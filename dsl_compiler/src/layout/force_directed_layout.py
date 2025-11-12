"""Force-directed layout optimization for circuit entities."""

from tqdm import tqdm
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from .layout_plan import EntityPlacement
from .signal_graph import SignalGraph


@dataclass
class LayoutConstraints:
    """Constraints for layout optimization."""

    max_wire_span: float = 9.0
    entity_spacing: float = 0.5
    boundary_penalty_start: float = 50.0
    boundary_penalty_strength: float = 0.01


@dataclass
class OptimizationResult:
    """Result of a single optimization attempt."""

    positions: Dict[str, Tuple[float, float]]
    energy: float
    violations: int
    success: bool


class _CollisionCache:
    """Pre-computed collision detection data for optimization."""

    def __init__(
        self, entity_ids: List[str], entity_placements: Dict[str, EntityPlacement]
    ):
        """
        Pre-compute footprint data that never changes during optimization.

        Args:
            entity_ids: Sorted list of entity identifiers
            entity_placements: Dict mapping entity_id to EntityPlacement
        """
        self.entity_ids = entity_ids
        self.collision_penalty = 1000.0

        # Cache footprint data - never changes during optimization
        self.footprints: Dict[str, Tuple[int, int]] = {}
        self.half_diagonals: Dict[str, float] = {}

        for entity_id in entity_ids:
            footprint = entity_placements[entity_id].properties["footprint"]
            self.footprints[entity_id] = footprint

            # Pre-compute half-diagonal for bounding circle (for early rejection)
            # This is the maximum distance from center to any corner
            width, height = footprint
            diagonal = math.sqrt(width * width + height * height)
            self.half_diagonals[entity_id] = diagonal * 0.5

        # Use spatial grid for large entity counts (threshold: 30 entities)
        self.use_spatial_grid = len(entity_ids) > 30

        if self.use_spatial_grid:
            # Grid cell size: use average entity size * 2 as heuristic
            avg_size = np.mean([max(fp) for fp in self.footprints.values()])
            self.grid_cell_size = max(3.0, avg_size * 2.0)


class ForceDirectedLayoutEngine:
    """
    Physics-based layout optimization using force-directed graph drawing.

    Uses:
    - Attractive spring forces between connected entities (proportional to distance)
    - Repulsive forces between all entities (inverse square)
    - Hard constraint penalties for wire span violations
    - Collision penalties for overlapping footprints
    - Boundary penalties to keep layout compact
    """

    def __init__(
        self,
        signal_graph: SignalGraph,
        entity_placements: Dict[str, EntityPlacement],
        diagnostics: ProgramDiagnostics,
        constraints: Optional[LayoutConstraints] = None,
    ):
        self.signal_graph = signal_graph
        self.entity_placements = entity_placements
        self.diagnostics = diagnostics
        self.constraints = constraints or LayoutConstraints()

        # Extract entity IDs in deterministic order
        self.entity_ids = sorted(entity_placements.keys())
        self.n_entities = len(self.entity_ids)

        # Build connectivity matrix
        self._build_connectivity()

        # Track fixed positions
        self.fixed_positions: Dict[str, Tuple[float, float]] = {}
        self._identify_fixed_positions()

        # Pre-compute collision detection cache
        self._collision_cache = _CollisionCache(self.entity_ids, self.entity_placements)

    def _build_connectivity(self) -> None:
        """Build adjacency matrix from signal graph."""
        self.connections: Set[Tuple[str, str]] = set()

        for signal_id, source_id, sink_id in self.signal_graph.iter_source_sink_pairs():
            if source_id in self.entity_ids and sink_id in self.entity_ids:
                # Store as sorted tuple for undirected graph
                edge = tuple(sorted([source_id, sink_id]))
                self.connections.add(edge)

    def _identify_fixed_positions(self) -> None:
        """Identify entities with fixed positions from user placement."""
        for entity_id, placement in self.entity_placements.items():
            # Check if position was explicitly set by user (from place() with fixed coords)
            if placement.position is not None:
                # Check if this was a user-specified position (not auto-generated)
                if placement.properties.get("user_specified_position"):
                    self.fixed_positions[entity_id] = placement.position

    def optimize(
        self,
        population_size: int = None,
        max_iterations: int = None,
        parallel: bool = True,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Optimize layout using population-based multi-start approach.

        Args:
            population_size: Number of random initializations to try (adaptive if None)
            max_iterations: Maximum optimization iterations per attempt (adaptive if None)
            parallel: Use parallel processing for multiple attempts

        Returns:
            Dict mapping entity_id to (x, y) position
        """
        self.diagnostics.info(
            f"Starting force-directed optimization: {self.n_entities} entities, "
            f"{len(self.connections)} connections"
        )

        if self.n_entities == 0:
            return {}

        # Adaptive parameters based on graph size
        if population_size is None:
            # Small graphs: 1-2 starts, medium: 2-3, large: 3-4
            if self.n_entities <= 3:
                population_size = 1  # Single attempt for tiny graphs
            elif self.n_entities <= 10:
                population_size = 2
            elif self.n_entities <= 30:
                population_size = 3
            else:
                population_size = 4

        if max_iterations is None:
            # Small graphs: quick convergence, large: more iterations
            if self.n_entities <= 3:
                max_iterations = 50
            elif self.n_entities <= 10:
                max_iterations = 100
            elif self.n_entities <= 30:
                max_iterations = 150
            else:
                max_iterations = 200

        self.diagnostics.info(
            f"Optimization parameters: population_size={population_size}, max_iterations={max_iterations}"
        )
        # Run multiple optimization attempts
        # Use parallel only for larger populations to avoid overhead
        # if parallel and population_size > 2:
        #     results = self._parallel_optimization(population_size, max_iterations)
        # else:
        results = self._sequential_optimization(population_size, max_iterations)

        # Select best result
        best_result = min(results, key=lambda r: r.energy)

        self.diagnostics.info(
            f"Optimization complete: energy={best_result.energy:.2f}, "
            f"violations={best_result.violations}"
        )

        return best_result.positions

    def _parallel_optimization(
        self, population_size: int, max_iterations: int
    ) -> List[OptimizationResult]:
        """Run multiple optimizations in parallel."""
        results = []

        with ProcessPoolExecutor(max_workers=min(population_size, 8)) as executor:
            futures = {
                executor.submit(
                    self._single_optimization_attempt, seed, max_iterations
                ): seed
                for seed in range(population_size)
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)

                    # Early stopping: if we found a perfect solution, stop
                    if result.violations == 0 and result.success:
                        self.diagnostics.info(
                            f"Found perfect solution early (attempt {len(results)}/{population_size})"
                        )
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break
                except Exception as e:
                    self.diagnostics.warning(f"Optimization attempt failed: {e}")

        return results if results else [self._fallback_result()]

    def _sequential_optimization(
        self, population_size: int, max_iterations: int
    ) -> List[OptimizationResult]:
        """Run multiple optimizations sequentially."""
        results = []

        for seed in tqdm(range(population_size)):
            # self.diagnostics.info(
            #     f"Starting optimization attempt {seed + 1}/{population_size}"
            # )
            try:
                result = self._single_optimization_attempt(seed, max_iterations)
                results.append(result)

                # Early stopping: if we found a perfect solution, stop
                if result.violations == 0 and result.success:
                    self.diagnostics.info(
                        f"Found perfect solution early (attempt {seed + 1}/{population_size})"
                    )
                    break
            except Exception as e:
                self.diagnostics.warning(f"Optimization attempt {seed} failed: {e}")

        return results if results else [self._fallback_result()]

    def _single_optimization_attempt(
        self, seed: int, max_iterations: int
    ) -> OptimizationResult:
        """Single optimization attempt with given random seed."""
        np.random.seed(seed)

        # Initialize positions
        x0 = self._initialize_positions(seed)

        # Optimize - use relaxed tolerances for faster convergence
        result = minimize(
            self._energy_function,
            x0,
            method="L-BFGS-B",
            options={
                "maxiter": max_iterations,
                "ftol": 1e-4,  # Relaxed from 1e-6 for speed
                "gtol": 1e-3,  # Relaxed from 1e-5 for speed
            },
        )

        # Convert to position dict
        positions = self._vector_to_positions(result.x)

        # Count violations
        violations = self._count_violations(positions)

        return OptimizationResult(
            positions=positions,
            energy=result.fun,
            violations=violations,
            success=result.success,
        )

    def _initialize_positions(self, seed: int) -> np.ndarray:
        """Initialize entity positions for optimization."""
        positions = {}

        # Start with fixed positions
        for entity_id, pos in self.fixed_positions.items():
            positions[entity_id] = pos

        # Initialize remaining entities
        if seed == 0:
            # First attempt: use simple grid layout as starting point
            self._init_grid_layout(positions)
        else:
            # Other attempts: use random perturbations
            self._init_random_layout(positions, seed)

        return self._positions_to_vector(positions)

    def _init_grid_layout(self, positions: Dict[str, Tuple[float, float]]) -> None:
        """Initialize with simple grid layout."""
        grid_size = int(math.ceil(math.sqrt(self.n_entities)))
        spacing = 3.0  # Conservative spacing for initial layout

        idx = 0
        for entity_id in self.entity_ids:
            if entity_id in positions:
                continue  # Already fixed

            row = idx // grid_size
            col = idx % grid_size
            positions[entity_id] = (col * spacing, row * spacing)
            idx += 1

    def _init_random_layout(
        self, positions: Dict[str, Tuple[float, float]], seed: int
    ) -> None:
        """Initialize with random positions."""
        np.random.seed(seed)

        # Estimate bounds from fixed positions or use default
        if self.fixed_positions:
            xs = [p[0] for p in self.fixed_positions.values()]
            ys = [p[1] for p in self.fixed_positions.values()]
            x_range = (min(xs) - 10, max(xs) + 10)
            y_range = (min(ys) - 10, max(ys) + 10)
        else:
            spread = math.sqrt(self.n_entities) * 5
            x_range = (-spread, spread)
            y_range = (-spread, spread)

        for entity_id in self.entity_ids:
            if entity_id in positions:
                continue  # Already fixed

            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            positions[entity_id] = (x, y)

    def _positions_to_vector(
        self, positions: Dict[str, Tuple[float, float]]
    ) -> np.ndarray:
        """Convert position dict to optimization vector."""
        vec = np.zeros(self.n_entities * 2)

        for i, entity_id in enumerate(self.entity_ids):
            x, y = positions[entity_id]
            vec[2 * i] = x
            vec[2 * i + 1] = y

        return vec

    def _vector_to_positions(self, vec: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Convert optimization vector to position dict."""
        positions = {}

        for i, entity_id in enumerate(self.entity_ids):
            x = vec[2 * i]
            y = vec[2 * i + 1]
            positions[entity_id] = (x, y)

        return positions

    def _energy_function(self, vec: np.ndarray) -> float:
        """
        Total energy function to minimize.

        Components:
        1. Spring forces (attractive) between connected entities
        2. Repulsive forces between all entities
        3. Collision penalties for overlapping footprints
        4. Hard constraint penalties for wire span violations
        5. Boundary penalties to keep layout compact
        6. Fixed position penalties (infinite for user-specified positions)
        """
        positions = self._vector_to_positions(vec)

        energy = 0.0

        # 1. Spring forces (attractive)
        energy += self._spring_energy(positions)

        # 2. Repulsive forces
        energy += self._repulsion_energy(positions)

        # 3. Collision penalties
        energy += self._collision_energy(positions)

        # 4. Wire span constraint penalties
        energy += self._span_constraint_energy(positions)

        # 5. Boundary penalties
        energy += self._boundary_energy(positions)

        # 6. Fixed position penalties
        energy += self._fixed_position_energy(positions)

        return energy

    def _spring_energy(self, positions: Dict[str, Tuple[float, float]]) -> float:
        """
        Optimized attractive spring forces between connected entities.

        Uses squared distance to avoid sqrt operations.
        """
        energy = 0.0
        ideal_distance = 3.0
        ideal_distance_sq = ideal_distance * ideal_distance  # Pre-compute
        spring_constant = 1.0

        for e1, e2 in self.connections:
            pos1 = positions[e1]
            pos2 = positions[e2]

            # Compute squared distance (avoid sqrt)
            dx = pos1[0] - pos2[0]
            dy = pos1[1] - pos2[1]
            dist_sq = dx * dx + dy * dy

            # For spring energy: E = 0.5 * k * (d - d0)^2
            # Expanding: E = 0.5 * k * (d^2 - 2*d*d0 + d0^2)
            # We need actual distance d for the deviation term
            # However, we can use a fast approximation or just compute sqrt here
            # Since this is only for connected pairs (much fewer than n^2),
            # the sqrt cost is acceptable
            dist = math.sqrt(dist_sq)

            # Quadratic spring: E = 0.5 * k * (d - d0)^2
            deviation = dist - ideal_distance
            energy += 0.5 * spring_constant * deviation * deviation

        return energy

    def _repulsion_energy(self, positions: Dict[str, Tuple[float, float]]) -> float:
        """
        Optimized repulsive forces between all entity pairs.

        Key optimizations:
        1. Use squared distance directly (avoid sqrt)
        2. Distance cutoff (distant entities contribute negligibly)
        3. Spatial grid for large graphs
        """
        cache = self._collision_cache

        if cache.use_spatial_grid:
            return self._repulsion_energy_grid(positions, cache)
        else:
            return self._repulsion_energy_cutoff(positions)

    def _repulsion_energy_cutoff(
        self, positions: Dict[str, Tuple[float, float]]
    ) -> float:
        """O(n²) repulsion with distance cutoff to skip distant pairs."""
        energy = 0.0
        repulsion_strength = 10.0
        min_distance_sq = 0.01  # min_distance = 0.1, squared

        # Distance cutoff: beyond this, repulsion is negligible
        # For k=10 and cutoff_energy=0.01: d_cutoff = sqrt(k/0.01) = sqrt(1000) ≈ 31.6
        cutoff_distance_sq = 1000.0  # (31.6)^2, repulsion < 0.01 beyond this

        for i, e1 in enumerate(self.entity_ids):
            pos1 = positions[e1]

            for e2 in self.entity_ids[i + 1 :]:
                pos2 = positions[e2]

                # Compute squared distance (avoid sqrt!)
                dx = pos1[0] - pos2[0]
                dy = pos1[1] - pos2[1]
                dist_sq = dx * dx + dy * dy

                # Skip distant pairs (negligible contribution)
                if dist_sq > cutoff_distance_sq:
                    continue

                # Clamp to minimum to prevent division by zero
                dist_sq = max(dist_sq, min_distance_sq)

                # Inverse square repulsion: E = k / d^2
                energy += repulsion_strength / dist_sq

        return energy

    def _repulsion_energy_grid(
        self, positions: Dict[str, Tuple[float, float]], cache: _CollisionCache
    ) -> float:
        """O(n) repulsion using spatial grid for large graphs."""
        from collections import defaultdict

        energy = 0.0
        repulsion_strength = 10.0
        min_distance_sq = 0.01
        cutoff_distance_sq = 1000.0

        # Build spatial grid with larger cells for repulsion
        # Repulsion has longer range than collision, so use larger cells
        grid_cell_size = cache.grid_cell_size * 3  # 3x larger than collision grid

        grid = defaultdict(list)
        for entity_id in self.entity_ids:
            pos = positions[entity_id]
            cell_x = int(pos[0] / grid_cell_size)
            cell_y = int(pos[1] / grid_cell_size)
            grid[(cell_x, cell_y)].append(entity_id)

        # Check repulsion only in nearby cells
        checked_pairs = set()

        # For repulsion, we need to check more distant cells
        # since repulsion has longer range than collision
        search_radius = 2  # Check 2 cells away (5x5 neighborhood)

        for (cell_x, cell_y), entities in grid.items():
            # Within same cell
            for i, e1 in enumerate(entities):
                pos1 = positions[e1]

                for e2 in entities[i + 1 :]:
                    pair = (e1, e2) if e1 < e2 else (e2, e1)
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)

                    pos2 = positions[e2]
                    dx = pos1[0] - pos2[0]
                    dy = pos1[1] - pos2[1]
                    dist_sq = dx * dx + dy * dy

                    if dist_sq > cutoff_distance_sq:
                        continue

                    dist_sq = max(dist_sq, min_distance_sq)
                    energy += repulsion_strength / dist_sq

            # Nearby cells (5x5 neighborhood for longer-range repulsion)
            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    if dx == 0 and dy == 0:
                        continue

                    neighbor_cell = (cell_x + dx, cell_y + dy)
                    if neighbor_cell not in grid:
                        continue

                    for e1 in entities:
                        pos1 = positions[e1]

                        for e2 in grid[neighbor_cell]:
                            pair = (e1, e2) if e1 < e2 else (e2, e1)
                            if pair in checked_pairs:
                                continue
                            checked_pairs.add(pair)

                            pos2 = positions[e2]
                            dx_dist = pos1[0] - pos2[0]
                            dy_dist = pos1[1] - pos2[1]
                            dist_sq = dx_dist * dx_dist + dy_dist * dy_dist

                            if dist_sq > cutoff_distance_sq:
                                continue

                            dist_sq = max(dist_sq, min_distance_sq)
                            energy += repulsion_strength / dist_sq

        return energy

    def _collision_energy(self, positions: Dict[str, Tuple[float, float]]) -> float:
        """
        Optimized collision detection with early rejection and spatial grid.

        For small graphs (n <= 30): O(n²) with bounding circle early rejection
        For large graphs (n > 30): O(n) with spatial grid hashing
        """
        cache = self._collision_cache

        if cache.use_spatial_grid:
            return self._collision_energy_grid(positions, cache)
        else:
            return self._collision_energy_early_rejection(positions, cache)

    def _collision_energy_early_rejection(
        self, positions: Dict[str, Tuple[float, float]], cache: _CollisionCache
    ) -> float:
        """
        O(n²) collision detection with bounding circle early rejection.

        Optimizations:
        1. Pre-cached footprint data (no dict lookups)
        2. Bounding circle test before expensive rectangular overlap
        3. Early exit from overlap calculation
        """
        energy = 0.0

        for i, e1 in enumerate(self.entity_ids):
            pos1 = positions[e1]
            half_diag1 = cache.half_diagonals[e1]
            footprint1 = cache.footprints[e1]

            for e2 in self.entity_ids[i + 1 :]:
                pos2 = positions[e2]
                half_diag2 = cache.half_diagonals[e2]

                # EARLY REJECTION: Check if bounding circles overlap
                # This is much cheaper than rectangular overlap calculation
                dx = pos1[0] - pos2[0]
                dy = pos1[1] - pos2[1]
                center_dist_sq = dx * dx + dy * dy
                min_dist = half_diag1 + half_diag2

                # If bounding circles don't overlap, entities definitely don't overlap
                if center_dist_sq > min_dist * min_dist:
                    continue

                # Only compute expensive rectangular overlap if bounding circles overlap
                footprint2 = cache.footprints[e2]
                overlap = self._compute_rectangular_overlap(
                    pos1, footprint1, pos2, footprint2
                )

                if overlap > 0:
                    energy += cache.collision_penalty * overlap

        return energy

    def _collision_energy_grid(
        self, positions: Dict[str, Tuple[float, float]], cache: _CollisionCache
    ) -> float:
        """
        O(n) collision detection using spatial grid hashing.

        Only checks entities in the same or adjacent grid cells.
        Typical speedup: 10-100x for large entity counts.
        """
        energy = 0.0

        # Build spatial hash grid - O(n)
        grid = defaultdict(list)
        for entity_id in self.entity_ids:
            pos = positions[entity_id]
            # Hash position to grid cell
            cell_x = int(pos[0] / cache.grid_cell_size)
            cell_y = int(pos[1] / cache.grid_cell_size)
            grid[(cell_x, cell_y)].append(entity_id)

        # Check collisions - only within same and adjacent cells
        checked_pairs = set()

        for (cell_x, cell_y), entities in grid.items():
            # Check within same cell
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1 :]:
                    pair = (e1, e2) if e1 < e2 else (e2, e1)
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)

                    energy += self._check_collision_pair(e1, e2, positions, cache)

            # Check adjacent cells (8 neighbors)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    neighbor_cell = (cell_x + dx, cell_y + dy)
                    if neighbor_cell not in grid:
                        continue

                    for e1 in entities:
                        for e2 in grid[neighbor_cell]:
                            pair = (e1, e2) if e1 < e2 else (e2, e1)
                            if pair in checked_pairs:
                                continue
                            checked_pairs.add(pair)

                            energy += self._check_collision_pair(
                                e1, e2, positions, cache
                            )

        return energy

    def _check_collision_pair(
        self,
        e1: str,
        e2: str,
        positions: Dict[str, Tuple[float, float]],
        cache: _CollisionCache,
    ) -> float:
        """Check collision between two entities with early rejection."""
        pos1 = positions[e1]
        pos2 = positions[e2]

        # Early rejection using bounding circles
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        center_dist_sq = dx * dx + dy * dy
        min_dist = cache.half_diagonals[e1] + cache.half_diagonals[e2]

        if center_dist_sq > min_dist * min_dist:
            return 0.0

        # Compute detailed rectangular overlap
        overlap = self._compute_rectangular_overlap(
            pos1, cache.footprints[e1], pos2, cache.footprints[e2]
        )

        return cache.collision_penalty * overlap if overlap > 0 else 0.0

    def _compute_rectangular_overlap(
        self,
        pos1: Tuple[float, float],
        footprint1: Tuple[int, int],
        pos2: Tuple[float, float],
        footprint2: Tuple[int, int],
    ) -> float:
        """
        Compute overlap area between two rectangles (no spacing).

        Optimized for tight packing on integer grid.
        """
        # Entity 1 bounds (center position to corners)
        half_w1 = footprint1[0] * 0.5
        half_h1 = footprint1[1] * 0.5

        x1_min = pos1[0] - half_w1
        x1_max = pos1[0] + half_w1
        y1_min = pos1[1] - half_h1
        y1_max = pos1[1] + half_h1

        # Entity 2 bounds
        half_w2 = footprint2[0] * 0.5
        half_h2 = footprint2[1] * 0.5

        x2_min = pos2[0] - half_w2
        x2_max = pos2[0] + half_w2
        y2_min = pos2[1] - half_h2
        y2_max = pos2[1] + half_h2

        # Compute overlap
        x_overlap = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))

        # Early exit if no x-axis overlap
        if x_overlap == 0.0:
            return 0.0

        y_overlap = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))

        return x_overlap * y_overlap

    def _span_constraint_energy(
        self, positions: Dict[str, Tuple[float, float]]
    ) -> float:
        """
        Heavy penalty for connections exceeding max wire span.

        Optimized to use squared distance.
        """
        energy = 0.0
        violation_penalty = 100.0
        max_span_sq = self.constraints.max_wire_span**2

        for e1, e2 in self.connections:
            pos1 = positions[e1]
            pos2 = positions[e2]

            # Compute squared distance
            dx = pos1[0] - pos2[0]
            dy = pos1[1] - pos2[1]
            dist_sq = dx * dx + dy * dy

            if dist_sq > max_span_sq:
                # Quadratic penalty for violations
                # E = penalty * (d - d_max)^2
                # We need actual distance here
                dist = math.sqrt(dist_sq)
                excess = dist - self.constraints.max_wire_span
                energy += violation_penalty * excess * excess

        return energy

    def _boundary_energy(self, positions: Dict[str, Tuple[float, float]]) -> float:
        """
        Soft penalty to keep entities within reasonable bounds.

        Optimized to use squared distance.
        """
        energy = 0.0
        penalty_start = self.constraints.boundary_penalty_start
        penalty_start_sq = penalty_start * penalty_start
        penalty_strength = self.constraints.boundary_penalty_strength

        for pos in positions.values():
            # Radial squared distance from origin
            dist_sq = pos[0] * pos[0] + pos[1] * pos[1]

            if dist_sq > penalty_start_sq:
                # E = strength * (d - d_start)^2
                # Need actual distance for the penalty calculation
                dist = math.sqrt(dist_sq)
                excess = dist - penalty_start
                energy += penalty_strength * excess * excess

        return energy

    def _fixed_position_energy(
        self, positions: Dict[str, Tuple[float, float]]
    ) -> float:
        """Infinite penalty for moving fixed positions."""
        energy = 0.0
        fixed_penalty = 1e9

        for entity_id, fixed_pos in self.fixed_positions.items():
            current_pos = positions[entity_id]

            # Squared distance from fixed position
            dx = current_pos[0] - fixed_pos[0]
            dy = current_pos[1] - fixed_pos[1]
            dist_sq = dx**2 + dy**2

            if dist_sq > 1e-6:  # Tolerance for numerical precision
                energy += fixed_penalty * dist_sq

        return energy

    def _count_violations(self, positions: Dict[str, Tuple[float, float]]) -> int:
        """
        Count number of wire span violations.

        Optimized to use squared distance.
        """
        violations = 0
        max_span_sq = self.constraints.max_wire_span**2

        for e1, e2 in self.connections:
            pos1 = positions[e1]
            pos2 = positions[e2]

            # Compute squared distance
            dx = pos1[0] - pos2[0]
            dy = pos1[1] - pos2[1]
            dist_sq = dx * dx + dy * dy

            if dist_sq > max_span_sq:
                violations += 1

        return violations

    def _fallback_result(self) -> OptimizationResult:
        """Create fallback result using simple grid layout."""
        positions = {}
        self._init_grid_layout(positions)

        return OptimizationResult(
            positions=positions,
            energy=float("inf"),
            violations=self._count_violations(positions),
            success=False,
        )
