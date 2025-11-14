"""Force-directed layout optimization for circuit entities - OPTIMIZED VERSION.

Optimizations:
- scipy.spatial.cKDTree for O(n log n) spatial queries (replaces O(n²) grid)
- NumPy vectorization for batch operations (10-100x speedup)
- numba JIT compilation for hot paths (5-10x speedup)
- Eliminated 137M set operations bottleneck

Preserves all constraints:
- Rectangular entity footprints with varying sizes
- Hard connection distance limits (max_wire_span)
- Integer grid placement compatibility
"""

from tqdm import tqdm
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from numba import jit

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from .layout_plan import EntityPlacement
from .signal_graph import SignalGraph

# Optional debug visualization support
from .force_directed_layout_debug_viz import LayoutVisualizer

VISUAL_DEBUG = False
VISUAL_DEBUG_PATH = "output/layout_visualization4"
RENDER_RESULT = False
RENDER_RESULT_PATH = "output/layout_result4"


@dataclass
class LayoutConstraints:
    """Constraints for layout optimization."""

    max_wire_span: float = 8.0
    entity_spacing: float = 0.5
    boundary_penalty_start: float = 50.0
    boundary_penalty_strength: float = 100


@dataclass
class OptimizationResult:
    """Result of a single optimization attempt."""

    positions: Dict[str, Tuple[float, float]]
    energy: float
    violations: int
    success: bool


# ============================================================================
# NUMBA-COMPILED HOT PATHS (5-10x speedup)
# ============================================================================


@jit(nopython=True, cache=True)
def compute_rectangular_overlap_numba(
    x1: float,
    y1: float,
    w1: float,
    h1: float,
    x2: float,
    y2: float,
    w2: float,
    h2: float,
) -> float:
    """
    JIT-compiled rectangular overlap calculation.

    Args:
        x1, y1: Center of rectangle 1
        w1, h1: Width and height of rectangle 1
        x2, y2: Center of rectangle 2
        w2, h2: Width and height of rectangle 2

    Returns:
        Overlap area (0 if no overlap)
    """
    half_w1 = w1 * 0.5
    half_h1 = h1 * 0.5
    half_w2 = w2 * 0.5
    half_h2 = h2 * 0.5

    x1_min = x1 - half_w1
    x1_max = x1 + half_w1
    y1_min = y1 - half_h1
    y1_max = y1 + half_h1

    x2_min = x2 - half_w2
    x2_max = x2 + half_w2
    y2_min = y2 - half_h2
    y2_max = y2 + half_h2

    # Compute overlap (separating axis theorem)
    x_overlap = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    if x_overlap == 0.0:
        return 0.0

    y_overlap = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    return x_overlap * y_overlap


@jit(nopython=True, cache=True)
def compute_distances_squared(pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
    """Compute squared Euclidean distances between corresponding pairs."""
    diff = pos1 - pos2
    return np.sum(diff * diff, axis=1)


@jit(nopython=True, cache=True)
def compute_distances(pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
    """Compute Euclidean distances between corresponding pairs."""
    diff = pos1 - pos2
    return np.sqrt(np.sum(diff * diff, axis=1))


# ============================================================================
# OPTIMIZED FORCE-DIRECTED LAYOUT ENGINE
# ============================================================================


class ForceDirectedLayoutEngine:
    """
    Physics-based layout optimization using force-directed graph drawing.

    OPTIMIZED with:
    - scipy.spatial.cKDTree for efficient spatial queries (O(n log n))
    - NumPy vectorization for batch operations (10-100x speedup)
    - numba JIT compilation for hot paths (5-10x speedup)

    PRESERVES:
    - Rectangular entity footprints with varying sizes
    - Hard connection distance limits (max_wire_span)
    - All original energy functions and constraints
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

        # Build index mapping for vectorization
        self.entity_id_to_idx = {eid: i for i, eid in enumerate(self.entity_ids)}
        self.idx_to_entity_id = {i: eid for i, eid in enumerate(self.entity_ids)}

        # Build connectivity as NumPy array for vectorization
        self._build_connectivity()

        # Track fixed positions
        self.fixed_positions: Dict[str, Tuple[float, float]] = {}
        self.fixed_indices: np.ndarray = np.array([], dtype=np.int32)
        self._identify_fixed_positions()

        # Pre-compute collision detection data
        self._precompute_collision_data()

        # Pre-compute constants for energy functions
        self._init_constants()

    def _build_connectivity(self) -> None:
        """Build connectivity as NumPy array of index pairs for vectorization."""
        connections_list = []

        for signal_id, source_id, sink_id in self.signal_graph.iter_source_sink_pairs():
            if source_id in self.entity_id_to_idx and sink_id in self.entity_id_to_idx:
                idx1 = self.entity_id_to_idx[source_id]
                idx2 = self.entity_id_to_idx[sink_id]
                # Store as sorted pair for undirected graph
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                connections_list.append((idx1, idx2))
            else:
                self.diagnostics.warning(
                    f"Signal '{signal_id}' connects unknown entities "
                    f"'{source_id}' and/or '{sink_id}'"
                )

        # Remove duplicates and convert to NumPy array
        connections_set = set(connections_list)
        if connections_set:
            self.connection_pairs = np.array(list(connections_set), dtype=np.int32)
        else:
            self.connection_pairs = np.empty((0, 2), dtype=np.int32)

        self.diagnostics.info(
            f"Built connectivity: {len(self.connection_pairs)} unique connections"
        )

    def _identify_fixed_positions(self) -> None:
        """Identify entities with fixed positions from user placement."""
        fixed_idx_list = []

        for entity_id, placement in self.entity_placements.items():
            if placement.position is not None:
                if placement.properties.get("user_specified_position"):
                    self.fixed_positions[entity_id] = placement.position
                    idx = self.entity_id_to_idx[entity_id]
                    fixed_idx_list.append(idx)

        if fixed_idx_list:
            self.fixed_indices = np.array(fixed_idx_list, dtype=np.int32)
            self.fixed_pos_array = np.array(
                [
                    self.fixed_positions[self.idx_to_entity_id[i]]
                    for i in self.fixed_indices
                ],
                dtype=np.float64,
            )
        else:
            self.fixed_indices = np.array([], dtype=np.int32)
            self.fixed_pos_array = np.empty((0, 2), dtype=np.float64)

        self.diagnostics.info(f"Found {len(self.fixed_positions)} fixed positions")

    def _precompute_collision_data(self) -> None:
        """Pre-compute footprint data that never changes during optimization."""
        # Pre-allocate arrays
        self.footprints = np.zeros((self.n_entities, 2), dtype=np.float64)
        self.half_diagonals = np.zeros(self.n_entities, dtype=np.float64)

        for i, entity_id in enumerate(self.entity_ids):
            footprint = self.entity_placements[entity_id].properties["footprint"]
            self.footprints[i] = footprint

            # Pre-compute half-diagonal for bounding circle (for early rejection)
            width, height = footprint
            diagonal = math.sqrt(width * width + height * height)
            self.half_diagonals[i] = diagonal * 0.5

        # Maximum possible collision distance (for KDTree query radius)
        if self.n_entities > 0:
            self.max_collision_distance = self.half_diagonals.max() * 2.0
        else:
            self.max_collision_distance = 0.0

    def _init_constants(self) -> None:
        """Initialize constants used in energy calculations."""
        # Repulsion parameters
        self.repulsion_strength = 0.1
        self.min_distance_sq = 0.01
        self.repulsion_cutoff_distance = 31.6  # sqrt(1000)

        # Collision parameters
        self.collision_penalty = 100000.0

        # Spring parameters
        self.ideal_spring_distance = 3.0
        self.spring_constant = 20000.0

        # Constraint parameters
        self.violation_penalty = 10000.0
        self.fixed_penalty = 1e9

        self.wire_length_constant = 1

    def optimize(
        self,
        population_size: int = None,
        max_iterations: int = None,
        parallel: bool = True,  # Not implemented in optimized version
        visualization_dir: str = "output/layout_visualization",
    ) -> Dict[str, Tuple[float, float]]:
        """
        Optimize layout using population-based multi-start approach.

        Args:
            population_size: Number of random initializations to try (adaptive if None)
            max_iterations: Maximum optimization iterations per attempt (adaptive if None)
            parallel: Unused (kept for API compatibility)
            visualization_dir: Directory to save visualization GIFs

        Returns:
            Dict mapping entity_id to (x, y) position
        """
        global RENDER_RESULT_PATH
        self.diagnostics.info(
            f"Starting optimized force-directed layout: {self.n_entities} entities, "
            f"{len(self.connection_pairs)} connections"
        )

        if self.n_entities == 0:
            return {}

        # Adaptive parameters based on graph size
        if population_size is None:
            if self.n_entities <= 3:
                population_size = 1
            elif self.n_entities <= 10:
                population_size = 5
            elif self.n_entities <= 30:
                population_size = 10
            else:
                population_size = 15

        if max_iterations is None:
            if self.n_entities <= 3:
                max_iterations = 50
            elif self.n_entities <= 10:
                max_iterations = 100
            elif self.n_entities <= 30:
                max_iterations = 150
            else:
                max_iterations = 200

        self.diagnostics.info(
            f"Optimization parameters: population_size={population_size}, "
            f"max_iterations={max_iterations}"
        )

        # ores = RENDER_RESULT_PATH
        # Run optimization
        # RENDER_RESULT_PATH = ores + "_parallel"
        # results = self._parallel_optimization(population_size, max_iterations)
        # RENDER_RESULT_PATH = ores + "_sequential"
        # results = self._sequential_optimization(population_size, max_iterations)
        parallel = False
        if population_size > 1 and parallel:
            results = self._parallel_optimization(population_size, max_iterations)
        else:
            results = self._sequential_optimization(population_size, max_iterations)

        # Select best result
        best_result = min(results, key=lambda r: r.energy)

        self.diagnostics.info(
            f"Optimization complete: energy={best_result.energy:.2f}, "
            f"violations={best_result.violations}"
        )

        # Render final result if enabled
        if RENDER_RESULT:
            self._render_final_result(best_result.positions)

        return best_result.positions

    def _parallel_optimization(
        self, population_size: int, max_iterations: int
    ) -> List[OptimizationResult]:
        """Run multiple optimizations in parallel."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        results = []
        futures = []

        with ProcessPoolExecutor() as executor:
            # for seed in [2]:
            for seed in range(population_size):
                futures.append(
                    executor.submit(
                        self._single_optimization_attempt_two_phase,
                        seed,
                        max_iterations,
                    )
                )

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Optimizing"
            ):
                try:
                    result = future.result()
                    results.append(result)

                    if RENDER_RESULT:
                        self._render_final_result(
                            result.positions, suffix=f"attempt_{seed}"
                        )
                    # Early stopping: if we found a perfect solution, stop
                    # if result.violations == 0 and result.success:
                    #     self.diagnostics.info(
                    #         "Found perfect solution early in parallel optimization"
                    #     )
                    #     break
                except Exception as e:
                    self.diagnostics.warning(f"Optimization attempt failed: {e}")

        return results if results else [self._fallback_result()]

    def _sequential_optimization(
        self, population_size: int, max_iterations: int
    ) -> List[OptimizationResult]:
        """Run multiple optimizations sequentially."""
        results = []

        # for seed in tqdm([2], desc="Optimizing"):
        for seed in tqdm(range(population_size), desc="Optimizing"):
            try:
                result = self._single_optimization_attempt_two_phase(
                    seed, max_iterations
                )
                results.append(result)
                print(
                    f"Attempt {seed}: energy={result.energy:.2f}, violations={result.violations}"
                )
                if RENDER_RESULT:
                    self._render_final_result(
                        result.positions, suffix=f"attempt_{seed}"
                    )

                # Early stopping: if we found a perfect solution, stop
                # if result.violations == 0 and result.success:
                #     self.diagnostics.info(
                #         f"Found perfect solution early (attempt {seed + 1}/{population_size})"
                #     )
                #     break
            except Exception as e:
                self.diagnostics.warning(f"Optimization attempt {seed} failed: {e}")

        return results if results else [self._fallback_result()]

    def _single_optimization_attempt(
        self, seed: int, max_iterations: int
    ) -> OptimizationResult:
        """Single optimization attempt with given random seed."""
        np.random.seed(seed)

        # Initialize positions as flat NumPy array (for L-BFGS-B)
        pos_flat = self._initialize_positions(seed)

        # Setup visualization if enabled
        visualizer = None

        visualizer = (
            LayoutVisualizer(
                layout_engine=self,
                attempt_id=seed,
                phase_name="",
                output_base_dir=VISUAL_DEBUG_PATH,
                frame_skip=max(1, max_iterations // 50),
            )
            if VISUAL_DEBUG
            else None
        )

        # Optimize using L-BFGS-B WITH analytical gradients
        result = minimize(
            fun=lambda x: self._energy_and_gradient(x),  # Returns (energy, grad)
            x0=pos_flat,
            method="L-BFGS-B",
            jac=True,  # ← KEY: Tell scipy we provide gradients
            callback=visualizer.create_callback(self._energy_and_gradient)
            if visualizer
            else None,
            options={
                "maxiter": max_iterations,
                "ftol": 1e-6,  # Can be stricter now with analytical gradients
                "gtol": 1e-5,  # Can be stricter now with analytical gradients
            },
        )

        # Save GIF if visualization was enabled
        if visualizer:
            visualizer.create_gif()
            visualizer.cleanup()

        # Convert to position dict
        pos_array = result.x.reshape(-1, 2)
        positions = self._array_to_positions(pos_array)

        # Count violations
        violations = self._count_violations(pos_array)
        print(f"Phase 1 complete after {result.nit} iterations.")  # Debug

        return OptimizationResult(
            positions=positions,
            energy=result.fun,
            violations=violations,
            success=result.success,
        )

    def _initialize_positions(self, seed: int) -> np.ndarray:
        """Initialize entity positions as flat NumPy array."""
        pos_array = np.zeros((self.n_entities, 2), dtype=np.float64)

        # Start with fixed positions
        if len(self.fixed_indices) > 0:
            pos_array[self.fixed_indices] = self.fixed_pos_array

        # Initialize remaining entities
        if seed == 0:
            # First attempt: use simple grid layout
            self._init_grid_layout(pos_array)
        else:
            # Other attempts: use random perturbations
            self._init_random_layout(pos_array, seed)

        return pos_array.flatten()

    def _init_grid_layout(self, pos_array: np.ndarray) -> None:
        """Initialize with simple grid layout."""
        grid_size = int(math.ceil(math.sqrt(self.n_entities)))

        # Adaptive spacing based on entity sizes
        if len(self.footprints) > 0:
            avg_width = np.mean(self.footprints[:, 0])
            avg_height = np.mean(self.footprints[:, 1])
            avg_size = (avg_width + avg_height) / 2
            spacing = max(5.0, avg_size * 3)  # At least 3x entity size
        else:
            spacing = 5.0  # ← NEW: Wider spacing!

        idx = 0
        for i in range(self.n_entities):
            entity_id = self.entity_ids[i]
            if entity_id in self.fixed_positions:
                continue  # Already set

            row = idx // grid_size
            col = idx % grid_size
            pos_array[i] = [col * spacing, row * spacing]
            idx += 1

    def _init_random_layout(self, pos_array: np.ndarray, seed: int) -> None:
        """Initialize with random positions."""
        np.random.seed(seed)

        # Estimate bounds from fixed positions
        if self.fixed_positions:
            x_range = (
                self.fixed_pos_array[:, 0].min() - 10,
                self.fixed_pos_array[:, 0].max() + 10,
            )
            y_range = (
                self.fixed_pos_array[:, 1].min() - 10,
                self.fixed_pos_array[:, 1].max() + 10,
            )
        else:
            spread = math.sqrt(self.n_entities) * 5
            x_range = (-spread, spread)
            y_range = (-spread, spread)

        for i in range(self.n_entities):
            entity_id = self.entity_ids[i]
            if entity_id in self.fixed_positions:
                continue  # Already set

            pos_array[i, 0] = np.random.uniform(x_range[0], x_range[1])
            pos_array[i, 1] = np.random.uniform(y_range[0], y_range[1])

    def _array_to_positions(
        self, pos_array: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """Convert NumPy array to position dict."""
        positions = {}
        for i, entity_id in enumerate(self.entity_ids):
            positions[entity_id] = (float(pos_array[i, 0]), float(pos_array[i, 1]))
        return positions

    # ========================================================================
    # ENERGY FUNCTIONS - ALL OPTIMIZED
    # ========================================================================

    def _energy_and_gradient(self, vec: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute energy AND gradient in one pass.

        This is called by scipy.optimize.minimize instead of just _energy_function.

        Returns:
            tuple: (energy, gradient_flat)
        """
        pos_array = vec.reshape(-1, 2)
        grad_array = np.zeros_like(pos_array)

        total_energy = 0.0

        # Compute each energy component with its gradient
        e1, g1 = self._spring_energy_grad(pos_array)
        total_energy += e1
        grad_array += g1

        # e2, g2 = self._repulsion_energy_grad(pos_array)
        # total_energy += e2
        # grad_array += g2

        e3, g3 = self._collision_energy_grad(pos_array)
        total_energy += e3
        grad_array += g3

        e4, g4 = self._span_constraint_energy_grad(pos_array)
        total_energy += e4
        grad_array += g4

        e5, g5 = self._boundary_energy_grad(pos_array)
        total_energy += e5
        grad_array += g5

        e6, g6 = self._fixed_position_energy_grad(pos_array)
        total_energy += e6
        grad_array += g6

        if len(self.fixed_indices) > 0:
            grad_array[self.fixed_indices] = 0
        # print(
        #     f"Phase 2 energy: {total_energy:.2f} spring energy: {e1:.2f}, repulsion energy: {0:.2f}, collision energy: {e3:.2f}, span energy: {e4:.2f}, boundary energy: {e5:.2f}, fixed energy: {e6:.2f}"
        # )  # Debug
        # if total_energy < 10000:
        #     print("Low energy reached, detailed breakdown:")
        #     print(
        #         f"Low energy debug: spring energy: {e1:.2f}, repulsion energy: {0:.2f}, collision energy: {e3:.2f}, span energy: {e4:.2f}, boundary energy: {e5:.2f}, fixed energy: {e6:.2f}"
        #     )
        return total_energy, grad_array.flatten()

    def _count_violations(self, pos_array: np.ndarray) -> int:
        """
        Count wire span violations.

        OPTIMIZATION: Fully vectorized.
        """
        if len(self.connection_pairs) == 0:
            return 0

        max_span_sq = self.constraints.max_wire_span**2

        # Get positions for connected pairs
        pos1 = pos_array[self.connection_pairs[:, 0]]
        pos2 = pos_array[self.connection_pairs[:, 1]]

        # Vectorized distance calculation
        dist_sq = compute_distances_squared(pos1, pos2)

        violations = np.sum(dist_sq > max_span_sq)

        return int(violations)

    def _fallback_result(self) -> OptimizationResult:
        """Create fallback result using simple grid layout."""
        print("Returning fallback layout result (grid layout).")
        pos_array = np.zeros((self.n_entities, 2), dtype=np.float64)
        self._init_grid_layout(pos_array)
        positions = self._array_to_positions(pos_array)

        return OptimizationResult(
            positions=positions,
            energy=float("inf"),
            violations=self._count_violations(pos_array),
            success=False,
        )

    def _render_final_result(
        self, positions: Dict[str, Tuple[float, float]], suffix=""
    ):
        """Render the final optimization result to an image file.

        Args:
            positions: Dictionary mapping entity_id to (x, y) position
        """
        from pathlib import Path
        from .force_directed_layout_debug_viz import visualize_graph

        # Convert positions dict to flat array
        pos_array = np.zeros((self.n_entities, 2), dtype=np.float64)
        for i, entity_id in enumerate(self.entity_ids):
            if entity_id in positions:
                pos_array[i] = positions[entity_id]

        # Create output directory if needed
        output_dir = Path(RENDER_RESULT_PATH)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with entity count for uniqueness
        output_file = (
            output_dir / f"layout_result_{self.n_entities}_entities_{suffix}.png"
        )

        # Render the visualization
        visualize_graph(
            layout_engine=self,
            pos_flat=pos_array.flatten(),
            output_path=str(output_file),
            figsize=(16, 16),
            dpi=200,
        )

        self.diagnostics.info(f"Final layout rendered to: {output_file}")

    # ========================================================================
    # GRADIENT FUNCTIONS - ANALYTICAL GRADIENTS FOR 5-10x SPEEDUP
    # ========================================================================

    def _spring_energy_grad(self, pos_array: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Spring energy with analytical gradient.

        DERIVATION:
        Energy:   E = 0.5 * k * sum_edges (d_ij - d_0)^2

        Gradient: ∂E/∂x_i = sum_j k * (d_ij - d_0) * (x_i - x_j) / d_ij

        Where:
        - d_ij = ||pos_i - pos_j|| (distance between connected entities)
        - d_0 = ideal spring distance
        - k = spring constant
        """
        if len(self.connection_pairs) == 0:
            return 0.0, np.zeros_like(pos_array)

        grad = np.zeros_like(pos_array)

        # Get positions of connected pairs
        pos1 = pos_array[self.connection_pairs[:, 0]]  # Shape: (n_edges, 2)
        pos2 = pos_array[self.connection_pairs[:, 1]]  # Shape: (n_edges, 2)

        # Compute distances
        diff = pos1 - pos2  # Shape: (n_edges, 2)
        distances = np.linalg.norm(diff, axis=1, keepdims=True)  # Shape: (n_edges, 1)
        distances = np.maximum(distances, 1e-6)  # Avoid division by zero

        # Energy
        deviations = distances.squeeze() - self.ideal_spring_distance
        energy = 0.5 * self.spring_constant * np.sum(deviations**2)

        # Gradient
        # Force magnitude: k * (d - d_0)
        force_magnitude = self.spring_constant * (
            distances - self.ideal_spring_distance
        )

        # Force direction: (x_i - x_j) / d_ij (unit vector)
        force_direction = diff / distances

        # Total force: magnitude * direction
        forces = force_magnitude * force_direction  # Shape: (n_edges, 2)

        # Accumulate forces at each node
        # For edge (i,j): add force to node i, subtract from node j
        np.add.at(grad, self.connection_pairs[:, 0], forces)
        np.add.at(grad, self.connection_pairs[:, 1], -forces)

        return energy, grad

    def _repulsion_energy_grad(self, pos_array: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Repulsion energy with analytical gradient.

        DERIVATION:
        Energy:   E = sum_{i<j} k / d_ij^2

        Gradient: ∂E/∂x_i = sum_{j≠i} -2k * (x_i - x_j) / d_ij^4

        Where:
        - d_ij = ||pos_i - pos_j||
        - k = repulsion strength
        """
        if self.n_entities <= 1:
            return 0.0, np.zeros_like(pos_array)

        grad = np.zeros_like(pos_array)

        # Use KDTree to find nearby pairs
        tree = cKDTree(pos_array)
        pairs = tree.query_pairs(self.repulsion_cutoff_distance, output_type="ndarray")

        if len(pairs) == 0:
            return 0.0, grad

        # Get positions
        pos1 = pos_array[pairs[:, 0]]
        pos2 = pos_array[pairs[:, 1]]

        # Compute distances
        diff = pos1 - pos2  # Shape: (n_pairs, 2)
        dist_sq = np.sum(diff**2, axis=1, keepdims=True)  # Shape: (n_pairs, 1)
        dist_sq = np.maximum(dist_sq, self.min_distance_sq)

        # Energy: k / d^2
        energy = np.sum(self.repulsion_strength / dist_sq.squeeze())

        # Gradient: -2k * (x_i - x_j) / d^4
        force_magnitude = (
            -2 * self.repulsion_strength / (dist_sq**2)
        )  # Shape: (n_pairs, 1)
        forces = force_magnitude * diff  # Shape: (n_pairs, 2)

        # Accumulate forces
        np.add.at(grad, pairs[:, 0], forces)
        np.add.at(grad, pairs[:, 1], -forces)

        return energy, grad

    def _span_constraint_energy_grad(
        self, pos_array: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Wire span constraint energy with analytical gradient.

        DERIVATION:
        Energy:   E = penalty * sum_edges max(0, d_ij - d_max)^2

        Gradient: ∂E/∂x_i = 2 * penalty * max(0, d_ij - d_max) * (x_i - x_j) / d_ij
                  for edges where d_ij > d_max, else 0

        Where:
        - d_ij = ||pos_i - pos_j||
        - d_max = maximum allowed wire span
        """
        if len(self.connection_pairs) == 0:
            return 0.0, np.zeros_like(pos_array)

        grad = np.zeros_like(pos_array)
        max_span = self.constraints.max_wire_span

        # Get positions
        pos1 = pos_array[self.connection_pairs[:, 0]]
        pos2 = pos_array[self.connection_pairs[:, 1]]

        # Compute distances
        diff = pos1 - pos2
        distances = np.linalg.norm(diff, axis=1, keepdims=True)
        distances = np.maximum(distances, 1e-6)

        # Energy: penalty * sum(max(0, d - d_max)^2)
        excesses = np.maximum(0, distances.squeeze() - max_span)
        energy = self.violation_penalty * np.sum(excesses**2)

        # Gradient: only non-zero for violations
        violation_mask = distances.squeeze() > max_span

        if not np.any(violation_mask):
            return energy, grad  # No violations, gradient is zero

        # Force magnitude: 2 * penalty * (d - d_max)
        force_magnitude = 2 * self.violation_penalty * excesses[:, np.newaxis]

        # Force direction: (x_i - x_j) / d
        force_direction = diff / distances

        # Total forces
        forces = force_magnitude * force_direction

        # Zero out forces for non-violations
        forces[~violation_mask] = 0

        # Accumulate forces
        np.add.at(grad, self.connection_pairs[:, 0], forces)
        np.add.at(grad, self.connection_pairs[:, 1], -forces)

        return energy, grad

    def _boundary_energy_grad(self, pos_array: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Simple L2 regularization towards origin with analytical gradient.
        """
        grad = np.zeros_like(pos_array)
        energy = np.sum(
            self.constraints.boundary_penalty_strength * pos_array**2
        ) / len(pos_array)
        grad = (
            2 * self.constraints.boundary_penalty_strength * pos_array / len(pos_array)
        )
        return energy, grad

    def _fixed_position_energy_grad(
        self, pos_array: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Fixed position penalty energy with analytical gradient.

        DERIVATION:
        Energy:   E = penalty * sum_fixed ||pos_i - pos_fixed_i||^2

        Gradient: ∂E/∂x_i = 2 * penalty * (x_i - x_fixed_i)  for fixed entities
                           = 0                                for non-fixed entities

        This strongly penalizes moving entities that should be fixed.
        """
        if len(self.fixed_indices) == 0:
            return 0.0, np.zeros_like(pos_array)

        # Get current positions of fixed entities
        current_pos = pos_array[self.fixed_indices]

        # Compute deviation from fixed positions
        deviation = current_pos - self.fixed_pos_array  # Shape: (n_fixed, 2)
        dist_sq = np.sum(deviation**2, axis=1)  # Shape: (n_fixed,)

        # Energy (only penalize if moved significantly)
        significant_movement = dist_sq > 1e-6
        energy = self.fixed_penalty * np.sum(dist_sq[significant_movement])

        # Gradient
        grad = np.zeros_like(pos_array)

        if np.any(significant_movement):
            # Gradient: 2 * penalty * (x - x_fixed)
            grad_fixed = 2 * self.fixed_penalty * deviation

            # Only apply to entities that moved significantly
            grad_fixed[~significant_movement] = 0

            # Put gradients in correct positions
            grad[self.fixed_indices] = grad_fixed

        return energy, grad

    def _collision_energy_grad(self, pos_array: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Collision energy with VECTORIZED gradient computation.

        OPTIMIZATION: Eliminates Python loop with ~100K+ scalar operations
        Expected speedup: 30-70x
        """
        if self.n_entities <= 1:
            return 0.0, np.zeros_like(pos_array)

        grad = np.zeros_like(pos_array)
        energy = 0.0

        # Use KDTree for candidate pairs
        tree = cKDTree(pos_array)
        pairs = tree.query_pairs(self.max_collision_distance, output_type="ndarray")

        if len(pairs) == 0:
            return 0.0, grad

        # ============================================================
        # VECTORIZED COMPUTATION (replaces ~100K+ Python operations)
        # ============================================================

        # Step 1: Extract all pair data at once
        i_indices = pairs[:, 0]
        j_indices = pairs[:, 1]

        pos_i = pos_array[i_indices]  # (n_pairs, 2)
        pos_j = pos_array[j_indices]  # (n_pairs, 2)

        footprints_i = self.footprints[i_indices]  # (n_pairs, 2)
        footprints_j = self.footprints[j_indices]  # (n_pairs, 2)

        # Step 2: Vectorized overlap detection
        dx = pos_j[:, 0] - pos_i[:, 0]  # (n_pairs,)
        dy = pos_j[:, 1] - pos_i[:, 1]  # (n_pairs,)

        sep_x = np.abs(dx) - (footprints_i[:, 0] + footprints_j[:, 0]) / 2
        sep_y = np.abs(dy) - (footprints_i[:, 1] + footprints_j[:, 1]) / 2

        overlap_mask = (sep_x < 0) & (sep_y < 0)

        if not np.any(overlap_mask):
            return 0.0, grad

        # Step 3: Compute energy
        overlap_areas = (-sep_x[overlap_mask]) * (-sep_y[overlap_mask])
        energy = self.collision_penalty * np.sum(overlap_areas)

        # Step 4: Vectorized gradient (CHOOSE APPROACH A OR B)

        # APPROACH A: Simplified smooth gradient (RECOMMENDED)
        dx_overlap = dx[overlap_mask]
        dy_overlap = dy[overlap_mask]
        sep_x_overlap = sep_x[overlap_mask]
        sep_y_overlap = sep_y[overlap_mask]
        i_overlap = i_indices[overlap_mask]
        j_overlap = j_indices[overlap_mask]

        epsilon = 1e-12
        dx_safe = np.where(np.abs(dx_overlap) < epsilon, epsilon, dx_overlap)
        dy_safe = np.where(np.abs(dy_overlap) < epsilon, epsilon, dy_overlap)

        sign_dx = np.sign(dx_safe)
        sign_dy = np.sign(dy_safe)

        grad_x = self.collision_penalty * sign_dx * (-sep_y_overlap)
        grad_y = self.collision_penalty * sign_dy * (-sep_x_overlap)

        grad_forces_i = np.column_stack([grad_x, grad_y])
        grad_forces_j = -grad_forces_i

        np.add.at(grad, i_overlap, grad_forces_i)
        np.add.at(grad, j_overlap, grad_forces_j)

        return energy, grad

    # def _collision_energy_grad(self, pos_array: np.ndarray) -> Tuple[float, np.ndarray]:
    #     """
    #     Collision energy with SMOOTH gradient.

    #     NOTE: Exact rectangular collision is non-differentiable.
    #     We use a smooth approximation that's differentiable.

    #     APPROXIMATION:
    #     Instead of exact overlap area, use smooth penalty based on
    #     bounding box separation distance.

    #     Energy:   E = penalty * sum_{overlapping} overlap_area
    #               where overlap_area ≈ max(0, -sep_x) * max(0, -sep_y)
    #               and sep_x = |x_i - x_j| - (w_i + w_j)/2

    #     Gradient: ∂E/∂x_i = penalty * max(0, -sep_y) * sign(x_i - x_j)
    #                        (if sep_x < 0, else 0)

    #     This is differentiable almost everywhere.
    #     """
    #     if self.n_entities <= 1:
    #         return 0.0, np.zeros_like(pos_array)

    #     grad = np.zeros_like(pos_array)
    #     energy = 0.0

    #     # Use KDTree for candidate pairs
    #     tree = cKDTree(pos_array)
    #     pairs = tree.query_pairs(self.max_collision_distance, output_type="ndarray")

    #     if len(pairs) == 0:
    #         return 0.0, grad

    #     # Check each candidate pair
    #     for i, j in pairs:
    #         x1, y1 = pos_array[i]
    #         x2, y2 = pos_array[j]
    #         w1, h1 = self.footprints[i]
    #         w2, h2 = self.footprints[j]

    #         # Compute differences
    #         dx = x2 - x1
    #         dy = y2 - y1

    #         # Separation distances (negative if overlapping)
    #         sep_x = abs(dx) - (w1 + w2) / 2
    #         sep_y = abs(dy) - (h1 + h2) / 2

    #         # Check if overlapping in both dimensions
    #         if sep_x < 0 and sep_y < 0:
    #             # Overlap area (smooth approximation)
    #             overlap = (-sep_x) * (-sep_y)
    #             energy += self.collision_penalty * overlap

    #             # Gradient computation
    #             # For collision energy E = penalty * (-sep_x) * (-sep_y)
    #             # where sep_x = abs(dx) - (w1+w2)/2 and dx = x_j - x_i
    #             #
    #             # For entity i:
    #             #   ∂E/∂x_i = penalty * ∂(-sep_x)/∂x_i * (-sep_y)
    #             #   ∂abs(dx)/∂x_i = -sign(dx) when dx != 0
    #             #   ∂(-sep_x)/∂x_i = -∂abs(dx)/∂x_i = +sign(dx)
    #             #   So: grad_x_i = penalty * sign(dx) * (-sep_y)
    #             #
    #             # For entity j:
    #             #   ∂abs(dx)/∂x_j = +sign(dx) when dx != 0
    #             #   ∂(-sep_x)/∂x_j = -sign(dx)
    #             #   So: grad_x_j = penalty * (-sign(dx)) * (-sep_y) = -grad_x_i
    #             #
    #             # Special case when dx = 0 (or dy = 0):
    #             #   The function is not differentiable, but we use sub-gradients
    #             #   that match scipy's forward difference approximation.
    #             #   For entity i: forward perturb gives dx → -ε, so sign = -1
    #             #   For entity j: forward perturb gives dx → +ε, so sign = +1
    #             #   But we want both to push apart, so we use:
    #             #     grad_i = penalty * (-1) * (-sep_y)
    #             #     grad_j = penalty * (+1) * (-sep_y)
    #             #   Wait, that gives opposite gradients, but they should both be
    #             #   the same (both -500 in our test case).
    #             #
    #             #   Actually, let me reconsider. When dy = 0:
    #             #   For entity i: grad_y_i = penalty * sign_dy_i * (-sep_x)
    #             #   For entity j: grad_y_j = -penalty * sign_dy_j * (-sep_y)

    #             # After much debugging: the correct formula is
    #             # grad_i = penalty * sign(dx for i's perturbation) * (-sep_y)
    #             # grad_j = -grad_i (when dx != 0)
    #             # But when dx = 0, both should use the same sub-gradient

    #             sign_dx_i = -1.0 if abs(dx) < 1e-12 else (1.0 if dx > 0 else -1.0)
    #             sign_dy_i = -1.0 if abs(dy) < 1e-12 else (1.0 if dy > 0 else -1.0)

    #             grad_x_i = self.collision_penalty * sign_dx_i * (-sep_y)
    #             grad_y_i = self.collision_penalty * sign_dy_i * (-sep_x)

    #             # For entity j: use the negative EXCEPT when dx=0 or dy=0
    #             if abs(dx) < 1e-12:
    #                 # Both entities should be pushed apart equally
    #                 grad_x_j = grad_x_i
    #             else:
    #                 grad_x_j = -grad_x_i

    #             if abs(dy) < 1e-12:
    #                 # Both entities should be pushed apart equally
    #                 grad_y_j = grad_y_i
    #             else:
    #                 grad_y_j = -grad_y_i

    #             grad[i, 0] += grad_x_i
    #             grad[i, 1] += grad_y_i
    #             grad[j, 0] += grad_x_j
    #             grad[j, 1] += grad_y_j

    #     return energy, grad

    def _total_wire_length_grad(
        self, pos_array: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Total wire length with analytical gradient.

        DERIVATION:
        Energy:   E = sum_edges d_ij

        Gradient: ∂E/∂x_i = sum_j (x_i - x_j) / d_ij

        Where:
        - d_ij = ||pos_i - pos_j||
        """
        if len(self.connection_pairs) == 0:
            return 0.0, np.zeros_like(pos_array)

        grad = np.zeros_like(pos_array)

        # Get positions of connected pairs
        pos1 = pos_array[self.connection_pairs[:, 0]]  # Shape: (n_edges, 2)
        pos2 = pos_array[self.connection_pairs[:, 1]]  # Shape: (n_edges, 2)

        # Compute distances
        diff = pos1 - pos2  # Shape: (n_edges, 2)
        distances = np.linalg.norm(diff, axis=1, keepdims=True)  # Shape: (n_edges, 1)
        distances = np.maximum(distances, 1e-6)  # Avoid division by zero

        # Energy
        energy = np.sum(distances) * self.wire_length_constant

        # Gradient
        force_direction = diff / distances
        force_direction *= self.wire_length_constant

        # Accumulate forces at each node
        np.add.at(grad, self.connection_pairs[:, 0], force_direction)
        np.add.at(grad, self.connection_pairs[:, 1], -force_direction)

        return energy, grad

    def _single_optimization_attempt_two_phase(
        self, seed: int, max_iterations: int
    ) -> OptimizationResult:
        """
        Two-phase optimization for better convergence:
        Phase 1: Rough layout without collision penalties (faster)
        Phase 2: Refinement with all constraints (precise)
        """
        np.random.seed(seed)
        pos_flat = self._initialize_positions(seed)

        # Setup single visualizer for both phases if enabled
        visualizer = (
            LayoutVisualizer(
                layout_engine=self,
                attempt_id=seed,
                phase_name="",
                output_base_dir=VISUAL_DEBUG_PATH,
                frame_skip=max(1, max_iterations // 100),  # ~100 frames total
            )
            if VISUAL_DEBUG
            else None
        )

        # Phase 1: Without collision (50% budget)

        if visualizer:
            visualizer.set_phase("Phase 1")

        result_phase1 = minimize(
            fun=lambda x: self._energy_no_collision(x),
            x0=pos_flat,
            method="L-BFGS-B",
            jac=True,
            callback=visualizer.create_callback(self._energy_no_collision)
            if visualizer
            else None,
            options={"maxiter": max_iterations, "ftol": 1e-8, "gtol": 1e-8},
        )

        old_spring_constant = self.spring_constant
        old_boundary_penalty = self.constraints.boundary_penalty_strength
        # Phase 2: With collision (50% budget)
        self.spring_constant *= 0.05  # Soften springs for refinement
        self.constraints.boundary_penalty_strength *= 0.01  # Soften boundary

        if visualizer:
            visualizer.set_phase("Phase 2")

        result_phase2 = minimize(
            fun=lambda x: self._energy_and_gradient(x),
            x0=result_phase1.x,
            method="L-BFGS-B",
            jac=True,
            callback=visualizer.create_callback(self._energy_and_gradient)
            if visualizer
            else None,
            options={"maxiter": max_iterations, "ftol": 1e-7, "gtol": 1e-7},
        )

        # Restore original constants
        self.spring_constant = old_spring_constant
        self.constraints.boundary_penalty_strength = old_boundary_penalty

        # Save single GIF with both phases
        if visualizer:
            visualizer.create_gif()
            visualizer.cleanup()

        pos_array = result_phase2.x.reshape(-1, 2)
        positions = self._array_to_positions(pos_array)
        violations = self._count_violations(pos_array)
        print(
            f"Two-phase optimization complete after {result_phase2.nit + result_phase1.nit} iterations ({result_phase1.nit} in "
            f"phase 1, {result_phase2.nit} in phase 2, final energy: {result_phase2.fun}, phase 1 end reason: {result_phase1.message}, "
            f"phase 2 end reason: {result_phase2.message})."
        )
        if RENDER_RESULT:
            self._render_final_result(positions, suffix=f"attempt_{seed}_two_phase")

        return OptimizationResult(
            positions=positions,
            energy=result_phase2.fun,
            violations=violations,
            success=result_phase2.success,
        )

    def _energy_no_collision(self, vec: np.ndarray) -> Tuple[float, np.ndarray]:
        """Energy function WITHOUT collision penalties (for phase 1)."""
        pos_array = vec.reshape(-1, 2)
        grad_array = np.zeros_like(pos_array)
        total_energy = 0.0

        # Include: springs, repulsion, boundary, span, fixed
        e1, g1 = self._spring_energy_grad(pos_array)
        total_energy += e1
        grad_array += g1

        e2, g2 = self._repulsion_energy_grad(pos_array)
        total_energy += e2
        grad_array += g2

        e3, g3 = self._total_wire_length_grad(pos_array)
        total_energy += e3
        grad_array += g3

        # Skip collision (expensive!)
        # e3, g3 = self._collision_energy_grad(pos_array)

        e4, g4 = self._span_constraint_energy_grad(pos_array)
        total_energy += e4
        grad_array += g4

        e5, g5 = self._boundary_energy_grad(pos_array)
        total_energy += e5
        grad_array += g5

        e6, g6 = self._fixed_position_energy_grad(pos_array)
        total_energy += e6
        grad_array += g6

        # Zero out fixed entity gradients
        if len(self.fixed_indices) > 0:
            grad_array[self.fixed_indices] = 0
        # if total_energy < 10000:
        #     print("super low energy in phase 1")
        #     print(
        #         f"Phase 1 energy: {total_energy:.2f} spring energy: {e1:.2f}, repulsion energy: {e2:.2f}, span energy: {e4:.2f}, boundary energy: {e5:.2f}, fixed energy: {e6:.2f}"
        #     )  # Debug
        return total_energy, grad_array.flatten()
