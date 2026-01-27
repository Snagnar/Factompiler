from dataclasses import dataclass
from typing import Any

import numpy as np
from ortools.sat.python import cp_model

from dsl_compiler.src.common.constants import DEFAULT_CONFIG, CompilerConfig
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics

from .layout_plan import EntityPlacement
from .signal_graph import SignalGraph


class EarlyStopCallback(cp_model.CpSolverSolutionCallback):
    """Callback to stop solver early when optimization plateaus.

    The callback tracks objective value improvements across solutions. It stops
    when:
    1. A solution with zero violations is found AND
    2. The objective has not improved significantly for several consecutive solutions

    This allows the solver to continue optimizing wire length even after finding
    a feasible (zero-violation) solution.
    """

    def __init__(
        self,
        max_violations: int = 0,
        plateau_threshold: int = 10,
        min_improvement: float = 0.005,
    ):
        """Initialize the callback.

        Args:
            max_violations: Maximum acceptable violations for early stopping
            plateau_threshold: Stop after this many solutions without improvement
            min_improvement: Minimum relative improvement to count as "improving"
        """
        super().__init__()
        self._solution_count = 0
        self._max_violations = max_violations
        self._plateau_threshold = plateau_threshold
        self._min_improvement = min_improvement
        self._best_objective = float("inf")
        self._solutions_without_improvement = 0
        self._has_feasible_solution = False
        self._stop_search = False

    def on_solution_callback(self) -> None:
        self._solution_count += 1
        obj = self.ObjectiveValue()

        # Check for improvement
        if self._best_objective > 0:
            improvement = (self._best_objective - obj) / self._best_objective
        else:
            improvement = 1.0 if obj < self._best_objective else 0.0

        if improvement >= self._min_improvement:
            # Significant improvement - reset plateau counter
            self._best_objective = obj
            self._solutions_without_improvement = 0
        else:
            # No significant improvement
            self._solutions_without_improvement += 1

        # Check if solution is feasible (has acceptable violations)
        # violation_weight is 10000, so we can check thresholds
        violation_threshold = 10000 * (self._max_violations + 1)
        if obj < violation_threshold:
            self._has_feasible_solution = True

        # Stop only if we have a feasible solution AND optimization has plateaued
        if (
            self._has_feasible_solution
            and self._solutions_without_improvement >= self._plateau_threshold
        ):
            self._stop_search = True
            self.StopSearch()

    @property
    def solution_count(self) -> int:
        return self._solution_count


@dataclass
class LayoutConstraints:
    """Constraints for layout optimization."""

    max_wire_span: float = 9.0
    max_coordinate: int = 500  # Will be overridden by config if provided


@dataclass
class OptimizationResult:
    """Result of layout optimization attempt.

    Note: positions are TILE positions (top-left corner on integer grid),
    NOT center positions. They must be converted to center coordinates
    before use in Draftsman/Factorio.
    """

    positions: dict[str, tuple[int, int]]  # entity_id -> (tile_x, tile_y)
    violations: int
    total_wire_length: int
    success: bool
    strategy_used: str
    solve_time: float


class IntegerLayoutEngine:
    """
    Integer-grid layout optimization using constraint programming.

    Uses Google OR-Tools CP-SAT solver to place entities on integer grid
    while respecting hard constraints (no overlaps) and soft constraints
    (wire span limits).
    """

    def __init__(
        self,
        signal_graph: SignalGraph,
        entity_placements: dict[str, EntityPlacement],
        diagnostics: ProgramDiagnostics,
        constraints: LayoutConstraints | None = None,
        config: CompilerConfig = DEFAULT_CONFIG,
        wire_merge_junctions: dict[str, dict] | None = None,
    ):
        self.signal_graph = signal_graph
        self.entity_placements = entity_placements
        self.diagnostics = diagnostics
        self.config = config
        self.wire_merge_junctions = wire_merge_junctions or {}
        if constraints is not None:
            self.constraints = constraints
        else:
            self.constraints = LayoutConstraints(max_coordinate=config.max_layout_coordinate)

        self.entity_ids = sorted(entity_placements.keys())  # Sort for deterministic order
        self.n_entities = len(self.entity_ids)

        # Identify power pole entities (for excluding from bounding box)
        self._power_pole_ids = set()
        for entity_id, placement in entity_placements.items():
            if placement.properties.get("is_power_pole"):
                self._power_pole_ids.add(entity_id)

        self._build_connectivity()
        self._identify_fixed_positions()
        self._extract_footprints()
        self._identify_star_topologies()

    def _build_connectivity(self) -> None:
        """Build connectivity as list of (source_id, sink_id) pairs.

        Wire merge nodes are expanded: if a source feeds into a wire merge,
        edges are created from that source to all sinks of the wire merge.
        This ensures the layout optimizer considers the full wire merge topology.
        """
        self.connections = []
        self._source_to_sinks: dict[str, list[str]] = {}  # Track source fanout

        # First, build a map of wire merge node -> its sinks
        wire_merge_sinks: dict[str, list[str]] = {}
        for _signal_id, source_id, sink_id in self.signal_graph.iter_source_sink_pairs():
            if source_id in self.wire_merge_junctions:
                if source_id not in wire_merge_sinks:
                    wire_merge_sinks[source_id] = []
                if sink_id in self.entity_ids:
                    wire_merge_sinks[source_id].append(sink_id)

        # Also build a map of wire merge node -> its sources (actual entity IDs)
        wire_merge_sources: dict[str, list[str]] = {}
        for merge_id, merge_info in self.wire_merge_junctions.items():
            inputs = merge_info.get("inputs", [])
            sources = []
            for input_ref in inputs:
                # Handle both SignalRef and BundleRef
                if hasattr(input_ref, "source_id"):
                    ir_source_id = input_ref.source_id
                    # Resolve IR node ID to actual entity ID using signal graph
                    actual_entity_id = self.signal_graph.get_source(ir_source_id)
                    if actual_entity_id is None:
                        actual_entity_id = ir_source_id
                    if actual_entity_id in self.entity_ids:
                        sources.append(actual_entity_id)
            wire_merge_sources[merge_id] = sources

        for _signal_id, source_id, sink_id in self.signal_graph.iter_source_sink_pairs():
            # Case 1: Both source and sink are real entities
            if source_id in self.entity_ids and sink_id in self.entity_ids:
                self.connections.append((source_id, sink_id))
                self._source_to_sinks.setdefault(source_id, []).append(sink_id)

            # Case 2: Source feeds into a wire merge (sink is wire merge node)
            elif source_id in self.entity_ids and sink_id in self.wire_merge_junctions:
                # Connect source to all sinks of the wire merge
                for actual_sink in wire_merge_sinks.get(sink_id, []):
                    self.connections.append((source_id, actual_sink))
                    self._source_to_sinks.setdefault(source_id, []).append(actual_sink)

            # Case 3: Wire merge feeds into a sink
            elif source_id in self.wire_merge_junctions and sink_id in self.entity_ids:
                # Connect all wire merge inputs to this sink
                for actual_source in wire_merge_sources.get(source_id, []):
                    self.connections.append((actual_source, sink_id))
                    self._source_to_sinks.setdefault(actual_source, []).append(sink_id)

        # Deduplicate and sort for deterministic iteration order
        self.connections = sorted(set(self.connections))

        self.diagnostics.info(f"Built connectivity: {len(self.connections)} unique connections")

    def _identify_fixed_positions(self) -> None:
        """Identify entities with fixed positions (user-specified or grid-placed).

        Fixed positions include:
        - User-specified positions (user_specified_position=True)
        - Grid-placed power poles (fixed_position=True)

        COORDINATE SYSTEM NOTES:
        - User-specified positions from place() calls are stored as TILE positions
          (the top-left corner of the entity).
        - Power poles (fixed_position=True) are stored as CENTER positions.

        We normalize both to tile positions for the solver, which works on an
        integer grid where positions represent top-left corners.
        """
        self.fixed_positions = {}

        for entity_id, placement in self.entity_placements.items():
            if placement.position is not None:
                # Check if position is marked as fixed
                is_user_specified = placement.properties.get("user_specified_position")
                is_fixed = placement.properties.get("fixed_position")

                if is_user_specified or is_fixed:
                    footprint = placement.properties.get("footprint", (1, 1))
                    width, height = footprint

                    if is_user_specified:
                        tile_x, tile_y = (
                            int(placement.position[0]),
                            int(placement.position[1]),
                        )
                    else:
                        # Power poles and other fixed entities store center positions
                        center_x, center_y = placement.position
                        tile_x = int(round(center_x - width / 2.0))
                        tile_y = int(round(center_y - height / 2.0))

                    self.fixed_positions[entity_id] = (tile_x, tile_y)

        self.diagnostics.info(f"Found {len(self.fixed_positions)} fixed positions")

    def _extract_footprints(self) -> None:
        """Extract entity footprints (width, height) as integers."""
        self.footprints = {}

        for entity_id in self.entity_ids:
            placement = self.entity_placements[entity_id]
            footprint = placement.properties.get("footprint")

            if footprint:
                width, height = footprint
                self.footprints[entity_id] = (int(np.ceil(width)), int(np.ceil(height)))
            else:
                self.footprints[entity_id] = (1, 1)

    def _get_downstream_fixed_position(self, entity_id: str) -> tuple[int, int] | None:
        """Get the fixed position of a downstream entity (if any).

        For entities that connect to user-placed entities (like lamps), this returns
        the fixed position of that downstream entity. This is used to sort star
        topology sinks by their actual spatial relationship.

        Args:
            entity_id: The entity to find downstream fixed position for

        Returns:
            (x, y) tuple of the downstream fixed position, or None if not found
        """
        # Check if this entity directly connects to any fixed-position entity
        for source, sink in self.connections:
            if source == entity_id and sink in self.fixed_positions:
                return self.fixed_positions[sink]
        return None

    def _sort_sinks_by_downstream_position(self, sinks: list[str]) -> list[str]:
        """Sort star topology sinks by their downstream fixed positions.

        For each sink, we look at what fixed-position entity (like a lamp) it
        connects to, and sort by that position. This ensures that combinators
        feeding adjacent lamps are considered adjacent in the MST model.

        Falls back to alphabetical sorting for sinks without downstream fixed
        positions (maintaining deterministic behavior).

        Args:
            sinks: List of sink entity IDs to sort

        Returns:
            Sorted list of sink entity IDs
        """

        def sort_key(sink_id: str) -> tuple:
            downstream_pos = self._get_downstream_fixed_position(sink_id)
            if downstream_pos is not None:
                # Primary sort by x, secondary by y, then by ID for ties
                return (0, downstream_pos[0], downstream_pos[1], sink_id)
            else:
                # No downstream fixed position - sort after positioned ones, by ID
                return (1, 0, 0, sink_id)

        return sorted(sinks, key=sort_key)

    def _identify_star_topologies(self) -> None:
        """Identify star topologies that will use MST routing.

        Star topologies are sources with multiple sinks (fanout >= MST_THRESHOLD).
        For these, the actual wiring uses minimum spanning tree (MST) instead of
        direct point-to-point connections, so we need to model their wire cost
        differently in the optimization objective.

        Creates:
        - self._star_sources: Set of source IDs with star topology
        - self._mst_connections: List of (entity_a, entity_b) edges for MST model
        - self._direct_connections: List of (source, sink) for 1-to-1 connections
        """
        MST_THRESHOLD = 3  # Minimum fanout to consider as star topology

        self._star_sources: set[str] = set()
        self._mst_connections: list[tuple[str, str]] = []
        self._direct_connections: list[tuple[str, str]] = []

        # Identify star sources
        for source_id, sinks in self._source_to_sinks.items():
            unique_sinks = list(set(sinks))
            if len(unique_sinks) >= MST_THRESHOLD:
                self._star_sources.add(source_id)

        if self._star_sources:
            self.diagnostics.info(
                f"Identified {len(self._star_sources)} star sources for MST-aware optimization"
            )

        # Partition connections into star (MST) vs direct
        for source, sink in self.connections:
            if source in self._star_sources:
                # This is part of a star topology - will be handled by MST model
                pass  # Don't add to direct connections
            else:
                self._direct_connections.append((source, sink))

        # For star sources, create MST-like connections:
        # Instead of N edges from source to each sink, model as:
        # - Edges between sinks (to form a chain/tree)
        # - One edge from source to the sink cluster
        for source_id in self._star_sources:
            sinks = list(set(self._source_to_sinks.get(source_id, [])))
            if len(sinks) < 2:
                continue

            # Model the MST within sinks as pairwise connections between
            # "adjacent" sinks. Since we don't know positions yet, we'll
            # create a spanning structure that encourages linear arrangement.
            #
            # CRITICAL: Sort sinks by their downstream fixed position (e.g., the lamp
            # they connect to), NOT by entity ID. Alphabetical sorting causes issues
            # when entity IDs like "arith_10" and "arith_101" are placed adjacent even
            # though they connect to lamps at x=0 and x=13 respectively.
            sorted_sinks = self._sort_sinks_by_downstream_position(sinks)
            for i in range(len(sorted_sinks) - 1):
                self._mst_connections.append((sorted_sinks[i], sorted_sinks[i + 1]))

            # Add one edge from source to the first sink (source connects to chain)
            self._mst_connections.append((source_id, sorted_sinks[0]))

        self.diagnostics.info(
            f"Connection partitioning: {len(self._direct_connections)} direct, "
            f"{len(self._mst_connections)} MST edges"
        )

    def optimize(self, time_limit_seconds: int = 60) -> dict[str, tuple[int, int]]:
        """
        Optimize layout with progressive relaxation strategy.

        Uses early stopping when a good-enough solution is found. For small
        graphs, tries a quick solve first before falling back to longer solves.

        The solver uses a violation progression where earlier (stricter) stages
        aim for fewer violations, while later stages accept more. This allows
        the solver to try harder for optimal solutions before accepting
        compromises.

        Args:
            time_limit_seconds: Total time budget for all strategies

        Returns:
            Dict mapping entity_id to (x, y) integer coordinates
        """

        self.diagnostics.info(
            f"Starting integer layout optimization: {self.n_entities} entities, "
            f"{len(self.connections)} connections"
        )

        if self.n_entities == 0:
            return {}

        if self.n_entities > 500:
            self.diagnostics.info(
                f"Large graph detected ({self.n_entities} entities), using subgraph decomposition"
            )
            return self._optimize_with_decomposition(time_limit_seconds)

        strategies = self._get_relaxation_strategies()

        # For small/medium graphs, try a very quick solve first with early stopping
        # This handles the common case where solutions are found in < 1 second
        if self.n_entities <= 20:
            quick_result = self._solve_with_strategy(strategies[0], time_limit=1, early_stop=True)
            if quick_result.success and quick_result.violations == 0:
                self.diagnostics.info(f"Quick solution found in {quick_result.solve_time:.2f}s")
                return quick_result.positions
            # Use strategy-specific threshold for early acceptance
            max_acceptable = strategies[0].get("max_acceptable_violations", 0)
            if quick_result.success and quick_result.violations <= max_acceptable:
                self.diagnostics.info(
                    f"Quick acceptable solution found with {quick_result.violations} violations "
                    f"in {quick_result.solve_time:.2f}s"
                )
                return quick_result.positions

        best_result = None
        # Distribute time budget across strategies, but give more to earlier (stricter) ones
        # Early stopping will handle convergence, so we don't need to worry about wasting time
        strict_time = max(5, time_limit_seconds // 2)  # Half the budget for strict strategy
        remaining_time = time_limit_seconds - strict_time
        remaining_per_strategy = (
            max(1, remaining_time // (len(strategies) - 1)) if len(strategies) > 1 else 0
        )

        for i, strategy in enumerate(strategies):
            per_strategy_limit = strict_time if i == 0 else remaining_per_strategy

            self.diagnostics.info(
                f"Attempting strategy {i + 1}/{len(strategies)}: {strategy['name']}"
            )

            result = self._solve_with_strategy(strategy, per_strategy_limit, early_stop=True)

            if result.success and result.violations == 0:
                self.diagnostics.info(
                    f"Perfect solution found with strategy '{strategy['name']}' "
                    f"in {result.solve_time:.2f}s"
                )
                return result.positions

            if result.success:
                if best_result is None or result.violations < best_result.violations:
                    best_result = result

                # Use strategy-specific threshold from progression
                max_acceptable = strategy.get(
                    "max_acceptable_violations", self.config.acceptable_layout_violations
                )
                if result.violations <= max_acceptable:
                    self.diagnostics.info(
                        f"Acceptable solution found with {result.violations} violations "
                        f"using strategy '{strategy['name']}' in {result.solve_time:.2f}s"
                    )
                    return result.positions
                else:
                    self.diagnostics.info(
                        f"Strategy '{strategy['name']}' produced "
                        f"{result.violations} violations, continuing"
                    )
            else:
                self.diagnostics.info(
                    f"Strategy '{strategy['name']}' failed to find feasible solution"
                )

        if best_result and best_result.success:
            self.diagnostics.warning(
                f"Best solution has {best_result.violations} violations "
                f"(strategy: {best_result.strategy_used})"
            )
            self._report_violations(best_result)
            return best_result.positions

        self._diagnose_failure()
        return self._fallback_grid_layout()

    def _get_relaxation_strategies(self) -> list[dict]:
        """Define progressive relaxation strategies.

        Each strategy has:
        - max_span: Wire span limit for violation detection
        - max_coord: Maximum coordinate for entity placement
        - violation_weight: Weight for violations in objective function
        - max_acceptable_violations: Threshold for early stopping

        The strict strategy uses the firm wire span limit (9.0 tiles in Factorio).
        No safety margins are applied - if Euclidean distance exceeds 9.0, it's
        a violation that requires relay poles.
        """
        # Use the firm wire span limit - no safety margins
        max_span = int(self.constraints.max_wire_span)
        max_coord = self.constraints.max_coordinate

        # Get violation progression from config, with fallback defaults
        progression = self.config.violation_progression
        if len(progression) < 5:
            # Extend with increasing values if too short
            progression = progression + tuple(range(len(progression), 5))

        return [
            {
                "name": "Strict",
                "max_span": max_span,  # Firm 9.0 limit
                "max_coord": max_coord,
                "violation_weight": 10000,
                "max_acceptable_violations": progression[0],
            },
            {
                "name": "Relaxed span (+33%)",
                "max_span": int(max_span * 1.33),  # Allow longer spans to find feasible layout
                "max_coord": max_coord,
                "violation_weight": 10000,
                "max_acceptable_violations": progression[1],
            },
            {
                "name": "Larger area (+50%)",
                "max_span": max_span,
                "max_coord": int(max_coord * 1.5),
                "violation_weight": 10000,
                "max_acceptable_violations": progression[2],
            },
            {
                "name": "Both relaxed",
                "max_span": int(max_span * 1.5),
                "max_coord": int(max_coord * 1.5),
                "violation_weight": 5000,
                "max_acceptable_violations": progression[3],
            },
            {
                "name": "Very relaxed",
                "max_span": int(max_span * 2),  # Allow very long spans for desperate cases
                "max_coord": int(max_coord * 2),
                "violation_weight": 1000,
                "max_acceptable_violations": progression[4],
            },
            {
                # Final fallback: accept any number of violations
                # The relay router will handle long-distance connections
                "name": "Unlimited (rely on relays)",
                "max_span": max_coord,  # Effectively unlimited span
                "max_coord": int(max_coord * 2),
                "violation_weight": 100,  # Minimize violations but don't block on them
                "max_acceptable_violations": 10000,  # Accept any number
            },
        ]

    def _solve_with_strategy(
        self, strategy: dict, time_limit: int, early_stop: bool = True
    ) -> OptimizationResult:
        """Solve layout with a specific strategy.

        Args:
            strategy: Strategy configuration dict
            time_limit: Maximum time in seconds
            early_stop: If True, stop as soon as a good solution is found
        """
        model = cp_model.CpModel()

        positions = self._create_position_variables(model, strategy["max_coord"])

        # Add solution hints based on simple layout to speed up search
        self._add_solution_hints(model, positions)

        # Add hard constraint: no overlaps
        self._add_no_overlap_constraint(model, positions)

        # Add edge layout constraints (inputs north, outputs south)
        self._add_edge_layout_constraints(model, positions)

        # Add soft constraint: wire span limit
        span_violations, wire_lengths = self._add_span_constraints(
            model, positions, strategy["max_span"]
        )

        self._create_objective(
            model,
            span_violations,
            wire_lengths,
            positions,
            strategy["violation_weight"],
            strategy["max_coord"],
        )

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit)
        solver.parameters.log_search_progress = False

        if early_stop:
            # Use strategy-specific max acceptable violations for early stopping
            max_acceptable = strategy.get(
                "max_acceptable_violations", self.config.acceptable_layout_violations
            )
            callback = EarlyStopCallback(max_violations=max_acceptable)
            status = solver.Solve(model, callback)
        else:
            status = solver.Solve(model)

        return self._extract_result(
            solver, status, positions, span_violations, wire_lengths, strategy["name"]
        )

    def _create_position_variables(
        self, model: cp_model.CpModel, max_coord: int
    ) -> dict[str, tuple]:
        """Create integer position variables for all entities."""
        positions = {}

        for entity_id in self.entity_ids:
            if entity_id in self.fixed_positions:
                # Fixed position: constrain to single value
                x_fixed, y_fixed = self.fixed_positions[entity_id]
                x = model.NewIntVar(x_fixed, x_fixed, f"x_{entity_id}")
                y = model.NewIntVar(y_fixed, y_fixed, f"y_{entity_id}")
            else:
                # Free position: can be anywhere in bounds
                x = model.NewIntVar(0, max_coord, f"x_{entity_id}")
                y = model.NewIntVar(0, max_coord, f"y_{entity_id}")

            positions[entity_id] = (x, y)

        return positions

    def _add_solution_hints(self, model: cp_model.CpModel, positions: dict) -> None:
        """Add solution hints based on a simple grid layout to speed up search.

        Solution hints give the solver a starting point, which can dramatically
        speed up finding an initial feasible solution.
        """
        # Compute a simple grid layout as hint
        grid_size = max(1, int(np.ceil(np.sqrt(self.n_entities))))
        spacing = 3  # Compact spacing for hints

        idx = 0
        for entity_id in self.entity_ids:
            if entity_id in self.fixed_positions:
                # Fixed positions already constrained, no hint needed
                continue

            x_var, y_var = positions[entity_id]
            row = idx // grid_size
            col = idx % grid_size
            hint_x = col * spacing
            hint_y = row * spacing

            model.AddHint(x_var, hint_x)
            model.AddHint(y_var, hint_y)
            idx += 1

    def _add_no_overlap_constraint(self, model: cp_model.CpModel, positions: dict) -> None:
        """Add hard no-overlap constraint using AddNoOverlap2D."""
        x_intervals = []
        y_intervals = []

        for entity_id in self.entity_ids:
            x, y = positions[entity_id]
            width, height = self.footprints[entity_id]

            x_interval = model.NewFixedSizeIntervalVar(x, width, f"x_int_{entity_id}")
            y_interval = model.NewFixedSizeIntervalVar(y, height, f"y_int_{entity_id}")

            x_intervals.append(x_interval)
            y_intervals.append(y_interval)

        model.AddNoOverlap2D(x_intervals, y_intervals)

    def _add_span_constraints(
        self, model: cp_model.CpModel, positions: dict, max_span: int
    ) -> tuple[list, list]:
        """Add soft wire span constraints with violation tracking.

        Uses squared Euclidean distance for BOTH violation detection AND
        wire length optimization. This matches how Factorio actually calculates
        wire spans - wires are drawn as straight lines, not along grid axes.

        For star topologies (sources with high fanout), we use MST-aware modeling:
        - Instead of N edges from source to each sink, we model:
          - Edges between sinks (to encourage linear arrangement)
          - One edge from source to the sink cluster
        - This better reflects actual MST wire costs

        NOTE: Wire lengths for fixed-to-fixed connections are excluded from the
        optimization objective since they can't be changed.
        """
        span_violations = []
        wire_lengths = []
        max_span_squared = max_span * max_span

        # Combine direct connections and MST connections for optimization
        # Direct connections: 1-to-1 edges (source with fanout < threshold)
        # MST connections: edges modeling MST topology for star sources
        all_optimization_edges = list(self._direct_connections) + list(self._mst_connections)

        for i, (entity_a, entity_b) in enumerate(all_optimization_edges):
            if entity_a not in positions or entity_b not in positions:
                continue  # Skip if entity not in model

            x1, y1 = positions[entity_a]
            x2, y2 = positions[entity_b]

            # Check if both entities are fixed (connection can't be optimized)
            a_fixed = entity_a in self.fixed_positions
            b_fixed = entity_b in self.fixed_positions
            is_fixed_to_fixed = a_fixed and b_fixed

            # Compute absolute differences for distance calculations
            dx = model.NewIntVar(0, max_span * 2, f"dx_{i}")
            dy = model.NewIntVar(0, max_span * 2, f"dy_{i}")
            model.AddAbsEquality(dx, x1 - x2)
            model.AddAbsEquality(dy, y1 - y2)

            # Squared Euclidean distance for both optimization and violation detection
            dx_squared = model.NewIntVar(0, max_span_squared * 4, f"dx2_{i}")
            dy_squared = model.NewIntVar(0, max_span_squared * 4, f"dy2_{i}")
            model.AddMultiplicationEquality(dx_squared, [dx, dx])
            model.AddMultiplicationEquality(dy_squared, [dy, dy])

            distance_squared = model.NewIntVar(0, max_span_squared * 8, f"dist2_{i}")
            model.Add(distance_squared == dx_squared + dy_squared)

            # Only include in wire_lengths if connection can be optimized
            if not is_fixed_to_fixed:
                wire_lengths.append(distance_squared)

            # Violation when squared Euclidean distance exceeds squared span limit
            is_violation = model.NewBoolVar(f"viol_{i}")
            model.Add(distance_squared > max_span_squared).OnlyEnforceIf(is_violation)
            model.Add(distance_squared <= max_span_squared).OnlyEnforceIf(is_violation.Not())

            span_violations.append(is_violation)

        # Also add violation tracking for ALL original connections
        # (important for star edges that were removed from optimization but still need
        # violation checking)
        for i, (source, sink) in enumerate(self.connections):
            if (source, sink) in set(all_optimization_edges):
                continue  # Already tracked above

            if source not in positions or sink not in positions:
                continue

            x1, y1 = positions[source]
            x2, y2 = positions[sink]

            dx = model.NewIntVar(0, max_span * 2, f"vdx_{i}")
            dy = model.NewIntVar(0, max_span * 2, f"vdy_{i}")
            model.AddAbsEquality(dx, x1 - x2)
            model.AddAbsEquality(dy, y1 - y2)

            dx_squared = model.NewIntVar(0, max_span_squared * 4, f"vdx2_{i}")
            dy_squared = model.NewIntVar(0, max_span_squared * 4, f"vdy2_{i}")
            model.AddMultiplicationEquality(dx_squared, [dx, dx])
            model.AddMultiplicationEquality(dy_squared, [dy, dy])

            distance_squared = model.NewIntVar(0, max_span_squared * 8, f"vdist2_{i}")
            model.Add(distance_squared == dx_squared + dy_squared)

            is_violation = model.NewBoolVar(f"vviol_{i}")
            model.Add(distance_squared > max_span_squared).OnlyEnforceIf(is_violation)
            model.Add(distance_squared <= max_span_squared).OnlyEnforceIf(is_violation.Not())

            span_violations.append(is_violation)

        return span_violations, wire_lengths

    def _add_edge_layout_constraints(self, model: cp_model.CpModel, positions: dict) -> None:
        """Add constraints for north-south edge layout.

        Strategy:
        - All inputs share Y = Y_input_line (a variable)
        - All outputs share Y = Y_output_line (a variable)
        - All intermediates have Y strictly between these lines
        - Optimizer minimizes bounding box, naturally placing:
          * Y_input_line at minimum feasible Y
          * Y_output_line at maximum feasible Y

        NOTE: We do NOT add X-ordering constraints here. The AddNoOverlap2D
        constraint already prevents overlaps, so the optimizer is free to
        reorder entities horizontally to minimize wire length.

        Args:
            model: CP-SAT constraint model
            positions: Dict mapping entity_id to (x, y) variables
        """
        # Categorize entities (skip user-fixed entities)
        input_entities = []
        output_entities = []
        intermediate_entities = []

        for entity_id in self.entity_ids:
            if entity_id in self.fixed_positions:
                continue

            placement = self.entity_placements.get(entity_id)
            if not placement:
                continue

            if placement.properties.get("is_input"):
                input_entities.append(entity_id)
            elif placement.properties.get("is_output"):
                output_entities.append(entity_id)
            else:
                intermediate_entities.append(entity_id)

        # If no inputs or outputs, no edge constraints needed
        if not input_entities and not output_entities:
            self.diagnostics.info("No edge layout constraints (no inputs/outputs marked)")
            return

        max_coord = self.constraints.max_coordinate

        # ========================================================================
        # Create shared Y-coordinate variables for input and output lines
        # ========================================================================

        Y_input_line = None
        max_input_height = 0

        if input_entities:
            Y_input_line = model.NewIntVar(0, max_coord, "Y_input_line")
            max_input_height = max(self.footprints.get(e, (1, 1))[1] for e in input_entities)
            self.diagnostics.info(
                f"Edge layout: {len(input_entities)} inputs, max height {max_input_height}"
            )

        Y_output_line = None

        if output_entities:
            Y_output_line = model.NewIntVar(0, max_coord, "Y_output_line")
            self.diagnostics.info(f"Edge layout: {len(output_entities)} outputs")

        # Ensure sufficient vertical gap between input and output lines
        if Y_input_line is not None and Y_output_line is not None:
            min_gap = max_input_height

            if intermediate_entities:
                max_intermediate_height = max(
                    self.footprints.get(e, (1, 1))[1] for e in intermediate_entities
                )
                min_gap += max_intermediate_height
            else:
                min_gap += 1  # At least 1 tile gap even with no intermediates

            model.Add(Y_output_line >= Y_input_line + min_gap)

            self.diagnostics.info(
                f"Edge layout: enforcing minimum gap of {min_gap} between input/output lines"
            )

        # Constrain inputs to share Y = Y_input_line
        # NOTE: No X-ordering - AddNoOverlap2D handles collision prevention,
        # and the optimizer will find the best horizontal arrangement.
        if input_entities and Y_input_line is not None:
            for entity_id in input_entities:
                _, y = positions[entity_id]
                model.Add(y == Y_input_line)

        # Constrain outputs to share Y = Y_output_line
        # NOTE: No X-ordering - same rationale as inputs.
        if output_entities and Y_output_line is not None:
            for entity_id in output_entities:
                _, y = positions[entity_id]
                model.Add(y == Y_output_line)

        # Constrain intermediates to be between input and output lines
        for entity_id in intermediate_entities:
            _, y = positions[entity_id]
            height = self.footprints.get(entity_id, (1, 1))[1]

            if Y_input_line is not None:
                model.Add(y >= Y_input_line + max_input_height)

            if Y_output_line is not None:
                model.Add(y + height <= Y_output_line)

    def _create_objective(
        self,
        model: cp_model.CpModel,
        span_violations: list,
        wire_lengths: list,
        positions: dict,
        violation_weight: int,
        max_coord: int,
    ) -> None:
        """Create multi-objective optimization function with priorities.

        Objective priorities (highest to lowest):
        1. Minimize span violations (connections exceeding 9 tiles)
        2. Minimize total wire length (sum of squared Euclidean distances)
        3. Minimize bounding box (perimeter = max_x + max_y)

        The weights are tuned so that:
        - A single violation is much worse than any wire length increase
        - Wire length and bounding box are balanced (both matter)

        Since wire_lengths contains squared distances (e.g., 43 connections
        with avg squared dist ~9 = 387), and bounding perimeter is typically
        20-50, we need to balance the weights accordingly.
        """
        num_violations = sum(span_violations) if span_violations else 0

        total_wire_length = sum(wire_lengths) if wire_lengths else 0

        # Exclude power poles from bounding box calculation
        non_pole_entities = [e for e in self.entity_ids if e not in self._power_pole_ids]

        if non_pole_entities:
            non_pole_x = [positions[e][0] for e in non_pole_entities]
            non_pole_y = [positions[e][1] for e in non_pole_entities]

            max_x = model.NewIntVar(0, max_coord, "max_x")
            max_y = model.NewIntVar(0, max_coord, "max_y")
            model.AddMaxEquality(max_x, non_pole_x)
            model.AddMaxEquality(max_y, non_pole_y)

            bounding_perimeter = max_x + max_y
        else:
            # Fallback if all entities are power poles (unlikely)
            bounding_perimeter_int = 0
            bounding_perimeter = model.NewIntVar(0, 0, "bounding_perimeter")
            model.Add(bounding_perimeter == bounding_perimeter_int)

        # Weight tuning:
        # - violation_weight: 10000 per violation (ensures violations are avoided first)
        # - wire_length_weight: Applied to sum of squared distances
        # - perimeter_weight: Applied to max_x + max_y
        #
        # For a typical circuit with 40 connections:
        # - Wire length sum: ~40 * 9 (avg d²) = 360, * 100 = 36,000
        # - Perimeter: ~30, * 1000 = 30,000
        # This makes them roughly equally important.
        wire_length_weight = 300
        perimeter_weight = 100

        objective = (
            violation_weight * num_violations
            + wire_length_weight * total_wire_length
            + perimeter_weight * bounding_perimeter
        )

        model.Minimize(objective)

    def _extract_result(
        self,
        solver: cp_model.CpSolver,
        status: Any,  # cp_model.CpSolverStatus
        positions: dict,
        span_violations: list,
        wire_lengths: list,
        strategy_name: str,
    ) -> OptimizationResult:
        """Extract optimization result from solved model."""
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            result_positions = {
                entity_id: (solver.Value(x), solver.Value(y))
                for entity_id, (x, y) in positions.items()
            }

            num_violations = sum(solver.Value(v) for v in span_violations) if span_violations else 0

            total_wire_length = sum(solver.Value(wl) for wl in wire_lengths) if wire_lengths else 0

            return OptimizationResult(
                positions=result_positions,
                violations=num_violations,
                total_wire_length=total_wire_length,
                success=True,
                strategy_used=strategy_name,
                solve_time=solver.WallTime(),
            )

        return OptimizationResult(
            positions={},
            violations=0,
            total_wire_length=0,
            success=False,
            strategy_used=strategy_name,
            solve_time=solver.WallTime(),
        )

    def _report_violations(self, result: OptimizationResult) -> None:
        """Report detailed information about constraint violations."""
        violated_connections = []

        for source, sink in self.connections:
            x1, y1 = result.positions[source]
            x2, y2 = result.positions[sink]
            distance = abs(x1 - x2) + abs(y1 - y2)

            if distance > self.constraints.max_wire_span:
                violated_connections.append((source, sink, distance))

        self.diagnostics.warning(f"Layout has {len(violated_connections)} wire span violations:")

        for source, sink, distance in violated_connections[:10]:
            self.diagnostics.warning(
                f"  {source} → {sink}: {distance} units (limit: {self.constraints.max_wire_span})"
            )

        if len(violated_connections) > 10:
            self.diagnostics.warning(f"  ... and {len(violated_connections) - 10} more violations")

    def _diagnose_failure(self) -> None:
        """Provide detailed diagnostic information about optimization failure."""

        error_msg = "Failed to find feasible layout with all strategies.\n"

        # Calculate areas excluding power poles for more accurate diagnostics
        non_pole_footprints = {
            k: v for k, v in self.footprints.items() if k not in self._power_pole_ids
        }
        total_entity_area = sum(w * h for w, h in non_pole_footprints.values())
        total_pole_area = sum(
            w * h for k, (w, h) in self.footprints.items() if k in self._power_pole_ids
        )
        total_area_needed = total_entity_area + total_pole_area
        available_area = self.constraints.max_coordinate**2

        error_msg += (
            f"Diagnostic information:\n"
            f"  Total entities: {self.n_entities}\n"
            f"    - Circuit entities: {len(non_pole_footprints)}\n"
            f"    - Power poles: {len(self._power_pole_ids)}\n"
            f"  Connections: {len(self.connections)}\n"
            f"  Fixed positions: {len(self.fixed_positions)}\n"
            f"  Entity area (circuit only): {total_entity_area}\n"
            f"  Power pole area: {total_pole_area}\n"
            f"  Total area needed: {total_area_needed}\n"
            f"  Available area: {available_area} ({self.constraints.max_coordinate}x{self.constraints.max_coordinate})\n"
            f"  Area utilization: {100 * total_area_needed / available_area:.1f}%"
        )

        if total_area_needed > available_area * 0.8:
            error_msg += (
                "  → Area utilization >80%: entities may not fit\n"
                "     Solution: Increase max_coordinate constraint"
            )

        if len(self.fixed_positions) > self.n_entities * 0.3:
            fixed_poles = len([p for p in self.fixed_positions if p in self._power_pole_ids])
            fixed_other = len(self.fixed_positions) - fixed_poles
            error_msg += (
                f"  → Many fixed positions ({len(self.fixed_positions)}/{self.n_entities})\n"
                f"     - Fixed power poles: {fixed_poles}\n"
                f"     - Fixed circuit entities: {fixed_other}\n"
                "     Solution: Reduce fixed position constraints or power pole grid density"
            )

        if self.connections:
            avg_degree = 2 * len(self.connections) / self.n_entities
            if avg_degree > 4:
                error_msg += (
                    f"  → Dense connectivity (avg degree: {avg_degree:.1f})\n"
                    "     Solution: Relax wire span constraint"
                )

        self.diagnostics.error(error_msg)

    def _optimize_with_decomposition(self, time_limit: int) -> dict[str, tuple[int, int]]:
        """Optimize large graphs using connected component decomposition."""
        components = self._find_connected_components()

        self.diagnostics.info(f"Decomposed graph into {len(components)} connected components")

        all_positions = {}
        current_offset_x = 0

        for i, component in enumerate(components):
            self.diagnostics.info(
                f"Optimizing component {i + 1}/{len(components)} ({len(component)} entities)"
            )

            component_set = set(component)
            component_connections = [
                (s, t) for s, t in self.connections if s in component_set and t in component_set
            ]

            original_connections = self.connections
            original_entity_ids = self.entity_ids

            self.connections = component_connections
            self.entity_ids = component

            strategies = self._get_relaxation_strategies()
            sub_positions = None
            # Use shorter time limits for decomposed components
            component_time_limit = max(1, time_limit // max(1, len(components)))

            for strategy in strategies:
                result = self._solve_with_strategy(strategy, component_time_limit, early_stop=True)
                if result.success:
                    sub_positions = result.positions
                    break

            self.connections = original_connections
            self.entity_ids = original_entity_ids

            if sub_positions:
                max_x = max(x for x, y in sub_positions.values())
                for entity_id, (x, y) in sub_positions.items():
                    all_positions[entity_id] = (x + current_offset_x, y)
                current_offset_x += max_x + 10
            else:
                self.diagnostics.warning(
                    f"Component {i + 1} optimization failed, using grid layout"
                )
                for j, entity_id in enumerate(component):
                    all_positions[entity_id] = (current_offset_x + j * 5, 0)
                current_offset_x += len(component) * 5 + 10

        return all_positions

    def _find_connected_components(self) -> list[list[str]]:
        """Find connected components using depth-first search."""
        adjacency: dict[str, set[str]] = {entity_id: set() for entity_id in self.entity_ids}

        for source, sink in self.connections:
            adjacency[source].add(sink)
            adjacency[sink].add(source)

        visited = set()
        components = []

        for entity_id in self.entity_ids:
            if entity_id not in visited:
                component = []
                stack = [entity_id]

                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        component.append(node)
                        stack.extend(adjacency[node] - visited)

                components.append(component)

        components.sort(key=len, reverse=True)

        return components

    def _fallback_grid_layout(self) -> dict[str, tuple[int, int]]:
        """Create simple grid layout as fallback when optimization fails."""
        self.diagnostics.info("Generating fallback grid layout")

        positions = {}
        grid_size = int(np.ceil(np.sqrt(self.n_entities)))
        spacing = 10

        idx = 0
        for entity_id in self.entity_ids:
            if entity_id in self.fixed_positions:
                positions[entity_id] = self.fixed_positions[entity_id]
            else:
                row = idx // grid_size
                col = idx % grid_size
                positions[entity_id] = (col * spacing, row * spacing)
                idx += 1

        return positions
