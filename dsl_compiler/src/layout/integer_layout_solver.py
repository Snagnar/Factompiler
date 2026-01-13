from dataclasses import dataclass
from typing import Any

import numpy as np
from ortools.sat.python import cp_model

from dsl_compiler.src.common.constants import DEFAULT_CONFIG, CompilerConfig
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics

from .layout_plan import EntityPlacement
from .signal_graph import SignalGraph


class EarlyStopCallback(cp_model.CpSolverSolutionCallback):
    """Callback to stop solver early when a good-enough solution is found."""

    def __init__(self, max_violations: int = 0):
        super().__init__()
        self._solution_count = 0
        self._max_violations = max_violations
        self._stop_search = False

    def on_solution_callback(self) -> None:
        self._solution_count += 1
        # Stop after finding any solution with acceptable violations
        # The objective function already prioritizes minimizing violations
        if self._solution_count >= 1:
            obj = self.ObjectiveValue()
            # If objective is low enough (few/no violations), stop early
            # violation_weight is typically 10000, so obj < 10000 means 0 violations
            if obj < 10000 * (self._max_violations + 1):
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

    def _build_connectivity(self) -> None:
        """Build connectivity as list of (source_id, sink_id) pairs.

        Wire merge nodes are expanded: if a source feeds into a wire merge,
        edges are created from that source to all sinks of the wire merge.
        This ensures the layout optimizer considers the full wire merge topology.
        """
        self.connections = []

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

            # Case 2: Source feeds into a wire merge (sink is wire merge node)
            elif source_id in self.entity_ids and sink_id in self.wire_merge_junctions:
                # Connect source to all sinks of the wire merge
                for actual_sink in wire_merge_sinks.get(sink_id, []):
                    self.connections.append((source_id, actual_sink))

            # Case 3: Wire merge feeds into a sink
            elif source_id in self.wire_merge_junctions and sink_id in self.entity_ids:
                # Connect all wire merge inputs to this sink
                for actual_source in wire_merge_sources.get(source_id, []):
                    self.connections.append((actual_source, sink_id))

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

    def optimize(self, time_limit_seconds: int = 60) -> dict[str, tuple[int, int]]:
        """
        Optimize layout with progressive relaxation strategy.

        Uses early stopping when a good-enough solution is found. For small
        graphs, tries a quick solve first before falling back to longer solves.

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
            if (
                quick_result.success
                and quick_result.violations <= self.config.acceptable_layout_violations
            ):
                self.diagnostics.info(
                    f"Quick acceptable solution found with {quick_result.violations} violations "
                    f"in {quick_result.solve_time:.2f}s"
                )
                return quick_result.positions

        best_result = None
        # Distribute time budget across strategies, favoring earlier (stricter) ones
        per_strategy_limit = max(1, time_limit_seconds // len(strategies))

        for i, strategy in enumerate(strategies):
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

                if result.violations <= self.config.acceptable_layout_violations:
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
        """Define progressive relaxation strategies."""
        max_span = int(self.constraints.max_wire_span)
        max_coord = self.constraints.max_coordinate

        return [
            {
                "name": "Strict",
                "max_span": max_span,
                "max_coord": max_coord,
                "violation_weight": 10000,
            },
            {
                "name": "Relaxed span (+33%)",
                "max_span": int(max_span * 1.33),
                "max_coord": max_coord,
                "violation_weight": 10000,
            },
            {
                "name": "Larger area (+50%)",
                "max_span": max_span,
                "max_coord": int(max_coord * 1.5),
                "violation_weight": 10000,
            },
            {
                "name": "Both relaxed",
                "max_span": int(max_span * 1.5),
                "max_coord": int(max_coord * 1.5),
                "violation_weight": 5000,
            },
            {
                "name": "Very relaxed",
                "max_span": int(max_span * 2),
                "max_coord": int(max_coord * 2),
                "violation_weight": 1000,
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
            callback = EarlyStopCallback(max_violations=self.config.acceptable_layout_violations)
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
        """Add soft wire span constraints with violation tracking."""
        span_violations = []
        wire_lengths = []

        for i, (source, sink) in enumerate(self.connections):
            x1, y1 = positions[source]
            x2, y2 = positions[sink]

            dx = model.NewIntVar(0, max_span * 2, f"dx_{i}")
            dy = model.NewIntVar(0, max_span * 2, f"dy_{i}")
            model.AddAbsEquality(dx, x1 - x2)
            model.AddAbsEquality(dy, y1 - y2)

            distance = model.NewIntVar(0, max_span * 4, f"dist_{i}")
            model.Add(distance == dx + dy)

            wire_lengths.append(distance)

            is_violation = model.NewBoolVar(f"viol_{i}")
            model.Add(distance > max_span).OnlyEnforceIf(is_violation)
            model.Add(distance <= max_span).OnlyEnforceIf(is_violation.Not())

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

        The bounding box is computed excluding power poles, since they are
        infrastructure and should not affect the compactness goal for the
        actual circuit entities.
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

        objective = (
            violation_weight * num_violations + 300 * total_wire_length + 100 * bounding_perimeter
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
