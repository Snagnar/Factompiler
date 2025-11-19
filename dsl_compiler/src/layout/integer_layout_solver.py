from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from ortools.sat.python import cp_model

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from .layout_plan import EntityPlacement
from .signal_graph import SignalGraph


@dataclass
class LayoutConstraints:
    """Constraints for layout optimization."""

    max_wire_span: float = 9.0
    max_coordinate: int = 200


@dataclass
class OptimizationResult:
    """Result of layout optimization attempt.

    Note: positions are TILE positions (top-left corner on integer grid),
    NOT center positions. They must be converted to center coordinates
    before use in Draftsman/Factorio.
    """

    positions: Dict[str, Tuple[int, int]]  # entity_id -> (tile_x, tile_y)
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
        entity_placements: Dict[str, EntityPlacement],
        diagnostics: ProgramDiagnostics,
        constraints: Optional[LayoutConstraints] = None,
    ):
        self.signal_graph = signal_graph
        self.entity_placements = entity_placements
        self.diagnostics = diagnostics
        self.constraints = constraints or LayoutConstraints()

        # Extract entity data
        # Preserve definition order (dict insertion order in Python 3.7+)
        # This maintains source code order for left-to-right layout
        self.entity_ids = list(entity_placements.keys())
        self.n_entities = len(self.entity_ids)

        # Build connectivity and extract entity properties
        self._build_connectivity()
        self._identify_fixed_positions()
        self._extract_footprints()

    def _build_connectivity(self) -> None:
        """Build connectivity as list of (source_id, sink_id) pairs."""
        self.connections = []

        for signal_id, source_id, sink_id in self.signal_graph.iter_source_sink_pairs():
            if source_id in self.entity_ids and sink_id in self.entity_ids:
                self.connections.append((source_id, sink_id))
            # Silently skip connections involving non-materialized nodes (e.g., wire merges)
            # These are virtual constructs that don't exist as physical entities

        # Remove duplicates
        self.connections = list(set(self.connections))

        self.diagnostics.info(
            f"Built connectivity: {len(self.connections)} unique connections"
        )

    def _identify_fixed_positions(self) -> None:
        """Identify entities with user-specified fixed positions."""
        self.fixed_positions = {}

        for entity_id, placement in self.entity_placements.items():
            if placement.position is not None:
                if placement.properties.get("user_specified_position"):
                    x, y = placement.position
                    self.fixed_positions[entity_id] = (int(x), int(y))

        self.diagnostics.info(f"Found {len(self.fixed_positions)} fixed positions")

    def _extract_footprints(self) -> None:
        """Extract entity footprints (width, height) as integers."""
        self.footprints = {}

        for entity_id in self.entity_ids:
            footprint = self.entity_placements[entity_id].properties.get("footprint")
            if footprint:
                width, height = footprint
                self.footprints[entity_id] = (int(np.ceil(width)), int(np.ceil(height)))
            else:
                self.footprints[entity_id] = (1, 1)

    def optimize(self, time_limit_seconds: int = 60) -> Dict[str, Tuple[int, int]]:
        """
        Optimize layout with progressive relaxation strategy.

        Tries multiple strategies with increasing relaxation until a good
        solution is found or all strategies are exhausted.

        Args:
            time_limit_seconds: Time limit per strategy attempt

        Returns:
            Dict mapping entity_id to (x, y) integer coordinates
        """

        self.diagnostics.info(
            f"Starting integer layout optimization: {self.n_entities} entities, "
            f"{len(self.connections)} connections"
        )

        if self.n_entities == 0:
            return {}

        # Check if we need subgraph decomposition
        if self.n_entities > 500:
            self.diagnostics.info(
                f"Large graph detected ({self.n_entities} entities), "
                "using subgraph decomposition"
            )
            return self._optimize_with_decomposition(time_limit_seconds)

        # Define progressive relaxation strategies
        strategies = self._get_relaxation_strategies()

        # Try each strategy until acceptable solution found
        best_result = None

        for i, strategy in enumerate(strategies):
            self.diagnostics.info(
                f"Attempting strategy {i + 1}/{len(strategies)}: {strategy['name']}"
            )

            result = self._solve_with_strategy(strategy, time_limit_seconds)

            if result.success and result.violations == 0:
                self.diagnostics.info(
                    f"Perfect solution found with strategy '{strategy['name']}' "
                    f"in {result.solve_time:.2f}s"
                )
                return result.positions

            if result.success:
                if best_result is None or result.violations < best_result.violations:
                    best_result = result

                if result.violations <= 5:
                    self.diagnostics.warning(
                        f"Acceptable solution found with {result.violations} violations "
                        f"using strategy '{strategy['name']}'"
                    )
                    self._report_violations(result)
                    return result.positions

        # Return best result found, if any
        if best_result and best_result.success:
            self.diagnostics.warning(
                f"Best solution has {best_result.violations} violations "
                f"(strategy: {best_result.strategy_used})"
            )
            self._report_violations(best_result)
            return best_result.positions

        # All strategies failed - provide detailed diagnostics
        self._diagnose_failure()
        return self._fallback_grid_layout()

    def _get_relaxation_strategies(self) -> List[Dict]:
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
        self, strategy: Dict, time_limit: int
    ) -> OptimizationResult:
        """Solve layout with a specific strategy."""
        model = cp_model.CpModel()

        # Create position variables
        positions = self._create_position_variables(model, strategy["max_coord"])

        # Add hard constraint: no overlaps
        self._add_no_overlap_constraint(model, positions)

        # Add edge layout constraints (inputs north, outputs south)
        self._add_edge_layout_constraints(model, positions)

        # Add soft constraint: wire span limit
        span_violations, wire_lengths = self._add_span_constraints(
            model, positions, strategy["max_span"]
        )

        # Create objective function
        self._create_objective(
            model,
            span_violations,
            wire_lengths,
            positions,
            strategy["violation_weight"],
            strategy["max_coord"],
        )

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit)
        solver.parameters.log_search_progress = False

        status = solver.Solve(model)

        # Extract and return result
        return self._extract_result(
            solver, status, positions, span_violations, wire_lengths, strategy["name"]
        )

    def _create_position_variables(
        self, model: cp_model.CpModel, max_coord: int
    ) -> Dict[str, Tuple]:
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

    def _add_no_overlap_constraint(
        self, model: cp_model.CpModel, positions: Dict
    ) -> None:
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
        self, model: cp_model.CpModel, positions: Dict, max_span: int
    ) -> Tuple[List, List]:
        """Add soft wire span constraints with violation tracking."""
        span_violations = []
        wire_lengths = []

        for i, (source, sink) in enumerate(self.connections):
            x1, y1 = positions[source]
            x2, y2 = positions[sink]

            # Manhattan distance (easier for solver than Euclidean)
            dx = model.NewIntVar(0, max_span * 2, f"dx_{i}")
            dy = model.NewIntVar(0, max_span * 2, f"dy_{i}")
            model.AddAbsEquality(dx, x1 - x2)
            model.AddAbsEquality(dy, y1 - y2)

            distance = model.NewIntVar(0, max_span * 4, f"dist_{i}")
            model.Add(distance == dx + dy)

            wire_lengths.append(distance)

            # Create boolean violation indicator
            is_violation = model.NewBoolVar(f"viol_{i}")
            model.Add(distance > max_span).OnlyEnforceIf(is_violation)
            model.Add(distance <= max_span).OnlyEnforceIf(is_violation.Not())

            span_violations.append(is_violation)

        return span_violations, wire_lengths

    def _add_edge_layout_constraints(
        self, model: cp_model.CpModel, positions: Dict
    ) -> None:
        """Add constraints for north-south edge layout.

        Strategy:
        - All inputs share Y = Y_input_line (a variable)
        - All outputs share Y = Y_output_line (a variable)
        - All intermediates have Y strictly between these lines
        - Horizontal ordering preserves definition order (no overlaps)
        - Optimizer minimizes bounding box, naturally placing:
          * Y_input_line at minimum feasible Y
          * Y_output_line at maximum feasible Y

        Args:
            model: CP-SAT constraint model
            positions: Dict mapping entity_id to (x, y) variables
        """
        # Categorize entities (skip user-fixed entities)
        input_entities = []
        output_entities = []
        intermediate_entities = []

        for entity_id in self.entity_ids:
            # Skip entities with user-specified positions
            if entity_id in self.fixed_positions:
                continue

            placement = self.entity_placements.get(entity_id)
            if not placement:
                continue

            # Categorize by role
            if placement.properties.get("is_input"):
                input_entities.append(entity_id)
            elif placement.properties.get("is_output"):
                output_entities.append(entity_id)
            else:
                intermediate_entities.append(entity_id)

        # If no inputs or outputs, no edge constraints needed
        if not input_entities and not output_entities:
            self.diagnostics.info(
                "No edge layout constraints (no inputs/outputs marked)"
            )
            return

        max_coord = self.constraints.max_coordinate

        # ========================================================================
        # Create shared Y-coordinate variables for input and output lines
        # ========================================================================

        Y_input_line = None
        max_input_height = 0

        if input_entities:
            Y_input_line = model.NewIntVar(0, max_coord, "Y_input_line")
            # Calculate maximum height among all inputs
            max_input_height = max(
                self.footprints.get(e, (1, 1))[1] for e in input_entities
            )
            self.diagnostics.info(
                f"Edge layout: {len(input_entities)} inputs, max height {max_input_height}"
            )

        Y_output_line = None

        if output_entities:
            Y_output_line = model.NewIntVar(0, max_coord, "Y_output_line")
            self.diagnostics.info(f"Edge layout: {len(output_entities)} outputs")

        # ========================================================================
        # Ensure sufficient vertical gap between input and output lines
        # ========================================================================

        if Y_input_line is not None and Y_output_line is not None:
            # Calculate minimum gap needed:
            # - max_input_height: space for inputs
            # - at least 1: minimum gap
            # - max_intermediate_height: space for intermediates (if any)
            min_gap = max_input_height

            if intermediate_entities:
                max_intermediate_height = max(
                    self.footprints.get(e, (1, 1))[1] for e in intermediate_entities
                )
                min_gap += max_intermediate_height
            else:
                min_gap += 1  # At least 1 tile gap even with no intermediates

            # Constrain: Y_output_line ≥ Y_input_line + min_gap
            model.Add(Y_output_line >= Y_input_line + min_gap)

            self.diagnostics.info(
                f"Edge layout: enforcing minimum gap of {min_gap} between input/output lines"
            )

        # ========================================================================
        # Constrain all inputs to share Y_input_line
        # ========================================================================

        if input_entities and Y_input_line is not None:
            for entity_id in input_entities:
                _, y = positions[entity_id]
                model.Add(y == Y_input_line)

            # Constrain horizontal ordering: maintain definition order (left-to-right)
            # Each input must start after the previous input ends (no overlap)
            for i in range(len(input_entities) - 1):
                curr_id = input_entities[i]
                next_id = input_entities[i + 1]

                curr_x, _ = positions[curr_id]
                next_x, _ = positions[next_id]

                curr_width = self.footprints.get(curr_id, (1, 1))[0]

                # Next entity X must be at least curr_x + curr_width
                # (no overlap, maintains left-to-right order)
                model.Add(next_x >= curr_x + curr_width)

        # ========================================================================
        # Constrain all outputs to share Y_output_line
        # ========================================================================

        if output_entities and Y_output_line is not None:
            for entity_id in output_entities:
                _, y = positions[entity_id]
                model.Add(y == Y_output_line)

            # Constrain horizontal ordering: maintain definition order (left-to-right)
            for i in range(len(output_entities) - 1):
                curr_id = output_entities[i]
                next_id = output_entities[i + 1]

                curr_x, _ = positions[curr_id]
                next_x, _ = positions[next_id]

                curr_width = self.footprints.get(curr_id, (1, 1))[0]

                model.Add(next_x >= curr_x + curr_width)

        # ========================================================================
        # Constrain intermediates to be strictly between input and output lines
        # ========================================================================

        for entity_id in intermediate_entities:
            _, y = positions[entity_id]
            height = self.footprints.get(entity_id, (1, 1))[1]

            # Intermediate top must be below input bottom
            if Y_input_line is not None:
                # y ≥ Y_input_line + max_input_height
                model.Add(y >= Y_input_line + max_input_height)

            # Intermediate bottom must be above output top
            if Y_output_line is not None:
                # y + height ≤ Y_output_line
                model.Add(y + height <= Y_output_line)

    def _create_objective(
        self,
        model: cp_model.CpModel,
        span_violations: List,
        wire_lengths: List,
        positions: Dict,
        violation_weight: int,
        max_coord: int,
    ) -> None:
        """Create multi-objective optimization function with priorities."""
        # Primary: minimize violations
        num_violations = sum(span_violations) if span_violations else 0

        # Secondary: minimize wire length
        total_wire_length = sum(wire_lengths) if wire_lengths else 0

        # Tertiary: minimize bounding box area (compact layout)
        all_x = [positions[e][0] for e in self.entity_ids]
        all_y = [positions[e][1] for e in self.entity_ids]

        max_x = model.NewIntVar(0, max_coord, "max_x")
        max_y = model.NewIntVar(0, max_coord, "max_y")
        model.AddMaxEquality(max_x, all_x)
        model.AddMaxEquality(max_y, all_y)

        bounding_perimeter = max_x + max_y

        # Combined objective with weighted priorities
        objective = (
            violation_weight * num_violations
            + 100 * total_wire_length
            + 100 * bounding_perimeter
        )

        model.Minimize(objective)

    def _extract_result(
        self,
        solver: cp_model.CpSolver,
        status: int,
        positions: Dict,
        span_violations: List,
        wire_lengths: List,
        strategy_name: str,
    ) -> OptimizationResult:
        """Extract optimization result from solved model."""
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            result_positions = {
                entity_id: (solver.Value(x), solver.Value(y))
                for entity_id, (x, y) in positions.items()
            }

            num_violations = (
                sum(solver.Value(v) for v in span_violations) if span_violations else 0
            )

            total_wire_length = (
                sum(solver.Value(wl) for wl in wire_lengths) if wire_lengths else 0
            )

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

        self.diagnostics.warning(
            f"Layout has {len(violated_connections)} wire span violations:"
        )

        # Report up to 10 violations
        for source, sink, distance in violated_connections[:10]:
            self.diagnostics.warning(
                f"  {source} → {sink}: {distance} units "
                f"(limit: {self.constraints.max_wire_span})"
            )

        if len(violated_connections) > 10:
            self.diagnostics.warning(
                f"  ... and {len(violated_connections) - 10} more violations"
            )

    def _diagnose_failure(self) -> None:
        """Provide detailed diagnostic information about optimization failure."""
        self.diagnostics.error("Failed to find feasible layout with all strategies")

        # Calculate statistics for diagnostics
        total_area_needed = sum(w * h for w, h in self.footprints.values())
        available_area = self.constraints.max_coordinate**2

        self.diagnostics.error(
            f"Diagnostic information:\n"
            f"  Entities: {self.n_entities}\n"
            f"  Connections: {len(self.connections)}\n"
            f"  Fixed positions: {len(self.fixed_positions)}\n"
            f"  Total entity area: {total_area_needed}\n"
            f"  Available area: {available_area}\n"
            f"  Area utilization: {100 * total_area_needed / available_area:.1f}%"
        )

        if total_area_needed > available_area * 0.8:
            self.diagnostics.error(
                "  → Area utilization >80%: entities may not fit\n"
                "     Solution: Increase max_coordinate constraint"
            )

        if len(self.fixed_positions) > self.n_entities * 0.3:
            self.diagnostics.error(
                f"  → Many fixed positions ({len(self.fixed_positions)}/{self.n_entities})\n"
                "     Solution: Reduce fixed position constraints"
            )

        if self.connections:
            avg_degree = 2 * len(self.connections) / self.n_entities
            if avg_degree > 4:
                self.diagnostics.error(
                    f"  → Dense connectivity (avg degree: {avg_degree:.1f})\n"
                    "     Solution: Relax wire span constraint"
                )

        self.diagnostics.error("Falling back to simple grid layout")

    def _optimize_with_decomposition(
        self, time_limit: int
    ) -> Dict[str, Tuple[int, int]]:
        """Optimize large graphs using connected component decomposition."""
        components = self._find_connected_components()

        self.diagnostics.info(
            f"Decomposed graph into {len(components)} connected components"
        )

        all_positions = {}
        current_offset_x = 0

        for i, component in enumerate(components):
            self.diagnostics.info(
                f"Optimizing component {i + 1}/{len(components)} "
                f"({len(component)} entities)"
            )

            # Filter connections for this component
            component_set = set(component)
            component_connections = [
                (s, t)
                for s, t in self.connections
                if s in component_set and t in component_set
            ]

            # Temporarily modify connections for sub-problem
            original_connections = self.connections
            original_entity_ids = self.entity_ids

            self.connections = component_connections
            self.entity_ids = component

            # Solve sub-problem (without decomposition)
            strategies = self._get_relaxation_strategies()
            sub_positions = None

            for strategy in strategies:
                result = self._solve_with_strategy(strategy, time_limit)
                if result.success:
                    sub_positions = result.positions
                    break

            # Restore original state
            self.connections = original_connections
            self.entity_ids = original_entity_ids

            # Apply positions with offset
            if sub_positions:
                max_x = max(x for x, y in sub_positions.values())
                for entity_id, (x, y) in sub_positions.items():
                    all_positions[entity_id] = (x + current_offset_x, y)
                current_offset_x += max_x + 10
            else:
                self.diagnostics.warning(
                    f"Component {i + 1} optimization failed, using grid layout"
                )
                # Use grid layout for failed component
                for j, entity_id in enumerate(component):
                    all_positions[entity_id] = (current_offset_x + j * 5, 0)
                current_offset_x += len(component) * 5 + 10

        return all_positions

    def _find_connected_components(self) -> List[List[str]]:
        """Find connected components using depth-first search."""
        adjacency = {entity_id: set() for entity_id in self.entity_ids}

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

        # Sort by size (largest first)
        components.sort(key=len, reverse=True)

        return components

    def _fallback_grid_layout(self) -> Dict[str, Tuple[int, int]]:
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
