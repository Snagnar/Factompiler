"""Main layout planning orchestrator."""

from __future__ import annotations
from .entity_placer import EntityPlacer

from typing import Any, Dict, Optional

from dsl_compiler.src.ir.nodes import IRNode
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.layout.integer_layout_solver import IntegerLayoutEngine

from .connection_planner import ConnectionPlanner
from .layout_plan import LayoutPlan
from .signal_analyzer import (
    SignalAnalyzer,
    SignalUsageEntry,
)
from .tile_grid import TileGrid


class LayoutPlanner:
    """Coordinate all layout-planning subsystems for a program."""

    def __init__(
        self,
        signal_type_map: Dict[str, str],
        diagnostics: ProgramDiagnostics,
        *,
        power_pole_type: Optional[str] = None,
        max_wire_span: float = 9.0,
    ) -> None:
        self.signal_type_map = signal_type_map
        self.diagnostics = diagnostics
        self.diagnostics.default_stage = "layout_planning"
        self.power_pole_type = power_pole_type
        self.max_wire_span = max_wire_span

        self.tile_grid = TileGrid()
        self.layout_plan = LayoutPlan()

        self.signal_analyzer: Optional[SignalAnalyzer] = None
        self.signal_usage: Dict[str, SignalUsageEntry] = {}
        self.connection_planner: Optional[ConnectionPlanner] = None
        self.signal_graph: Any = None
        self._memory_modules: Dict[str, Any] = {}
        self._wire_merge_junctions: Dict[str, Any] = {}

    def plan_layout(
        self,
        ir_operations: list[IRNode],
        blueprint_label: str = "DSL Generated",
        blueprint_description: str = "",
    ) -> LayoutPlan:
        """Produce a fresh layout plan for the provided IR operations.

        FLOW:
        1. Signal analysis & materialization
        2. Build signal graph
        3. Add power pole grid BEFORE optimization (with estimated bounds)
           - Poles get fixed_position=True so layout won't move them
           - Combinators will be placed around them
        4. Create entities (without positions)
        5. Optimize positions using integer layout (respects fixed pole positions)
        6. Trim power poles that don't cover any entities
        7. Plan connections with relay routing
        8. Set metadata

        Args:
            ir_operations: List of IR nodes representing the program
            blueprint_label: Label for the resulting blueprint
            blueprint_description: Description text for the blueprint

        Returns:
            LayoutPlan containing entity placements and wire connections
        """
        self._setup_signal_analysis(ir_operations)

        self._create_entities(ir_operations)

        self._add_power_pole_grid()

        self._optimize_positions()

        self._trim_power_poles()

        self._update_tile_grid()
        self._plan_connections()
        self._update_tile_grid()  # Again after relays added

        self._set_metadata(blueprint_label, blueprint_description)

        return self.layout_plan

    def _setup_signal_analysis(self, ir_operations: list[IRNode]) -> None:
        """Initialize and run signal analysis with materialization."""
        self.signal_analyzer = SignalAnalyzer(self.diagnostics, self.signal_type_map)
        self.signal_usage = self.signal_analyzer.analyze(ir_operations)

    def _create_entities(self, ir_operations: list[IRNode]) -> None:
        """Create entity records without positions.

        Entities are created with footprints and properties, but positions
        are assigned later by force-directed optimization.
        """

        placer = EntityPlacer(
            self.tile_grid,
            self.layout_plan,
            self.signal_analyzer,
            self.diagnostics,
        )

        for op in ir_operations:
            placer.place_ir_operation(op)

        placer.create_output_anchors()

        placer.cleanup_unused_entities()

        self.signal_graph = placer.signal_graph
        self._memory_modules = placer._memory_modules
        self._wire_merge_junctions = placer._wire_merge_junctions

    def _optimize_positions(self) -> None:
        """Optimize entity positions using force-directed layout."""
        layout_engine = IntegerLayoutEngine(
            signal_graph=self.signal_graph,
            entity_placements=self.layout_plan.entity_placements,
            diagnostics=self.diagnostics,
        )
        optimized_positions = layout_engine.optimize(time_limit_seconds=30)

        for entity_id, (tile_x, tile_y) in optimized_positions.items():
            placement = self.layout_plan.entity_placements.get(entity_id)
            if placement:
                footprint = placement.properties.get("footprint", (1, 1))
                width, height = footprint

                center_x = tile_x + width / 2.0
                center_y = tile_y + height / 2.0

                placement.position = (center_x, center_y)

        self.diagnostics.info("Entity positions optimized using integer layout engine.")

    def _update_tile_grid(self) -> None:
        """Update tile grid's occupied tiles from entity placements."""
        self.tile_grid.rebuild_from_placements(self.layout_plan.entity_placements)

    def _plan_connections(self) -> None:
        """Plan wire connections between entities."""
        self.connection_planner = ConnectionPlanner(
            self.layout_plan,
            self.signal_usage,
            self.diagnostics,
            self.tile_grid,
            max_wire_span=self.max_wire_span,
            power_pole_type=self.power_pole_type,
        )

        self.connection_planner._memory_modules = self._memory_modules

        locked_colors = self._determine_locked_wire_colors()

        self.connection_planner.plan_connections(
            self.signal_graph,
            self.layout_plan.entity_placements,
            wire_merge_junctions=self._wire_merge_junctions,
            locked_colors=locked_colors,
        )

        self._inject_wire_colors_into_placements()

    def _resolve_source_entity(self, signal_id: Any) -> Optional[str]:
        """Resolve source entity ID from a signal ID."""
        if isinstance(signal_id, str) and "@" in signal_id:
            return signal_id.split("@")[1]
        if hasattr(signal_id, "source_id"):
            return signal_id.source_id
        return self.signal_graph.get_source(str(signal_id))

    def _inject_operand_wire_color(
        self, placement: Any, operand_key: str, injected_count: int
    ) -> int:
        """Inject wire color for a single operand. Returns updated count."""
        signal = placement.properties.get(f"{operand_key}_operand")
        signal_id = placement.properties.get(f"{operand_key}_operand_signal_id")

        if not signal or isinstance(signal, int) or not signal_id:
            return injected_count

        source_entity = self._resolve_source_entity(signal_id)

        if source_entity:
            color = self.connection_planner.get_wire_color_for_edge(
                source_entity, placement.ir_node_id, signal
            )
            placement.properties[f"{operand_key}_operand_wires"] = {color}
            self.diagnostics.info(
                f"Injected {operand_key}_operand_wires={{'{color}'}} for {placement.ir_node_id} "
                f"signal={signal} from {source_entity}"
            )
            return injected_count + 1
        else:
            placement.properties[f"{operand_key}_operand_wires"] = {"red", "green"}
            self.diagnostics.warning(
                f"No source found for {operand_key} operand signal {signal_id} of {placement.ir_node_id}"
            )
            return injected_count

    def _inject_wire_colors_into_placements(self) -> None:
        """Store wire color information in combinator placement properties.

        After wire connections are planned, we know which wire colors (red/green)
        deliver each signal to each entity. Store this information in the placement
        properties so the entity emitter can configure wire filters on combinators.
        """
        injected_count = 0
        for placement in self.layout_plan.entity_placements.values():
            if placement.entity_type not in (
                "arithmetic-combinator",
                "decider-combinator",
            ):
                continue

            injected_count = self._inject_operand_wire_color(
                placement, "left", injected_count
            )
            injected_count = self._inject_operand_wire_color(
                placement, "right", injected_count
            )

        self.diagnostics.info(
            f"Wire color injection: {injected_count} operands configured"
        )

    def _add_power_pole_grid(self) -> None:
        """Add power poles in a grid pattern BEFORE layout optimization.

        This is called BEFORE layout optimization so that the layout engine
        treats poles as fixed obstacles and places combinators around them.
        After layout, we trim unused poles with _trim_power_poles().
        """
        if not self.power_pole_type:
            return

        from .power_planner import PowerPlanner

        power_planner = PowerPlanner(
            self.tile_grid,
            self.layout_plan,
            self.diagnostics,
            connection_planner=None,
        )

        # Place poles with fixed positions - layout will place entities around them
        power_planner.add_power_pole_grid(self.power_pole_type)

    def _trim_power_poles(self) -> None:
        """Remove power poles that don't cover any entities.

        This is called AFTER layout optimization when we know the actual
        entity positions. Removes poles that are outside the entity bounds
        or don't provide coverage to any entity.
        """
        if not self.power_pole_type:
            return

        from .power_planner import POWER_POLE_CONFIG

        config = POWER_POLE_CONFIG.get(self.power_pole_type.lower())
        if config is None:
            return

        supply_radius = float(config["supply_radius"])

        entity_positions = []
        for placement in self.layout_plan.entity_placements.values():
            if placement.position is None:
                continue
            if placement.properties.get("is_power_pole"):
                continue
            entity_positions.append(placement.position)

        if not entity_positions:
            return

        poles_to_remove = []
        for pole_id, placement in list(self.layout_plan.entity_placements.items()):
            if not placement.properties.get("is_power_pole"):
                continue

            pole_pos = placement.position
            if pole_pos is None:
                continue

            covers_any = False
            for entity_pos in entity_positions:
                dx = abs(entity_pos[0] - pole_pos[0])
                dy = abs(entity_pos[1] - pole_pos[1])
                if dx <= supply_radius and dy <= supply_radius:
                    covers_any = True
                    break

            if not covers_any:
                poles_to_remove.append(pole_id)

        for pole_id in poles_to_remove:
            del self.layout_plan.entity_placements[pole_id]
            self.layout_plan.power_poles = [
                p for p in self.layout_plan.power_poles if p.pole_id != pole_id
            ]

        if poles_to_remove:
            remaining = len(
                [
                    p
                    for p in self.layout_plan.entity_placements.values()
                    if p.properties.get("is_power_pole")
                ]
            )
            self.diagnostics.info(
                f"Trimmed {len(poles_to_remove)} unused power poles, {remaining} remaining"
            )

    def _set_metadata(self, blueprint_label: str, blueprint_description: str) -> None:
        """Set blueprint metadata."""
        self.layout_plan.blueprint_label = blueprint_label
        self.layout_plan.blueprint_description = blueprint_description

    def _determine_locked_wire_colors(self) -> Dict[tuple[str, str], str]:
        """Determine wire colors that must be locked for correctness.

        For SR latch memories:
        - Data/feedback channel: RED (signal-B or memory signal)
        - Control channel: GREEN (signal-W)

        This prevents signal summation at combinator inputs.
        """
        from .memory_builder import MemoryModule

        locked = {}

        for module in self._memory_modules.values():
            if isinstance(module, MemoryModule):
                if module.optimization is None:
                    if module.write_gate:
                        locked[(module.write_gate.ir_node_id, module.signal_type)] = (
                            "red"
                        )
                    if module.hold_gate:
                        locked[(module.hold_gate.ir_node_id, module.signal_type)] = (
                            "red"
                        )

        for signal_id, source_ids, sink_ids in self.signal_graph.iter_edges():
            usage = self.signal_usage.get(signal_id)
            resolved_name = usage.resolved_signal_name if usage else None

            if resolved_name == "signal-W" or signal_id == "signal-W":
                for source_id in source_ids:
                    locked[(source_id, "signal-W")] = "green"

        for module in self._memory_modules.values():
            if isinstance(module, MemoryModule):
                if module.optimization is None and module.write_gate:
                    write_gate_id = module.write_gate.ir_node_id
                    data_signal = module.signal_type  # e.g., "signal-B"

                    for (
                        signal_id,
                        source_ids,
                        sink_ids,
                    ) in self.signal_graph.iter_edges():
                        if write_gate_id in sink_ids:
                            usage = self.signal_usage.get(signal_id)
                            resolved_name = (
                                usage.resolved_signal_name if usage else None
                            )

                            if resolved_name == data_signal or signal_id == data_signal:
                                for source_id in source_ids:
                                    if (
                                        source_id != write_gate_id
                                        and source_id != module.hold_gate.ir_node_id
                                    ):
                                        locked[(source_id, data_signal)] = "red"

        for entity_id, placement in self.layout_plan.entity_placements.items():
            if placement.properties.get("has_self_feedback"):
                feedback_signal = placement.properties.get("feedback_signal")
                if feedback_signal:
                    locked[(entity_id, feedback_signal)] = "red"
                    self.diagnostics.info(
                        f"Locked {entity_id} feedback signal '{feedback_signal}' to red wire"
                    )

        return locked
