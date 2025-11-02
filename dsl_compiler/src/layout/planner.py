"""Main layout planning orchestrator."""

from __future__ import annotations

from typing import Any, Dict, Optional

from dsl_compiler.src.ir import IRNode
from dsl_compiler.src.common import ProgramDiagnostics

from .connection_planner import ConnectionPlanner
from .layout_engine import LayoutEngine
from .layout_plan import LayoutPlan
from .power_planner import PowerPlanner
from .signal_analyzer import (
    SignalAnalyzer,
    SignalMaterializer,
    SignalUsageEntry,
)
from .signal_resolver import SignalResolver


class LayoutPlanner:
    """Coordinate all layout-planning subsystems for a program."""

    def __init__(
        self,
        signal_type_map: Dict[str, str],
        diagnostics: Optional[ProgramDiagnostics] = None,
        *,
        power_pole_type: Optional[str] = None,
        max_wire_span: float = 9.0,
    ) -> None:
        self.signal_type_map = signal_type_map
        self.diagnostics = diagnostics or ProgramDiagnostics()
        self.power_pole_type = power_pole_type
        self.max_wire_span = max_wire_span

        self.layout_engine = LayoutEngine()
        self.layout_plan = LayoutPlan()

        self.signal_analyzer: Optional[SignalAnalyzer] = None
        self.signal_usage: Dict[str, SignalUsageEntry] = {}
        self.materializer: Optional[SignalMaterializer] = None
        self.signal_resolver: Optional[SignalResolver] = None
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

        Args:
            ir_operations: List of IR nodes representing the program
            blueprint_label: Label for the resulting blueprint
            blueprint_description: Description text for the blueprint

        Returns:
            LayoutPlan containing entity placements and wire connections
        """
        self._reset_state()
        self._setup_signal_analysis(ir_operations)
        self._setup_materialization()
        self._setup_signal_resolver()
        self._place_entities(ir_operations)
        self._plan_connections()
        self._plan_power_if_requested()
        self._set_metadata(blueprint_label, blueprint_description)
        return self.layout_plan

    def _reset_state(self) -> None:
        """Reset state for a fresh planning run."""
        self.layout_engine = LayoutEngine()
        self.layout_plan = LayoutPlan()
        self.signal_analyzer = None
        self.signal_usage = {}
        self.materializer = None
        self.signal_resolver = None
        self.connection_planner = None
        self.signal_graph = None
        self._memory_modules = {}
        self._wire_merge_junctions = {}

    def _setup_signal_analysis(self, ir_operations: list[IRNode]) -> None:
        """Initialize and run signal analysis."""
        self.signal_analyzer = SignalAnalyzer(self.diagnostics)
        self.signal_usage = self.signal_analyzer.analyze(ir_operations)

    def _setup_materialization(self) -> None:
        """Initialize materialization decisions."""
        self.materializer = SignalMaterializer(
            self.signal_usage,
            self.signal_type_map,
            self.diagnostics,
        )
        self.materializer.finalize()

    def _setup_signal_resolver(self) -> None:
        """Initialize signal resolver."""
        self.signal_resolver = SignalResolver(
            self.signal_type_map,
            self.diagnostics,
            materializer=self.materializer,
            signal_usage=self.signal_usage,
        )

    def _plan_connections(self) -> None:
        """Plan wire connections between entities."""
        self.connection_planner = ConnectionPlanner(
            self.layout_plan,
            self.signal_usage,
            self.diagnostics,
            self.layout_engine,
            max_wire_span=self.max_wire_span,
        )

        locked_colors = self._determine_locked_wire_colors()
        self.connection_planner.plan_connections(
            self.signal_graph,
            self.layout_plan.entity_placements,
            wire_merge_junctions=self._wire_merge_junctions,
            locked_colors=locked_colors,
        )

    def _plan_power_if_requested(self) -> None:
        """Plan power grid if power pole type is specified."""
        if not self.power_pole_type:
            return

        power_planner = PowerPlanner(
            self.layout_engine,
            self.layout_plan,
            self.diagnostics,
        )
        power_planner.plan_power_grid(self.power_pole_type)

    def _set_metadata(self, blueprint_label: str, blueprint_description: str) -> None:
        """Set blueprint metadata."""
        self.layout_plan.blueprint_label = blueprint_label
        self.layout_plan.blueprint_description = blueprint_description

    def _place_entities(self, ir_operations: list[IRNode]) -> None:
        """Place all entities in the layout plan."""
        from .entity_placer import EntityPlacer

        placer = EntityPlacer(
            self.layout_engine,
            self.layout_plan,
            self.signal_usage,
            self.materializer,
            self.signal_resolver,
            self.diagnostics,
        )

        for op in ir_operations:
            placer.place_ir_operation(op)

        # ✅ Clean up optimized-away entities
        placer.cleanup_unused_entities()

        # Store signal graph and memory info for connection planning
        self.signal_graph = placer.signal_graph
        self._memory_modules = placer._memory_modules
        self._wire_merge_junctions = placer._wire_merge_junctions

    def _determine_locked_wire_colors(self) -> Dict[tuple[str, str], str]:
        """Collect wire color locks for memory modules."""
        locked = {}
        for module in self._memory_modules.values():
            for component_name, placement in module.items():
                if component_name == "hold_gate":
                    if not hasattr(placement, "properties"):
                        continue
                    # Memory hold gate output must use red wire for feedback loop stability
                    signal_name = placement.properties.get("output_signal")
                    if signal_name:
                        locked[(placement.ir_node_id, signal_name)] = "red"
        return locked
