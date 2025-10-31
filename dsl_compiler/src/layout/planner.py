"""Main layout planning orchestrator."""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from dsl_compiler.src.ir import IRNode
from dsl_compiler.src.semantic import DiagnosticCollector

from .connection_planner import ConnectionPlanner
from .layout_engine import LayoutEngine
from .layout_plan import LayoutPlan
from .power_planner import PowerPlanner
from .signal_analyzer import (
    SignalAnalyzer,
    SignalMaterializer,
    SignalUsageEntry,
)

if TYPE_CHECKING:  # pragma: no cover - type checking helper
    from dsl_compiler.src.emission.emitter import LegacyBlueprintEmitter


class LayoutPlanner:
    """Coordinate all layout-planning subsystems for a program."""

    def __init__(
        self,
        signal_type_map: Dict[str, str],
        diagnostics: Optional[DiagnosticCollector] = None,
        *,
        power_pole_type: Optional[str] = None,
        max_wire_span: float = 9.0,
    ) -> None:
        self.signal_type_map = signal_type_map
        self.diagnostics = diagnostics or DiagnosticCollector()
        self.power_pole_type = power_pole_type
        self.max_wire_span = max_wire_span

        self.layout_engine = LayoutEngine()
        self.layout_plan = LayoutPlan()

        self.signal_analyzer: Optional[SignalAnalyzer] = None
        self.signal_usage: Dict[str, SignalUsageEntry] = {}
        self.materializer: Optional[SignalMaterializer] = None
        self.connection_planner: Optional[ConnectionPlanner] = None
        self.signal_graph: Any = None
        self.entities: Dict[str, Any] = {}
        self._emitter: Optional["LegacyBlueprintEmitter"] = None

    def plan_layout(
        self,
        ir_operations: list[IRNode],
        blueprint_label: str = "DSL Generated",
        blueprint_description: str = "",
    ) -> LayoutPlan:
        """Produce a fresh layout plan for the provided IR operations."""

        # Reset state for a fresh planning run.
        self.layout_engine = LayoutEngine()
        self.layout_plan = LayoutPlan()
        self.signal_analyzer = None
        self.signal_usage = {}
        self.materializer = None
        self.connection_planner = None
        self.signal_graph = None
        self.entities = {}
        self._emitter = None

        emitter = self._place_entities(ir_operations)

        self._emitter = emitter
        self.signal_analyzer = emitter.signal_analyzer
        self.signal_usage = emitter.signal_usage or {}
        self.materializer = emitter.materializer
        self.layout_engine = emitter.layout

        source_plan = emitter.layout_plan
        filtered_placements = {
            entity_id: placement
            for entity_id, placement in source_plan.entity_placements.items()
            if getattr(placement, "role", None) != "wire_relay"
        }

        self.layout_plan = LayoutPlan(
            entity_placements=dict(filtered_placements),
            signal_materializations=dict(source_plan.signal_materializations),
            signal_graph=dict(source_plan.signal_graph or {}),
        )
        self.layout_plan.blueprint_label = blueprint_label
        self.layout_plan.blueprint_description = blueprint_description

        self.signal_graph = emitter.signal_graph
        self.entities = {
            entity_id: placement
            for entity_id, placement in emitter.entities.items()
            if getattr(placement, "role", None) != "wire_relay"
        }

        self.connection_planner = ConnectionPlanner(
            self.layout_plan,
            self.signal_usage,
            self.diagnostics,
            self.layout_engine,
            max_wire_span=self.max_wire_span,
        )

        locked_colors = emitter._determine_locked_wire_colors()
        self.connection_planner.plan_connections(
            emitter.signal_graph,
            emitter.entities,
            wire_merge_junctions=emitter._wire_merge_junctions,
            locked_colors=locked_colors,
        )

        if self.power_pole_type:
            power_planner = PowerPlanner(
                self.layout_engine,
                self.layout_plan,
                self.diagnostics,
            )
            power_planner.plan_power_grid(self.power_pole_type)

        return self.layout_plan

    def _place_entities(self, ir_operations: list[IRNode]) -> "LegacyBlueprintEmitter":
        """Delegate to existing emitter logic to populate the layout plan."""

        from dsl_compiler.src.emission.emitter import LegacyBlueprintEmitter

        emitter = LegacyBlueprintEmitter(
            self.signal_type_map,
            power_pole_type=self.power_pole_type,
            plan_only_mode=True,
        )
        emitter.diagnostics = self.diagnostics
        emitter.prepare(ir_operations)
        for op in emitter._prepared_operations:
            emitter.emit_ir_operation(op)
        emitter._ensure_export_anchors()
        return emitter
