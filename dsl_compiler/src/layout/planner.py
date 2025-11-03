"""Main layout planning orchestrator."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from dsl_compiler.src.ir.nodes import (
    IRNode,
    IRValue,
    SignalRef,
    IR_Arith,
    IR_Decider,
    IR_MemWrite,
    IR_EntityPropWrite,
    IR_WireMerge,
)
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics

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
from .signal_graph import SignalGraph
from .cluster_analyzer import ClusterAnalyzer, Cluster


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

        # Clustering infrastructure
        self.cluster_analyzer: Optional[ClusterAnalyzer] = None
        self.clusters: List[Cluster] = []
        self.entity_to_cluster: Dict[str, int] = {}

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

        # NEW: Build signal graph before clustering
        self._build_signal_graph(ir_operations)

        # NEW: Clustering phase
        self._analyze_clusters(ir_operations)

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
        self.cluster_analyzer = None
        self.clusters = []
        self.entity_to_cluster = {}

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

    def _build_signal_graph(self, ir_operations: list[IRNode]) -> None:
        """Build signal graph from IR before placement."""

        self.signal_graph = SignalGraph()

        # Add all value-producing operations as sources
        for op in ir_operations:
            if isinstance(op, IRValue):
                self.signal_graph.set_source(op.node_id, op.node_id)

        # Add all consumption edges
        for op in ir_operations:
            if isinstance(op, IR_Arith):
                self._add_value_ref_sink(op.left, op.node_id)
                self._add_value_ref_sink(op.right, op.node_id)
            elif isinstance(op, IR_Decider):
                self._add_value_ref_sink(op.left, op.node_id)
                self._add_value_ref_sink(op.right, op.node_id)
                self._add_value_ref_sink(op.output_value, op.node_id)
            elif isinstance(op, IR_MemWrite):
                self._add_value_ref_sink(op.data_signal, op.node_id)
                self._add_value_ref_sink(op.write_enable, op.node_id)
            elif isinstance(op, IR_EntityPropWrite):
                self._add_value_ref_sink(op.value, op.node_id)
            elif isinstance(op, IR_WireMerge):
                for source in op.sources:
                    self._add_value_ref_sink(source, op.node_id)

    def _add_value_ref_sink(self, value_ref: Any, consumer_id: str) -> None:
        """Helper to add signal sink from ValueRef."""
        if isinstance(value_ref, SignalRef):
            self.signal_graph.add_sink(value_ref.source_id, consumer_id)

    def _analyze_clusters(self, ir_operations: list[IRNode]) -> None:
        """Analyze and form connectivity-based clusters."""

        self.cluster_analyzer = ClusterAnalyzer(self.diagnostics)
        self.clusters = self.cluster_analyzer.analyze(ir_operations, self.signal_graph)
        self.entity_to_cluster = self.cluster_analyzer.entity_to_cluster

    def _plan_connections(self) -> None:
        """Plan wire connections between entities."""
        self.connection_planner = ConnectionPlanner(
            self.layout_plan,
            self.signal_usage,
            self.diagnostics,
            self.layout_engine,
            max_wire_span=self.max_wire_span,
            clusters=self.clusters,
            entity_to_cluster=self.entity_to_cluster,
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
            clusters=self.clusters,
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
            clusters=self.clusters,
            entity_to_cluster=self.entity_to_cluster,
        )

        # Setup cluster bounds before placement
        placer.setup_cluster_bounds()

        for op in ir_operations:
            placer.place_ir_operation(op)

        # ✅ Clean up optimized-away entities
        placer.cleanup_unused_entities()

        # Store signal graph and memory info for connection planning
        # Note: signal_graph already built, but entity_placer might have modified it
        self.signal_graph = placer.signal_graph
        self._memory_modules = placer._memory_modules
        self._wire_merge_junctions = placer._wire_merge_junctions

    def _determine_locked_wire_colors(self) -> Dict[tuple[str, str], str]:
        """Determine wire colors that must be locked for correctness.

        Returns:
            Dict mapping (entity_id, signal_name) → wire_color
        """
        locked = {}

        # Lock colors for non-optimized memory hold gates (standard SR latch)
        for module in self._memory_modules.values():
            for component_name, placement in module.items():
                if component_name == "hold_gate":
                    if not hasattr(placement, "properties"):
                        continue
                    # Memory hold gate output must use red wire for feedback loop stability
                    signal_name = placement.properties.get("output_signal")
                    if signal_name:
                        locked[(placement.ir_node_id, signal_name)] = "red"

        # ✅ NEW: Lock colors for optimized arithmetic feedback combinators
        # These use self-feedback wires hardcoded to red, so we must prevent
        # other signals from using red to the same combinator
        for entity_id, placement in self.layout_plan.entity_placements.items():
            if placement.properties.get("has_self_feedback"):
                feedback_signal = placement.properties.get("feedback_signal")
                if feedback_signal:
                    locked[(entity_id, feedback_signal)] = "red"
                    self.diagnostics.info(
                        f"Locked {entity_id} feedback signal '{feedback_signal}' to red wire"
                    )

        return locked
