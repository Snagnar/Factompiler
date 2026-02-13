"""Main layout planning orchestrator."""

from __future__ import annotations

from typing import Any

from dsl_compiler.src.common.constants import DEFAULT_CONFIG, CompilerConfig
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.ir.nodes import IRNode
from dsl_compiler.src.layout.integer_layout_solver import IntegerLayoutEngine

from .connection_planner import ConnectionPlanner
from .entity_placer import EntityPlacer
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
        signal_type_map: dict[str, str],
        diagnostics: ProgramDiagnostics,
        *,
        signal_refs: dict[str, Any] | None = None,
        referenced_signal_names: set | None = None,
        power_pole_type: str | None = None,
        max_wire_span: float = 9.0,
        config: CompilerConfig = DEFAULT_CONFIG,
        use_mst_optimization: bool = True,
        max_layout_retries: int = 3,
    ) -> None:
        self.signal_type_map = signal_type_map
        self.signal_refs = signal_refs or {}
        self.referenced_signal_names = referenced_signal_names or set()
        self.diagnostics = diagnostics
        self.diagnostics.default_stage = "layout_planning"
        self.power_pole_type = power_pole_type
        self.max_wire_span = max_wire_span
        self.config = config
        self.use_mst_optimization = use_mst_optimization
        self.max_layout_retries = max_layout_retries

        self.tile_grid = TileGrid()
        self.layout_plan = LayoutPlan()

        self.signal_analyzer: SignalAnalyzer | None = None
        self.signal_usage: dict[str, SignalUsageEntry] = {}
        self.connection_planner: ConnectionPlanner | None = None
        self.signal_graph: Any = None
        self._memory_modules: dict[str, Any] = {}
        self._wire_merge_junctions: dict[str, Any] = {}
        self._merge_membership: dict[str, set] = {}

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

        If relay routing fails, the layout is retried with relaxed parameters up to
        max_layout_retries times before giving up.

        Args:
            ir_operations: List of IR nodes representing the program
            blueprint_label: Label for the resulting blueprint
            blueprint_description: Description text for the blueprint

        Returns:
            LayoutPlan containing entity placements and wire connections
        """
        self._setup_signal_analysis(ir_operations)

        for attempt in range(self.max_layout_retries + 1):
            # Reset state for this attempt
            self._reset_layout_state()

            # Run layout phases
            self._create_entities(ir_operations)
            self._add_power_pole_grid()

            # Vary optimization parameters on retries for different results
            time_multiplier = 1.0 + (attempt * 0.5)  # More time on retries
            self._optimize_positions(time_multiplier=time_multiplier)

            self._trim_power_poles()
            self._update_tile_grid()

            # Plan connections - returns False if relay routing failed
            routing_succeeded = self._plan_connections()

            if not routing_succeeded:
                if attempt < self.max_layout_retries:
                    self.diagnostics.warning(
                        f"Layout attempt {attempt + 1} failed due to relay routing issues, "
                        f"retrying with relaxed parameters ({self.max_layout_retries - attempt} retries remaining)..."
                    )
                    continue
                # Final attempt failed
                self.diagnostics.error(
                    f"Layout failed after {self.max_layout_retries + 1} attempts: "
                    f"relay routing could not find valid paths. "
                    f"The layout may be too spread out for the available wire span."
                )
                break

            # Success - no routing failures
            if attempt > 0:
                self.diagnostics.warning(f"Layout succeeded after {attempt + 1} attempt(s).")
            break

        self._update_tile_grid()  # Final update after relays added
        self._set_metadata(blueprint_label, blueprint_description)

        return self.layout_plan

    def _reset_layout_state(self) -> None:
        """Reset layout state for a fresh attempt."""
        self.tile_grid = TileGrid()
        self.layout_plan = LayoutPlan()
        self.connection_planner = None
        self._memory_modules = {}
        self._wire_merge_junctions = {}
        self._merge_membership = {}

    def _setup_signal_analysis(self, ir_operations: list[IRNode]) -> None:
        """Initialize and run signal analysis with materialization."""
        self.signal_analyzer = SignalAnalyzer(
            self.diagnostics,
            self.signal_type_map,
            self.signal_refs,
            self.referenced_signal_names,
        )
        self.signal_usage = self.signal_analyzer.analyze(ir_operations)

    def _create_entities(self, ir_operations: list[IRNode]) -> None:
        """Create entity records without positions.

        Entities are created with footprints and properties, but positions
        are assigned later by force-directed optimization.
        """

        if self.signal_analyzer is None:
            raise RuntimeError("Signal analyzer not initialized")

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
        self._merge_membership = placer.get_merge_membership()

    def _optimize_positions(self, time_multiplier: float = 1.0) -> None:
        """Optimize entity positions using force-directed layout.

        Args:
            time_multiplier: Multiplier for the solver time limit, used during retries
                           to give the solver more time to find better solutions.
        """
        layout_engine = IntegerLayoutEngine(
            signal_graph=self.signal_graph,
            entity_placements=self.layout_plan.entity_placements,
            diagnostics=self.diagnostics,
            config=self.config,
            wire_merge_junctions=self._wire_merge_junctions,
        )

        time_limit = self.config.layout_solver_time_limit * time_multiplier
        optimized_positions = layout_engine.optimize(time_limit_seconds=int(time_limit))

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

    def _plan_connections(self) -> bool:
        """Plan wire connections between entities.

        Returns:
            True if all connections were successfully routed, False if any
            relay routing failed.
        """
        self.connection_planner = ConnectionPlanner(
            self.layout_plan,
            self.signal_usage,
            self.diagnostics,
            self.tile_grid,
            max_wire_span=self.max_wire_span,
            power_pole_type=self.power_pole_type,
            config=self.config,
            use_mst_optimization=self.use_mst_optimization,
        )

        self.connection_planner._memory_modules = self._memory_modules

        return self.connection_planner.plan_connections(
            self.signal_graph,
            self.layout_plan.entity_placements,
            wire_merge_junctions=self._wire_merge_junctions,
            merge_membership=self._merge_membership,
        )

    def _add_power_pole_grid(self) -> None:
        """Add power poles in a grid pattern BEFORE layout optimization.

        This is called BEFORE layout optimization so that the layout engine
        treats poles as fixed obstacles and places combinators around them.
        After layout, we trim unused poles with _trim_power_poles().
        """
        if not self.power_pole_type:
            return

        # Update tile grid with user-specified positions BEFORE placing poles
        # This prevents poles from overlapping with user-placed entities
        self._update_tile_grid()

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

        supply_radius_raw = config["supply_radius"]
        # Cast object to float, or use 0.0 if None
        if supply_radius_raw is None:
            supply_radius = 0.0
        elif isinstance(supply_radius_raw, (int, float)):
            supply_radius = float(supply_radius_raw)
        else:
            supply_radius = 0.0

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
