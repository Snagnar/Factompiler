"""Main layout planning orchestrator."""

from __future__ import annotations
from .entity_placer import EntityPlacer

from typing import Any, Dict, Optional, List, Tuple

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
        3. Create entities (without positions)
        4. Add power pole grid with fixed positions (if requested)
        5. Optimize positions using integer layout (respecting fixed pole positions)
        6. Plan connections with relay routing
        7. Set metadata

        Args:
            ir_operations: List of IR nodes representing the program
            blueprint_label: Label for the resulting blueprint
            blueprint_description: Description text for the blueprint

        Returns:
            LayoutPlan containing entity placements and wire connections
        """
        # Phase 1: Analysis
        self._setup_signal_analysis(ir_operations)

        # Phase 2: Create entities (no positions yet, signal graph built during placement)
        self._create_entities(ir_operations)

        # Phase 3: Add power pole grid BEFORE optimization (if requested)
        # Power poles are given fixed positions in a grid pattern
        self._add_power_pole_grid()

        # Phase 4: Optimize positions using integer layout (respects fixed pole positions)
        self._optimize_positions()

        # Phase 5: Infrastructure & connections
        self._update_tile_grid()
        self._plan_connections()
        self._update_tile_grid()  # Again after relays added

        # Phase 6: Metadata
        self._set_metadata(blueprint_label, blueprint_description)

        return self.layout_plan

    def _setup_signal_analysis(self, ir_operations: list[IRNode]) -> None:
        """Initialize and run signal analysis with materialization."""
        self.signal_analyzer = SignalAnalyzer(self.diagnostics, self.signal_type_map)
        self.signal_usage = self.signal_analyzer.analyze(ir_operations)
        # Note: analyze() now calls finalize_materialization() internally

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

        # Create all entities (positions will be None)
        for op in ir_operations:
            placer.place_ir_operation(op)

        # Create output anchor combinators for signals with no consumers
        placer.create_output_anchors()

        # Clean up optimized-away entities
        placer.cleanup_unused_entities()

        # Store signal graph and metadata
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

        # Convert tile positions (integer grid) to center positions (may be half-integer)
        # The solver returns tile coordinates (top-left corner), but Factorio/Draftsman
        # uses entity center positions.
        for entity_id, (tile_x, tile_y) in optimized_positions.items():
            placement = self.layout_plan.entity_placements.get(entity_id)
            if placement:
                footprint = placement.properties.get("footprint", (1, 1))
                width, height = footprint

                # Convert tile position to center position
                # For a 1x1 entity at tile (0,0): center is (0.5, 0.5)
                # For a 1x2 entity at tile (0,0): center is (0.5, 1.0)
                # For a 2x2 entity at tile (0,0): center is (1.0, 1.0)
                center_x = tile_x + width / 2.0
                center_y = tile_y + height / 2.0

                placement.position = (center_x, center_y)

        self.diagnostics.info("Entity positions optimized using integer layout engine.")

    def _detect_overlaps(
        self, tile_positions: Dict[str, Tuple[int, int]]
    ) -> List[Tuple[str, str]]:
        """Detect overlapping entities using exact tile-based collision detection.

        Args:
            tile_positions: Dict mapping entity_id to (tile_x, tile_y) top-left corner

        Returns:
            List of (entity_id1, entity_id2) pairs that overlap
        """
        overlaps = []
        entities = list(tile_positions.keys())

        for i, entity_id1 in enumerate(entities):
            for entity_id2 in entities[i + 1 :]:
                if self._entities_overlap(entity_id1, entity_id2, tile_positions):
                    overlaps.append((entity_id1, entity_id2))

        return overlaps

    def _entities_overlap(
        self,
        entity_id1: str,
        entity_id2: str,
        tile_positions: Dict[str, Tuple[int, int]],
    ) -> bool:
        """Check if two entities overlap using tile-based AABB collision.

        Args:
            entity_id1: First entity ID
            entity_id2: Second entity ID
            tile_positions: Dict mapping entity_id to (tile_x, tile_y)

        Returns:
            True if entities overlap
        """
        placement1 = self.layout_plan.entity_placements.get(entity_id1)
        placement2 = self.layout_plan.entity_placements.get(entity_id2)

        if not placement1 or not placement2:
            return False

        tile_x1, tile_y1 = tile_positions[entity_id1]
        tile_x2, tile_y2 = tile_positions[entity_id2]

        footprint1 = placement1.properties.get("footprint", (1, 1))
        footprint2 = placement2.properties.get("footprint", (1, 1))

        width1, height1 = footprint1
        width2, height2 = footprint2

        # AABB (Axis-Aligned Bounding Box) overlap test
        # Two rectangles overlap if they overlap on both X and Y axes

        # X-axis overlap: [x1, x1+w1) overlaps [x2, x2+w2)
        x_overlap = tile_x1 < tile_x2 + width2 and tile_x2 < tile_x1 + width1

        # Y-axis overlap: [y1, y1+h1) overlaps [y2, y2+h2)
        y_overlap = tile_y1 < tile_y2 + height2 and tile_y2 < tile_y1 + height1

        return x_overlap and y_overlap

    def _resolve_overlaps(
        self,
        tile_positions: Dict[str, Tuple[int, int]],
        overlaps: List[Tuple[str, str]],
    ) -> Dict[str, Tuple[int, int]]:
        """Resolve overlaps by moving entities to nearby free positions.

        Strategy:
        1. Sort entities by priority (larger/more connected entities have higher priority)
        2. Keep high-priority entities in place
        3. Move low-priority entities to nearest free position

        Args:
            tile_positions: Dict mapping entity_id to (tile_x, tile_y)
            overlaps: List of overlapping (entity_id1, entity_id2) pairs

        Returns:
            Dict mapping entity_id to resolved (tile_x, tile_y) positions
        """
        # Create a copy to modify
        resolved_positions = dict(tile_positions)

        # Build a set of entities involved in overlaps
        entities_to_move = set()
        for entity_id1, entity_id2 in overlaps:
            entities_to_move.add(entity_id1)
            entities_to_move.add(entity_id2)

        # Sort entities by priority (keep more important entities in place)
        priority_entities = self._sort_by_priority(list(entities_to_move))

        # Track occupied tiles
        occupied_tiles = self._build_occupied_tiles_map(resolved_positions)

        # Process entities from lowest to highest priority
        # (high priority entities keep their positions)
        for entity_id in reversed(priority_entities):
            # Check if this entity still overlaps after previous moves
            current_tile_pos = resolved_positions[entity_id]

            if self._position_has_overlap(
                entity_id, current_tile_pos, resolved_positions, occupied_tiles
            ):
                # Find nearest free position
                footprint = self.layout_plan.entity_placements[
                    entity_id
                ].properties.get("footprint", (1, 1))

                new_tile_pos = self._find_nearest_free_position(
                    current_tile_pos, footprint, occupied_tiles, max_search_radius=20
                )

                if new_tile_pos:
                    # Update position
                    old_pos = resolved_positions[entity_id]
                    resolved_positions[entity_id] = new_tile_pos

                    # Update occupied tiles map
                    self._remove_from_occupied_tiles(entity_id, old_pos, occupied_tiles)
                    self._add_to_occupied_tiles(
                        entity_id, new_tile_pos, footprint, occupied_tiles
                    )

                    self.diagnostics.info(
                        f"Moved overlapping entity {entity_id} from {old_pos} to {new_tile_pos}"
                    )
                else:
                    self.diagnostics.warning(
                        f"Could not find free position for entity {entity_id} within search radius"
                    )

        return resolved_positions

    def _sort_by_priority(self, entity_ids: List[str]) -> List[str]:
        """Sort entities by priority for overlap resolution.

        Priority (highest to lowest):
        1. Entities with user-specified positions (don't move)
        2. Entities with many connections (hub entities)
        3. Larger entities (harder to place)
        4. Everything else

        Returns list sorted from HIGH to LOW priority (keep high priority in place)
        """

        def priority_score(entity_id: str) -> Tuple[int, int, int]:
            placement = self.layout_plan.entity_placements.get(entity_id)
            if not placement:
                return (0, 0, 0)

            # Priority 1: User-specified positions (highest priority)
            user_specified = placement.properties.get("user_specified_position", False)

            # Priority 2: Connection count
            connection_count = 0
            for (
                signal_id,
                source_id,
                sink_id,
            ) in self.signal_graph.iter_source_sink_pairs():
                if source_id == entity_id or sink_id == entity_id:
                    connection_count += 1

            # Priority 3: Entity size (larger = higher priority)
            footprint = placement.properties.get("footprint", (1, 1))
            size = footprint[0] * footprint[1]

            return (
                1 if user_specified else 0,  # User-specified has highest priority
                connection_count,
                size,
            )

        return sorted(entity_ids, key=priority_score)

    def _build_occupied_tiles_map(
        self, tile_positions: Dict[str, Tuple[int, int]]
    ) -> Dict[Tuple[int, int], str]:
        """Build a map of occupied tiles.

        Returns:
            Dict mapping (tile_x, tile_y) to entity_id occupying that tile
        """
        occupied = {}

        for entity_id, (tile_x, tile_y) in tile_positions.items():
            placement = self.layout_plan.entity_placements.get(entity_id)
            if not placement:
                continue

            footprint = placement.properties.get("footprint", (1, 1))
            width, height = footprint

            for dx in range(width):
                for dy in range(height):
                    tile = (tile_x + dx, tile_y + dy)
                    occupied[tile] = entity_id

        return occupied

    def _position_has_overlap(
        self,
        entity_id: str,
        tile_pos: Tuple[int, int],
        tile_positions: Dict[str, Tuple[int, int]],
        occupied_tiles: Dict[Tuple[int, int], str],
    ) -> bool:
        """Check if an entity at given position overlaps with any other entity."""
        placement = self.layout_plan.entity_placements.get(entity_id)
        if not placement:
            return False

        tile_x, tile_y = tile_pos
        footprint = placement.properties.get("footprint", (1, 1))
        width, height = footprint

        for dx in range(width):
            for dy in range(height):
                tile = (tile_x + dx, tile_y + dy)
                occupier = occupied_tiles.get(tile)
                if occupier and occupier != entity_id:
                    return True

        return False

    def _find_nearest_free_position(
        self,
        center_pos: Tuple[int, int],
        footprint: Tuple[int, int],
        occupied_tiles: Dict[Tuple[int, int], str],
        max_search_radius: int = 20,
    ) -> Optional[Tuple[int, int]]:
        """Find nearest free position for an entity using spiral search.

        Args:
            center_pos: Starting position (tile_x, tile_y)
            footprint: (width, height) in tiles
            occupied_tiles: Map of occupied tiles
            max_search_radius: Maximum search distance

        Returns:
            (tile_x, tile_y) of free position, or None if not found
        """
        center_x, center_y = center_pos
        width, height = footprint

        # Spiral search outward from center
        for radius in range(max_search_radius + 1):
            # Try positions at this radius
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # Only check positions on the perimeter of current radius
                    if abs(dx) != radius and abs(dy) != radius:
                        continue

                    candidate_x = center_x + dx
                    candidate_y = center_y + dy

                    # Check if position is free
                    if self._is_position_free(
                        candidate_x, candidate_y, footprint, occupied_tiles
                    ):
                        return (candidate_x, candidate_y)

        return None

    def _is_position_free(
        self,
        tile_x: int,
        tile_y: int,
        footprint: Tuple[int, int],
        occupied_tiles: Dict[Tuple[int, int], str],
    ) -> bool:
        """Check if a position is completely free."""
        width, height = footprint

        for dx in range(width):
            for dy in range(height):
                tile = (tile_x + dx, tile_y + dy)
                if tile in occupied_tiles:
                    return False

        return True

    def _remove_from_occupied_tiles(
        self,
        entity_id: str,
        tile_pos: Tuple[int, int],
        occupied_tiles: Dict[Tuple[int, int], str],
    ) -> None:
        """Remove entity from occupied tiles map."""
        placement = self.layout_plan.entity_placements.get(entity_id)
        if not placement:
            return

        tile_x, tile_y = tile_pos
        footprint = placement.properties.get("footprint", (1, 1))
        width, height = footprint

        for dx in range(width):
            for dy in range(height):
                tile = (tile_x + dx, tile_y + dy)
                if occupied_tiles.get(tile) == entity_id:
                    del occupied_tiles[tile]

    def _add_to_occupied_tiles(
        self,
        entity_id: str,
        tile_pos: Tuple[int, int],
        footprint: Tuple[int, int],
        occupied_tiles: Dict[Tuple[int, int], str],
    ) -> None:
        """Add entity to occupied tiles map."""
        tile_x, tile_y = tile_pos
        width, height = footprint

        for dx in range(width):
            for dy in range(height):
                tile = (tile_x + dx, tile_y + dy)
                occupied_tiles[tile] = entity_id

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

        # ✅ Pass memory modules for feedback edge detection
        self.connection_planner._memory_modules = self._memory_modules

        locked_colors = self._determine_locked_wire_colors()

        self.connection_planner.plan_connections(
            self.signal_graph,
            self.layout_plan.entity_placements,
            wire_merge_junctions=self._wire_merge_junctions,
            locked_colors=locked_colors,
        )

        # Inject wire color information into combinator placements
        self._inject_wire_colors_into_placements()

    def _inject_wire_colors_into_placements(self) -> None:
        """Store wire color information in combinator placement properties.

        After wire connections are planned, we know which wire colors (red/green)
        deliver each signal to each entity. Store this information in the placement
        properties so the entity emitter can configure wire filters on combinators.

        For combinators where both operands use the same signal name but from different
        sources (e.g., both read signal-A but from different deciders), we use the
        IR signal IDs to determine which source entity produces each operand, then
        look up the wire color for that specific edge.
        """
        injected_count = 0
        for placement in self.layout_plan.entity_placements.values():
            if placement.entity_type not in (
                "arithmetic-combinator",
                "decider-combinator",
            ):
                continue

            # Get the operand signals from placement properties
            # These are RESOLVED signal names (e.g., "signal-A") not IR types (e.g., "__v1")
            left_signal = placement.properties.get("left_operand")
            right_signal = placement.properties.get("right_operand")

            # Get the IR signal IDs for each operand to find their source entities
            left_signal_id = placement.properties.get("left_operand_signal_id")
            right_signal_id = placement.properties.get("right_operand_signal_id")

            # Get wire colors for each operand based on its source entity
            # Skip constants (integers) - they don't need wire filtering
            if left_signal and not isinstance(left_signal, int) and left_signal_id:
                # The signal ID might be a string representation like "signal-A@decider_3"
                # Extract the source_id from it
                if isinstance(left_signal_id, str) and "@" in left_signal_id:
                    # Parse "signal_type@source_id" format
                    source_entity = left_signal_id.split("@")[1]
                elif hasattr(left_signal_id, "source_id"):
                    # It's a SignalRef object
                    source_entity = left_signal_id.source_id
                else:
                    # Try to find source via signal_graph
                    source_entity = self.signal_graph.get_source(str(left_signal_id))

                if source_entity:
                    # Look up the wire color for this specific edge
                    left_color = self.connection_planner.get_wire_color_for_edge(
                        source_entity, placement.ir_node_id, left_signal
                    )
                    # Store as a set with only this color
                    placement.properties["left_operand_wires"] = {left_color}
                    self.diagnostics.info(
                        f"Injected left_operand_wires={{'{left_color}'}} for {placement.ir_node_id} "
                        f"signal={left_signal} from {source_entity}"
                    )
                    injected_count += 1
                else:
                    # Fallback: read from both wires if source not found
                    placement.properties["left_operand_wires"] = {"red", "green"}
                    self.diagnostics.warning(
                        f"No source found for left operand signal {left_signal_id} of {placement.ir_node_id}"
                    )

            if right_signal and not isinstance(right_signal, int) and right_signal_id:
                # The signal ID might be a string representation like "signal-A@decider_3"
                # Extract the source_id from it
                if isinstance(right_signal_id, str) and "@" in right_signal_id:
                    # Parse "signal_type@source_id" format
                    source_entity = right_signal_id.split("@")[1]
                elif hasattr(right_signal_id, "source_id"):
                    # It's a SignalRef object
                    source_entity = right_signal_id.source_id
                else:
                    # Try to find source via signal_graph
                    source_entity = self.signal_graph.get_source(str(right_signal_id))

                if source_entity:
                    # Look up the wire color for this specific edge
                    right_color = self.connection_planner.get_wire_color_for_edge(
                        source_entity, placement.ir_node_id, right_signal
                    )
                    # Store as a set with only this color
                    placement.properties["right_operand_wires"] = {right_color}
                    self.diagnostics.info(
                        f"Injected right_operand_wires={{'{right_color}'}} for {placement.ir_node_id} "
                        f"signal={right_signal} from {source_entity}"
                    )
                    injected_count += 1
                else:
                    # Fallback: read from both wires if source not found
                    placement.properties["right_operand_wires"] = {"red", "green"}
                    self.diagnostics.warning(
                        f"No source found for right operand signal {right_signal_id} of {placement.ir_node_id}"
                    )

        self.diagnostics.info(
            f"Wire color injection: {injected_count} operands configured"
        )

    def _add_power_pole_grid(self) -> None:
        """Add power poles in a grid pattern with fixed positions.

        This creates a regular grid of power poles that covers the estimated
        blueprint area plus a safety margin. The poles have fixed positions
        that are respected by the layout optimizer.
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

        power_planner.add_power_pole_grid(self.power_pole_type)

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

        # ===================================================================
        # PART 1: Lock memory gate outputs
        # ===================================================================
        for module in self._memory_modules.values():
            if isinstance(module, MemoryModule):
                # Only for standard SR latches (not optimized memories)
                if module.optimization is None:
                    # Lock data/feedback channel to RED for both gates
                    if module.write_gate:
                        locked[(module.write_gate.ir_node_id, module.signal_type)] = (
                            "red"
                        )
                    if module.hold_gate:
                        locked[(module.hold_gate.ir_node_id, module.signal_type)] = (
                            "red"
                        )

        # ===================================================================
        # PART 2: Lock signal-W (control signal) to GREEN globally
        # ===================================================================
        # signal-W is reserved for memory control and must use GREEN wire
        # to separate from data signals that use RED

        # Find all sources that output signal-W and lock them to GREEN
        for signal_id, source_ids, sink_ids in self.signal_graph.iter_edges():
            # Check if this is signal-W (the memory control signal)
            usage = self.signal_usage.get(signal_id)
            resolved_name = usage.resolved_signal_name if usage else None

            if resolved_name == "signal-W" or signal_id == "signal-W":
                # Lock all sources of signal-W to GREEN
                for source_id in source_ids:
                    locked[(source_id, "signal-W")] = "green"

        # ===================================================================
        # PART 3: Lock data signals going TO memory gates to RED
        # ===================================================================
        # For each memory, find what signal is being written (the data signal)
        # and lock its source to RED to match the feedback channel

        for module in self._memory_modules.values():
            if isinstance(module, MemoryModule):
                if module.optimization is None and module.write_gate:
                    write_gate_id = module.write_gate.ir_node_id
                    data_signal = module.signal_type  # e.g., "signal-B"

                    # Find sources that feed this data signal to the write gate
                    for (
                        signal_id,
                        source_ids,
                        sink_ids,
                    ) in self.signal_graph.iter_edges():
                        # Check if this edge connects to write_gate with the data signal
                        if write_gate_id in sink_ids:
                            usage = self.signal_usage.get(signal_id)
                            resolved_name = (
                                usage.resolved_signal_name if usage else None
                            )

                            # If this is the data signal, lock its sources to RED
                            if resolved_name == data_signal or signal_id == data_signal:
                                for source_id in source_ids:
                                    # Don't re-lock the memory gates themselves
                                    if (
                                        source_id != write_gate_id
                                        and source_id != module.hold_gate.ir_node_id
                                    ):
                                        locked[(source_id, data_signal)] = "red"

        # ===================================================================
        # PART 4: Lock optimized arithmetic feedback (existing code)
        # ===================================================================
        for entity_id, placement in self.layout_plan.entity_placements.items():
            if placement.properties.get("has_self_feedback"):
                feedback_signal = placement.properties.get("feedback_signal")
                if feedback_signal:
                    locked[(entity_id, feedback_signal)] = "red"
                    self.diagnostics.info(
                        f"Locked {entity_id} feedback signal '{feedback_signal}' to red wire"
                    )

        return locked
