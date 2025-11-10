from __future__ import annotations
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple
from dsl_compiler.src.ir.builder import SignalRef
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from .layout_engine import LayoutEngine
from .layout_plan import LayoutPlan, WireConnection, EntityPlacement
from .signal_analyzer import SignalUsageEntry

from .wire_router import (
    CircuitEdge,
    WIRE_COLORS,
    collect_circuit_edges,
    plan_wire_colors,
)
from collections import defaultdict


"""Connection planning for wire routing."""


class RelayNode:
    """Represents a pole that can relay circuit signals."""

    def __init__(self, position: Tuple[float, float], entity_id: str, pole_type: str):
        self.position = position
        self.entity_id = entity_id
        self.pole_type = pole_type
        # Track which wire colors are in use on this pole
        # Key: signal_name, Value: wire_color
        self.signal_assignments: Dict[str, str] = {}

    def can_carry_signal(self, signal_name: str, preferred_color: str) -> Optional[str]:
        """
        Check if this pole can carry a signal.
        Returns wire color to use, or None if pole is full.
        """
        if signal_name in self.signal_assignments:
            return self.signal_assignments[signal_name]

        # Check if preferred color available
        signals_on_red = [s for s, c in self.signal_assignments.items() if c == "red"]
        signals_on_green = [
            s for s, c in self.signal_assignments.items() if c == "green"
        ]

        if preferred_color == "red" and len(signals_on_red) == 0:
            return "red"
        elif preferred_color == "green" and len(signals_on_green) == 0:
            return "green"

        # Try opposite color
        other_color = "green" if preferred_color == "red" else "red"
        if other_color == "red" and len(signals_on_red) == 0:
            return "red"
        elif other_color == "green" and len(signals_on_green) == 0:
            return "green"

        return None  # Pole is full (both colors used)

    def assign_signal(self, signal_name: str, wire_color: str):
        """Record that this signal uses this wire color on this pole."""
        self.signal_assignments[signal_name] = wire_color


class RelayNetwork:
    """Manages shared relay infrastructure for inter-cluster routing."""

    def __init__(
        self,
        layout_engine,
        clusters,
        entity_to_cluster,
        max_span: float,
        layout_plan,
        diagnostics,
    ):
        self.layout_engine = layout_engine
        self.clusters = clusters
        self.entity_to_cluster = entity_to_cluster
        self.max_span = max_span
        self.layout_plan = layout_plan
        self.diagnostics = diagnostics
        # Map: pole_position → RelayNode
        self.relay_nodes: Dict[Tuple[int, int], RelayNode] = {}
        # Spatial index for finding nearby poles
        self.pole_grid: Dict[Tuple[int, int], List[RelayNode]] = defaultdict(list)
        self._relay_counter = 0

    def add_relay_node(
        self, position: Tuple[float, float], entity_id: str, pole_type: str
    ) -> RelayNode:
        """Add a pole to the relay network."""
        tile_pos = (int(round(position[0])), int(round(position[1])))

        if tile_pos in self.relay_nodes:
            return self.relay_nodes[tile_pos]

        node = RelayNode(position, entity_id, pole_type)
        self.relay_nodes[tile_pos] = node

        # Add to spatial grid
        grid_x = tile_pos[0] // 10
        grid_y = tile_pos[1] // 10
        self.pole_grid[(grid_x, grid_y)].append(node)

        return node

    def find_relay_near(
        self, position: Tuple[float, float], max_distance: float
    ) -> Optional[RelayNode]:
        """Find existing relay pole near position, or None."""
        grid_x = int(position[0]) // 10
        grid_y = int(position[1]) // 10

        # Check nearby grid cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for node in self.pole_grid.get((grid_x + dx, grid_y + dy), []):
                    dist = math.dist(position, node.position)
                    if dist <= max_distance:
                        return node
        return None

    def route_signal(
        self,
        source_pos: Tuple[float, float],
        sink_pos: Tuple[float, float],
        signal_name: str,
        wire_color: str,
    ) -> List[Tuple[str, str]]:  # Returns list of (entity_id, wire_color) along path
        """
        Find or create relay path from source to sink.
        Returns list of (entity_id, wire_color) pairs representing the routing path.
        """
        distance = math.dist(source_pos, sink_pos)

        # Account for entity size and wire reach offset (same as in _validate_relay_coverage)
        span_limit = max(1.0, float(self.max_span) - 1.8)

        # Direct connection if close enough
        # Use 0.95 factor to account for entity footprints and placement imprecision
        if distance <= span_limit * 0.95:
            return []  # No relays needed

        # Determine how many relay points needed
        # Use very conservative 0.55x of span_limit as safe interval to account for:
        # - Entity footprints (up to 1 tile each side)
        # - Placement imprecision (relays might not land exactly on ideal spot)
        # - Reserve failures requiring nearby placement
        # - Entity center positions might not be tile-aligned
        safe_interval = span_limit * 0.55
        num_relays = max(1, int(math.ceil(distance / safe_interval)) - 1)

        path = []

        for i in range(1, num_relays + 1):
            # Calculate ideal relay position
            ratio = i / (num_relays + 1)
            relay_x = source_pos[0] + (sink_pos[0] - source_pos[0]) * ratio
            relay_y = source_pos[1] + (sink_pos[1] - source_pos[1]) * ratio
            ideal_pos = (relay_x, relay_y)

            # Try to find existing relay nearby, but only if it's very close (within 0.5 tiles)
            # This prevents accumulating distance errors when reusing relays
            relay_node = self.find_relay_near(ideal_pos, max_distance=0.7)

            if relay_node is None:
                # Create new relay
                tile_pos = (int(round(relay_x)), int(round(relay_y)))
                # Try exact position first
                reserved_pos = self.layout_engine.reserve_exact(
                    tile_pos, footprint=(1, 1), for_infrastructure=True
                )
                if reserved_pos is None:
                    # Fall back to finding nearby position (without for_infrastructure parameter)
                    reserved_pos = self.layout_engine.reserve_near(
                        tile_pos, max_radius=3, footprint=(1, 1)
                    )

                if reserved_pos:
                    self._relay_counter += 1
                    relay_id = f"__relay_{self._relay_counter}"
                    # Convert tile to center position for draftsman
                    center_pos = (reserved_pos[0] + 0.5, reserved_pos[1] + 0.5)
                    relay_node = self.add_relay_node(
                        center_pos, relay_id, "medium-electric-pole"
                    )

                    # Create entity placement for this relay
                    relay_placement = EntityPlacement(
                        ir_node_id=relay_id,
                        entity_type="medium-electric-pole",
                        position=center_pos,
                        properties={
                            "debug_info": {
                                "variable": f"relay_{self._relay_counter}",
                                "operation": "infrastructure",
                                "details": "wire_relay",
                                "role": "relay",
                            }
                        },
                        role="wire_relay",
                    )
                    self.layout_plan.add_placement(relay_placement)

            if relay_node:
                # Try to assign signal to this relay
                assigned_color = relay_node.can_carry_signal(signal_name, wire_color)
                if assigned_color:
                    relay_node.assign_signal(signal_name, assigned_color)
                    path.append((relay_node.entity_id, assigned_color))
                else:
                    # Pole is full, need to create new one nearby
                    placed_fallback = False
                    # Try offset positions
                    for offset_dx in [-1, 1, 0]:
                        for offset_dy in [-1, 1, 0]:
                            if offset_dx == 0 and offset_dy == 0:
                                continue
                            offset_pos = (relay_x + offset_dx, relay_y + offset_dy)
                            tile_offset = (
                                int(round(offset_pos[0])),
                                int(round(offset_pos[1])),
                            )
                            reserved_pos = self.layout_engine.reserve_exact(
                                tile_offset, footprint=(1, 1), for_infrastructure=True
                            )
                            if reserved_pos:
                                self._relay_counter += 1
                                new_relay_id = f"__relay_{self._relay_counter}"
                                center_pos = (
                                    reserved_pos[0] + 0.5,
                                    reserved_pos[1] + 0.5,
                                )
                                new_node = self.add_relay_node(
                                    center_pos, new_relay_id, "medium-electric-pole"
                                )

                                # Create entity placement
                                relay_placement = EntityPlacement(
                                    ir_node_id=new_relay_id,
                                    entity_type="medium-electric-pole",
                                    position=center_pos,
                                    properties={
                                        "debug_info": {
                                            "variable": f"relay_{self._relay_counter}",
                                            "operation": "infrastructure",
                                            "details": "wire_relay",
                                            "role": "relay",
                                        }
                                    },
                                    role="wire_relay",
                                )
                                self.layout_plan.add_placement(relay_placement)

                                new_node.assign_signal(signal_name, wire_color)
                                path.append((new_node.entity_id, wire_color))
                                placed_fallback = True
                                break
                        if placed_fallback:
                            break

                    if not placed_fallback:
                        self.diagnostics.warning(
                            f"Failed to place relay for {signal_name} - pole at {ideal_pos} is full and no nearby positions available"
                        )
            else:
                # Relay placement completely failed
                self.diagnostics.warning(
                    f"Failed to place relay for {signal_name} at {ideal_pos} - no space available"
                )

        return path


class ConnectionPlanner:
    """Plans all wire connections for a blueprint.

    Uses a greedy straight-line approach for relay placement:
    1. Calculates direct path between source and sink
    2. Places relays at evenly-spaced intervals if distance exceeds span limit
    3. Snaps relay positions to grid for clean layouts
    4. Reuses existing relays within 30% of span limit
    5. Assigns wire colors to isolate conflicting signal producers

    This approach ensures predictable, clean layouts without complex pathfinding.
    """

    def __init__(
        self,
        layout_plan: LayoutPlan,
        signal_usage: Dict[str, SignalUsageEntry],
        diagnostics: ProgramDiagnostics,
        layout_engine: LayoutEngine,
        max_wire_span: float = 9.0,
        clusters: Optional[List[Any]] = None,
        entity_to_cluster: Optional[Dict[str, int]] = None,
        power_pole_type: Optional[str] = None,
    ) -> None:
        self.layout_plan = layout_plan
        self.signal_usage = signal_usage
        self.diagnostics = diagnostics
        self.max_wire_span = max_wire_span
        self.layout_engine = layout_engine
        self.power_pole_type = power_pole_type

        self._circuit_edges: List[CircuitEdge] = []
        self._node_color_assignments: Dict[Tuple[str, str], str] = {}
        self._edge_color_map: Dict[Tuple[str, str, str], str] = {}
        self._coloring_conflicts = []
        self._coloring_success = True
        self._relay_counter = 0

        # Cluster support
        self.clusters = clusters or []
        self.entity_to_cluster = entity_to_cluster or {}

        # Create relay network for shared infrastructure
        self.relay_network = RelayNetwork(
            self.layout_engine,
            self.clusters,
            self.entity_to_cluster,
            self.max_wire_span,
            self.layout_plan,
            self.diagnostics,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_connections(
        self,
        signal_graph: Any,
        entities: Dict[str, Any],
        wire_merge_junctions: Optional[Dict[str, Any]] = None,
        locked_colors: Optional[Dict[Tuple[str, str], str]] = None,
    ) -> None:
        """Compute all wire connections with color assignments."""
        preserved_connections = list(self.layout_plan.wire_connections)
        self.layout_plan.wire_connections.clear()
        self._circuit_edges = []
        self._node_color_assignments = {}
        self._edge_color_map = {}
        self._coloring_conflicts = []
        self._coloring_success = True
        self._relay_counter = 0

        # ✅ Add self-feedback for optimized arithmetic memories
        self._add_self_feedback_connections()

        base_edges = collect_circuit_edges(signal_graph, self.signal_usage, entities)
        expanded_edges = self._expand_merge_edges(
            base_edges, wire_merge_junctions, entities
        )
        self._circuit_edges = expanded_edges

        self._log_multi_source_conflicts(expanded_edges, entities)

        coloring_result = plan_wire_colors(expanded_edges, locked_colors)
        self._node_color_assignments = coloring_result.assignments
        self._coloring_conflicts = coloring_result.conflicts
        self._coloring_success = coloring_result.is_bipartite

        edge_color_map: Dict[Tuple[str, str, str], str] = {}
        for edge in expanded_edges:
            if not edge.source_entity_id:
                continue
            node_key = (edge.source_entity_id, edge.resolved_signal_name)
            color = self._node_color_assignments.get(node_key, WIRE_COLORS[0])
            edge_color_map[
                (edge.source_entity_id, edge.sink_entity_id, edge.resolved_signal_name)
            ] = color

        self._edge_color_map = edge_color_map
        self._log_color_summary()
        self._log_unresolved_conflicts()
        self._populate_wire_connections()
        if preserved_connections:
            self.layout_plan.wire_connections.extend(preserved_connections)

        # Validate relay placement results
        self._validate_relay_coverage()

    def get_wire_color(
        self, source_id: str, sink_id: str, resolved_signal: str
    ) -> Optional[str]:
        """Lookup planned wire color for a specific connection."""

        return self._edge_color_map.get((source_id, sink_id, resolved_signal))

    def iter_circuit_edges(self) -> Sequence[CircuitEdge]:
        """Expose planned circuit edges."""

        return tuple(self._circuit_edges)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expand_merge_edges(
        self,
        edges: Sequence[CircuitEdge],
        wire_merge_junctions: Optional[Dict[str, Any]],
        entities: Dict[str, Any],
    ) -> List[CircuitEdge]:
        if not wire_merge_junctions:
            return list(edges)

        expanded: List[CircuitEdge] = []
        for edge in edges:
            source_id = edge.source_entity_id or ""
            merge_info = wire_merge_junctions.get(source_id)
            if not merge_info:
                expanded.append(edge)
                continue

            for source_ref in merge_info.get("sources", []):
                if not isinstance(source_ref, SignalRef):
                    continue
                actual_source_id = source_ref.source_id
                source_entity_type = None
                placement = entities.get(actual_source_id)
                if placement is not None:
                    source_entity_type = getattr(placement, "entity_type", None)
                    if source_entity_type is None:
                        entity = getattr(placement, "entity", None)
                        if entity is not None:
                            source_entity_type = type(entity).__name__

                expanded.append(
                    CircuitEdge(
                        logical_signal_id=edge.logical_signal_id,
                        resolved_signal_name=edge.resolved_signal_name,
                        source_entity_id=actual_source_id,
                        sink_entity_id=edge.sink_entity_id,
                        source_entity_type=source_entity_type,
                        sink_entity_type=edge.sink_entity_type,
                        sink_role=edge.sink_role,
                    )
                )

        return expanded

    def _add_self_feedback_connections(self) -> None:
        """Add self-feedback connections for arithmetic feedback memories."""
        for entity_id, placement in self.layout_plan.entity_placements.items():
            if placement.properties.get("has_self_feedback"):
                feedback_signal = placement.properties.get("feedback_signal")
                if not feedback_signal:
                    continue

                # Add red self-feedback wire
                feedback_conn = WireConnection(
                    source_entity_id=entity_id,
                    sink_entity_id=entity_id,
                    signal_name=feedback_signal,
                    wire_color="red",
                    source_side="output",
                    sink_side="input",
                )
                self.layout_plan.add_wire_connection(feedback_conn)

                self.diagnostics.info(f"Added self-feedback to {entity_id}")

    def _log_multi_source_conflicts(
        self, edges: Sequence[CircuitEdge], entities: Dict[str, Any]
    ) -> None:
        conflict_map: Dict[str, Dict[str, set[str]]] = {}

        for edge in edges:
            if not edge.source_entity_id:
                continue
            sink_conflicts = conflict_map.setdefault(edge.sink_entity_id, {})
            sink_conflicts.setdefault(edge.resolved_signal_name, set()).add(
                edge.source_entity_id
            )

        for sink_id, conflict_entries in conflict_map.items():
            for resolved_signal, sources in conflict_entries.items():
                if len(sources) <= 1:
                    continue

                source_labels = []
                for source_entity_id in sorted(sources):
                    placement = entities.get(source_entity_id)
                    label = getattr(placement, "entity_id", None)
                    if label:
                        source_labels.append(label)
                    else:
                        source_labels.append(source_entity_id)

                sink_label = sink_id
                placement = entities.get(sink_id)
                if placement is not None:
                    sink_label = getattr(placement, "entity_id", sink_id)

                source_desc = ", ".join(source_labels)

                self.diagnostics.warning(
                    "Detected multiple producers for signal "
                    f"'{resolved_signal}' feeding sink '{sink_label}'; attempting wire coloring to isolate networks (sources: {source_desc})."
                )

    def _log_color_summary(self) -> None:
        if not self._edge_color_map:
            return

        color_counts = Counter(self._edge_color_map.values())
        summaries = []
        for color in WIRE_COLORS:
            count = color_counts.get(color, 0)
            if count:
                summaries.append(f"{count} {color}")
        if summaries:
            self.diagnostics.info(
                "Wire color planner assignments: " + ", ".join(summaries)
            )

    def _log_unresolved_conflicts(self) -> None:
        if self._coloring_success or not self._coloring_conflicts:
            return

        for conflict in self._coloring_conflicts:
            resolved_signal = conflict.nodes[0][1]
            source_desc = ", ".join(sorted({node_id for node_id, _ in conflict.nodes}))
            sink_desc = (
                ", ".join(sorted(conflict.sinks)) if conflict.sinks else "unknown sinks"
            )
            self.diagnostics.warning(
                "Two-color routing could not isolate signal "
                f"'{resolved_signal}' across sinks [{sink_desc}]; falling back to single-channel wiring for involved entities ({source_desc})."
            )
        self._edge_color_map = {}

    def _get_connection_side(self, entity_id: str, is_source: bool) -> Optional[str]:
        """Determine if entity needs 'input'/'output' side specified.

        Args:
            entity_id: Entity to check
            is_source: True if this entity is producing the signal

        Returns:
            'output' for source combinators, 'input' for sink combinators, None otherwise
        """
        placement = self.layout_plan.get_placement(entity_id)
        if not placement:
            return None

        # Combinators have distinct input/output sides
        combinator_types = {
            "arithmetic-combinator",
            "decider-combinator",
        }

        if placement.entity_type in combinator_types:
            return "output" if is_source else "input"

        # Other entities don't need sides specified
        return None

    def _populate_wire_connections(self) -> None:
        for edge in self._circuit_edges:
            if not edge.source_entity_id or not edge.sink_entity_id:
                continue

            color = self._edge_color_map.get(
                (edge.source_entity_id, edge.sink_entity_id, edge.resolved_signal_name)
            )
            if color is None:
                color = WIRE_COLORS[0]

            # Determine connection sides for combinators
            source_side = self._get_connection_side(
                edge.source_entity_id, is_source=True
            )
            sink_side = self._get_connection_side(edge.sink_entity_id, is_source=False)

            self._route_connection_with_relays(edge, color, source_side, sink_side)

    # ------------------------------------------------------------------
    # Relay placement helpers
    # ------------------------------------------------------------------

    def _route_connection_with_relays(
        self,
        edge: CircuitEdge,
        wire_color: str,
        source_side: Optional[str] = None,
        sink_side: Optional[str] = None,
    ) -> None:
        """Route a connection with relays if needed using shared relay infrastructure."""
        source = self.layout_plan.get_placement(edge.source_entity_id)
        sink = self.layout_plan.get_placement(edge.sink_entity_id)

        if source is None or sink is None:
            self.diagnostics.warning(
                "Skipped wiring for '%s' due to missing placement (%s -> %s)."
                % (
                    edge.resolved_signal_name,
                    edge.source_entity_id,
                    edge.sink_entity_id,
                )
            )
            return

        # Get relay path (shared infrastructure)
        relay_path = self.relay_network.route_signal(
            source.position, sink.position, edge.resolved_signal_name, wire_color
        )

        if not relay_path:
            # Direct connection
            self.layout_plan.add_wire_connection(
                WireConnection(
                    source_entity_id=edge.source_entity_id,
                    sink_entity_id=edge.sink_entity_id,
                    signal_name=edge.resolved_signal_name,
                    wire_color=wire_color,
                    source_side=source_side,
                    sink_side=sink_side,
                )
            )
        else:
            # Multi-hop through relays
            current_id = edge.source_entity_id
            current_side = source_side

            for relay_id, relay_color in relay_path:
                self.layout_plan.add_wire_connection(
                    WireConnection(
                        source_entity_id=current_id,
                        sink_entity_id=relay_id,
                        signal_name=edge.resolved_signal_name,
                        wire_color=relay_color,  # Use color assigned by relay network
                        source_side=current_side,
                        sink_side=None,
                    )
                )
                current_id = relay_id
                current_side = None

            # Final hop to sink
            self.layout_plan.add_wire_connection(
                WireConnection(
                    source_entity_id=current_id,
                    sink_entity_id=edge.sink_entity_id,
                    signal_name=edge.resolved_signal_name,
                    wire_color=relay_path[-1][1],  # Use last relay's color
                    source_side=None,
                    sink_side=sink_side,
                )
            )

    # ========================================================================
    # DEPRECATED METHODS - Replaced by RelayNetwork class
    # These methods have been replaced and should not be called anymore
    # ========================================================================

    def _validate_relay_coverage(self) -> None:
        """Validate that all wire connections have adequate relay coverage.

        Logs warnings for any connections that exceed span limits.
        """
        max_span = (
            self.max_wire_span if self.max_wire_span and self.max_wire_span > 0 else 9.0
        )
        span_limit = max(1.0, float(max_span) - 1.8)
        epsilon = 1e-6

        violation_count = 0

        for connection in self.layout_plan.wire_connections:
            source = self.layout_plan.get_placement(connection.source_entity_id)
            sink = self.layout_plan.get_placement(connection.sink_entity_id)

            if not source or not sink:
                continue

            distance = math.dist(source.position, sink.position)

            if distance > span_limit + epsilon:
                violation_count += 1
                if violation_count <= 5:  # Only log first 5 to avoid spam
                    self.diagnostics.warning(
                        f"Wire connection exceeds span limit: {distance:.1f} > {span_limit:.1f} "
                        f"({connection.source_entity_id} -> {connection.sink_entity_id} "
                        f"on {connection.signal_name})"
                    )

        if violation_count > 5:
            self.diagnostics.warning(
                f"Total {violation_count} wire connections exceed span limit "
                f"(showing first 5)"
            )

        relay_count = sum(
            1
            for p in self.layout_plan.entity_placements.values()
            if getattr(p, "role", None) == "wire_relay"
        )

        if relay_count > 0:
            self.diagnostics.info(f"Placed {relay_count} wire relay poles")

    # ------------------------------------------------------------------
    # Convenience helpers for emitters
    # ------------------------------------------------------------------

    def compute_wire_distance(self, source_id: str, sink_id: str) -> Optional[float]:
        source = self.layout_plan.get_placement(source_id)
        sink = self.layout_plan.get_placement(sink_id)
        if not source or not sink:
            return None
        sx, sy = source.position
        tx, ty = sink.position
        return math.dist((sx, sy), (tx, ty))

    def edge_color_map(self) -> Dict[Tuple[str, str, str], str]:
        """Expose raw edge→color assignments."""

        return dict(self._edge_color_map)
