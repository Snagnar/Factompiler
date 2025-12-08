from __future__ import annotations
import math
from dataclasses import dataclass
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple
from dsl_compiler.src.ir.builder import SignalRef
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.common.constants import CompilerConfig, DEFAULT_CONFIG
from .tile_grid import TileGrid
from .layout_plan import LayoutPlan, WireConnection, EntityPlacement
from .signal_analyzer import SignalUsageEntry

from .wire_router import (
    CircuitEdge,
    WIRE_COLORS,
    collect_circuit_edges,
    plan_wire_colors,
)


"""Connection planning for wire routing."""


@dataclass
class RelayNode:
    """A relay pole for routing circuit signals."""

    position: Tuple[float, float]
    entity_id: str
    pole_type: str


class RelayNetwork:
    """Manages shared relay infrastructure for inter-cluster routing."""

    def __init__(
        self,
        tile_grid,
        clusters,
        entity_to_cluster,
        max_span: float,
        layout_plan,
        diagnostics,
        config: CompilerConfig = DEFAULT_CONFIG,
    ):
        self.tile_grid = tile_grid
        self.clusters = clusters
        self.entity_to_cluster = entity_to_cluster
        self.max_span = max_span
        self.layout_plan = layout_plan
        self.diagnostics = diagnostics
        self.config = config
        self.relay_nodes: Dict[Tuple[int, int], RelayNode] = {}
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

        return node

    def find_relay_near(
        self, position: Tuple[float, float], max_distance: float
    ) -> Optional[RelayNode]:
        """Find existing relay pole near position, or None."""
        for node in self.relay_nodes.values():
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
    ) -> Optional[
        List[Tuple[str, str]]
    ]:  # Returns list of (entity_id, wire_color) or None on failure
        """
        Find or create relay path from source to sink.
        Returns list of (entity_id, wire_color) pairs representing the routing path.
        Returns empty list if no relays needed, None if routing failed.
        """
        distance = math.dist(source_pos, sink_pos)

        span_limit = max(
            1.0, float(self.max_span) - self.config.wire_span_safety_margin
        )

        if distance <= span_limit * 0.95:
            return []  # No relays needed

        safe_interval = span_limit * 0.8
        num_relays = max(1, int(math.ceil(distance / safe_interval)) - 1)

        path = []

        for i in range(1, num_relays + 1):
            ratio = i / (num_relays + 1)
            relay_x = source_pos[0] + (sink_pos[0] - source_pos[0]) * ratio
            relay_y = source_pos[1] + (sink_pos[1] - source_pos[1]) * ratio
            ideal_pos = (relay_x, relay_y)

            # This prevents accumulating distance errors when reusing relays
            relay_node = self.find_relay_near(ideal_pos, max_distance=0.7)

            if relay_node is None:
                tile_pos = (int(round(relay_x)), int(round(relay_y)))
                if self.tile_grid.reserve_exact(tile_pos, footprint=(1, 1)):
                    reserved_pos = tile_pos
                else:
                    reserved_pos = None
                    # Search in expanding rings up to 5 tiles away
                    # Prioritize positions closer to the ideal position
                    for radius in range(1, 6):
                        if reserved_pos:
                            break
                        for dx in range(-radius, radius + 1):
                            if reserved_pos:
                                break
                            for dy in range(-radius, radius + 1):
                                # Only check tiles at the current radius (ring edges)
                                if abs(dx) != radius and abs(dy) != radius:
                                    continue
                                nearby_pos = (tile_pos[0] + dx, tile_pos[1] + dy)
                                if self.tile_grid.reserve_exact(
                                    nearby_pos, footprint=(1, 1)
                                ):
                                    reserved_pos = nearby_pos
                                    break

                if reserved_pos:
                    self._relay_counter += 1
                    relay_id = f"__relay_{self._relay_counter}"
                    # ✅ FIX: Ensure reserved_pos is integer tile position
                    tile_x = int(round(reserved_pos[0]))
                    tile_y = int(round(reserved_pos[1]))
                    center_pos = (tile_x + 0.5, tile_y + 0.5)
                    relay_node = self.add_relay_node(
                        center_pos, relay_id, "medium-electric-pole"
                    )

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
                path.append((relay_node.entity_id, wire_color))
            else:
                self.diagnostics.warning(
                    f"Failed to place relay for {signal_name} at {ideal_pos} - no space available"
                )
                return None  # Return None to signal failure - incomplete relay chain is broken

        return path

    def can_route_signal(
        self,
        source_pos: Tuple[float, float],
        sink_pos: Tuple[float, float],
    ) -> bool:
        """Check if a signal can be routed between two positions.

        Returns True if:
        - The distance is within wire span limit (no relays needed), or
        - All necessary relay positions have available space

        This is a dry-run check that doesn't create any relays.
        """
        distance = math.dist(source_pos, sink_pos)

        span_limit = max(
            1.0, float(self.max_span) - self.config.wire_span_safety_margin
        )

        if distance <= span_limit * 0.95:
            return True  # No relays needed

        safe_interval = span_limit * 0.8
        num_relays = max(1, int(math.ceil(distance / safe_interval)) - 1)

        for i in range(1, num_relays + 1):
            ratio = i / (num_relays + 1)
            relay_x = source_pos[0] + (sink_pos[0] - source_pos[0]) * ratio
            relay_y = source_pos[1] + (sink_pos[1] - source_pos[1]) * ratio
            ideal_pos = (relay_x, relay_y)

            # Check if there's an existing relay nearby
            relay_node = self.find_relay_near(ideal_pos, max_distance=0.7)
            if relay_node is not None:
                continue  # Can reuse existing relay

            # Check if we could place a new relay
            tile_pos = (int(round(relay_x)), int(round(relay_y)))
            if self.tile_grid.is_available(tile_pos, footprint=(1, 1)):
                continue  # Can place new relay here

            # Check offset positions
            found_space = False
            for offset in [
                (0, 1),
                (1, 0),
                (0, -1),
                (-1, 0),
                (1, 1),
                (-1, 1),
                (1, -1),
                (-1, -1),
            ]:
                nearby_pos = (tile_pos[0] + offset[0], tile_pos[1] + offset[1])
                if self.tile_grid.is_available(nearby_pos, footprint=(1, 1)):
                    found_space = True
                    break

            if not found_space:
                return False  # Can't place relay at this position

        return True


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
        tile_grid: TileGrid,
        max_wire_span: float = 9.0,
        power_pole_type: Optional[str] = None,
        config: CompilerConfig = DEFAULT_CONFIG,
        use_mst_optimization: bool = True,
    ) -> None:
        self.layout_plan = layout_plan
        self.signal_usage = signal_usage
        self.diagnostics = diagnostics
        self.max_wire_span = max_wire_span
        self.tile_grid = tile_grid
        self.power_pole_type = power_pole_type
        self.use_mst_optimization = use_mst_optimization
        self.config = config

        self._circuit_edges: List[CircuitEdge] = []
        self._node_color_assignments: Dict[Tuple[str, str], str] = {}
        self._edge_color_map: Dict[Tuple[str, str, str], str] = {}
        self._coloring_conflicts = []
        self._coloring_success = True
        self._relay_counter = 0

        self._memory_modules: Dict[str, Any] = {}

        self._edge_wire_colors: Dict[Tuple[str, str, str], str] = {}

        self.relay_network = RelayNetwork(
            self.tile_grid,
            None,  # No clusters
            {},  # No entity_to_cluster mapping
            self.max_wire_span,
            self.layout_plan,
            self.diagnostics,
            self.config,
        )

    def plan_connections(
        self,
        signal_graph: Any,
        entities: Dict[str, Any],
        wire_merge_junctions: Optional[Dict[str, Any]] = None,
        locked_colors: Optional[Dict[Tuple[str, str], str]] = None,
    ) -> None:
        """Compute all wire connections with color assignments."""
        self._register_power_poles_as_relays()

        self._add_self_feedback_connections()

        preserved_connections = list(self.layout_plan.wire_connections)
        self.layout_plan.wire_connections.clear()
        self._circuit_edges = []
        self._node_color_assignments = {}
        self._edge_color_map = {}
        self._coloring_conflicts = []
        self._coloring_success = True
        self._relay_counter = 0

        base_edges = collect_circuit_edges(signal_graph, self.signal_usage, entities)
        expanded_edges = self._expand_merge_edges(
            base_edges, wire_merge_junctions, entities
        )

        wire_merge_sinks = set()
        if wire_merge_junctions:
            for _, merge_info in wire_merge_junctions.items():
                for edge in expanded_edges:
                    if edge.source_entity_id in [
                        ref.source_id
                        for ref in merge_info.get("inputs", [])
                        if hasattr(ref, "source_id")
                    ]:
                        wire_merge_sinks.add(edge.sink_entity_id)

        filtered_edges = []
        for edge in expanded_edges:
            if self._is_internal_feedback_signal(edge.resolved_signal_name):
                self.diagnostics.info(
                    f"Filtered out internal feedback signal edge: "
                    f"{edge.source_entity_id} -> {edge.sink_entity_id} ({edge.resolved_signal_name})"
                )
                continue
            filtered_edges.append(edge)

        self.diagnostics.info(
            f"Filtered {len(expanded_edges) - len(filtered_edges)} internal feedback edges "
            f"({len(filtered_edges)} edges remaining for wire planning)"
        )
        expanded_edges = filtered_edges
        self._circuit_edges = expanded_edges

        self._log_multi_source_conflicts(expanded_edges, entities)

        # Memory feedback edges always use RED wire and should not participate
        # in the bipartite graph coloring algorithm
        non_feedback_edges = [
            edge
            for edge in expanded_edges
            if not self._is_memory_feedback_edge(
                edge.source_entity_id, edge.sink_entity_id, edge.resolved_signal_name
            )
        ]

        if len(expanded_edges) != len(non_feedback_edges):
            self.diagnostics.info(
                f"Filtered {len(expanded_edges) - len(non_feedback_edges)} memory feedback edges from wire coloring "
                f"({len(non_feedback_edges)} edges remaining)"
            )

        # Wire merge inputs intentionally combine the same signal on the same wire
        non_merge_edges = [
            edge
            for edge in non_feedback_edges
            if edge.sink_entity_id not in wire_merge_sinks
        ]

        if len(non_feedback_edges) != len(non_merge_edges):
            self.diagnostics.info(
                f"Filtered {len(non_feedback_edges) - len(non_merge_edges)} wire merge edges from wire coloring "
                f"({len(non_merge_edges)} edges remaining for conflict resolution)"
            )

        coloring_result = plan_wire_colors(non_merge_edges, locked_colors)
        self._node_color_assignments = coloring_result.assignments
        self._coloring_conflicts = coloring_result.conflicts
        self._coloring_success = coloring_result.is_bipartite

        edge_color_map: Dict[Tuple[str, str, str], str] = {}
        for edge in non_merge_edges:
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

        self._validate_relay_coverage()

    def get_wire_color_for_edge(
        self, source_entity_id: str, sink_entity_id: str, signal_name: str
    ) -> str:
        """Get the wire color for a specific edge.

        Args:
            source_entity_id: The entity producing the signal
            sink_entity_id: The entity consuming the signal
            signal_name: The RESOLVED Factorio signal name (e.g., "signal-A")

        Returns:
            Wire color "red" or "green", defaults to "red" if not found
        """
        edge_key = (source_entity_id, sink_entity_id, signal_name)
        return self._edge_wire_colors.get(edge_key, "red")

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
            if edge.sink_entity_id in wire_merge_junctions:
                continue

            source_id = edge.source_entity_id or ""
            merge_info = wire_merge_junctions.get(source_id)
            if not merge_info:
                expanded.append(edge)
                continue

            for source_ref in merge_info.get("inputs", []):
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

    def _register_power_poles_as_relays(self) -> None:
        """Register existing power pole entities as available relays for circuit routing.

        This allows the relay network to reuse power poles that were placed during
        layout optimization, reducing the total number of poles needed.
        """
        from .power_planner import POWER_POLE_CONFIG

        power_pole_count = 0

        for entity_id, placement in self.layout_plan.entity_placements.items():
            if not placement.properties.get("is_power_pole"):
                continue

            if placement.position is None:
                continue

            pole_type = placement.properties.get("pole_type", "medium")
            config = POWER_POLE_CONFIG.get(pole_type.lower())
            if not config:
                continue

            prototype = str(config["prototype"])

            self.relay_network.add_relay_node(placement.position, entity_id, prototype)
            power_pole_count += 1

        if power_pole_count > 0:
            self.diagnostics.info(
                f"Registered {power_pole_count} existing power poles as available relays"
            )

    def _add_self_feedback_connections(self) -> None:
        """Add self-feedback connections for arithmetic feedback memories."""
        for entity_id, placement in self.layout_plan.entity_placements.items():
            if placement.properties.get("has_self_feedback"):
                feedback_signal = placement.properties.get("feedback_signal")
                if not feedback_signal:
                    continue

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

        combinator_types = {
            "arithmetic-combinator",
            "decider-combinator",
        }

        if placement.entity_type in combinator_types:
            return "output" if is_source else "input"

        return None

    def _is_memory_feedback_edge(
        self, source_id: str, sink_id: str, signal_name: str
    ) -> bool:
        """Check if an edge is a memory SR latch feedback connection.

        Feedback edges (write_gate ↔ hold_gate) must use GREEN wire while
        data/enable connections use RED wire to prevent signal interference.

        NOTE: With the new approach, feedback edges are created directly and
        internal feedback signal IDs are filtered out before wire planning.
        This function is now mainly for documentation and edge cases.
        """
        from .memory_builder import MemoryModule

        if not self._memory_modules:
            return False

        if self._is_internal_feedback_signal(signal_name):
            return True

        for module in self._memory_modules.values():
            if not isinstance(module, MemoryModule):
                continue

            if module.optimization is not None:
                continue

            if not module.write_gate or not module.hold_gate:
                continue

            write_id = module.write_gate.ir_node_id
            hold_id = module.hold_gate.ir_node_id

            if (
                source_id == write_id
                and sink_id == hold_id
                and signal_name == module.signal_type
            ):
                return True

            if (
                source_id == hold_id
                and sink_id == write_id
                and signal_name == module.signal_type
            ):
                return True

        return False

    def _is_internal_feedback_signal(self, signal_name: str) -> bool:
        """Check if a signal name is an internal feedback identifier.

        Internal feedback signals are used in signal_graph for layout proximity
        but should not be wired (direct wire connections are created instead).
        """
        from .memory_builder import MemoryModule

        if not signal_name.startswith("__feedback_"):
            return False

        for module in self._memory_modules.values():
            if not isinstance(module, MemoryModule):
                continue

            if hasattr(module, "_feedback_signal_ids"):
                if signal_name in module._feedback_signal_ids:
                    return True

        if signal_name.startswith("__feedback_"):
            self.diagnostics.warning(
                f"Found feedback-like signal '{signal_name}' but no matching module"
            )
            return True

        return False

    def _populate_wire_connections(self) -> None:
        """Create wire connections, using MST optimization for safe star patterns.

        Uses per-source analysis: for each source entity, we check if its fanout
        to multiple sinks can be optimized with MST. Bidirectional edges (feedback
        loops) are always routed directly.
        """

        # Step 1: Group edges by (signal_name, wire_color)
        signal_groups: Dict[Tuple[str, str], List[Tuple[CircuitEdge, str]]] = {}

        for edge in self._circuit_edges:
            if not edge.source_entity_id or not edge.sink_entity_id:
                continue

            # Determine wire color
            if self._is_memory_feedback_edge(
                edge.source_entity_id, edge.sink_entity_id, edge.resolved_signal_name
            ):
                color = "red"
            else:
                color = self._edge_color_map.get(
                    (
                        edge.source_entity_id,
                        edge.sink_entity_id,
                        edge.resolved_signal_name,
                    )
                )
                if color is None:
                    color = WIRE_COLORS[0]

            group_key = (edge.resolved_signal_name, color)
            if group_key not in signal_groups:
                signal_groups[group_key] = []
            signal_groups[group_key].append((edge, color))

        # Step 2: Process each signal group with per-source analysis
        mst_star_count = 0
        direct_routed_count = 0

        # Sort for deterministic iteration order
        for (signal_name, wire_color), edge_color_pairs in sorted(
            signal_groups.items()
        ):
            edges = [pair[0] for pair in edge_color_pairs]

            # Debug: log signal group processing
            if "arith_15" in str([e.source_entity_id for e in edges]):
                self.diagnostics.info(
                    f"Processing signal group ({signal_name}, {wire_color}) with {len(edges)} edges, "
                    f"sources: {set(e.source_entity_id for e in edges)}"
                )

            # Find all bidirectional pairs in this signal group
            bidirectional_pairs = self._find_bidirectional_pairs(edges)

            # Group edges by source
            by_source: Dict[str, List[CircuitEdge]] = {}
            for edge in edges:
                if edge.source_entity_id not in by_source:
                    by_source[edge.source_entity_id] = []
                by_source[edge.source_entity_id].append(edge)

            # Process each source's fanout independently
            # Sort for deterministic iteration order
            for source_id, source_edges in sorted(by_source.items()):
                sink_ids = [e.sink_entity_id for e in source_edges]

                # Separate sinks into bidirectional (with this source) and safe
                bidir_sinks = set()
                for sink in sink_ids:
                    if (source_id, sink) in bidirectional_pairs:
                        bidir_sinks.add(sink)

                safe_sinks = [s for s in sink_ids if s not in bidir_sinks]

                # Apply MST to safe sinks if we have 2+ of them AND MST is enabled
                mst_succeeded = False
                if self.use_mst_optimization and len(safe_sinks) >= 2:
                    self.diagnostics.info(
                        f"Applying MST optimization for source '{source_id}' "
                        f"to {len(safe_sinks)} safe sinks for signal '{signal_name}'"
                    )
                    mst_succeeded = self._apply_mst_to_source_fanout(
                        source_id, safe_sinks, signal_name, wire_color
                    )
                    if mst_succeeded:
                        mst_star_count += 1
                    else:
                        self.diagnostics.info(
                            f"MST routing failed for '{signal_name}', falling back to direct routing"
                        )

                if not mst_succeeded:
                    # Route safe sinks directly (MST disabled, failed, or 0/1 sink)
                    for sink in safe_sinks:
                        edge = next(e for e in source_edges if e.sink_entity_id == sink)
                        self._route_edge_directly(edge, wire_color)
                        direct_routed_count += 1

                # Always route bidirectional sinks directly
                # Sort for deterministic iteration order
                for sink in sorted(bidir_sinks):
                    edge = next(e for e in source_edges if e.sink_entity_id == sink)
                    self._route_edge_directly(edge, wire_color)
                    direct_routed_count += 1

        if mst_star_count > 0:
            self.diagnostics.info(
                f"MST optimization: {mst_star_count} source fanouts optimized, "
                f"{direct_routed_count} direct edges"
            )

    def _find_bidirectional_pairs(self, edges: List[CircuitEdge]) -> set:
        """Find all bidirectional edge pairs (A→B and B→A both exist).

        Returns:
            Set of (source, sink) tuples that are part of bidirectional pairs.
            Both directions are included: if A↔B, returns {(A,B), (B,A)}.
        """
        pairs = set()
        edge_set = {(e.source_entity_id, e.sink_entity_id) for e in edges}

        for edge in edges:
            reverse = (edge.sink_entity_id, edge.source_entity_id)
            if reverse in edge_set:
                pairs.add((edge.source_entity_id, edge.sink_entity_id))
                pairs.add(reverse)

        return pairs

    def _route_edge_directly(self, edge: CircuitEdge, wire_color: str) -> None:
        """Route a single edge directly (no MST optimization)."""
        edge_key = (
            edge.source_entity_id,
            edge.sink_entity_id,
            edge.resolved_signal_name,
        )
        self._edge_wire_colors[edge_key] = wire_color

        source_side = self._get_connection_side(edge.source_entity_id, is_source=True)
        sink_side = self._get_connection_side(edge.sink_entity_id, is_source=False)

        self._route_connection_with_relays(edge, wire_color, source_side, sink_side)

    def _apply_mst_to_source_fanout(
        self, source_id: str, sink_ids: List[str], signal_name: str, wire_color: str
    ) -> bool:
        """Apply MST optimization to a source's fanout to multiple sinks.

        Args:
            source_id: The source entity ID
            sink_ids: List of sink entity IDs (must be >= 2)
            signal_name: The signal being routed
            wire_color: The wire color to use

        Returns:
            True if all MST edges were routed successfully, False if any failed.
        """
        # Build MST over source + all sinks
        # Use sorted to ensure deterministic order
        all_entities = [source_id] + sorted(set(sink_ids))
        mst_edges = self._build_minimum_spanning_tree(all_entities)

        # Verify source is connected in MST
        source_in_mst = any(source_id in edge for edge in mst_edges)
        if not source_in_mst and mst_edges:
            self.diagnostics.warning(
                f"MST bug: source '{source_id}' not connected in MST edges: {mst_edges}"
            )
            return False

        # Pre-validate ALL MST edges are short enough to NOT need relays
        # If any edge needs relays, skip MST entirely to avoid relay conflicts
        # between different signal groups
        span_limit = self.relay_network.max_span - self.config.wire_span_safety_margin
        for ent_a, ent_b in mst_edges:
            placement_a = self.layout_plan.get_placement(ent_a)
            placement_b = self.layout_plan.get_placement(ent_b)

            if not placement_a or not placement_b:
                self.diagnostics.info(
                    f"MST pre-check failed for {signal_name}: missing placement for {ent_a} or {ent_b}"
                )
                return False

            if not placement_a.position or not placement_b.position:
                self.diagnostics.info(
                    f"MST pre-check failed for {signal_name}: missing position for {ent_a} or {ent_b}"
                )
                return False

            distance = math.dist(placement_a.position, placement_b.position)
            if distance > span_limit * 0.95:
                # This edge would need relays - skip MST to avoid relay conflicts
                self.diagnostics.info(
                    f"MST skipped for {signal_name}: edge {ent_a} ↔ {ent_b} "
                    f"distance {distance:.1f} exceeds span limit {span_limit:.1f}"
                )
                return False

        self.diagnostics.info(
            f"MST for {signal_name}: source={source_id} → {len(sink_ids)} sinks "
            f"({len(sink_ids)} edges → {len(mst_edges)} MST edges)"
        )

        # IMPORTANT: Also register the original logical edges (source → each sink)
        # so that get_wire_color_for_edge() can find them for operand wire injection.
        # The MST edges are physical routing paths, but operand lookup needs
        # the logical source→sink edges.
        for sink_id in sink_ids:
            self._edge_wire_colors[(source_id, sink_id, signal_name)] = wire_color
            # Also register reverse for bidirectional lookups
            self._edge_wire_colors[(sink_id, source_id, signal_name)] = wire_color

        all_succeeded = True
        for ent_a, ent_b in mst_edges:
            # Determine sides: source uses OUTPUT, sinks use INPUT
            side_a = self._get_connection_side(ent_a, is_source=(ent_a == source_id))
            side_b = self._get_connection_side(ent_b, is_source=(ent_b == source_id))

            # Store wire color for both directions (MST edges are undirected)
            self._edge_wire_colors[(ent_a, ent_b, signal_name)] = wire_color
            self._edge_wire_colors[(ent_b, ent_a, signal_name)] = wire_color

            # Route the connection
            if not self._route_mst_edge(
                ent_a, ent_b, signal_name, wire_color, side_a, side_b
            ):
                all_succeeded = False

        return all_succeeded

    def _build_minimum_spanning_tree(
        self, entity_ids: List[str]
    ) -> List[Tuple[str, str]]:
        """Build minimum spanning tree over entities using Prim's algorithm.

        Args:
            entity_ids: List of entity IDs to connect

        Returns:
            List of (entity_a, entity_b) edges forming the MST
        """
        if len(entity_ids) <= 1:
            return []

        # Collect positions for entities that have valid placements
        positions: Dict[str, Tuple[float, float]] = {}
        for entity_id in entity_ids:
            placement = self.layout_plan.get_placement(entity_id)
            if placement and placement.position:
                positions[entity_id] = placement.position

        valid_entities = [e for e in entity_ids if e in positions]
        if len(valid_entities) <= 1:
            return []

        # Prim's algorithm: greedily grow MST from first entity (should be source)
        in_tree = {valid_entities[0]}
        mst_edges: List[Tuple[str, str]] = []

        while len(in_tree) < len(valid_entities):
            best_edge: Optional[Tuple[str, str]] = None
            best_distance = float("inf")

            # Find shortest edge from tree to non-tree vertex
            # Sort in_tree for deterministic iteration order
            for tree_entity in sorted(in_tree):
                tree_pos = positions[tree_entity]
                for candidate in valid_entities:
                    if candidate in in_tree:
                        continue
                    distance = math.dist(tree_pos, positions[candidate])
                    # Use tuple comparison for tie-breaking to ensure determinism
                    if distance < best_distance or (
                        distance == best_distance
                        and (tree_entity, candidate) < (best_edge or ("", ""))
                    ):
                        best_distance = distance
                        best_edge = (tree_entity, candidate)

            if best_edge is None:
                break

            mst_edges.append(best_edge)
            in_tree.add(best_edge[1])

        return mst_edges

    def _route_mst_edge(
        self,
        entity_a: str,
        entity_b: str,
        signal_name: str,
        wire_color: str,
        side_a: Optional[str],
        side_b: Optional[str],
    ) -> bool:
        """Create wire connection for MST edge, with relay poles if needed.

        Returns:
            True if routing succeeded, False if it failed.
        """

        placement_a = self.layout_plan.get_placement(entity_a)
        placement_b = self.layout_plan.get_placement(entity_b)

        if not placement_a or not placement_b:
            self.diagnostics.warning(
                f"Skipped MST edge for '{signal_name}': missing placement "
                f"({entity_a} or {entity_b})"
            )
            return False

        if not placement_a.position or not placement_b.position:
            self.diagnostics.warning(
                f"Skipped MST edge for '{signal_name}': missing position "
                f"({entity_a} or {entity_b})"
            )
            return False

        # Use relay network for long edges
        relay_path = self.relay_network.route_signal(
            placement_a.position, placement_b.position, signal_name, wire_color
        )

        if relay_path is None:
            # Relay routing failed - connection cannot be established
            # This is a critical error - the circuit will be broken without this connection
            self.diagnostics.error(
                f"MST edge for '{signal_name}' cannot be routed: "
                f"relay placement failed between {entity_a} and {entity_b}. "
                f"The layout may be too spread out for the available wire span."
            )
            return False

        if len(relay_path) == 0:
            # Direct connection - no relays needed
            self.layout_plan.add_wire_connection(
                WireConnection(
                    source_entity_id=entity_a,
                    sink_entity_id=entity_b,
                    signal_name=signal_name,
                    wire_color=wire_color,
                    source_side=side_a,
                    sink_side=side_b,
                )
            )
        else:
            # Chain through relay poles
            current_id = entity_a
            current_side = side_a

            for relay_id, relay_color in relay_path:
                self.layout_plan.add_wire_connection(
                    WireConnection(
                        source_entity_id=current_id,
                        sink_entity_id=relay_id,
                        signal_name=signal_name,
                        wire_color=relay_color,
                        source_side=current_side,
                        sink_side=None,
                    )
                )
                current_id = relay_id
                current_side = None

            # Final connection to entity_b
            self.layout_plan.add_wire_connection(
                WireConnection(
                    source_entity_id=current_id,
                    sink_entity_id=entity_b,
                    signal_name=signal_name,
                    wire_color=relay_path[-1][1],
                    source_side=None,
                    sink_side=side_b,
                )
            )

        return True

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

        relay_path = self.relay_network.route_signal(
            source.position, sink.position, edge.resolved_signal_name, wire_color
        )

        if relay_path is None:
            # Relay routing failed - connection cannot be established
            # This is a critical error - the circuit will be broken without this connection
            self.diagnostics.error(
                f"Connection for '{edge.resolved_signal_name}' cannot be routed: "
                f"relay placement failed between {edge.source_entity_id} and {edge.sink_entity_id}. "
                f"The layout may be too spread out for the available wire span."
            )
            return

        if len(relay_path) == 0:
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

    def _validate_relay_coverage(self) -> None:
        """Validate that all wire connections have adequate relay coverage.

        Logs warnings for any connections that exceed span limits.
        """
        max_span = (
            self.max_wire_span if self.max_wire_span and self.max_wire_span > 0 else 9.0
        )
        span_limit = max(1.0, float(max_span) - self.config.wire_span_safety_margin)
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

    def edge_color_map(self) -> Dict[Tuple[str, str, str], str]:
        """Expose raw edge→color assignments."""

        return dict(self._edge_color_map)
