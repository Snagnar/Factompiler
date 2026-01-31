from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from dsl_compiler.src.common.constants import DEFAULT_CONFIG, CompilerConfig
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.common.entity_data import is_dual_circuit_connectable
from dsl_compiler.src.ir.builder import BundleRef, SignalRef

from .layout_plan import LayoutPlan, WireConnection
from .signal_analyzer import SignalUsageEntry
from .tile_grid import TileGrid
from .wire_router import (
    WIRE_COLORS,
    CircuitEdge,
    ConflictEdge,
    collect_circuit_edges,
    plan_wire_colors,
)

"""Connection planning for wire routing."""


@dataclass
class RelayNode:
    """A relay pole for routing circuit signals.

    Tracks which circuit networks are using each wire color to prevent signal mixing.
    In Factorio, all entities connected by the same color wire form a circuit network.
    If two signals from DIFFERENT networks share a relay on the same color, they mix.

    We track network IDs (not signal names) because:
    - Signals going to the SAME sink are on the SAME network (safe to share relay)
    - Signals going to DIFFERENT sinks are on DIFFERENT networks (must not share relay)
    """

    position: tuple[float, float]
    entity_id: str
    pole_type: str
    networks_red: set[int] = field(default_factory=set)  # Network IDs on red wire
    networks_green: set[int] = field(default_factory=set)  # Network IDs on green wire

    def can_route_network(self, network_id: int, wire_color: str) -> bool:
        """Check if this relay can route the given network on the given color.

        Returns True if:
        - No networks are currently using this color (can start fresh), OR
        - The same network is already on this color (can extend/reuse)
        """
        networks = self.networks_red if wire_color == "red" else self.networks_green
        return len(networks) == 0 or network_id in networks

    def add_network(self, network_id: int, wire_color: str) -> None:
        """Mark a network as using this relay on the given color."""
        if wire_color == "red":
            self.networks_red.add(network_id)
        else:
            self.networks_green.add(network_id)


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
        relay_search_radius: float = 5.0,
    ):
        self.tile_grid = tile_grid
        self.clusters = clusters
        self.entity_to_cluster = entity_to_cluster
        self.max_span = max_span
        self.layout_plan = layout_plan
        self.diagnostics = diagnostics
        self.config = config
        self.relay_search_radius = relay_search_radius
        self.relay_nodes: dict[tuple[int, int], RelayNode] = {}
        self._relay_counter = 0

    @property
    def span_limit(self) -> float:
        """Maximum wire span limit (firm 9.0 tiles for Factorio)."""
        return float(self.max_span)

    def add_relay_node(
        self, position: tuple[float, float], entity_id: str, pole_type: str
    ) -> RelayNode:
        """Add a pole to the relay network."""
        # Use floor to get consistent tile positions
        # (center positions like 31.5 should map to tile 31, not round to 32)
        tile_pos = (int(math.floor(position[0])), int(math.floor(position[1])))

        if tile_pos in self.relay_nodes:
            return self.relay_nodes[tile_pos]

        node = RelayNode(position, entity_id, pole_type)
        self.relay_nodes[tile_pos] = node

        return node

    def find_relay_near(
        self, position: tuple[float, float], max_distance: float
    ) -> RelayNode | None:
        """Find the closest existing relay pole within max_distance, or None."""
        best_node = None
        best_dist = float("inf")
        for node in self.relay_nodes.values():
            dist = math.dist(position, node.position)
            if dist <= max_distance and dist < best_dist:
                best_node = node
                best_dist = dist
        return best_node

    def route_signal(
        self,
        source_pos: tuple[float, float],
        sink_pos: tuple[float, float],
        signal_name: str,
        wire_color: str,
        network_id: int = 0,
    ) -> list[tuple[str, str]] | None:  # Returns list of (entity_id, wire_color) or None on failure
        """
        Find or create relay path from source to sink.
        Returns list of (entity_id, wire_color) pairs representing the routing path.
        Returns empty list if no relays needed, None if routing failed.

        Uses a two-phase approach:
        1. Try to find a path through existing relays using A*
        2. If no path exists, plan relay positions along the source-sink line
           and create them sequentially, ensuring each is reachable from the previous

        Args:
            source_pos: Position of source entity
            sink_pos: Position of sink entity
            signal_name: Name of the signal being routed (for logging)
            wire_color: Color of the wire ("red" or "green")
            network_id: ID of the circuit network this signal belongs to.
                        Signals with the same network_id can share relays.
        """
        distance = math.dist(source_pos, sink_pos)

        if distance <= self.span_limit:
            return []  # No relays needed

        # Phase 1: Try to find a path through existing relays
        existing_path = self._find_path_through_existing_relays(
            source_pos, sink_pos, self.span_limit, wire_color, network_id
        )
        if existing_path is not None:
            # Found complete path! Register the network usage on each relay
            for relay_id, relay_color in existing_path:
                node = self._get_relay_node_by_id(relay_id)
                if node:
                    node.add_network(network_id, relay_color)
                    self.diagnostics.info(
                        f"Relay {relay_id} at {node.position} now carries network {network_id} ({signal_name}) on {relay_color}"
                    )
            return existing_path

        # Phase 2: Plan and create relays along the source-sink line
        return self._plan_and_create_relay_path(
            source_pos, sink_pos, self.span_limit, signal_name, wire_color, network_id
        )

    def _find_path_through_existing_relays(
        self,
        source_pos: tuple[float, float],
        sink_pos: tuple[float, float],
        span_limit: float,
        wire_color: str,
        network_id: int,
    ) -> list[tuple[str, str]] | None:
        """Try to find a path through existing relays using A*.

        Returns:
            List of (relay_id, wire_color) if path found, None otherwise.
        """
        import heapq

        # Build graph of all reachable nodes from source
        nodes: dict[str, tuple[float, float]] = {}
        nodes["__source__"] = source_pos
        nodes["__sink__"] = sink_pos
        for node in self.relay_nodes.values():
            # Only consider relays that can route this network
            if node.can_route_network(network_id, wire_color):
                nodes[node.entity_id] = node.position

        # A* search from source
        open_set: list[tuple[float, float, str, list]] = [
            (0.0, 0.0, "__source__", [])
        ]  # (f_score, g_score, node_id, path)
        visited = set()

        while open_set:
            _, g, current, path = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            current_pos = nodes[current]

            # Check if we can reach sink directly
            if math.dist(current_pos, sink_pos) <= span_limit:
                return [(relay_id, wire_color) for relay_id in path]

            # Explore neighbors
            for node_id, node_pos in nodes.items():
                if node_id == "__source__" or node_id in visited:
                    continue

                dist = math.dist(current_pos, node_pos)
                if dist > span_limit:
                    continue  # Too far

                new_g = g + dist
                new_h = math.dist(node_pos, sink_pos)
                new_f = new_g + new_h

                if node_id == "__sink__":
                    return [(relay_id, wire_color) for relay_id in path]
                else:
                    new_path = path + [node_id]
                    heapq.heappush(open_set, (new_f, new_g, node_id, new_path))

        return None  # No path found through existing relays

    def _plan_and_create_relay_path(
        self,
        source_pos: tuple[float, float],
        sink_pos: tuple[float, float],
        span_limit: float,
        signal_name: str,
        wire_color: str,
        network_id: int,
    ) -> list[tuple[str, str]] | None:
        """Plan relay positions along source-sink line and create them.

        Uses a greedy approach: starting from source, place relays at regular
        intervals along the line to sink, adjusting positions when blocked.
        Each relay must be reachable from the previous point.
        """
        distance = math.dist(source_pos, sink_pos)

        # Calculate step size - use 80% of span for safety margin
        step_size = span_limit * 0.8

        # Calculate number of relays needed
        num_relays = max(1, int(math.ceil(distance / step_size)) - 1)

        path: list[tuple[str, str]] = []
        current_pos = source_pos

        for i in range(num_relays + 5):  # +5 for safety margin if relays get placed off-path
            # Calculate remaining distance and direction FROM CURRENT POSITION
            remaining_dist = math.dist(current_pos, sink_pos)
            if remaining_dist <= span_limit:
                break  # Can already reach sink, no more relays needed

            # Recalculate direction vector from current position to sink
            # This is crucial when relays get placed off the ideal path
            dir_x = (sink_pos[0] - current_pos[0]) / remaining_dist
            dir_y = (sink_pos[1] - current_pos[1]) / remaining_dist

            # Calculate ideal position along the line
            actual_step = min(step_size, remaining_dist * 0.6)  # Don't overshoot
            ideal_x = current_pos[0] + dir_x * actual_step
            ideal_y = current_pos[1] + dir_y * actual_step
            ideal_pos = (ideal_x, ideal_y)

            # Try to find or create a relay near this position
            relay_node = self._find_or_create_relay_near(
                ideal_pos,
                current_pos,
                sink_pos,
                span_limit,
                signal_name,
                wire_color,
                network_id,
            )

            if relay_node is None:
                self.diagnostics.info(
                    f"Failed to create relay {i + 1} for {signal_name} "
                    f"at ideal position {ideal_pos}"
                )
                return None

            # Verify the relay is reachable from current position
            relay_dist = math.dist(current_pos, relay_node.position)
            if relay_dist > span_limit:
                self.diagnostics.info(
                    f"Relay {relay_node.entity_id} at {relay_node.position} is too far "
                    f"({relay_dist:.1f}) from current position {current_pos}"
                )
                return None

            # Add relay to path and update current position
            relay_node.add_network(network_id, wire_color)
            path.append((relay_node.entity_id, wire_color))
            self.diagnostics.info(
                f"Relay {relay_node.entity_id} at {relay_node.position} now carries "
                f"network {network_id} ({signal_name}) on {wire_color}"
            )
            current_pos = relay_node.position

        # Verify we can reach sink from the last relay
        final_dist = math.dist(current_pos, sink_pos)
        if final_dist > span_limit:
            self.diagnostics.info(
                f"Cannot reach sink from last relay, distance {final_dist:.1f} > {span_limit:.1f}"
            )
            return None

        return path

    def _find_or_create_relay_near(
        self,
        ideal_pos: tuple[float, float],
        source_pos: tuple[float, float],
        sink_pos: tuple[float, float],
        span_limit: float,
        signal_name: str,
        wire_color: str,
        network_id: int,
    ) -> RelayNode | None:
        """Find an existing relay or create a new one near the ideal position.

        When searching for positions, prioritizes positions that:
        1. Are reachable from source_pos (within span_limit)
        2. Are closer to sink_pos (makes progress toward destination)

        Args:
            ideal_pos: The ideal position for the relay
            source_pos: The current position we're routing from
            sink_pos: The final destination
            span_limit: Maximum wire span
            signal_name: For logging
            wire_color: Wire color for network isolation
            network_id: Network ID for isolation checking
        """
        # First, check if there's an existing relay we can reuse near the ideal position
        for node in self.relay_nodes.values():
            node_dist_to_ideal = math.dist(node.position, ideal_pos)
            node_dist_to_source = math.dist(node.position, source_pos)

            # Check if this relay is usable:
            # - Within span limit from source
            # - Within 3 tiles of ideal position
            # - Can route this network
            if (
                node_dist_to_source <= span_limit
                and node_dist_to_ideal <= 3.0
                and node.can_route_network(network_id, wire_color)
            ):
                return node

        # Need to create a new relay - find the best available position
        return self._create_relay_directed(ideal_pos, source_pos, sink_pos, span_limit, signal_name)

    def _create_relay_directed(
        self,
        ideal_pos: tuple[float, float],
        source_pos: tuple[float, float],
        sink_pos: tuple[float, float],
        span_limit: float,
        signal_name: str,
    ) -> RelayNode | None:
        """Create a new relay, prioritizing positions toward the sink.

        Unlike the previous ring search, this method scores candidate positions
        based on:
        1. Distance from ideal position (closer is better)
        2. Progress toward sink (closer to sink is better)
        3. Reachability from source (must be within span_limit)
        """
        tile_pos = (int(round(ideal_pos[0])), int(round(ideal_pos[1])))

        # First try the exact ideal position
        if self.tile_grid.reserve_exact(tile_pos, footprint=(1, 1)):
            return self._finalize_relay_creation(tile_pos, signal_name, ideal_pos)

        # Collect all candidate positions within search radius
        candidates: list[tuple[float, tuple[int, int]]] = []  # (score, position)
        search_radius = 6  # tiles

        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                if dx == 0 and dy == 0:
                    continue  # Already tried ideal position

                candidate_pos = (tile_pos[0] + dx, tile_pos[1] + dy)

                # Skip if not available
                if not self.tile_grid.is_available(candidate_pos, footprint=(1, 1)):
                    continue

                # Calculate center position
                center = (candidate_pos[0] + 0.5, candidate_pos[1] + 0.5)

                # Check if reachable from source
                dist_to_source = math.dist(center, source_pos)
                if dist_to_source > span_limit:
                    continue  # Too far from source

                # Score: prefer positions that are
                # 1. Close to ideal position (weight: 1)
                # 2. Closer to sink (weight: 2 - progress is more important)
                dist_to_ideal = math.dist(center, ideal_pos)
                dist_to_sink = math.dist(center, sink_pos)

                # Lower score is better
                score = dist_to_ideal + 2.0 * dist_to_sink
                candidates.append((score, candidate_pos))

        if not candidates:
            return None

        # Sort by score and try to reserve the best positions
        candidates.sort(key=lambda x: x[0])

        for _score, candidate_pos in candidates:
            if self.tile_grid.reserve_exact(candidate_pos, footprint=(1, 1)):
                return self._finalize_relay_creation(candidate_pos, signal_name, ideal_pos)

        return None

    def _finalize_relay_creation(
        self,
        tile_pos: tuple[int, int],
        signal_name: str,
        ideal_pos: tuple[float, float],
    ) -> RelayNode:
        """Finalize relay creation with entity placement."""
        self._relay_counter += 1
        relay_id = f"__relay_{self._relay_counter}"
        center_pos = (tile_pos[0] + 0.5, tile_pos[1] + 0.5)

        self.diagnostics.info(
            f"Creating NEW relay {relay_id} at {center_pos} for {signal_name} "
            f"(ideal was {ideal_pos})"
        )

        relay_node = self.add_relay_node(center_pos, relay_id, "medium-electric-pole")

        self.layout_plan.create_and_add_placement(
            ir_node_id=relay_id,
            entity_type="medium-electric-pole",
            position=center_pos,
            footprint=(1, 1),
            role="wire_relay",
            debug_info={
                "variable": f"relay_{self._relay_counter}",
                "operation": "infrastructure",
                "details": "wire_relay",
                "role": "relay",
            },
        )

        return relay_node

    def _get_relay_node_by_id(self, relay_id: str) -> RelayNode | None:
        """Get a relay node by its entity ID."""
        for node in self.relay_nodes.values():
            if node.entity_id == relay_id:
                return node
        return None


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
        signal_usage: dict[str, SignalUsageEntry],
        diagnostics: ProgramDiagnostics,
        tile_grid: TileGrid,
        max_wire_span: float = 9.0,
        power_pole_type: str | None = None,
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

        self._circuit_edges: list[CircuitEdge] = []
        self._node_color_assignments: dict[tuple[str, str], str] = {}
        self._edge_color_map: dict[tuple[str, str, str], str] = {}
        self._coloring_conflicts: list[ConflictEdge] = []
        self._coloring_success = True
        self._relay_counter = 0
        self._routing_failed = False  # Track if any relay routing failed

        self._memory_modules: dict[str, Any] = {}

        self._edge_wire_colors: dict[tuple[str, str, str], str] = {}

        # Network IDs for relay isolation - computed from edge connectivity
        # Edges in the same network can share relays on the same wire color
        self._edge_network_ids: dict[tuple[str, str, str], int] = {}

        # Calculate relay search radius based on power pole grid spacing
        # For a grid with spacing S, the max distance from any point to nearest
        # grid point is S/2 * sqrt(2) ≈ 0.707 * S. Use S/2 * 1.5 for safety margin.
        self._relay_search_radius = self._compute_relay_search_radius()

        self.relay_network = RelayNetwork(
            self.tile_grid,
            None,  # No clusters
            {},  # No entity_to_cluster mapping
            self.max_wire_span,
            self.layout_plan,
            self.diagnostics,
            self.config,
            relay_search_radius=self._relay_search_radius,
        )

    def _compute_relay_search_radius(self) -> float:
        """Calculate optimal search radius for finding existing relays.

        Based on power pole grid spacing: for a grid with spacing S,
        the maximum distance from any point to the nearest grid point
        is S/2 * sqrt(2) ≈ 0.707 * S. We use S/2 * 1.5 for safety margin.
        """
        from .power_planner import POWER_POLE_CONFIG

        if self.power_pole_type:
            config = POWER_POLE_CONFIG.get(self.power_pole_type.lower())
            if config:
                supply_radius = float(config["supply_radius"])  # type: ignore[arg-type]
                grid_spacing = 2.0 * supply_radius
                # S/2 * 1.5 gives good coverage with safety margin
                return grid_spacing * 0.75

        # Default: use a generous radius when no power poles
        return 3.0

    def _compute_network_ids(self, edges: Sequence[CircuitEdge]) -> None:
        """Compute network IDs for relay isolation.

        Network isolation is based on SIGNAL SOURCES, not entity connectivity.
        Two edges can share a relay on the same wire color ONLY if they
        originate from the SAME source entity. This prevents signal mixing
        between different producers.

        The key insight: in Factorio, wires of the same color connected to
        the same entity form a single network where all signals get merged.
        So if signal-A from source1 and signal-B from source2 share a relay,
        both signals will appear at both destinations - causing flickering.
        """
        # Group edges by (source_entity, wire_color)
        # Each unique (source, color) pair gets its own network ID
        # This ensures signals from different sources never share relays

        next_network_id = 1
        source_color_to_network: dict[tuple[str, str], int] = {}

        for edge in edges:
            if not edge.source_entity_id:
                continue

            edge_key = (
                edge.source_entity_id,
                edge.sink_entity_id,
                edge.resolved_signal_name,
            )
            color = self._edge_color_map.get(edge_key, "red")

            # Create network ID based on (source, color) pair
            source_color_key = (edge.source_entity_id, color)
            if source_color_key not in source_color_to_network:
                source_color_to_network[source_color_key] = next_network_id
                next_network_id += 1

            network_id = source_color_to_network[source_color_key]
            self._edge_network_ids[edge_key] = network_id

        num_networks = len(source_color_to_network)
        self.diagnostics.info(f"Computed network IDs: {num_networks} isolated source networks")

    def plan_connections(
        self,
        signal_graph: Any,
        entities: dict[str, Any],
        wire_merge_junctions: dict[str, Any] | None = None,
        locked_colors: dict[tuple[str, str], str] | None = None,
        merge_membership: dict[str, set] | None = None,
    ) -> bool:
        """Compute all wire connections with color assignments.

        Returns:
            True if all connections were successfully routed, False if any relay
            routing failed (layout may need to be retried with different parameters).
        """
        self._register_power_poles_as_relays()

        self._add_self_feedback_connections()

        preserved_connections = list(self.layout_plan.wire_connections)
        self.layout_plan.wire_connections.clear()
        self._circuit_edges = []
        self._node_color_assignments = {}
        self._edge_color_map = {}
        self._routing_failed = False  # Reset routing failure flag
        self._coloring_conflicts = []
        self._coloring_success = True
        self._relay_counter = 0

        base_edges = collect_circuit_edges(signal_graph, self.signal_usage, entities)
        expanded_edges = self._expand_merge_edges(
            base_edges, wire_merge_junctions, entities, signal_graph
        )

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
            if edge.source_entity_id is not None
            and not self._is_memory_feedback_edge(
                edge.source_entity_id, edge.sink_entity_id, edge.resolved_signal_name
            )
        ]

        if len(expanded_edges) != len(non_feedback_edges):
            self.diagnostics.info(
                f"Filtered {len(expanded_edges) - len(non_feedback_edges)} memory feedback edges from wire coloring "
                f"({len(non_feedback_edges)} edges remaining)"
            )

        # Compute edge-level locked colors for sources that participate in multiple merges
        # This ensures that edges from the same source to different merge chains use different colors
        edge_locked_colors = self._compute_edge_locked_colors(
            non_feedback_edges, merge_membership or {}, signal_graph
        )

        # Combine with caller-provided locked colors
        all_locked_colors = dict(locked_colors or {})
        all_locked_colors.update(edge_locked_colors)

        coloring_result = plan_wire_colors(non_feedback_edges, all_locked_colors)
        self._node_color_assignments = coloring_result.assignments
        self._coloring_conflicts = coloring_result.conflicts
        self._coloring_success = coloring_result.is_bipartite

        # Build edge-level color map
        # First, use node-level assignments as default, then apply edge-level overrides
        edge_color_map: dict[tuple[str, str, str], str] = {}
        for edge in non_feedback_edges:
            if not edge.source_entity_id:
                continue

            edge_key = (edge.source_entity_id, edge.sink_entity_id, edge.resolved_signal_name)

            # Check for edge-level locked color first (based on merge origin)
            if edge.originating_merge_id:
                # Use (source, merge_id) as key for edge-level locks
                edge_lock_key = (edge.source_entity_id, edge.originating_merge_id)
                if edge_lock_key in edge_locked_colors:
                    edge_color_map[edge_key] = edge_locked_colors[edge_lock_key]
                    continue

            # Fall back to node-level assignment
            node_key = (edge.source_entity_id, edge.resolved_signal_name)
            color = self._node_color_assignments.get(node_key, WIRE_COLORS[0])
            edge_color_map[edge_key] = color

        self._edge_color_map = edge_color_map

        # Compute network IDs for relay isolation
        # Edges in the same connected component (per wire color) can share relays
        self._compute_network_ids(non_feedback_edges)

        self._log_color_summary()
        self._log_unresolved_conflicts()
        self._populate_wire_connections()
        if preserved_connections:
            self.layout_plan.wire_connections.extend(preserved_connections)

        self._validate_relay_coverage()

        return not self._routing_failed

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

    def get_network_id_for_edge(
        self, source_entity_id: str, sink_entity_id: str, signal_name: str
    ) -> int:
        """Get the network ID for a specific edge.

        Args:
            source_entity_id: The entity producing the signal
            sink_entity_id: The entity consuming the signal
            signal_name: The RESOLVED Factorio signal name (e.g., "signal-A")

        Returns:
            Network ID (0 if not found, which allows sharing with any network)
        """
        edge_key = (source_entity_id, sink_entity_id, signal_name)
        return self._edge_network_ids.get(edge_key, 0)

    def _compute_edge_locked_colors(
        self,
        edges: Sequence[CircuitEdge],
        merge_membership: dict[str, set],
        signal_graph: Any = None,
    ) -> dict[tuple[str, str], str]:
        """Compute edge-level locked colors for sources participating in multiple merges.

        When a source entity participates in multiple independent merges, the edges
        from that source to different merge chains need to use different wire colors
        to keep the networks separated - BUT only when those different paths would
        arrive at the SAME final entity.

        For example, in a balanced loader:
        - Chest1 output participates in both 'total' merge (all chests → combinator)
          and 'input1' merge (chest1 + neg_avg → inserter1)
        - Chest1's signal arrives at inserter1 via two paths:
          1. chest1 → combinator → inserter1 (computes negative average)
          2. chest1 → inserter1 (direct individual content)
        - These paths MUST use different colors to prevent double-counting

        But for the combinator output:
        - It participates in merge_65..merge_70 (6 inserter input merges)
        - Each goes to a DIFFERENT inserter - no shared destination
        - Same color is fine for all (no conflict)

        Args:
            edges: All circuit edges after merge expansion
            merge_membership: Maps source_id -> set of merge_ids the source belongs to
            signal_graph: Signal graph for resolving IR node IDs to entity IDs

        Returns:
            Dict mapping (actual_entity_id, merge_id) -> wire color
        """
        locked: dict[tuple[str, str], str] = {}

        # Build a reverse map: for each edge, map (source_entity, merge_id) to edges
        edge_source_merges: dict[tuple[str, str], list[CircuitEdge]] = {}
        for edge in edges:
            if edge.originating_merge_id and edge.source_entity_id:
                key = (edge.source_entity_id, edge.originating_merge_id)
                edge_source_merges.setdefault(key, []).append(edge)

        # Build map of merge_id -> set of source entity IDs (to detect transitive conflicts)
        merge_to_sources: dict[str, set] = {}
        for edge in edges:
            if edge.originating_merge_id and edge.source_entity_id:
                merge_to_sources.setdefault(edge.originating_merge_id, set()).add(
                    edge.source_entity_id
                )

        # Build map of merge_id -> set of sink entity IDs
        merge_to_sinks: dict[str, set] = {}
        for edge in edges:
            if edge.originating_merge_id:
                merge_to_sinks.setdefault(edge.originating_merge_id, set()).add(edge.sink_entity_id)

        # Find sources that participate in multiple merges
        for source_id, merge_ids in merge_membership.items():
            if len(merge_ids) <= 1:
                continue

            # Resolve the source_id (which might be an IR node ID like entity_output_ir_43)
            # to the actual entity ID (like entity_ir_31)
            actual_source_id = source_id
            if signal_graph is not None:
                resolved = signal_graph.get_source(source_id)
                if resolved:
                    actual_source_id = resolved

            # Find which merges this source has edges for
            source_merge_edges: dict[str, list[CircuitEdge]] = {}
            for merge_id in merge_ids:
                key = (actual_source_id, merge_id)
                if key in edge_source_merges:
                    source_merge_edges[merge_id] = edge_source_merges[key]

            if len(source_merge_edges) <= 1:
                continue

            # Check for transitive conflict:
            # If one merge's sink is a source in another merge, there's a path conflict
            # This means the source's signal can arrive at a final entity via two paths
            merge_list = sorted(source_merge_edges.keys())
            has_conflict = False
            for i, m1 in enumerate(merge_list):
                sinks1 = merge_to_sinks.get(m1, set())
                for m2 in merge_list[i + 1 :]:
                    sources2 = merge_to_sources.get(m2, set())
                    # If m1's sink is a source in m2, there's a transitive path
                    if sinks1 & sources2:
                        has_conflict = True
                        self.diagnostics.info(
                            f"Transitive conflict detected: {actual_source_id} in {m1} (sink {sinks1}) "
                            f"feeds into source of {m2} (sources {sources2 & sinks1})"
                        )
                        break
                    # Check the reverse direction too
                    sinks2 = merge_to_sinks.get(m2, set())
                    sources1 = merge_to_sources.get(m1, set())
                    if sinks2 & sources1:
                        has_conflict = True
                        self.diagnostics.info(
                            f"Transitive conflict detected (reverse): {actual_source_id} in {m2} (sink {sinks2}) "
                            f"feeds into source of {m1} (sources {sources1 & sinks2})"
                        )
                        break
                if has_conflict:
                    break

            if not has_conflict:
                # No transitive conflict - skip color locking for this source
                continue

            # Assign alternating colors to different merges
            # Use sorted order for determinism
            for i, merge_id in enumerate(merge_list):
                color = WIRE_COLORS[i % 2]  # red for index 0, green for index 1, ...
                locked[(actual_source_id, merge_id)] = color

            self.diagnostics.info(
                f"Locked wire colors for {actual_source_id} (from {source_id}) across {len(merge_list)} merges: "
                + ", ".join(f"{m}={locked.get((actual_source_id, m), '?')}" for m in merge_list)
            )

        return locked

    def _expand_merge_edges(
        self,
        edges: Sequence[CircuitEdge],
        wire_merge_junctions: dict[str, Any] | None,
        entities: dict[str, Any],
        signal_graph: Any = None,
    ) -> list[CircuitEdge]:
        if not wire_merge_junctions:
            return list(edges)

        expanded: list[CircuitEdge] = []
        for edge in edges:
            if edge.sink_entity_id in wire_merge_junctions:
                continue

            source_id = edge.source_entity_id or ""
            merge_info = wire_merge_junctions.get(source_id)
            if not merge_info:
                expanded.append(edge)
                continue

            # Track that this edge came from expanding a merge
            originating_merge_id = source_id

            for source_ref in merge_info.get("inputs", []):
                # Handle both SignalRef and BundleRef
                if isinstance(source_ref, (SignalRef, BundleRef)):
                    ir_source_id = source_ref.source_id
                else:
                    continue

                # Resolve IR node ID to actual entity ID using signal graph
                actual_source_id = ir_source_id
                if signal_graph is not None:
                    entity_id = signal_graph.get_source(ir_source_id)
                    if entity_id:
                        actual_source_id = entity_id

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
                        originating_merge_id=originating_merge_id,
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
            self.diagnostics.info(
                f"Registered power pole {entity_id} at {placement.position} as relay"
            )
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
        self, edges: Sequence[CircuitEdge], entities: dict[str, Any]
    ) -> None:
        conflict_map: dict[str, dict[str, set[str]]] = {}

        for edge in edges:
            if not edge.source_entity_id:
                continue
            sink_conflicts = conflict_map.setdefault(edge.sink_entity_id, {})
            sink_conflicts.setdefault(edge.resolved_signal_name, set()).add(edge.source_entity_id)

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

                self.diagnostics.info(
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
            self.diagnostics.info("Wire color planner assignments: " + ", ".join(summaries))

    def _log_unresolved_conflicts(self) -> None:
        if self._coloring_success or not self._coloring_conflicts:
            return

        for conflict in self._coloring_conflicts:
            resolved_signal = conflict.nodes[0][1]
            source_desc = ", ".join(sorted({node_id for node_id, _ in conflict.nodes}))
            sink_desc = ", ".join(sorted(conflict.sinks)) if conflict.sinks else "unknown sinks"
            self.diagnostics.info(
                "Two-color routing could not isolate signal "
                f"'{resolved_signal}' across sinks [{sink_desc}]; falling back to single-channel wiring for involved entities ({source_desc})."
            )

    def _get_connection_side(self, entity_id: str, is_source: bool) -> str | None:
        """Determine if entity needs 'input'/'output' side specified.

        Entities with dual circuit connectors (like arithmetic-combinator, decider-combinator,
        selector-combinator) have separate input and output sides. When wiring these entities,
        we need to specify which side to connect to.

        Args:
            entity_id: Entity to check
            is_source: True if this entity is producing the signal

        Returns:
            'output' for source side of dual-connectable entities,
            'input' for sink side, None otherwise
        """
        placement = self.layout_plan.get_placement(entity_id)
        if not placement:
            return None

        if is_dual_circuit_connectable(placement.entity_type):
            return "output" if is_source else "input"

        return None

    def _is_memory_feedback_edge(self, source_id: str, sink_id: str, signal_name: str) -> bool:
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

            if source_id == write_id and sink_id == hold_id and signal_name == module.signal_type:
                return True

            if source_id == hold_id and sink_id == write_id and signal_name == module.signal_type:
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

            if (
                hasattr(module, "_feedback_signal_ids")
                and signal_name in module._feedback_signal_ids
            ):
                return True

        if signal_name.startswith("__feedback_"):
            self.diagnostics.info(
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
        signal_groups: dict[tuple[str, str], list[tuple[CircuitEdge, str]]] = {}

        for edge in self._circuit_edges:
            if not edge.source_entity_id or not edge.sink_entity_id:
                continue

            # Determine wire color
            color: str
            if self._is_memory_feedback_edge(
                edge.source_entity_id, edge.sink_entity_id, edge.resolved_signal_name
            ):
                color = "red"
            else:
                color_opt = self._edge_color_map.get(
                    (
                        edge.source_entity_id,
                        edge.sink_entity_id,
                        edge.resolved_signal_name,
                    )
                )
                color = color_opt if color_opt is not None else WIRE_COLORS[0]

            group_key = (edge.resolved_signal_name, color)
            if group_key not in signal_groups:
                signal_groups[group_key] = []
            signal_groups[group_key].append((edge, color))

        # Step 2: Process each signal group with per-source analysis
        mst_star_count = 0
        direct_routed_count = 0

        # Sort for deterministic iteration order
        for (signal_name, wire_color), edge_color_pairs in sorted(signal_groups.items()):
            edges = [pair[0] for pair in edge_color_pairs]

            # Debug: log signal group processing
            if "arith_15" in str([e.source_entity_id for e in edges]):
                self.diagnostics.info(
                    f"Processing signal group ({signal_name}, {wire_color}) with {len(edges)} edges, "
                    f"sources: { {e.source_entity_id for e in edges} }"
                )

            # Find all bidirectional pairs in this signal group
            bidirectional_pairs = self._find_bidirectional_pairs(edges)

            # Group edges by source
            by_source: dict[str, list[CircuitEdge]] = {}
            for edge in edges:
                if edge.source_entity_id is None:
                    continue
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

    def _find_bidirectional_pairs(self, edges: list[CircuitEdge]) -> set:
        """Find all bidirectional edge pairs (A→B and B→A both exist).

        Returns:
            Set of (source, sink) tuples that are part of bidirectional pairs.
            Both directions are included: if A↔B, returns {(A,B), (B,A)}.
        """
        pairs = set()
        edge_set = {
            (e.source_entity_id, e.sink_entity_id) for e in edges if e.source_entity_id is not None
        }

        for edge in edges:
            if edge.source_entity_id is None:
                continue
            reverse = (edge.sink_entity_id, edge.source_entity_id)
            if reverse in edge_set:
                pairs.add((edge.source_entity_id, edge.sink_entity_id))
                pairs.add(reverse)

        return pairs

    def _route_edge_directly(self, edge: CircuitEdge, wire_color: str) -> bool:
        """Route a single edge directly (no MST optimization).

        Returns:
            True if routing succeeded, False if relay placement failed.
        """
        if edge.source_entity_id is None:
            return True

        edge_key = (
            edge.source_entity_id,
            edge.sink_entity_id,
            edge.resolved_signal_name,
        )
        self._edge_wire_colors[edge_key] = wire_color

        source_side = self._get_connection_side(edge.source_entity_id, is_source=True)
        sink_side = self._get_connection_side(edge.sink_entity_id, is_source=False)

        return self._route_connection_with_relays(edge, wire_color, source_side, sink_side)

    def _apply_mst_to_source_fanout(
        self, source_id: str, sink_ids: list[str], signal_name: str, wire_color: str
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
            self.diagnostics.info(
                f"MST bug: source '{source_id}' not connected in MST edges: {mst_edges}"
            )
            return False

        # Pre-validate ALL MST edges are short enough to NOT need relays
        # If any edge needs relays, skip MST entirely to avoid relay conflicts
        # between different signal groups
        span_limit = self.relay_network.span_limit
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
            if distance > span_limit:
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

            # Store wire color for MST edges, but DON'T overwrite existing assignments.
            # This prevents MST routing from clobbering wire colors that were already
            # correctly assigned for edges belonging to a different signal group (color).
            # Example: arith_7 -> decider_8 might be GREEN for output_spec copying,
            # but arith_3's MST might include arith_7 ↔ decider_8 as a routing path.
            edge_key_ab = (ent_a, ent_b, signal_name)
            edge_key_ba = (ent_b, ent_a, signal_name)
            if edge_key_ab not in self._edge_wire_colors:
                self._edge_wire_colors[edge_key_ab] = wire_color
            if edge_key_ba not in self._edge_wire_colors:
                self._edge_wire_colors[edge_key_ba] = wire_color

            # Get network ID from the first original edge (all share the same network)
            network_id = self.get_network_id_for_edge(source_id, sink_ids[0], signal_name)

            # Route the connection
            if not self._route_mst_edge(
                ent_a, ent_b, signal_name, wire_color, side_a, side_b, network_id
            ):
                all_succeeded = False

        return all_succeeded

    def _build_minimum_spanning_tree(self, entity_ids: list[str]) -> list[tuple[str, str]]:
        """Build minimum spanning tree over entities using Prim's algorithm.

        Args:
            entity_ids: List of entity IDs to connect

        Returns:
            List of (entity_a, entity_b) edges forming the MST
        """
        if len(entity_ids) <= 1:
            return []

        # Collect positions for entities that have valid placements
        positions: dict[str, tuple[float, float]] = {}
        for entity_id in entity_ids:
            placement = self.layout_plan.get_placement(entity_id)
            if placement and placement.position:
                positions[entity_id] = placement.position

        valid_entities = [e for e in entity_ids if e in positions]
        if len(valid_entities) <= 1:
            return []

        # Prim's algorithm: greedily grow MST from first entity (should be source)
        in_tree = {valid_entities[0]}
        mst_edges: list[tuple[str, str]] = []

        while len(in_tree) < len(valid_entities):
            best_edge: tuple[str, str] | None = None
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
        side_a: str | None,
        side_b: str | None,
        network_id: int = 0,
    ) -> bool:
        """Create wire connection for MST edge, with relay poles if needed.

        Returns:
            True if routing succeeded, False if it failed.
        """

        placement_a = self.layout_plan.get_placement(entity_a)
        placement_b = self.layout_plan.get_placement(entity_b)

        if not placement_a or not placement_b:
            self.diagnostics.info(
                f"Skipped MST edge for '{signal_name}': missing placement "
                f"({entity_a} or {entity_b})"
            )
            return False

        if not placement_a.position or not placement_b.position:
            self.diagnostics.info(
                f"Skipped MST edge for '{signal_name}': missing position ({entity_a} or {entity_b})"
            )
            return False

        # Use relay network for long edges
        relay_path = self.relay_network.route_signal(
            placement_a.position,
            placement_b.position,
            signal_name,
            wire_color,
            network_id,
        )

        if relay_path is None:
            # Relay routing failed - connection cannot be established
            self.diagnostics.warning(
                f"MST edge for '{signal_name}' cannot be routed: "
                f"relay placement failed between {entity_a} and {entity_b}. "
                f"The layout may be too spread out for the available wire span."
            )
            self._routing_failed = True
            return False

        self._create_relay_chain(
            entity_a, entity_b, signal_name, wire_color, relay_path, side_a, side_b
        )

        return True

    def _create_relay_chain(
        self,
        source_id: str,
        sink_id: str,
        signal_name: str,
        wire_color: str,
        relay_path: list[tuple[str, str]],
        source_side: str | None = None,
        sink_side: str | None = None,
    ) -> None:
        """Create wire connections through a relay path.

        This is the shared implementation for chaining connections through
        relay poles. Used by both MST edge routing and regular connection routing.

        Args:
            source_id: Starting entity ID.
            sink_id: Ending entity ID.
            signal_name: Name of the signal being routed.
            wire_color: Wire color for direct connection (used if relay_path is empty).
            relay_path: List of (relay_id, wire_color) tuples from relay network.
            source_side: Circuit side for source entity (None for poles).
            sink_side: Circuit side for sink entity (None for poles).
        """
        if len(relay_path) == 0:
            # Direct connection - no relays needed
            self.layout_plan.add_wire_connection(
                WireConnection(
                    source_entity_id=source_id,
                    sink_entity_id=sink_id,
                    signal_name=signal_name,
                    wire_color=wire_color,
                    source_side=source_side,
                    sink_side=sink_side,
                )
            )
        else:
            # Chain through relay poles
            current_id = source_id
            current_side = source_side

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

            # Final connection to sink
            self.layout_plan.add_wire_connection(
                WireConnection(
                    source_entity_id=current_id,
                    sink_entity_id=sink_id,
                    signal_name=signal_name,
                    wire_color=relay_path[-1][1],
                    source_side=None,
                    sink_side=sink_side,
                )
            )

    def _route_connection_with_relays(
        self,
        edge: CircuitEdge,
        wire_color: str,
        source_side: str | None = None,
        sink_side: str | None = None,
    ) -> bool:
        """Route a connection with relays if needed using shared relay infrastructure.

        Returns:
            True if routing succeeded, False if relay placement failed.
        """
        if edge.source_entity_id is None:
            return True

        source = self.layout_plan.get_placement(edge.source_entity_id)
        sink = self.layout_plan.get_placement(edge.sink_entity_id)

        if source is None or sink is None or source.position is None or sink.position is None:
            self.diagnostics.info(
                f"Skipped wiring for '{edge.resolved_signal_name}' due to missing placement ({edge.source_entity_id} -> {edge.sink_entity_id})."
            )
            return True

        # Get network ID for this edge
        network_id = self.get_network_id_for_edge(
            edge.source_entity_id, edge.sink_entity_id, edge.resolved_signal_name
        )

        relay_path = self.relay_network.route_signal(
            source.position,
            sink.position,
            edge.resolved_signal_name,
            wire_color,
            network_id,
        )

        if relay_path is None:
            # Relay routing failed - connection cannot be established
            self.diagnostics.warning(
                f"Connection for '{edge.resolved_signal_name}' cannot be routed: "
                f"relay placement failed between {edge.source_entity_id} and {edge.sink_entity_id}. "
                f"The layout may be too spread out for the available wire span."
            )
            self._routing_failed = True
            return False

        self._create_relay_chain(
            edge.source_entity_id,
            edge.sink_entity_id,
            edge.resolved_signal_name,
            wire_color,
            relay_path,
            source_side,
            sink_side,
        )
        return True

    def _validate_relay_coverage(self) -> None:
        """Validate that all wire connections have adequate relay coverage.

        Logs warnings for any connections that exceed span limits.
        """
        span_limit = self.relay_network.span_limit
        epsilon = 1e-6

        violation_count = 0

        for connection in self.layout_plan.wire_connections:
            source = self.layout_plan.get_placement(connection.source_entity_id)
            sink = self.layout_plan.get_placement(connection.sink_entity_id)

            if not source or not sink or not source.position or not sink.position:
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
                f"Total {violation_count} wire connections exceed span limit (showing first 5)"
            )

        relay_count = sum(
            1
            for p in self.layout_plan.entity_placements.values()
            if getattr(p, "role", None) == "wire_relay"
        )

        if relay_count > 0:
            self.diagnostics.warning(
                f"Blueprint complexity required {relay_count} wire relay pole(s) to route signals"
            )

    def edge_color_map(self) -> dict[tuple[str, str, str], str]:
        """Expose raw edge→color assignments."""

        return dict(self._edge_color_map)
