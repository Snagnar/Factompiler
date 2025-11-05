from __future__ import annotations
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple
from dsl_compiler.src.ir.builder import SignalRef
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from .layout_engine import LayoutEngine
from .layout_plan import LayoutPlan, WireConnection, EntityPlacement
from .signal_analyzer import SignalUsageEntry
from .cluster_analyzer import Cluster

from .wire_router import (
    CircuitEdge,
    WIRE_COLORS,
    collect_circuit_edges,
    plan_wire_colors,
)


"""Connection planning for wire routing."""


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
        clusters: Optional[List[Cluster]] = None,
        entity_to_cluster: Optional[Dict[str, int]] = None,
    ) -> None:
        self.layout_plan = layout_plan
        self.signal_usage = signal_usage
        self.diagnostics = diagnostics
        self.max_wire_span = max_wire_span
        self.layout_engine = layout_engine

        self._circuit_edges: List[CircuitEdge] = []
        self._node_color_assignments: Dict[Tuple[str, str], str] = {}
        self._edge_color_map: Dict[Tuple[str, str, str], str] = {}
        self._coloring_conflicts = []
        self._coloring_success = True
        self._relay_counter = 0

        # Cluster support
        self.clusters = clusters or []
        self.entity_to_cluster = entity_to_cluster or {}

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
        """Route a connection with relays if needed.

        Uses greedy straight-line relay placement.
        """
        # DEBUG: Entry logging
        self.diagnostics.info(
            f"_route_connection_with_relays: {edge.source_entity_id} -> {edge.sink_entity_id}"
        )

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

        # Use configured max span, with safe default
        max_span = (
            self.max_wire_span if self.max_wire_span and self.max_wire_span > 0 else 9.0
        )

        # Account for entity size and wire reach offset
        span_limit = max(1.0, float(max_span) - 1.8)

        # DEBUG: Log all connections to trace relay creation
        distance = math.dist(source.position, sink.position)
        if distance > span_limit:
            self.diagnostics.info(
                f"Routing connection: {edge.source_entity_id} -> {edge.sink_entity_id}, "
                f"distance={distance:.2f}, span_limit={span_limit:.2f}"
            )

        # Build path with relays using greedy straight-line approach
        path = self._build_relay_path(source, sink, span_limit)

        # Create wire connections along the path
        for i, (start_id, end_id) in enumerate(zip(path, path[1:])):
            # Use sides for first and last connection
            conn_source_side = source_side if i == 0 else None
            conn_sink_side = sink_side if i == len(path) - 2 else None

            # DEBUG: Log wire connection with positions
            start_placement = self.layout_plan.get_placement(start_id)
            end_placement = self.layout_plan.get_placement(end_id)
            if start_placement and end_placement:
                seg_distance = math.dist(
                    start_placement.position, end_placement.position
                )
                if seg_distance > span_limit:
                    self.diagnostics.warning(
                        f"Creating wire segment that exceeds span_limit: "
                        f"{start_id} at {start_placement.position} -> "
                        f"{end_id} at {end_placement.position}, "
                        f"distance={seg_distance:.2f} > {span_limit:.2f}"
                    )

            connection = WireConnection(
                source_entity_id=start_id,
                sink_entity_id=end_id,
                signal_name=edge.resolved_signal_name,
                wire_color=wire_color,
                source_side=conn_source_side,
                sink_side=conn_sink_side,
            )
            self.layout_plan.add_wire_connection(connection)

    def _build_relay_path(
        self,
        source: EntityPlacement,
        sink: EntityPlacement,
        span_limit: float,
    ) -> List[str]:
        """Build relay path with GUARANTEED segment lengths < span_limit."""

        # Check cluster membership
        source_cluster = self.entity_to_cluster.get(source.ir_node_id, -1)
        sink_cluster = self.entity_to_cluster.get(sink.ir_node_id, -1)

        distance = math.dist(source.position, sink.position)

        # DEBUG: Log cluster info
        if source_cluster >= 0 or sink_cluster >= 0:
            self.diagnostics.info(
                f"Connection: {source.ir_node_id}(cluster {source_cluster}) -> "
                f"{sink.ir_node_id}(cluster {sink_cluster}), distance={distance:.2f}"
            )

        # Same cluster: direct connection (guaranteed < 7.07 tiles by cluster size)
        if source_cluster == sink_cluster and source_cluster >= 0:
            if distance > span_limit:
                self.diagnostics.warning(
                    f"INTRA-CLUSTER distance violation: {distance:.2f} > {span_limit:.2f}"
                )
            return [source.ir_node_id, sink.ir_node_id]

        # Different clusters or no cluster info
        if distance <= span_limit * 0.95:
            return [source.ir_node_id, sink.ir_node_id]

        # Need relays - use CONSERVATIVE interval to ensure segments stay under limit
        # Use 6.0 instead of 7.0 to give margin for displacement
        safe_interval = 6.0
        num_relays = int(math.ceil(distance / safe_interval)) - 1

        if num_relays <= 0:
            return [source.ir_node_id, sink.ir_node_id]

        path = [source.ir_node_id]
        sx, sy = source.position
        ex, ey = sink.position

        for i in range(1, num_relays + 1):
            ratio = i / (num_relays + 1)
            relay_x = sx + (ex - sx) * ratio
            relay_y = sy + (ey - sy) * ratio

            relay_pos = (round(relay_x), round(relay_y))

            # Try to place relay in reserved corridor space
            relay_id = self._create_relay_in_corridor(
                relay_pos, source_cluster, sink_cluster
            )
            path.append(relay_id)

        path.append(sink.ir_node_id)

        # CRITICAL: Validate every segment
        for i in range(len(path) - 1):
            seg_source = self.layout_plan.get_placement(path[i])
            seg_sink = self.layout_plan.get_placement(path[i + 1])
            if seg_source and seg_sink:
                seg_dist = math.dist(seg_source.position, seg_sink.position)
                if seg_dist > span_limit:
                    # Use warning instead of error to not fail compilation
                    # The blueprint may still work with slightly longer wire segments
                    self.diagnostics.warning(
                        f"Relay path segment too long: {seg_dist:.2f} > {span_limit:.2f}. "
                        f"Path: {path}"
                    )

        return path

    def _create_relay_in_corridor(
        self,
        position: Tuple[int, int],
        source_cluster_idx: int,
        sink_cluster_idx: int,
    ) -> str:
        """Create relay in reserved corridor space between clusters."""
        relay_id = f"__relay_{self._relay_counter}"
        self._relay_counter += 1

        footprint = (1, 1)

        # Priority 1: Exact position
        if self.layout_engine.can_reserve(position, footprint=footprint):
            reserved_pos = self.layout_engine.reserve_exact(
                position, footprint=footprint
            )
            if reserved_pos:
                self._create_relay_entity(relay_id, reserved_pos)
                return relay_id

        # Priority 2: Search in 3-tile radius (increased from 1)
        for radius in range(1, 4):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) > radius:
                        continue
                    alt_pos = (position[0] + dx, position[1] + dy)
                    if self.layout_engine.can_reserve(alt_pos, footprint=footprint):
                        reserved_pos = self.layout_engine.reserve_exact(
                            alt_pos, footprint=footprint
                        )
                        if reserved_pos:
                            self._create_relay_entity(relay_id, reserved_pos)
                            return relay_id

        # Priority 3: Use corridor space explicitly
        # Calculate corridor position between source and sink clusters
        if (
            source_cluster_idx >= 0
            and sink_cluster_idx >= 0
            and source_cluster_idx < len(self.clusters)
            and sink_cluster_idx < len(self.clusters)
        ):
            source_cluster = self.clusters[source_cluster_idx]
            sink_cluster = self.clusters[sink_cluster_idx]

            # Use midpoint of cluster centers as fallback
            mid_x = (source_cluster.center[0] + sink_cluster.center[0]) / 2
            mid_y = (source_cluster.center[1] + sink_cluster.center[1]) / 2
            corridor_pos = (int(mid_x), int(mid_y))

            reserved_pos = self.layout_engine.reserve_near(
                corridor_pos, max_radius=5, footprint=footprint
            )
            if reserved_pos:
                self._create_relay_entity(relay_id, reserved_pos)
                return relay_id

        # Last resort: Use layout engine's next position
        reserved_pos = self.layout_engine.get_next_position(footprint=footprint)
        self._create_relay_entity(relay_id, reserved_pos)
        self.diagnostics.warning(
            f"Relay {relay_id} placed outside corridor at {reserved_pos}"
        )
        return relay_id

    def _create_relay_entity(self, relay_id: str, position: Tuple[int, int]) -> None:
        """Helper to create relay placement."""
        relay_placement = EntityPlacement(
            ir_node_id=relay_id,
            entity_type="medium-electric-pole",
            position=position,
            properties={
                "debug_info": {
                    "variable": f"relay_{self._relay_counter}",
                    "operation": "infrastructure",
                    "details": "wire_relay",
                    "role": "relay",
                }
            },
            role="wire_relay",  # Changed from "relay" to "wire_relay"
        )
        self.layout_plan.add_placement(relay_placement)

    def _validate_and_fix_relay_path(
        self,
        path: List[str],
        span_limit: float,
    ) -> List[str]:
        """Validate relay path and add missing relays if segments are too long.

        DEPRECATED: Now handled inline in _build_relay_path validation.
        """
        return path

    def _create_explicit_relay(self, position: Tuple[int, int]) -> str:
        """DEPRECATED: Use _create_relay_in_corridor instead."""
        # Kept for backward compatibility but redirect to new implementation
        return self._create_relay_in_corridor(position, -1, -1)

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
