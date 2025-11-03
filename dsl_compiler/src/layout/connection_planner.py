from __future__ import annotations
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple
from dsl_compiler.src.ir import SignalRef
from dsl_compiler.src.common import ProgramDiagnostics
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

    def _get_connection_side(
        self, 
        entity_id: str, 
        is_source: bool
    ) -> Optional[str]:
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
            source_side = self._get_connection_side(edge.source_entity_id, is_source=True)
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
        max_span = self.max_wire_span if self.max_wire_span and self.max_wire_span > 0 else 9.0
        
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
                seg_distance = math.dist(start_placement.position, end_placement.position)
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
        """Build relay path with EXPLICIT positioning every 7 tiles."""
        
        # DEBUG: Entry point logging
        self.diagnostics.info(f"_build_relay_path called: {source.ir_node_id} -> {sink.ir_node_id}")
        
        # DEBUG: Check if cluster mapping exists
        if not self.entity_to_cluster:
            self.diagnostics.warning(
                f"entity_to_cluster is EMPTY - cannot determine cluster membership!"
            )
        
        # Check if in same cluster
        source_cluster = self.entity_to_cluster.get(source.ir_node_id, -1)
        sink_cluster = self.entity_to_cluster.get(sink.ir_node_id, -1)
        
        # Calculate distance first
        distance = math.dist(source.position, sink.position)
        
        # DEBUG: Log cluster info
        if source_cluster >= 0 or sink_cluster >= 0:
            self.diagnostics.info(
                f"Connection: {source.ir_node_id}(cluster {source_cluster}) -> "
                f"{sink.ir_node_id}(cluster {sink_cluster}), distance={distance:.2f}"
            )
        
        if source_cluster == sink_cluster and source_cluster >= 0:
            # Same cluster - direct connection guaranteed by cluster sizing
            # DEBUG: Verify intra-cluster distances
            if distance > span_limit:
                self.diagnostics.warning(
                    f"INTRA-CLUSTER distance violation: {source.ir_node_id} -> "
                    f"{sink.ir_node_id} in cluster {source_cluster}, distance={distance:.2f} > {span_limit:.2f}"
                )
            return [source.ir_node_id, sink.ir_node_id]
        
        # Different clusters - add relays along straight line
        
        if distance <= span_limit * 0.95:  # 5% safety margin
            # Close enough - direct connection
            self.diagnostics.info(
                f"Direct connection (within span): {source.ir_node_id} -> {sink.ir_node_id}, "
                f"distance={distance:.2f} <= {span_limit * 0.95:.2f}"
            )
            return [source.ir_node_id, sink.ir_node_id]
        
        # Need relays - calculate interval to ensure max segment length < span_limit
        # Using 7.0 to reduce collision frequency
        # sqrt(7^2 + 7^2) = 9.90 > 9 - but actual path distances may be acceptable
        relay_interval = 7.0
        num_relays = int(math.ceil(distance / relay_interval)) - 1
        
        # DEBUG: Log inter-cluster connections
        self.diagnostics.info(
            f"NEEDS RELAYS: {source.ir_node_id}(cluster {source_cluster}) -> "
            f"{sink.ir_node_id}(cluster {sink_cluster}), distance={distance:.2f}, "
            f"num_relays={num_relays}, relay_interval={relay_interval}"
        )
        
        if num_relays <= 0:
            return [source.ir_node_id, sink.ir_node_id]
        
        # Create relay positions
        path = [source.ir_node_id]
        
        sx, sy = source.position
        ex, ey = sink.position
        
        for i in range(1, num_relays + 1):
            ratio = i / (num_relays + 1)
            relay_x = sx + (ex - sx) * ratio
            relay_y = sy + (ey - sy) * ratio
            
            # Snap to grid
            relay_pos = (round(relay_x), round(relay_y))
            
            # Create relay at EXPLICIT position
            relay_id = self._create_explicit_relay(relay_pos)
            path.append(relay_id)
        
        path.append(sink.ir_node_id)
        
        # CRITICAL: Validate and fix the path after all relays placed
        path = self._validate_and_fix_relay_path(path, span_limit)
        
        return path
    
    def _validate_and_fix_relay_path(
        self,
        path: List[str],
        span_limit: float,
    ) -> List[str]:
        """Validate relay path and add missing relays if segments are too long.
        
        After relays are placed, their actual positions may differ from intended
        due to collision avoidance. This function checks all segments and adds
        additional relays where needed.
        
        DISABLED: This creates an infinite loop - adding intermediate relays
        that themselves get displaced, requiring more relays...
        
        The real solution is to ensure relays are NEVER displaced, or use
        existing power pole grid as relay infrastructure.
        """
        # For now, just return the original path and accept some warnings
        # TODO: Implement proper relay corridor reservation or power pole reuse
        return path

    def _create_explicit_relay(self, position: Tuple[int, int]) -> str:
        """Create relay pole at explicit position."""
        
        relay_id = f"__relay_{self._relay_counter}"
        self._relay_counter += 1
        
        # Reserve position through layout engine to avoid overlaps
        footprint = (1, 1)  # Power poles are 1x1
        reserved_pos = self.layout_engine.reserve_exact(position, footprint=footprint)
        if reserved_pos is None:
            # Try within 1 tile radius
            reserved_pos = self.layout_engine._find_nearest_available(
                position, max_radius=1, footprint=footprint, padding=0, alignment=1
            )
            if reserved_pos:
                reserved_pos = self.layout_engine._claim_position(reserved_pos, footprint, padding=0)
        
        if reserved_pos is None:
            # Fallback to any available position - will cause warnings but won't fail compilation
            reserved_pos = self.layout_engine.get_next_position(footprint=footprint, padding=0)
        
        relay_placement = EntityPlacement(
            ir_node_id=relay_id,
            entity_type="medium-electric-pole",
            position=reserved_pos,
            properties={
                "debug_info": {
                    "variable": f"relay_{self._relay_counter}",
                    "operation": "infrastructure",
                    "details": "inter_cluster_relay",
                    "role": "relay",
                }
            },
            role="relay",
        )
        self.layout_plan.add_placement(relay_placement)
        return relay_id

    def _validate_relay_coverage(self) -> None:
        """Validate that all wire connections have adequate relay coverage.
        
        Logs warnings for any connections that exceed span limits.
        """
        max_span = self.max_wire_span if self.max_wire_span and self.max_wire_span > 0 else 9.0
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
            1 for p in self.layout_plan.entity_placements.values()
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
