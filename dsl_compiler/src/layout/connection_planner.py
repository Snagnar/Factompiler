from __future__ import annotations
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple
from dsl_compiler.src.ir import SignalRef
from dsl_compiler.src.common import ProgramDiagnostics
from .layout_engine import LayoutEngine
from .layout_plan import LayoutPlan, WireConnection, EntityPlacement
from .signal_analyzer import SignalUsageEntry

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
        
        # Build path with relays using greedy straight-line approach
        path = self._build_relay_path(source, sink, span_limit)

        # Create wire connections along the path
        for i, (start_id, end_id) in enumerate(zip(path, path[1:])):
            # Use sides for first and last connection
            conn_source_side = source_side if i == 0 else None
            conn_sink_side = sink_side if i == len(path) - 2 else None
            
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
        """Build a relay path using greedy straight-line placement.
        
        Places relays at evenly-spaced intervals along the direct line
        between source and sink. Reuses existing relays when possible.
        
        Args:
            source: Source entity placement
            sink: Sink entity placement  
            span_limit: Maximum wire span distance
            
        Returns:
            List of entity IDs forming the path from source to sink
        """
        epsilon = 1e-6
        base_distance = math.dist(source.position, sink.position)
        
        # No relays needed if within span
        if base_distance <= span_limit + epsilon:
            return [source.ir_node_id, sink.ir_node_id]
        
        # Calculate number of relays needed
        num_relays = math.ceil(base_distance / span_limit) - 1
        
        if num_relays <= 0:
            return [source.ir_node_id, sink.ir_node_id]
        
        # Build path with relays
        path_ids = [source.ir_node_id]
        
        for i in range(1, num_relays + 1):
            # Calculate ideal position along straight line
            ratio = i / (num_relays + 1)
            ideal_x = source.position[0] + (sink.position[0] - source.position[0]) * ratio
            ideal_y = source.position[1] + (sink.position[1] - source.position[1]) * ratio
            ideal_pos = (ideal_x, ideal_y)
            
            # Try to find or create relay at this position
            relay_id = self._find_or_create_relay(ideal_pos, span_limit)
            if relay_id:
                path_ids.append(relay_id)
            else:
                # Failed to place relay - log warning and continue with partial path
                self.diagnostics.warning(
                    f"Could not place relay {i}/{num_relays} at ({ideal_x:.1f}, {ideal_y:.1f}) "
                    f"for path {source.ir_node_id} -> {sink.ir_node_id}"
                )
        
        path_ids.append(sink.ir_node_id)
        
        # Validate that all segments are within span
        for i in range(len(path_ids) - 1):
            start_placement = self.layout_plan.get_placement(path_ids[i])
            end_placement = self.layout_plan.get_placement(path_ids[i + 1])
            
            if start_placement and end_placement:
                segment_dist = math.dist(start_placement.position, end_placement.position)
                if segment_dist > span_limit + epsilon:
                    self.diagnostics.warning(
                        f"Relay path segment exceeds span limit: {segment_dist:.1f} > {span_limit:.1f} "
                        f"({path_ids[i]} -> {path_ids[i + 1]})"
                    )
        
        return path_ids

    def _find_or_create_relay(
        self,
        ideal_position: Tuple[float, float],
        span_limit: float,
    ) -> Optional[str]:
        """Find existing relay near position or create new one.
        
        Strategy:
        1. Snap ideal position to grid
        2. Search for existing relays within reuse_radius
        3. If found, validate it doesn't exceed span to neighbors
        4. If not found, try to reserve grid position
        5. Create new relay at reserved position
        
        Args:
            ideal_position: Desired (x, y) position for relay
            span_limit: Maximum wire span distance
            
        Returns:
            Entity ID of found or created relay, or None if placement failed
        """
        # Snap to grid
        snapped_pos = self.layout_engine.snap_to_grid(ideal_position)
        
        # Define reuse radius (existing relays within this distance can be reused)
        reuse_radius = span_limit * 0.3  # 30% of span limit
        
        # Search for existing relay within reuse radius
        existing_relay = self._find_existing_relay_near(snapped_pos, reuse_radius)
        if existing_relay:
            return existing_relay
        
        # Try to reserve the snapped position
        reserved_pos = self.layout_engine.reserve_exact(
            snapped_pos,
            footprint=(1, 1),
            padding=0,
        )
        
        if reserved_pos:
            return self._create_intermediate_relay(reserved_pos)
        
        # Try nearby positions in a small radius
        search_radius = 3  # Search up to 3 grid positions away
        for radius in range(1, search_radius + 1):
            for offset_x in range(-radius, radius + 1):
                for offset_y in range(-radius, radius + 1):
                    # Only check positions at current radius (not interior)
                    if abs(offset_x) != radius and abs(offset_y) != radius:
                        continue
                    
                    candidate_x = snapped_pos[0] + offset_x * self.layout_engine.entity_spacing
                    candidate_y = snapped_pos[1] + offset_y * self.layout_engine.row_height
                    candidate_pos = (candidate_x, candidate_y)
                    
                    reserved_pos = self.layout_engine.reserve_exact(
                        candidate_pos,
                        footprint=(1, 1),
                        padding=0,
                    )
                    
                    if reserved_pos:
                        return self._create_intermediate_relay(reserved_pos)
        
        # Could not place relay
        return None

    def _find_existing_relay_near(
        self,
        position: Tuple[int, int],
        max_distance: float,
    ) -> Optional[str]:
        """Find an existing wire relay within max_distance of position.
        
        Args:
            position: Center position to search from
            max_distance: Maximum distance to search
            
        Returns:
            Entity ID of nearest relay, or None if none found within range
        """
        best_relay_id = None
        best_distance = float('inf')
        
        for entity_id, placement in self.layout_plan.entity_placements.items():
            # Only consider wire relays
            if getattr(placement, "role", None) != "wire_relay":
                continue
            
            # Calculate distance
            distance = math.dist(position, placement.position)
            
            # Check if within range and closer than current best
            if distance <= max_distance and distance < best_distance:
                best_relay_id = entity_id
                best_distance = distance
        
        return best_relay_id

    def _create_intermediate_relay(
        self,
        position: Tuple[int, int],
    ) -> str:
        """Create a wire relay power pole at the specified position.
        
        Args:
            position: Grid-aligned (x, y) position
            
        Returns:
            Entity ID of the created relay
        """
        relay_id = f"__wire_relay_{self._relay_counter}"
        self._relay_counter += 1

        relay_placement = EntityPlacement(
            ir_node_id=relay_id,
            entity_type="medium-electric-pole",
            position=position,
            properties={
                "debug_info": {
                    "variable": f"relay_{self._relay_counter}",
                    "operation": "infrastructure",
                    "details": f"wire_relay at ({position[0]}, {position[1]})",
                    "role": "wire_relay",
                }
            },
            role="wire_relay",
            zone="infrastructure",
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
