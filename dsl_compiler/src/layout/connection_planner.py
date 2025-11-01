"""Connection planning for wire routing."""

from __future__ import annotations

import heapq
import math
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from dsl_compiler.src.ir import SignalRef
from dsl_compiler.src.semantic import DiagnosticCollector

from .layout_engine import LayoutEngine
from .layout_plan import LayoutPlan, WireConnection, EntityPlacement
from .signal_analyzer import SignalUsageEntry
from .wire_router import (
    CircuitEdge,
    WIRE_COLORS,
    collect_circuit_edges,
    plan_wire_colors,
)


class ConnectionPlanner:
    """Plans all wire connections for a blueprint."""

    def __init__(
        self,
        layout_plan: LayoutPlan,
        signal_usage: Dict[str, SignalUsageEntry],
        diagnostics: DiagnosticCollector,
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
            self.diagnostics.error(
                "Two-color routing could not isolate signal "
                f"'{resolved_signal}' across sinks [{sink_desc}]; falling back to single-channel wiring for involved entities ({source_desc})."
            )
        self._edge_color_map = {}

    def _populate_wire_connections(self) -> None:
        for edge in self._circuit_edges:
            if not edge.source_entity_id or not edge.sink_entity_id:
                continue

            color = self._edge_color_map.get(
                (edge.source_entity_id, edge.sink_entity_id, edge.resolved_signal_name)
            )
            if color is None:
                color = WIRE_COLORS[0]

            self._route_connection_with_relays(edge, color)

    # ------------------------------------------------------------------
    # Relay placement helpers
    # ------------------------------------------------------------------

    def _route_connection_with_relays(
        self,
        edge: CircuitEdge,
        wire_color: str,
    ) -> None:
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

        max_span = self.max_wire_span or 0.0
        if max_span <= 0:
            max_span = 9.0

        span_limit = max(1.0, float(max_span) - 1.8)
        path = self._build_relay_path(source, sink, span_limit)

        for start_id, end_id in zip(path, path[1:]):
            connection = WireConnection(
                source_entity_id=start_id,
                sink_entity_id=end_id,
                signal_name=edge.resolved_signal_name,
                wire_color=wire_color,
            )
            self.layout_plan.add_wire_connection(connection)

    def _build_relay_path(
        self,
        source: EntityPlacement,
        sink: EntityPlacement,
        span_limit: float,
    ) -> List[str]:
        ids: List[str] = [source.ir_node_id, sink.ir_node_id]
        epsilon = 1e-6
        base_span = math.dist(source.position, sink.position)
        base_segments = max(1, math.ceil(max(base_span, span_limit) / span_limit))
        relay_cap = max(4096, base_segments * 8)
        relays_inserted = 0

        idx = 0
        while idx < len(ids) - 1:
            start_id = ids[idx]
            end_id = ids[idx + 1]
            start = self.layout_plan.get_placement(start_id)
            end = self.layout_plan.get_placement(end_id)

            if start is None or end is None:
                idx += 1
                continue

            distance = math.dist(start.position, end.position)
            if distance <= span_limit + epsilon:
                idx += 1
                continue

            existing_relay_id = self._find_existing_relay_bridge(
                start,
                end,
                distance,
                span_limit,
                epsilon,
                ids,
            )

            if existing_relay_id is not None:
                ids.insert(idx + 1, existing_relay_id)
                continue

            relay_position = self._select_relay_position(
                start,
                end,
                distance,
                span_limit,
                epsilon,
            )

            if relay_position is None:
                bridge_sequence = self._find_segment_bridge(
                    start,
                    end,
                    span_limit,
                    epsilon,
                    ids,
                )

                if bridge_sequence:
                    new_relays = 0
                    insertion_offset = 1
                    for relay_id, pos in bridge_sequence:
                        current_id = relay_id
                        if current_id is None:
                            if relays_inserted + new_relays >= relay_cap:
                                break
                            reserved = self.layout_engine.reserve_exact(pos)
                            if reserved is None:
                                break
                            current_id = self._create_intermediate_relay(reserved)
                            new_relays += 1
                        ids.insert(idx + insertion_offset, current_id)
                        insertion_offset += 1

                    if insertion_offset > 1:
                        relays_inserted += new_relays
                        continue

                self.diagnostics.warning(
                    "Unable to insert relay between %s and %s; segment spans %.2f tiles (limit %.2f)."
                    % (start_id, end_id, distance, span_limit)
                )
                idx += 1
                continue

            relay_id = self._create_intermediate_relay(relay_position)
            ids.insert(idx + 1, relay_id)
            relays_inserted += 1

            if relays_inserted > relay_cap:
                self.diagnostics.warning(
                    "Relay insertion exceeded safety cap while routing %s -> %s; inserted %d relays (cap %d)."
                    % (
                        source.ir_node_id,
                        sink.ir_node_id,
                        relays_inserted,
                        relay_cap,
                    )
                )
                break

        return ids

    def _find_existing_relay_bridge(
        self,
        start: EntityPlacement,
        end: EntityPlacement,
        distance: float,
        span_limit: float,
        epsilon: float,
        current_path: Sequence[str],
    ) -> Optional[str]:
        best_id: Optional[str] = None
        best_score = (float("inf"), float("inf"))

        for entity_id, placement in self.layout_plan.entity_placements.items():
            if getattr(placement, "role", None) != "wire_relay":
                continue
            if entity_id in current_path:
                continue
            if entity_id == start.ir_node_id or entity_id == end.ir_node_id:
                continue

            start_distance = math.dist(start.position, placement.position)
            end_distance = math.dist(placement.position, end.position)

            if (
                start_distance >= distance - epsilon
                or end_distance >= distance - epsilon
            ):
                continue

            penalty = 0.0
            if start_distance > span_limit + epsilon:
                penalty += start_distance - span_limit
            if end_distance > span_limit + epsilon:
                penalty += end_distance - span_limit

            span_score = max(start_distance, end_distance)
            composite = (penalty, span_score)
            if composite >= best_score:
                continue

            best_id = entity_id
            best_score = composite

        return best_id

    def _find_segment_bridge(
        self,
        start: EntityPlacement,
        end: EntityPlacement,
        span_limit: float,
        epsilon: float,
        current_path: Sequence[str],
    ) -> Optional[List[Tuple[Optional[str], Tuple[int, int]]]]:
        spacing_x = max(1, self.layout_engine.entity_spacing)
        spacing_y = max(1, self.layout_engine.row_height)
        dx = abs(start.position[0] - end.position[0])
        dy = abs(start.position[1] - end.position[1])
        margin = int(math.ceil(span_limit * 4 + max(dx, dy)))

        min_x = (
            (min(start.position[0], end.position[0]) - margin) // spacing_x
        ) * spacing_x
        max_x = (
            (max(start.position[0], end.position[0]) + margin) // spacing_x
        ) * spacing_x
        min_y = (
            (min(start.position[1], end.position[1]) - margin) // spacing_y
        ) * spacing_y
        max_y = (
            (max(start.position[1], end.position[1]) + margin) // spacing_y
        ) * spacing_y

        nodes: List[Dict[str, Any]] = []
        index_by_pos: Dict[Tuple[int, int], int] = {}

        def add_node(position: Tuple[int, int], relay_id: Optional[str]) -> int:
            existing = index_by_pos.get(position)
            if existing is not None:
                if relay_id is not None:
                    nodes[existing]["relay_id"] = relay_id
                return existing
            index = len(nodes)
            nodes.append({"position": position, "relay_id": relay_id})
            index_by_pos[position] = index
            return index

        start_idx = add_node(start.position, start.ir_node_id)
        end_idx = add_node(end.position, end.ir_node_id)

        for entity_id, placement in self.layout_plan.entity_placements.items():
            if getattr(placement, "role", None) != "wire_relay":
                continue
            if entity_id in current_path:
                continue
            add_node(placement.position, entity_id)

        max_nodes = 2500
        for x in range(min_x, max_x + spacing_x, spacing_x):
            if len(nodes) >= max_nodes:
                break
            for y in range(min_y, max_y + spacing_y, spacing_y):
                if len(nodes) >= max_nodes:
                    break
                pos = (x, y)
                if pos in index_by_pos:
                    continue
                if not self.layout_engine.can_reserve(pos):
                    continue
                add_node(pos, None)

        if len(nodes) <= 2:
            return None

        parents: Dict[int, Optional[int]] = {start_idx: None}
        queue: deque[int] = deque([start_idx])

        while queue:
            current = queue.popleft()
            current_pos = nodes[current]["position"]

            for idx, node in enumerate(nodes):
                if idx in parents:
                    continue
                if math.dist(current_pos, node["position"]) > span_limit + epsilon:
                    continue
                parents[idx] = current
                if idx == end_idx:
                    queue.clear()
                    break
                queue.append(idx)

            if end_idx in parents:
                break

        if end_idx not in parents:
            return None

        path_indices: List[int] = []
        node_idx: Optional[int] = end_idx
        while node_idx is not None:
            path_indices.append(node_idx)
            node_idx = parents.get(node_idx)

        path_indices.reverse()
        if len(path_indices) <= 2:
            return None

        sequence: List[Tuple[Optional[str], Tuple[int, int]]] = []
        for idx in path_indices[1:-1]:
            node = nodes[idx]
            sequence.append((node.get("relay_id"), node["position"]))

        return sequence

    def _select_relay_position(
        self,
        start: EntityPlacement,
        end: EntityPlacement,
        distance: float,
        span_limit: float,
        epsilon: float,
    ) -> Optional[Tuple[int, int]]:
        spacing_x = max(1, self.layout_engine.entity_spacing)
        spacing_y = max(1, self.layout_engine.row_height)
        min_spacing = max(1, min(spacing_x, spacing_y))
        span_radius = math.ceil(span_limit / min_spacing) + 2
        distance_radius = math.ceil(distance / min_spacing) + 2
        search_radius = min(64, max(span_radius, distance_radius))

        candidate_ratios = self._candidate_relay_ratios(distance, span_limit)
        seen_positions: Set[Tuple[int, int]] = set()

        for ratio in candidate_ratios:
            candidate = self._locate_relay_position(
                start.position,
                end.position,
                ratio,
                search_radius,
            )

            if candidate is None:
                continue
            if candidate == start.position or candidate == end.position:
                continue
            if candidate in seen_positions:
                continue

            new_start = math.dist(start.position, candidate)
            new_end = math.dist(candidate, end.position)

            if new_start >= distance - epsilon or new_end >= distance - epsilon:
                continue

            reserved = self.layout_engine.reserve_exact(candidate)
            if reserved is None:
                continue

            seen_positions.add(reserved)
            return reserved

        step_candidates = self._scan_step_candidates(
            start.position,
            end.position,
            span_limit,
            distance,
            epsilon,
        )

        if not step_candidates:
            step_candidates = self._scan_step_candidates(
                end.position,
                start.position,
                span_limit,
                distance,
                epsilon,
            )

        for _, _, candidate in step_candidates:
            if candidate == start.position or candidate == end.position:
                continue
            if candidate in seen_positions:
                continue
            reserved = self.layout_engine.reserve_exact(candidate)
            if reserved is None:
                continue
            seen_positions.add(reserved)
            return reserved

        return None

    def _candidate_relay_ratios(
        self, distance: float, span_limit: float
    ) -> List[float]:
        ratios: List[float] = []
        pivot_values = [0.5, 0.25, 0.75, 0.125, 0.875, 0.375, 0.625]

        if distance > 0:
            offset = span_limit / distance
            if 0.0 < offset < 1.0:
                pivot_values.extend([offset, 1.0 - offset])

        seen = set()
        for value in pivot_values:
            if value <= 0.0 or value >= 1.0:
                continue
            key = round(value, 6)
            if key in seen:
                continue
            seen.add(key)
            ratios.append(value)

        return ratios

    def _locate_relay_position(
        self,
        start_pos: Tuple[int, int],
        end_pos: Tuple[int, int],
        ratio: float,
        search_radius: int,
    ) -> Optional[Tuple[int, int]]:
        ratio = min(max(ratio, 1e-6), 1 - 1e-6)
        target_x = start_pos[0] + (end_pos[0] - start_pos[0]) * ratio
        target_y = start_pos[1] + (end_pos[1] - start_pos[1]) * ratio
        return self._find_available_near((target_x, target_y), search_radius)

    def _find_available_near(
        self,
        target: Tuple[float, float],
        search_radius: int,
    ) -> Optional[Tuple[int, int]]:
        snapped = self.layout_engine.snap_to_grid(target)
        spacing_x = max(1, self.layout_engine.entity_spacing)
        spacing_y = max(1, self.layout_engine.row_height)

        visited: Set[Tuple[int, int]] = set()
        heap: List[Tuple[float, int, int, Tuple[int, int]]] = []

        def push(pos: Tuple[int, int]) -> None:
            if pos in visited:
                return
            visited.add(pos)
            dx = abs(pos[0] - snapped[0]) / spacing_x
            dy = abs(pos[1] - snapped[1]) / spacing_y
            distance = max(dx, dy)
            heapq.heappush(
                heap,
                (
                    distance,
                    abs(pos[1] - snapped[1]),
                    abs(pos[0] - snapped[0]),
                    pos,
                ),
            )

        push(snapped)

        while heap:
            distance, _, _, pos = heapq.heappop(heap)
            if distance > search_radius:
                break
            if self.layout_engine.can_reserve(pos):
                return pos
            for neighbor in self._iter_neighbor_positions(pos):
                push(neighbor)

        return None

    def _iter_neighbor_positions(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        spacing_x = max(1, self.layout_engine.entity_spacing)
        spacing_y = max(1, self.layout_engine.row_height)
        x, y = pos
        offsets = (
            (spacing_x, 0),
            (-spacing_x, 0),
            (0, spacing_y),
            (0, -spacing_y),
            (spacing_x, spacing_y),
            (spacing_x, -spacing_y),
            (-spacing_x, spacing_y),
            (-spacing_x, -spacing_y),
        )
        return [(x + dx, y + dy) for dx, dy in offsets]

    def _scan_step_candidates(
        self,
        origin: Tuple[int, int],
        target: Tuple[int, int],
        span_limit: float,
        original_distance: float,
        epsilon: float,
    ) -> List[Tuple[float, float, Tuple[int, int]]]:
        spacing_x = max(1, self.layout_engine.entity_spacing)
        spacing_y = max(1, self.layout_engine.row_height)
        min_spacing = max(1, min(spacing_x, spacing_y))
        tile_radius = math.ceil(span_limit / min_spacing) + 2

        visited: Set[Tuple[int, int]] = set()
        heap: List[Tuple[float, int, int, Tuple[int, int]]] = []
        candidates: List[Tuple[float, float, Tuple[int, int]]] = []

        def push(pos: Tuple[int, int]) -> None:
            if pos in visited:
                return
            visited.add(pos)
            dx = abs(pos[0] - origin[0]) / spacing_x
            dy = abs(pos[1] - origin[1]) / spacing_y
            priority = max(dx, dy)
            heapq.heappush(
                heap,
                (
                    priority,
                    abs(pos[1] - origin[1]),
                    abs(pos[0] - origin[0]),
                    pos,
                ),
            )

        push(origin)

        while heap:
            priority, _, _, pos = heapq.heappop(heap)
            if priority > tile_radius:
                continue

            if pos != origin:
                origin_distance = math.dist(origin, pos)
                if origin_distance > span_limit + epsilon:
                    continue
                if not self.layout_engine.can_reserve(pos):
                    # Explore neighbors even if occupied to look for free spots nearby.
                    for neighbor in self._iter_neighbor_positions(pos):
                        push(neighbor)
                    continue
                remaining = math.dist(pos, target)
                if remaining >= original_distance - epsilon:
                    for neighbor in self._iter_neighbor_positions(pos):
                        push(neighbor)
                    continue

                candidates.append((remaining, origin_distance, pos))

            for neighbor in self._iter_neighbor_positions(pos):
                push(neighbor)

        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates

    def _create_intermediate_relay(
        self,
        position: Tuple[int, int],
    ) -> str:
        relay_id = f"__wire_relay_{self._relay_counter}"
        self._relay_counter += 1

        relay_placement = EntityPlacement(
            ir_node_id=relay_id,
            entity_type="medium-electric-pole",
            position=position,
            properties={},
            role="wire_relay",
            zone="infrastructure",
        )
        self.layout_plan.add_placement(relay_placement)
        return relay_id

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
