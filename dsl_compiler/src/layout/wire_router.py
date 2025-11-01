from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

"""Wire routing and color assignment algorithms."""


WIRE_COLORS: Tuple[str, str] = ("red", "green")


@dataclass(frozen=True)
class CircuitEdge:
    """Represents a physical source→sink wiring requirement."""

    logical_signal_id: str
    resolved_signal_name: str
    source_entity_id: Optional[str]
    sink_entity_id: str
    source_entity_type: Optional[str] = None
    sink_entity_type: Optional[str] = None
    sink_role: Optional[str] = None


@dataclass
class ConflictNode:
    """A node in the conflict graph representing a source entity for a signal."""

    node_key: Tuple[str, str]
    edges: Set[str] = field(default_factory=set)
    locked_color: Optional[str] = None


@dataclass
class ConflictEdge:
    """Edge between two conflict nodes that must not share a wire color."""

    nodes: Tuple[Tuple[str, str], Tuple[str, str]]
    sinks: Set[str] = field(default_factory=set)


@dataclass
class ColoringResult:
    assignments: Dict[Tuple[str, str], str]
    conflicts: List[ConflictEdge]
    is_bipartite: bool


def _resolve_entity_type(placement: Any) -> Optional[str]:
    """Best-effort extraction of an entity type from a placement object."""

    if placement is None:
        return None

    entity_type = getattr(placement, "entity_type", None)
    if entity_type:
        return entity_type

    entity = getattr(placement, "entity", None)
    if entity is not None:
        return type(entity).__name__

    proto = getattr(placement, "prototype", None)
    if proto:
        return str(proto)

    return None


def collect_circuit_edges(
    signal_graph: Any,
    signal_usage: Dict[str, Any],
    entities: Dict[str, Any],
) -> List[CircuitEdge]:
    """Compute all source→sink edges with resolved signal metadata."""

    edges: List[CircuitEdge] = []

    for (
        logical_id,
        source_entity_id,
        sink_entity_id,
    ) in signal_graph.iter_source_sink_pairs():
        usage_entry = signal_usage.get(logical_id)
        resolved_signal_name = (
            usage_entry.resolved_signal_name
            if usage_entry and usage_entry.resolved_signal_name
            else logical_id
        )

        source_entity_type: Optional[str] = None
        sink_entity_type: Optional[str] = None
        sink_role: Optional[str] = None

        if source_entity_id:
            source_placement = entities.get(source_entity_id)
            source_entity_type = _resolve_entity_type(source_placement)

        sink_placement = entities.get(sink_entity_id)
        if sink_placement is not None:
            sink_entity_type = _resolve_entity_type(sink_placement)
            sink_role = getattr(sink_placement, "role", None)

        if sink_role is None and sink_entity_id.endswith("_export_anchor"):
            sink_role = "export"

        edges.append(
            CircuitEdge(
                logical_signal_id=logical_id,
                resolved_signal_name=resolved_signal_name,
                source_entity_id=source_entity_id,
                sink_entity_id=sink_entity_id,
                source_entity_type=source_entity_type,
                sink_entity_type=sink_entity_type,
                sink_role=sink_role,
            )
        )

    return edges


def detect_multi_source_conflicts(edges: Sequence[CircuitEdge]) -> List[ConflictEdge]:
    """Identify sinks that receive the same resolved signal from multiple sources."""

    conflicts_by_pair: Dict[Tuple[str, str], ConflictEdge] = {}
    sinks_by_name: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for edge in edges:
        if edge.source_entity_id is None:
            continue
        key = (edge.sink_entity_id, edge.resolved_signal_name)
        sinks_by_name[key].add(edge.source_entity_id)

    for (sink_id, resolved_name), sources in sinks_by_name.items():
        if len(sources) <= 1:
            continue
        sources_list = list(sources)
        for idx in range(len(sources_list)):
            for jdx in range(idx + 1, len(sources_list)):
                pair = tuple(
                    sorted(
                        [
                            (sources_list[idx], resolved_name),
                            (sources_list[jdx], resolved_name),
                        ]
                    )
                )
                conflict = conflicts_by_pair.get(pair)
                if not conflict:
                    conflict = ConflictEdge(nodes=pair)  # type: ignore[arg-type]
                    conflicts_by_pair[pair] = conflict
                conflict.sinks.add(sink_id)

    return list(conflicts_by_pair.values())


def plan_wire_colors(
    edges: Sequence[CircuitEdge],
    locked_colors: Optional[Dict[Tuple[str, str], str]] = None,
) -> ColoringResult:
    """Assign red/green colors to signal sources using conflict-aware coloring."""

    locked = locked_colors or {}

    graph: Dict[Tuple[str, str], Set[Tuple[str, str]]] = defaultdict(set)
    edge_sinks: Dict[Tuple[Tuple[str, str], Tuple[str, str]], Set[str]] = defaultdict(
        set
    )

    # Ensure all nodes appear in the graph even if conflict-free
    for edge in edges:
        if not edge.source_entity_id:
            continue
        node_key = (edge.source_entity_id, edge.resolved_signal_name)
        graph.setdefault(node_key, set())

    # Build conflict adjacency by grouping sink inputs that share the same signal name
    sink_groups: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
    for edge in edges:
        if not edge.source_entity_id:
            continue
        node_key = (edge.source_entity_id, edge.resolved_signal_name)
        sink_groups[(edge.sink_entity_id, edge.resolved_signal_name)].append(node_key)

    for (sink_id, resolved_name), nodes in sink_groups.items():
        unique_nodes = list(dict.fromkeys(nodes))
        if len(unique_nodes) <= 1:
            continue

        for idx in range(len(unique_nodes)):
            for jdx in range(idx + 1, len(unique_nodes)):
                a = unique_nodes[idx]
                b = unique_nodes[jdx]
                if a == b:
                    continue
                graph[a].add(b)
                graph[b].add(a)
                pair = tuple(sorted((a, b)))
                edge_sinks[pair].add(sink_id)

    assignments: Dict[Tuple[str, str], str] = {}
    conflicts: List[ConflictEdge] = []
    conflict_pairs_recorded: Set[Tuple[Tuple[str, str], Tuple[str, str]]] = set()
    is_bipartite = True

    pending_nodes = set(graph.keys()) | set(locked.keys())

    for start_node in pending_nodes:
        if start_node in assignments:
            continue

        start_color = locked.get(start_node, WIRE_COLORS[0])
        queue: deque[Tuple[Tuple[str, str], str]] = deque()
        queue.append((start_node, start_color))

        while queue:
            node, desired_color = queue.popleft()

            locked_color = locked.get(node)
            if locked_color:
                desired_color = locked_color

            existing = assignments.get(node)
            if existing:
                if existing != desired_color:
                    is_bipartite = False
                continue

            assignments[node] = desired_color

            neighbors = graph.get(node, set())
            if not neighbors:
                continue

            opposite_color = (
                WIRE_COLORS[1] if desired_color == WIRE_COLORS[0] else WIRE_COLORS[0]
            )

            for neighbor in neighbors:
                neighbor_locked = locked.get(neighbor)
                neighbor_desired = neighbor_locked or opposite_color

                neighbor_existing = assignments.get(neighbor)
                if neighbor_existing:
                    if neighbor_existing != neighbor_desired:
                        is_bipartite = False
                        pair = tuple(sorted((node, neighbor)))
                        if pair not in conflict_pairs_recorded:
                            conflict_pairs_recorded.add(pair)
                            sinks = edge_sinks.get(pair, set())
                            conflicts.append(ConflictEdge(nodes=pair, sinks=set(sinks)))
                    continue

                if neighbor_locked and neighbor_locked == desired_color:
                    is_bipartite = False
                    pair = tuple(sorted((node, neighbor)))
                    if pair not in conflict_pairs_recorded:
                        conflict_pairs_recorded.add(pair)
                        sinks = edge_sinks.get(pair, set())
                        conflicts.append(ConflictEdge(nodes=pair, sinks=set(sinks)))

                queue.append((neighbor, neighbor_desired))

    return ColoringResult(
        assignments=assignments, conflicts=conflicts, is_bipartite=is_bipartite
    )
