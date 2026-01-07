from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

"""Wire routing and color assignment algorithms."""


WIRE_COLORS: tuple[str, str] = ("red", "green")


@dataclass(frozen=True)
class CircuitEdge:
    """Represents a physical source→sink wiring requirement."""

    logical_signal_id: str
    resolved_signal_name: str
    source_entity_id: str | None
    sink_entity_id: str
    source_entity_type: str | None = None
    sink_entity_type: str | None = None
    sink_role: str | None = None
    originating_merge_id: str | None = None  # Track which merge this edge came from


@dataclass
class ConflictEdge:
    """Edge between two conflict nodes that must not share a wire color."""

    nodes: tuple[tuple[str, str], tuple[str, str]]
    sinks: set[str] = field(default_factory=set)


@dataclass
class ColoringResult:
    assignments: dict[tuple[str, str], str]
    conflicts: list[ConflictEdge]
    is_bipartite: bool


def _resolve_entity_type(placement: Any) -> str | None:
    """Best-effort extraction of an entity type from a placement object."""

    if placement is None:
        return None

    entity_type = getattr(placement, "entity_type", None)
    if entity_type:
        return str(entity_type)

    entity = getattr(placement, "entity", None)
    if entity is not None:
        return type(entity).__name__

    proto = getattr(placement, "prototype", None)
    if proto:
        return str(proto)

    return None


def collect_circuit_edges(
    signal_graph: Any,
    signal_usage: dict[str, Any],
    entities: dict[str, Any],
) -> list[CircuitEdge]:
    """Compute all source→sink edges with resolved signal metadata."""

    edges: list[CircuitEdge] = []

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

        source_entity_type: str | None = None
        sink_entity_type: str | None = None
        sink_role: str | None = None

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


def plan_wire_colors(
    edges: Sequence[CircuitEdge],
    locked_colors: dict[tuple[str, str], str] | None = None,
) -> ColoringResult:
    """Assign red/green colors to signal sources using conflict-aware coloring."""

    locked = locked_colors or {}

    graph: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    edge_sinks: dict[tuple[tuple[str, str], tuple[str, str]], set[str]] = defaultdict(set)

    # Ensure all nodes appear in the graph even if conflict-free
    for edge in edges:
        if not edge.source_entity_id:
            continue
        node_key = (edge.source_entity_id, edge.resolved_signal_name)
        graph.setdefault(node_key, set())

    # Group edges by (sink_id, resolved_signal_name)
    # Each group entry is (node_key, originating_merge_id)
    sink_groups: dict[tuple[str, str], list[tuple[tuple[str, str], str | None]]] = defaultdict(list)
    for edge in edges:
        if not edge.source_entity_id:
            continue
        node_key = (edge.source_entity_id, edge.resolved_signal_name)
        sink_groups[(edge.sink_entity_id, edge.resolved_signal_name)].append(
            (node_key, edge.originating_merge_id)
        )

    # Sort for deterministic iteration order
    for (sink_id, _resolved_name), nodes_with_merge in sorted(sink_groups.items()):
        # Deduplicate by node_key, keeping first occurrence
        seen_nodes = {}
        for node_key, merge_id in nodes_with_merge:
            if node_key not in seen_nodes:
                seen_nodes[node_key] = merge_id

        unique_entries = list(seen_nodes.items())
        if len(unique_entries) <= 1:
            continue

        # Only create conflict edges between nodes from DIFFERENT merges
        # Nodes with the same originating_merge_id are intentionally merging
        for idx in range(len(unique_entries)):
            a, merge_a = unique_entries[idx]
            for jdx in range(idx + 1, len(unique_entries)):
                b, merge_b = unique_entries[jdx]
                if a == b:
                    continue

                # If both edges come from the same merge (or both have no merge),
                # they should be on the same wire - no conflict edge needed
                if merge_a is not None and merge_a == merge_b:
                    continue

                # Different merges or mixed merge/non-merge: potential conflict
                graph[a].add(b)
                graph[b].add(a)
                sorted_pair = sorted((a, b))
                pair: tuple[tuple[str, str], tuple[str, str]] = (sorted_pair[0], sorted_pair[1])
                edge_sinks[pair].add(sink_id)

    assignments: dict[tuple[str, str], str] = {}
    conflicts: list[ConflictEdge] = []
    conflict_pairs_recorded: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    is_bipartite = True

    pending_nodes = set(graph.keys()) | set(locked.keys())

    # Sort for deterministic iteration order
    for start_node in sorted(pending_nodes):
        if start_node in assignments:
            continue

        start_color = locked.get(start_node, WIRE_COLORS[0])
        queue: deque[tuple[tuple[str, str], str]] = deque()
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

            opposite_color = WIRE_COLORS[1] if desired_color == WIRE_COLORS[0] else WIRE_COLORS[0]

            # Sort neighbors for deterministic iteration order
            for neighbor in sorted(neighbors):
                neighbor_locked = locked.get(neighbor)
                neighbor_desired = neighbor_locked or opposite_color

                neighbor_existing = assignments.get(neighbor)
                if neighbor_existing:
                    if neighbor_existing != neighbor_desired:
                        is_bipartite = False
                        sorted_pair3 = sorted((node, neighbor))
                        pair3: tuple[tuple[str, str], tuple[str, str]] = (
                            sorted_pair3[0],
                            sorted_pair3[1],
                        )
                        if pair3 not in conflict_pairs_recorded:
                            conflict_pairs_recorded.add(pair3)
                            sinks = edge_sinks.get(pair3, set())
                            conflicts.append(ConflictEdge(nodes=pair3, sinks=set(sinks)))
                    continue

                if neighbor_locked and neighbor_locked == desired_color:
                    is_bipartite = False
                    sorted_pair4 = sorted((node, neighbor))
                    pair4: tuple[tuple[str, str], tuple[str, str]] = (
                        sorted_pair4[0],
                        sorted_pair4[1],
                    )
                    if pair4 not in conflict_pairs_recorded:
                        conflict_pairs_recorded.add(pair4)
                        sinks = edge_sinks.get(pair4, set())
                        conflicts.append(ConflictEdge(nodes=pair4, sinks=set(sinks)))

                queue.append((neighbor, neighbor_desired))

    return ColoringResult(assignments=assignments, conflicts=conflicts, is_bipartite=is_bipartite)


def detect_merge_color_conflicts(
    merge_membership: dict[str, set[str]],
    signal_graph: Any,
) -> dict[tuple[str, str], str]:
    """Detect paths that need locked colors due to merge conflicts.

    When a signal source participates in multiple independent wire merges
    that both connect to the same final sink, they must use different wire colors
    to prevent double-counting.

    Args:
        merge_membership: Maps source_id -> set of merge_ids the source belongs to
        signal_graph: Signal graph for finding downstream sinks

    Returns:
        Dict mapping (source_id, merge_id) -> locked color
    """
    from itertools import combinations

    locked_colors: dict[tuple[str, str], str] = {}

    # For each source that's in multiple merges
    for source_id, merge_ids in merge_membership.items():
        if len(merge_ids) <= 1:
            continue

        # Check each pair of merges containing this source
        for merge_a, merge_b in combinations(sorted(merge_ids), 2):
            # Get sinks that receive from each merge
            sinks_a = set(signal_graph.get_sinks(merge_a)) if signal_graph else set()
            sinks_b = set(signal_graph.get_sinks(merge_b)) if signal_graph else set()

            # If both merges connect to the same sink, need different colors
            common_sinks = sinks_a & sinks_b
            if common_sinks:
                # Lock merge_a to red, merge_b to green
                # Use (merge_id, source_id) as key since that's what affects the wire color
                locked_colors[(merge_a, source_id)] = "red"
                locked_colors[(merge_b, source_id)] = "green"

    return locked_colors
