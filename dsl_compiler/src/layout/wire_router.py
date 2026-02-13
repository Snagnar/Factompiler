"""Wire color assignment via edge-level constraint solving.

Replaces the old node-level bipartite graph coloring with an edge-level
constraint model.  Every wiring requirement is expressed as a WireEdge,
and correctness rules are expressed as constraints on those edges.

The solver uses union-find to merge edges that must share a color (merge
constraints), then BFS 2-coloring on the contracted constraint graph.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

WIRE_COLORS: tuple[str, str] = ("red", "green")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WireEdge:
    """A logical wiring requirement between two entities."""

    source_entity_id: str
    sink_entity_id: str
    signal_name: str  # Resolved Factorio signal name
    logical_signal_id: str  # IR-level signal ID (for tracing)
    merge_group: str | None = None  # If part of a wire merge

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source_entity_id, self.sink_entity_id, self.signal_name)


@dataclass
class SeparationConstraint:
    """Two edges that MUST use different colors at the same sink."""

    edge_a: WireEdge
    edge_b: WireEdge
    reason: str


@dataclass
class MergeConstraint:
    """A set of edges that MUST share the same wire color."""

    edges: list[WireEdge]
    merge_id: str


@dataclass
class ColorAssignment:
    """Result of the wire color solver."""

    edge_colors: dict[WireEdge, str]
    is_bipartite: bool
    conflicts: list[SeparationConstraint]  # unresolvable conflicts


# ---------------------------------------------------------------------------
# Union-Find for merge groups
# ---------------------------------------------------------------------------


class _UnionFind:
    """Simple union-find over WireEdge instances."""

    def __init__(self) -> None:
        self._parent: dict[WireEdge, WireEdge] = {}
        self._rank: dict[WireEdge, int] = {}

    def make_set(self, edge: WireEdge) -> None:
        if edge not in self._parent:
            self._parent[edge] = edge
            self._rank[edge] = 0

    def find(self, edge: WireEdge) -> WireEdge:
        root = edge
        while self._parent[root] is not root:
            root = self._parent[root]
        # Path compression
        while self._parent[edge] is not root:
            self._parent[edge], edge = root, self._parent[edge]
        return root

    def union(self, a: WireEdge, b: WireEdge) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra is rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


# ---------------------------------------------------------------------------
# WireColorSolver
# ---------------------------------------------------------------------------


class WireColorSolver:
    """Constraint-based wire color solver.

    Usage::

        solver = WireColorSolver()
        solver.add_edge(edge1)
        solver.add_edge(edge2)
        solver.add_hard_constraint(edge1, "red", "memory data")
        solver.add_separation(edge1, edge2, "same signal same sink")
        solver.add_merge([edge3, edge4], "merge_42")
        result = solver.solve()
        # result.edge_colors maps each WireEdge → "red" | "green"
    """

    def __init__(self) -> None:
        self._edges: list[WireEdge] = []
        self._edge_set: set[WireEdge] = set()
        self._hard: dict[WireEdge, tuple[str, str]] = {}  # edge → (color, reason)
        self._separations: list[SeparationConstraint] = []
        self._merges: list[MergeConstraint] = []

    # -- Building API -------------------------------------------------------

    def add_edge(self, edge: WireEdge) -> None:
        if edge not in self._edge_set:
            self._edges.append(edge)
            self._edge_set.add(edge)

    def add_hard_constraint(self, edge: WireEdge, color: str, reason: str) -> None:
        if edge not in self._hard:
            self._hard[edge] = (color, reason)

    def add_separation(self, edge_a: WireEdge, edge_b: WireEdge, reason: str) -> None:
        self._separations.append(SeparationConstraint(edge_a, edge_b, reason))

    def add_merge(self, edges: list[WireEdge], merge_id: str) -> None:
        if len(edges) >= 2:
            self._merges.append(MergeConstraint(edges, merge_id))

    # -- Solving ------------------------------------------------------------

    def solve(self) -> ColorAssignment:
        """Solve wire color assignment.

        Algorithm:
        1. Initialize union-find with all edges.
        2. Merge all edges in the same merge group.
        3. Propagate hard constraints to representatives.
        4. Build contracted conflict graph from separation constraints.
        5. BFS 2-color the contracted graph.
        """
        if not self._edges:
            return ColorAssignment(edge_colors={}, is_bipartite=True, conflicts=[])

        # Step 1: Union-find
        uf = _UnionFind()
        for edge in self._edges:
            uf.make_set(edge)

        # Step 2: Union merge groups
        for mc in self._merges:
            anchor = mc.edges[0]
            for other in mc.edges[1:]:
                uf.union(anchor, other)

        # Step 3: Propagate hard constraints to representative edges
        rep_color: dict[WireEdge, str] = {}
        for edge, (color, _reason) in self._hard.items():
            rep = uf.find(edge)
            existing = rep_color.get(rep)
            if existing is None or existing == color:
                rep_color[rep] = color
            # else: conflicting hard constraints inside a merge group — keep first

        # Step 4: Build contracted conflict graph
        adj: dict[WireEdge, set[WireEdge]] = defaultdict(set)
        contracted_separations: list[tuple[WireEdge, WireEdge, SeparationConstraint]] = []

        for sep in self._separations:
            ra = uf.find(sep.edge_a)
            rb = uf.find(sep.edge_b)
            if ra is rb:
                continue  # Both edges are in same merge group; unresolvable
            adj[ra].add(rb)
            adj[rb].add(ra)
            contracted_separations.append((ra, rb, sep))

        # Step 5: BFS 2-coloring on representatives
        assignment: dict[WireEdge, str] = {}
        is_bipartite = True

        def _sort_key(e: WireEdge) -> tuple[str, str, str]:
            return (e.source_entity_id, e.sink_entity_id, e.signal_name)

        all_reps = {uf.find(e) for e in self._edges}

        for start in sorted(all_reps, key=_sort_key):
            if start in assignment:
                continue

            start_color = rep_color.get(start, WIRE_COLORS[0])
            queue: deque[tuple[WireEdge, str]] = deque([(start, start_color)])

            while queue:
                node, desired = queue.popleft()

                locked = rep_color.get(node)
                if locked:
                    desired = locked

                existing = assignment.get(node)
                if existing is not None:
                    if existing != desired:
                        is_bipartite = False
                    continue

                assignment[node] = desired
                opposite = WIRE_COLORS[1] if desired == WIRE_COLORS[0] else WIRE_COLORS[0]

                for neighbor in sorted(adj.get(node, set()), key=_sort_key):
                    nb_locked = rep_color.get(neighbor)
                    nb_desired = nb_locked or opposite

                    nb_existing = assignment.get(neighbor)
                    if nb_existing is not None:
                        if nb_existing != nb_desired:
                            is_bipartite = False
                        continue

                    if nb_locked and nb_locked == desired:
                        is_bipartite = False

                    queue.append((neighbor, nb_desired))

        # Record unresolvable conflicts
        conflicts: list[SeparationConstraint] = []
        if not is_bipartite:
            for ra, rb, sep in contracted_separations:
                ca = assignment.get(ra)
                cb = assignment.get(rb)
                if ca and cb and ca == cb:
                    conflicts.append(sep)

        # Step 6: Map representatives back to all edges
        edge_colors: dict[WireEdge, str] = {}
        for edge in self._edges:
            rep = uf.find(edge)
            edge_colors[edge] = assignment.get(rep, WIRE_COLORS[0])

        return ColorAssignment(
            edge_colors=edge_colors,
            is_bipartite=is_bipartite,
            conflicts=conflicts,
        )
