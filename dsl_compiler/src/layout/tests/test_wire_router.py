"""Tests for layout/wire_router.py — constraint-based wire color solver."""

import pytest

from dsl_compiler.src.layout.wire_router import (
    ColorAssignment,
    MergeConstraint,
    SeparationConstraint,
    WireColorSolver,
    WireEdge,
    _UnionFind,
)


def edge(src: str, snk: str, sig: str, *, merge: str | None = None) -> WireEdge:
    """Shorthand WireEdge builder."""
    return WireEdge(
        source_entity_id=src,
        sink_entity_id=snk,
        signal_name=sig,
        logical_signal_id=f"{src}->{snk}",
        merge_group=merge,
    )


# ── WireEdge ──────────────────────────────────────────────────────────────


class TestWireEdge:
    def test_creation(self):
        e = WireEdge("s", "t", "signal-A", "id1")
        assert e.source_entity_id == "s"
        assert e.sink_entity_id == "t"
        assert e.signal_name == "signal-A"
        assert e.logical_signal_id == "id1"
        assert e.merge_group is None

    def test_key(self):
        e = edge("s", "t", "signal-A")
        assert e.key == ("s", "t", "signal-A")

    def test_frozen(self):
        e = edge("s", "t", "signal-A")
        with pytest.raises(AttributeError):
            e.source_entity_id = "other"  # type: ignore[misc]

    def test_equality_and_hashing(self):
        e1 = edge("s", "t", "signal-A")
        e2 = edge("s", "t", "signal-A")
        assert e1 == e2
        assert hash(e1) == hash(e2)
        assert len({e1, e2}) == 1

    def test_merge_group(self):
        e = edge("s", "t", "signal-A", merge="m1")
        assert e.merge_group == "m1"


# ── Constraint dataclasses ────────────────────────────────────────────────


class TestConstraints:
    def test_separation_constraint(self):
        a, b = edge("s1", "t", "signal-A"), edge("s2", "t", "signal-A")
        sc = SeparationConstraint(a, b, "test")
        assert sc.edge_a is a
        assert sc.edge_b is b
        assert sc.reason == "test"

    def test_merge_constraint(self):
        edges = [edge("s1", "t", "sig"), edge("s2", "t", "sig")]
        mc = MergeConstraint(edges, "merge_1")
        assert mc.edges == edges
        assert mc.merge_id == "merge_1"


# ── ColorAssignment ───────────────────────────────────────────────────────


class TestColorAssignment:
    def test_empty(self):
        ca = ColorAssignment(edge_colors={}, is_bipartite=True, conflicts=[])
        assert ca.edge_colors == {}
        assert ca.is_bipartite
        assert ca.conflicts == []


# ── _UnionFind ────────────────────────────────────────────────────────────


class TestUnionFind:
    def test_make_set_and_find(self):
        uf = _UnionFind()
        e = edge("s", "t", "sig")
        uf.make_set(e)
        assert uf.find(e) is e

    def test_union_and_find(self):
        uf = _UnionFind()
        a = edge("s1", "t", "sig")
        b = edge("s2", "t", "sig")
        uf.make_set(a)
        uf.make_set(b)
        uf.union(a, b)
        assert uf.find(a) is uf.find(b)

    def test_idempotent_make_set(self):
        uf = _UnionFind()
        e = edge("s", "t", "sig")
        uf.make_set(e)
        uf.make_set(e)
        assert uf.find(e) is e

    def test_three_way_union(self):
        uf = _UnionFind()
        a, b, c = (edge(f"s{i}", "t", "sig") for i in range(3))
        for x in (a, b, c):
            uf.make_set(x)
        uf.union(a, b)
        uf.union(b, c)
        assert uf.find(a) is uf.find(c)


# ── WireColorSolver ──────────────────────────────────────────────────────


class TestWireColorSolverBasic:
    def test_empty_solve(self):
        result = WireColorSolver().solve()
        assert result.edge_colors == {}
        assert result.is_bipartite

    def test_single_edge_defaults_red(self):
        s = WireColorSolver()
        e = edge("s", "t", "sig")
        s.add_edge(e)
        r = s.solve()
        assert r.edge_colors[e] == "red"
        assert r.is_bipartite

    def test_duplicate_add_edge_ignored(self):
        s = WireColorSolver()
        e = edge("s", "t", "sig")
        s.add_edge(e)
        s.add_edge(e)
        r = s.solve()
        assert len(r.edge_colors) == 1


class TestWireColorSolverHardConstraints:
    def test_hard_constraint_red(self):
        s = WireColorSolver()
        e = edge("s", "t", "sig")
        s.add_edge(e)
        s.add_hard_constraint(e, "red", "test")
        assert s.solve().edge_colors[e] == "red"

    def test_hard_constraint_green(self):
        s = WireColorSolver()
        e = edge("s", "t", "sig")
        s.add_edge(e)
        s.add_hard_constraint(e, "green", "test")
        assert s.solve().edge_colors[e] == "green"


class TestWireColorSolverSeparation:
    def test_two_edges_separated(self):
        s = WireColorSolver()
        a = edge("s1", "t", "sig")
        b = edge("s2", "t", "sig")
        s.add_edge(a)
        s.add_edge(b)
        s.add_separation(a, b, "conflict")
        r = s.solve()
        assert r.is_bipartite
        assert r.edge_colors[a] != r.edge_colors[b]

    def test_separation_respects_hard_constraint(self):
        s = WireColorSolver()
        a = edge("s1", "t", "sig")
        b = edge("s2", "t", "sig")
        s.add_edge(a)
        s.add_edge(b)
        s.add_hard_constraint(a, "green", "locked")
        s.add_separation(a, b, "conflict")
        r = s.solve()
        assert r.edge_colors[a] == "green"
        assert r.edge_colors[b] == "red"

    def test_three_way_conflict_not_bipartite(self):
        """Three mutual separations form an odd cycle → not bipartite."""
        s = WireColorSolver()
        a = edge("s1", "t", "sig")
        b = edge("s2", "t", "sig")
        c = edge("s3", "t", "sig")
        for e in (a, b, c):
            s.add_edge(e)
        s.add_separation(a, b, "AB")
        s.add_separation(b, c, "BC")
        s.add_separation(a, c, "AC")
        r = s.solve()
        assert r.is_bipartite is False
        assert len(r.conflicts) > 0


class TestWireColorSolverMerge:
    def test_merge_same_color(self):
        s = WireColorSolver()
        a = edge("s1", "t1", "sig", merge="m1")
        b = edge("s2", "t2", "sig", merge="m1")
        s.add_edge(a)
        s.add_edge(b)
        s.add_merge([a, b], "m1")
        r = s.solve()
        assert r.edge_colors[a] == r.edge_colors[b]

    def test_merge_propagates_hard_constraint(self):
        s = WireColorSolver()
        a = edge("s1", "t1", "sig", merge="m1")
        b = edge("s2", "t2", "sig", merge="m1")
        s.add_edge(a)
        s.add_edge(b)
        s.add_merge([a, b], "m1")
        s.add_hard_constraint(a, "green", "test")
        r = s.solve()
        assert r.edge_colors[a] == "green"
        assert r.edge_colors[b] == "green"

    def test_merge_plus_separation(self):
        """Merged pair separated from a third edge."""
        s = WireColorSolver()
        a = edge("s1", "t", "sig", merge="m1")
        b = edge("s2", "t", "sig", merge="m1")
        c = edge("s3", "t", "sig")
        for e in (a, b, c):
            s.add_edge(e)
        s.add_merge([a, b], "m1")
        s.add_separation(a, c, "conflict")
        r = s.solve()
        assert r.edge_colors[a] == r.edge_colors[b]  # merged
        assert r.edge_colors[a] != r.edge_colors[c]  # separated

    def test_single_edge_merge_ignored(self):
        """A merge group with fewer than 2 edges is a no-op."""
        s = WireColorSolver()
        a = edge("s", "t", "sig")
        s.add_edge(a)
        s.add_merge([a], "m1")
        r = s.solve()
        assert r.edge_colors[a] == "red"  # default

    def test_separation_within_merge_group_ignored(self):
        """Separation between two edges in same merge is unresolvable — solver proceeds."""
        s = WireColorSolver()
        a = edge("s1", "t", "sig", merge="m1")
        b = edge("s2", "t", "sig", merge="m1")
        s.add_edge(a)
        s.add_edge(b)
        s.add_merge([a, b], "m1")
        s.add_separation(a, b, "impossible")
        r = s.solve()
        # Both in same merge group, so same color regardless of separation
        assert r.edge_colors[a] == r.edge_colors[b]


class TestWireColorSolverComplex:
    def test_chain_of_separations(self):
        """A-B conflict, B-C conflict → A and C should be same color."""
        s = WireColorSolver()
        a = edge("s1", "t", "sig")
        b = edge("s2", "t", "sig")
        c = edge("s3", "t", "sig")
        for e in (a, b, c):
            s.add_edge(e)
        s.add_separation(a, b, "AB")
        s.add_separation(b, c, "BC")
        r = s.solve()
        assert r.is_bipartite
        assert r.edge_colors[a] != r.edge_colors[b]
        assert r.edge_colors[b] != r.edge_colors[c]
        assert r.edge_colors[a] == r.edge_colors[c]

    def test_disconnected_components(self):
        """Two independent groups get default coloring independently."""
        s = WireColorSolver()
        a = edge("s1", "t1", "sig1")
        b = edge("s2", "t1", "sig1")
        c = edge("s3", "t2", "sig2")
        d = edge("s4", "t2", "sig2")
        for e in (a, b, c, d):
            s.add_edge(e)
        s.add_separation(a, b, "group1")
        s.add_separation(c, d, "group2")
        r = s.solve()
        assert r.is_bipartite
        assert r.edge_colors[a] != r.edge_colors[b]
        assert r.edge_colors[c] != r.edge_colors[d]

    def test_hard_constraint_conflict_in_merge(self):
        """Conflicting hard constraints within a merge — first wins."""
        s = WireColorSolver()
        a = edge("s1", "t", "sig", merge="m1")
        b = edge("s2", "t", "sig", merge="m1")
        s.add_edge(a)
        s.add_edge(b)
        s.add_merge([a, b], "m1")
        s.add_hard_constraint(a, "red", "lock a")
        s.add_hard_constraint(b, "green", "lock b")
        r = s.solve()
        assert r.edge_colors[a] == r.edge_colors[b]

    def test_memory_pattern(self):
        """Model a memory: data → RED, write-enable → GREEN, separated."""
        s = WireColorSolver()
        data = edge("data_src", "write_gate", "signal-A")
        write = edge("ctrl_src", "write_gate", "signal-W")
        s.add_edge(data)
        s.add_edge(write)
        s.add_hard_constraint(data, "red", "memory data")
        s.add_hard_constraint(write, "green", "write-enable")
        s.add_separation(data, write, "same sink different signals")
        r = s.solve()
        assert r.is_bipartite
        assert r.edge_colors[data] == "red"
        assert r.edge_colors[write] == "green"

    def test_many_edges_bipartite(self):
        """Large bipartite graph with alternating separations."""
        s = WireColorSolver()
        edges = []
        for i in range(20):
            e = edge(f"s{i}", "t", "sig")
            s.add_edge(e)
            edges.append(e)
        for i in range(0, 20, 2):
            for j in range(1, 20, 2):
                s.add_separation(edges[i], edges[j], f"{i}-{j}")
        r = s.solve()
        assert r.is_bipartite
        for i in range(0, 20, 2):
            for j in range(1, 20, 2):
                assert r.edge_colors[edges[i]] != r.edge_colors[edges[j]]
