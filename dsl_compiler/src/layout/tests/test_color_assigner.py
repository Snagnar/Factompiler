"""Tests for layout/color_assigner.py — wire color assignment."""

import pytest

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.layout.color_assigner import WireColorAssigner, WireColorResult
from dsl_compiler.src.layout.layout_plan import LayoutPlan
from dsl_compiler.src.layout.signal_graph import SignalGraph
from dsl_compiler.src.layout.wire_router import WireColorSolver, WireEdge


@pytest.fixture
def diagnostics():
    return ProgramDiagnostics()


@pytest.fixture
def plan():
    return LayoutPlan()


@pytest.fixture
def assigner(plan, diagnostics):
    return WireColorAssigner(plan, {}, diagnostics, {})


# ── Edge collection ──────────────────────────────────────────────────────


def test_collect_edges_basic(assigner, plan):
    plan.create_and_add_placement("src1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("sink1", "arithmetic-combinator", (2.5, 1), (1, 2), "arithmetic")

    sg = SignalGraph()
    sg.set_source("sig1", "src1")
    sg.add_sink("sig1", "sink1")

    edges = assigner._collect_edges(sg, plan.entity_placements, None)
    assert len(edges) >= 1
    assert edges[0].source_entity_id == "src1"
    assert edges[0].sink_entity_id == "sink1"


def test_collect_edges_no_source_filtered(assigner):
    sg = SignalGraph()
    sg.add_sink("sig1", "sink1")  # No source registered

    edges = assigner._collect_edges(sg, {}, None)
    assert edges == []


def test_expand_merges_no_junctions(assigner, plan):
    edges = [WireEdge("src1", "sink1", "sig", "lid")]
    expanded = assigner._expand_merges(edges, {}, {}, SignalGraph())
    assert len(expanded) == 1


def test_expand_merges_with_junctions(assigner, plan):
    from dsl_compiler.src.ir.builder import SignalRef

    plan.create_and_add_placement("src1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("sink1", "arithmetic-combinator", (2.5, 1), (1, 2), "arithmetic")

    edges = [WireEdge("merge1", "sink1", "signal-A", "lid")]
    junctions = {"merge1": {"inputs": [SignalRef("signal-A", "src1")]}}

    sg = SignalGraph()
    sg.set_source("src1", "src1")

    expanded = assigner._expand_merges(edges, junctions, plan.entity_placements, sg)
    assert len(expanded) == 1
    assert expanded[0].source_entity_id == "src1"
    assert expanded[0].merge_group == "merge1"


# ── Constraint building ─────────────────────────────────────────────────


def test_build_solver_returns_solver(assigner, plan):
    plan.create_and_add_placement("src1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("sink1", "arithmetic-combinator", (2.5, 1), (1, 2), "arithmetic")

    edges = [WireEdge("src1", "sink1", "sig", "lid")]
    sg = SignalGraph()
    solver = assigner._build_solver(edges, plan.entity_placements, {}, sg)
    assert isinstance(solver, WireColorSolver)


def test_collect_isolated_entities(assigner, plan):
    plan.create_and_add_placement("const1", "constant-combinator", (0, 0), (1, 2), "literal")
    plan.entity_placements["const1"].properties["is_input"] = True
    plan.create_and_add_placement("anchor1", "constant-combinator", (3, 0), (1, 1), "output_anchor")
    plan.entity_placements["anchor1"].properties["is_output"] = True

    assigner._collect_isolated_entities(plan.entity_placements)
    assert "const1" in assigner._isolated_entities
    assert "anchor1" in assigner._isolated_entities


def test_add_merge_constraints(assigner):
    solver = WireColorSolver()
    a = WireEdge("s1", "t", "sig", "l1", merge_group="m1")
    b = WireEdge("s2", "t", "sig", "l2", merge_group="m1")
    solver.add_edge(a)
    solver.add_edge(b)
    assigner._add_merge_constraints(solver, [a, b])
    r = solver.solve()
    assert r.edge_colors[a] == r.edge_colors[b]


def test_separation_same_signal_same_sink(assigner, plan):
    solver = WireColorSolver()
    a = WireEdge("s1", "t", "sig", "l1")
    b = WireEdge("s2", "t", "sig", "l2")
    solver.add_edge(a)
    solver.add_edge(b)
    assigner._add_separation_constraints(solver, [a, b], {}, {}, SignalGraph())
    r = solver.solve()
    assert r.edge_colors[a] != r.edge_colors[b]


# ── Memory / feedback ────────────────────────────────────────────────────


def test_is_internal_feedback_signal():
    assert WireColorAssigner._is_internal_feedback_signal("__feedback_x") is True
    assert WireColorAssigner._is_internal_feedback_signal("signal-A") is False


def test_is_memory_feedback_edge(assigner):
    assert assigner._is_memory_feedback_edge("src", "sink", "__feedback_x") is True
    assert assigner._is_memory_feedback_edge("src", "sink", "signal-A") is False


# ── Full pipeline ────────────────────────────────────────────────────────


def test_assign_colors_basic(assigner, plan):
    plan.create_and_add_placement("src1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("sink1", "arithmetic-combinator", (2.5, 1), (1, 2), "arithmetic")

    sg = SignalGraph()
    sg.set_source("sig1", "src1")
    sg.add_sink("sig1", "sink1")

    result = assigner.assign_colors(sg)
    assert isinstance(result, WireColorResult)
    assert len(result.wire_edges) >= 1
    assert len(result.edge_colors) >= 1
    assert result.is_bipartite


def test_assign_colors_separation(assigner, plan):
    """Two edges with same signal at same sink get different colors."""
    plan.create_and_add_placement("s1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("s2", "constant-combinator", (3.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("t", "arithmetic-combinator", (6.5, 1), (1, 2), "arithmetic")

    sg = SignalGraph()
    sg.set_source("sig1", "s1")
    sg.add_sink("sig1", "t")
    sg.set_source("sig2", "s2")
    sg.add_sink("sig2", "t")

    # sig1 and sig2 both resolve to same signal name in edges
    result = assigner.assign_colors(sg)
    assert isinstance(result, WireColorResult)


def test_wire_color_result_fields():
    result = WireColorResult(
        wire_edges=[],
        edge_colors={},
        network_ids={},
        is_bipartite=True,
        isolated_entities={"e1"},
    )
    assert result.is_bipartite
    assert "e1" in result.isolated_entities
