"""Tests for layout/connection_planner.py — new constraint-based pipeline."""

import pytest

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.layout.connection_planner import (
    ConnectionPlanner,
    RelayNetwork,
    RelayNode,
)
from dsl_compiler.src.layout.layout_plan import LayoutPlan
from dsl_compiler.src.layout.signal_graph import SignalGraph
from dsl_compiler.src.layout.tile_grid import TileGrid
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer


@pytest.fixture
def diagnostics():
    return ProgramDiagnostics()


@pytest.fixture
def plan():
    return LayoutPlan()


@pytest.fixture
def planner(plan, diagnostics):
    return ConnectionPlanner(plan, {}, diagnostics, TileGrid())


# ── RelayNode ─────────────────────────────────────────────────────────────


def test_relay_node_init():
    node = RelayNode((1.0, 2.0), "pole1", "medium-electric-pole")
    assert node.position == (1.0, 2.0)
    assert node.entity_id == "pole1"
    assert node.pole_type == "medium-electric-pole"


def test_relay_node_can_route_network():
    node = RelayNode((1.0, 2.0), "pole1", "medium-electric-pole")
    assert node.can_route_network(1, "red") is True
    node.add_network(1, "red")
    assert node.can_route_network(1, "green") is True
    node.add_network(2, "red")
    assert node.can_route_network(3, "red") is False


def test_relay_node_add_network():
    node = RelayNode((1.0, 2.0), "pole1", "medium-electric-pole")
    node.add_network(1, "red")
    node.add_network(2, "green")
    assert 1 in node.networks_red
    assert 2 in node.networks_green


# ── RelayNetwork ──────────────────────────────────────────────────────────


def test_relay_network_init(plan, diagnostics):
    net = RelayNetwork(TileGrid(), 9.0, plan, diagnostics)
    assert net.span_limit == 9.0
    assert net.relay_nodes == {}


def test_relay_network_add_relay_node(plan, diagnostics):
    net = RelayNetwork(TileGrid(), 9.0, plan, diagnostics)
    node = net.add_relay_node((5.0, 5.0), "pole1", "medium-electric-pole")
    assert node.entity_id == "pole1"
    assert (5, 5) in net.relay_nodes


def test_relay_network_span_limit(plan, diagnostics):
    net = RelayNetwork(TileGrid(), 9.0, plan, diagnostics)
    assert net.span_limit == 9.0


def test_relay_network_route_signal_short(plan, diagnostics):
    """Within span → empty relay path."""
    net = RelayNetwork(TileGrid(), 9.0, plan, diagnostics)
    result = net.route_signal((0, 0), (3, 0), "test_sig", "red", 1)
    assert result == []


def test_relay_network_route_signal_long(plan, diagnostics):
    """Beyond span → creates relay path or returns None."""
    net = RelayNetwork(TileGrid(), 9.0, plan, diagnostics)
    result = net.route_signal((0, 0), (25, 0), "test_sig", "red", 1)
    assert result is None or isinstance(result, list)


def test_relay_network_get_node_by_id(plan, diagnostics):
    net = RelayNetwork(TileGrid(), 9.0, plan, diagnostics)
    node = net.add_relay_node((5, 5), "relay_x", "medium-electric-pole")
    found = net._get_node_by_id("relay_x")
    assert found is node
    assert net._get_node_by_id("nonexistent") is None


# ── ConnectionPlanner init + public API ──────────────────────────────────


def test_connection_planner_init(planner):
    assert planner.layout_plan is not None
    assert planner.diagnostics is not None


def test_connection_planner_plan_connections_empty(planner):
    sg = SignalGraph()
    result = planner.plan_connections(sg, {})
    assert result is True


def test_connection_planner_plan_connections_basic(planner, plan):
    plan.create_and_add_placement("src1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("sink1", "arithmetic-combinator", (2.5, 1), (1, 2), "arithmetic")

    sg = SignalGraph()
    sg.set_source("sig1", "src1")
    sg.add_sink("sig1", "sink1")

    result = planner.plan_connections(sg, plan.entity_placements)
    assert isinstance(result, bool)


def test_connection_planner_get_wire_color_for_edge(planner, plan):
    plan.create_and_add_placement("src1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("sink1", "arithmetic-combinator", (2.5, 1), (1, 2), "arithmetic")

    sg = SignalGraph()
    sg.set_source("sig1", "src1")
    sg.add_sink("sig1", "sink1")
    planner.plan_connections(sg, plan.entity_placements)

    color = planner.get_wire_color_for_edge("src1", "sink1", "sig1")
    assert color in ("red", "green")


def test_connection_planner_get_wire_color_for_entity_pair(planner, plan):
    plan.create_and_add_placement("src1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("sink1", "arithmetic-combinator", (2.5, 1), (1, 2), "arithmetic")

    sg = SignalGraph()
    sg.set_source("sig1", "src1")
    sg.add_sink("sig1", "sink1")
    planner.plan_connections(sg, plan.entity_placements)

    color = planner.get_wire_color_for_entity_pair("src1", "sink1")
    assert color in ("red", "green", None)


def test_connection_planner_edge_color_map(planner, plan):
    plan.create_and_add_placement("src1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("sink1", "arithmetic-combinator", (2.5, 1), (1, 2), "arithmetic")

    sg = SignalGraph()
    sg.set_source("sig1", "src1")
    sg.add_sink("sig1", "sink1")
    planner.plan_connections(sg, plan.entity_placements)

    ecm = planner.edge_color_map()
    assert isinstance(ecm, dict)


# ── Internal: edge collection ────────────────────────────────────────────


def test_collect_edges_basic(planner, plan):
    plan.create_and_add_placement("src1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("sink1", "arithmetic-combinator", (2.5, 1), (1, 2), "arithmetic")

    sg = SignalGraph()
    sg.set_source("sig1", "src1")
    sg.add_sink("sig1", "sink1")

    edges = planner._collect_edges(sg, plan.entity_placements, None)
    assert len(edges) >= 1
    assert edges[0].source_entity_id == "src1"
    assert edges[0].sink_entity_id == "sink1"


def test_collect_edges_no_source_filtered(planner):
    sg = SignalGraph()
    sg.add_sink("sig1", "sink1")  # No source registered

    edges = planner._collect_edges(sg, {}, None)
    assert edges == []


def test_expand_merges_no_junctions(planner, plan):
    from dsl_compiler.src.layout.wire_router import WireEdge

    edges = [WireEdge("src1", "sink1", "sig", "lid")]
    expanded = planner._expand_merges(edges, {}, {}, SignalGraph())
    assert len(expanded) == 1


def test_expand_merges_with_junctions(planner, plan):
    from dsl_compiler.src.ir.builder import SignalRef
    from dsl_compiler.src.layout.wire_router import WireEdge

    plan.create_and_add_placement("src1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("sink1", "arithmetic-combinator", (2.5, 1), (1, 2), "arithmetic")

    edges = [WireEdge("merge1", "sink1", "signal-A", "lid")]
    junctions = {"merge1": {"inputs": [SignalRef("signal-A", "src1")]}}

    sg = SignalGraph()
    sg.set_source("src1", "src1")

    expanded = planner._expand_merges(edges, junctions, plan.entity_placements, sg)
    assert len(expanded) == 1
    assert expanded[0].source_entity_id == "src1"
    assert expanded[0].merge_group == "merge1"


# ── Internal: constraint building ────────────────────────────────────────


def test_build_solver_returns_solver(planner, plan):
    from dsl_compiler.src.layout.wire_router import WireColorSolver, WireEdge

    plan.create_and_add_placement("src1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("sink1", "arithmetic-combinator", (2.5, 1), (1, 2), "arithmetic")

    edges = [WireEdge("src1", "sink1", "sig", "lid")]
    sg = SignalGraph()
    solver = planner._build_solver(edges, plan.entity_placements, {}, sg)
    assert isinstance(solver, WireColorSolver)


def test_collect_isolated_entities(planner, plan):
    plan.create_and_add_placement("const1", "constant-combinator", (0, 0), (1, 2), "literal")
    plan.entity_placements["const1"].properties["is_input"] = True
    plan.create_and_add_placement("anchor1", "constant-combinator", (3, 0), (1, 1), "output_anchor")
    plan.entity_placements["anchor1"].properties["is_output"] = True

    planner._collect_isolated_entities(plan.entity_placements)
    assert "const1" in planner._isolated_entities
    assert "anchor1" in planner._isolated_entities


def test_add_merge_constraints(planner):
    from dsl_compiler.src.layout.wire_router import WireColorSolver, WireEdge

    solver = WireColorSolver()
    a = WireEdge("s1", "t", "sig", "l1", merge_group="m1")
    b = WireEdge("s2", "t", "sig", "l2", merge_group="m1")
    solver.add_edge(a)
    solver.add_edge(b)
    planner._add_merge_constraints(solver, [a, b])
    r = solver.solve()
    assert r.edge_colors[a] == r.edge_colors[b]


def test_separation_same_signal_same_sink(planner, plan):
    from dsl_compiler.src.layout.wire_router import WireColorSolver, WireEdge

    solver = WireColorSolver()
    a = WireEdge("s1", "t", "sig", "l1")
    b = WireEdge("s2", "t", "sig", "l2")
    solver.add_edge(a)
    solver.add_edge(b)
    planner._add_separation_constraints(solver, [a, b], {}, {}, SignalGraph())
    r = solver.solve()
    assert r.edge_colors[a] != r.edge_colors[b]


# ── Internal: memory / feedback ──────────────────────────────────────────


def test_is_internal_feedback_signal(planner):
    assert planner._is_internal_feedback_signal("__feedback_x") is True
    assert planner._is_internal_feedback_signal("signal-A") is False


def test_is_memory_feedback_edge(planner):
    assert planner._is_memory_feedback_edge("src", "sink", "__feedback_x") is True
    assert planner._is_memory_feedback_edge("src", "sink", "signal-A") is False


# ── Internal: physical connections ───────────────────────────────────────


def test_get_connection_side(planner, plan):
    plan.create_and_add_placement("arith1", "arithmetic-combinator", (0, 0), (1, 2), "arithmetic")
    plan.create_and_add_placement("const1", "constant-combinator", (3, 0), (1, 2), "literal")

    result_src = planner._get_connection_side("arith1", is_source=True)
    result_snk = planner._get_connection_side("arith1", is_source=False)
    result_const = planner._get_connection_side("const1", is_source=True)

    assert result_src == "output"
    assert result_snk == "input"
    # Constant combinator is not dual-circuit-connectable, so should be None
    assert result_const is None


def test_build_mst_two_entities(planner, plan):
    plan.create_and_add_placement("e1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("e2", "arithmetic-combinator", (2.5, 1), (1, 2), "arithmetic")

    mst = planner._build_mst(["e1", "e2"])
    assert len(mst) == 1
    assert ("e1", "e2") in mst or ("e2", "e1") in mst


def test_build_mst_three_entities(planner, plan):
    plan.create_and_add_placement("e1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("e2", "arithmetic-combinator", (2.5, 1), (1, 2), "arithmetic")
    plan.create_and_add_placement("e3", "arithmetic-combinator", (4.5, 1), (1, 2), "arithmetic")

    mst = planner._build_mst(["e1", "e2", "e3"])
    assert len(mst) == 2


def test_build_mst_single_entity(planner, plan):
    plan.create_and_add_placement("e1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    assert planner._build_mst(["e1"]) == []


# ── Internal: self-feedback ──────────────────────────────────────────────


def test_add_self_feedback_connections(planner, plan):
    plan.create_and_add_placement("latch1", "decider-combinator", (0.5, 1), (1, 2), "latch")
    plan.entity_placements["latch1"].properties["has_self_feedback"] = True
    plan.entity_placements["latch1"].properties["feedback_signal"] = "signal-A"

    initial = len(plan.wire_connections)
    planner._add_self_feedback_connections()
    assert len(plan.wire_connections) > initial


# ── Internal: relay validation ───────────────────────────────────────────


def test_validate_relay_coverage(planner, plan):
    plan.create_and_add_placement("src1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("sink1", "arithmetic-combinator", (2.5, 1), (1, 2), "arithmetic")

    sg = SignalGraph()
    sg.set_source("sig1", "src1")
    sg.add_sink("sig1", "sink1")
    planner.plan_connections(sg, plan.entity_placements)

    # Should not crash
    planner._validate_relay_coverage()


# ── Internal: power pole relay registration ──────────────────────────────


def test_register_power_poles_as_relays(planner, plan):
    plan.create_and_add_placement("pole1", "medium-electric-pole", (5, 5), (1, 1), "power_pole")
    plan.entity_placements["pole1"].properties["is_power_pole"] = True
    plan.entity_placements["pole1"].properties["pole_type"] = "medium"
    planner._register_power_poles_as_relays()
    assert len(planner.relay_network.relay_nodes) >= 1


# ── Full pipeline integration tests ─────────────────────────────────────


def compile_to_ir(source: str):
    """Helper to compile source to IR."""
    diags = ProgramDiagnostics()
    parser = DSLParser()
    ast = parser.parse(source, "<test>")
    analyzer = SemanticAnalyzer(diagnostics=diags)
    analyzer.visit(ast)
    lowerer = ASTLowerer(analyzer, diags)
    ir_ops = lowerer.lower_program(ast)
    return ir_ops, lowerer, diags


class TestConnectionPlannerCoverageGaps:
    def test_simple_arithmetic(self):
        source = """
        Signal a = 100;
        Signal b = a + 1;
        Signal c = b + 1;
        """
        compile_to_ir(source)

    def test_multi_operand(self):
        source = """
        Signal a = 10;
        Signal b = 20;
        Signal merged = a + b;
        Signal result = merged * 2;
        """
        compile_to_ir(source)

    def test_fan_out(self):
        source = """
        Signal a = 10;
        Signal b = a + 1;
        Signal c = a + 2;
        Signal d = a + 3;
        """
        compile_to_ir(source)

    def test_memory_basic(self):
        source = """
        Memory counter: "signal-A";
        Signal x = 1;
        counter.write(x);
        Signal out = counter;
        """
        compile_to_ir(source)

    def test_chain(self):
        source = """
        Signal a = 10;
        Signal b = a + 1;
        Signal c = b + 1;
        Signal d = c + 1;
        Signal e = d + 1;
        """
        compile_to_ir(source)
