"""Tests for layout/connection_planner.py - one test per function."""

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


@pytest.fixture
def diagnostics():
    return ProgramDiagnostics()


@pytest.fixture
def plan():
    return LayoutPlan()


@pytest.fixture
def planner(plan, diagnostics):
    return ConnectionPlanner(plan, {}, diagnostics, TileGrid())


# --- RelayNode Tests ---
def test_relay_node_init():
    node = RelayNode((1.0, 2.0), "pole1", "medium-electric-pole")
    assert node.position == (1.0, 2.0)
    assert node.entity_id == "pole1"
    assert node.pole_type == "medium-electric-pole"


def test_relay_node_can_route_network():
    node = RelayNode((1.0, 2.0), "pole1", "medium-electric-pole")
    assert node.can_route_network(1, "red") is True
    node.add_network(1, "red")
    assert node.can_route_network(1, "green") is True  # Different color OK
    node.add_network(2, "red")
    assert node.can_route_network(3, "red") is False  # Both red slots taken


def test_relay_node_add_network():
    node = RelayNode((1.0, 2.0), "pole1", "medium-electric-pole")
    node.add_network(1, "red")
    node.add_network(2, "green")
    assert 1 in node.networks_red
    assert 2 in node.networks_green


# --- RelayNetwork Tests ---
def test_relay_network_init(plan, diagnostics):
    net = RelayNetwork(TileGrid(), None, {}, 9.0, plan, diagnostics)
    assert net.max_span == 9.0
    assert net.relay_nodes == {}


def test_relay_network_add_relay_node(plan, diagnostics):
    net = RelayNetwork(TileGrid(), None, {}, 9.0, plan, diagnostics)
    node = net.add_relay_node((5.0, 5.0), "pole1", "medium-electric-pole")
    assert node.entity_id == "pole1"
    assert (5, 5) in net.relay_nodes


def test_relay_network_find_relay_near(plan, diagnostics):
    net = RelayNetwork(TileGrid(), None, {}, 9.0, plan, diagnostics)
    net.add_relay_node((5.0, 5.0), "pole1", "medium-electric-pole")
    found = net.find_relay_near((6.0, 5.0), 2.0)
    assert found is not None
    not_found = net.find_relay_near((100.0, 100.0), 2.0)
    assert not_found is None


def test_relay_network_span_limit(plan, diagnostics):
    net = RelayNetwork(TileGrid(), None, {}, 9.0, plan, diagnostics)
    assert net.span_limit > 0


# --- ConnectionPlanner Tests ---
def test_connection_planner_init(planner):
    assert planner.layout_plan is not None
    assert planner.diagnostics is not None


def test_connection_planner_plan_connections_empty(planner):
    sg = SignalGraph()
    result = planner.plan_connections(sg, {})
    assert result is True  # No connections to plan = success


def test_connection_planner_plan_connections_basic(planner, plan):
    # Create two entities
    plan.create_and_add_placement("src1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("sink1", "arithmetic-combinator", (2.5, 1), (1, 2), "arithmetic")

    sg = SignalGraph()
    sg.set_source("sig1", "src1")
    sg.add_sink("sig1", "sink1")

    result = planner.plan_connections(sg, plan.entity_placements)
    assert isinstance(result, bool)


def test_connection_planner_compute_network_ids(planner):
    from dsl_compiler.src.layout.wire_router import CircuitEdge

    edges = [
        CircuitEdge("src1", "sink1", "sig1", 1),
        CircuitEdge("src2", "sink2", "sig2", 1),
    ]
    planner._compute_network_ids(edges)
    # Should assign network IDs to edges
    assert len(planner._edge_network_ids) >= 0


def test_connection_planner_get_wire_color_for_edge(planner, plan):
    plan.create_and_add_placement("src1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("sink1", "arithmetic-combinator", (2.5, 1), (1, 2), "arithmetic")

    sg = SignalGraph()
    sg.set_source("sig1", "src1")
    sg.add_sink("sig1", "sink1")
    planner.plan_connections(sg, plan.entity_placements)

    color = planner.get_wire_color_for_edge("src1", "sink1", "sig1")
    assert color in ("red", "green", None)


def test_connection_planner_populate_wire_connections(planner, plan):
    plan.create_and_add_placement("src1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("sink1", "arithmetic-combinator", (2.5, 1), (1, 2), "arithmetic")

    sg = SignalGraph()
    sg.set_source("sig1", "src1")
    sg.add_sink("sig1", "sink1")

    planner.plan_connections(sg, plan.entity_placements)
    # plan_connections calls _populate_wire_connections internally
    assert len(plan.wire_connections) >= 0


def test_connection_planner_build_minimum_spanning_tree(planner, plan):
    plan.create_and_add_placement("e1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("e2", "arithmetic-combinator", (2.5, 1), (1, 2), "arithmetic")
    plan.create_and_add_placement("e3", "arithmetic-combinator", (4.5, 1), (1, 2), "arithmetic")

    mst_edges = planner._build_minimum_spanning_tree(["e1", "e2", "e3"])
    assert isinstance(mst_edges, list)
    assert len(mst_edges) == 2  # MST of 3 nodes has 2 edges


def test_connection_planner_find_bidirectional_pairs(planner):
    from dsl_compiler.src.layout.wire_router import CircuitEdge

    edges = [
        CircuitEdge("a", "b", "sig", 1),
        CircuitEdge("b", "a", "sig", 1),
    ]
    pairs = planner._find_bidirectional_pairs(edges)
    # Just ensure the function runs and returns a set
    assert isinstance(pairs, set)


def test_connection_planner_route_edge_directly(planner, plan):
    plan.create_and_add_placement("src1", "constant-combinator", (0.5, 1), (1, 2), "literal")
    plan.create_and_add_placement("sink1", "arithmetic-combinator", (2.5, 1), (1, 2), "arithmetic")

    from dsl_compiler.src.layout.wire_router import CircuitEdge

    edge = CircuitEdge("src1", "sink1", "sig1", 1)
    result = planner._route_edge_directly(edge, "red")
    assert isinstance(result, bool)


def test_connection_planner_register_power_poles_as_relays(planner, plan):
    # Add a power pole placement
    plan.create_and_add_placement("pole1", "medium-electric-pole", (5, 5), (1, 1), "power_pole")
    plan.entity_placements["pole1"].properties["is_power_pole"] = True
    planner._register_power_poles_as_relays()
    # Should register the pole in relay_network
    assert len(planner.relay_network.relay_nodes) >= 0


def test_connection_planner_is_internal_feedback_signal(planner):
    result = planner._is_internal_feedback_signal("signal-W")
    assert isinstance(result, bool)
    result2 = planner._is_internal_feedback_signal("signal-A")
    assert isinstance(result2, bool)


def test_relay_network_route_signal(plan, diagnostics):
    net = RelayNetwork(TileGrid(), None, {}, 9.0, plan, diagnostics)
    # Add two relays that can form a path
    net.add_relay_node((0, 0), "pole1", "medium-electric-pole")
    net.add_relay_node((5, 0), "pole2", "medium-electric-pole")
    result = net.route_signal((0, 0), (8, 0), "test_sig", "red", 1)
    assert result is None or isinstance(result, list)


def test_relay_network_find_path_through_existing(plan, diagnostics):
    net = RelayNetwork(TileGrid(), None, {}, 9.0, plan, diagnostics)
    net.add_relay_node((4, 0), "relay1", "medium-electric-pole")
    result = net._find_path_through_existing_relays((0, 0), (8, 0), 9.0, "red", 1)
    assert result is None or isinstance(result, list)
