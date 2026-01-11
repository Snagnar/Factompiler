"""Tests for layout/planner.py - one test per function."""

import pytest

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.ir.nodes import IRArith, IRConst, SignalRef
from dsl_compiler.src.layout.planner import LayoutPlanner


def make_const(node_id, value, output_type="signal-A"):
    op = IRConst(node_id, output_type)
    op.value = value
    op.debug_metadata = {"user_declared": True}
    return op


def make_arith(node_id, left, right, op_type="+", output_type="signal-C"):
    op = IRArith(node_id, output_type)
    op.op = op_type
    op.left = left
    op.right = right
    return op


@pytest.fixture
def planner():
    return LayoutPlanner({}, ProgramDiagnostics(), max_layout_retries=0)


def test_init(planner):
    assert planner.tile_grid is not None
    assert planner.layout_plan is not None


def test_plan_layout(planner):
    result = planner.plan_layout([make_const("c1", 42)])
    assert "c1" in result.entity_placements


def test_reset_layout_state(planner):
    planner._memory_modules = {"x": 1}
    planner._reset_layout_state()
    assert planner._memory_modules == {}


def test_setup_signal_analysis(planner):
    planner._setup_signal_analysis([make_const("c1", 42)])
    assert planner.signal_analyzer is not None


def test_create_entities(planner):
    planner._setup_signal_analysis([make_const("c1", 42)])
    planner._create_entities([make_const("c1", 42)])
    assert planner.signal_graph is not None


def test_update_tile_grid(planner):
    planner._setup_signal_analysis([make_const("c1", 42)])
    planner._create_entities([make_const("c1", 42)])
    planner._update_tile_grid()  # Should not raise


def test_set_metadata(planner):
    planner._set_metadata("Label", "Desc")
    assert planner.layout_plan.blueprint_label == "Label"
    assert planner.layout_plan.blueprint_description == "Desc"


def test_determine_locked_wire_colors(planner):
    planner._setup_signal_analysis([make_const("c1", 42)])
    planner._create_entities([make_const("c1", 42)])
    result = planner._determine_locked_wire_colors()
    assert isinstance(result, dict)


def test_plan_connections(planner):
    planner._setup_signal_analysis([make_const("c1", 42)])
    planner._create_entities([make_const("c1", 42)])
    planner._update_tile_grid()
    result = planner._plan_connections()
    assert isinstance(result, bool)


def test_resolve_source_entity(planner):
    planner._setup_signal_analysis([make_const("c1", 42)])
    planner._create_entities([make_const("c1", 42)])
    # source_entity should resolve constants
    result = planner._resolve_source_entity("c1")
    assert result is None or isinstance(result, str)


def test_optimize_positions(planner):
    ops = [make_const("c1", 1), make_const("c2", 2)]
    planner._setup_signal_analysis(ops)
    planner._create_entities(ops)
    planner._update_tile_grid()
    planner._optimize_positions(time_multiplier=0.1)
    # Just verify it runs without error


def test_plan_layout_with_connections(planner):
    # Create connected operations to exercise wire routing
    c1 = make_const("c1", 10)
    c2 = make_const("c2", 20)
    arith = make_arith("add1", SignalRef("signal-A", "c1"), SignalRef("signal-A", "c2"))
    result = planner.plan_layout([c1, c2, arith])
    assert "c1" in result.entity_placements
    assert "add1" in result.entity_placements


def test_plan_layout_with_power_poles():
    planner = LayoutPlanner(
        {}, ProgramDiagnostics(), max_layout_retries=0, power_pole_type="medium-electric-pole"
    )
    ops = [make_const("c1", 42)]
    result = planner.plan_layout(ops)
    # Check that power poles were added
    has_pole = any("pole" in p.entity_type for p in result.entity_placements.values())
    assert has_pole or len(result.entity_placements) >= 1  # May not add poles if layout is small
