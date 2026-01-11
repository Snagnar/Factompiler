"""Tests for layout/integer_layout_solver.py - one test per function."""

import pytest

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.layout.integer_layout_solver import (
    EarlyStopCallback,
    IntegerLayoutEngine,
    LayoutConstraints,
    OptimizationResult,
)
from dsl_compiler.src.layout.layout_plan import LayoutPlan
from dsl_compiler.src.layout.signal_graph import SignalGraph


@pytest.fixture
def diagnostics():
    return ProgramDiagnostics()


@pytest.fixture
def signal_graph():
    return SignalGraph()


@pytest.fixture
def simple_placements():
    plan = LayoutPlan()
    plan.create_and_add_placement("e1", "constant-combinator", (0.5, 1.0), (1, 2), "literal")
    plan.create_and_add_placement("e2", "arithmetic-combinator", (2.5, 1.0), (1, 2), "arithmetic")
    return plan.entity_placements


# --- EarlyStopCallback Tests ---
def test_early_stop_callback_init():
    cb = EarlyStopCallback(max_violations=5)
    assert cb.solution_count == 0


# --- LayoutConstraints Tests ---
def test_layout_constraints_defaults():
    lc = LayoutConstraints()
    assert lc.max_wire_span == 9.0
    assert lc.max_coordinate == 500


def test_layout_constraints_custom():
    lc = LayoutConstraints(max_wire_span=15.0, max_coordinate=100)
    assert lc.max_wire_span == 15.0
    assert lc.max_coordinate == 100


# --- IntegerLayoutEngine Tests ---
def test_engine_init(signal_graph, simple_placements, diagnostics):
    engine = IntegerLayoutEngine(signal_graph, simple_placements, diagnostics)
    assert engine.n_entities == 2
    assert len(engine.entity_ids) == 2


def test_engine_build_connectivity(signal_graph, simple_placements, diagnostics):
    signal_graph.set_source("sig1", "e1")
    signal_graph.add_sink("sig1", "e2")
    engine = IntegerLayoutEngine(signal_graph, simple_placements, diagnostics)
    assert len(engine.connections) == 1


def test_engine_identify_fixed_positions(signal_graph, simple_placements, diagnostics):
    engine = IntegerLayoutEngine(signal_graph, simple_placements, diagnostics)
    # Method is called in __init__, verify fixed_positions exists
    assert hasattr(engine, "fixed_positions")


def test_engine_extract_footprints(signal_graph, simple_placements, diagnostics):
    engine = IntegerLayoutEngine(signal_graph, simple_placements, diagnostics)
    # Method is called in __init__, verify footprints exist
    assert hasattr(engine, "footprints")
    assert len(engine.footprints) == 2


def test_engine_optimize(signal_graph, simple_placements, diagnostics):
    signal_graph.set_source("sig1", "e1")
    signal_graph.add_sink("sig1", "e2")
    engine = IntegerLayoutEngine(signal_graph, simple_placements, diagnostics)
    result = engine.optimize(time_limit_seconds=2)
    assert isinstance(result, dict)
    assert "e1" in result
    assert "e2" in result


def test_engine_fallback_grid_layout(signal_graph, simple_placements, diagnostics):
    engine = IntegerLayoutEngine(signal_graph, simple_placements, diagnostics)
    result = engine._fallback_grid_layout()
    assert isinstance(result, dict)
    assert "e1" in result
    assert "e2" in result


def test_engine_find_connected_components(signal_graph, simple_placements, diagnostics):
    signal_graph.set_source("sig1", "e1")
    signal_graph.add_sink("sig1", "e2")
    engine = IntegerLayoutEngine(signal_graph, simple_placements, diagnostics)
    components = engine._find_connected_components()
    assert isinstance(components, list)
    assert len(components) >= 1


def test_engine_get_relaxation_strategies(signal_graph, simple_placements, diagnostics):
    engine = IntegerLayoutEngine(signal_graph, simple_placements, diagnostics)
    strategies = engine._get_relaxation_strategies()
    assert isinstance(strategies, list)
    assert len(strategies) > 0


def test_engine_with_custom_constraints(signal_graph, simple_placements, diagnostics):
    constraints = LayoutConstraints(max_wire_span=5.0, max_coordinate=50)
    engine = IntegerLayoutEngine(signal_graph, simple_placements, diagnostics, constraints)
    assert engine.constraints.max_wire_span == 5.0


def test_engine_empty_graph(diagnostics):
    plan = LayoutPlan()
    plan.create_and_add_placement("e1", "constant-combinator", (0.5, 1.0), (1, 2), "literal")
    sg = SignalGraph()
    engine = IntegerLayoutEngine(sg, plan.entity_placements, diagnostics)
    result = engine.optimize(time_limit_seconds=1)
    assert isinstance(result, dict)


def test_optimization_result_dataclass():
    result = OptimizationResult(
        positions={"e1": (0, 0)},
        violations=0,
        total_wire_length=5,
        success=True,
        strategy_used="normal",
        solve_time=0.1,
    )
    assert result.success is True
    assert result.violations == 0


def test_engine_with_many_entities(signal_graph, diagnostics):
    """Test with more entities to cover strategy selection."""
    plan = LayoutPlan()
    # Create 10 entities in a chain
    for i in range(10):
        plan.create_and_add_placement(
            f"e{i}", "arithmetic-combinator", (i * 2 + 0.5, 1.0), (1, 2), "arithmetic"
        )
    # Connect them in sequence
    for i in range(9):
        signal_graph.set_source(f"sig{i}", f"e{i}")
        signal_graph.add_sink(f"sig{i}", f"e{i + 1}")

    engine = IntegerLayoutEngine(signal_graph, plan.entity_placements, diagnostics)
    result = engine.optimize(time_limit_seconds=5)
    assert len(result) == 10


def test_engine_disconnected_components(diagnostics):
    """Test with disconnected entity groups."""
    plan = LayoutPlan()
    # Group 1
    plan.create_and_add_placement("a1", "constant-combinator", (0.5, 1.0), (1, 2), "literal")
    plan.create_and_add_placement("a2", "arithmetic-combinator", (2.5, 1.0), (1, 2), "arithmetic")
    # Group 2 (disconnected)
    plan.create_and_add_placement("b1", "constant-combinator", (10.5, 1.0), (1, 2), "literal")
    plan.create_and_add_placement("b2", "arithmetic-combinator", (12.5, 1.0), (1, 2), "arithmetic")

    sg = SignalGraph()
    sg.set_source("sig1", "a1")
    sg.add_sink("sig1", "a2")
    sg.set_source("sig2", "b1")
    sg.add_sink("sig2", "b2")

    engine = IntegerLayoutEngine(sg, plan.entity_placements, diagnostics)
    components = engine._find_connected_components()
    assert len(components) == 2


def test_engine_with_fixed_positions(signal_graph, diagnostics):
    """Test handling of fixed positions."""
    plan = LayoutPlan()
    plan.create_and_add_placement("e1", "constant-combinator", (5.0, 5.0), (1, 2), "literal")
    plan.entity_placements["e1"].properties["user_specified_position"] = True
    plan.create_and_add_placement("e2", "arithmetic-combinator", (7.5, 5.0), (1, 2), "arithmetic")

    signal_graph.set_source("sig1", "e1")
    signal_graph.add_sink("sig1", "e2")

    engine = IntegerLayoutEngine(signal_graph, plan.entity_placements, diagnostics)
    assert "e1" in engine.fixed_positions
    result = engine.optimize(time_limit_seconds=2)
    # Fixed position should be respected
    assert result["e1"] == (5, 5)


def test_engine_solve_with_strategy(signal_graph, simple_placements, diagnostics):
    """Test _solve_with_strategy directly."""
    signal_graph.set_source("sig1", "e1")
    signal_graph.add_sink("sig1", "e2")
    engine = IntegerLayoutEngine(signal_graph, simple_placements, diagnostics)
    strategies = engine._get_relaxation_strategies()
    result = engine._solve_with_strategy(strategies[0], time_limit=2, early_stop=False)
    assert hasattr(result, "success")


def test_engine_position_variables(signal_graph, simple_placements, diagnostics):
    """Test position variable creation."""
    from ortools.sat.python import cp_model

    engine = IntegerLayoutEngine(signal_graph, simple_placements, diagnostics)
    model = cp_model.CpModel()
    positions = engine._create_position_variables(model, 100)
    assert len(positions) == 2
    assert "e1" in positions
    assert "e2" in positions
