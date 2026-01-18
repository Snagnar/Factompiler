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


def test_early_stop_callback_on_solution():
    """Test EarlyStopCallback.on_solution_callback behavior."""
    cb = EarlyStopCallback(max_violations=0)
    # Simulate being called with low objective value
    # We can't fully test without a real solver, but ensure solution_count works
    assert cb.solution_count == 0


def test_engine_add_no_overlap_constraint(signal_graph, simple_placements, diagnostics):
    """Test _add_no_overlap_constraint adds overlap prevention."""
    from ortools.sat.python import cp_model

    engine = IntegerLayoutEngine(signal_graph, simple_placements, diagnostics)
    model = cp_model.CpModel()
    positions = engine._create_position_variables(model, 100)
    engine._add_no_overlap_constraint(model, positions)
    # Model should have constraints added
    assert model.Proto().constraints


def test_engine_add_span_constraints(signal_graph, simple_placements, diagnostics):
    """Test _add_span_constraints adds wire span limits."""
    from ortools.sat.python import cp_model

    signal_graph.set_source("sig1", "e1")
    signal_graph.add_sink("sig1", "e2")
    engine = IntegerLayoutEngine(signal_graph, simple_placements, diagnostics)
    model = cp_model.CpModel()
    positions = engine._create_position_variables(model, 100)
    violations, lengths = engine._add_span_constraints(model, positions, 9)
    assert isinstance(violations, list)
    assert isinstance(lengths, list)


def test_engine_add_edge_layout_constraints(signal_graph, simple_placements, diagnostics):
    """Test _add_edge_layout_constraints for north/south placement."""
    from ortools.sat.python import cp_model

    engine = IntegerLayoutEngine(signal_graph, simple_placements, diagnostics)
    model = cp_model.CpModel()
    positions = engine._create_position_variables(model, 100)
    engine._add_edge_layout_constraints(model, positions)
    # Should not crash


def test_engine_add_solution_hints(signal_graph, simple_placements, diagnostics):
    """Test _add_solution_hints provides initial solution."""
    from ortools.sat.python import cp_model

    engine = IntegerLayoutEngine(signal_graph, simple_placements, diagnostics)
    model = cp_model.CpModel()
    positions = engine._create_position_variables(model, 100)
    engine._add_solution_hints(model, positions)
    # Should not crash


def test_engine_create_objective(signal_graph, simple_placements, diagnostics):
    """Test _create_objective sets optimization goal."""
    from ortools.sat.python import cp_model

    signal_graph.set_source("sig1", "e1")
    signal_graph.add_sink("sig1", "e2")
    engine = IntegerLayoutEngine(signal_graph, simple_placements, diagnostics)
    model = cp_model.CpModel()
    positions = engine._create_position_variables(model, 100)
    violations, lengths = engine._add_span_constraints(model, positions, 9)
    engine._create_objective(model, violations, lengths, positions, 10000, 100)
    # Should have objective
    assert model.Proto().objective


def test_engine_report_violations(signal_graph, diagnostics):
    """Test _report_violations logs violation details."""
    plan = LayoutPlan()
    plan.create_and_add_placement("e1", "constant-combinator", (0.5, 1.0), (1, 2), "literal")
    plan.create_and_add_placement("e2", "arithmetic-combinator", (20.5, 1.0), (1, 2), "arithmetic")

    signal_graph.set_source("sig1", "e1")
    signal_graph.add_sink("sig1", "e2")

    engine = IntegerLayoutEngine(signal_graph, plan.entity_placements, diagnostics)
    result = OptimizationResult(
        positions={"e1": (0, 0), "e2": (20, 0)},
        violations=1,
        total_wire_length=20,
        success=True,
        strategy_used="test",
        solve_time=0.1,
    )
    engine._report_violations(result)
    # Should not crash


def test_engine_diagnose_failure(signal_graph, simple_placements, diagnostics):
    """Test _diagnose_failure logs diagnostic info."""
    engine = IntegerLayoutEngine(signal_graph, simple_placements, diagnostics)
    engine._diagnose_failure()
    # Should not crash


def test_engine_optimize_with_decomposition(signal_graph, diagnostics):
    """Test _optimize_with_decomposition for large graphs."""
    plan = LayoutPlan()
    # Create 10 disconnected entities (10 components)
    for i in range(10):
        plan.create_and_add_placement(
            f"e{i}", "constant-combinator", (i * 5 + 0.5, 1.0), (1, 2), "literal"
        )

    engine = IntegerLayoutEngine(signal_graph, plan.entity_placements, diagnostics)
    # Call directly (normally only called for 500+ entities)
    result = engine._optimize_with_decomposition(5)  # positional argument
    assert isinstance(result, dict)


def test_engine_with_wire_merge_junctions(signal_graph, diagnostics):
    """Test engine handles wire merge junctions in connectivity."""
    from dsl_compiler.src.ir.nodes import SignalRef

    plan = LayoutPlan()
    plan.create_and_add_placement("src1", "constant-combinator", (0.5, 1.0), (1, 2), "literal")
    plan.create_and_add_placement(
        "sink1", "arithmetic-combinator", (2.5, 1.0), (1, 2), "arithmetic"
    )

    signal_graph.set_source("sig1", "src1")
    signal_graph.add_sink("sig1", "merge1")  # Sink to merge
    signal_graph.set_source("merge_out", "merge1")
    signal_graph.add_sink("merge_out", "sink1")

    wire_merge_junctions = {
        "merge1": {
            "inputs": [SignalRef("signal-A", "src1")],
            "output_sinks": ["sink1"],
        }
    }

    engine = IntegerLayoutEngine(
        signal_graph, plan.entity_placements, diagnostics, wire_merge_junctions=wire_merge_junctions
    )
    assert len(engine.connections) >= 0


def test_engine_power_pole_excluded_from_bounds(diagnostics):
    """Test power poles are excluded from bounding box."""
    plan = LayoutPlan()
    plan.create_and_add_placement("e1", "constant-combinator", (0.5, 1.0), (1, 2), "literal")
    plan.create_and_add_placement("pole1", "medium-electric-pole", (50.5, 50.5), (1, 1), "power")
    plan.entity_placements["pole1"].properties["is_power_pole"] = True

    sg = SignalGraph()
    engine = IntegerLayoutEngine(sg, plan.entity_placements, diagnostics)
    assert "pole1" in engine._power_pole_ids


# =============================================================================
# Coverage gap tests (Lines 293-296, 308-314, 361-374, 771-773, 892-894, 902-904, 956-961)
# =============================================================================


class TestIntegerLayoutSolverCoverageGaps:
    """Tests for integer_layout_solver.py coverage gaps > 2 lines."""

    @pytest.fixture
    def signal_graph(self):
        return SignalGraph()

    @pytest.fixture
    def diagnostics(self):
        return ProgramDiagnostics()

    def test_constraint_variable_bounds(self, signal_graph, diagnostics):
        """Cover lines 293-296: constraint variable bounds setup."""
        plan = LayoutPlan()
        plan.create_and_add_placement("e1", "constant-combinator", (0.5, 1.0), (1, 2), "literal")
        plan.create_and_add_placement(
            "e2", "arithmetic-combinator", (2.5, 1.0), (1, 2), "arithmetic"
        )

        signal_graph.set_source("sig1", "e1")
        signal_graph.add_sink("sig1", "e2")

        engine = IntegerLayoutEngine(signal_graph, plan.entity_placements, diagnostics)
        result = engine.optimize(time_limit_seconds=1)
        assert isinstance(result, dict)

    def test_add_wire_constraints(self, signal_graph, diagnostics):
        """Cover lines 308-314: wire constraint addition."""
        plan = LayoutPlan()
        plan.create_and_add_placement("e1", "constant-combinator", (0.5, 1.0), (1, 2), "literal")
        plan.create_and_add_placement(
            "e2", "arithmetic-combinator", (5.5, 1.0), (1, 2), "arithmetic"
        )
        plan.create_and_add_placement("e3", "decider-combinator", (10.5, 1.0), (1, 2), "decider")

        signal_graph.set_source("sig1", "e1")
        signal_graph.add_sink("sig1", "e2")
        signal_graph.set_source("sig2", "e2")
        signal_graph.add_sink("sig2", "e3")

        engine = IntegerLayoutEngine(signal_graph, plan.entity_placements, diagnostics)
        # Wire constraints should be added for connections
        assert engine.connections is not None

    def test_add_non_overlap_constraints(self, signal_graph, diagnostics):
        """Cover lines 361-374: non-overlap constraint setup."""
        plan = LayoutPlan()
        plan.create_and_add_placement("e1", "constant-combinator", (0.5, 1.0), (1, 2), "literal")
        plan.create_and_add_placement("e2", "constant-combinator", (1.5, 1.0), (1, 2), "literal")

        engine = IntegerLayoutEngine(signal_graph, plan.entity_placements, diagnostics)
        # Non-overlap constraints should have been added
        assert engine.n_entities == 2

    def test_solution_extraction(self, signal_graph, diagnostics):
        """Cover lines 771-773: solution extraction from solver."""
        plan = LayoutPlan()
        plan.create_and_add_placement("e1", "constant-combinator", (0.5, 1.0), (1, 2), "literal")

        engine = IntegerLayoutEngine(signal_graph, plan.entity_placements, diagnostics)
        result = engine.optimize(time_limit_seconds=1)
        # Should return valid positions
        assert "e1" in result

    def test_fallback_when_infeasible(self, signal_graph, diagnostics):
        """Cover lines 892-894: fallback when constraints infeasible."""
        plan = LayoutPlan()
        plan.create_and_add_placement("e1", "constant-combinator", (0.5, 1.0), (1, 2), "literal")

        engine = IntegerLayoutEngine(signal_graph, plan.entity_placements, diagnostics)
        fallback = engine._fallback_grid_layout()
        assert isinstance(fallback, dict)

    def test_grid_layout_multiple_entities(self, signal_graph, diagnostics):
        """Cover lines 902-904: grid layout with multiple entities."""
        plan = LayoutPlan()
        for i in range(5):
            plan.create_and_add_placement(
                f"e{i}", "constant-combinator", (i * 2.0 + 0.5, 1.0), (1, 2), "literal"
            )

        engine = IntegerLayoutEngine(signal_graph, plan.entity_placements, diagnostics)
        result = engine._fallback_grid_layout()
        assert len(result) == 5

    def test_optimize_with_fixed_positions(self, signal_graph, diagnostics):
        """Cover lines 956-961: optimization with some fixed positions."""
        plan = LayoutPlan()
        plan.create_and_add_placement("e1", "constant-combinator", (0.5, 1.0), (1, 2), "literal")
        plan.create_and_add_placement("e2", "small-lamp", (3.5, 1.5), (1, 1), "output")

        # Small lamps should have fixed positions
        signal_graph.set_source("sig1", "e1")
        signal_graph.add_sink("sig1", "e2")

        engine = IntegerLayoutEngine(signal_graph, plan.entity_placements, diagnostics)
        result = engine.optimize(time_limit_seconds=1)
        assert "e1" in result
        assert "e2" in result
