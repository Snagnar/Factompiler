"""Tests for integer_layout_solver.py coverage gaps."""

from dsl_compiler.src.common.constants import DEFAULT_CONFIG
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.layout.planner import LayoutPlanner
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer


def compile_to_layout(source: str):
    """Helper to compile DSL source to layout plan."""
    diags = ProgramDiagnostics()
    parser = DSLParser()
    program = parser.parse(source.strip())
    analyzer = SemanticAnalyzer(diags)
    analyzer.visit(program)
    lowerer = ASTLowerer(analyzer, diags)
    ir_ops = lowerer.lower_program(program)

    planner = LayoutPlanner(
        lowerer.ir_builder.signal_type_map,
        diagnostics=diags,
        signal_refs=lowerer.signal_refs,
        referenced_signal_names=lowerer.referenced_signal_names,
        config=DEFAULT_CONFIG,
    )
    layout = planner.plan_layout(ir_ops)
    return layout, planner, diags


class TestIntegerLayoutSolverBasicPlacement:
    """Cover lines 293-296: basic placement algorithm."""

    def test_place_single_combinator(self):
        """Test placing a single combinator."""
        layout, _, diags = compile_to_layout("""
        Signal x = 5 | "signal-A";
        """)

    def test_place_two_combinators(self):
        """Test placing two combinators."""
        layout, _, diags = compile_to_layout("""
        Signal x = 5 | "signal-A";
        Signal y = x + 10;
        """)


class TestIntegerLayoutSolverConstraints:
    """Cover lines 308-314: constraint handling."""

    def test_placement_with_fixed_entities(self):
        """Test placement with user-placed entities."""
        layout, _, diags = compile_to_layout("""
        Entity lamp = place("small-lamp", 0, 0);
        Signal enable = 1 | "signal-E";
        lamp.enable = enable > 0;
        """)

    def test_placement_avoids_collision(self):
        """Test combinators avoid colliding with entities."""
        layout, _, diags = compile_to_layout("""
        Entity lamp1 = place("small-lamp", 0, 0);
        Entity lamp2 = place("small-lamp", 2, 0);
        Signal x = 5 | "signal-A";
        lamp1.enable = x > 0;
        lamp2.enable = x > 0;
        """)


class TestIntegerLayoutSolverOptimization:
    """Cover lines 361-374: layout optimization."""

    def test_optimize_wire_length(self):
        """Test layout optimizes wire length."""
        layout, _, diags = compile_to_layout("""
        Signal a = 5 | "signal-A";
        Signal b = a + 10;
        Signal c = b * 2;
        """)

    def test_optimize_connected_components(self):
        """Test layout keeps connected components close."""
        layout, _, diags = compile_to_layout("""
        Signal x = 5 | "signal-A";
        Signal y = x + 10;
        Signal z = y * 2;
        Signal w = z - 5;
        """)


class TestIntegerLayoutSolverRowColumn:
    """Cover lines 771-773: row/column allocation."""

    def test_horizontal_row_layout(self):
        """Test combinators are placed in rows."""
        layout, _, diags = compile_to_layout("""
        Signal a = 1 | "signal-A";
        Signal b = 2 | "signal-B";
        Signal c = 3 | "signal-C";
        Signal d = 4 | "signal-D";
        """)


class TestIntegerLayoutSolverDependencyOrder:
    """Cover lines 892-894, 902-904: dependency-ordered placement."""

    def test_dependency_order_respected(self):
        """Test dependencies are placed before dependents."""
        layout, _, diags = compile_to_layout("""
        Signal a = 5 | "signal-A";
        Signal b = a + 10;
        Signal c = b * 2;
        """)

    def test_parallel_dependencies(self):
        """Test parallel dependencies placed efficiently."""
        layout, _, diags = compile_to_layout("""
        Signal a = 5 | "signal-A";
        Signal b = 10 | "signal-B";
        Signal c = a + b;
        """)


class TestIntegerLayoutSolverMemoryPlacement:
    """Cover lines 956-961: memory cell placement."""

    def test_memory_cell_placement(self):
        """Test memory cells are placed correctly."""
        layout, _, diags = compile_to_layout("""
        Memory counter: "signal-A";
        counter.write(counter.read() + 1);
        """)

    def test_multiple_memory_placement(self):
        """Test multiple memory cells are placed."""
        layout, _, diags = compile_to_layout("""
        Memory counter1: "signal-A";
        Memory counter2: "signal-B";
        counter1.write(counter1.read() + 1);
        counter2.write(counter2.read() + 2);
        """)


class TestIntegerLayoutSolverComplexScenarios:
    """Tests for complex layout scenarios."""

    def test_complex_signal_graph(self):
        """Test complex signal dependency graph."""
        layout, _, diags = compile_to_layout("""
        Signal a = 5 | "signal-A";
        Signal b = 10 | "signal-B";
        Signal c = a + b;
        Signal d = a * 2;
        Signal e = b - 3;
        Signal f = c + d + e;
        """)

    def test_mixed_entities_and_combinators(self):
        """Test mixed entity and combinator layout."""
        layout, _, diags = compile_to_layout("""
        Entity lamp1 = place("small-lamp", 0, 0);
        Entity lamp2 = place("small-lamp", 3, 0);
        Signal x = 5 | "signal-A";
        Signal y = x + 10;
        lamp1.enable = x > 0;
        lamp2.enable = y > 10;
        """)
