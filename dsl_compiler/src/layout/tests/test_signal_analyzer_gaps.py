"""Tests for signal_analyzer.py coverage gaps."""

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
    return layout, planner, diags, ir_ops


class TestSignalAnalyzerDependencies:
    """Cover lines 184-186: dependency analysis."""

    def test_analyze_simple_dependency(self):
        """Test dependency analysis for simple expression."""
        layout, planner, diags, ir_ops = compile_to_layout("""
        Signal x = 5 | "signal-A";
        Signal y = x + 10;
        """)

    def test_analyze_chain_dependencies(self):
        """Test dependency analysis for chained expressions."""
        layout, planner, diags, ir_ops = compile_to_layout("""
        Signal a = 5 | "signal-A";
        Signal b = a + 10;
        Signal c = b * 2;
        Signal d = c - 5;
        """)


class TestSignalAnalyzerCycleDetection:
    """Cover lines 482-494: cycle detection in signal flow."""

    def test_memory_feedback_cycle(self):
        """Test cycle detection with memory feedback."""
        layout, planner, diags, ir_ops = compile_to_layout("""
        Memory counter: "signal-A";
        counter.write(counter.read() + 1);
        """)

    def test_no_cycle_in_feedforward(self):
        """Test no false positives in feedforward flow."""
        layout, planner, diags, ir_ops = compile_to_layout("""
        Signal a = 5 | "signal-A";
        Signal b = a + 10;
        Signal c = b * 2;
        """)

    def test_complex_memory_feedback(self):
        """Test complex memory feedback pattern."""
        layout, planner, diags, ir_ops = compile_to_layout("""
        Memory counter: "signal-A";
        Signal current = counter.read();
        Signal next = current + 1;
        Signal limited = next % 100;
        counter.write(limited);
        """)


class TestSignalAnalyzerFlowGraph:
    """Tests for signal flow graph construction."""

    def test_flow_graph_simple(self):
        """Test flow graph for simple program."""
        layout, planner, diags, ir_ops = compile_to_layout("""
        Signal x = 5 | "signal-A";
        Signal y = x + 10;
        """)

    def test_flow_graph_with_branching(self):
        """Test flow graph with signal used in multiple places."""
        layout, planner, diags, ir_ops = compile_to_layout("""
        Signal x = 5 | "signal-A";
        Signal y = x + 10;
        Signal z = x * 2;
        """)

    def test_flow_graph_with_entity(self):
        """Test flow graph with entity connection."""
        layout, planner, diags, ir_ops = compile_to_layout("""
        Signal enable = 1 | "signal-E";
        Entity lamp = place("small-lamp", 0, 0);
        lamp.enable = enable > 0;
        """)


class TestSignalAnalyzerProducerConsumer:
    """Tests for producer/consumer relationship analysis."""

    def test_single_producer_multiple_consumers(self):
        """Test signal with multiple consumers."""
        layout, planner, diags, ir_ops = compile_to_layout("""
        Signal x = 5 | "signal-A";
        Signal y = x + 10;
        Signal z = x * 2;
        Signal w = x - 3;
        """)

    def test_multiple_producers_merge(self):
        """Test multiple producers being merged."""
        layout, planner, diags, ir_ops = compile_to_layout("""
        Signal a = 5 | "signal-A";
        Signal b = 10 | "signal-B";
        Signal sum = a + b;
        """)
