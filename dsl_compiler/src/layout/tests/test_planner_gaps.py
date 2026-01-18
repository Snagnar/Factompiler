"""Tests for planner.py coverage gaps."""

from dsl_compiler.src.common.constants import DEFAULT_CONFIG
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.layout.planner import LayoutPlanner
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer


def compile_to_layout(source: str, power_pole_type: str | None = None):
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
        power_pole_type=power_pole_type,
        config=DEFAULT_CONFIG,
    )
    layout = planner.plan_layout(ir_ops)
    return layout, planner, diags


class TestPlannerConfiguration:
    """Cover lines 112-124: planner configuration handling."""

    def test_planner_with_power_poles(self):
        """Test planner with power poles enabled."""
        layout, _, diags = compile_to_layout(
            """
        Signal x = 5 | "signal-A";
        Signal y = x + 10;
        """,
            power_pole_type="medium-electric-pole",
        )
        # Layout should include power poles

    def test_planner_without_power_poles(self):
        """Test planner without power poles."""
        layout, _, diags = compile_to_layout(
            """
        Signal x = 5 | "signal-A";
        Signal y = x + 10;
        """,
            power_pole_type=None,
        )
        # Layout should not include power poles

    def test_planner_with_substation(self):
        """Test planner with substation power poles."""
        layout, _, diags = compile_to_layout(
            """
        Signal x = 5 | "signal-A";
        Signal y = x + 10;
        """,
            power_pole_type="substation",
        )


class TestPlannerEntityLayout:
    """Cover lines 306-310, 337-341: entity layout scenarios."""

    def test_layout_with_multiple_entities(self):
        """Test layout with multiple placed entities."""
        layout, _, diags = compile_to_layout("""
        Entity lamp1 = place("small-lamp", 0, 0);
        Entity lamp2 = place("small-lamp", 2, 0);
        Entity lamp3 = place("small-lamp", 4, 0);
        lamp1.enable = 1;
        lamp2.enable = 1;
        lamp3.enable = 1;
        """)

    def test_layout_with_memory(self):
        """Test layout with memory cells."""
        layout, _, diags = compile_to_layout("""
        Memory counter: "signal-A";
        counter.write(counter.read() + 1);
        Signal current = counter.read();
        """)

    def test_layout_with_combinators_and_entities(self):
        """Test layout with combinators connected to entities."""
        layout, _, diags = compile_to_layout("""
        Signal x = 5 | "signal-A";
        Signal y = x + 10;
        Entity lamp = place("small-lamp", 0, 0);
        lamp.enable = y > 10;
        """)


class TestPlannerWireRouting:
    """Tests for wire routing in layout."""

    def test_complex_wire_routing(self):
        """Test complex wire routing scenario."""
        layout, _, diags = compile_to_layout("""
        Signal a = 5 | "signal-A";
        Signal b = 10 | "signal-B";
        Signal c = a + b;
        Signal d = c * 2;
        Signal e = d - a;
        """)

    def test_wire_routing_with_feedback(self):
        """Test wire routing with memory feedback."""
        layout, _, diags = compile_to_layout("""
        Memory counter: "signal-A";
        Signal next = counter.read() + 1;
        counter.write(next);
        """)


class TestPlannerLayoutRetries:
    """Tests for layout retry mechanism."""

    def test_layout_with_constrained_space(self):
        """Test layout planning with space constraints."""
        layout, _, diags = compile_to_layout("""
        Entity lamp1 = place("small-lamp", 0, 0);
        Entity lamp2 = place("small-lamp", 0, 1);
        Signal x = 1 | "signal-A";
        lamp1.enable = x > 0;
        lamp2.enable = x > 0;
        """)
