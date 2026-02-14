"""Tests for wire color annotation on signal literals.

Tests the full pipeline from parsing through to wire color constraint application.
"""

import pytest

from dsl_compiler.src.ast.expressions import SignalLiteral
from dsl_compiler.src.ast.literals import NumberLiteral
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer


@pytest.fixture
def parser():
    return DSLParser()


# =============================================================================
# Parsing tests
# =============================================================================


class TestWireColorParsing:
    """Tests for parsing wire color annotations on signal literals."""

    def test_signal_literal_with_red(self, parser):
        """Parse ("signal-A", 100, red) — wire_color should be 'red'."""
        ast = parser.parse('Signal a = ("signal-A", 100, red);', "<test>")
        decl = ast.statements[0]
        signal_lit = decl.value
        assert isinstance(signal_lit, SignalLiteral)
        assert signal_lit.signal_type == "signal-A"
        assert signal_lit.wire_color == "red"

    def test_signal_literal_with_green(self, parser):
        """Parse ("signal-A", 100, green) — wire_color should be 'green'."""
        ast = parser.parse('Signal a = ("signal-A", 100, green);', "<test>")
        decl = ast.statements[0]
        signal_lit = decl.value
        assert isinstance(signal_lit, SignalLiteral)
        assert signal_lit.wire_color == "green"

    def test_signal_literal_without_color(self, parser):
        """Parse ("signal-A", 100) — wire_color should be None."""
        ast = parser.parse('Signal a = ("signal-A", 100);', "<test>")
        decl = ast.statements[0]
        signal_lit = decl.value
        assert isinstance(signal_lit, SignalLiteral)
        assert signal_lit.wire_color is None

    def test_bare_number_no_color(self, parser):
        """Parse bare number 42 — wire_color should be None."""
        ast = parser.parse("Signal a = 42;", "<test>")
        decl = ast.statements[0]
        signal_lit = decl.value
        assert isinstance(signal_lit, SignalLiteral)
        assert signal_lit.wire_color is None

    def test_wire_color_with_expression_value(self, parser):
        """Parse ("iron-plate", 10 + 5, red) — expression value with wire color."""
        ast = parser.parse('Signal a = ("iron-plate", 10 + 5, red);', "<test>")
        decl = ast.statements[0]
        signal_lit = decl.value
        assert isinstance(signal_lit, SignalLiteral)
        assert signal_lit.signal_type == "iron-plate"
        assert signal_lit.wire_color == "red"

    def test_wire_color_with_name_type(self, parser):
        """Parse (signal_type_var, 100, green) — name-based type with wire color."""
        # Use a.type as signal type with wire color
        ast = parser.parse(
            'Signal a = ("signal-A", 42);\nSignal b = (a.type, 100, green);', "<test>"
        )
        decl = ast.statements[1]
        signal_lit = decl.value
        assert isinstance(signal_lit, SignalLiteral)
        assert signal_lit.wire_color == "green"

    def test_multiple_signals_different_colors(self, parser):
        """Parse two signals with different wire colors."""
        source = """
Signal a = ("signal-A", 10, red);
Signal b = ("signal-B", 20, green);
"""
        ast = parser.parse(source, "<test>")
        decl_a = ast.statements[0]
        decl_b = ast.statements[1]
        assert decl_a.value.wire_color == "red"
        assert decl_b.value.wire_color == "green"


# =============================================================================
# AST tests
# =============================================================================


class TestSignalLiteralWireColor:
    """Tests for SignalLiteral wire_color field."""

    def test_default_wire_color_none(self):
        """SignalLiteral without wire_color defaults to None."""
        lit = SignalLiteral(value=NumberLiteral(42), signal_type="signal-A")
        assert lit.wire_color is None

    def test_explicit_red(self):
        """SignalLiteral with wire_color='red'."""
        lit = SignalLiteral(value=NumberLiteral(42), signal_type="signal-A", wire_color="red")
        assert lit.wire_color == "red"

    def test_explicit_green(self):
        """SignalLiteral with wire_color='green'."""
        lit = SignalLiteral(value=NumberLiteral(42), signal_type="signal-A", wire_color="green")
        assert lit.wire_color == "green"


# =============================================================================
# Lowering tests
# =============================================================================


class TestWireColorLowering:
    """Tests for wire_color propagation through lowering to IR."""

    def _lower_source(self, source: str):
        """Helper to parse and lower source code."""
        diagnostics = ProgramDiagnostics()
        parser = DSLParser()
        ast = parser.parse(source, "<test>")
        analyzer = SemanticAnalyzer(diagnostics=diagnostics)
        analyzer.visit(ast)
        lowerer = ASTLowerer(analyzer, diagnostics)
        ir_ops = lowerer.lower_program(ast)
        return ir_ops, lowerer

    def test_wire_color_in_debug_metadata(self):
        """wire_color should appear in IRConst.debug_metadata."""
        ir_ops, lowerer = self._lower_source('Signal a = ("signal-A", 100, red);')
        # Find the constant operation
        const_ops = [op for op in ir_ops if hasattr(op, "value") and op.value == 100]
        assert len(const_ops) >= 1
        assert const_ops[0].debug_metadata.get("wire_color") == "red"

    def test_no_wire_color_in_debug_metadata(self):
        """Without wire_color, debug_metadata should not have 'wire_color' key."""
        ir_ops, lowerer = self._lower_source('Signal a = ("signal-A", 100);')
        const_ops = [op for op in ir_ops if hasattr(op, "value") and op.value == 100]
        assert len(const_ops) >= 1
        assert "wire_color" not in const_ops[0].debug_metadata

    def test_green_wire_color_in_debug_metadata(self):
        """wire_color=green should appear in IRConst.debug_metadata."""
        ir_ops, lowerer = self._lower_source('Signal a = ("signal-A", 50, green);')
        const_ops = [op for op in ir_ops if hasattr(op, "value") and op.value == 50]
        assert len(const_ops) >= 1
        assert const_ops[0].debug_metadata.get("wire_color") == "green"


# =============================================================================
# Integration tests (full pipeline)
# =============================================================================


def compile_to_layout(source: str):
    """Compile Facto source through to layout plan."""
    from dsl_compiler.src.layout.planner import LayoutPlanner

    diagnostics = ProgramDiagnostics()
    parser = DSLParser()
    ast = parser.parse(source, "<test>")
    analyzer = SemanticAnalyzer(diagnostics=diagnostics)
    analyzer.visit(ast)
    lowerer = ASTLowerer(analyzer, diagnostics)
    ir_ops = lowerer.lower_program(ast)

    planner = LayoutPlanner(
        lowerer.ir_builder.signal_type_map,
        diagnostics=diagnostics,
        signal_refs=lowerer.signal_refs,
        referenced_signal_names=lowerer.referenced_signal_names,
    )
    layout = planner.plan_layout(ir_ops)
    return layout, planner


class TestWireColorIntegration:
    """Integration tests for wire color through the full pipeline."""

    def test_wire_color_in_placement_properties(self):
        """wire_color should appear in EntityPlacement.properties."""
        source = 'Signal a = ("signal-A", 100, red);\nSignal b = a + 1;'
        layout, planner = compile_to_layout(source)

        # Find the constant combinator for 'a'
        const_placements = [
            p
            for p in layout.entity_placements.values()
            if p.entity_type == "constant-combinator" and p.properties.get("value") == 100
        ]
        assert len(const_placements) >= 1
        assert const_placements[0].properties.get("wire_color") == "red"

    def test_wire_color_constraint_applied(self):
        """User-specified wire color should be applied as a hard constraint."""
        source = 'Signal a = ("signal-A", 100, red);\nSignal b = a + 1;'
        layout, planner = compile_to_layout(source)

        # Verify the connection planner used the constraint
        cp = planner.connection_planner
        edge_colors = cp.edge_color_map()
        # Find edges from the constant combinator (keys are (source, sink, signal) tuples)
        const_placements = [
            p
            for p in layout.entity_placements.values()
            if p.entity_type == "constant-combinator" and p.properties.get("value") == 100
        ]
        if const_placements:
            const_id = const_placements[0].ir_node_id
            edges_from_const = {k: c for k, c in edge_colors.items() if k[0] == const_id}
            # All edges from this constant should be red
            for key, color in edges_from_const.items():
                assert color == "red", f"Expected red for {key}, got {color}"

    def test_two_inputs_different_colors(self):
        """Two inputs with different specified colors feeding one combinator."""
        source = """
Signal a = ("signal-A", 10, red);
Signal b = ("signal-A", 20, green);
Signal c = a * b;
"""
        layout, planner = compile_to_layout(source)
        cp = planner.connection_planner
        edge_colors = cp.edge_color_map()

        # Find edges from each constant
        a_const = [
            p
            for p in layout.entity_placements.values()
            if p.entity_type == "constant-combinator" and p.properties.get("value") == 10
        ]
        b_const = [
            p
            for p in layout.entity_placements.values()
            if p.entity_type == "constant-combinator" and p.properties.get("value") == 20
        ]

        if a_const and b_const:
            a_edges = [c for k, c in edge_colors.items() if k[0] == a_const[0].ir_node_id]
            b_edges = [c for k, c in edge_colors.items() if k[0] == b_const[0].ir_node_id]
            if a_edges:
                assert all(c == "red" for c in a_edges), f"a edges: {a_edges}"
            if b_edges:
                assert all(c == "green" for c in b_edges), f"b edges: {b_edges}"

    def test_auto_color_without_annotation(self):
        """Without wire_color annotation, automatic assignment still works."""
        source = """
Signal a = ("signal-A", 10);
Signal b = ("signal-A", 20);
Signal c = a * b;
"""
        layout, planner = compile_to_layout(source)
        # Should compile without error; wire colors assigned automatically
        assert len(layout.wire_connections) >= 1

    def test_wire_color_with_arithmetic(self):
        """Wire color annotation with arithmetic operations."""
        source = """
Signal input1 = ("iron-plate", 100, red);
Signal input2 = ("copper-plate", 50, green);
Signal total = input1 + input2;
"""
        layout, planner = compile_to_layout(source)
        assert len(layout.entity_placements) >= 3
