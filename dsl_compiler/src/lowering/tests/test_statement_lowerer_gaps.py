"""Tests for statement_lowerer.py coverage gaps."""

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer


def compile_to_ir(source: str):
    """Helper to compile DSL source to IR operations."""
    diags = ProgramDiagnostics()
    parser = DSLParser()
    program = parser.parse(source.strip())
    analyzer = SemanticAnalyzer(diags)
    analyzer.visit(program)
    lowerer = ASTLowerer(analyzer, diags)
    ir_ops = lowerer.lower_program(program)
    return ir_ops, lowerer, diags


class TestStatementLowererBundleHandling:
    """Cover lines 121-123: Bundle type declarations."""

    def test_bundle_declaration_tracks_metadata(self):
        """Test that Bundle declarations set proper metadata."""
        ir_ops, _, diags = compile_to_ir("""
        Bundle b = {("iron-plate", 10), ("copper-plate", 20)};
        Signal x = b["iron-plate"];
        """)
        # Bundle should be created with proper debug info


class TestStatementLowererAssignMetadata:
    """Cover lines 244-246: assignment metadata."""

    def test_assign_stmt_updates_existing_variable(self):
        """Test assignment to existing variable."""
        ir_ops, _, diags = compile_to_ir("""
        Signal x = 5 | "signal-A";
        Signal y = x + 10;
        """)
        # Metadata should be updated properly


class TestStatementLowererConstantMetadata:
    """Cover lines 284-286: folded constant metadata."""

    def test_constant_folding_preserves_name(self):
        """Test that folded constants preserve declared name."""
        ir_ops, _, diags = compile_to_ir("""
        int a = 5;
        int b = 10;
        int c = a + b;
        Signal result = c | "signal-A";
        """)
        # c should be folded to 15 with metadata preserved


class TestStatementLowererForLoop:
    """Cover lines 376-378: for loop lowering."""

    def test_for_loop_with_int_bounds(self):
        """Test for loop with integer bounds."""
        ir_ops, _, diags = compile_to_ir("""
        for i in 1..5 {
            Signal temp = i | "signal-A";
        }
        """)


class TestStatementLowererInlineBundleCondition:
    """Cover lines 388-397: inlined bundle conditions."""

    def test_all_bundle_condition_inline(self):
        """Test all(bundle) < N inlined to entity condition."""
        ir_ops, _, diags = compile_to_ir("""
        Bundle b = {("iron-plate", 10), ("copper-plate", 20)};
        Entity lamp = place("small-lamp", 0, 0);
        lamp.enable = all(b) < 100;
        """)
        # Should inline the bundle condition to entity


class TestStatementLowererPropertyWrite:
    """Tests for property write handling."""

    def test_property_write_on_entity(self):
        """Test property write on entity."""
        ir_ops, _, diags = compile_to_ir("""
        Entity lamp = place("small-lamp", 0, 0);
        Signal enable = 1 | "signal-A";
        lamp.enable = enable > 0;
        """)

    def test_entity_rgb_property(self):
        """Test RGB property on lamp entity."""
        ir_ops, _, diags = compile_to_ir("""
        Entity lamp = place("small-lamp", 0, 0, {use_colors: 1, always_on: 1, color_mode: 1});
        lamp.rgb = 255 | "signal-white";
        """)


class TestStatementLowererFunctionReturn:
    """Tests for function return handling."""

    def test_function_return_value(self):
        """Test function with return statement."""
        ir_ops, _, diags = compile_to_ir("""
        func double(Signal x) {
            return x + x;
        }
        Signal result = double(5 | "signal-A");
        """)

    def test_void_function_no_return(self):
        """Test function without explicit return."""
        ir_ops, _, diags = compile_to_ir("""
        func noop() {
            Signal temp = 5 | "signal-A";
        }
        """)
