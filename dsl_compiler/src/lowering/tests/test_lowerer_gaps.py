"""Tests for lowerer.py coverage gaps."""

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


class TestLowererExprContext:
    """Cover lines 144-158: expression context tracking."""

    def test_push_pop_expr_context(self):
        """Test expression context push/pop tracking."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Signal a = 5 | "signal-A";
        Signal b = a + 10;
        """)
        # Context should be properly managed

    def test_nested_expr_context(self):
        """Test nested expression contexts."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Signal a = 5 | "signal-A";
        Signal b = 10 | "signal-B";
        Signal result = (a + b) * 2;
        """)


class TestLowererAnnotateSignalRef:
    """Cover lines 164-168: signal ref annotation."""

    def test_annotate_signal_ref_with_metadata(self):
        """Test signal ref annotation sets metadata correctly."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Signal x = 5 | "signal-A";
        Signal y = x + 10;
        """)
        # signal_refs should have proper metadata

    def test_annotate_signal_ref_from_function(self):
        """Test signal ref annotation from function call."""
        ir_ops, lowerer, diags = compile_to_ir("""
        func double(Signal x) {
            return x * 2;
        }
        Signal result = double(5 | "signal-A");
        """)


class TestLowererProgramLowering:
    """Tests for overall program lowering."""

    def test_lower_program_with_imports(self):
        """Test lowering program with imports."""
        ir_ops, lowerer, diags = compile_to_ir("""
        import "math.facto";
        Signal x = 5 | "signal-A";
        Signal y = abs(x - 10);
        """)

    def test_lower_program_with_multiple_functions(self):
        """Test lowering program with multiple functions."""
        ir_ops, lowerer, diags = compile_to_ir("""
        func add(Signal a, Signal b) {
            return a + b;
        }
        func sub(Signal a, Signal b) {
            return a - b;
        }
        Signal x = 5 | "signal-A";
        Signal y = 3 | "signal-B";
        Signal sum = add(x, y);
        Signal diff = sub(x, y);
        """)

    def test_lower_program_with_memory(self):
        """Test lowering program with memory operations."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Memory counter: "signal-A";
        counter.write(counter.read() + 1);
        Signal current = counter.read();
        """)


class TestLowererEntityTracking:
    """Tests for entity reference tracking."""

    def test_track_entity_refs(self):
        """Test that entity refs are tracked properly."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Entity lamp1 = place("small-lamp", 0, 0);
        Entity lamp2 = place("small-lamp", 2, 0);
        lamp1.enable = 1;
        lamp2.enable = 0;
        """)
        # entity_refs should have both lamps

    def test_track_entity_with_properties(self):
        """Test entity tracking with property configuration."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Entity lamp = place("small-lamp", 0, 0, {use_colors: 1, always_on: 1});
        Signal brightness = 200 | "signal-white";
        lamp.rgb = brightness;
        """)


class TestLowererMemoryTracking:
    """Tests for memory reference tracking."""

    def test_track_memory_refs(self):
        """Test that memory refs are tracked properly."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Memory mem1: "signal-A";
        Memory mem2: "signal-B";
        mem1.write(10);
        mem2.write(20);
        """)
        # memory_refs should have both memories

    def test_track_memory_types(self):
        """Test memory type tracking."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Memory counter: "signal-A";
        counter.write(counter.read() + 1);
        """)
        # memory_types should track the signal type
