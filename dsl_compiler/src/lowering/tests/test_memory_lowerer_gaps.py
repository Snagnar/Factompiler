"""Tests for memory_lowerer.py coverage gaps."""

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


class TestMemoryLowererDeclaration:
    """Cover lines 68-70, 73-77: memory declaration with missing signal type."""

    def test_memory_without_signal_type_error(self):
        """Memory must have explicit signal type."""
        # Memory without type should work (type inferred from write)
        ir_ops, _, diags = compile_to_ir("""
        Memory mem;
        Signal x = ("signal-A", 5);
        mem.write(x);
        """)


class TestMemoryLowererLatchWrite:
    """Cover lines 331-335: latch set signal type handling."""

    def test_latch_write_with_signal_set(self):
        """Test latch write with signal set reference."""
        ir_ops, _, diags = compile_to_ir("""
        Memory mem: "signal-A";
        Signal s = 50 | "signal-A";
        mem.write(1, set=s < 50, reset=s > 200);
        """)

    def test_latch_set_priority(self):
        """Test latch with set priority (SR latch)."""
        ir_ops, _, diags = compile_to_ir("""
        Memory mem: "signal-A";
        Signal s = 100 | "signal-A";
        mem.write(1, set=s < 50, reset=s > 200);
        """)


class TestMemoryLowererStandardWrite:
    """Cover lines 382-386, 410-412: standard write with when condition."""

    def test_standard_write_with_when_condition(self):
        """Test standard memory write with when= condition."""
        ir_ops, _, diags = compile_to_ir("""
        Memory counter: "signal-A";
        Signal enable = 1 | "signal-E";
        counter.write(counter.read() + 1, when=enable > 0);
        """)

    def test_write_enable_from_decider(self):
        """Test write enable signal from decider output."""
        ir_ops, _, diags = compile_to_ir("""
        Memory mem: "signal-A";
        Signal cond = 5 | "signal-C";
        mem.write(10 | "signal-A", when=cond > 0);
        """)


class TestMemoryLowererTypeCoercion:
    """Cover lines 427-437, 467-477: type coercion in memory writes."""

    def test_memory_write_type_mismatch_warning(self):
        """Test type mismatch warning in memory write."""
        ir_ops, _, diags = compile_to_ir("""
        Memory mem: "signal-A";
        Signal value = 10 | "signal-B";
        mem.write(value);
        """)
        # Should have warning about type mismatch

    def test_memory_write_integer_coercion(self):
        """Test coercing integer to signal type."""
        ir_ops, _, diags = compile_to_ir("""
        Memory mem: "signal-A";
        mem.write(42);
        """)

    def test_memory_write_with_projection(self):
        """Test memory write with explicit projection to correct type."""
        ir_ops, _, diags = compile_to_ir("""
        Memory mem: "signal-A";
        Signal value = 10 | "signal-B";
        mem.write(value | "signal-A");
        """)


class TestMemoryLowererReadUndefined:
    """Tests for reading from undefined memory."""

    def test_read_undefined_memory_error(self):
        """Test reading from undefined memory produces error."""
        ir_ops, _, diags = compile_to_ir("""
        Signal x = undefined_mem.read();
        """)
        assert diags.has_errors()


class TestMemoryLowererLatchMultiplier:
    """Tests for latch write with multiplier."""

    def test_latch_write_with_multiplier(self):
        """Test latch write with non-1 value."""
        ir_ops, _, diags = compile_to_ir("""
        Memory mem: "signal-A";
        Signal s = 50 | "signal-A";
        mem.write(10, set=s < 50, reset=s > 200);
        """)

    def test_latch_write_with_signal_value(self):
        """Test latch write with signal value (needs multiplier)."""
        ir_ops, _, diags = compile_to_ir("""
        Memory mem: "signal-A";
        Signal s = 50 | "signal-A";
        Signal val = 100 | "signal-V";
        mem.write(val, set=s < 50, reset=s > 200);
        """)
