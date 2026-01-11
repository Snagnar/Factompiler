"""
Tests for lowering/memory_lowerer.py - Memory operations to IR lowering.

This module tests the MemoryLowerer class which handles converting
memory declarations and operations to IR.
"""

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.ir.nodes import IRMemCreate, IRMemRead, IRMemWrite
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer


def compile_to_ir(source: str):
    """Helper to compile source to IR operations."""
    parser = DSLParser()
    diagnostics = ProgramDiagnostics()
    analyzer = SemanticAnalyzer(diagnostics)
    program = parser.parse(source)
    analyzer.visit(program)
    lowerer = ASTLowerer(analyzer, diagnostics)
    ir_ops = lowerer.lower_program(program)
    return ir_ops, lowerer, diagnostics


class TestLowerMemDecl:
    """Tests for lower_mem_decl method."""

    def test_lower_memory_decl(self):
        """Test lowering a simple memory declaration."""
        ir_ops, lowerer, diags = compile_to_ir('Memory counter: "signal-A";')
        assert not diags.has_errors()
        assert "counter" in lowerer.memory_refs
        # Should have created memory
        mem_creates = [op for op in ir_ops if isinstance(op, IRMemCreate)]
        assert len(mem_creates) == 1

    def test_lower_memory_decl_item_signal(self):
        """Test lowering memory with item signal type."""
        ir_ops, lowerer, diags = compile_to_ir('Memory items: "iron-plate";')
        assert not diags.has_errors()
        assert "items" in lowerer.memory_refs


class TestLowerReadExpr:
    """Tests for lower_read_expr method."""

    def test_lower_memory_read(self):
        """Test lowering a memory read expression."""
        ir_ops, _, diags = compile_to_ir("""
        Memory m: "signal-A";
        Signal x = m.read();
        """)
        assert not diags.has_errors()
        mem_reads = [op for op in ir_ops if isinstance(op, IRMemRead)]
        assert len(mem_reads) >= 1

    def test_lower_memory_read_in_expression(self):
        """Test memory read used in an expression."""
        ir_ops, _, diags = compile_to_ir("""
        Memory counter: "signal-A";
        Signal doubled = counter.read() * 2;
        """)
        assert not diags.has_errors()


class TestLowerWriteExpr:
    """Tests for lower_write_expr method."""

    def test_lower_simple_memory_write(self):
        """Test lowering a simple memory write."""
        ir_ops, _, diags = compile_to_ir("""
        Memory m: "signal-A";
        m.write(42);
        """)
        assert not diags.has_errors()
        mem_writes = [op for op in ir_ops if isinstance(op, IRMemWrite)]
        assert len(mem_writes) >= 1

    def test_lower_memory_write_with_feedback(self):
        """Test lowering memory write with read feedback."""
        ir_ops, _, diags = compile_to_ir("""
        Memory counter: "signal-A";
        counter.write(counter.read() + 1);
        """)
        assert not diags.has_errors()
        mem_writes = [op for op in ir_ops if isinstance(op, IRMemWrite)]
        assert len(mem_writes) >= 1

    def test_lower_memory_write_with_when(self):
        """Test lowering memory write with when condition."""
        ir_ops, _, diags = compile_to_ir("""
        Memory m: "signal-A";
        Signal enable = 1;
        m.write(10, when=enable > 0);
        """)
        assert not diags.has_errors()

    def test_lower_sr_latch_write(self):
        """Test lowering SR latch memory write."""
        ir_ops, _, diags = compile_to_ir("""
        Memory latch: "signal-A";
        latch.write(1, set=("signal-B", 1), reset=("signal-C", 1));
        """)
        assert not diags.has_errors()

    def test_lower_rs_latch_write(self):
        """Test lowering RS latch memory write (reset priority)."""
        ir_ops, _, diags = compile_to_ir("""
        Memory latch: "signal-A";
        latch.write(1, reset=("signal-B", 1), set=("signal-C", 1));
        """)
        assert not diags.has_errors()


class TestMemoryFeedbackPatterns:
    """Tests for memory feedback loop patterns."""

    def test_counter_pattern(self):
        """Test counter pattern: mem = mem + 1."""
        ir_ops, _, diags = compile_to_ir("""
        Memory counter: "signal-A";
        counter.write(counter.read() + 1);
        """)
        assert not diags.has_errors()

    def test_accumulator_pattern(self):
        """Test accumulator pattern: mem = mem + input."""
        ir_ops, _, diags = compile_to_ir("""
        Memory sum: "signal-A";
        Signal input = 5;
        sum.write(sum.read() + input);
        """)
        assert not diags.has_errors()

    def test_conditional_counter(self):
        """Test conditional counter with when clause."""
        ir_ops, _, diags = compile_to_ir("""
        Memory counter: "signal-A";
        Signal enable = 1;
        counter.write(counter.read() + 1, when=enable > 0);
        """)
        assert not diags.has_errors()
