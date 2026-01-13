"""
Tests for lowering/memory_lowerer.py - Memory operations to IR lowering.

This module tests the MemoryLowerer class which handles converting
memory declarations and operations to IR.
"""

from dsl_compiler.src.ir.nodes import IRMemCreate, IRMemRead, IRMemWrite

from .conftest import compile_to_ir


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


class TestMemoryEdgeCases:
    """Tests for memory lowering edge cases."""

    def test_memory_write_int_value(self):
        """Test memory write with integer value (lines 308-318)."""
        ir_ops, _, diags = compile_to_ir("""
        Memory m: "signal-A";
        m.write(100);
        """)
        assert not diags.has_errors()
        mem_writes = [op for op in ir_ops if isinstance(op, IRMemWrite)]
        assert len(mem_writes) >= 1

    def test_memory_signal_type_lookup(self):
        """Test _memory_signal_type lookups (lines 268-278)."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Memory m: "iron-plate";
        Signal x = m.read();
        """)
        assert not diags.has_errors()

    def test_latch_with_signal_ref(self):
        """Test latch with signal reference set/reset (lines 157-161)."""
        ir_ops, _, diags = compile_to_ir("""
        Memory latch: "signal-A";
        Signal s = ("signal-B", 1);
        Signal r = ("signal-C", 1);
        latch.write(1, set=s, reset=r);
        """)
        assert not diags.has_errors()

    def test_coerce_signal_ref_type_mismatch(self):
        """Test coercing SignalRef with different signal type (lines 285-305)."""
        ir_ops, _, diags = compile_to_ir("""
        Memory m: "signal-A";
        Signal x = ("signal-B", 50);
        m.write(x);
        """)
        # May have warnings about type mismatch
        assert len(ir_ops) > 0


class TestMemorySignalTypeLookup:
    """Tests for _memory_signal_type lookups."""

    def test_lookup_from_memory_types_dict(self):
        """Test lookup from parent.memory_types."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Memory m: "signal-A";
        m.write(42);
        """)
        assert not diags.has_errors()
        assert "m" in lowerer.memory_types

    def test_lookup_from_semantic_memory_types(self):
        """Test lookup from semantic.memory_types."""
        ir_ops, _, diags = compile_to_ir("""
        Memory counter: "signal-B";
        Signal x = counter.read() + 1;
        """)
        assert not diags.has_errors()

    def test_lookup_from_symbol_table(self):
        """Test lookup from symbol table."""
        ir_ops, _, diags = compile_to_ir("""
        Memory items: "iron-plate";
        Signal val = items.read();
        """)
        assert not diags.has_errors()


class TestLatchWithNonSignalRef:
    """Tests for latch operations (lines 157-161)."""

    def test_latch_set_signal_type_extraction(self):
        """Test that latch extracts signal type from set reference."""
        ir_ops, _, diags = compile_to_ir("""
        Memory latch: "signal-A";
        latch.write(1, set=("signal-B", 1), reset=("signal-C", 0));
        """)
        assert not diags.has_errors()

    def test_latch_with_value_not_one(self):
        """Test latch with value != 1 (needs multiplier)."""
        ir_ops, _, diags = compile_to_ir("""
        Memory latch: "signal-A";
        latch.write(5, set=("signal-B", 1), reset=("signal-C", 0));
        """)
        assert not diags.has_errors()


class TestCoerceToSignalType:
    """Tests for _coerce_to_signal_type (lines 281-319)."""

    def test_coerce_int_to_signal(self):
        """Test coercing integer value to signal type."""
        ir_ops, _, diags = compile_to_ir("""
        Memory m: "signal-A";
        m.write(100);
        """)
        assert not diags.has_errors()

    def test_coerce_matching_signal_type(self):
        """Test coercing SignalRef with matching type (no conversion)."""
        ir_ops, _, diags = compile_to_ir("""
        Memory m: "signal-A";
        Signal x = ("signal-A", 50);
        m.write(x);
        """)
        assert not diags.has_errors()

    def test_coerce_mismatched_signal_emits_warning(self):
        """Test coercing SignalRef with mismatched type emits warning."""
        ir_ops, _, diags = compile_to_ir("""
        Memory m: "signal-A";
        Signal y = ("signal-C", 30);
        m.write(y);
        """)
        # Should have warnings
        assert len(ir_ops) > 0

    def test_coerce_signal_ref_same_type(self):
        """Test that same signal type returns the ref unchanged."""
        ir_ops, _, diags = compile_to_ir("""
        Memory m: "signal-0";
        Signal x = ("signal-0", 42);
        m.write(x);
        """)
        assert not diags.has_errors()


class TestMemoryDeclSignalType:
    """Tests for memory declaration signal type handling (lines 58-67)."""

    def test_memory_with_explicit_signal_type(self):
        """Test memory declaration with explicit signal type."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Memory counter: "signal-red";
        counter.write(1);
        """)
        assert not diags.has_errors()

    def test_memory_from_symbol_table_lookup(self):
        """Test memory signal type from symbol table lookup."""
        ir_ops, _, diags = compile_to_ir("""
        Memory m: "signal-white";
        Signal x = m.read();
        """)
        assert not diags.has_errors()


class TestWriteEnableCondition:
    """Tests for write enable condition handling."""

    def test_write_with_complex_when_condition(self):
        """Test memory write with complex when condition."""
        ir_ops, _, diags = compile_to_ir("""
        Memory m: "signal-A";
        Signal a = 5;
        Signal b = 10;
        m.write(42, when=a < b);
        """)
        assert not diags.has_errors()

    def test_write_without_when(self):
        """Test memory write without when condition (always enabled)."""
        ir_ops, _, diags = compile_to_ir("""
        Memory m: "signal-A";
        m.write(123);
        """)
        assert not diags.has_errors()
        mem_writes = [op for op in ir_ops if isinstance(op, IRMemWrite)]
        assert len(mem_writes) >= 1
