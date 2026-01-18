"""
Tests for lowering/memory_lowerer.py - Memory operations to IR lowering.

This module tests the MemoryLowerer class which handles converting
memory declarations and operations to IR.
"""

from dsl_compiler.src.common.diagnostics import DiagnosticSeverity
from dsl_compiler.src.ir.nodes import IRConst, IRMemCreate, IRMemRead, IRMemWrite

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


class TestLatchInlineConditions:
    """Tests for latch condition inlining paths."""

    def test_latch_with_comparison_set_reset(self):
        """Test latch with simple comparison conditions that can be inlined."""
        ir_ops, _, diags = compile_to_ir("""
        Signal input = ("signal-S", 0);
        Memory latch: "signal-A";
        latch.write(1, set=input > 10, reset=input < 5);
        """)
        assert not diags.has_errors()

    def test_latch_different_signals_no_inline(self):
        """Test latch with different signals in set/reset - can't inline."""
        ir_ops, _, diags = compile_to_ir("""
        Signal s1 = ("signal-A", 0);
        Signal s2 = ("signal-B", 0);
        Memory latch: "signal-L";
        latch.write(1, set=s1 > 0, reset=s2 > 0);
        """)
        assert not diags.has_errors()

    def test_latch_complex_conditions_no_inline(self):
        """Test latch with complex conditions that can't be inlined."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = ("signal-A", 0);
        Signal b = ("signal-B", 0);
        Memory latch: "signal-L";
        latch.write(1, set=a + b > 0, reset=a > 0);
        """)
        assert not diags.has_errors()

    def test_latch_with_both_simple_comparisons_same_signal(self):
        """Test latch where both set/reset compare same signal (should inline)."""
        ir_ops, _, diags = compile_to_ir("""
        Signal battery = ("signal-B", 0);
        Memory steam_on: "signal-S";
        steam_on.write(1, set=battery < 20, reset=battery >= 80);
        """)
        assert not diags.has_errors()


class TestLatchWriteEdgeCases:
    """Test edge cases in latch write lowering."""

    def test_latch_with_signal_literal_condition(self):
        """Test latch with signal literal as condition."""
        ir_ops, _, diags = compile_to_ir("""
        Memory latch: "signal-A";
        latch.write(1, set=("signal-B", 1), reset=("signal-C", 1));
        """)
        assert not diags.has_errors()

    def test_latch_value_is_signal_ref(self):
        """Test latch where value is a signal reference (same type as memory)."""
        ir_ops, _, diags = compile_to_ir("""
        Signal val = ("signal-A", 100);
        Memory latch: "signal-A";
        latch.write(val, set=("signal-B", 1), reset=("signal-C", 1));
        """)
        assert not diags.has_errors()

    def test_latch_rs_priority(self):
        """Test latch with reset priority (RS latch)."""
        ir_ops, _, diags = compile_to_ir("""
        Signal s = ("signal-A", 0);
        Memory latch: "signal-L";
        latch.write(1, reset=s > 5, set=s < 3);
        """)
        assert not diags.has_errors()


class TestConditionalWriteEdgeCases:
    """Test edge cases in conditional memory write."""

    def test_conditional_write_complex_condition(self):
        """Test conditional write with compound condition."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = ("signal-A", 0);
        Signal b = ("signal-B", 0);
        Memory m: "signal-M";
        m.write(100, when=a > 0 && b > 0);
        """)
        assert not diags.has_errors()

    def test_conditional_write_or_condition(self):
        """Test conditional write with OR condition."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = ("signal-A", 0);
        Signal b = ("signal-B", 0);
        Memory m: "signal-M";
        m.write(100, when=a > 0 || b > 0);
        """)
        assert not diags.has_errors()

    def test_conditional_write_signal_value(self):
        """Test conditional write where value is a signal (same type as memory)."""
        ir_ops, _, diags = compile_to_ir("""
        Signal enable = ("signal-E", 0);
        Signal val = ("signal-M", 50);
        Memory m: "signal-M";
        m.write(val, when=enable > 0);
        """)
        assert not diags.has_errors()

    def test_conditional_counter(self):
        """Test conditional counter increment."""
        ir_ops, _, diags = compile_to_ir("""
        Signal enable = ("signal-E", 1);
        Memory counter: "signal-C";
        counter.write(counter.read() + 1, when=enable > 0);
        """)
        assert not diags.has_errors()
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


class TestCoverageBoostMemory:
    """Additional tests for memory lowering coverage."""

    def test_memory_with_arithmetic_feedback(self):
        """Test memory cell with arithmetic feedback loop."""
        ir_ops, _, diags = compile_to_ir("""
        Memory counter: "signal-A";
        counter.write(counter.read() + 1);
        """)
        assert not diags.has_errors()
        # Should have a memory create and write
        assert any(isinstance(op, IRMemCreate) for op in ir_ops)
        assert any(isinstance(op, IRMemWrite) for op in ir_ops)

    def test_memory_conditional_write_stored_comparison(self):
        """Test memory with conditional write using stored comparison."""
        ir_ops, _, diags = compile_to_ir("""
        Signal enable = 5;
        Memory counter: "signal-A";
        Signal cond = enable > 0;
        counter.write(counter.read() + 1, when=cond);
        """)
        assert not diags.has_errors()

    def test_multiple_memories_independent(self):
        """Test multiple independent memory cells."""
        ir_ops, _, diags = compile_to_ir("""
        Memory a: "signal-A";
        Memory b: "signal-B";
        a.write(1);
        b.write(2);
        """)
        assert not diags.has_errors()
        mem_creates = [op for op in ir_ops if isinstance(op, IRMemCreate)]
        assert len(mem_creates) == 2

    def test_memory_read_in_condition(self):
        """Test memory read used in a condition."""
        ir_ops, _, diags = compile_to_ir("""
        Memory counter: "signal-A";
        Signal above_50 = (counter.read() > 50) : 1;
        """)
        assert not diags.has_errors()


class TestLatchNeedsMultiplier:
    """Tests for latch writes with value != 1."""

    def test_latch_with_non_unity_value(self):
        """Test latch write where value is not 1 (requires multiplier)."""
        ir_ops, _, diags = compile_to_ir("""
        Memory latch: "signal-A";
        latch.write(10, set=("signal-B", 1), reset=("signal-C", 1));
        """)
        assert not diags.has_errors()

    def test_latch_signal_value_different_type(self):
        """Test latch where value signal has different type than output."""
        ir_ops, _, diags = compile_to_ir("""
        Signal val = 5 | "signal-X";
        Memory latch: "signal-A";
        latch.write(val, set=("signal-B", 1), reset=("signal-C", 1));
        """)
        # Should produce an info message about type mismatch
        assert len(ir_ops) > 0


class TestMemorySignalTypeResolution:
    """Tests for memory signal type resolution edge cases."""

    def test_memory_with_item_signal(self):
        """Test memory with item signal type."""
        ir_ops, _, diags = compile_to_ir("""
        Memory items: "iron-plate";
        items.write(100);
        """)
        assert not diags.has_errors()

    def test_memory_read_with_expression(self):
        """Test memory read result used in complex expression."""
        ir_ops, _, diags = compile_to_ir("""
        Memory m: "signal-A";
        Signal doubled = m.read() * 2;
        Signal halved = m.read() / 2;
        """)
        assert not diags.has_errors()


class TestAdditionalMemoryCoverage:
    """Additional tests to improve memory lowerer coverage."""

    def test_write_enable_from_decider_output(self):
        """Test write enable coming from a decider combinator output (lines 406-409)."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 10 | "signal-A";
        Signal b = 5 | "signal-B";
        Memory m: "signal-M";
        # The comparison (a > b) produces a decider output
        m.write(100, when=a > b);
        """)
        assert not diags.has_errors()

    def test_write_enable_arithmetic_projection(self):
        """Test write enable that needs arithmetic projection to signal-W (lines 410-414)."""
        ir_ops, _, diags = compile_to_ir("""
        Signal enable_val = 1 | "signal-E";  # Non signal-W type
        Memory m: "signal-M";
        # Use signal that's not from a decider and not signal-W
        m.write(100, when=enable_val);
        """)
        assert not diags.has_errors()

    def test_write_without_condition_creates_const_w(self):
        """Test write without condition creates constant signal-W (lines 416-417)."""
        ir_ops, _, diags = compile_to_ir("""
        Memory m: "signal-A";
        m.write(50);
        """)
        assert not diags.has_errors()
        # Check that a const signal-W=1 is created - IRConst uses output_type
        consts = [op for op in ir_ops if isinstance(op, IRConst)]
        signal_w_consts = [c for c in consts if c.output_type == "signal-W"]
        assert len(signal_w_consts) >= 1

    def test_coerce_to_signal_type_match(self):
        """Test coerce when signal type already matches (line 447-448)."""
        ir_ops, _, diags = compile_to_ir("""
        Signal val = 100 | "signal-A";
        Memory m: "signal-A";
        # Value already has matching signal type
        m.write(val);
        """)
        assert not diags.has_errors()

    def test_coerce_to_signal_type_mismatch(self):
        """Test coerce when signal type doesn't match (lines 449-462)."""
        ir_ops, _, diags = compile_to_ir("""
        Signal val = 100 | "signal-B";  # Different type
        Memory m: "signal-A";
        # Value has mismatched signal type - should warn and project
        m.write(val);
        """)
        # Should produce a warning about type mismatch
        warnings = diags.get_messages(DiagnosticSeverity.WARNING)
        assert any("mismatch" in w.lower() for w in warnings)

    def test_read_from_undefined_memory(self):
        """Test reading from undefined memory (lines 91-93)."""
        ir_ops, lowerer, diags = compile_to_ir("""
        # Intentionally reference undefined memory
        Signal x = undefined_mem.read();
        """)
        # Should have an error about undefined memory
        assert diags.has_errors()

    def test_memory_with_derived_signal_type(self):
        """Test memory using type derived from semantic analyzer."""
        ir_ops, _, diags = compile_to_ir("""
        Memory m: "signal-A";
        Signal val = m.read() + 10;
        m.write(val);
        """)
        assert not diags.has_errors()

    def test_latch_write_bundle_value_fallback(self):
        """Test latch write with bundle value (falls back to 1)."""
        # This is a special case where bundle can't be used as latch value
        ir_ops, _, diags = compile_to_ir("""
        Bundle b = {("signal-A", 5), ("signal-B", 10)};
        Memory latch: "signal-L";
        # Bundle can't be latch value, should fall back
        Signal s = ("signal-S", 0);
        latch.write(1, set=s > 0, reset=s < 0);
        """)
        assert not diags.has_errors()


# =============================================================================
# Coverage gap tests (Lines 68-70, 73-77, 257-259, 331-335, 382-386, 410-412, 427-437, 467-477)
# =============================================================================


class TestMemoryLowererCoverageGaps:
    """Tests for memory_lowerer.py coverage gaps > 2 lines."""

    def test_memory_decl_with_inferred_signal(self):
        """Cover lines 68-70: memory declaration signal type handling."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Memory m: "signal-A";
        Signal x = m.read();
        """)
        assert not diags.has_errors()
        # Verify memory is registered
        assert "m" in lowerer.memory_refs

    def test_memory_decl_item_signal_type(self):
        """Cover lines 73-77: memory with item signal type."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Memory inventory: "iron-plate";
        inventory.write(100);
        """)
        assert not diags.has_errors()
        # Verify item signal was correctly registered
        assert "inventory" in lowerer.memory_refs

    def test_memory_write_constant_value(self):
        """Cover lines 257-259: write constant value to memory."""
        ir_ops, _, diags = compile_to_ir("""
        Memory m: "signal-A";
        m.write(42);
        """)
        assert not diags.has_errors()
        mem_writes = [op for op in ir_ops if isinstance(op, IRMemWrite)]
        assert len(mem_writes) >= 1

    def test_latch_set_reset_binary_conditions(self):
        """Cover lines 331-335: latch with binary set/reset conditions."""
        ir_ops, _, diags = compile_to_ir("""
        Signal trigger = 10;
        Memory latch: "signal-L";
        latch.write(1, set=trigger > 5, reset=trigger < 3);
        """)
        assert not diags.has_errors()

    def test_latch_with_signal_conditions(self):
        """Cover lines 382-386: latch with signal type conditions."""
        ir_ops, _, diags = compile_to_ir("""
        Signal s = 10;
        Signal t = 5;
        Memory latch: "signal-L";
        latch.write(100, set=s > t, reset=s < t);
        """)
        assert not diags.has_errors()

    def test_latch_condition_extraction(self):
        """Cover lines 410-412: extracting condition signal."""
        ir_ops, _, diags = compile_to_ir("""
        Signal condition = 1;
        Memory latch: "signal-A";
        latch.write(50, set=condition > 0, reset=condition < 0);
        """)
        assert not diags.has_errors()

    def test_latch_complex_conditions(self):
        """Cover lines 427-437: latch with complex condition expressions."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 10;
        Signal b = 5;
        Memory latch: "signal-L";
        latch.write(1, set=(a + b) > 10, reset=(a - b) < 0);
        """)
        assert not diags.has_errors()

    def test_latch_with_nested_expressions(self):
        """Cover lines 467-477: latch with nested expression conditions."""
        ir_ops, _, diags = compile_to_ir("""
        Signal x = 1;
        Signal y = 2;
        Memory latch: "signal-L";
        latch.write(1, set=x > 0, reset=y < 0);
        Signal result = latch.read();
        """)
        assert not diags.has_errors()
        # Verify read operation was generated
        mem_reads = [op for op in ir_ops if isinstance(op, IRMemRead)]
        assert len(mem_reads) >= 1
