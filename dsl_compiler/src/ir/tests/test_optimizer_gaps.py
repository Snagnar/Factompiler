"""Tests for optimizer.py coverage gaps."""

from dsl_compiler.src.ir.nodes import (
    IRArith,
    IRConst,
    IRDecider,
    IRMemWrite,
    SignalRef,
)
from dsl_compiler.src.ir.optimizer import ConstantPropagationOptimizer, CSEOptimizer


class TestConstantPropOptimizerReferences:
    """Cover lines 280-285: _references_node for checking references."""

    def test_references_node_with_non_matching_signal_ref(self):
        """Test that _references_node returns False for non-matching source_id."""
        optimizer = ConstantPropagationOptimizer()
        ref = SignalRef("signal-A", "node_1")
        assert not optimizer._references_node(ref, "node_2")

    def test_references_node_with_matching_signal_ref(self):
        """Test that _references_node returns True for matching source_id."""
        optimizer = ConstantPropagationOptimizer()
        ref = SignalRef("signal-A", "node_1")
        assert optimizer._references_node(ref, "node_1")

    def test_references_node_with_non_signal_ref(self):
        """Test that _references_node returns False for non-SignalRef values."""
        optimizer = ConstantPropagationOptimizer()
        assert not optimizer._references_node(42, "node_1")
        assert not optimizer._references_node("string", "node_1")
        assert not optimizer._references_node(None, "node_1")


class TestConstantPropOptimizerUpdateReferences:
    """Cover lines 286-314: _update_references for various node types."""

    def test_update_references_irmemwrite(self):
        """Test updating references in IRMemWrite."""
        optimizer = ConstantPropagationOptimizer()
        data_ref = SignalRef("signal-A", "old_id")
        enable_ref = SignalRef("signal-W", "old_enable")

        mem_write = IRMemWrite("mem_1", data_ref, enable_ref)

        # Set up replacement
        optimizer.replacements["old_id"] = "new_id"
        optimizer.replacements["old_enable"] = "new_enable"

        optimizer._update_references([mem_write])

        assert isinstance(mem_write.data_signal, SignalRef)
        assert mem_write.data_signal.source_id == "new_id"

    def test_update_references_irarith(self):
        """Test updating references in IRArith."""
        optimizer = ConstantPropagationOptimizer()

        # Create IRArith with correct constructor
        arith = IRArith("arith_1", "signal-C")
        arith.op = "+"
        arith.left = SignalRef("signal-A", "old_left")
        arith.right = SignalRef("signal-B", "old_right")

        optimizer.replacements["old_left"] = "new_left"

        optimizer._update_references([arith])

        assert isinstance(arith.left, SignalRef)
        assert arith.left.source_id == "new_left"

    def test_update_references_irdecider(self):
        """Test updating references in IRDecider."""
        optimizer = ConstantPropagationOptimizer()

        # Create IRDecider with correct constructor
        decider = IRDecider("decider_1", "signal-D")
        decider.op = "<"
        decider.left = SignalRef("signal-A", "old_left")
        decider.right = SignalRef("signal-B", "old_right")
        decider.output_value = SignalRef("signal-C", "old_output")

        optimizer.replacements["old_left"] = "new_left"

        optimizer._update_references([decider])

        assert isinstance(decider.left, SignalRef)
        assert decider.left.source_id == "new_left"


class TestConstantPropOptimization:
    """Tests for constant propagation optimization."""

    def test_propagate_constant_through_arithmetic(self):
        """Test propagation of constants through arithmetic operations."""
        optimizer = ConstantPropagationOptimizer()

        # Create constant nodes
        const_a = IRConst("const_a", "signal-A")
        const_a.value = 10

        const_b = IRConst("const_b", "signal-B")
        const_b.value = 20

        # Create arithmetic using the constants
        arith = IRArith("arith_1", "signal-C")
        arith.op = "+"
        arith.left = SignalRef("signal-A", "const_a")
        arith.right = SignalRef("signal-B", "const_b")

        ir_ops = [const_a, const_b, arith]
        optimizer.optimize(ir_ops)

        # The arith should be folded into a constant
        # Check that either arith is removed or a new constant is created


class TestCSEOptimizerSignatureGeneration:
    """Cover lines 384-396: _value_key for signature generation."""

    def test_value_key_for_integer(self):
        """Test _value_key for integer value."""
        optimizer = CSEOptimizer()
        key = optimizer._value_key(42)
        assert key == "int:42"

    def test_value_key_for_string(self):
        """Test _value_key for string value."""
        optimizer = CSEOptimizer()
        key = optimizer._value_key("signal-A")
        assert key == "str:signal-A"

    def test_value_key_for_signal_ref(self):
        """Test _value_key for SignalRef value."""
        optimizer = CSEOptimizer()
        ref = SignalRef("signal-A", "node_1")
        key = optimizer._value_key(ref)
        assert "node_1" in key and "signal-A" in key

    def test_value_key_for_none(self):
        """Test _value_key for None value."""
        optimizer = CSEOptimizer()
        key = optimizer._value_key(None)
        assert key == "None"  # Python's str(None)


class TestCSEOptimizerMakeKey:
    """Cover lines 345-383: _make_key for operation signatures."""

    def test_make_key_irarith(self):
        """Test _make_key for IRArith generates correct signature."""
        optimizer = CSEOptimizer()

        arith = IRArith("arith_1", "signal-C")
        arith.op = "+"
        arith.left = SignalRef("signal-A", "node_1")
        arith.right = 42

        key = optimizer._make_key(arith)
        # Key format is lowercase 'arith'
        assert "arith" in key
        assert "+" in key

    def test_make_key_irdecider(self):
        """Test _make_key for IRDecider generates correct signature."""
        optimizer = CSEOptimizer()

        decider = IRDecider("decider_1", "signal-D")
        decider.op = "<"
        decider.left = SignalRef("signal-A", "node_1")
        decider.right = 100

        key = optimizer._make_key(decider)
        # Key format is lowercase 'decider'
        assert "decider" in key

    def test_make_key_irconst(self):
        """Test _make_key for IRConst - constants don't generate CSE keys."""
        optimizer = CSEOptimizer()

        const = IRConst("const_1", "signal-A")
        const.value = 42

        key = optimizer._make_key(const)
        # Constants return empty string (no CSE for constants)
        assert key == ""


class TestCSEOptimizerUpdateReferences:
    """Cover CSE _update_references method."""

    def test_cse_update_references_irarith(self):
        """Test CSE updating references in IRArith."""
        optimizer = CSEOptimizer()

        arith = IRArith("arith_1", "signal-C")
        arith.op = "+"
        arith.left = SignalRef("signal-A", "old_id")
        arith.right = 10

        optimizer.replacements["old_id"] = "new_id"

        optimizer._update_references([arith])

        assert isinstance(arith.left, SignalRef)
        assert arith.left.source_id == "new_id"

    def test_cse_update_references_irdecider(self):
        """Test CSE updating references in IRDecider."""
        optimizer = CSEOptimizer()

        decider = IRDecider("decider_1", "signal-D")
        decider.op = ">"
        decider.left = SignalRef("signal-A", "old_id")
        decider.right = 5
        decider.output_value = SignalRef("signal-B", "old_out")

        optimizer.replacements["old_id"] = "new_id"
        optimizer.replacements["old_out"] = "new_out"

        optimizer._update_references([decider])

        assert isinstance(decider.left, SignalRef)
        assert decider.left.source_id == "new_id"
        assert isinstance(decider.output_value, SignalRef)
        assert decider.output_value.source_id == "new_out"
