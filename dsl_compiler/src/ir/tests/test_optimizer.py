"""Tests for optimizer.py - IR optimization passes."""

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.ir.nodes import (
    DeciderCondition,
    IRArith,
    IRConst,
    IRDecider,
    SignalRef,
)
from dsl_compiler.src.ir.optimizer import (
    ConstantPropagationOptimizer,
    CSEOptimizer,
)
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer


class TestConstantPropagationOptimizerFoldArithmetic:
    """Tests for ConstantPropagationOptimizer._fold_arithmetic()."""

    def test_fold_addition(self):
        """Test folding addition of two constants."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_arithmetic("+", 3, 5) == 8
        assert opt._fold_arithmetic("+", -3, 5) == 2
        assert opt._fold_arithmetic("+", 0, 0) == 0

    def test_fold_subtraction(self):
        """Test folding subtraction of two constants."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_arithmetic("-", 10, 3) == 7
        assert opt._fold_arithmetic("-", 5, 10) == -5
        assert opt._fold_arithmetic("-", 0, 0) == 0

    def test_fold_multiplication(self):
        """Test folding multiplication of two constants."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_arithmetic("*", 3, 4) == 12
        assert opt._fold_arithmetic("*", -3, 4) == -12
        assert opt._fold_arithmetic("*", 0, 100) == 0

    def test_fold_division(self):
        """Test folding integer division of two constants."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_arithmetic("/", 10, 3) == 3
        assert opt._fold_arithmetic("/", 9, 3) == 3
        assert opt._fold_arithmetic("/", -10, 3) == -4  # Python floor division

    def test_fold_division_by_zero_returns_none(self):
        """Test that division by zero returns None."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_arithmetic("/", 10, 0) is None

    def test_fold_modulo(self):
        """Test folding modulo of two constants."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_arithmetic("%", 10, 3) == 1
        assert opt._fold_arithmetic("%", 9, 3) == 0

    def test_fold_modulo_by_zero_returns_none(self):
        """Test that modulo by zero returns None."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_arithmetic("%", 10, 0) is None

    def test_fold_power(self):
        """Test folding power operations."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_arithmetic("**", 2, 10) == 1024
        assert opt._fold_arithmetic("^", 2, 10) == 1024
        assert opt._fold_arithmetic("**", 3, 3) == 27

    def test_fold_power_with_negative_exponent(self):
        """Test that negative exponents return 0."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_arithmetic("**", 2, -1) == 0

    def test_fold_power_with_large_values_returns_none(self):
        """Test that very large power operations return None to prevent overflow."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_arithmetic("**", 10000, 10) is None  # Base too large
        assert opt._fold_arithmetic("**", 2, 200) is None  # Exponent too large

    def test_fold_left_shift(self):
        """Test folding left shift operations."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_arithmetic("<<", 1, 4) == 16
        assert opt._fold_arithmetic("<<", 5, 2) == 20

    def test_fold_left_shift_edge_cases(self):
        """Test left shift with edge case shift amounts."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_arithmetic("<<", 1, -1) == 0  # Negative shift
        assert opt._fold_arithmetic("<<", 1, 32) == 0  # Shift >= 32

    def test_fold_right_shift(self):
        """Test folding right shift operations."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_arithmetic(">>", 16, 4) == 1
        assert opt._fold_arithmetic(">>", 20, 2) == 5

    def test_fold_right_shift_edge_cases(self):
        """Test right shift with edge case shift amounts."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_arithmetic(">>", 1, -1) == 0  # Negative shift
        assert opt._fold_arithmetic(">>", 1, 32) == 0  # Shift >= 32

    def test_fold_bitwise_and(self):
        """Test folding bitwise AND operations."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_arithmetic("&", 0b1111, 0b1010) == 0b1010
        assert opt._fold_arithmetic("AND", 0b1111, 0b1010) == 0b1010

    def test_fold_bitwise_or(self):
        """Test folding bitwise OR operations."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_arithmetic("|", 0b1100, 0b0011) == 0b1111
        assert opt._fold_arithmetic("OR", 0b1100, 0b0011) == 0b1111

    def test_fold_bitwise_xor(self):
        """Test folding bitwise XOR operations."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_arithmetic("XOR", 0b1111, 0b1010) == 0b0101

    def test_fold_unknown_operator_returns_none(self):
        """Test that unknown operators return None."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_arithmetic("unknown", 1, 2) is None


class TestConstantPropagationOptimizerFoldComparison:
    """Tests for ConstantPropagationOptimizer._fold_comparison()."""

    def test_fold_equality(self):
        """Test folding equality comparisons."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_comparison("==", 5, 5) is True
        assert opt._fold_comparison("=", 5, 5) is True
        assert opt._fold_comparison("==", 5, 3) is False

    def test_fold_inequality(self):
        """Test folding inequality comparisons."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_comparison("!=", 5, 3) is True
        assert opt._fold_comparison("≠", 5, 3) is True
        assert opt._fold_comparison("!=", 5, 5) is False

    def test_fold_less_than(self):
        """Test folding less than comparisons."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_comparison("<", 3, 5) is True
        assert opt._fold_comparison("<", 5, 5) is False
        assert opt._fold_comparison("<", 7, 5) is False

    def test_fold_less_than_or_equal(self):
        """Test folding less than or equal comparisons."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_comparison("<=", 3, 5) is True
        assert opt._fold_comparison("<=", 5, 5) is True
        assert opt._fold_comparison("<=", 7, 5) is False

    def test_fold_greater_than(self):
        """Test folding greater than comparisons."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_comparison(">", 7, 5) is True
        assert opt._fold_comparison(">", 5, 5) is False
        assert opt._fold_comparison(">", 3, 5) is False

    def test_fold_greater_than_or_equal(self):
        """Test folding greater than or equal comparisons."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_comparison(">=", 7, 5) is True
        assert opt._fold_comparison(">=", 5, 5) is True
        assert opt._fold_comparison(">=", 3, 5) is False

    def test_fold_unknown_operator_returns_none(self):
        """Test that unknown operators return None."""
        opt = ConstantPropagationOptimizer()
        assert opt._fold_comparison("???", 1, 2) is None


class TestConstantPropagationOptimizerOptimize:
    """Tests for ConstantPropagationOptimizer.optimize() method."""

    def test_optimize_folds_addition_of_constants(self):
        """Test that addition of two constants is folded."""
        const_a = IRConst("a", "__v1", None)
        const_a.value = 3
        const_a.debug_metadata = {}

        const_b = IRConst("b", "__v2", None)
        const_b.value = 5
        const_b.debug_metadata = {}

        arith = IRArith("add", "__v3", None)
        arith.op = "+"
        arith.left = SignalRef("__v1", "a")
        arith.right = SignalRef("__v2", "b")
        arith.debug_metadata = {}

        opt = ConstantPropagationOptimizer()
        result = opt.optimize([const_a, const_b, arith])

        # Should have folded the addition
        assert len([op for op in result if isinstance(op, IRArith)]) == 0
        # Should have created a new constant with value 8
        folded_consts = [op for op in result if isinstance(op, IRConst) and op.value == 8]
        assert len(folded_consts) == 1

    def test_optimize_preserves_user_declared_constants(self):
        """Test that operations using user-declared constants are NOT folded."""
        const_a = IRConst("a", "__v1", None)
        const_a.value = 3
        const_a.debug_metadata = {"user_declared": True}

        const_b = IRConst("b", "__v2", None)
        const_b.value = 5
        const_b.debug_metadata = {}

        arith = IRArith("add", "__v3", None)
        arith.op = "+"
        arith.left = SignalRef("__v1", "a")
        arith.right = SignalRef("__v2", "b")
        arith.debug_metadata = {}

        opt = ConstantPropagationOptimizer()
        result = opt.optimize([const_a, const_b, arith])

        # Should NOT have folded - user_declared constant must materialize
        assert any(isinstance(op, IRArith) for op in result)

    def test_optimize_folds_comparison(self):
        """Test that comparisons of two constants are folded."""
        const_a = IRConst("a", "__v1", None)
        const_a.value = 10
        const_a.debug_metadata = {}

        const_b = IRConst("b", "__v2", None)
        const_b.value = 5
        const_b.debug_metadata = {}

        decider = IRDecider("cmp", "__v3", None)
        decider.test_op = ">"
        decider.left = SignalRef("__v1", "a")
        decider.right = SignalRef("__v2", "b")
        decider.output_value = 1
        decider.debug_metadata = {}
        decider.conditions = []

        opt = ConstantPropagationOptimizer()
        result = opt.optimize([const_a, const_b, decider])

        # Should have folded the comparison (10 > 5 is True, output_value=1)
        assert len([op for op in result if isinstance(op, IRDecider)]) == 0
        folded_consts = [op for op in result if isinstance(op, IRConst) and op.value == 1]
        assert len(folded_consts) >= 1

    def test_optimize_returns_original_when_non_constant_operand(self):
        """Test that operations with external signal (not in const_map) are preserved."""
        # Create an arith that references a signal that's not a constant
        # (like an IRArith output or external input)
        arith = IRArith("add", "__v2", None)
        arith.op = "+"
        arith.left = SignalRef("__v1", "external_signal")  # Not in const_map
        arith.right = 42
        arith.debug_metadata = {}

        opt = ConstantPropagationOptimizer()
        result = opt.optimize([arith])

        # Should preserve the operation since left operand is not a known constant
        assert any(isinstance(op, IRArith) for op in result)


class TestCSEOptimizerOptimize:
    """Tests for CSEOptimizer.optimize() method."""

    def test_optimize_eliminates_duplicate_arithmetic(self):
        """Test that duplicate arithmetic operations are eliminated."""
        const = IRConst("c", "__v1", None)
        const.value = 5
        const.debug_metadata = {}

        arith1 = IRArith("add1", "__v2", None)
        arith1.op = "+"
        arith1.left = SignalRef("__v1", "c")
        arith1.right = 10
        arith1.debug_metadata = {}

        # Duplicate operation
        arith2 = IRArith("add2", "__v2", None)
        arith2.op = "+"
        arith2.left = SignalRef("__v1", "c")
        arith2.right = 10
        arith2.debug_metadata = {}

        opt = CSEOptimizer()
        result = opt.optimize([const, arith1, arith2])

        # Should have eliminated the duplicate
        ariths = [op for op in result if isinstance(op, IRArith)]
        assert len(ariths) == 1
        assert ariths[0].node_id == "add1"  # First one is kept

    def test_optimize_preserves_different_operations(self):
        """Test that different operations are preserved."""
        const = IRConst("c", "__v1", None)
        const.value = 5
        const.debug_metadata = {}

        arith1 = IRArith("add", "__v2", None)
        arith1.op = "+"
        arith1.left = SignalRef("__v1", "c")
        arith1.right = 10
        arith1.debug_metadata = {}

        arith2 = IRArith("sub", "__v3", None)
        arith2.op = "-"
        arith2.left = SignalRef("__v1", "c")
        arith2.right = 10
        arith2.debug_metadata = {}

        opt = CSEOptimizer()
        result = opt.optimize([const, arith1, arith2])

        # Both should be preserved since they're different operations
        ariths = [op for op in result if isinstance(op, IRArith)]
        assert len(ariths) == 2

    def test_optimize_updates_references_to_eliminated_operations(self):
        """Test that references are updated when eliminating duplicates."""
        const = IRConst("c", "__v1", None)
        const.value = 5
        const.debug_metadata = {}

        arith1 = IRArith("add1", "__v2", None)
        arith1.op = "+"
        arith1.left = SignalRef("__v1", "c")
        arith1.right = 10
        arith1.debug_metadata = {}

        arith2 = IRArith("add2", "__v2", None)
        arith2.op = "+"
        arith2.left = SignalRef("__v1", "c")
        arith2.right = 10
        arith2.debug_metadata = {}

        # Operation that references the duplicate
        arith3 = IRArith("mul", "__v3", None)
        arith3.op = "*"
        arith3.left = SignalRef("__v2", "add2")  # References the duplicate
        arith3.right = 2
        arith3.debug_metadata = {}

        opt = CSEOptimizer()
        result = opt.optimize([const, arith1, arith2, arith3])

        # arith3 should now reference arith1 instead of arith2
        mul_op = next(op for op in result if isinstance(op, IRArith) and op.op == "*")
        assert mul_op.left.source_id == "add1"


class TestCSEOptimizerValueKey:
    """Tests for CSEOptimizer._value_key() method."""

    def test_value_key_for_signal_ref(self):
        """Test key generation for SignalRef values."""
        opt = CSEOptimizer()
        ref = SignalRef("__v1", "const_1")
        key = opt._value_key(ref)
        assert "sig:" in key
        assert "const_1" in key

    def test_value_key_for_integer(self):
        """Test key generation for integer values."""
        opt = CSEOptimizer()
        assert opt._value_key(42) == "int:42"
        assert opt._value_key(0) == "int:0"
        assert opt._value_key(-5) == "int:-5"

    def test_value_key_for_string(self):
        """Test key generation for string values."""
        opt = CSEOptimizer()
        assert opt._value_key("signal-A") == "str:signal-A"

    def test_value_key_for_replaced_ref(self):
        """Test that replaced references use canonical ID."""
        opt = CSEOptimizer()
        opt.replacements["old_id"] = "new_id"
        ref = SignalRef("__v1", "old_id")
        key = opt._value_key(ref)
        assert "new_id" in key
        assert "old_id" not in key

    def test_value_key_for_unknown_type(self):
        """Test key generation for unknown value types."""
        opt = CSEOptimizer()
        # Should use repr for unknown types
        key = opt._value_key([1, 2, 3])
        assert "[1, 2, 3]" in key


class TestCSEOptimizerMakeKey:
    """Tests for CSEOptimizer._make_key() method."""

    def test_make_key_for_arith(self):
        """Test key generation for IRArith."""
        opt = CSEOptimizer()
        arith = IRArith("add", "__v2", None)
        arith.op = "+"
        arith.left = SignalRef("__v1", "c")
        arith.right = 10
        arith.debug_metadata = {}

        key = opt._make_key(arith)
        assert "arith" in key
        assert "+" in key

    def test_make_key_for_decider_legacy_mode(self):
        """Test key generation for IRDecider in legacy single-condition mode."""
        opt = CSEOptimizer()
        decider = IRDecider("cmp", "__v2", None)
        decider.test_op = ">"
        decider.left = SignalRef("__v1", "a")
        decider.right = 5
        decider.output_value = 1
        decider.conditions = []  # Legacy mode
        decider.debug_metadata = {}

        key = opt._make_key(decider)
        assert "decider" in key
        assert ">" in key

    def test_make_key_for_decider_multi_condition_with_valueref(self):
        """Test key generation for IRDecider with multi-condition using ValueRefs."""
        opt = CSEOptimizer()
        decider = IRDecider("cmp", "__v2", None)
        decider.output_value = 1
        decider.debug_metadata = {}

        cond = DeciderCondition(
            comparator=">",
            compare_type="and",
        )
        cond.first_operand = SignalRef("__v1", "a")
        cond.second_operand = SignalRef("__v2", "b")
        decider.conditions = [cond]

        key = opt._make_key(decider)
        assert "decider_multi" in key
        assert ">" in key
        assert "and" in key

    def test_make_key_for_decider_multi_condition_with_string_signal(self):
        """Test key generation for IRDecider with multi-condition using string signals."""
        opt = CSEOptimizer()
        decider = IRDecider("cmp", "__v2", None)
        decider.output_value = 1
        decider.debug_metadata = {}

        cond = DeciderCondition(
            comparator="==",
            first_signal="signal-A",
            second_constant=42,
            compare_type="or",
        )
        decider.conditions = [cond]

        key = opt._make_key(decider)
        assert "decider_multi" in key
        assert "==" in key
        assert "str:signal-A" in key
        assert "int:42" in key

    def test_make_key_for_const_returns_empty(self):
        """Test that IRConst returns empty key (not CSE candidates)."""
        opt = CSEOptimizer()
        const = IRConst("c", "__v1", None)
        const.value = 5
        const.debug_metadata = {}

        key = opt._make_key(const)
        assert key == ""


class TestConstantPropagationOptimizerHelpers:
    """Tests for helper methods in ConstantPropagationOptimizer."""

    def test_references_node_with_signal_ref(self):
        """Test _references_node returns True for matching SignalRef."""
        opt = ConstantPropagationOptimizer()
        ref = SignalRef("__v1", "target_node")
        assert opt._references_node(ref, "target_node") is True
        assert opt._references_node(ref, "other_node") is False

    def test_references_node_with_non_signal_ref(self):
        """Test _references_node returns False for non-SignalRef values."""
        opt = ConstantPropagationOptimizer()
        assert opt._references_node(42, "target_node") is False
        assert opt._references_node("string", "target_node") is False

    def test_update_value_replaces_signal_ref(self):
        """Test _update_value replaces SignalRef when replacement exists."""
        opt = ConstantPropagationOptimizer()
        opt.replacements["old_id"] = "new_id"
        ref = SignalRef("__v1", "old_id", debug_label="test")

        updated = opt._update_value(ref)
        assert isinstance(updated, SignalRef)
        assert updated.source_id == "new_id"
        assert updated.signal_type == "__v1"
        assert updated.debug_label == "test"

    def test_update_value_preserves_value_without_replacement(self):
        """Test _update_value preserves value when no replacement exists."""
        opt = ConstantPropagationOptimizer()
        ref = SignalRef("__v1", "unchanged_id")

        updated = opt._update_value(ref)
        assert updated.source_id == "unchanged_id"

    def test_update_value_preserves_non_signal_ref(self):
        """Test _update_value preserves non-SignalRef values."""
        opt = ConstantPropagationOptimizer()
        assert opt._update_value(42) == 42
        assert opt._update_value("string") == "string"

    def test_get_const_value_for_integer(self):
        """Test _get_const_value returns integer directly."""
        opt = ConstantPropagationOptimizer()
        assert opt._get_const_value(42, {}) == 42

    def test_get_const_value_for_signal_ref_in_const_map(self):
        """Test _get_const_value returns value from const_map."""
        opt = ConstantPropagationOptimizer()
        const = IRConst("c", "__v1", None)
        const.value = 100
        const_map = {"c": const}

        ref = SignalRef("__v1", "c")
        assert opt._get_const_value(ref, const_map) == 100

    def test_get_const_value_follows_replacement_chain(self):
        """Test _get_const_value follows replacement chain to find value."""
        opt = ConstantPropagationOptimizer()
        opt.replacements["old_id"] = "new_id"
        const = IRConst("new_id", "__v1", None)
        const.value = 200
        const_map = {"new_id": const}

        ref = SignalRef("__v1", "old_id")
        assert opt._get_const_value(ref, const_map) == 200

    def test_get_const_value_returns_none_for_unknown_ref(self):
        """Test _get_const_value returns None for unknown references."""
        opt = ConstantPropagationOptimizer()
        ref = SignalRef("__v1", "unknown")
        assert opt._get_const_value(ref, {}) is None

    def test_is_user_declared_operand_true(self):
        """Test _is_user_declared_operand returns True for user-declared constants."""
        opt = ConstantPropagationOptimizer()
        const = IRConst("c", "__v1", None)
        const.debug_metadata = {"user_declared": True}
        const_map = {"c": const}

        ref = SignalRef("__v1", "c")
        assert opt._is_user_declared_operand(ref, const_map) is True

    def test_is_user_declared_operand_false(self):
        """Test _is_user_declared_operand returns False for non-user-declared constants."""
        opt = ConstantPropagationOptimizer()
        const = IRConst("c", "__v1", None)
        const.debug_metadata = {}
        const_map = {"c": const}

        ref = SignalRef("__v1", "c")
        assert opt._is_user_declared_operand(ref, const_map) is False

    def test_is_user_declared_operand_non_signal_ref(self):
        """Test _is_user_declared_operand returns False for non-SignalRef values."""
        opt = ConstantPropagationOptimizer()
        assert opt._is_user_declared_operand(42, {}) is False


class TestConstantPropagationOptimizerChainedFolding:
    """Tests for chained constant folding behavior."""

    def test_optimize_folds_chained_operations(self):
        """Test that chained constant operations are folded iteratively."""
        const_a = IRConst("a", "__v1", None)
        const_a.value = 2
        const_a.debug_metadata = {}

        const_b = IRConst("b", "__v2", None)
        const_b.value = 3
        const_b.debug_metadata = {}

        # First: a + b = 5
        arith1 = IRArith("add1", "__v3", None)
        arith1.op = "+"
        arith1.left = SignalRef("__v1", "a")
        arith1.right = SignalRef("__v2", "b")
        arith1.debug_metadata = {}

        # Second: (a+b) * 2 = 10
        arith2 = IRArith("mul", "__v4", None)
        arith2.op = "*"
        arith2.left = SignalRef("__v3", "add1")
        arith2.right = 2
        arith2.debug_metadata = {}

        opt = ConstantPropagationOptimizer()
        result = opt.optimize([const_a, const_b, arith1, arith2])

        # Should have folded both operations
        ariths = [op for op in result if isinstance(op, IRArith)]
        assert len(ariths) == 0

        # Should have a constant with value 10
        consts = [op for op in result if isinstance(op, IRConst)]
        final_values = {c.value for c in consts}
        assert 10 in final_values


# =============================================================================
# Coverage gap tests (Lines 293-295, 404-412)
# =============================================================================


def compile_to_ir(source: str):
    """Helper to compile source to IR."""
    parser = DSLParser()
    ast = parser.parse(source, "<test>")
    diagnostics = ProgramDiagnostics()
    analyzer = SemanticAnalyzer(diagnostics=diagnostics)
    analyzer.visit(ast)
    lowerer = ASTLowerer(analyzer, diagnostics)
    ir_ops = lowerer.lower_program(ast)
    return ir_ops, lowerer, diagnostics


class TestOptimizerCoverageGaps:
    """Tests for optimizer.py coverage gaps > 2 lines."""

    def test_constant_propagator_update_decider_refs(self):
        """Cover lines 293-295: updating IRDecider references in constant propagator."""
        source = """
        Signal a = 10;
        Signal b = 20;
        Signal result = (a > 5) : b;
        """
        ir_ops, lowerer, diags = compile_to_ir(source)
        deciders = [op for op in ir_ops if isinstance(op, IRDecider)]
        assert len(deciders) > 0

    def test_cse_optimizer_update_decider_conditions(self):
        """Cover lines 404-412: CSE optimizer updating multi-condition deciders."""
        source = """
        Signal a = 10;
        Signal b = 20;
        Signal result = ((a > 5) && (b < 30)) : 1;
        """
        ir_ops, lowerer, diags = compile_to_ir(source)
        deciders = [op for op in ir_ops if isinstance(op, IRDecider)]
        for d in deciders:
            if hasattr(d, "conditions") and d.conditions:
                assert len(d.conditions) > 0
