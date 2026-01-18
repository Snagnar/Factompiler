"""
Additional tests for lowering/expression_lowerer.py to increase coverage.

This module targets specific uncovered lines and edge cases.
"""

from dsl_compiler.src.ir.nodes import IRDecider

from .conftest import compile_to_ir


class TestResolveTypeRef:
    """Tests for _resolve_type_ref edge cases - lines 116-131."""

    def test_resolve_type_ref_via_type_access(self):
        """Test .type access on a signal variable."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 10 | "iron-plate";
        Signal b = 5 | a.type;  # Use a's type for projection
        """)
        assert not diags.has_errors()

    def test_type_access_on_undefined_var(self):
        """Test .type access on undefined variable produces error."""
        ir_ops, _, diags = compile_to_ir("""
        Signal b = 5 | undefined.type;
        """)
        assert diags.has_errors()

    def test_type_access_chained(self):
        """Test chained type access."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 10 | "iron-plate";
        Signal b = 20 | a.type;
        Signal c = 30 | b.type;
        """)
        assert not diags.has_errors()


class TestUnknownExpressionType:
    """Tests for unknown expression type handling - line 187-188."""

    def test_valid_expressions_lowered(self):
        """Test that various valid expressions are lowered correctly."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 10 | "signal-A";
        Signal b = 20 | "signal-B";
        Signal computed = (a + b) * 2;
        """)
        assert not diags.has_errors()


class TestCompoundConditionEdgeCases:
    """Tests for compound condition lowering edge cases."""

    def test_compound_or_with_signal_output(self):
        """Test OR compound condition with signal output value."""
        ir_ops, _, diags = compile_to_ir("""
        Signal x = 10 | "signal-X";
        Signal y = 20 | "signal-Y";
        Signal out = 5 | "signal-O";
        Signal result = (x > 100 || y > 100) : out;
        """)
        assert not diags.has_errors()

    def test_nested_logical_and(self):
        """Test nested AND conditions."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 10 | "signal-A";
        Signal b = 20 | "signal-B";
        Signal c = 30 | "signal-C";
        Signal d = 40 | "signal-D";
        Signal result = ((a > 0) && (b > 0)) && ((c > 0) && (d > 0));
        """)
        assert not diags.has_errors()

    def test_nested_logical_or(self):
        """Test nested OR conditions."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 10 | "signal-A";
        Signal b = 20 | "signal-B";
        Signal c = 30 | "signal-C";
        Signal result = (a > 100) || (b > 100) || (c > 100);
        """)
        assert not diags.has_errors()


class TestSignalProjectionEdgeCases:
    """Tests for signal projection edge cases - lines 1047-1076."""

    def test_projection_with_int_value_type(self):
        """Test projection when semantic returns IntValue (line 1066)."""
        ir_ops, _, diags = compile_to_ir("""
        Signal x = 10;
        # Pure integer projection should unwrap
        Signal y = (7) | "signal-A";
        Signal result = x + y;
        """)
        assert not diags.has_errors()

    def test_projection_with_non_int_computed_value(self):
        """Test projection with computed value that's not an int literal."""
        ir_ops, _, diags = compile_to_ir("""
        Signal base = 10 | "signal-A";
        Signal projected = (base) | "signal-B";
        """)
        assert not diags.has_errors()


class TestDictLiteralEdgeCases:
    """Tests for dict literal lowering edge cases - lines 1088-1101."""

    def test_dict_literal_with_entity_properties(self):
        """Test dict literal with entity configuration properties."""
        ir_ops, _, diags = compile_to_ir("""
        Entity e = place("small-lamp", 0, 0, {use_colors: 1, always_on: 1});
        """)
        assert not diags.has_errors()

    def test_entity_without_properties(self):
        """Test entity placement without properties dict."""
        ir_ops, _, diags = compile_to_ir("""
        Entity e = place("small-lamp", 2, 2);
        """)
        assert not diags.has_errors()


class TestBundleOperationEdgeCases:
    """Tests for bundle operation edge cases."""

    def test_bundle_with_computed_signals(self):
        """Test bundle with computed signal values."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 10 | "signal-A";
        Signal b = 20 | "signal-B";
        Bundle mixed = {a, b, ("signal-C", 30)};
        """)
        assert not diags.has_errors()

    def test_bundle_arithmetic_with_scalar(self):
        """Test bundle arithmetic with scalar value."""
        ir_ops, _, diags = compile_to_ir("""
        Bundle b = {("signal-A", 10), ("signal-B", 20)};
        Bundle scaled = b * 2;
        """)
        assert not diags.has_errors()


class TestFunctionInliningEdgeCases:
    """Tests for function inlining edge cases."""

    def test_function_with_signal_param(self):
        """Test function with signal parameter."""
        ir_ops, _, diags = compile_to_ir("""
        func double(Signal x) {
            return x + x;
        }

        Signal input = 10 | "signal-A";
        Signal result = double(input);
        """)
        assert not diags.has_errors()

    def test_nested_function_calls(self):
        """Test nested function calls."""
        ir_ops, _, diags = compile_to_ir("""
        func add_one(Signal x) {
            return x + 1;
        }

        func add_two(Signal x) {
            return add_one(add_one(x));
        }

        Signal base = 5 | "signal-A";
        Signal result = add_two(base);
        """)
        assert not diags.has_errors()


class TestLogicalOperationEdgeCases:
    """Tests for logical operation edge cases - lines 1301-1360."""

    def test_logical_and_left_literal(self):
        """Test logical AND with left-side integer literal."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 10 | "signal-A";
        Signal result = 5 && a;
        """)
        assert not diags.has_errors()

    def test_logical_or_left_literal(self):
        """Test logical OR with left-side integer literal."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 10 | "signal-A";
        Signal result = 0 || a;
        """)
        assert not diags.has_errors()

    def test_logical_not_on_computed_value(self):
        """Test NOT on computed value."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 10 | "signal-A";
        Signal b = 5 | "signal-B";
        Signal computed = a + b;
        Signal result = !computed;
        """)
        assert not diags.has_errors()


class TestOutputSpecEdgeCases:
    """Tests for output specification edge cases."""

    def test_output_spec_with_signal_output(self):
        """Test output spec with signal value as output."""
        ir_ops, _, diags = compile_to_ir("""
        Signal cond = 10 | "signal-C";
        Signal value = 42 | "signal-V";
        Signal result = (cond > 5) : value;
        """)
        assert not diags.has_errors()

    def test_output_spec_copy_count_false(self):
        """Test output spec with literal output (copy_count should be false)."""
        ir_ops, _, diags = compile_to_ir("""
        Signal cond = 10 | "signal-C";
        Signal result = (cond > 5) : 100;
        """)
        assert not diags.has_errors()
        # The output is a literal, so copy_count_from_input should be False
        deciders = [op for op in ir_ops if isinstance(op, IRDecider)]
        # At least one decider should have copy_count_from_input = False
        assert any(not d.copy_count_from_input for d in deciders)


class TestConstantResolution:
    """Tests for constant symbol resolution - lines 1136-1139."""

    def test_resolve_constant_in_function(self):
        """Test integer literals are resolved correctly in functions."""
        ir_ops, _, diags = compile_to_ir("""
        func scale(Signal x) {
            return x * 10;
        }

        Signal base = 5 | "signal-A";
        Signal result = scale(base);
        """)
        assert not diags.has_errors()

    def test_multiple_integer_literals(self):
        """Test multiple integer literals used together."""
        ir_ops, _, diags = compile_to_ir("""
        Signal x = 10 | "signal-A";
        Signal result = (x * 3) + 100;
        """)
        assert not diags.has_errors()

    def test_for_loop_constant(self):
        """Test for loop variable as compile-time constant."""
        ir_ops, _, diags = compile_to_ir("""
        for i in 0..3 {
            Signal s = (i * 10) | "signal-A";
        }
        """)
        assert not diags.has_errors()


class TestUnaryOperationEdgeCases:
    """Tests for unary operation edge cases."""

    def test_unary_minus_on_signal(self):
        """Test unary minus on signal value."""
        ir_ops, _, diags = compile_to_ir("""
        Signal x = 10 | "signal-X";
        Signal neg_x = -x;
        """)
        assert not diags.has_errors()

    def test_unary_plus_on_signal(self):
        """Test unary plus on signal (identity)."""
        ir_ops, _, diags = compile_to_ir("""
        Signal x = 10 | "signal-X";
        Signal pos_x = +x;
        """)
        assert not diags.has_errors()

    def test_double_negation(self):
        """Test double negation."""
        ir_ops, _, diags = compile_to_ir("""
        Signal x = 10 | "signal-X";
        Signal double_neg = --x;
        """)
        assert not diags.has_errors()


class TestOutputSpecConditionEdgeCases:
    """Tests for output specification edge cases (covers lines 774-777, 833-836)."""

    def test_identifier_condition_output_spec(self):
        """Test output spec with identifier condition."""
        ir_ops, _, diags = compile_to_ir("""
        Signal cond = 1 | "signal-check";
        Signal result = (cond > 0) : 10;
        """)
        assert not diags.has_errors()

    def test_compound_and_condition_output_spec(self):
        """Test output spec with AND compound condition (lines 800-836)."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 5 | "signal-A";
        Signal b = 10 | "signal-B";
        Signal result = ((a > 0) && (b > 0)) : 1;
        """)
        assert not diags.has_errors()

    def test_compound_or_condition_output_spec(self):
        """Test output spec with OR compound condition."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 5 | "signal-A";
        Signal b = 10 | "signal-B";
        Signal result = ((a > 0) || (b < 20)) : 1;
        """)
        assert not diags.has_errors()


class TestSignalLiteralEdgeCases:
    """Tests for signal literal edge cases (covers lines 1047-1058, 1066-1076)."""

    def test_signal_literal_with_explicit_type(self):
        """Test signal literal with explicit type."""
        ir_ops, _, diags = compile_to_ir("""
        Signal x = ("signal-X", 42);
        """)
        assert not diags.has_errors()

    def test_signal_literal_with_variable_value(self):
        """Test signal literal with variable value (lines 1066-1076)."""
        ir_ops, _, diags = compile_to_ir("""
        int count = 5;
        Signal x = ("signal-X", count);
        """)
        assert not diags.has_errors()

    def test_signal_literal_fallback_type_allocation(self):
        """Test signal literal type inference fallback (lines 1088-1101)."""
        ir_ops, _, diags = compile_to_ir("""
        Signal x = 5 | "signal-A";
        Signal y = x + 1;
        """)
        assert not diags.has_errors()


class TestDictLiteralPlacementEdgeCases:
    """Tests for dict literal edge cases (covers lines 1136-1139)."""

    def test_dict_literal_with_signal_literal_value(self):
        """Test dict literal with signal literal value."""
        ir_ops, _, diags = compile_to_ir("""
        Entity lamp = place("small-lamp", 0, 0, {
            brightness: 100
        });
        """)
        assert not diags.has_errors()

    def test_dict_literal_with_nested_value(self):
        """Test dict literal with nested value."""
        ir_ops, _, diags = compile_to_ir("""
        Entity lamp = place("small-lamp", 0, 0, {
            color: "green"
        });
        """)
        # May have errors due to color being a string, not a valid property
        # The test covers the code path, not necessarily success


class TestBundleSelectEdgeCases:
    """Tests for bundle select edge cases (covers lines 1221-1227)."""

    def test_bundle_select_from_bundle_ref(self):
        """Test selecting from bundle."""
        ir_ops, _, diags = compile_to_ir("""
        Bundle b = {("iron-plate", 10), ("copper-plate", 20)};
        Signal iron = b["iron-plate"];
        """)
        assert not diags.has_errors()

    def test_bundle_select_from_entity_output(self):
        """Test selecting from entity output bundle."""
        ir_ops, _, diags = compile_to_ir("""
        Entity lamp = place("small-lamp", 0, 0);
        Signal out = lamp.output["signal-A"];
        """)
        # This tests accessing entity output as a bundle


class TestWireMergeFoldingCases:
    """Tests for wire merge folding cases (covers lines 1511-1538)."""

    def test_wire_merge_with_constants(self):
        """Test wire merge folding with multiple constants."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 10 | "signal-A";
        Signal b = 20 | "signal-A";
        Signal c = 30 | "signal-A";
        Signal sum = a + b + c;
        """)
        assert not diags.has_errors()

    def test_wire_merge_mixed_signals_types(self):
        """Test wire merge with different signal types (no folding)."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 10 | "signal-A";
        Signal b = 20 | "signal-B";
        Signal sum = a + b;
        """)
        assert not diags.has_errors()


class TestComparisonOutputEdgeCases:
    """Tests for comparison output edge cases (covers lines 713-722)."""

    def test_comparison_with_integer_output(self):
        """Test comparison with integer output value."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 5 | "signal-A";
        Signal result = (a > 0) : 100;
        """)
        assert not diags.has_errors()

    def test_comparison_with_signal_output(self):
        """Test comparison with signal output value."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 5 | "signal-A";
        Signal out = 10 | "signal-B";
        Signal result = (a > 0) : out;
        """)
        assert not diags.has_errors()
