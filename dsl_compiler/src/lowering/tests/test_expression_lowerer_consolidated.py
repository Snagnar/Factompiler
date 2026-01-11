"""
Consolidated tests for lowering/expression_lowerer.py - Expression to IR lowering.

This module contains comprehensive test programs that cover the ExpressionLowerer
functionality with fewer, larger test programs instead of many small ones.
"""

from unittest.mock import MagicMock

from dsl_compiler.src.ast.expressions import (
    BinaryOp,
    IdentifierExpr,
    SignalLiteral,
    UnaryOp,
)
from dsl_compiler.src.ast.literals import NumberLiteral, StringLiteral
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.common.source_location import SourceLocation
from dsl_compiler.src.ir.builder import IRBuilder
from dsl_compiler.src.ir.nodes import IRArith, IRDecider
from dsl_compiler.src.lowering.expression_lowerer import ExpressionLowerer
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


class TestArithmeticAndBinaryOperations:
    """Tests for all arithmetic, shift, bitwise, and binary operations."""

    def test_all_arithmetic_operators(self):
        """Test all arithmetic operators in a single program."""
        ir_ops, _, diags = compile_to_ir("""
        # Basic arithmetic
        Signal add_result = 10 + 20;
        Signal sub_result = 30 - 15;
        Signal mul_result = 5 * 4;
        Signal div_result = 20 / 4;
        Signal mod_result = 17 % 5;
        Signal pow_result = 2 ** 8;

        # Shift operations
        Signal shift_left = 1 << 4;
        Signal shift_right = 256 >> 2;

        # Bitwise operations
        Signal bit_and = 15 AND 7;
        Signal bit_or = 8 OR 4;
        Signal bit_xor = 15 XOR 9;

        # Chained arithmetic
        Signal chain1 = 10 + 20 + 30;
        Signal chain2 = 5 * 2 + 10;
        Signal chain3 = 100 / 2 - 25;

        # Mixed signal types in arithmetic
        Signal iron = ("iron-plate", 100);
        Signal more_iron = iron + 50;
        Signal result_from_int = 10 + iron;
        Signal iron_plus_iron = iron + iron;

        # Negation and unary
        Signal neg = -10;
        Signal neg_sig = -iron;
        Signal pos = +iron;
        """)
        assert not diags.has_errors()
        # Many operations are constant-folded, so we just verify it compiles
        arith_ops = [op for op in ir_ops if isinstance(op, IRArith)]
        assert len(arith_ops) >= 1

    def test_arithmetic_with_signal_types(self):
        """Test arithmetic operations that preserve/propagate signal types."""
        ir_ops, _, diags = compile_to_ir("""
        Signal iron = ("iron-plate", 100);
        Signal copper = ("copper-plate", 50);

        # Signal + int (signal type preserved)
        Signal iron_plus = iron + 25;

        # int + Signal (right type propagates)
        Signal from_int = 25 + iron;

        # Signal + Signal (left type preserved)
        Signal mixed = iron + copper;

        # Signal - int
        Signal iron_minus = iron - 10;

        # Signal * int
        Signal iron_mult = iron * 2;

        # Signal / int
        Signal iron_div = iron / 5;

        # Signal % int
        Signal iron_mod = iron % 30;

        # Power of signal
        Signal iron_pow = iron ** 2;
        """)
        assert not diags.has_errors()


class TestComparisonAndLogicalOperations:
    """Tests for comparison and logical operators."""

    def test_all_comparison_operators(self):
        """Test all comparison operators."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 10;
        Signal b = 20;
        Signal c = 10;

        # All comparison operators
        Signal eq = (a == c): 1;
        Signal ne = (a != b): 1;
        Signal lt = (a < b): 1;
        Signal le = (a <= c): 1;
        Signal gt = (b > a): 1;
        Signal ge = (a >= c): 1;

        # Signal vs signal comparisons
        Signal sig_vs_sig = (a > b): 1;
        Signal sig_eq_sig = (a == c): 1;

        # Comparison with output value
        Signal out_42 = (a > 5): 42;
        Signal out_large = (a > 0): 1000000;
        Signal out_sig = (a > 5): a;
        """)
        assert not diags.has_errors()
        deciders = [op for op in ir_ops if isinstance(op, IRDecider)]
        assert len(deciders) >= 5

    def test_logical_operators_and_conditions(self):
        """Test logical operators with various input types."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 10;
        Signal b = 20;
        Signal c = 30;

        # Logical with boolean producers (comparisons)
        Signal and_cmp = (a > 5) && (b < 30);
        Signal or_cmp = (a > 100) || (b > 0);

        # Chained logical operators
        Signal three_and = (a > 0) && (b > 0) && (c > 0);
        Signal three_or = (a > 100) || (b > 100) || (c > 100);

        # Logical with non-boolean signals (needs materialization)
        Signal x = 5;
        Signal y = 3;
        Signal non_bool_and = x && y;
        Signal non_bool_or = x || y;

        # Integer literals in logical ops (needs materialization)
        Signal lit_and = a && 5;
        Signal lit_or = a || 5;
        Signal left_lit_and = 5 && a;

        # NOT operator
        Signal not_sig = !a;
        Signal not_zero = !0;
        Signal not_one = !1;
        """)
        assert not diags.has_errors()

    def test_condition_folding(self):
        """Test condition folding optimization for multi-condition deciders."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = ("signal-A", 10);
        Signal b = ("signal-B", 20);
        Signal c = ("signal-C", 30);

        # Two conditions folded into one decider
        Signal fold_and_two = (a > 0) && (b > 0);

        # Three conditions folded
        Signal fold_and_three = (a > 0) && (b > 0) && (c > 0);

        # Mixed comparison operators
        Signal fold_mixed = (a > 5) && (b < 10);

        # OR folding
        Signal fold_or = (a > 0) || (b > 0);
        """)
        assert not diags.has_errors()


class TestBundleOperations:
    """Tests for bundle literals and operations."""

    def test_bundle_literals(self):
        """Test various bundle literal patterns."""
        ir_ops, _, diags = compile_to_ir("""
        # All constant bundle
        Bundle const_bundle = {("iron-plate", 10), ("copper-plate", 20), ("signal-A", 30)};

        # Single element bundle
        Bundle single = {("iron-plate", 10)};

        # Bundle with computed signals
        Signal a = 10;
        Signal b = 20;
        Bundle computed = {a, b};

        # Mixed constant and computed
        Bundle mixed = {a, ("copper-plate", 20)};

        # Nested bundles
        Bundle inner = {("iron-plate", 10)};
        Bundle outer = {inner, ("copper-plate", 20)};

        # Bundle with single computed signal
        Bundle single_computed = {a};
        """)
        assert not diags.has_errors()

    def test_bundle_arithmetic(self):
        """Test arithmetic operations on bundles."""
        ir_ops, _, diags = compile_to_ir("""
        Bundle b = {("iron-plate", 100), ("copper-plate", 50)};

        # All basic arithmetic
        Bundle plus = b + 10;
        Bundle minus = b - 10;
        Bundle mult = b * 2;
        Bundle div = b / 5;
        Bundle mod = b % 30;
        Bundle pow = b ** 2;

        # Shift operations on bundle
        Bundle shifted = b << 2;
        """)
        assert not diags.has_errors()

    def test_bundle_select_and_any_all(self):
        """Test bundle selection and any/all functions."""
        ir_ops, _, diags = compile_to_ir("""
        Bundle b = {("iron-plate", 10), ("copper-plate", 20), ("signal-A", 30)};

        # Select specific signals
        Signal iron = b["iron-plate"];
        Signal copper = b["copper-plate"];

        # Use selected signal in arithmetic
        Signal doubled = iron * 2;

        # any() and all() functions
        Signal has_any = (any(b) > 0): 1;
        Signal all_positive = (all(b) > 0): 1;

        # any/all in conditions
        Signal any_cond = (any(b) > 5): 1;
        Signal all_cond = (all(b) > 5): 1;
        """)
        assert not diags.has_errors()


class TestProjectionAndSignalLiterals:
    """Tests for projection and signal literal expressions."""

    def test_projection_operations(self):
        """Test projection expressions."""
        ir_ops, _, diags = compile_to_ir("""
        Signal iron = ("iron-plate", 10);

        # Projection to different type
        Signal as_copper = iron | "copper-plate";

        # Projection from int to signal
        Signal int_proj = 5 | "signal-A";

        # Projection to same type (no-op)
        Signal same = iron | "iron-plate";

        # Type access in projection
        Signal dynamic = 20 | iron.type;
        """)
        assert not diags.has_errors()

    def test_signal_literals(self):
        """Test signal literal expressions."""
        ir_ops, _, diags = compile_to_ir("""
        # Item signals
        Signal iron = ("iron-plate", 100);
        Signal copper = ("copper-plate", 50);

        # Virtual signals
        Signal sig_a = ("signal-A", 42);
        Signal sig_0 = ("signal-0", 123);

        # Fluid signals
        Signal water = ("water", 1000);
        Signal oil = ("crude-oil", 500);

        # Signal literal with expression value
        Signal base = 10;
        Signal computed_lit = ("iron-plate", base + 5);
        """)
        assert not diags.has_errors()


class TestFunctionInlining:
    """Tests for function inlining."""

    def test_function_inlining_patterns(self):
        """Test various function inlining patterns."""
        ir_ops, _, diags = compile_to_ir("""
        # Simple function
        func double(int x) { return x * 2; }

        # Function with signal param
        func add_ten(Signal s) { return s + 10; }

        # Function with type access
        func same_type(Signal s) { return s | s.type; }

        # Function with multiple params
        func calc(int a, int b) {
            Signal temp = a * 2;
            return temp + b;
        }

        # Function calls
        Signal r1 = double(5);
        Signal r2 = double(10);

        Signal iron = ("iron-plate", 100);
        Signal r3 = add_ten(iron);
        Signal r4 = same_type(iron);

        Signal r5 = calc(5, 10);

        # Nested function calls
        func triple(int x) { return x * 3; }
        Signal nested = double(5) + triple(3);

        # Function with entity param
        func light_entity(Entity e) {
            e.enable = 1 > 0;
            return 1;
        }
        Entity lamp = place("small-lamp", 0, 0);
        Signal r6 = light_entity(lamp);

        # Function receiving int as Signal param
        func process(Signal s) { return s + 1; }
        Signal r7 = process(5);
        """)
        assert not diags.has_errors()


class TestEntityAndPlaceOperations:
    """Tests for entity placement and property access."""

    def test_entity_placement(self):
        """Test entity placement operations."""
        ir_ops, _, diags = compile_to_ir("""
        # Basic placement
        Entity lamp1 = place("small-lamp", 0, 0);
        Entity lamp2 = place("small-lamp", 2, 0);
        Entity lamp3 = place("small-lamp", 4, 0);

        # Placement with properties
        Entity belt = place("transport-belt", 0, 2, {direction: 4});

        # Different positions
        Entity lamp_far = place("small-lamp", 10, 20);

        # Entity output as bundle
        Entity chest = place("iron-chest", 0, 4);
        Bundle contents = chest.output;
        """)
        assert not diags.has_errors()

    def test_entity_properties(self):
        """Test entity property access and assignment."""
        ir_ops, _, diags = compile_to_ir("""
        Entity lamp = place("small-lamp", 0, 0);

        # Enable property with comparison
        Signal cond = 10;
        lamp.enable = cond > 5;

        # Color property
        lamp.color = ("signal-red", 255);
        """)
        assert not diags.has_errors()


class TestMemoryOperations:
    """Tests for memory read/write expressions."""

    def test_memory_expressions(self):
        """Test memory read and write expressions."""
        ir_ops, _, diags = compile_to_ir("""
        Memory counter: "signal-A";
        Memory counter2: "signal-B";

        # Read expression
        Signal current = counter.read();

        # Read in arithmetic
        Signal doubled = counter.read() * 2;

        # Read in condition
        Signal is_positive = (counter.read() > 0): 1;

        # Write with when condition
        Signal trigger = 1;
        counter.write(42, when=trigger > 0);

        # Write with compound condition (different memory cell)
        Signal a = 10;
        Signal b = 20;
        counter2.write(100, when=(a > 5) && (b < 30));
        """)
        assert not diags.has_errors()


class TestOutputSpecPatterns:
    """Tests for output specifier expression patterns."""

    def test_output_spec_expressions(self):
        """Test output specifier expression patterns."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 10;
        Signal b = 20;

        # Constant true condition
        Signal always_42 = (1 > 0): 42;

        # Constant false condition
        Signal never_42 = (0 > 1): 42;

        # Variable condition with constant output
        Signal cond_out = (a > 5): 1;

        # Variable condition with signal output (copy from input)
        Signal copy_out = (a > 5): a;

        # Large output value
        Signal large = (a > 0): 1000000;

        # Output specifier with computed output
        Signal computed = (a > 5): (a * 2);

        # Integer output
        Signal int_out = (a > 5): 42;
        """)
        assert not diags.has_errors()


class TestConstantFolding:
    """Tests for constant folding optimizations."""

    def test_constant_folding_all_operators(self):
        """Test constant folding for all operators."""
        ir_ops, _, diags = compile_to_ir("""
        # Arithmetic folding
        Signal add = 10 + 20;
        Signal sub = 30 - 10;
        Signal mul = 5 * 4;
        Signal div = 20 / 4;
        Signal mod = 17 % 5;
        Signal pow = 2 ** 8;

        # Shift folding
        Signal shl = 1 << 4;
        Signal shr = 256 >> 4;

        # Bitwise folding
        Signal band = 15 AND 7;
        Signal bor = 8 OR 4;
        Signal bxor = 15 XOR 9;

        # Comparison folding (true and false cases)
        Signal cmp_true = (10 > 5): 1;
        Signal cmp_false = (5 > 10): 1;

        # Complex expression folding
        Signal complex = (10 + 20) * 2;

        # Negation folding
        Signal neg = -42;
        Signal not_zero = !0;
        """)
        assert not diags.has_errors()


class TestForLoopExpressions:
    """Tests for expressions within for loops."""

    def test_for_loop_expressions(self):
        """Test expressions using loop iterators."""
        ir_ops, _, diags = compile_to_ir("""
        # Iterator in expression
        for i in 0..3 {
            Signal val = i * 10;
        }

        # Iterator in place coordinates
        for i in 0..3 {
            Entity lamp = place("small-lamp", i, 0);
        }

        # Iterator with external signal
        Signal offset = 100;
        for i in 0..3 {
            Signal val = offset + i * 10;
        }
        """)
        assert not diags.has_errors()


class TestBooleanProducerDetection:
    """Tests for boolean producer detection optimization."""

    def test_boolean_producer_detection(self):
        """Test detection of boolean-producing expressions."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 10;
        Signal b = 20;

        # Comparison produces boolean
        Signal cond1 = (a > 5): 1;
        Signal cond2 = (b > 10): 1;

        # Multiplication of booleans
        Signal both = cond1 && cond2;

        # Identity projection preserves boolean
        Signal identity = cond1 + 0;

        # Constant 0/1 as boolean
        Signal const_bool = 1;
        Signal and_with_const = cond1 && const_bool;
        """)
        assert not diags.has_errors()


class TestTypeAccessAndResolution:
    """Tests for type access and resolution."""

    def test_type_access_patterns(self):
        """Test type access patterns."""
        ir_ops, _, diags = compile_to_ir("""
        Signal iron = ("iron-plate", 10);
        Signal copper = ("copper-plate", 20);

        # Type access on signal
        Signal same = iron | iron.type;

        # Function with type access
        func preserve_type(Signal s) { return s | s.type; }
        Signal preserved = preserve_type(iron);

        # Type access in projection
        Signal proj = 20 | iron.type;
        """)
        assert not diags.has_errors()


class TestIntConstants:
    """Tests for int constants and their resolution."""

    def test_int_constant_usage(self):
        """Test int constant usage patterns."""
        ir_ops, _, diags = compile_to_ir("""
        int OFFSET = 5;
        int FACTOR = 10;

        # Int in arithmetic
        Signal a = 100;
        Signal result1 = a + OFFSET;
        Signal result2 = a * FACTOR;

        # Int in function call
        func double(int x) { return x * 2; }
        Signal result3 = double(FACTOR);

        # Int in complex expression
        Signal result4 = (FACTOR + OFFSET) * 2;
        """)
        assert not diags.has_errors()


class TestErrorPaths:
    """Tests for error handling paths."""

    def test_undefined_identifier_error(self):
        """Test error for undefined identifier."""
        ir_ops, _, diags = compile_to_ir("""
        Signal x = undefined_var + 5;
        """)
        assert diags.has_errors()

    def test_output_spec_non_comparison_error(self):
        """Test error for output spec with non-comparison."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 10;
        Signal b = 20;
        Signal result = (a + b): 1;
        """)
        assert diags.has_errors()

    def test_function_wrong_arg_count_error(self):
        """Test error for function with wrong argument count."""
        ir_ops, _, diags = compile_to_ir("""
        func add(int a, int b) { return a + b; }
        Signal result = add(1);
        """)
        assert diags.has_errors()


class TestWireMergeOptimization:
    """Tests for wire merge optimization."""

    def test_wire_merge_patterns(self):
        """Test wire merge optimization patterns."""
        ir_ops, _, diags = compile_to_ir("""
        Signal a = 10;
        Signal b = 20;
        Signal c = 30;

        # Two-way merge (same type)
        Signal sum2 = a + b;

        # Three-way merge
        Signal sum3 = a + b + c;

        # Four-way merge
        Signal d = 40;
        Signal sum4 = a + b + c + d;

        # Different types don't merge
        Signal iron = ("iron-plate", 100);
        Signal copper = ("copper-plate", 50);
        Signal mixed_sum = iron + copper;
        """)
        assert not diags.has_errors()


# =============================================================================
# DIRECT UNIT TESTS FOR INTERNAL METHODS
# =============================================================================
# These tests directly call internal methods of ExpressionLowerer to cover
# specific code paths that aren't reachable via full compilation.


def make_loc() -> SourceLocation:
    """Create a dummy source location."""
    return SourceLocation("<test>", 1, 1)


def create_minimal_lowerer() -> tuple[ExpressionLowerer, IRBuilder, ProgramDiagnostics]:
    """Create a minimal ExpressionLowerer for direct testing."""
    diagnostics = ProgramDiagnostics()
    ir_builder = IRBuilder()
    semantic = MagicMock()
    semantic.get_expr_type = MagicMock(return_value=None)
    semantic.current_scope = MagicMock()
    semantic.current_scope.lookup = MagicMock(return_value=None)

    parent = MagicMock()
    parent.param_values = {}
    parent.signal_refs = {}
    parent.referenced_signal_names = set()
    parent.get_expr_context = MagicMock(return_value=None)
    parent.ensure_signal_registered = MagicMock()
    parent.ir_builder = ir_builder
    parent.semantic = semantic
    parent.diagnostics = diagnostics
    parent._error = MagicMock(side_effect=lambda msg, node=None: diagnostics.error(msg, node))

    expr_lowerer = ExpressionLowerer(parent)
    return expr_lowerer, ir_builder, diagnostics


class TestResolveSignalTypeDirect:
    """Direct unit tests for _resolve_signal_type method."""

    def test_resolve_signal_type_with_string(self):
        """_resolve_signal_type with a string type returns it directly."""
        lowerer, _, _ = create_minimal_lowerer()
        result = lowerer._resolve_signal_type("iron-plate", None)
        assert result == "iron-plate"


class TestIsBooleanProducerDirect:
    """Direct unit tests for _is_boolean_producer method."""

    def test_int_zero_is_boolean(self):
        """Integer 0 is a boolean producer."""
        lowerer, _, _ = create_minimal_lowerer()
        assert lowerer._is_boolean_producer(0) is True

    def test_int_one_is_boolean(self):
        """Integer 1 is a boolean producer."""
        lowerer, _, _ = create_minimal_lowerer()
        assert lowerer._is_boolean_producer(1) is True

    def test_int_other_not_boolean(self):
        """Other integers are not boolean producers."""
        lowerer, _, _ = create_minimal_lowerer()
        assert lowerer._is_boolean_producer(42) is False


class TestResolveConstantSymbolDirect:
    """Direct unit tests for _resolve_constant_symbol method."""

    def test_resolve_constant_symbol_from_param(self):
        """_resolve_constant_symbol finds int in param_values."""
        lowerer, _, _ = create_minimal_lowerer()
        lowerer.parent.param_values["x"] = 42
        assert lowerer._resolve_constant_symbol("x") == 42

    def test_resolve_constant_symbol_non_int_param(self):
        """_resolve_constant_symbol returns None for non-int param."""
        lowerer, ir_builder, _ = create_minimal_lowerer()
        lowerer.parent.param_values["x"] = ir_builder.const("sig", 10)
        assert lowerer._resolve_constant_symbol("x") is None

    def test_resolve_constant_symbol_not_found(self):
        """_resolve_constant_symbol returns None for undefined symbol."""
        lowerer, _, _ = create_minimal_lowerer()
        assert lowerer._resolve_constant_symbol("undefined") is None


class TestIsSimpleOperandDirect:
    """Direct unit tests for _is_simple_operand method."""

    def test_number_literal_is_simple(self):
        """NumberLiteral is a simple operand."""
        lowerer, _, _ = create_minimal_lowerer()
        expr = NumberLiteral(42, make_loc())
        assert lowerer._is_simple_operand(expr) is True

    def test_signal_literal_is_simple(self):
        """SignalLiteral is a simple operand."""
        lowerer, _, _ = create_minimal_lowerer()
        expr = SignalLiteral("iron-plate", NumberLiteral(10, make_loc()), make_loc())
        assert lowerer._is_simple_operand(expr) is True

    def test_identifier_is_simple(self):
        """IdentifierExpr is a simple operand."""
        lowerer, _, _ = create_minimal_lowerer()
        expr = IdentifierExpr("x", make_loc())
        assert lowerer._is_simple_operand(expr) is True

    def test_binary_op_not_simple(self):
        """BinaryOp is not a simple operand."""
        lowerer, _, _ = create_minimal_lowerer()
        expr = BinaryOp(NumberLiteral(1, make_loc()), "+", NumberLiteral(2, make_loc()), make_loc())
        assert lowerer._is_simple_operand(expr) is False


class TestLowerUnaryOpDirect:
    """Direct unit tests for lower_unary_op method."""

    def test_unknown_unary_op_logs_error(self):
        """Unknown unary operator logs error."""
        lowerer, _, diags = create_minimal_lowerer()
        expr = UnaryOp("?", NumberLiteral(5, make_loc()), make_loc())
        lowerer.lower_unary_op(expr)
        # Check that an error was logged
        assert diags.has_errors()


class TestSuppressMaterializationDirect:
    """Direct unit tests for _suppress_value_ref_materialization method."""

    def test_suppress_non_signal_ref_does_nothing(self):
        """Suppressing a non-SignalRef does nothing."""
        lowerer, _, _ = create_minimal_lowerer()
        # Should not raise
        lowerer._suppress_value_ref_materialization(42)

    def test_suppress_signal_ref_marks_metadata(self):
        """Suppressing a SignalRef marks debug_metadata."""
        lowerer, ir_builder, _ = create_minimal_lowerer()
        ref = ir_builder.const("signal-A", 10)
        lowerer._suppress_value_ref_materialization(ref)
        op = ir_builder.get_operation(ref.source_id)
        assert op.debug_metadata.get("suppress_materialization") is True


class TestTryExtractConstValueDirect:
    """Direct unit tests for _try_extract_const_value method."""

    def test_extract_from_int(self):
        """Extracting from int returns the int."""
        lowerer, _, _ = create_minimal_lowerer()
        assert lowerer._try_extract_const_value(42) == 42

    def test_extract_from_const_op(self):
        """Extracting from IRConst returns the value."""
        lowerer, ir_builder, _ = create_minimal_lowerer()
        ref = ir_builder.const("signal-A", 123)
        assert lowerer._try_extract_const_value(ref) == 123

    def test_extract_from_non_const_returns_none(self):
        """Extracting from non-constant SignalRef returns None."""
        lowerer, ir_builder, _ = create_minimal_lowerer()
        # Create an arithmetic op
        ref = ir_builder.arithmetic("+", 1, 2, "signal-A")
        assert lowerer._try_extract_const_value(ref) is None


class TestLowerDictLiteralDirect:
    """Direct unit tests for lower_dict_literal method."""

    def test_dict_literal_with_number(self):
        """DictLiteral with NumberLiteral value."""
        from dsl_compiler.src.ast.literals import DictLiteral

        lowerer, _, _ = create_minimal_lowerer()

        entries = {"direction": NumberLiteral(4, make_loc())}
        expr = DictLiteral(entries, make_loc())
        result = lowerer.lower_dict_literal(expr)
        assert result == {"direction": 4}

    def test_dict_literal_with_string(self):
        """DictLiteral with StringLiteral value."""
        from dsl_compiler.src.ast.literals import DictLiteral

        lowerer, _, _ = create_minimal_lowerer()

        entries = {"filter": StringLiteral("iron-plate", make_loc())}
        expr = DictLiteral(entries, make_loc())
        result = lowerer.lower_dict_literal(expr)
        assert result == {"filter": "iron-plate"}
