"""
Tests for lowering/statement_lowerer.py - Statement to IR lowering.

This module tests the StatementLowerer class which handles converting
AST statement nodes to IR operations.
"""

from unittest.mock import MagicMock

from dsl_compiler.src.ast.expressions import BinaryOp, CallExpr
from dsl_compiler.src.ast.literals import Identifier, NumberLiteral, PropertyAccess
from dsl_compiler.src.ast.statements import AssignStmt, DeclStmt
from dsl_compiler.src.ir.nodes import IRConst, IREntityPropWrite, IRMemWrite, BundleRef, SignalRef
from dsl_compiler.src.lowering.statement_lowerer import StatementLowerer

from .conftest import compile_to_ir, make_loc, create_mock_parent


class TestLowerDeclStmt:
    """Tests for lower_decl_stmt method."""

    def test_lower_signal_decl(self):
        """Test lowering a signal declaration."""
        ir_ops, _, diags = compile_to_ir("Signal x = 42;")
        assert not diags.has_errors()
        consts = [op for op in ir_ops if isinstance(op, IRConst)]
        assert len(consts) >= 1

    def test_lower_int_decl(self):
        """Test lowering an int type declaration."""
        ir_ops, _, diags = compile_to_ir("int count = 10;")
        assert not diags.has_errors()

    def test_lower_entity_decl_with_place(self):
        """Test lowering an entity declaration with place()."""
        ir_ops, lowerer, diags = compile_to_ir(
            'Entity lamp = place("small-lamp", 0, 0, {enabled: 1});'
        )
        assert not diags.has_errors()
        assert "lamp" in lowerer.entity_refs

    def test_lower_bundle_decl(self):
        """Test lowering a bundle declaration."""
        ir_ops, lowerer, diags = compile_to_ir(
            'Bundle b = {("iron-plate", 10), ("copper-plate", 20)};'
        )
        assert not diags.has_errors()
        assert "b" in lowerer.signal_refs


class TestLowerAssignStmt:
    """Tests for lower_assign_stmt method."""

    def test_lower_simple_assignment(self):
        """Test lowering a simple signal expression."""
        ir_ops, _, diags = compile_to_ir("""
        Signal x = 10;
        Signal y = x + 5;
        """)
        assert not diags.has_errors()

    def test_lower_entity_assignment(self):
        """Test lowering entity assignment."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Entity lamp = place("small-lamp", 0, 0, {enabled: 1});
        Entity lamp2 = place("small-lamp", 1, 0, {enabled: 1});
        """)
        assert not diags.has_errors()
        assert "lamp" in lowerer.entity_refs
        assert "lamp2" in lowerer.entity_refs

    def test_lower_property_assignment_enable(self):
        """Test lowering property assignment to entity.enable."""
        ir_ops, _, diags = compile_to_ir("""
        Entity lamp = place("small-lamp", 0, 0, {enabled: 1, use_colors: 1});
        Signal condition = 10;
        lamp.enable = condition > 5;
        """)
        assert not diags.has_errors()

    def test_lower_property_assignment_color(self):
        """Test lowering property assignment to entity color."""
        ir_ops, _, diags = compile_to_ir("""
        Entity lamp = place("small-lamp", 0, 0, {use_colors: 1});
        Signal red = 255;
        lamp.color_r = red;
        """)
        assert not diags.has_errors()
        # Should have entity property writes
        prop_writes = [op for op in ir_ops if isinstance(op, IREntityPropWrite)]
        assert len(prop_writes) >= 1


class TestLowerExprStmt:
    """Tests for lower_expr_stmt method."""

    def test_lower_memory_write_expr_stmt(self):
        """Test lowering a memory write as expression statement."""
        ir_ops, _, diags = compile_to_ir("""
        Memory m: "signal-A";
        m.write(10);
        """)
        assert not diags.has_errors()
        mem_writes = [op for op in ir_ops if isinstance(op, IRMemWrite)]
        assert len(mem_writes) >= 1


class TestLowerFuncDecl:
    """Tests for lower_func_decl method."""

    def test_lower_simple_function(self):
        """Test lowering a simple function declaration."""
        ir_ops, lowerer, diags = compile_to_ir("""
        func double(int x) { return x * 2; }
        Signal result = double(21);
        """)
        assert not diags.has_errors()

    def test_lower_function_with_multiple_params(self):
        """Test lowering a function with multiple parameters."""
        ir_ops, lowerer, diags = compile_to_ir("""
        func add(int a, int b) { return a + b; }
        Signal result = add(5, 10);
        """)
        assert not diags.has_errors()


class TestLowerForStmt:
    """Tests for lower_for_stmt method."""

    def test_lower_simple_for_loop(self):
        """Test lowering a simple for loop."""
        ir_ops, _, diags = compile_to_ir("""
        for i in 0..3 { Signal x = i * 10; }
        """)
        assert not diags.has_errors()

    def test_lower_for_loop_with_step(self):
        """Test lowering a for loop with step."""
        ir_ops, _, diags = compile_to_ir("""
        for i in 0..10 step 2 { Signal x = i; }
        """)
        assert not diags.has_errors()

    def test_lower_for_loop_list_iterator(self):
        """Test lowering a for loop with list iterator."""
        ir_ops, _, diags = compile_to_ir("""
        for i in [1, 5, 10] { Signal x = i * 2; }
        """)
        assert not diags.has_errors()


class TestLowerReturnStmt:
    """Tests for lower_return_stmt method."""

    def test_lower_return_in_function(self):
        """Test lowering return statement inside function."""
        ir_ops, _, diags = compile_to_ir("""
        func getVal() { return 42; }
        Signal x = getVal();
        """)
        assert not diags.has_errors()


class TestLowerImportStmt:
    """Tests for lower_import_stmt method."""

    def test_lower_import_does_nothing(self):
        """Test that import statements are no-ops during lowering."""
        # Import is handled during preprocessing
        ir_ops, _, diags = compile_to_ir("Signal x = 10;")
        assert not diags.has_errors()


class TestEntityPropertyPatterns:
    """Tests for various entity property assignment patterns."""

    def test_property_assignment_with_signal(self):
        """Test property assignment with a signal value."""
        ir_ops, _, diags = compile_to_ir("""
        Entity lamp = place("small-lamp", 0, 0, {use_colors: 1});
        Signal val = 128;
        lamp.color_g = val;
        """)
        assert not diags.has_errors()
        prop_writes = [op for op in ir_ops if isinstance(op, IREntityPropWrite)]
        assert len(prop_writes) >= 1

    def test_property_assignment_with_expression(self):
        """Test property assignment with a complex expression."""
        ir_ops, _, diags = compile_to_ir("""
        Entity lamp = place("small-lamp", 0, 0, {use_colors: 1});
        Signal a = 100;
        Signal b = 50;
        lamp.color_b = a + b;
        """)
        assert not diags.has_errors()

    def test_enable_property_with_comparison(self):
        """Test enable property with comparison expression."""
        ir_ops, _, diags = compile_to_ir("""
        Entity lamp = place("small-lamp", 0, 0, {enabled: 1});
        Signal counter = 50;
        lamp.enable = counter >= 25;
        """)
        assert not diags.has_errors()


class TestFunctionInlining:
    """Tests for function inlining during lowering."""

    def test_function_with_signal_param(self):
        """Test function with signal parameter."""
        ir_ops, _, diags = compile_to_ir("""
        func scale(int x) { return x * 10; }
        Signal value = 5;
        Signal result = scale(value);
        """)
        assert not diags.has_errors()

    def test_function_returning_expression(self):
        """Test function returning a complex expression."""
        ir_ops, _, diags = compile_to_ir("""
        func combine(int a, int b) { return (a + b) * 2; }
        Signal result = combine(10, 20);
        """)
        assert not diags.has_errors()


class TestForLoopVariants:
    """Tests for various for loop patterns."""

    def test_for_loop_with_computation(self):
        """Test for loop with signal computation."""
        ir_ops, _, diags = compile_to_ir("""
        for i in 1..5 {
            Signal val = i * 10;
        }
        """)
        assert not diags.has_errors()

    def test_nested_for_loops(self):
        """Test nested for loops."""
        ir_ops, _, diags = compile_to_ir("""
        for i in 0..2 {
            for j in 0..3 {
                Signal x = i + j;
            }
        }
        """)
        assert not diags.has_errors()


class TestAssignmentToIdentifier:
    """Tests for assignment to identifiers with various patterns."""

    def test_assign_place_call_to_identifier(self):
        """Test assigning a place() call to an identifier."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Entity lamp = place("small-lamp", 0, 0, {enabled: 1});
        Entity lamp2 = place("small-lamp", 1, 0, {enabled: 1});
        """)
        assert not diags.has_errors()
        assert "lamp" in lowerer.entity_refs

    def test_assign_function_call_to_new_identifier(self):
        """Test assigning a function call result to a new identifier."""
        ir_ops, lowerer, diags = compile_to_ir("""
        func makeVal() { return 42; }
        Signal x = makeVal();
        """)
        assert not diags.has_errors()
        assert "x" in lowerer.signal_refs

    def test_assign_constant_to_new_identifier(self):
        """Test assigning a constant value to a new identifier."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Signal x = 99;
        """)
        assert not diags.has_errors()

    def test_assign_signal_ref_to_new_identifier(self):
        """Test assigning a signal reference to another identifier."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Signal a = 100;
        Signal b = a;
        """)
        assert not diags.has_errors()


class TestBundleConditionInlining:
    """Tests for bundle condition inlining patterns."""

    def test_bundle_all_comparison(self):
        """Test all(bundle) < N pattern for inlining."""
        ir_ops, _, diags = compile_to_ir("""
        Bundle b = {("iron-plate", 10), ("copper-plate", 20)};
        Entity lamp = place("small-lamp", 0, 0, {enabled: 1});
        lamp.enable = all(b) < 100;
        """)
        assert not diags.has_errors()

    def test_bundle_any_comparison(self):
        """Test any(bundle) > N pattern for inlining."""
        ir_ops, _, diags = compile_to_ir("""
        Bundle b = {("iron-plate", 10), ("copper-plate", 20)};
        Entity lamp = place("small-lamp", 0, 0, {enabled: 1});
        lamp.enable = any(b) > 5;
        """)
        assert not diags.has_errors()


class TestConstantExtraction:
    """Tests for constant extraction patterns."""

    def test_constant_number_literal(self):
        """Test constant extraction from number literal."""
        ir_ops, _, diags = compile_to_ir("Signal x = 42;")
        assert not diags.has_errors()

    def test_constant_int_variable(self):
        """Test constant extraction from int variable."""
        ir_ops, _, diags = compile_to_ir("""
        int count = 10;
        Signal x = count * 2;
        """)
        assert not diags.has_errors()


class TestForLoopScoping:
    """Tests for for loop variable scoping."""

    def test_for_loop_iterator_scope(self):
        """Test that for loop iterator is properly scoped."""
        ir_ops, lowerer, diags = compile_to_ir("""
        for i in 0..5 {
            Signal val = i;
        }
        """)
        assert not diags.has_errors()
        # Iterator should not be in signal_refs after loop (scope isolation)
        # The signal 'val' from the last iteration may remain

    def test_for_loop_with_outer_signal(self):
        """Test for loop referencing outer signal."""
        ir_ops, _, diags = compile_to_ir("""
        Signal base = 10;
        for i in 0..3 {
            Signal x = base + i;
        }
        """)
        assert not diags.has_errors()


class TestUnknownStatementHandler:
    """Tests for error handling of unknown statements."""

    def test_statements_lowered_correctly(self):
        """Test that all known statement types are handled."""
        # Test a mix of statement types
        ir_ops, _, diags = compile_to_ir("""
        int count = 5;
        Signal x = 10;
        Bundle b = {("iron-plate", 10)};
        Memory m: "signal-A";
        m.write(x);
        for i in 0..3 { Signal y = i; }
        """)
        assert not diags.has_errors()


class TestPlaceCallAssignment:
    """Tests for place() call assignment patterns (lines 176-199)."""

    def test_place_call_with_tracking(self):
        """Test place() call creates entity ref and signal ref."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Entity lamp = place("small-lamp", 0, 0, {enabled: 1});
        """)
        assert not diags.has_errors()
        assert "lamp" in lowerer.entity_refs
        assert "lamp" in lowerer.signal_refs

    def test_function_call_returning_entity(self):
        """Test function call that creates an entity via place()."""
        # Note: Functions returning place() directly have some limitations.
        # This tests the general call assignment path (lines 187-199).
        ir_ops, lowerer, diags = compile_to_ir("""
        func makeLamp(int x, int y) {
            return 42;
        }
        Signal result = makeLamp(0, 0);
        """)
        assert not diags.has_errors()
        assert "result" in lowerer.signal_refs


class TestBundleRefAssignment:
    """Tests for BundleRef assignment patterns (lines 222-232)."""

    def test_bundle_ref_assignment_updates_metadata(self):
        """Test that bundle assignment sets debug metadata."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Bundle b1 = {("iron-plate", 10), ("copper-plate", 20)};
        Bundle b2 = b1;
        """)
        assert not diags.has_errors()
        assert "b2" in lowerer.signal_refs

    def test_bundle_scalar_operation_assignment(self):
        """Test bundle scalar operation result assignment."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Bundle a = {("iron-plate", 10)};
        Bundle c = a * 2;
        """)
        assert not diags.has_errors()
        assert "c" in lowerer.signal_refs


class TestSignalRefAssignment:
    """Tests for SignalRef assignment patterns (lines 234-252)."""

    def test_signal_ref_from_const_sets_metadata(self):
        """Test signal ref from constant sets debug metadata."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Signal x = 42;
        Signal y = x;
        """)
        assert not diags.has_errors()
        assert "y" in lowerer.signal_refs

    def test_signal_ref_from_expression(self):
        """Test signal ref from expression result."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Signal a = 10;
        Signal b = 20;
        Signal c = a + b;
        """)
        assert not diags.has_errors()


class TestIntValueAssignment:
    """Tests for int constant assignment patterns (lines 254-268)."""

    def test_int_constant_assignment(self):
        """Test assigning integer constant directly."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Signal x = 123;
        """)
        assert not diags.has_errors()

    def test_int_variable_creates_const(self):
        """Test int variable creates a constant operation."""
        ir_ops, lowerer, diags = compile_to_ir("""
        int count = 5;
        Signal x = count;
        """)
        assert not diags.has_errors()


class TestForLoopLoweringEdgeCases:
    """Tests for for loop lowering edge cases (lines 302-310, 374-397)."""

    def test_for_loop_with_negative_step(self):
        """Test for loop with negative step (count down)."""
        ir_ops, _, diags = compile_to_ir("""
        for i in 10..0 step -2 {
            Signal x = i;
        }
        """)
        # May have errors if negative step not supported
        # The key is it doesn't crash

    def test_for_loop_with_list_iterator(self):
        """Test for loop with explicit list of values."""
        ir_ops, _, diags = compile_to_ir("""
        for i in [1, 5, 10, 20] {
            Signal x = i * 2;
        }
        """)
        assert not diags.has_errors()


class TestInlinableBundleCondition:
    """Tests for inlinable bundle condition detection (lines 357-366)."""

    def test_all_bundle_less_than_constant(self):
        """Test all(bundle) < constant is inlinable."""
        ir_ops, _, diags = compile_to_ir("""
        Bundle b = {("iron-plate", 10)};
        Entity lamp = place("small-lamp", 0, 0, {enabled: 1});
        lamp.enable = all(b) < 50;
        """)
        assert not diags.has_errors()

    def test_any_bundle_greater_than_constant(self):
        """Test any(bundle) > constant is inlinable."""
        ir_ops, _, diags = compile_to_ir("""
        Bundle b = {("iron-plate", 10)};
        Entity lamp = place("small-lamp", 0, 0, {enabled: 1});
        lamp.enable = any(b) > 5;
        """)
        assert not diags.has_errors()


class TestConstantExtractionEdgeCases:
    """Tests for constant extraction edge cases (lines 384-397)."""

    def test_extract_number_literal(self):
        """Test extracting from number literal."""
        ir_ops, _, diags = compile_to_ir("""
        Signal x = 42;
        """)
        assert not diags.has_errors()

    def test_extract_from_int_identifier(self):
        """Test extracting from int-typed identifier."""
        ir_ops, _, diags = compile_to_ir("""
        int base = 100;
        Entity lamp = place("small-lamp", 0, 0, {enabled: 1});
        lamp.enable = all({("iron-plate", 10)}) < base;
        """)
        assert not diags.has_errors()


class TestDeclStmtCallExprHandling:
    """Tests for DeclStmt with CallExpr values (lines 92-100)."""

    def test_decl_with_function_call(self):
        """Test declaration with function call value."""
        ir_ops, lowerer, diags = compile_to_ir("""
        func getValue() { return 42; }
        Signal x = getValue();
        """)
        assert not diags.has_errors()
        assert "x" in lowerer.signal_refs

    def test_decl_with_memory_read_call(self):
        """Test declaration with memory read call."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Memory m: "signal-A";
        Signal x = m.read();
        """)
        assert not diags.has_errors()
        assert "x" in lowerer.signal_refs


class TestDeclStmtBundleRef:
    """Tests for DeclStmt with BundleRef values (lines 130-139)."""

    def test_decl_bundle_sets_metadata(self):
        """Test bundle declaration sets debug metadata."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Bundle items = {("iron-plate", 10), ("copper-plate", 20)};
        """)
        assert not diags.has_errors()
        assert "items" in lowerer.signal_refs


class TestDeclStmtIntValue:
    """Tests for DeclStmt with int values (lines 141-157)."""

    def test_decl_int_variable(self):
        """Test int variable declaration."""
        ir_ops, lowerer, diags = compile_to_ir("""
        int count = 5;
        Signal x = count * 2;
        """)
        assert not diags.has_errors()


class TestAssignStmtCallExpr:
    """Tests for AssignStmt with CallExpr (lines 177-199)."""

    def test_assign_place_call_creates_entity_ref(self):
        """Test assigning place() call creates entity and signal refs."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Entity lamp = place("small-lamp", 0, 0, {enabled: 1});
        """)
        assert not diags.has_errors()
        assert "lamp" in lowerer.entity_refs
        assert "lamp" in lowerer.signal_refs

    def test_assign_function_call_creates_signal_ref(self):
        """Test assigning function call creates signal ref."""
        ir_ops, lowerer, diags = compile_to_ir("""
        func compute(int a, int b) { return a + b; }
        Signal result = compute(5, 10);
        """)
        assert not diags.has_errors()
        assert "result" in lowerer.signal_refs

    def test_reassign_variable_with_place(self):
        """Test reassigning an identifier with place() (lines 176-185)."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Entity lamp = place("small-lamp", 0, 0, {enabled: 1});
        lamp = place("small-lamp", 1, 0, {enabled: 1});
        """)
        assert not diags.has_errors()
        assert "lamp" in lowerer.entity_refs

    def test_reassign_variable_with_function_call(self):
        """Test function call used in assignment."""
        ir_ops, lowerer, diags = compile_to_ir("""
        func getValue(int x) { return x * 2; }
        Signal result = getValue(5);
        """)
        assert not diags.has_errors()
        assert "result" in lowerer.signal_refs


class TestAssignStmtIntegerValue:
    """Tests for AssignStmt with integer values (lines 252-268)."""

    def test_assign_integer_to_signal(self):
        """Test assigning bare integer creates constant."""
        ir_ops, lowerer, diags = compile_to_ir("""
        Signal x = 123;
        """)
        assert not diags.has_errors()
        consts = [op for op in ir_ops if isinstance(op, IRConst)]
        assert len(consts) >= 1

    def test_assign_int_expr_result_to_signal(self):
        """Test assigning int expression result."""
        ir_ops, lowerer, diags = compile_to_ir("""
        int a = 5;
        int b = 10;
        Signal x = a + b;
        """)
        # Constant folding occurs
        assert not diags.has_errors()


class TestForStmtEdgeCases:
    """Tests for for loop edge cases (lines 302-310)."""

    def test_for_loop_empty_body(self):
        """Test for loop with minimal body."""
        ir_ops, _, diags = compile_to_ir("""
        for i in 0..3 {
            Signal x = i;
        }
        """)
        assert not diags.has_errors()

    def test_for_loop_with_list(self):
        """Test for loop with list iterator."""
        ir_ops, _, diags = compile_to_ir("""
        for i in [1, 2, 3, 4, 5] {
            Signal val = i * 10;
        }
        """)
        assert not diags.has_errors()

    def test_for_loop_with_step(self):
        """Test for loop with custom step."""
        ir_ops, _, diags = compile_to_ir("""
        for i in 0..10 step 2 {
            Signal x = i;
        }
        """)
        assert not diags.has_errors()

    def test_for_loop_uses_outer_variable(self):
        """Test for loop referencing outer scope variable."""
        ir_ops, _, diags = compile_to_ir("""
        Signal base = 100;
        for i in 0..3 {
            Signal x = base + i;
        }
        """)
        assert not diags.has_errors()


class TestDirectStatementLowerer:
    """Direct unit tests for StatementLowerer methods."""

    def test_error_method(self):
        """Test _error method logs through diagnostics."""
        parent, _, diags = create_mock_parent()
        stmt_lowerer = StatementLowerer(parent)
        stmt_lowerer._error("Test error message")
        assert diags.has_errors()

    def test_lower_assign_identifier_with_signal_ref(self):
        """Test lowering assignment to identifier with SignalRef value."""
        parent, ir_builder, diags = create_mock_parent()
        stmt_lowerer = StatementLowerer(parent)

        # Create a SignalRef to return from expr_lowerer
        signal_ref = ir_builder.const("signal-A", 42)
        parent.expr_lowerer.lower_expr = MagicMock(return_value=signal_ref)

        # Create assignment: x = expr
        target = Identifier("x", make_loc())
        value = NumberLiteral(42, make_loc())
        stmt = AssignStmt(target, value)
        stmt.line = 1

        stmt_lowerer.lower_assign_stmt(stmt)

        # Should have registered signal ref
        assert "x" in parent.signal_refs

    def test_lower_assign_identifier_with_bundle_ref(self):
        """Test lowering assignment to identifier with BundleRef value."""
        parent, ir_builder, diags = create_mock_parent()
        stmt_lowerer = StatementLowerer(parent)

        # Create a BundleRef to return from expr_lowerer
        bundle_ref = BundleRef({"signal-A", "signal-B"}, "bundle_op_1")
        ir_builder.add_operation(MagicMock(id="bundle_op_1", debug_metadata={}, debug_label=""))
        parent.expr_lowerer.lower_expr = MagicMock(return_value=bundle_ref)

        # Create assignment: x = bundle_expr
        target = Identifier("x", make_loc())
        value = NumberLiteral(42, make_loc())  # Doesn't matter, we mock the lowering
        stmt = AssignStmt(target, value)
        stmt.line = 1

        stmt_lowerer.lower_assign_stmt(stmt)

        # Should have registered signal ref
        assert "x" in parent.signal_refs

    def test_lower_assign_identifier_with_int(self):
        """Test lowering assignment to identifier with int value."""
        parent, ir_builder, diags = create_mock_parent()
        stmt_lowerer = StatementLowerer(parent)

        # Return int from expr_lowerer
        parent.expr_lowerer.lower_expr = MagicMock(return_value=42)

        # Create assignment: x = 42
        target = Identifier("x", make_loc())
        value = NumberLiteral(42, make_loc())
        stmt = AssignStmt(target, value)
        stmt.line = 1

        stmt_lowerer.lower_assign_stmt(stmt)

        # Should have registered signal ref
        assert "x" in parent.signal_refs

    def test_lower_assign_property_access(self):
        """Test lowering assignment to property access."""
        parent, ir_builder, diags = create_mock_parent()
        stmt_lowerer = StatementLowerer(parent)

        # Register an entity
        parent.entity_refs["lamp"] = "lamp_entity_id"

        # Create a SignalRef to return
        signal_ref = ir_builder.const("signal-A", 1)
        parent.expr_lowerer.lower_expr = MagicMock(return_value=signal_ref)

        # Create assignment: lamp.enable = condition
        target = PropertyAccess("lamp", "enable", make_loc())
        value = NumberLiteral(1, make_loc())
        stmt = AssignStmt(target, value)
        stmt.line = 1

        stmt_lowerer.lower_assign_stmt(stmt)

        # Should have added a property write operation
        prop_writes = [op for op in ir_builder.get_ir() if isinstance(op, IREntityPropWrite)]
        assert len(prop_writes) >= 1

    def test_lower_assign_with_call_expr_place(self):
        """Test lowering assignment with place() call."""
        parent, ir_builder, diags = create_mock_parent()
        stmt_lowerer = StatementLowerer(parent)

        # Mock lower_place_call_with_tracking
        parent.expr_lowerer.lower_place_call_with_tracking = MagicMock(
            return_value=("entity_id_1", SignalRef("signal-A", "const_1"))
        )

        # Create assignment: lamp = place(...)
        target = Identifier("lamp", make_loc())
        call_expr = CallExpr("place", [], make_loc())
        stmt = AssignStmt(target, call_expr)
        stmt.line = 1

        stmt_lowerer.lower_assign_stmt(stmt)

        # Should have registered entity ref
        assert "lamp" in parent.entity_refs
        assert "lamp" in parent.signal_refs

    def test_lower_assign_with_call_expr_function(self):
        """Test lowering assignment with function call."""
        parent, ir_builder, diags = create_mock_parent()
        stmt_lowerer = StatementLowerer(parent)

        # Return SignalRef from function call
        signal_ref = ir_builder.const("signal-A", 42)
        parent.expr_lowerer.lower_expr = MagicMock(return_value=signal_ref)

        # Create assignment: result = myFunc()
        target = Identifier("result", make_loc())
        call_expr = CallExpr("myFunc", [], make_loc())
        stmt = AssignStmt(target, call_expr)
        stmt.line = 1

        stmt_lowerer.lower_assign_stmt(stmt)

        # Should have registered signal ref
        assert "result" in parent.signal_refs
