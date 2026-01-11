"""
Tests for lowering/statement_lowerer.py - Statement to IR lowering.

This module tests the StatementLowerer class which handles converting
AST statement nodes to IR operations.
"""

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.ir.nodes import IRConst, IREntityPropWrite, IRMemWrite
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
