"""
Tests for ast/statements.py - Statement AST nodes.
"""

import pytest

from dsl_compiler.src.ast.base import ASTNode
from dsl_compiler.src.ast.literals import Identifier, NumberLiteral
from dsl_compiler.src.ast.statements import (
    AssignStmt,
    DeclStmt,
    ExprStmt,
    ForStmt,
    FuncDecl,
    ImportStmt,
    MemDecl,
    Program,
    ReturnStmt,
    Statement,
    TypedParam,
)


class TestDeclStmt:
    """Tests for DeclStmt (declaration statement) node."""

    def test_decl_stmt_creation(self):
        """Test DeclStmt node stores type, name, and value."""
        value = NumberLiteral(42)
        node = DeclStmt("Signal", "x", value)
        assert node.type_name == "Signal"
        assert node.name == "x"
        assert node.value == value

    def test_decl_stmt_inherits_from_statement(self):
        """DeclStmt should inherit from Statement."""
        node = DeclStmt("Signal", "x", NumberLiteral(5))
        assert isinstance(node, Statement)
        assert isinstance(node, ASTNode)


class TestAssignStmt:
    """Tests for AssignStmt (assignment statement) node."""

    def test_assign_stmt_creation(self):
        """Test AssignStmt node stores target and value."""
        target = Identifier("x")
        value = NumberLiteral(42)
        node = AssignStmt(target, value)
        assert node.target == target
        assert node.value == value

    def test_assign_stmt_inherits_from_statement(self):
        """AssignStmt should inherit from Statement."""
        node = AssignStmt(Identifier("x"), NumberLiteral(5))
        assert isinstance(node, Statement)
        assert isinstance(node, ASTNode)


class TestMemDecl:
    """Tests for MemDecl (memory declaration) node."""

    def test_mem_decl_creation_with_signal_type(self):
        """Test MemDecl node with explicit signal type."""
        node = MemDecl("counter", signal_type="iron-plate")
        assert node.name == "counter"
        assert node.signal_type == "iron-plate"

    def test_mem_decl_inherits_from_statement(self):
        """MemDecl should inherit from Statement."""
        node = MemDecl("mem", "signal-A", NumberLiteral(0))
        assert isinstance(node, Statement)
        assert isinstance(node, ASTNode)


class TestExprStmt:
    """Tests for ExprStmt (expression statement) node."""

    def test_expr_stmt_creation(self):
        """Test ExprStmt node wraps an expression."""
        expr = NumberLiteral(42)
        node = ExprStmt(expr)
        assert node.expr == expr

    def test_expr_stmt_inherits_from_statement(self):
        """ExprStmt should inherit from Statement."""
        node = ExprStmt(NumberLiteral(42))
        assert isinstance(node, Statement)
        assert isinstance(node, ASTNode)


class TestReturnStmt:
    """Tests for ReturnStmt node."""

    def test_return_stmt_creation(self):
        """Test ReturnStmt node stores expression."""
        expr = NumberLiteral(42)
        node = ReturnStmt(expr)
        assert node.expr == expr

    def test_return_stmt_inherits_from_statement(self):
        """ReturnStmt should inherit from Statement."""
        node = ReturnStmt(NumberLiteral(42))
        assert isinstance(node, Statement)
        assert isinstance(node, ASTNode)


class TestImportStmt:
    """Tests for ImportStmt node."""

    def test_import_stmt_creation(self):
        """Test ImportStmt node stores path and alias."""
        node = ImportStmt("module_name", "alias")
        assert node.path == "module_name"
        assert node.alias == "alias"

    def test_import_stmt_inherits_from_statement(self):
        """ImportStmt should inherit from Statement."""
        node = ImportStmt("module", None)
        assert isinstance(node, Statement)
        assert isinstance(node, ASTNode)


class TestFuncDecl:
    """Tests for FuncDecl (function declaration) node."""

    def test_func_decl_creation(self):
        """Test FuncDecl node stores name, params, and body."""
        params = ["a", "b"]
        body = [ExprStmt(NumberLiteral(1))]
        node = FuncDecl("test_func", params, body)
        assert node.name == "test_func"
        assert node.params == params
        assert node.body == body

    def test_func_decl_inherits_from_ast_node(self):
        """FuncDecl should inherit from ASTNode."""
        node = FuncDecl("f", [], [])
        assert isinstance(node, ASTNode)


class TestProgram:
    """Tests for Program (top-level) node."""

    def test_program_creation(self):
        """Test Program node stores list of statements."""
        statements = [DeclStmt("Signal", "x", NumberLiteral(42))]
        node = Program(statements)
        assert node.statements == statements

    def test_program_inherits_from_ast_node(self):
        """Program should inherit from ASTNode."""
        node = Program([])
        assert isinstance(node, ASTNode)


class TestStatementHierarchy:
    """Tests for statement node hierarchy and inheritance."""

    def test_all_statements_inherit_from_statement(self):
        """All statement types should inherit from Statement and ASTNode."""
        statements = [
            DeclStmt("Signal", "x", NumberLiteral(5)),
            AssignStmt(Identifier("x"), NumberLiteral(5)),
            MemDecl("mem", "signal-A", NumberLiteral(0)),
            ExprStmt(NumberLiteral(42)),
            ReturnStmt(NumberLiteral(42)),
            ImportStmt("module", "alias"),
        ]

        for stmt in statements:
            assert isinstance(stmt, Statement)
            assert isinstance(stmt, ASTNode)


class TestTypedParam:
    """Tests for TypedParam dataclass."""

    def test_typed_param_creation(self):
        """Test TypedParam stores type and name."""
        param = TypedParam("Signal", "x")
        assert param.type_name == "Signal"
        assert param.name == "x"

    def test_typed_param_with_location(self):
        """Test TypedParam with location info."""
        param = TypedParam("int", "count", line=10, column=5)
        assert param.line == 10
        assert param.column == 5


class TestForStmt:
    """Tests for ForStmt (for loop statement) node."""

    def test_for_stmt_creation_with_range(self):
        """Test ForStmt creation with range iterator."""
        body = [ExprStmt(NumberLiteral(1))]
        node = ForStmt(
            iterator_name="i",
            start=0,
            stop=10,
            step=1,
            values=None,
            body=body,
        )
        assert node.iterator_name == "i"
        assert node.start == 0
        assert node.stop == 10
        assert node.step == 1
        assert node.values is None
        assert node.body == body

    def test_for_stmt_creation_with_values(self):
        """Test ForStmt creation with explicit values list."""
        body = []
        node = ForStmt(
            iterator_name="x",
            start=None,
            stop=None,
            step=None,
            values=[1, 2, 3],
            body=body,
        )
        assert node.values == [1, 2, 3]

    def test_for_stmt_inherits_from_statement(self):
        """ForStmt should inherit from Statement."""
        node = ForStmt("i", 0, 10, 1, None, [])
        assert isinstance(node, Statement)
        assert isinstance(node, ASTNode)

    def test_get_iteration_values_with_list(self):
        """Test get_iteration_values returns values list."""
        node = ForStmt("x", None, None, None, [5, 10, 15], [])
        assert node.get_iteration_values() == [5, 10, 15]

    def test_get_iteration_values_with_range(self):
        """Test get_iteration_values returns range values."""
        node = ForStmt("i", 0, 5, 1, None, [])
        assert node.get_iteration_values() == [0, 1, 2, 3, 4]

    def test_get_iteration_values_range_with_step(self):
        """Test get_iteration_values with step."""
        node = ForStmt("i", 0, 10, 2, None, [])
        assert node.get_iteration_values() == [0, 2, 4, 6, 8]

    def test_get_iteration_values_auto_negative_step(self):
        """Test auto-detected negative step for descending range."""
        node = ForStmt("i", 5, 0, None, None, [])
        assert node.get_iteration_values() == [5, 4, 3, 2, 1]

    def test_get_iteration_values_auto_positive_step(self):
        """Test auto-detected positive step for ascending range."""
        node = ForStmt("i", 0, 3, None, None, [])
        assert node.get_iteration_values() == [0, 1, 2]

    def test_get_iteration_values_explicit_negative_step(self):
        """Test explicit negative step."""
        node = ForStmt("i", 10, 0, -2, None, [])
        assert node.get_iteration_values() == [10, 8, 6, 4, 2]

    def test_get_iteration_values_with_variable_bounds(self):
        """Test get_iteration_values with variable bounds and resolver."""
        node = ForStmt("i", "start_var", "end_var", None, None, [])

        def resolver(name: str) -> int:
            return {"start_var": 1, "end_var": 4}[name]

        assert node.get_iteration_values(resolver) == [1, 2, 3]

    def test_get_iteration_values_variable_without_resolver_raises(self):
        """Test that variable bounds without resolver raises error."""
        node = ForStmt("i", "start_var", 10, None, None, [])
        with pytest.raises(ValueError) as exc_info:
            node.get_iteration_values()
        assert "requires a constant resolver" in str(exc_info.value)

    def test_get_iteration_values_empty_range(self):
        """Test get_iteration_values returns empty for zero-length range."""
        node = ForStmt("i", 5, 5, 1, None, [])
        assert node.get_iteration_values() == []
