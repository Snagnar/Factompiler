"""
Tests for ast/statements.py - Statement AST nodes.
"""

from dsl_compiler.src.ast.base import ASTNode
from dsl_compiler.src.ast.literals import Identifier, NumberLiteral
from dsl_compiler.src.ast.statements import (
    AssignStmt,
    DeclStmt,
    ExprStmt,
    FuncDecl,
    ImportStmt,
    MemDecl,
    Program,
    ReturnStmt,
    Statement,
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
