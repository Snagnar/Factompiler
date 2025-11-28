"""
Tests for dsl_ast.py - AST node classes and structures.
"""

from dsl_compiler.src.ast.literals import (
    NumberLiteral,
    StringLiteral,
    PropertyAccess,
    Identifier,
    LValue,
)
from dsl_compiler.src.ast.expressions import BinaryOp, UnaryOp, Expr
from dsl_compiler.src.ast.statements import (
    DeclStmt,
    AssignStmt,
    MemDecl,
    ExprStmt,
    ReturnStmt,
    ImportStmt,
    FuncDecl,
    Statement,
    Program,
)
from dsl_compiler.src.ast.expressions import IdentifierExpr
from dsl_compiler.src.ast.base import ASTNode


class TestLiteralNodes:
    """Test literal AST nodes."""

    def test_number_literal(self):
        """Test NumberLiteral node."""
        node = NumberLiteral(42)
        assert node.value == 42

    def test_string_literal(self):
        """Test StringLiteral node."""
        node = StringLiteral("test")
        assert node.value == "test"


class TestIdentifierNodes:
    """Test identifier and property access nodes."""

    def test_identifier(self):
        """Test Identifier node."""
        node = Identifier("test_var")
        assert node.name == "test_var"

    def test_identifier_expr(self):
        """Test IdentifierExpr node."""
        node = IdentifierExpr("var_name")
        assert node.name == "var_name"

    def test_property_access(self):
        """Test PropertyAccess node."""
        node = PropertyAccess("bundle", "field")
        assert node.object_name == "bundle"
        assert node.property_name == "field"


class TestExpressionNodes:
    """Test expression AST nodes."""

    def test_binary_op(self):
        """Test BinaryOp node."""
        left = NumberLiteral(5)
        right = NumberLiteral(3)
        node = BinaryOp("+", left, right)
        assert node.left == left
        assert node.op == "+"
        assert node.right == right

    def test_unary_op(self):
        """Test UnaryOp node."""
        operand = NumberLiteral(5)
        node = UnaryOp("-", operand)
        assert node.op == "-"
        assert node.expr == operand


class TestStatementNodes:
    """Test statement AST nodes."""

    def test_decl_statement(self):
        """Test DeclStmt node."""
        value = NumberLiteral(42)
        node = DeclStmt("Signal", "x", value)
        assert node.type_name == "Signal"
        assert node.name == "x"
        assert node.value == value

    def test_assign_statement(self):
        """Test AssignStmt node."""
        target = Identifier("x")
        value = NumberLiteral(42)
        node = AssignStmt(target, value)
        assert node.target == target
        assert node.value == value

    def test_memory_declaration(self):
        """Test MemDecl node with explicit signal type."""
        node = MemDecl("counter", signal_type="iron-plate")
        assert node.name == "counter"
        assert node.signal_type == "iron-plate"

    def test_expr_statement(self):
        """Test ExprStmt node."""
        expr = NumberLiteral(42)
        node = ExprStmt(expr)
        assert node.expr == expr

    def test_return_statement(self):
        """Test ReturnStmt node."""
        expr = NumberLiteral(42)
        node = ReturnStmt(expr)
        assert node.expr == expr

    def test_import_statement(self):
        """Test ImportStmt node."""
        node = ImportStmt("module_name", "alias")
        assert node.path == "module_name"
        assert node.alias == "alias"


class TestTopLevelNodes:
    """Test top-level declaration nodes."""

    def test_function_declaration(self):
        """Test FuncDecl node."""
        params = ["a", "b"]  # FuncDecl expects List[str], not List[Identifier]
        body = [ExprStmt(NumberLiteral(1))]
        node = FuncDecl("test_func", params, body)
        assert node.name == "test_func"
        assert node.params == params
        assert node.body == body

    def test_program_node(self):
        """Test Program node."""
        statements = [DeclStmt("Signal", "x", NumberLiteral(42))]
        node = Program(statements)
        assert node.statements == statements


class TestASTHierarchy:
    """Test AST node inheritance and type relationships."""

    def test_ast_node_base(self):
        """Test ASTNode base class functionality."""
        # Most concrete tests are done through subclasses
        # Just verify that all AST nodes inherit from ASTNode
        assert isinstance(NumberLiteral(5), ASTNode)
        assert isinstance(Identifier("x"), ASTNode)
        assert isinstance(BinaryOp(NumberLiteral(1), "+", NumberLiteral(2)), ASTNode)

    def test_statement_hierarchy(self):
        """Test statement node inheritance."""
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

    def test_expression_hierarchy(self):
        """Test expression node inheritance."""
        expressions = [
            NumberLiteral(5),
            StringLiteral("test"),
            IdentifierExpr("x"),
            BinaryOp(NumberLiteral(1), "+", NumberLiteral(2)),
            UnaryOp("-", NumberLiteral(5)),
        ]

        for expr in expressions:
            assert isinstance(expr, Expr)
            assert isinstance(expr, ASTNode)

    def test_lvalue_hierarchy(self):
        """Test LValue node inheritance."""
        lvalues = [
            Identifier("x"),
            PropertyAccess(Identifier("obj"), "field"),
        ]

        for lvalue in lvalues:
            assert isinstance(lvalue, LValue)
            assert isinstance(lvalue, ASTNode)
