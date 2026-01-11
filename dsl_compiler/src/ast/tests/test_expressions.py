"""
Tests for ast/expressions.py - Expression AST nodes.
"""

from dsl_compiler.src.ast.base import ASTNode
from dsl_compiler.src.ast.expressions import BinaryOp, Expr, IdentifierExpr, UnaryOp
from dsl_compiler.src.ast.literals import NumberLiteral


class TestBinaryOp:
    """Tests for BinaryOp expression node."""

    def test_binary_op_creation(self):
        """Test BinaryOp node creation with left, op, and right."""
        left = NumberLiteral(5)
        right = NumberLiteral(3)
        node = BinaryOp("+", left, right)
        assert node.left == left
        assert node.op == "+"
        assert node.right == right

    def test_binary_op_inherits_from_expr(self):
        """BinaryOp should inherit from Expr."""
        node = BinaryOp("+", NumberLiteral(1), NumberLiteral(2))
        assert isinstance(node, Expr)
        assert isinstance(node, ASTNode)


class TestUnaryOp:
    """Tests for UnaryOp expression node."""

    def test_unary_op_creation(self):
        """Test UnaryOp node creation with op and operand."""
        operand = NumberLiteral(5)
        node = UnaryOp("-", operand)
        assert node.op == "-"
        assert node.expr == operand

    def test_unary_op_inherits_from_expr(self):
        """UnaryOp should inherit from Expr."""
        node = UnaryOp("-", NumberLiteral(5))
        assert isinstance(node, Expr)
        assert isinstance(node, ASTNode)


class TestIdentifierExpr:
    """Tests for IdentifierExpr expression node."""

    def test_identifier_expr_creation(self):
        """Test IdentifierExpr node creation with name."""
        node = IdentifierExpr("var_name")
        assert node.name == "var_name"

    def test_identifier_expr_inherits_from_expr(self):
        """IdentifierExpr should inherit from Expr."""
        node = IdentifierExpr("x")
        assert isinstance(node, Expr)
        assert isinstance(node, ASTNode)


class TestExpressionHierarchy:
    """Tests for expression node hierarchy and inheritance."""

    def test_all_expressions_inherit_from_expr(self):
        """All expression types should inherit from Expr and ASTNode."""
        expressions = [
            NumberLiteral(5),
            IdentifierExpr("x"),
            BinaryOp("+", NumberLiteral(1), NumberLiteral(2)),
            UnaryOp("-", NumberLiteral(5)),
        ]

        for expr in expressions:
            assert isinstance(expr, Expr)
            assert isinstance(expr, ASTNode)
