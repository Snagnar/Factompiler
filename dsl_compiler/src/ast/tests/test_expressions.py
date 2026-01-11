"""
Tests for ast/expressions.py - Expression AST nodes.
"""

from dsl_compiler.src.ast.base import ASTNode
from dsl_compiler.src.ast.expressions import (
    BinaryOp,
    EntityOutputExpr,
    Expr,
    IdentifierExpr,
    OutputSpecExpr,
    PropertyAccessExpr,
    SignalTypeAccess,
    UnaryOp,
)
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


class TestPropertyAccessExpr:
    """Tests for PropertyAccessExpr expression node."""

    def test_property_access_expr_creation(self):
        """Test PropertyAccessExpr node creation."""
        node = PropertyAccessExpr("object", "property")
        assert node.object_name == "object"
        assert node.property_name == "property"

    def test_property_access_expr_inherits_from_expr(self):
        """PropertyAccessExpr should inherit from Expr."""
        node = PropertyAccessExpr("obj", "prop")
        assert isinstance(node, Expr)
        assert isinstance(node, ASTNode)


class TestOutputSpecExpr:
    """Tests for OutputSpecExpr expression node."""

    def test_output_spec_expr_creation(self):
        """Test OutputSpecExpr node creation."""
        condition = BinaryOp("<", IdentifierExpr("x"), NumberLiteral(10))
        output = IdentifierExpr("y")
        node = OutputSpecExpr(condition, output)
        assert node.condition is condition
        assert node.output_value is output

    def test_output_spec_expr_inherits_from_expr(self):
        """OutputSpecExpr should inherit from Expr."""
        condition = BinaryOp("<", IdentifierExpr("x"), NumberLiteral(10))
        output = NumberLiteral(1)
        node = OutputSpecExpr(condition, output)
        assert isinstance(node, Expr)
        assert isinstance(node, ASTNode)


class TestSignalTypeAccess:
    """Tests for SignalTypeAccess expression node."""

    def test_signal_type_access_creation(self):
        """Test SignalTypeAccess node creation."""
        node = SignalTypeAccess("signal_var", "type")
        assert node.object_name == "signal_var"
        assert node.property_name == "type"

    def test_signal_type_access_inherits_from_expr(self):
        """SignalTypeAccess should inherit from Expr."""
        node = SignalTypeAccess("sig", "type")
        assert isinstance(node, Expr)
        assert isinstance(node, ASTNode)


class TestEntityOutputExpr:
    """Tests for EntityOutputExpr expression node."""

    def test_entity_output_expr_creation(self):
        """Test EntityOutputExpr node creation."""
        node = EntityOutputExpr("chest")
        assert node.entity_name == "chest"

    def test_entity_output_expr_inherits_from_expr(self):
        """EntityOutputExpr should inherit from Expr."""
        node = EntityOutputExpr("tank")
        assert isinstance(node, Expr)
        assert isinstance(node, ASTNode)
