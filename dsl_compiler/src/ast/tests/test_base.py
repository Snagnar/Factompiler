"""
Tests for ast/base.py - Base AST node classes.
"""

from dsl_compiler.src.ast.base import ASTNode, ASTVisitor
from dsl_compiler.src.ast.expressions import BinaryOp
from dsl_compiler.src.ast.literals import Identifier, NumberLiteral


class TestASTNode:
    """Tests for the ASTNode base class."""

    def test_number_literal_inherits_from_ast_node(self):
        """NumberLiteral should inherit from ASTNode."""
        node = NumberLiteral(5)
        assert isinstance(node, ASTNode)

    def test_identifier_inherits_from_ast_node(self):
        """Identifier should inherit from ASTNode."""
        node = Identifier("x")
        assert isinstance(node, ASTNode)

    def test_binary_op_inherits_from_ast_node(self):
        """BinaryOp should inherit from ASTNode."""
        node = BinaryOp(NumberLiteral(1), "+", NumberLiteral(2))
        assert isinstance(node, ASTNode)

    def test_ast_node_with_location(self):
        """Test ASTNode with line/column."""
        node = NumberLiteral(value=42, line=10, column=5)
        assert node.line == 10
        assert node.column == 5


class TestASTVisitor:
    """Tests for the ASTVisitor base class."""

    def test_visitor_generic_visit(self):
        """Test generic_visit is called for unhandled nodes."""
        visitor = ASTVisitor()
        node = NumberLiteral(42)
        result = visitor.visit(node)
        # generic_visit returns None by default
        assert result is None

    def test_visitor_calls_specific_method(self):
        """Test visitor calls specific visit_ClassName method."""

        class TestVisitor(ASTVisitor):
            def visit_NumberLiteral(self, node):
                return f"visited number: {node.value}"

        visitor = TestVisitor()
        node = NumberLiteral(42)
        result = visitor.visit(node)
        assert result == "visited number: 42"

    def test_visitor_falls_back_to_generic(self):
        """Test visitor falls back to generic_visit for unhandled types."""

        class TestVisitor(ASTVisitor):
            def __init__(self):
                self.generic_called = False

            def generic_visit(self, node):
                self.generic_called = True
                return "generic"

        visitor = TestVisitor()
        node = NumberLiteral(42)
        result = visitor.visit(node)
        assert visitor.generic_called
        assert result == "generic"

    def test_visitor_different_node_types(self):
        """Test visitor dispatches to correct methods."""

        class TestVisitor(ASTVisitor):
            def visit_NumberLiteral(self, node):
                return "number"

            def visit_Identifier(self, node):
                return "identifier"

        visitor = TestVisitor()
        assert visitor.visit(NumberLiteral(42)) == "number"
        assert visitor.visit(Identifier("x")) == "identifier"
