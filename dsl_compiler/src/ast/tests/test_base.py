"""
Tests for ast/base.py - Base AST node classes.
"""

from dsl_compiler.src.ast.base import ASTNode
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
