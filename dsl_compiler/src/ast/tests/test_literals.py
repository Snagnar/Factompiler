"""
Tests for ast/literals.py - Literal AST nodes.
"""

from dsl_compiler.src.ast.base import ASTNode
from dsl_compiler.src.ast.expressions import Expr
from dsl_compiler.src.ast.literals import (
    DictLiteral,
    Identifier,
    LValue,
    NumberLiteral,
    PropertyAccess,
    StringLiteral,
)


class TestNumberLiteral:
    """Tests for NumberLiteral node."""

    def test_number_literal_creation(self):
        """Test NumberLiteral node stores integer value."""
        node = NumberLiteral(42)
        assert node.value == 42

    def test_number_literal_inherits_from_expr(self):
        """NumberLiteral should inherit from Expr."""
        node = NumberLiteral(5)
        assert isinstance(node, Expr)
        assert isinstance(node, ASTNode)


class TestStringLiteral:
    """Tests for StringLiteral node."""

    def test_string_literal_creation(self):
        """Test StringLiteral node stores string value."""
        node = StringLiteral("test")
        assert node.value == "test"

    def test_string_literal_inherits_from_expr(self):
        """StringLiteral should inherit from Expr."""
        node = StringLiteral("hello")
        assert isinstance(node, Expr)
        assert isinstance(node, ASTNode)


class TestIdentifier:
    """Tests for Identifier node."""

    def test_identifier_creation(self):
        """Test Identifier node stores name."""
        node = Identifier("test_var")
        assert node.name == "test_var"

    def test_identifier_inherits_from_lvalue(self):
        """Identifier should inherit from LValue."""
        node = Identifier("x")
        assert isinstance(node, LValue)
        assert isinstance(node, ASTNode)


class TestPropertyAccess:
    """Tests for PropertyAccess node."""

    def test_property_access_creation(self):
        """Test PropertyAccess node stores object and property names."""
        node = PropertyAccess("bundle", "field")
        assert node.object_name == "bundle"
        assert node.property_name == "field"

    def test_property_access_inherits_from_lvalue(self):
        """PropertyAccess should inherit from LValue."""
        node = PropertyAccess(Identifier("obj"), "field")
        assert isinstance(node, LValue)
        assert isinstance(node, ASTNode)


class TestLValueHierarchy:
    """Tests for LValue node hierarchy."""

    def test_all_lvalues_inherit_from_lvalue(self):
        """All LValue types should inherit from LValue and ASTNode."""
        lvalues = [
            Identifier("x"),
            PropertyAccess(Identifier("obj"), "field"),
        ]

        for lvalue in lvalues:
            assert isinstance(lvalue, LValue)
            assert isinstance(lvalue, ASTNode)


class TestDictLiteral:
    """Tests for DictLiteral node."""

    def test_dict_literal_creation(self):
        """Test DictLiteral node stores entries."""
        entries = {"key1": NumberLiteral(1), "key2": NumberLiteral(2)}
        node = DictLiteral(entries)
        assert node.entries == entries
        assert len(node.entries) == 2

    def test_dict_literal_inherits_from_expr(self):
        """DictLiteral should inherit from Expr."""
        node = DictLiteral({})
        assert isinstance(node, Expr)
        assert isinstance(node, ASTNode)

    def test_dict_literal_empty(self):
        """Test DictLiteral with empty entries."""
        node = DictLiteral({})
        assert node.entries == {}
