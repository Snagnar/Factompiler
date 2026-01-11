"""
Tests for semantic/symbol_table.py - Symbol table implementation.
"""

import pytest

from dsl_compiler.src.ast.literals import NumberLiteral
from dsl_compiler.src.semantic.symbol_table import (
    SemanticError,
    Symbol,
    SymbolTable,
    SymbolType,
)
from dsl_compiler.src.semantic.type_system import BundleValue, IntValue


class TestSymbolType:
    """Tests for SymbolType enum."""

    def test_symbol_types_exist(self):
        """Test all expected symbol types exist."""
        assert SymbolType.VARIABLE.value == "variable"
        assert SymbolType.MEMORY.value == "memory"
        assert SymbolType.FUNCTION.value == "function"
        assert SymbolType.PARAMETER.value == "parameter"
        assert SymbolType.ENTITY.value == "entity"
        assert SymbolType.MODULE.value == "module"


class TestSemanticError:
    """Tests for SemanticError exception."""

    def test_semantic_error_without_node(self):
        """Test SemanticError without node."""
        error = SemanticError("test error message")
        assert error.message == "test error message"
        assert error.node is None
        assert str(error) == "test error message"

    def test_semantic_error_with_node(self):
        """Test SemanticError with node having location."""
        node = NumberLiteral(value=42, line=10, column=5)
        error = SemanticError("test error", node)
        assert error.message == "test error"
        assert error.node is node
        assert " at 10:5" in str(error)

    def test_semantic_error_with_node_no_location(self):
        """Test SemanticError with node having no location (line 0)."""
        node = NumberLiteral(value=42, line=0, column=0)
        error = SemanticError("test error", node)
        # Line 0 means no location info
        assert str(error) == "test error"


class TestSymbol:
    """Tests for Symbol dataclass."""

    def test_symbol_creation(self):
        """Test creating a Symbol."""
        node = NumberLiteral(value=42, line=1, column=1)
        symbol = Symbol(
            name="x",
            symbol_type=SymbolType.VARIABLE,
            value_type=IntValue(),
            defined_at=node,
        )
        assert symbol.name == "x"
        assert symbol.symbol_type == SymbolType.VARIABLE
        assert symbol.is_mutable is False
        assert symbol.properties is None
        assert symbol.function_def is None
        assert symbol.debug_info == {}

    def test_symbol_with_mutable(self):
        """Test Symbol with is_mutable flag."""
        node = NumberLiteral(value=42, line=1, column=1)
        symbol = Symbol(
            name="mem",
            symbol_type=SymbolType.MEMORY,
            value_type=IntValue(),
            defined_at=node,
            is_mutable=True,
        )
        assert symbol.is_mutable is True

    def test_symbol_with_properties(self):
        """Test Symbol with properties dict."""
        node = NumberLiteral(value=42, line=1, column=1)
        prop_symbol = Symbol(
            name="x",
            symbol_type=SymbolType.VARIABLE,
            value_type=IntValue(),
            defined_at=node,
        )
        symbol = Symbol(
            name="bundle",
            symbol_type=SymbolType.VARIABLE,
            value_type=BundleValue(),
            defined_at=node,
            properties={"x": prop_symbol},
        )
        assert symbol.properties is not None
        assert "x" in symbol.properties


class TestSymbolTable:
    """Tests for SymbolTable class."""

    def test_symbol_table_init(self):
        """Test SymbolTable initialization."""
        table = SymbolTable()
        assert table.parent is None
        assert table.symbols == {}
        assert table.children == []

    def test_symbol_table_with_parent(self):
        """Test SymbolTable with parent."""
        parent = SymbolTable()
        child = SymbolTable(parent=parent)
        assert child.parent is parent

    def test_define_symbol(self):
        """Test defining a symbol."""
        table = SymbolTable()
        node = NumberLiteral(value=42, line=1, column=1)
        symbol = Symbol(
            name="x",
            symbol_type=SymbolType.VARIABLE,
            value_type=IntValue(),
            defined_at=node,
        )
        table.define(symbol)
        assert "x" in table.symbols
        assert table.symbols["x"] is symbol

    def test_define_duplicate_raises_error(self):
        """Test defining duplicate symbol raises SemanticError."""
        table = SymbolTable()
        node = NumberLiteral(value=42, line=1, column=1)
        symbol1 = Symbol(
            name="x",
            symbol_type=SymbolType.VARIABLE,
            value_type=IntValue(),
            defined_at=node,
        )
        symbol2 = Symbol(
            name="x",
            symbol_type=SymbolType.VARIABLE,
            value_type=IntValue(),
            defined_at=node,
        )
        table.define(symbol1)
        with pytest.raises(SemanticError) as exc_info:
            table.define(symbol2)
        assert "already defined" in str(exc_info.value)

    def test_lookup_existing_symbol(self):
        """Test looking up an existing symbol."""
        table = SymbolTable()
        node = NumberLiteral(value=42, line=1, column=1)
        symbol = Symbol(
            name="x",
            symbol_type=SymbolType.VARIABLE,
            value_type=IntValue(),
            defined_at=node,
        )
        table.define(symbol)
        result = table.lookup("x")
        assert result is symbol

    def test_lookup_nonexistent_symbol(self):
        """Test looking up a nonexistent symbol returns None."""
        table = SymbolTable()
        result = table.lookup("undefined")
        assert result is None

    def test_lookup_in_parent_scope(self):
        """Test looking up symbol in parent scope."""
        parent = SymbolTable()
        child = SymbolTable(parent=parent)
        node = NumberLiteral(value=42, line=1, column=1)
        symbol = Symbol(
            name="x",
            symbol_type=SymbolType.VARIABLE,
            value_type=IntValue(),
            defined_at=node,
        )
        parent.define(symbol)
        # Child should find symbol in parent
        result = child.lookup("x")
        assert result is symbol

    def test_lookup_shadows_parent(self):
        """Test child scope shadows parent scope."""
        parent = SymbolTable()
        child = SymbolTable(parent=parent)
        node = NumberLiteral(value=42, line=1, column=1)
        parent_symbol = Symbol(
            name="x",
            symbol_type=SymbolType.VARIABLE,
            value_type=IntValue(),
            defined_at=node,
        )
        child_symbol = Symbol(
            name="x",
            symbol_type=SymbolType.PARAMETER,
            value_type=IntValue(),
            defined_at=node,
        )
        parent.define(parent_symbol)
        child.define(child_symbol)
        # Child lookup should return child's symbol
        result = child.lookup("x")
        assert result is child_symbol
        # Parent lookup should return parent's symbol
        parent_result = parent.lookup("x")
        assert parent_result is parent_symbol

    def test_create_child_scope(self):
        """Test creating child scope."""
        parent = SymbolTable()
        child = parent.create_child_scope()
        assert child.parent is parent
        assert child in parent.children

    def test_create_multiple_child_scopes(self):
        """Test creating multiple child scopes."""
        parent = SymbolTable()
        child1 = parent.create_child_scope()
        child2 = parent.create_child_scope()
        assert len(parent.children) == 2
        assert child1 in parent.children
        assert child2 in parent.children
        assert child1 is not child2
