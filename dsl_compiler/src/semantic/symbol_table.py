from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from dsl_compiler.src.ast.statements import ASTNode

from .type_system import ValueInfo

"""Symbol table implementation for semantic analysis."""


class SymbolType(Enum):
    """Symbol types in the DSL."""

    VARIABLE = "variable"
    MEMORY = "memory"
    FUNCTION = "function"
    PARAMETER = "parameter"
    ENTITY = "entity"
    MODULE = "module"


class SemanticError(Exception):
    """Exception raised for semantic analysis errors."""

    def __init__(self, message: str, node: ASTNode | None = None) -> None:
        self.message = message
        self.node = node
        location = f" at {node.line}:{node.column}" if node and node.line > 0 else ""
        super().__init__(f"{message}{location}")


@dataclass
class Symbol:
    """Symbol table entry."""

    name: str
    symbol_type: SymbolType  # Type of symbol (variable, memory, function, etc.)
    value_type: ValueInfo
    defined_at: ASTNode
    is_mutable: bool = False
    properties: dict[str, "Symbol"] | None = None
    function_def: ASTNode | None = None
    debug_info: dict[str, object] = field(default_factory=dict)


class SymbolTable:
    """Hierarchical symbol table supporting lexical scoping."""

    def __init__(self, parent: Optional["SymbolTable"] = None) -> None:
        self.parent = parent
        self.symbols: dict[str, Symbol] = {}
        self.children: list[SymbolTable] = []

    def define(self, symbol: Symbol) -> None:
        """Define a symbol in the current scope."""

        if symbol.name in self.symbols:
            raise SemanticError(
                f"Symbol '{symbol.name}' already defined",
                symbol.defined_at,
            )
        self.symbols[symbol.name] = symbol

    def lookup(self, name: str) -> Symbol | None:
        """Look up a symbol by name, searching parent scopes as needed."""

        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def create_child_scope(self) -> "SymbolTable":
        """Create and return a child scope."""

        child = SymbolTable(parent=self)
        self.children.append(child)
        return child
