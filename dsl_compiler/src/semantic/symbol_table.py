from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
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

    def __init__(self, message: str, node: Optional[ASTNode] = None) -> None:
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
    properties: Optional[Dict[str, "Symbol"]] = None
    function_def: Optional[ASTNode] = None
    debug_info: Dict[str, object] = field(default_factory=dict)


class SymbolTable:
    """Hierarchical symbol table supporting lexical scoping."""

    def __init__(self, parent: Optional["SymbolTable"] = None) -> None:
        self.parent = parent
        self.symbols: Dict[str, Symbol] = {}
        self.children: List["SymbolTable"] = []

    def define(self, symbol: Symbol) -> None:
        """Define a symbol in the current scope."""

        if symbol.name in self.symbols:
            raise SemanticError(
                f"Symbol '{symbol.name}' already defined",
                symbol.defined_at,
            )
        self.symbols[symbol.name] = symbol

    def lookup(self, name: str) -> Optional[Symbol]:
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
