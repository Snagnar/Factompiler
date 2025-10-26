"""Base classes and utilities for AST traversal."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ASTNode(ABC):
    """Base class for all AST nodes."""

    def __init__(
        self,
        line: int = 0,
        column: int = 0,
        source_file: Optional[str] = None,
        raw_text: Optional[str] = None,
    ) -> None:
        self.line = line
        self.column = column
        self.source_file = source_file
        self.raw_text = raw_text


class ASTVisitor(ABC):
    """Base class for AST traversal visitors."""

    @abstractmethod
    def visit(self, node: ASTNode) -> Any:
        """Visit a node and return result."""
        method_name = f"visit_{type(node).__name__}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(node)
        return self.generic_visit(node)

    def generic_visit(self, node: ASTNode) -> Any:
        """Default visitor for unhandled node types."""
        pass


class ASTTransformer(ASTVisitor):
    """Base class for AST transformation visitors."""

    def visit(self, node: ASTNode) -> ASTNode:  # type: ignore[override]
        """Visit and transform a node."""
        method_name = f"visit_{type(node).__name__}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(node)
        return self.generic_visit(node)

    def generic_visit(self, node: ASTNode) -> ASTNode:  # type: ignore[override]
        """Default transformer - returns node unchanged."""
        return node


def ast_to_dict(node: ASTNode) -> Any:
    """Convert AST node to dictionary representation for debugging."""
    if not isinstance(node, ASTNode):
        return node

    result: Dict[str, Any] = {"type": type(node).__name__}
    for field_name, field_value in node.__dict__.items():
        if field_name in ("line", "column"):
            continue
        if isinstance(field_value, list):
            result[field_name] = [ast_to_dict(item) for item in field_value]
        elif isinstance(field_value, ASTNode):
            result[field_name] = ast_to_dict(field_value)
        else:
            result[field_name] = field_value

    return result


def print_ast(node: ASTNode, indent: int = 0) -> None:
    """Pretty-print AST structure for debugging."""
    spaces = "  " * indent
    print(f"{spaces}{type(node).__name__}")

    for field_name, field_value in node.__dict__.items():
        if field_name in ("line", "column"):
            continue
        print(f"{spaces}  {field_name}:", end="")
        if isinstance(field_value, ASTNode):
            print()
            print_ast(field_value, indent + 2)
        elif isinstance(field_value, list):
            print()
            for item in field_value:
                if isinstance(item, ASTNode):
                    print_ast(item, indent + 2)
                else:
                    print(f"{spaces}    {item}")
        else:
            print(f" {field_value}")
