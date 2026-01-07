from __future__ import annotations

from abc import ABC
from typing import Any

"""Base classes and utilities for AST traversal."""


class ASTNode(ABC):
    """Base class for all AST nodes."""

    def __init__(
        self,
        line: int = 0,
        column: int = 0,
        source_file: str | None = None,
        raw_text: str | None = None,
    ) -> None:
        self.line = line
        self.column = column
        self.source_file = source_file
        self.raw_text = raw_text


class ASTVisitor(ABC):
    """Base class for AST traversal visitors."""

    def visit(self, node: ASTNode) -> Any:
        """Visit a node and return result."""
        method_name = f"visit_{type(node).__name__}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(node)
        return self.generic_visit(node)

    def generic_visit(self, node: ASTNode) -> Any:
        """Default visitor for unhandled node types."""
        pass
