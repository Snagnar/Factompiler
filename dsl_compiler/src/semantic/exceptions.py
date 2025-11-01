from typing import Optional
from dsl_compiler.src.ast import ASTNode

"""Semantic analysis exceptions."""


class SemanticError(Exception):
    """Exception raised for semantic analysis errors."""

    def __init__(self, message: str, node: Optional[ASTNode] = None) -> None:
        self.message = message
        self.node = node
        location = f" at {node.line}:{node.column}" if node and node.line > 0 else ""
        super().__init__(f"{message}{location}")
