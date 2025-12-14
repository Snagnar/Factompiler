from __future__ import annotations
from typing import Any, Dict, List, Optional
from .base import ASTNode

"""Expression node definitions for the Factorio Circuit DSL."""


class Expr(ASTNode):
    """Base class for all expressions."""

    def __init__(
        self, line: int = 0, column: int = 0, raw_text: Optional[str] = None
    ) -> None:
        super().__init__(line, column, raw_text=raw_text)


class BinaryOp(Expr):
    """Binary operation: left op right"""

    def __init__(
        self, op: str, left: "Expr", right: "Expr", line: int = 0, column: int = 0
    ) -> None:
        super().__init__(line, column)
        self.op = op  # +, -, *, /, %, ==, !=, <, <=, >, >=, &&, ||
        self.left = left
        self.right = right


class UnaryOp(Expr):
    """Unary operation: op expr"""

    def __init__(self, op: str, expr: "Expr", line: int = 0, column: int = 0) -> None:
        super().__init__(line, column)
        self.op = op  # +, -, !
        self.expr = expr


class CallExpr(Expr):
    """Function call: func(args...)"""

    def __init__(
        self, name: str, args: List["Expr"], line: int = 0, column: int = 0
    ) -> None:
        super().__init__(line, column)
        self.name = name
        self.args = args
        self.metadata: Dict[str, Any] = {}


class ReadExpr(Expr):
    """read(memory_name)"""

    def __init__(self, memory_name: str, line: int = 0, column: int = 0) -> None:
        super().__init__(line, column)
        self.memory_name = memory_name


class WriteExpr(Expr):
    """write(value, memory_name, when=enable)"""

    def __init__(
        self,
        value: "Expr",
        memory_name: str,
        when: Optional["Expr"] = None,
        *,
        line: int = 0,
        column: int = 0,
    ) -> None:
        super().__init__(line, column)
        self.value = value
        self.memory_name = memory_name
        self.when = when


class ProjectionExpr(Expr):
    """expr | "type" - project signal/bundle to specific channel"""

    def __init__(
        self, expr: "Expr", target_type: str, line: int = 0, column: int = 0
    ) -> None:
        super().__init__(line, column)
        self.expr = expr
        self.target_type = target_type  # the type literal after |


class SignalLiteral(Expr):
    """Signal literal: ("type", value) or just value"""

    def __init__(
        self,
        value: "Expr",
        signal_type: Optional[str] = None,
        line: int = 0,
        column: int = 0,
        raw_text: Optional[str] = None,
    ) -> None:
        super().__init__(line, column, raw_text=raw_text)
        self.value = value
        self.signal_type = signal_type  # None for implicit type, string for explicit


class IdentifierExpr(Expr):
    """Variable reference in expression context."""

    def __init__(
        self, name: str, line: int = 0, column: int = 0, raw_text: Optional[str] = None
    ) -> None:
        super().__init__(line, column, raw_text=raw_text)
        self.name = name


class PropertyAccessExpr(Expr):
    """Property access in expression context: entity.property"""

    def __init__(
        self,
        object_name: str,
        property_name: str,
        line: int = 0,
        column: int = 0,
        raw_text: Optional[str] = None,
    ) -> None:
        super().__init__(line, column, raw_text=raw_text)
        self.object_name = object_name
        self.property_name = property_name


class OutputSpecExpr(Expr):
    """Comparison with output specifier: (condition) : output_value

    When condition is true, outputs output_value instead of constant 1.
    Maps to Factorio decider combinator's "copy count from input" mode.
    """

    def __init__(
        self,
        condition: "Expr",
        output_value: "Expr",
        line: int = 0,
        column: int = 0,
        raw_text: Optional[str] = None,
    ) -> None:
        super().__init__(line, column, raw_text=raw_text)
        self.condition = condition  # Must be a comparison (BinaryOp with COMP_OP)
        self.output_value = output_value  # Value to output when true
