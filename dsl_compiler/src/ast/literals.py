"""Literal and L-value node definitions."""

from __future__ import annotations

from typing import Dict, Optional

from .base import ASTNode
from .expressions import Expr


class Literal(Expr):
    """Base class for literal values."""

    def __init__(
        self, line: int = 0, column: int = 0, raw_text: Optional[str] = None
    ) -> None:
        super().__init__(line, column, raw_text=raw_text)


class NumberLiteral(Literal):
    """Numeric literal: 42, -17"""

    def __init__(
        self,
        value: int,
        line: int = 0,
        column: int = 0,
        raw_text: Optional[str] = None,
    ) -> None:
        super().__init__(line, column, raw_text=raw_text)
        self.value = value


class StringLiteral(Literal):
    """String literal: "iron-plate" """

    def __init__(
        self,
        value: str,
        line: int = 0,
        column: int = 0,
        raw_text: Optional[str] = None,
    ) -> None:
        super().__init__(line, column, raw_text=raw_text)
        self.value = value


class DictLiteral(Literal):
    """Dictionary literal: { key: value, ... }"""

    def __init__(
        self,
        entries: Dict[str, "Expr"],
        line: int = 0,
        column: int = 0,
        raw_text: Optional[str] = None,
    ) -> None:
        super().__init__(line, column, raw_text=raw_text)
        self.entries = entries


class LValue(ASTNode):
    """Base class for assignment targets."""

    def __init__(
        self, line: int = 0, column: int = 0, raw_text: Optional[str] = None
    ) -> None:
        super().__init__(line, column, raw_text=raw_text)


class Identifier(LValue):
    """Simple variable reference."""

    def __init__(
        self, name: str, line: int = 0, column: int = 0, raw_text: Optional[str] = None
    ) -> None:
        super().__init__(line, column, raw_text=raw_text)
        self.name = name


class PropertyAccess(LValue):
    """entity.property reference."""

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
