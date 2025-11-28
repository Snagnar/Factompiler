from __future__ import annotations
from typing import List, Optional
from .base import ASTNode
from .expressions import Expr
from .literals import LValue

"""Statement node definitions for the Factorio Circuit DSL."""


class Program(ASTNode):
    """Root node representing an entire DSL program."""

    def __init__(
        self, statements: List["Statement"], line: int = 0, column: int = 0
    ) -> None:
        super().__init__(line, column)
        self.statements = statements


class TopDecl(ASTNode):
    """Base class for top-level declarations."""

    def __init__(self, line: int = 0, column: int = 0) -> None:
        super().__init__(line, column)


class Statement(ASTNode):
    """Base class for all statements."""

    def __init__(
        self, line: int = 0, column: int = 0, raw_text: Optional[str] = None
    ) -> None:
        super().__init__(line, column, raw_text=raw_text)


class DeclStmt(Statement):
    """type variable = expression;"""

    def __init__(
        self, type_name: str, name: str, value: Expr, line: int = 0, column: int = 0
    ) -> None:
        super().__init__(line, column)
        self.type_name = type_name  # "int", "Signal", "SignalType", "Entity", "Bundle"
        self.name = name
        self.value = value


class AssignStmt(Statement):
    """variable = expression; or variable.property = expression;"""

    def __init__(
        self, target: LValue, value: Expr, line: int = 0, column: int = 0
    ) -> None:
        super().__init__(line, column)
        self.target = target
        self.value = value


class MemDecl(Statement):
    """Memory declaration: Memory name [: "signal-type"];"""

    def __init__(
        self,
        name: str,
        signal_type: Optional[str] = None,
        line: int = 0,
        column: int = 0,
    ) -> None:
        super().__init__(line, column)
        self.name = name
        self.signal_type = signal_type


class ExprStmt(Statement):
    """expression; (standalone expression statement)"""

    def __init__(self, expr: Expr, line: int = 0, column: int = 0) -> None:
        super().__init__(line, column)
        self.expr = expr


class ReturnStmt(Statement):
    """return expression;"""

    def __init__(self, expr: Expr, line: int = 0, column: int = 0) -> None:
        super().__init__(line, column)
        self.expr = expr


class ImportStmt(Statement):
    """import "file" [as alias];"""

    def __init__(
        self, path: str, alias: Optional[str] = None, line: int = 0, column: int = 0
    ) -> None:
        super().__init__(line, column)
        self.path = path
        self.alias = alias


class FuncDecl(TopDecl):
    """func name(params...) { statements... }"""

    def __init__(
        self,
        name: str,
        params: List[str],
        body: List[Statement],
        line: int = 0,
        column: int = 0,
    ) -> None:
        super().__init__(line, column)
        self.name = name
        self.params = params
        self.body = body
