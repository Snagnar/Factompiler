from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from .base import ASTNode
from .expressions import Expr
from .literals import LValue

"""Statement node definitions for the Factorio Circuit DSL."""


@dataclass
class TypedParam:
    """A typed function parameter."""

    type_name: str  # "int", "Signal", "Entity"
    name: str
    line: int = 0
    column: int = 0
    source_file: Optional[str] = None


class Program(ASTNode):
    """Root node representing an entire DSL program."""

    def __init__(
        self, statements: List["Statement"], line: int = 0, column: int = 0
    ) -> None:
        super().__init__(line, column)
        self.statements = statements


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


class FuncDecl(Statement):
    """func name(params...) { statements... }"""

    def __init__(
        self,
        name: str,
        params: List[TypedParam],
        body: List[Statement],
        line: int = 0,
        column: int = 0,
    ) -> None:
        super().__init__(line, column)
        self.name = name
        self.params = params
        self.body = body


class ForStmt(Statement):
    """for variable in iterator { body }
    
    Represents a compile-time for loop that is fully unrolled.
    """

    def __init__(
        self,
        iterator_name: str,
        start: Optional[int],
        stop: Optional[int],
        step: Optional[int],
        values: Optional[List[int]],
        body: List["Statement"],
        line: int = 0,
        column: int = 0,
    ) -> None:
        super().__init__(line, column)
        self.iterator_name = iterator_name
        self.start = start  # Start value for range (inclusive), None for list
        self.stop = stop    # Stop value for range (exclusive), None for list
        self.step = step    # Step value for range, None for list
        self.values = values  # List of values for list iterator, None for range
        self.body = body

    def get_iteration_values(self) -> List[int]:
        """Returns the sequence of values the iterator takes."""
        if self.values is not None:
            return list(self.values)
        # Range iteration
        result = []
        if self.step is None:
            step = 1 if self.start < self.stop else -1
        else:
            step = self.step
        if step > 0:
            i = self.start
            while i < self.stop:
                result.append(i)
                i += step
        elif step < 0:
            i = self.start
            while i > self.stop:
                result.append(i)
                i += step
        return result
