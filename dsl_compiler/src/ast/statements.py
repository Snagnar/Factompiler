from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .base import ASTNode
from .expressions import Expr
from .literals import LValue

"""Statement node definitions for the Facto."""


@dataclass
class TypedParam:
    """A typed function parameter."""

    type_name: str  # "int", "Signal", "Entity"
    name: str
    line: int = 0
    column: int = 0
    source_file: str | None = None


class Program(ASTNode):
    """Root node representing an entire DSL program."""

    def __init__(self, statements: list[Statement], line: int = 0, column: int = 0) -> None:
        super().__init__(line, column)
        self.statements = statements


class Statement(ASTNode):
    """Base class for all statements."""

    def __init__(self, line: int = 0, column: int = 0, raw_text: str | None = None) -> None:
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

    def __init__(self, target: LValue, value: Expr, line: int = 0, column: int = 0) -> None:
        super().__init__(line, column)
        self.target = target
        self.value = value


class MemDecl(Statement):
    """Memory declaration: Memory name [: "signal-type"];

    The memory type (standard vs latch) is determined by how it's written to:
    - Standard write: mem.write(value, when=cond) - 2 decider latch
    - Latch write: mem.write(value, set=s, reset=r) - 1 decider latch
    """

    def __init__(
        self,
        name: str,
        signal_type: str | None = None,
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

    def __init__(self, path: str, alias: str | None = None, line: int = 0, column: int = 0) -> None:
        super().__init__(line, column)
        self.path = path
        self.alias = alias


class FuncDecl(Statement):
    """func name(params...) { statements... }"""

    def __init__(
        self,
        name: str,
        params: list[TypedParam],
        body: list[Statement],
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

    Range bounds (start, stop, step) can be:
    - int: A literal integer value
    - str: A variable name that references a compile-time constant int

    These must be resolved before calling get_iteration_values().
    """

    def __init__(
        self,
        iterator_name: str,
        start: int | str | None,
        stop: int | str | None,
        step: int | str | None,
        values: list[int] | None,
        body: list[Statement],
        line: int = 0,
        column: int = 0,
    ) -> None:
        super().__init__(line, column)
        self.iterator_name = iterator_name
        self.start = start  # Start value for range (inclusive), None for list, can be var name
        self.stop = stop  # Stop value for range (exclusive), None for list, can be var name
        self.step = step  # Step value for range, None for list, can be var name
        self.values = values  # List of values for list iterator, None for range
        self.body = body

    def get_iteration_values(
        self, constant_resolver: Callable[[str], int] | None = None
    ) -> list[int]:
        """Returns the sequence of values the iterator takes.

        Args:
            constant_resolver: Optional callable that resolves variable names to their
                              compile-time constant int values. Required if any bounds
                              are variable references (strings).
        """
        if self.values is not None:
            return list(self.values)

        # Resolve bounds
        def resolve(value: int | str) -> int:
            if isinstance(value, int):
                return value
            if constant_resolver is None:
                raise ValueError(
                    f"Variable '{value}' in for loop range requires a constant resolver"
                )
            return constant_resolver(value)

        start = resolve(self.start)
        stop = resolve(self.stop)
        step = resolve(self.step) if self.step is not None else None

        # Range iteration
        result = []
        if step is None:
            step = 1 if start < stop else -1
        if step > 0:
            i = start
            while i < stop:
                result.append(i)
                i += step
        elif step < 0:
            i = start
            while i > stop:
                result.append(i)
                i += step
        return result
