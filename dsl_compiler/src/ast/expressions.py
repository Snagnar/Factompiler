from __future__ import annotations

from typing import Any

from .base import ASTNode

"""Expression node definitions for the Facto."""


class Expr(ASTNode):
    """Base class for all expressions."""

    def __init__(self, line: int = 0, column: int = 0, raw_text: str | None = None) -> None:
        super().__init__(line, column, raw_text=raw_text)


class BinaryOp(Expr):
    """Binary operation: left op right"""

    def __init__(self, op: str, left: Expr, right: Expr, line: int = 0, column: int = 0) -> None:
        super().__init__(line, column)
        self.op = op  # +, -, *, /, %, ==, !=, <, <=, >, >=, &&, ||
        self.left = left
        self.right = right


class UnaryOp(Expr):
    """Unary operation: op expr"""

    def __init__(self, op: str, expr: Expr, line: int = 0, column: int = 0) -> None:
        super().__init__(line, column)
        self.op = op  # +, -, !
        self.expr = expr


class CallExpr(Expr):
    """Function call: func(args...)"""

    def __init__(self, name: str, args: list[Expr], line: int = 0, column: int = 0) -> None:
        super().__init__(line, column)
        self.name = name
        self.args = args
        self.metadata: dict[str, Any] = {}


class ReadExpr(Expr):
    """read(memory_name)"""

    def __init__(self, memory_name: str, line: int = 0, column: int = 0) -> None:
        super().__init__(line, column)
        self.memory_name = memory_name


class WriteExpr(Expr):
    """Memory write expression.

    Supports two modes:
    - Standard write: write(value, when=enable) - write-gated latch
    - Latch write: write(value, set=s, reset=r) - single combinator latch

    For latch mode, set_priority determines the conflict resolution:
    - True (SR latch): set wins when both conditions true
    - False (RS latch): reset wins when both conditions true
    """

    # Added by semantic analysis
    enable_type: Any  # ValueInfo - type of enable signal
    value_type: Any  # ValueInfo - type of value being written

    def __init__(
        self,
        value: Expr,
        memory_name: str,
        when: Expr | None = None,
        set_signal: Expr | None = None,
        reset_signal: Expr | None = None,
        set_priority: bool = True,  # True = SR (set priority), False = RS (reset priority)
        *,
        line: int = 0,
        column: int = 0,
    ) -> None:
        super().__init__(line, column)
        self.value = value
        self.memory_name = memory_name
        self.when = when
        self.set_signal = set_signal
        self.reset_signal = reset_signal
        self.set_priority = set_priority

    def is_latch_write(self) -> bool:
        """Returns True if this is a latch write (set/reset mode)."""
        return self.set_signal is not None and self.reset_signal is not None


class ProjectionExpr(Expr):
    """expr | "type" - project signal/bundle to specific channel

    target_type can be:
    - A string (e.g., "iron-plate", "signal-A")
    - A SignalTypeAccess (e.g., a.type - resolved at compile time)
    """

    def __init__(
        self, expr: Expr, target_type: str | SignalTypeAccess, line: int = 0, column: int = 0
    ) -> None:
        super().__init__(line, column)
        self.expr = expr
        self.target_type = target_type  # the type literal after |


class SignalLiteral(Expr):
    """Signal literal: ("type", value) or just value

    signal_type can be:
    - None for implicit type (compiler allocates virtual signal)
    - A string (e.g., "iron-plate", "signal-A")
    - A SignalTypeAccess (e.g., a.type - resolved at compile time)
    """

    def __init__(
        self,
        value: Expr,
        signal_type: str | SignalTypeAccess | None = None,
        line: int = 0,
        column: int = 0,
        raw_text: str | None = None,
        wire_color: str | None = None,
    ) -> None:
        super().__init__(line, column, raw_text=raw_text)
        self.value = value
        self.signal_type = signal_type  # None for implicit type, string for explicit
        self.wire_color = wire_color  # "red" | "green" | None for automatic


class IdentifierExpr(Expr):
    """Variable reference in expression context."""

    def __init__(
        self, name: str, line: int = 0, column: int = 0, raw_text: str | None = None
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
        raw_text: str | None = None,
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
        condition: Expr,
        output_value: Expr,
        line: int = 0,
        column: int = 0,
        raw_text: str | None = None,
    ) -> None:
        super().__init__(line, column, raw_text=raw_text)
        self.condition = condition  # Must be a comparison (BinaryOp with COMP_OP)
        self.output_value = output_value  # Value to output when true


class BundleLiteral(Expr):
    """Bundle literal: { signal1, signal2, ... }

    A bundle is an unordered collection of signals that can be operated on
    as a unit. Elements can be signal literals, signal variables, or other bundles.
    """

    def __init__(
        self,
        elements: list[Expr],
        line: int = 0,
        column: int = 0,
        raw_text: str | None = None,
    ) -> None:
        super().__init__(line, column, raw_text=raw_text)
        self.elements = elements  # List of signal/bundle expressions


class BundleSelectExpr(Expr):
    """Bundle element selection: bundle["signal-type"]

    Extracts a single signal from a bundle by its signal type.
    Returns a Signal that can be used in any expression.
    """

    def __init__(
        self,
        bundle: Expr,
        signal_type: str,
        line: int = 0,
        column: int = 0,
        raw_text: str | None = None,
    ) -> None:
        super().__init__(line, column, raw_text=raw_text)
        self.bundle = bundle  # The bundle expression
        self.signal_type = signal_type  # The signal type to select


class BundleAnyExpr(Expr):
    """any(bundle) for 'anything' comparisons.

    Used in comparisons like: any(bundle) > value
    Maps to Factorio's 'signal-anything' in decider combinators.
    """

    def __init__(
        self,
        bundle: Expr,
        line: int = 0,
        column: int = 0,
        raw_text: str | None = None,
    ) -> None:
        super().__init__(line, column, raw_text=raw_text)
        self.bundle = bundle


class BundleAllExpr(Expr):
    """all(bundle) for 'everything' comparisons.

    Used in comparisons like: all(bundle) > value
    Maps to Factorio's 'signal-everything' in decider combinators.
    """

    def __init__(
        self,
        bundle: Expr,
        line: int = 0,
        column: int = 0,
        raw_text: str | None = None,
    ) -> None:
        super().__init__(line, column, raw_text=raw_text)
        self.bundle = bundle


class SignalTypeAccess(Expr):
    """Access to a signal's type: signal_var.type

    Used in projections and signal literals to dynamically reference
    the type of another signal variable. The type is resolved at compile time.

    Example:
        Signal a = ("iron-plate", 60);
        Signal b = 50 | a.type;  # b is projected to iron-plate
    """

    def __init__(
        self,
        object_name: str,
        property_name: str,
        line: int = 0,
        column: int = 0,
        raw_text: str | None = None,
    ) -> None:
        super().__init__(line, column, raw_text=raw_text)
        self.object_name = object_name
        self.property_name = property_name


class EntityOutputExpr(Expr):
    """Access to entity's circuit output: entity.output

    Represents reading the circuit network signals that an entity outputs.
    For chests: all items in the chest as a bundle
    For tanks: fluid amount as a signal
    For train stops with read_from_train: train contents as a bundle

    Example:
        Entity chest = place("steel-chest", 0, 0, {read_contents: 1});
        Bundle contents = chest.output;  # All items in chest as bundle
    """

    def __init__(
        self,
        entity_name: str,
        line: int = 0,
        column: int = 0,
        raw_text: str | None = None,
    ) -> None:
        super().__init__(line, column, raw_text=raw_text)
        self.entity_name = entity_name
