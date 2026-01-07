"""Compile-time constant folding helpers for the lowering pass."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dsl_compiler.src.ast.expressions import (
    BinaryOp,
    IdentifierExpr,
    SignalLiteral,
    UnaryOp,
)
from dsl_compiler.src.ast.literals import NumberLiteral
from dsl_compiler.src.ast.statements import (
    ASTNode,
    Expr,
)

# Type alias for symbol resolver callback
SymbolResolver = Callable[[str], int | None]


class ConstantFolder:
    """Utility helpers for evaluating expressions during lowering."""

    @staticmethod
    def fold_binary_operation(
        op: str,
        left: int,
        right: int,
        node: ASTNode,
        diagnostics: Any = None,
    ) -> int | None:
        """Evaluate a binary integer operation at compile time."""

        # Arithmetic
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            if right == 0:
                if diagnostics is not None:
                    diagnostics.warning(
                        "Division by zero in constant expression",
                        stage="lowering",
                        node=node,
                    )
                return 0
            return left // right
        if op == "%":
            if right == 0:
                if diagnostics is not None:
                    diagnostics.warning(
                        "Modulo by zero in constant expression",
                        stage="lowering",
                        node=node,
                    )
                return 0
            return left % right

        # Power
        if op == "**":
            if right < 0:
                if diagnostics is not None:
                    diagnostics.warning(
                        "Negative exponent in constant expression",
                        stage="lowering",
                        node=node,
                    )
                return 0
            try:
                return left**right
            except (OverflowError, ValueError):
                if diagnostics is not None:
                    diagnostics.warning(
                        "Power operation overflow in constant expression",
                        stage="lowering",
                        node=node,
                    )
                return 0

        # Shifts - Factorio uses 32-bit signed integers
        # Shift amounts >= 32 or < 0 result in 0
        if op == "<<":
            if right < 0 or right >= 32:
                return 0
            return (left << right) & 0xFFFFFFFF  # Mask to 32-bit
        if op == ">>":
            if right < 0 or right >= 32:
                return 0
            # Arithmetic right shift for signed integers
            return left >> right

        # Bitwise
        if op == "AND":
            return left & right
        if op == "OR":
            return left | right
        if op == "XOR":
            return left ^ right

        # Comparisons - MUST be folded when both operands are constants
        # because Factorio decider combinators cannot compare two constants.
        # At least one operand must be a signal on the input wire.
        if op == "==":
            return 1 if left == right else 0
        if op == "!=":
            return 1 if left != right else 0
        if op == "<":
            return 1 if left < right else 0
        if op == "<=":
            return 1 if left <= right else 0
        if op == ">":
            return 1 if left > right else 0
        if op == ">=":
            return 1 if left >= right else 0

        # Logical operators - also must be folded for constant operands
        if op == "&&":
            return 1 if (left != 0 and right != 0) else 0
        if op == "||":
            return 1 if (left != 0 or right != 0) else 0

        return None

    @classmethod
    def extract_constant_int(
        cls,
        expr: Expr,
        diagnostics: Any = None,
        symbol_resolver: SymbolResolver | None = None,
    ) -> int | None:
        """Attempt to evaluate an expression to an integer constant.

        Recursively evaluates constant expressions at compile time, including:
        - NumberLiteral: immediate integer values
        - SignalLiteral with NumberLiteral value
        - IdentifierExpr: resolved via symbol_resolver if provided
        - UnaryOp: +/- on constant expressions
        - BinaryOp: arithmetic/bitwise/comparison on constant expressions

        Args:
            expr: The expression to evaluate
            diagnostics: Optional diagnostics for reporting warnings
            symbol_resolver: Optional callback to resolve identifier names to
                constant int values. If provided and an IdentifierExpr is
                encountered, this callback will be called with the identifier
                name and should return the constant value or None.

        Returns None if any part of the expression is not a compile-time constant.
        """

        if isinstance(expr, NumberLiteral):
            return expr.value

        if isinstance(expr, IdentifierExpr):
            if symbol_resolver is not None:
                return symbol_resolver(expr.name)
            return None

        if isinstance(expr, SignalLiteral):
            inner = expr.value
            if isinstance(inner, NumberLiteral):
                return inner.value
            # Try to resolve inner expression (e.g., identifier reference)
            return cls.extract_constant_int(inner, diagnostics, symbol_resolver)

        if isinstance(expr, UnaryOp):
            inner_val = cls.extract_constant_int(expr.expr, diagnostics, symbol_resolver)
            if inner_val is None:
                return None
            if expr.op == "+":
                return inner_val
            if expr.op == "-":
                return -inner_val
            return None

        if isinstance(expr, BinaryOp):
            left_const = cls.extract_constant_int(expr.left, diagnostics, symbol_resolver)
            right_const = cls.extract_constant_int(expr.right, diagnostics, symbol_resolver)
            if left_const is None or right_const is None:
                return None
            return cls.fold_binary_operation(
                expr.op, left_const, right_const, expr, diagnostics
            )

        return None
