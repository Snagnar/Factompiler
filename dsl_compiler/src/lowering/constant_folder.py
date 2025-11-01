"""Compile-time constant folding helpers for the lowering pass."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from dsl_compiler.src.ast import (
    ASTNode,
    BinaryOp,
    Expr,
    NumberLiteral,
    SignalLiteral,
    UnaryOp,
)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from dsl_compiler.src.common import ProgramDiagnostics


class ConstantFolder:
    """Utility helpers for evaluating expressions during lowering."""

    @staticmethod
    def fold_binary_operation(
        op: str,
        left: int,
        right: int,
        node: ASTNode,
        diagnostics: Optional["ProgramDiagnostics"] = None,
    ) -> Optional[int]:
        """Evaluate a binary integer operation at compile time."""

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
        if op == "&&":
            return 1 if (left != 0 and right != 0) else 0
        if op == "||":
            return 1 if (left != 0 or right != 0) else 0

        return None

    @classmethod
    def extract_constant_int(
        cls,
        expr: Expr,
        diagnostics: Optional["ProgramDiagnostics"] = None,
    ) -> Optional[int]:
        """Attempt to evaluate an expression to an integer constant."""

        if isinstance(expr, NumberLiteral):
            return expr.value

        if isinstance(expr, SignalLiteral):
            inner = expr.value
            if isinstance(inner, NumberLiteral):
                return inner.value

        if isinstance(expr, UnaryOp):
            inner_val = cls.extract_constant_int(expr.expr, diagnostics)
            if inner_val is None:
                return None
            if expr.op == "+":
                return inner_val
            if expr.op == "-":
                return -inner_val
            return None

        if isinstance(expr, BinaryOp):
            left_const = cls.extract_constant_int(expr.left, diagnostics)
            right_const = cls.extract_constant_int(expr.right, diagnostics)
            if left_const is None or right_const is None:
                return None
            return cls.fold_binary_operation(
                expr.op, left_const, right_const, expr, diagnostics
            )

        return None
