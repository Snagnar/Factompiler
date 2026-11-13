"""Factorio's integer arithmetic, for compile-time constant folding.

Factorio circuit values are signed 32-bit integers. Arithmetic wraps on
overflow, ``/`` truncates toward zero and ``%`` takes the sign of the
dividend (C++ semantics). Python's ``//`` and ``%`` floor instead, so
``-7 // 2 == -4`` and ``-7 % 3 == 2``, whereas in game the results are
``-3`` and ``-1``. Constant folding must match the game, otherwise a
folded constant differs from what the same expression computes at runtime.
"""

from __future__ import annotations

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


def in_int32(value: int) -> bool:
    return INT32_MIN <= value <= INT32_MAX


def wrap32(value: int) -> int:
    """Wrap an arbitrary integer to signed 32-bit (two's complement)."""
    return ((value + 2**31) % 2**32) - 2**31


def trunc_div(left: int, right: int) -> int:
    """Integer division rounding toward zero (right must be non-zero)."""
    q = abs(left) // abs(right)
    return q if (left >= 0) == (right >= 0) else -q


def trunc_mod(left: int, right: int) -> int:
    """Remainder with the sign of the dividend (right must be non-zero)."""
    return left - right * trunc_div(left, right)


def fold_int_op(op: str, left: int, right: int) -> int | None:
    """Fold an arithmetic/bitwise operator like Factorio does.

    Returns ``None`` for operators this helper does not handle. Division
    and modulo by zero yield 0, matching the arithmetic combinator.
    Results are wrapped to 32 bits.
    """
    if op == "+":
        raw = left + right
    elif op == "-":
        raw = left - right
    elif op == "*":
        raw = left * right
    elif op == "/":
        raw = 0 if right == 0 else trunc_div(left, right)
    elif op == "%":
        raw = 0 if right == 0 else trunc_mod(left, right)
    elif op in ("**", "^"):
        if right < 0:
            return 0
        raw = pow(left, right, 2**32)  # modular: no huge intermediates
    elif op == "<<":
        if right < 0 or right >= 32:
            return 0
        raw = left << right
    elif op == ">>":
        if right < 0 or right >= 32:
            return 0
        raw = left >> right  # arithmetic shift, like C++ on int32
    elif op in ("AND", "&"):
        raw = left & right
    elif op in ("OR", "|"):
        raw = left | right
    elif op == "XOR":
        raw = left ^ right
    else:
        return None
    return wrap32(raw)


def overflows(op: str, left: int, right: int) -> bool:
    """True if the mathematically exact result does not fit in 32 bits."""
    if op == "+":
        return not in_int32(left + right)
    if op == "-":
        return not in_int32(left - right)
    if op == "*":
        return not in_int32(left * right)
    if op in ("**", "^") and right >= 0:
        if abs(left) <= 1 or right <= 31:
            return not in_int32(left**right)
        return True
    return False
