"""Factorio 32-bit integer semantics used by constant folding."""

import pytest

from dsl_compiler.src.common.int32 import (
    INT32_MAX,
    INT32_MIN,
    fold_int_op,
    overflows,
    trunc_div,
    trunc_mod,
    wrap32,
)


@pytest.mark.parametrize(
    "value,expected",
    [(0, 0), (INT32_MAX, INT32_MAX), (INT32_MAX + 1, INT32_MIN), (-(2**31) - 1, INT32_MAX)],
)
def test_wrap32(value, expected):
    assert wrap32(value) == expected


@pytest.mark.parametrize(
    "a,b,q,r",
    [(7, 2, 3, 1), (-7, 2, -3, -1), (7, -2, -3, 1), (-7, -2, 3, -1), (-7, 3, -2, -1)],
)
def test_truncating_division_and_modulo(a, b, q, r):
    assert trunc_div(a, b) == q
    assert trunc_mod(a, b) == r
    assert q * b + r == a


@pytest.mark.parametrize(
    "op,a,b,expected",
    [
        ("+", INT32_MAX, 1, INT32_MIN),
        ("*", 65536, 65536, 0),
        ("/", -7, 2, -3),
        ("%", -7, 3, -1),
        ("/", 5, 0, 0),
        ("%", 5, 0, 0),
        ("<<", 1, 31, INT32_MIN),
        ("<<", 1, 32, 0),
        (">>", -8, 1, -4),
        ("**", 2, -1, 0),
        ("XOR", 6, 3, 5),
        ("?", 1, 2, None),
    ],
)
def test_fold_int_op(op, a, b, expected):
    assert fold_int_op(op, a, b) == expected


def test_overflows():
    assert overflows("+", INT32_MAX, 1)
    assert not overflows("+", INT32_MAX, 0)
    assert overflows("**", 3, 40)
    assert not overflows("**", 1, 1000)
