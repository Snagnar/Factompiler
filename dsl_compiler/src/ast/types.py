"""Type information classes used during semantic analysis."""


class TypeInfo:
    """Type information attached to expressions during semantic analysis."""

    pass


class SignalType(TypeInfo):
    """Single-channel signal type: (type_name, count)"""

    def __init__(self, type_name: str, is_implicit: bool = False) -> None:
        self.type_name = type_name  # e.g. "iron-plate", "signal-A", "__v1"
        self.is_implicit = is_implicit  # True for compiler-allocated virtual signals


class IntType(TypeInfo):
    """Plain integer type (not a signal)"""

    pass
