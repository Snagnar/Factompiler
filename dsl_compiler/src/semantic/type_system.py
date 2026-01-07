from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dsl_compiler.src.ast.statements import ASTNode, Expr
from dsl_compiler.src.common.signal_registry import SignalTypeInfo

"""Type system primitives used by semantic analysis."""


@dataclass
class SignalValue:
    """A single-channel signal value."""

    signal_type: SignalTypeInfo
    count_expr: Expr | None = None  # Expression computing the signal count


@dataclass
class SignalDebugInfo:
    """Metadata describing a logical signal in the source program."""

    identifier: str
    signal_key: str | None
    factorio_signal: str | None
    source_node: ASTNode
    declared_type: str | None = None
    location: str | None = None
    category: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.identifier,
            "signal_key": self.signal_key,
            "factorio_signal": self.factorio_signal,
            "declared_type": self.declared_type,
            "location": self.location,
            "category": self.category,
            "source_ast": self.source_node,
        }


@dataclass
class IntValue:
    """A plain integer value."""

    value: int | None = None  # None for computed values


@dataclass
class FunctionValue:
    """Value type for functions."""

    param_types: list[ValueInfo] = field(default_factory=list)
    return_type: ValueInfo = field(default_factory=lambda: IntValue())


@dataclass
class EntityValue:
    """An entity reference value."""

    entity_id: str | None = None
    prototype: str | None = None


@dataclass
class VoidValue:
    """Represents the absence of a value (void return type)."""

    pass


@dataclass
class BundleValue:
    """A bundle of signals that can be operated on as a unit.

    Bundles contain zero or more signals, each with a distinct signal type.
    When used in arithmetic operations, bundles compile to Factorio's "each"
    signal combinators. When used in comparisons with any/all, they use
    "anything" or "everything" virtual signals.
    """

    signal_types: set[str] = field(default_factory=set)  # Set of signal type names


@dataclass
class DynamicBundleValue(BundleValue):
    """Bundle with runtime-determined signal types.

    Used for entity outputs where the actual signals depend on runtime state
    (e.g., chest contents, train cargo). Signal types are unknown at compile
    time.

    Example:
        Entity chest = place("steel-chest", 0, 0, {read_contents: 1});
        Bundle contents = chest.output;  # DynamicBundleValue - types unknown
    """

    source_entity_id: str = ""
    is_dynamic: bool = True


ValueInfo = (
    SignalValue
    | IntValue
    | FunctionValue
    | EntityValue
    | VoidValue
    | BundleValue
    | DynamicBundleValue
)


@dataclass
class MemoryInfo:
    """Type information captured for memory declarations.

    The memory type (standard vs latch) is determined by how it's written to,
    not by the declaration. This allows flexible usage of memory cells.
    """

    name: str
    symbol: Any
    signal_type: str | None = None
    signal_info: SignalTypeInfo | None = None
    explicit: bool = False


def get_signal_type_name(value_type: ValueInfo) -> str | None:
    """Extract signal type name from ValueInfo if it's a SignalValue.

    Args:
        value_type: A ValueInfo instance

    Returns:
        The signal type name if value_type is a SignalValue with a signal_type,
        otherwise None.
    """
    if isinstance(value_type, SignalValue) and value_type.signal_type:
        return value_type.signal_type.name
    return None
