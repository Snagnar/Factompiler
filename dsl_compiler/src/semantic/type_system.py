from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from dsl_compiler.src.ast.statements import ASTNode, Expr

"""Type system primitives used by semantic analysis."""


@dataclass
class SignalTypeInfo:
    """Information about a signal type."""

    name: str  # e.g. "iron-plate", "signal-A", "__v1"
    is_implicit: bool = False  # True for compiler-allocated virtual signals
    is_virtual: bool = False  # True for Factorio virtual signals


@dataclass
class SignalValue:
    """A single-channel signal value."""

    signal_type: SignalTypeInfo
    count_expr: Optional[Expr] = None  # Expression computing the signal count


@dataclass
class SignalDebugInfo:
    """Metadata describing a logical signal in the source program."""

    identifier: str
    signal_key: Optional[str]
    factorio_signal: Optional[str]
    source_node: ASTNode
    declared_type: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
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

    value: Optional[int] = None  # None for computed values


@dataclass
class FunctionValue:
    """Value type for functions."""

    param_types: List["ValueInfo"] = field(default_factory=list)
    return_type: "ValueInfo" = field(default_factory=lambda: IntValue())


@dataclass
class EntityValue:
    """An entity reference value."""

    entity_id: Optional[str] = None
    prototype: Optional[str] = None


@dataclass
class VoidValue:
    """Represents the absence of a value (void return type)."""

    pass


ValueInfo = Union[SignalValue, IntValue, FunctionValue, EntityValue, VoidValue]


@dataclass
class MemoryInfo:
    """Type information captured for memory declarations."""

    name: str
    symbol: Any
    signal_type: Optional[str] = None
    signal_info: Optional[SignalTypeInfo] = None
    explicit: bool = False
