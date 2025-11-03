from typing import Optional
from dsl_compiler.src.common.signal_registry import SignalTypeRegistry
from .type_system import SignalTypeInfo

"""Implicit signal allocation for semantic analysis."""


class SignalAllocator:
    """Manages allocation of implicit virtual signal types.

    Allocates new implicit virtual signal types (__v1, __v2, ...) during semantic
    analysis. Wraps SignalTypeRegistry for allocation-specific logic.

    See dsl_compiler.src.common.signal_types for the complete architecture overview.
    """

    def __init__(self, signal_registry: Optional[SignalTypeRegistry] = None) -> None:
        if signal_registry is None:
            self.signal_registry = SignalTypeRegistry()
        else:
            self.signal_registry = signal_registry

    def allocate_implicit_type(self) -> SignalTypeInfo:
        """Allocate and record a new implicit virtual signal."""
        implicit_name = self.signal_registry.allocate_implicit()
        return SignalTypeInfo(name=implicit_name, is_implicit=True, is_virtual=True)
