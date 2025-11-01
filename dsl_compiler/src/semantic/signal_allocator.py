"""Implicit signal allocation for semantic analysis."""

from typing import Dict, Optional

from dsl_compiler.src.common import SignalTypeRegistry
from .type_system import SignalTypeInfo


class SignalAllocator:
    """Manages allocation of implicit virtual signal types."""

    def __init__(self, signal_registry: Optional[SignalTypeRegistry] = None) -> None:
        if signal_registry is None:
            self.signal_registry = SignalTypeRegistry()
        else:
            self.signal_registry = signal_registry

    @property
    def signal_type_map(self) -> Dict[str, str]:
        """Backward compatibility: get signal type map from registry as strings."""
        # Old code expected Dict[str, str] mapping signal keys to factorio signal names
        # New registry stores Dict[str, Dict] with "name" and "type" keys
        # Extract just the "name" for backward compatibility
        result = {}
        for key, value in self.signal_registry.get_all_mappings().items():
            if isinstance(value, dict):
                result[key] = value.get("name", key)
            else:
                result[key] = value
        return result

    def allocate_implicit_type(self) -> SignalTypeInfo:
        """Allocate and record a new implicit virtual signal."""
        implicit_name = self.signal_registry.allocate_implicit()
        return SignalTypeInfo(name=implicit_name, is_implicit=True, is_virtual=True)
