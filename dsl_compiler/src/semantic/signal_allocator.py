"""Implicit signal allocation for semantic analysis."""

from typing import Dict

from .type_system import SignalTypeInfo


class SignalAllocator:
    """Manages allocation of implicit virtual signal types."""

    def __init__(self) -> None:
        self.implicit_type_counter = 0
        self.signal_type_map: Dict[str, str] = {}

    def allocate_implicit_type(self) -> SignalTypeInfo:
        """Allocate and record a new implicit virtual signal."""

        self.implicit_type_counter += 1
        implicit_name = f"__v{self.implicit_type_counter}"
        factorio_signal = self._virtual_signal_name(self.implicit_type_counter)
        self.signal_type_map[implicit_name] = factorio_signal
        return SignalTypeInfo(name=implicit_name, is_implicit=True, is_virtual=True)

    def _virtual_signal_name(self, index: int) -> str:
        """Map an implicit index to a unique Factorio virtual signal name."""

        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        index -= 1  # convert to zero-based
        name = ""
        while index >= 0:
            name = alphabet[index % 26] + name
            index = index // 26 - 1
        return f"signal-{name}"
