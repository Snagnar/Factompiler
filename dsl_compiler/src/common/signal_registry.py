"""Centralized signal type management."""

from typing import Dict, Optional, Any

try:
    from draftsman.data import signals as signal_data
except ImportError:
    signal_data = None


class SignalTypeRegistry:
    """Centralized registry for signal type mappings.

    This class maintains the mapping between DSL signal types (like __v1, iron-plate)
    and their Factorio representations. It provides a single source of truth for
    signal type information across all compiler stages.
    """

    def __init__(self):
        self._type_map: Dict[str, Any] = {}
        self._implicit_counter = 0

    def register(
        self, signal_key: str, factorio_signal: str, signal_type: str = "virtual"
    ) -> None:
        """Register a signal type mapping.

        Args:
            signal_key: DSL signal identifier (e.g., "__v1", "iron-plate")
            factorio_signal: Factorio signal name (e.g., "signal-A", "iron-plate")
            signal_type: Category (virtual, item, fluid, etc.)
        """
        self._type_map[signal_key] = {"name": factorio_signal, "type": signal_type}

        # Register with draftsman if available
        if signal_data is not None and factorio_signal not in signal_data.raw:
            try:
                signal_data.add_signal(factorio_signal, signal_type)
            except Exception:
                pass  # Already exists or invalid

    def allocate_implicit(self) -> str:
        """Allocate a new implicit virtual signal type.

        Returns:
            The DSL signal key (e.g., "__v1")
        """
        self._implicit_counter += 1
        implicit_key = f"__v{self._implicit_counter}"
        factorio_signal = self._virtual_signal_name(self._implicit_counter)
        self.register(implicit_key, factorio_signal, "virtual")
        return implicit_key

    def resolve(self, signal_key: str) -> Optional[Dict[str, str]]:
        """Get the Factorio signal information for a DSL signal key.

        Args:
            signal_key: DSL signal identifier

        Returns:
            Dict with "name" and "type" keys, or None if not found
        """
        return self._type_map.get(signal_key)

    def resolve_name(self, signal_key: str) -> Optional[str]:
        """Get just the Factorio signal name.

        Args:
            signal_key: DSL signal identifier

        Returns:
            Factorio signal name or None
        """
        mapping = self.resolve(signal_key)
        return mapping["name"] if mapping else None

    def resolve_type(self, signal_key: str) -> Optional[str]:
        """Get just the signal category.

        Args:
            signal_key: DSL signal identifier

        Returns:
            Signal category (virtual, item, fluid) or None
        """
        mapping = self.resolve(signal_key)
        return mapping["type"] if mapping else None

    def get_all_mappings(self) -> Dict[str, Any]:
        """Get a copy of all signal type mappings."""
        return self._type_map.copy()

    def _virtual_signal_name(self, index: int) -> str:
        """Map an implicit index to a unique Factorio virtual signal name."""
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        index -= 1  # convert to zero-based
        name = ""
        while index >= 0:
            name = alphabet[index % 26] + name
            index = index // 26 - 1
        return f"signal-{name}"

    def __len__(self) -> int:
        """Get the number of registered signal types."""
        return len(self._type_map)
