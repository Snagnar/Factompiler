"""Centralized signal type management.

Storage Format: Dict[signal_key, {"name": factorio_name, "type": category}]
Example: {"__v1": {"name": "signal-A", "type": "virtual"}}

This is the ONLY format used. All code must handle dict values.
"""

from typing import Dict, Optional, Any, Tuple

from draftsman.data import signals as signal_data


def is_valid_factorio_signal(signal_name: str) -> Tuple[bool, Optional[str]]:
    """Check if a signal name exists in Factorio's signal database.

    Args:
        signal_name: The signal name to validate (e.g., "iron-plate", "signal-A")

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    if not signal_name:
        return False, "Signal name cannot be empty"

    # Implicit compiler-generated signals are always valid (they get resolved later)
    if signal_name.startswith("__v"):
        return True, None

    # Check against draftsman's signal database
    if signal_data is not None:
        # Check in raw signal data (includes items, fluids, virtual signals)
        if signal_name in signal_data.raw:
            return True, None
        # Check in type_of mapping
        if signal_name in signal_data.type_of:
            return True, None

    # If we get here, the signal is not in draftsman's database
    # Provide a helpful error message
    return False, (
        f"Unknown signal '{signal_name}'. "
        "Signal must be a valid Factorio item, fluid, or virtual signal name "
        "(e.g., 'iron-plate', 'water', 'signal-A')."
    )


class SignalTypeRegistry:
    """Centralized registry for signal type mappings.

    This class maintains the mapping between DSL signal types (like __v1, iron-plate)
    and their Factorio representations. It provides a single source of truth for
    signal type information across all compiler stages.

    See dsl_compiler.src.common.signal_types for the complete architecture overview.
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

        Note: The implicit signal is NOT immediately mapped to a Factorio signal.
        This allows the layout phase to determine the actual Factorio signal based
        on whether the signal needs materialization.
        """
        self._implicit_counter += 1
        implicit_key = f"__v{self._implicit_counter}"
        # Don't register Factorio mapping yet - let layout phase decide
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
