"""Entity data extraction from draftsman."""

import difflib
import math
import warnings
from functools import lru_cache

from draftsman.data import entities as entity_data
from draftsman.entity import new_entity


class EntityDataHelper:
    """Helper to extract entity information from draftsman data."""

    @staticmethod
    @lru_cache(maxsize=128)
    def is_dual_circuit_connectable(prototype: str) -> bool:
        """Check if an entity type has separate input/output circuit connectors.

        Entities like arithmetic-combinator, decider-combinator, and selector-combinator
        have dual circuit connectors (input side and output side). When connecting wires
        to these entities, you must specify which side (input/output) the wire connects to.

        Args:
            prototype: Entity prototype name (e.g., "selector-combinator")

        Returns:
            True if the entity has dual circuit connectors, False otherwise
        """
        try:
            entity = new_entity(prototype)
            return getattr(entity, "dual_circuit_connectable", False)
        except Exception:
            return False

    @staticmethod
    def get_footprint(prototype: str) -> tuple[int, int]:
        """Get entity footprint size dynamically from draftsman.

        Args:
            prototype: Entity prototype name (e.g., "small-lamp")

        Returns:
            (width, height) in tiles
        """
        try:
            entity_info = entity_data.raw.get(prototype, {})

            width = entity_info.get("tile_width")
            height = entity_info.get("tile_height")

            if width is not None and height is not None:
                return (max(1, int(width)), max(1, int(height)))

            collision_box = entity_info.get("collision_box")
            if collision_box:
                width = max(1, math.ceil(collision_box[1][0] - collision_box[0][0]))
                height = max(1, math.ceil(collision_box[1][1] - collision_box[0][1]))
                return (width, height)

            return (1, 1)

        except Exception:
            return (1, 1)

    @staticmethod
    def get_alignment(prototype: str) -> int:
        """Get entity alignment requirement.

        Args:
            prototype: Entity prototype name

        Returns:
            Alignment grid size (1 or 2)
        """
        footprint = EntityDataHelper.get_footprint(prototype)
        return 2 if max(footprint) >= 2 else 1


get_entity_footprint = EntityDataHelper.get_footprint
get_entity_alignment = EntityDataHelper.get_alignment
is_dual_circuit_connectable = EntityDataHelper.is_dual_circuit_connectable


# Entity properties with special meaning in Facto (not draftsman attributes).
FACTO_ENTITY_PROPERTIES = frozenset({"enable", "input", "output", "wire"})


def is_known_prototype(prototype: str) -> bool:
    """True if draftsman knows the entity prototype (vanilla + Space Age data)."""
    return prototype in entity_data.raw


def suggest_prototype(prototype: str) -> str | None:
    """Closest known prototype name, for 'did you mean' hints."""
    matches = difflib.get_close_matches(prototype, list(entity_data.raw), n=1, cutoff=0.6)
    return matches[0] if matches else None


@lru_cache(maxsize=256)
def entity_attributes(prototype: str) -> frozenset[str] | None:
    """Public attribute names of the draftsman entity, or None if unknown."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            entity = new_entity(prototype)
    except Exception:
        return None
    return frozenset(a for a in dir(entity) if not a.startswith("_"))


def unknown_property_hint(prototype: str, prop: str) -> str | None:
    """Return a warning text if ``prop`` is not a property of ``prototype``.

    Returns None when the property exists, or when the prototype is unknown
    (that case is reported separately).
    """
    if prop in FACTO_ENTITY_PROPERTIES:
        return None
    attrs = entity_attributes(prototype)
    if attrs is None or prop in attrs:
        return None
    if prop in ("r", "g", "b") and "red_signal" in attrs:
        return None  # lamp colour components (lamp.r = signal)
    candidates = sorted(attrs | FACTO_ENTITY_PROPERTIES)
    close = difflib.get_close_matches(prop, candidates, n=1, cutoff=0.6)
    hint = f" Did you mean '{close[0]}'?" if close else ""
    return f"'{prototype}' has no property '{prop}'; it will be ignored.{hint}"
