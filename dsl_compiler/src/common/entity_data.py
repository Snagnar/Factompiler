"""Entity data extraction from draftsman."""

from typing import Tuple
import math

try:
    from draftsman.data import entities as entity_data
except ImportError:
    entity_data = None


class EntityDataHelper:
    """Helper to extract entity information from draftsman data."""

    @staticmethod
    def get_footprint(prototype: str) -> Tuple[int, int]:
        """Get entity footprint size dynamically from draftsman.

        Args:
            prototype: Entity prototype name (e.g., "small-lamp")

        Returns:
            (width, height) in tiles
        """
        if entity_data is None:
            return (1, 1)  # Fallback

        try:
            entity_info = entity_data.raw.get(prototype, {})

            # Try collision_box first (most accurate)
            collision_box = entity_info.get("collision_box")
            if collision_box:
                # collision_box is [[x1, y1], [x2, y2]]
                width = max(1, math.ceil(collision_box[1][0] - collision_box[0][0]))
                height = max(1, math.ceil(collision_box[1][1] - collision_box[0][1]))
                return (width, height)

            # Fallback to tile_width/tile_height
            width = entity_info.get("tile_width", 1)
            height = entity_info.get("tile_height", 1)
            return (max(1, width), max(1, height))

        except Exception:
            return (1, 1)  # Fallback for any errors

    @staticmethod
    def get_alignment(prototype: str) -> int:
        """Get entity alignment requirement.

        Args:
            prototype: Entity prototype name

        Returns:
            Alignment grid size (1 or 2)
        """
        footprint = EntityDataHelper.get_footprint(prototype)
        # Entities 2x2 or larger need alignment=2
        return 2 if max(footprint) >= 2 else 1


# Export for convenience
get_entity_footprint = EntityDataHelper.get_footprint
get_entity_alignment = EntityDataHelper.get_alignment
