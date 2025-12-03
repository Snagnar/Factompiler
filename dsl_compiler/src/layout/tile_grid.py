"""Simple tile occupancy tracking for entity placement."""

from typing import Set, Tuple, Dict
from .layout_plan import EntityPlacement


class TileGrid:
    """Tracks which tiles are occupied by entities.

    After force-directed layout optimization, we only need simple
    occupancy tracking for:
    1. Validating final positions don't overlap
    2. Reserving positions for relay poles during connection planning
    """

    def __init__(self):
        self._occupied: Set[Tuple[int, int]] = set()

    def is_available(
        self, tile_pos: Tuple[int, int], footprint: Tuple[int, int]
    ) -> bool:
        """Check if a tile position is available for given footprint.

        Args:
            tile_pos: (tile_x, tile_y) top-left corner position
            footprint: (width, height) in tiles
        """
        width, height = footprint
        for dx in range(width):
            for dy in range(height):
                if (tile_pos[0] + dx, tile_pos[1] + dy) in self._occupied:
                    return False
        return True

    def mark_occupied(
        self, tile_pos: Tuple[int, int], footprint: Tuple[int, int]
    ) -> None:
        """Mark tiles as occupied.

        Args:
            tile_pos: (tile_x, tile_y) top-left corner position
            footprint: (width, height) in tiles
        """
        width, height = footprint
        for dx in range(width):
            for dy in range(height):
                self._occupied.add((tile_pos[0] + dx, tile_pos[1] + dy))

    def reserve_exact(
        self, tile_pos: Tuple[int, int], footprint: Tuple[int, int]
    ) -> bool:
        """Try to reserve exact tile position.

        Returns True if successful, False if position unavailable.

        Args:
            tile_pos: (tile_x, tile_y) top-left corner position
            footprint: (width, height) in tiles
        """
        if not self.is_available(tile_pos, footprint):
            return False
        self.mark_occupied(tile_pos, footprint)
        return True

    def rebuild_from_placements(self, placements: Dict[str, EntityPlacement]) -> None:
        """Rebuild occupancy from entity placements.

        Used after position optimization to update tile tracking.

        Args:
            placements: Dict of entity_id -> EntityPlacement
        """
        self._occupied.clear()

        for placement in placements.values():
            if placement.position is None:
                continue

            footprint = placement.properties.get("footprint", (1, 1))

            tile_x = int(placement.position[0] - footprint[0] / 2.0)
            tile_y = int(placement.position[1] - footprint[1] / 2.0)

            self.mark_occupied((tile_x, tile_y), footprint)
