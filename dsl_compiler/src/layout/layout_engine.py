import heapq
import math
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union


class LayoutEngine:
    """Heuristic layout engine that keeps related entities clustered."""

    def __init__(self):
        self.entity_spacing = 1
        self.row_height = 2
        self._origin = (0, 0)
        self.used_positions: Set[Tuple[int, int]] = set()
        self._occupied_tiles: Set[Tuple[int, int]] = set()
        self._zone_states: Dict[str, Dict[str, Any]] = {}
        self._zone_defaults: Dict[str, int] = {
            "north_literals": -self.row_height * 3,
            "south_exports": self.row_height * 3,
        }
        # Cluster-based placement state
        self._next_row_x = 0
        self._next_row_y = 0
        self._cluster_bounds: Optional[Tuple[int, int, int, int]] = None
        self._current_row_max_height = 1  # Track tallest entity in current row

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_next_position(
        self,
        footprint: Tuple[int, int] = (1, 1),
        padding: int = 0,
        alignment: int = 1,
    ) -> Tuple[int, int]:
        """Get next available position - SIMPLIFIED for cluster-bounded placement."""
        
        spacing_x = max(alignment, self.entity_spacing)
        spacing_y = max(alignment, self.row_height)
        
        # Try current position
        while True:
            candidate = (self._next_row_x, self._next_row_y)
            
            # Check alignment
            if candidate[0] % alignment == 0 and candidate[1] % alignment == 0:
                if self._position_available(candidate, footprint, padding):
                    self._next_row_x += spacing_x
                    return self._claim_position(candidate, footprint, padding, enqueue_neighbors=False)
            
            # Move to next position
            self._next_row_x += spacing_x
            if self._next_row_x > 20:  # Simple row wrap
                self._next_row_x = 0
                self._next_row_y += spacing_y

    def reserve_near(
        self,
        target: Tuple[int, int],
        max_radius: int = 12,
        footprint: Tuple[int, int] = (1, 1),
        padding: int = 0,
        alignment: int = 1,
    ) -> Tuple[int, int]:
        snapped = self._snap_to_alignment(target, alignment)

        if snapped not in self.used_positions and self._position_available(
            snapped, footprint, padding
        ):
            return self._claim_position(snapped, footprint, padding)

        candidate = self._find_nearest_available(
            snapped, max_radius, footprint, padding, alignment
        )
        if candidate is not None:
            return self._claim_position(candidate, footprint, padding)

        return self.get_next_position(
            footprint=footprint, padding=padding, alignment=alignment
        )

    def reserve_along_path(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        ratio: float,
        *,
        strategy: str = "euclidean",
        max_radius: int = 12,
        footprint: Tuple[int, int] = (1, 1),
        padding: int = 0,
    ) -> Tuple[int, int]:
        ratio = min(max(ratio, 0.0), 1.0)
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        if strategy == "manhattan":
            total_x = abs(dx)
            total_y = abs(dy)
            total = total_x + total_y
            if total == 0:
                target_x, target_y = start
            else:
                target_distance = ratio * total
                take_x = min(total_x, target_distance)
                take_y = max(0.0, target_distance - take_x)
                target_x = start[0] + math.copysign(take_x, dx)
                target_y = start[1] + math.copysign(take_y, dy)
        else:
            target_x = start[0] + dx * ratio
            target_y = start[1] + dy * ratio

        return self.reserve_near(
            (target_x, target_y),
            max_radius=max_radius,
            footprint=footprint,
            padding=padding,
        )

    def reserve_in_zone(
        self,
        zone: str,
        footprint: Tuple[int, int] = (1, 1),
        padding: int = 0,
    ) -> Tuple[int, int]:
        state = self._zone_states.setdefault(zone, self._create_zone_state(zone))
        spacing_x = max(1, self.entity_spacing)

        while True:
            offset_index = state["next_index"]
            state["next_index"] += 1
            x_offset = self._zone_offset(offset_index, spacing_x)
            pos = (state["origin"][0] + x_offset, state["origin"][1])
            snapped = self.snap_to_grid(pos)

            if snapped in self.used_positions:
                continue
            if not self._position_available(snapped, footprint, padding):
                continue
            return self._claim_position(
                snapped,
                footprint,
                padding,
                enqueue_neighbors=False,
            )

    def reserve_exact(
        self,
        pos: Tuple[int, int],
        *,
        footprint: Tuple[int, int] = (1, 1),
        padding: int = 0,
    ) -> Optional[Tuple[int, int]]:
        """Attempt to claim the exact snapped position if it is currently unused."""

        snapped = self.snap_to_grid(pos)
        if snapped in self.used_positions:
            return None
        if not self._position_available(snapped, footprint, padding):
            return None
        return self._claim_position(snapped, footprint, padding)

    def can_reserve(
        self,
        pos: Tuple[int, int],
        *,
        footprint: Tuple[int, int] = (1, 1),
        padding: int = 0,
    ) -> bool:
        """Check whether a position is available without mutating layout state."""

        snapped = self.snap_to_grid(pos)
        if snapped in self.used_positions:
            return False
        return self._position_available(snapped, footprint, padding)

    def snap_to_grid(
        self, pos: Tuple[Union[int, float], Union[int, float]]
    ) -> Tuple[int, int]:
        x, y = pos
        spacing_x = max(1, self.entity_spacing)
        spacing_y = max(1, self.row_height)

        snapped_x = int(round(x / spacing_x) * spacing_x)
        snapped_y = int(round(y / spacing_y) * spacing_y)

        return (snapped_x, snapped_y)

    def set_cluster_bounds(self, bounds: Optional[Tuple[int, int, int, int]]) -> None:
        """Set bounds for next placements (x1, y1, x2, y2) or None to clear."""
        self._cluster_bounds = bounds
        if bounds is not None:
            # Reset row placement to start of bounds
            self._next_row_x = bounds[0]
            self._next_row_y = bounds[1]
            self._current_row_max_height = 1
        else:
            # Clear bounds, reset to normal placement
            self._next_row_x = 0
            self._next_row_y = 0
            self._current_row_max_height = 1

    def get_cluster_position(
        self,
        footprint: Tuple[int, int] = (1, 1),
        alignment: int = 1,
    ) -> Tuple[int, int]:
        """Get position within cluster bounds."""
        
        if self._cluster_bounds is None:
            return self.get_next_position(footprint, 0, alignment)
        
        x1, y1, x2, y2 = self._cluster_bounds
        
        # Simple row placement within bounds
        while self._next_row_y < y2:
            while self._next_row_x < x2:
                candidate = (self._next_row_x, self._next_row_y)
                
                if candidate[0] % alignment == 0 and candidate[1] % alignment == 0:
                    if self._position_available(candidate, footprint, 0):
                        if self._next_row_x + footprint[0] <= x2 and self._next_row_y + footprint[1] <= y2:
                            result = self._claim_position(candidate, footprint, 0, enqueue_neighbors=False)
                            # Track the tallest entity in this row
                            self._current_row_max_height = max(self._current_row_max_height, footprint[1])
                            self._next_row_x += max(alignment, footprint[0])
                            return result
                
                self._next_row_x += alignment
            
            # Move to next row - advance by the tallest entity in the previous row
            self._next_row_x = x1
            self._next_row_y += max(alignment, self._current_row_max_height)
            self._current_row_max_height = 1  # Reset for next row
        
        # Cluster full - fall back to unrestricted placement
        # This allows overflow entities to be placed outside the cluster bounds
        return self.get_next_position(footprint, 0, alignment)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _claim_position(
        self,
        pos: Tuple[int, int],
        footprint: Tuple[int, int],
        padding: int,
        *,
        enqueue_neighbors: bool = True,
    ) -> Tuple[int, int]:
        self.used_positions.add(pos)
        self._mark_occupied(pos, footprint, padding)
        return pos

    def _snap_to_alignment(
        self, pos: Tuple[Union[int, float], Union[int, float]], alignment: int
    ) -> Tuple[int, int]:
        """Snap position to alignment grid."""
        x, y = pos

        if alignment == 1:
            # Normal snapping
            return self.snap_to_grid(pos)

        # Align to multiple of alignment
        snapped_x = int(round(x / alignment) * alignment)
        snapped_y = int(round(y / alignment) * alignment)

        return (snapped_x, snapped_y)

    def _find_nearest_available(
        self,
        target: Tuple[int, int],
        max_radius: int,
        footprint: Tuple[int, int],
        padding: int,
        alignment: int = 1,
    ) -> Optional[Tuple[int, int]]:
        spacing_x = max(alignment, self.entity_spacing)
        spacing_y = max(alignment, self.row_height)

        visited: Set[Tuple[int, int]] = set()
        heap: List[Tuple[float, int, int, Tuple[int, int]]] = []

        def push(pos: Tuple[int, int]) -> None:
            if pos in visited:
                return
            # Only consider positions that respect alignment
            if pos[0] % alignment != 0 or pos[1] % alignment != 0:
                return
            visited.add(pos)
            dx = (pos[0] - target[0]) / spacing_x
            dy = (pos[1] - target[1]) / spacing_y
            distance = max(abs(dx), abs(dy))
            heapq.heappush(
                heap,
                (
                    distance,
                    abs(pos[1] - target[1]),
                    abs(pos[0] - target[0]),
                    pos,
                ),
            )

        push(target)

        while heap:
            distance, _, _, pos = heapq.heappop(heap)
            if distance > max_radius:
                break
            if pos not in self.used_positions and self._position_available(
                pos, footprint, padding
            ):
                return pos
            # Generate neighbors inline
            x, y = pos
            offsets = (
                (spacing_x, 0),
                (-spacing_x, 0),
                (0, spacing_y),
                (0, -spacing_y),
                (spacing_x, spacing_y),
                (spacing_x, -spacing_y),
                (-spacing_x, spacing_y),
                (-spacing_x, -spacing_y),
            )
            for dx, dy in offsets:
                push((x + dx, y + dy))

        return None

    # ------------------------------------------------------------------
    # Footprint helpers
    # ------------------------------------------------------------------

    def _position_available(
        self, pos: Tuple[int, int], footprint: Tuple[int, int], padding: int
    ) -> bool:
        for tile in self._iter_footprint_tiles(pos, footprint, padding):
            if tile in self._occupied_tiles:
                return False
        return True

    def _mark_occupied(
        self, pos: Tuple[int, int], footprint: Tuple[int, int], padding: int
    ) -> None:
        self._occupied_tiles.update(self._iter_footprint_tiles(pos, footprint, padding))

    def _iter_footprint_tiles(
        self, pos: Tuple[int, int], footprint: Tuple[int, int], padding: int
    ) -> Iterable[Tuple[int, int]]:
        width = max(1, int(footprint[0]))
        height = max(1, int(footprint[1]))
        pad = max(0, int(padding))

        start_x = pos[0] - pad
        start_y = pos[1] - pad
        end_x = pos[0] + width + pad - 1
        end_y = pos[1] + height + pad - 1

        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                yield (x, y)

    def _create_zone_state(self, zone: str) -> Dict[str, Any]:
        origin_y = self._zone_defaults.get(zone)
        if origin_y is None:
            # Place additional zones below existing ones to avoid overlap.
            origin_y = (len(self._zone_states) + 1) * self.row_height * 4
        origin = (0, origin_y)
        return {
            "origin": origin,
            "next_index": 0,
        }

    def _zone_offset(self, index: int, spacing: int) -> int:
        if index == 0:
            return 0
        step = ((index + 1) // 2) * spacing
        return step if index % 2 == 1 else -step


__all__ = ["LayoutEngine"]
