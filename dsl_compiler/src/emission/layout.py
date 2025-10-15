import heapq
import math
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union


class LayoutEngine:
    """Heuristic layout engine that keeps related entities clustered."""

    def __init__(self):
        self.entity_spacing = 2
        self.row_height = 2
        self._origin = (0, 0)
        self.used_positions: Set[Tuple[int, int]] = set()
        self._occupied_tiles: Set[Tuple[int, int]] = set()
        self._candidate_heap: List[Tuple[float, int, Tuple[int, int]]] = []
        self._queued_positions: Set[Tuple[int, int]] = set()
        self._sequence_counter = 0
        self._zone_states: Dict[str, Dict[str, Any]] = {}
        self._zone_defaults: Dict[str, int] = {
            "north_literals": -self.row_height * 3,
            "south_exports": self.row_height * 3,
        }
        self._push_candidate(self._origin)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_next_position(
        self,
        footprint: Tuple[int, int] = (1, 1),
        padding: int = 0,
    ) -> Tuple[int, int]:
        while self._candidate_heap:
            _, _, pos = heapq.heappop(self._candidate_heap)
            if pos in self.used_positions:
                continue
            if not self._position_available(pos, footprint, padding):
                continue
            return self._claim_position(pos, footprint, padding)

        # Fallback: expand search ring vertically near the origin and retry.
        spacing_y = max(1, self.row_height)
        fallback = (0, (len(self.used_positions) + 1) * spacing_y)
        self._push_candidate(fallback)
        return self.get_next_position(footprint=footprint, padding=padding)

    def reserve_near(
        self,
        target: Tuple[int, int],
        max_radius: int = 12,
        footprint: Tuple[int, int] = (1, 1),
        padding: int = 0,
    ) -> Tuple[int, int]:
        snapped = self.snap_to_grid(target)

        if snapped not in self.used_positions and self._position_available(
            snapped, footprint, padding
        ):
            return self._claim_position(snapped, footprint, padding)

        candidate = self._find_nearest_available(
            snapped, max_radius, footprint, padding
        )
        if candidate is not None:
            return self._claim_position(candidate, footprint, padding)

        return self.get_next_position(footprint=footprint, padding=padding)

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

    def snap_to_grid(
        self, pos: Tuple[Union[int, float], Union[int, float]]
    ) -> Tuple[int, int]:
        x, y = pos
        spacing_x = max(1, self.entity_spacing)
        spacing_y = max(1, self.row_height)

        snapped_x = int(round(x / spacing_x) * spacing_x)
        snapped_y = int(round(y / spacing_y) * spacing_y)

        return (snapped_x, snapped_y)

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
        self._queued_positions.discard(pos)
        self._mark_occupied(pos, footprint, padding)
        if enqueue_neighbors:
            self._enqueue_neighbors(pos)
        return pos

    def _push_candidate(
        self, pos: Tuple[int, int], reference: Tuple[int, int] = None
    ) -> None:
        if pos in self.used_positions or pos in self._queued_positions:
            return

        ref = reference or self._origin
        spacing_x = max(1, self.entity_spacing)
        spacing_y = max(1, self.row_height)
        dx = (pos[0] - ref[0]) / spacing_x
        dy = (pos[1] - ref[1]) / spacing_y
        priority = max(abs(dx), abs(dy))

        heapq.heappush(self._candidate_heap, (priority, self._sequence_counter, pos))
        self._sequence_counter += 1
        self._queued_positions.add(pos)

    def _enqueue_neighbors(self, pos: Tuple[int, int]) -> None:
        for neighbor in self._iter_neighbor_positions(pos):
            self._push_candidate(neighbor, reference=pos)

    def _iter_neighbor_positions(self, pos: Tuple[int, int]):
        spacing_x = max(1, self.entity_spacing)
        spacing_y = max(1, self.row_height)
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
            yield (x + dx, y + dy)

    def _find_nearest_available(
        self,
        target: Tuple[int, int],
        max_radius: int,
        footprint: Tuple[int, int],
        padding: int,
    ) -> Optional[Tuple[int, int]]:
        spacing_x = max(1, self.entity_spacing)
        spacing_y = max(1, self.row_height)

        visited: Set[Tuple[int, int]] = set()
        heap: List[Tuple[float, int, int, Tuple[int, int]]] = []

        def push(pos: Tuple[int, int]) -> None:
            if pos in visited:
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
            for neighbor in self._iter_neighbor_positions(pos):
                push(neighbor)

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
        self._occupied_tiles.update(
            self._iter_footprint_tiles(pos, footprint, padding)
        )

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
