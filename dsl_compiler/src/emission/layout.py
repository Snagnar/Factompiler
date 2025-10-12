from typing import Set, Tuple, Union


class LayoutEngine:
    """Simple layout engine for entity placement."""

    def __init__(self):
        self.next_x = -30
        self.next_y = 0
        self.row_height = 2
        self.entities_per_row = 6
        self.current_row_count = 0
        self.used_positions: Set[Tuple[int, int]] = set()
        self.entity_spacing = 2

    def get_next_position(self) -> Tuple[int, int]:
        pos = (int(self.next_x), int(self.next_y))

        while pos in self.used_positions:
            self._advance_position()
            pos = (int(self.next_x), int(self.next_y))

        self.used_positions.add(pos)
        self._advance_position()
        return pos

    def reserve_near(
        self, target: Tuple[int, int], max_radius: int = 6
    ) -> Tuple[int, int]:
        tx, ty = self.snap_to_grid(target)

        if (tx, ty) not in self.used_positions:
            self.used_positions.add((tx, ty))
            return (tx, ty)

        spacing_x = max(1, self.entity_spacing)
        spacing_y = max(1, self.row_height)

        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    pos = (tx + dx * spacing_x, ty + dy * spacing_y)
                    if pos not in self.used_positions:
                        self.used_positions.add(pos)
                        return pos

        return self.get_next_position()

    def snap_to_grid(
        self, pos: Tuple[Union[int, float], Union[int, float]]
    ) -> Tuple[int, int]:
        x, y = pos
        spacing_x = max(1, self.entity_spacing)
        spacing_y = max(1, self.row_height)

        snapped_x = int(round(x / spacing_x) * spacing_x)
        snapped_y = int(round(y / spacing_y) * spacing_y)

        return (snapped_x, snapped_y)

    def _advance_position(self):
        self.current_row_count += 1
        if self.current_row_count >= self.entities_per_row:
            self.next_x = -30
            self.next_y += self.row_height
            self.current_row_count = 0
        else:
            self.next_x += self.entity_spacing


__all__ = ["LayoutEngine"]
