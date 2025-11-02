from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
from dsl_compiler.src.common import ProgramDiagnostics
from .layout_engine import LayoutEngine
from .layout_plan import LayoutPlan, EntityPlacement, PowerPolePlacement

"""Power infrastructure planning for the layout module."""


@dataclass(frozen=True)
class PlannedPowerPole:
    """Result of the power planner describing a single pole placement."""

    position: Tuple[int, int]
    prototype: str


POWER_POLE_CONFIG: Dict[str, Dict[str, object]] = {
    "small": {
        "prototype": "small-electric-pole",
        "footprint": (1, 1),
        "supply_radius": 2,
        "padding": 0,
        "wire_reach": 9,
    },
    "medium": {
        "prototype": "medium-electric-pole",
        "footprint": (1, 1),
        "supply_radius": 3,
        "padding": 0,
        "wire_reach": 9,
    },
    "big": {
        "prototype": "big-electric-pole",
        "footprint": (2, 2),
        "supply_radius": 5,
        "padding": 1,
        "wire_reach": 30,
    },
    "substation": {
        "prototype": "substation",
        "footprint": (2, 2),
        "supply_radius": 9,
        "padding": 1,
        "wire_reach": 18,
    },
}


class PowerPlanner:
    """Plan power pole placements that cover all circuit entities."""

    def __init__(
        self,
        layout: LayoutEngine,
        layout_plan: LayoutPlan,
        diagnostics: ProgramDiagnostics,
    ) -> None:
        self.layout: LayoutEngine = layout
        self.layout_plan: LayoutPlan = layout_plan
        self.diagnostics: ProgramDiagnostics = diagnostics

        self._planned: List[PlannedPowerPole] = []
        self._footprint: Tuple[int, int] = (1, 1)
        self._padding: int = 0
        self._supply_radius: float = 0.0
        self._wire_reach: float = 0.0

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def plan_power_grid(self, pole_type: str) -> List[PlannedPowerPole]:
        """Compute power pole placements for the requested ``pole_type``."""

        if not pole_type:
            return []

        config = POWER_POLE_CONFIG.get(pole_type)
        if config is None:
            self.diagnostics.warning(
                f"Unknown power pole type '{pole_type}'; skipping power grid deployment"
            )
            return []

        self._planned = []
        self._footprint = tuple(int(v) for v in config.get("footprint", (1, 1)))
        self._padding = int(config.get("padding", 0))
        self._supply_radius = float(config.get("supply_radius", 0.0))
        self._wire_reach = float(config.get("wire_reach", 0.0))
        self.layout_plan.power_poles.clear()

        placements = [
            placement
            for placement in self.layout_plan.entity_placements.values()
            if placement.role != "power"
        ]

        if not placements:
            return []

        coverage_tiles = self._collect_coverage_tiles(placements)
        if not coverage_tiles:
            return []

        bounds = self._compute_bounds(coverage_tiles)
        uncovered_tiles = set(coverage_tiles)
        coverage_cache: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}

        while uncovered_tiles:
            target_tile = next(iter(uncovered_tiles))
            best_pos: Optional[Tuple[int, int]] = None
            best_total_cover: Set[Tuple[int, int]] = set()
            best_cover: Set[Tuple[int, int]] = set()
            best_distance = float("inf")

            for candidate in self._candidate_positions(target_tile, bounds):
                total_cover = coverage_cache.get(candidate)
                if total_cover is None:
                    total_cover = self._tiles_covered_by_position(
                        candidate,
                        coverage_tiles,
                    )
                    coverage_cache[candidate] = total_cover

                new_cover = total_cover & uncovered_tiles
                if not new_cover:
                    continue

                distance = self._distance_tile_to_position(target_tile, candidate)

                if len(new_cover) > len(best_cover) or (
                    len(new_cover) == len(best_cover) and distance < best_distance
                ):
                    best_pos = candidate
                    best_total_cover = total_cover
                    best_cover = new_cover
                    best_distance = distance

            if best_pos is not None:
                claimed = self._reserve_preferred(best_pos)
                planned = self._record_power_pole(claimed, config["prototype"])
                if claimed == best_pos and best_total_cover:
                    coverage = best_total_cover
                else:
                    coverage = self._tiles_covered_by_position(
                        planned.position,
                        coverage_tiles,
                    )
            else:
                claimed = self._reserve_near_tile(target_tile)
                planned = self._record_power_pole(claimed, config["prototype"])
                coverage = self._tiles_covered_by_position(
                    planned.position,
                    coverage_tiles,
                )

            if not coverage:
                coverage = {target_tile}

            coverage_cache[planned.position] = coverage
            uncovered_tiles.difference_update(coverage)

        residual_uncovered = self._tiles_missing_power(coverage_tiles)
        safety_guard = 0
        max_attempts = len(coverage_tiles) * 2 + 8

        while residual_uncovered and safety_guard < max_attempts:
            target_tile = residual_uncovered.pop()
            claimed = self._reserve_near_tile(target_tile)
            self._record_power_pole(claimed, config["prototype"])
            residual_uncovered = self._tiles_missing_power(coverage_tiles)
            safety_guard += 1

        if residual_uncovered:
            self.diagnostics.warning(
                "Unable to guarantee power coverage for all tiles after fallback placement"
            )

        if not self._planned:
            claimed = self.layout.get_next_position(
                footprint=self._footprint,
                padding=self._padding,
            )
            self._record_power_pole(claimed, config["prototype"])

        self._ensure_connectivity(config["prototype"])
        return list(self._planned)

    # ------------------------------------------------------------------
    # Coverage helpers
    # ------------------------------------------------------------------

    def _collect_coverage_tiles(
        self, placements: Iterable[EntityPlacement]
    ) -> Set[Tuple[int, int]]:
        tiles: Set[Tuple[int, int]] = set()
        for placement in placements:
            tiles.update(self._iter_entity_tiles(placement))
        return tiles

    def _compute_bounds(self, tiles: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        margin = max(
            2,
            math.ceil(self._supply_radius) + max(self._footprint) + self._padding,
        )
        min_x = min(tile[0] for tile in tiles) - margin
        min_y = min(tile[1] for tile in tiles) - margin
        max_x = max(tile[0] for tile in tiles) + margin
        max_y = max(tile[1] for tile in tiles) + margin
        return (min_x, min_y, max_x, max_y)

    def _iter_entity_tiles(
        self, placement: EntityPlacement
    ) -> Iterable[Tuple[int, int]]:
        footprint = placement.properties.get("footprint") or (1, 1)
        width = max(1, int(footprint[0]))
        height = max(1, int(footprint[1]))
        base_x, base_y = placement.position
        for dx in range(width):
            for dy in range(height):
                yield (base_x + dx, base_y + dy)

    def _candidate_positions(
        self,
        target_tile: Tuple[int, int],
        bounds: Tuple[int, int, int, int],
    ) -> Iterable[Tuple[int, int]]:
        tx, ty = target_tile
        min_x, min_y, max_x, max_y = bounds
        search_radius = max(1, math.ceil(self._supply_radius) + max(self._footprint))
        seen: Set[Tuple[int, int]] = set()

        for dx in range(-search_radius, search_radius + max(self._footprint) + 1):
            for dy in range(-search_radius, search_radius + max(self._footprint) + 1):
                raw_x = tx + dx
                raw_y = ty + dy
                if raw_x < min_x or raw_x > max_x or raw_y < min_y or raw_y > max_y:
                    continue
                candidate = self.layout.snap_to_grid((raw_x, raw_y))
                if candidate in seen:
                    continue
                seen.add(candidate)
                if not self.layout.can_reserve(
                    candidate,
                    footprint=self._footprint,
                    padding=self._padding,
                ):
                    continue
                yield candidate

    def _tiles_covered_by_position(
        self,
        pos: Tuple[int, int],
        tiles: Iterable[Tuple[int, int]],
    ) -> Set[Tuple[int, int]]:
        cx, cy = self._position_center(pos)
        radius = float(self._supply_radius) + 0.45
        radius_sq = radius * radius
        covered: Set[Tuple[int, int]] = set()
        for tile in tiles:
            tx = tile[0] + 0.5
            ty = tile[1] + 0.5
            if (tx - cx) ** 2 + (ty - cy) ** 2 <= radius_sq:
                covered.add(tile)
        return covered

    def _tiles_missing_power(
        self,
        tiles: Iterable[Tuple[int, int]],
    ) -> Set[Tuple[int, int]]:
        centers = [self._position_center(plan.position) for plan in self._planned]
        if not centers:
            return set(tiles)

        radius_sq = (float(self._supply_radius) + 0.5) ** 2
        uncovered: Set[Tuple[int, int]] = set()
        for tile in tiles:
            tx = tile[0] + 0.5
            ty = tile[1] + 0.5
            if not any(
                (tx - cx) ** 2 + (ty - cy) ** 2 <= radius_sq for cx, cy in centers
            ):
                uncovered.add(tile)
        return uncovered

    # ------------------------------------------------------------------
    # Reservation helpers
    # ------------------------------------------------------------------

    def _reserve_preferred(self, position: Tuple[int, int]) -> Tuple[int, int]:
        claimed = self.layout.reserve_exact(
            position,
            footprint=self._footprint,
            padding=self._padding,
        )
        if claimed is None:
            claimed = self.layout.reserve_near(
                position,
                max_radius=max(
                    6, math.ceil(self._supply_radius) + max(self._footprint)
                ),
                footprint=self._footprint,
                padding=self._padding,
            )
        if claimed is None:
            claimed = self.layout.get_next_position(
                footprint=self._footprint,
                padding=self._padding,
            )
        return claimed

    def _reserve_near_tile(
        self,
        tile: Tuple[int, int],
    ) -> Tuple[int, int]:
        placement_pos = self.layout.reserve_near(
            tile,
            max_radius=max(
                8,
                math.ceil(self._supply_radius) + max(self._footprint) + self._padding,
            ),
            footprint=self._footprint,
            padding=self._padding,
        )
        if placement_pos is None:
            placement_pos = self.layout.get_next_position(
                footprint=self._footprint,
                padding=self._padding,
            )
        return placement_pos

    def _record_power_pole(
        self,
        position: Tuple[int, int],
        prototype: str,
    ) -> PlannedPowerPole:
        pole = PlannedPowerPole(position=position, prototype=prototype)
        self._planned.append(pole)
        pole_id = f"power_pole_{len(self.layout_plan.power_poles) + 1}"
        self.layout_plan.add_power_pole(
            PowerPolePlacement(
                pole_id=pole_id,
                pole_type=prototype,
                position=position,
            )
        )
        return pole

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _position_center(self, pos: Tuple[int, int]) -> Tuple[float, float]:
        return (
            pos[0] + self._footprint[0] / 2.0,
            pos[1] + self._footprint[1] / 2.0,
        )

    def _distance_tile_to_position(
        self,
        tile: Tuple[int, int],
        pos: Tuple[int, int],
    ) -> float:
        tile_center = (tile[0] + 0.5, tile[1] + 0.5)
        position_center = self._position_center(pos)
        return math.hypot(
            tile_center[0] - position_center[0],
            tile_center[1] - position_center[1],
        )

    # ------------------------------------------------------------------
    # Connectivity helpers
    # ------------------------------------------------------------------

    def _ensure_connectivity(self, prototype: str) -> None:
        if self._wire_reach <= 0 or len(self._planned) < 2:
            return

        reach_limit = max(0.0, self._wire_reach - 0.35)
        if reach_limit <= 0:
            reach_limit = self._wire_reach

        reach_sq = reach_limit * reach_limit
        # ✅ FIX: Cap iterations to prevent infinite loops on complex blueprints
        max_attempts = min(len(self._planned) + 8, 50)
        attempts = 0
        last_component_count = float('inf')  # ✅ Track progress

        while attempts < max_attempts:
            components = self._compute_components(reach_sq)
            if len(components) <= 1:
                break

            # Add progress check - if no improvement after 5 iterations, abort
            if attempts > 0 and attempts % 5 == 0:
                if len(components) == last_component_count:
                    self.diagnostics.warning(
                        f"Power connectivity stalled after {attempts} attempts with "
                        f"{len(components)} components remaining. Stopping early."
                    )
                    break
            
            last_component_count = len(components)

            base_component = components[0]
            best_distance = float("inf")
            closest_pair: Optional[Tuple[PlannedPowerPole, PlannedPowerPole]] = None

            for candidate_component in components[1:]:
                for pole_a in base_component:
                    center_a = self._position_center(pole_a.position)
                    for pole_b in candidate_component:
                        center_b = self._position_center(pole_b.position)
                        distance = math.dist(center_a, center_b)
                        if distance < best_distance:
                            best_distance = distance
                            closest_pair = (pole_a, pole_b)

            if not closest_pair:
                break

            if best_distance <= reach_limit:
                attempts += 1
                continue

            pole_a, pole_b = closest_pair
            span_cap = reach_limit if reach_limit > 0 else self._wire_reach
            needed = max(1, math.ceil(best_distance / span_cap) - 1)
            center_a = self._position_center(pole_a.position)
            center_b = self._position_center(pole_b.position)

            for index in range(1, needed + 1):
                ratio = index / (needed + 1)
                target_center = (
                    center_a[0] + (center_b[0] - center_a[0]) * ratio,
                    center_a[1] + (center_b[1] - center_a[1]) * ratio,
                )
                approx_top_left = (
                    target_center[0] - self._footprint[0] / 2.0,
                    target_center[1] - self._footprint[1] / 2.0,
                )
                placement_pos = self.layout.reserve_exact(
                    approx_top_left,
                    footprint=self._footprint,
                    padding=self._padding,
                )
                if placement_pos is None:
                    placement_pos = self.layout.reserve_near(
                        approx_top_left,
                        max_radius=max(6, math.ceil(self._wire_reach)),
                        footprint=self._footprint,
                        padding=self._padding,
                    )
                if placement_pos is None:
                    placement_pos = self.layout.get_next_position(
                        footprint=self._footprint,
                        padding=self._padding,
                    )
                self._record_power_pole(placement_pos, prototype)

            attempts += 1

        if attempts >= max_attempts and len(components) > 1:
            self.diagnostics.warning(
                f"Could not fully connect power grid after {max_attempts} attempts. "
                f"{len(components)} disconnected components remain."
            )

    def _compute_components(
        self,
        reach_sq: float,
    ) -> List[List[PlannedPowerPole]]:
        if not self._planned:
            return []

        centers = [self._position_center(pole.position) for pole in self._planned]
        components: List[List[PlannedPowerPole]] = []
        visited: Set[int] = set()

        for idx in range(len(self._planned)):
            if idx in visited:
                continue
            stack = [idx]
            component_indices: List[int] = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component_indices.append(current)
                cx, cy = centers[current]
                for other in range(len(self._planned)):
                    if other == current or other in visited:
                        continue
                    ox, oy = centers[other]
                    if (ox - cx) ** 2 + (oy - cy) ** 2 <= reach_sq:
                        stack.append(other)
            components.append([self._planned[i] for i in component_indices])

        return components
