from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from .layout_engine import LayoutEngine
from .layout_plan import LayoutPlan, EntityPlacement, PowerPolePlacement
from .cluster_analyzer import Cluster

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
        clusters: Optional[List[Cluster]] = None,
    ) -> None:
        self.layout: LayoutEngine = layout
        self.layout_plan: LayoutPlan = layout_plan
        self.diagnostics: ProgramDiagnostics = diagnostics

        self._planned: List[PlannedPowerPole] = []
        self._footprint: Tuple[int, int] = (1, 1)
        self._padding: int = 0
        self._supply_radius: float = 0.0
        self._wire_reach: float = 0.0

        # Cluster support
        self.clusters = clusters or []

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def plan_power_grid(self, pole_type: str) -> List[PlannedPowerPole]:
        """Compute power pole placements using pre-reserved positions."""

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

        if self.clusters:
            # Use pre-reserved power pole positions from clusters
            for cluster in self.clusters:
                if hasattr(cluster, "power_pole_position"):
                    position = cluster.power_pole_position
                    self._record_power_pole(position, config["prototype"])
                    self.diagnostics.info(
                        f"Placed power pole at reserved position {position}"
                    )
                else:
                    self.diagnostics.warning(
                        f"Cluster {cluster.cluster_id} has no reserved power pole position!"
                    )
        else:
            # Fallback to coverage-based (existing algorithm)
            return self._plan_coverage_based_grid(pole_type, config)

        # Enable connectivity (was previously disabled)
        self._ensure_connectivity_fixed(config["prototype"])

        return list(self._planned)

    def _plan_coverage_based_grid(
        self, pole_type: str, config: Dict
    ) -> List[PlannedPowerPole]:
        """Fallback coverage-based power grid (existing algorithm)."""
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

        return list(self._planned)

    def _position_has_power_coverage(self, pos: Tuple[int, int]) -> bool:
        """Check if position has power coverage."""
        for pole in self._planned:
            if math.dist(pos, pole.position) <= self._supply_radius:
                return True
        return False

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
        # Check for duplicate position
        for existing in self._planned:
            if existing.position == position:
                return existing  # Return existing pole instead of creating duplicate

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

    def _ensure_connectivity_fixed(self, prototype: str) -> None:
        """Ensure power poles connect, with fixed logic that prevents infinite loops."""
        if self._wire_reach <= 0 or len(self._planned) < 2:
            return

        reach_limit = max(0.0, self._wire_reach - 0.35)
        reach_sq = reach_limit * reach_limit

        # Find disconnected components
        components = self._compute_components(reach_sq)

        if len(components) <= 1:
            return  # Already connected

        # Connect components with single intermediate pole
        base_component = components[0]

        for candidate_component in components[1:]:
            # Find closest pair between components
            best_distance = float("inf")
            closest_pair = None

            for pole_a in base_component:
                center_a = self._position_center(pole_a.position)
                for pole_b in candidate_component:
                    center_b = self._position_center(pole_b.position)
                    distance = math.dist(center_a, center_b)
                    if distance < best_distance:
                        best_distance = distance
                        closest_pair = (pole_a, pole_b)

            if not closest_pair or best_distance <= reach_limit:
                continue  # Already connected or no valid pair

            # Place ONE intermediate pole at midpoint
            pole_a, pole_b = closest_pair
            center_a = self._position_center(pole_a.position)
            center_b = self._position_center(pole_b.position)

            mid_x = (center_a[0] + center_b[0]) / 2
            mid_y = (center_a[1] + center_b[1]) / 2

            # Try to place intermediate pole
            placement_pos = self.layout.reserve_near(
                (mid_x, mid_y),
                max_radius=3,
                footprint=self._footprint,
                padding=self._padding,
            )

            if placement_pos:
                self._record_power_pole(placement_pos, prototype)
                self.diagnostics.info(
                    f"Added intermediate power pole at {placement_pos} to connect components"
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
