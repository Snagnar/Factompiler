from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from .tile_grid import TileGrid
from .layout_plan import LayoutPlan, PowerPolePlacement, EntityPlacement

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
        "supply_radius": 2.5,
        "wire_reach": 9,
    },
    "medium": {
        "prototype": "medium-electric-pole",
        "footprint": (1, 1),
        "supply_radius": 3.5,
        "wire_reach": 9,
    },
    "big": {
        "prototype": "big-electric-pole",
        "footprint": (2, 2),
        "supply_radius": 5,
        "wire_reach": 30,
    },
    "substation": {
        "prototype": "substation",
        "footprint": (2, 2),
        "supply_radius": 9,
        "wire_reach": 18,
    },
}


class PowerPlanner:
    """Plan power pole placements using simple geometric corner/perimeter placement."""

    def __init__(
        self,
        tile_grid: TileGrid,
        layout_plan: LayoutPlan,
        diagnostics: ProgramDiagnostics,
        connection_planner: Optional[Any] = None,
    ) -> None:
        self.tile_grid = tile_grid
        self.layout_plan = layout_plan
        self.diagnostics = diagnostics
        self.connection_planner = connection_planner
        self._planned: List[PlannedPowerPole] = []

    def _compute_entity_bounds(
        self, exclude_power_poles: bool = True
    ) -> Optional[Tuple[float, float, float, float]]:
        """Compute bounding box of all positioned entities.

        Args:
            exclude_power_poles: If True, exclude power poles from bounds calculation
        """
        if not self.layout_plan.entity_placements:
            return None

        positions = []
        for p in self.layout_plan.entity_placements.values():
            if p.position is None:
                continue
            if exclude_power_poles and p.properties.get("is_power_pole"):
                continue
            positions.append(p.position)

        if not positions:
            return None

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        return (min(xs), min(ys), max(xs), max(ys))

    def _place_pole(
        self, position: Tuple[int, int], prototype: str, footprint: Tuple[int, int]
    ) -> PlannedPowerPole:
        """Place a single power pole at the given position.

        Args:
            position: (x, y) tile position (top-left for 2x2 poles)
            prototype: Factorio entity prototype name
            footprint: (width, height) in tiles

        Returns:
            PlannedPowerPole object
        """
        # ✅ FIX: Snap position to integer grid
        snapped_x = int(round(position[0]))
        snapped_y = int(round(position[1]))
        snapped_position = (snapped_x, snapped_y)

        for existing in self._planned:
            if existing.position == snapped_position:
                return existing  # Return existing pole

        pole = PlannedPowerPole(position=snapped_position, prototype=prototype)
        self._planned.append(pole)

        pole_id = f"power_pole_{len(self.layout_plan.power_poles) + 1}"
        self.layout_plan.add_power_pole(
            PowerPolePlacement(
                pole_id=pole_id,
                pole_type=prototype,
                position=snapped_position,
            )
        )

        center_x = snapped_position[0] + footprint[0] / 2.0
        center_y = snapped_position[1] + footprint[1] / 2.0

        placement = EntityPlacement(
            ir_node_id=pole_id,
            entity_type=prototype,
            position=(center_x, center_y),
            properties={
                "footprint": footprint,
                "is_power_pole": True,
                "debug_info": {
                    "variable": pole_id,
                    "operation": "power",
                    "details": "electricity supply",
                },
            },
            role="power_pole",
        )
        self.layout_plan.add_placement(placement)

        self.tile_grid.mark_occupied(snapped_position, footprint)

        if self.connection_planner and hasattr(
            self.connection_planner, "relay_network"
        ):
            center_position = (center_x, center_y)
            self.connection_planner.relay_network.add_relay_node(
                center_position, pole_id, prototype
            )

        return pole

    def add_power_pole_grid(self, pole_type: str) -> None:
        """Add power poles in a regular grid pattern with fixed positions.

        NOTE: This method is DEPRECATED. Use add_power_pole_grid_from_bounds instead,
        which is called after layout optimization and uses actual entity positions.

        Strategy:
        1. Estimate the final blueprint bounding box based on entity count
        2. Create a grid of power poles spaced by supply radius
        3. Assign fixed positions to poles (layout optimizer won't move them)

        Args:
            pole_type: Type of power pole (small/medium/big/substation)
        """
        config = POWER_POLE_CONFIG.get(pole_type.lower())
        if config is None:
            self.diagnostics.warning(
                f"Unknown power pole type '{pole_type}'; skipping power grid"
            )
            return

        entity_count = len(self.layout_plan.entity_placements)
        if entity_count == 0:
            return

        # Calculate bounds from entity count estimate
        avg_entity_area = 3.5 * 3.5  # Entity + spacing
        total_area = entity_count * avg_entity_area
        approx_side = math.sqrt(total_area)
        safety_margin = 5.0
        estimated_width = approx_side + 2 * safety_margin
        estimated_height = approx_side + 2 * safety_margin

        # Also consider user-specified entity positions to ensure full coverage
        user_min_x, user_min_y = 0.0, 0.0
        user_max_x, user_max_y = 0.0, 0.0
        has_user_positions = False

        for placement in self.layout_plan.entity_placements.values():
            if placement.position is not None:
                is_user = placement.properties.get("user_specified_position")
                if is_user:
                    has_user_positions = True
                    footprint_w, footprint_h = placement.properties.get(
                        "footprint", (1, 1)
                    )
                    px, py = placement.position
                    user_min_x = min(user_min_x, px)
                    user_min_y = min(user_min_y, py)
                    user_max_x = max(user_max_x, px + footprint_w)
                    user_max_y = max(user_max_y, py + footprint_h)

        # Use the larger of estimated bounds and actual user bounds
        if has_user_positions:
            # Extend bounds to cover all user-specified positions plus margin
            width = max(estimated_width, user_max_x - user_min_x + 2 * safety_margin)
            height = max(estimated_height, user_max_y - user_min_y + 2 * safety_margin)
            # Adjust start offset to cover negative positions if any
            start_x_offset = min(0.0, user_min_x) - safety_margin
            start_y_offset = min(0.0, user_min_y) - safety_margin
        else:
            width = estimated_width
            height = estimated_height
            start_x_offset = 0.0
            start_y_offset = 0.0

        supply_radius = float(config["supply_radius"])
        # This ensures overlapping coverage while leaving maximum room for entities
        spacing = 2.0 * supply_radius

        prototype = str(config["prototype"])
        footprint = tuple(int(v) for v in config.get("footprint", (1, 1)))

        # Start at negative offset to ensure coverage of entities at (0,0)+
        base_offset = -spacing / 2.0 + footprint[0] / 2.0
        start_x = start_x_offset + base_offset
        start_y = start_y_offset + base_offset
        end_x = start_x_offset + width
        end_y = start_y_offset + height

        poles_added = 0
        x = start_x
        while x < end_x:
            y = start_y
            while y < end_y:
                tile_x = int(round(x))
                tile_y = int(round(y))

                if not self.tile_grid.is_available((tile_x, tile_y), footprint):
                    y += spacing
                    continue

                pole_id = f"power_pole_{poles_added + 1}"

                center_x = tile_x + footprint[0] / 2.0
                center_y = tile_y + footprint[1] / 2.0

                placement = EntityPlacement(
                    ir_node_id=pole_id,
                    entity_type=prototype,
                    position=(center_x, center_y),  # Fixed position
                    properties={
                        "footprint": footprint,
                        "is_power_pole": True,
                        "pole_type": pole_type,
                        "fixed_position": True,  # Don't optimize this position
                        "debug_info": {
                            "variable": f"power_pole_{poles_added + 1}",
                            "operation": "power",
                            "details": "electricity supply",
                        },
                    },
                    role="power_pole",
                )

                self.layout_plan.add_placement(placement)

                self.tile_grid.mark_occupied((tile_x, tile_y), footprint)

                poles_added += 1

                y += spacing
            x += spacing

        self.diagnostics.info(
            f"Added {poles_added} {pole_type} power poles in {int(width)}x{int(height)} grid "
            f"(spacing: {spacing:.1f} tiles, coverage: {supply_radius:.1f} tiles)"
        )

    def add_power_pole_grid_from_bounds(self, pole_type: str) -> None:
        """Add power poles based on actual entity positions.

        This method should be called AFTER layout optimization to place
        power poles based on actual entity positions rather than estimates.

        Strategy:
        1. Compute bounding box of all placed entities
        2. Create a minimal grid of power poles covering the bounds
        3. Only place poles where there's actually something to power

        Args:
            pole_type: Type of power pole (small/medium/big/substation)
        """
        config = POWER_POLE_CONFIG.get(pole_type.lower())
        if config is None:
            self.diagnostics.warning(
                f"Unknown power pole type '{pole_type}'; skipping power grid"
            )
            return

        # Get actual entity bounds (excluding power poles)
        bounds = self._compute_entity_bounds(exclude_power_poles=True)
        if bounds is None:
            return

        min_x, min_y, max_x, max_y = bounds

        supply_radius = float(config["supply_radius"])
        # Spacing should provide overlapping coverage
        spacing = 2.0 * supply_radius

        prototype = str(config["prototype"])
        footprint = tuple(int(v) for v in config.get("footprint", (1, 1)))

        # Expand bounds by supply radius to ensure edge coverage
        grid_min_x = min_x - supply_radius
        grid_min_y = min_y - supply_radius
        grid_max_x = max_x + supply_radius
        grid_max_y = max_y + supply_radius

        poles_added = 0

        # Start at the first pole position that would cover the minimum bound
        x = grid_min_x
        while x <= grid_max_x:
            y = grid_min_y
            while y <= grid_max_y:
                tile_x = int(round(x))
                tile_y = int(round(y))

                if not self.tile_grid.is_available((tile_x, tile_y), footprint):
                    # Try to find a nearby available position
                    found = False
                    for offset in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1)]:
                        alt_x = tile_x + offset[0]
                        alt_y = tile_y + offset[1]
                        if self.tile_grid.is_available((alt_x, alt_y), footprint):
                            tile_x, tile_y = alt_x, alt_y
                            found = True
                            break
                    if not found:
                        y += spacing
                        continue

                pole_id = f"power_pole_{poles_added + 1}"

                center_x = tile_x + footprint[0] / 2.0
                center_y = tile_y + footprint[1] / 2.0

                placement = EntityPlacement(
                    ir_node_id=pole_id,
                    entity_type=prototype,
                    position=(center_x, center_y),
                    properties={
                        "footprint": footprint,
                        "is_power_pole": True,
                        "pole_type": pole_type,
                        "debug_info": {
                            "variable": f"power_pole_{poles_added + 1}",
                            "operation": "power",
                            "details": "electricity supply",
                        },
                    },
                    role="power_pole",
                )

                self.layout_plan.add_placement(placement)
                self.tile_grid.mark_occupied((tile_x, tile_y), footprint)

                poles_added += 1

                y += spacing
            x += spacing

        self.diagnostics.info(
            f"Added {poles_added} {pole_type} power poles for entity bounds "
            f"({min_x:.0f},{min_y:.0f})-({max_x:.0f},{max_y:.0f}) "
            f"(spacing: {spacing:.1f} tiles, coverage: {supply_radius:.1f} tiles)"
        )
