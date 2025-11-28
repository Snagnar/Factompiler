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

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def plan_power_grid(self, pole_type: str) -> List[PlannedPowerPole]:
        """
        Compute power pole placements based on entity positions.

        Uses adaptive placement:
        1. Estimate bounding box from entity positions
        2. Place poles in a grid covering the area
        3. Ensure all entities are within supply radius
        """
        if not pole_type:
            return []

        config = POWER_POLE_CONFIG.get(pole_type.lower())
        if config is None:
            self.diagnostics.warning(
                f"Unknown power pole type '{pole_type}'; skipping power grid"
            )
            return []

        # Get bounding box of all entities
        bounds = self._compute_entity_bounds()
        if bounds is None:
            self.diagnostics.warning("No entities to place power poles for")
            return []

        x_min, y_min, x_max, y_max = bounds

        # Add margin
        margin = config["supply_radius"]
        x_min -= margin
        y_min -= margin
        x_max += margin
        y_max += margin

        # Place poles in grid
        supply_radius = config["supply_radius"]
        spacing = supply_radius * 1.5  # Overlap for reliability

        prototype = config["prototype"]
        footprint = tuple(int(v) for v in config.get("footprint", (1, 1)))

        self._planned = []
        self.layout_plan.power_poles.clear()

        x = x_min
        while x <= x_max:
            y = y_min
            while y <= y_max:
                self._place_pole((int(x), int(y)), prototype, footprint)
                y += spacing
            x += spacing

        self.diagnostics.info(
            f"Placed {len(self._planned)} {pole_type} power poles in adaptive grid"
        )

        return list(self._planned)

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
            # Optionally skip power poles
            if exclude_power_poles and p.properties.get("is_power_pole"):
                continue
            positions.append(p.position)

        if not positions:
            return None

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        return (min(xs), min(ys), max(xs), max(ys))

    # -------------------------------------------------------------------------
    # Pole placement helper
    # -------------------------------------------------------------------------

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

        # Check for duplicate at this exact position
        for existing in self._planned:
            if existing.position == snapped_position:
                return existing  # Return existing pole

        # Create new pole
        pole = PlannedPowerPole(position=snapped_position, prototype=prototype)
        self._planned.append(pole)

        # Add to layout plan
        pole_id = f"power_pole_{len(self.layout_plan.power_poles) + 1}"
        self.layout_plan.add_power_pole(
            PowerPolePlacement(
                pole_id=pole_id,
                pole_type=prototype,
                position=snapped_position,
            )
        )

        # Also register as an entity placement so it appears in the blueprint
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

        # Mark tile as occupied
        self.tile_grid.mark_occupied(snapped_position, footprint)

        # Register with relay network so it can be reused for circuit routing
        if self.connection_planner and hasattr(
            self.connection_planner, "relay_network"
        ):
            center_position = (center_x, center_y)
            self.connection_planner.relay_network.add_relay_node(
                center_position, pole_id, prototype
            )

        return pole

    # -------------------------------------------------------------------------
    # Grid-based power pole placement from actual bounds
    # -------------------------------------------------------------------------

    def add_power_pole_grid_from_bounds(self, pole_type: str) -> None:
        """Add power poles in a grid pattern covering actual entity positions.

        This method is called AFTER layout optimization, so it uses the actual
        entity positions to determine the required power pole grid.

        Strategy:
        1. Compute actual bounding box from positioned entities
        2. Create a minimal grid of power poles to cover the area
        3. Only place poles that are actually needed

        Args:
            pole_type: Type of power pole (small/medium/big/substation)
        """
        config = POWER_POLE_CONFIG.get(pole_type.lower())
        if config is None:
            self.diagnostics.warning(
                f"Unknown power pole type '{pole_type}'; skipping power grid"
            )
            return

        # Get actual entity bounds (excluding any existing power poles)
        bounds = self._compute_entity_bounds(exclude_power_poles=True)
        if bounds is None:
            self.diagnostics.warning("No positioned entities for power grid")
            return

        x_min, y_min, x_max, y_max = bounds

        # In Factorio, power pole supply area is a SQUARE, not a circle
        # The supply_radius is the distance from pole center to edge of the square
        # So a pole at position (px, py) covers the square:
        #   x: [px - supply_radius, px + supply_radius]
        #   y: [py - supply_radius, py + supply_radius]
        supply_radius = float(config["supply_radius"])
        prototype = str(config["prototype"])
        footprint = tuple(int(v) for v in config.get("footprint", (1, 1)))

        # Clear any existing power poles
        self._planned = []
        self.layout_plan.power_poles.clear()

        # Remove any existing power pole placements
        to_remove = [
            pid
            for pid, p in self.layout_plan.entity_placements.items()
            if p.properties.get("is_power_pole")
        ]
        for pid in to_remove:
            del self.layout_plan.entity_placements[pid]

        # Calculate area dimensions
        area_width = x_max - x_min
        area_height = y_max - y_min

        # The supply area diameter (full side length of coverage square)
        coverage_side = supply_radius * 2.0

        # Spacing between poles - use the full coverage width with small overlap
        # to ensure reliable coverage at all points
        spacing = coverage_side * 0.95

        # For proper grid coverage with square areas:
        # We need poles spaced such that their coverage squares overlap
        # Place first pole at position that covers x_min/y_min edge

        # Calculate the starting position for the first pole
        # We want the pole's coverage to start at or before the entity bounds
        start_x = x_min
        start_y = y_min

        # Calculate how many poles in each direction
        # A pole at position p covers [p - supply_radius, p + supply_radius]
        # First pole at x_min covers up to x_min + supply_radius
        # We need enough poles that the last one covers x_max
        #
        # If area_width <= coverage_side, a single pole in the middle covers all
        # But if first pole is at x_min, it only covers to x_min + supply_radius
        # So we need: x_min + supply_radius >= x_max
        #          => supply_radius >= area_width
        #          => coverage_side / 2 >= area_width

        if area_width <= supply_radius:
            # A single pole at x_min will cover to x_min + supply_radius >= x_max
            n_cols = 1
        else:
            # Need multiple poles
            # First pole at x_min covers to x_min + supply_radius
            # Additional poles needed to cover the remaining distance
            remaining_width = area_width - supply_radius
            n_additional = int(math.ceil(remaining_width / spacing))
            n_cols = 1 + n_additional

        if area_height <= supply_radius:
            n_rows = 1
        else:
            remaining_height = area_height - supply_radius
            n_additional = int(math.ceil(remaining_height / spacing))
            n_rows = 1 + n_additional

        # Place poles in grid
        poles_added = 0
        for col in range(n_cols):
            for row in range(n_rows):
                x = start_x + col * spacing
                y = start_y + row * spacing

                # Snap to integer tile position, offset to avoid entity positions
                # Place poles at half-tile offsets to avoid overlapping with entities
                tile_x = int(round(x))
                tile_y = int(round(y))

                # Try to find a non-overlapping position nearby
                placed = False
                for dy_offset in [0, -1, 1, -2, 2]:
                    for dx_offset in [0, -1, 1, -2, 2]:
                        test_x = tile_x + dx_offset
                        test_y = tile_y + dy_offset
                        if self.tile_grid.is_available((test_x, test_y), footprint):
                            self._place_pole((test_x, test_y), prototype, footprint)
                            poles_added += 1
                            placed = True
                            break
                    if placed:
                        break

                if not placed:
                    # Force placement even if overlapping - warn user
                    self._place_pole((tile_x, tile_y), prototype, footprint)
                    poles_added += 1
                    self.diagnostics.warning(
                        f"Power pole at ({tile_x}, {tile_y}) may overlap with entities"
                    )

        # Calculate actual grid dimensions for logging
        grid_width = area_width
        grid_height = area_height

        self.diagnostics.info(
            f"Added {poles_added} {pole_type} power poles covering "
            f"{grid_width:.1f}x{grid_height:.1f} area "
            f"(entity bounds: {x_min:.1f},{y_min:.1f} to {x_max:.1f},{y_max:.1f})"
        )

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

        # Estimate blueprint area based on entity count
        entity_count = len(self.layout_plan.entity_placements)
        if entity_count == 0:
            return

        # Rough estimate: entities arranged in a roughly square layout
        # Average entity footprint ~1.5 tiles, with spacing ~2 tiles between entities
        avg_entity_area = 3.5 * 3.5  # Entity + spacing
        total_area = entity_count * avg_entity_area

        # Approximate dimensions (square-ish layout)
        approx_side = math.sqrt(total_area)

        # Add safety margin
        safety_margin = 5.0
        width = approx_side + 2 * safety_margin
        height = approx_side + 2 * safety_margin

        # Grid spacing based on supply radius (Factorio 2.0)
        supply_radius = float(config["supply_radius"])
        # Space poles by the full coverage diameter (2*supply_radius)
        # This ensures overlapping coverage while leaving maximum room for entities
        # Poles at positions (0,0), (2*r,0), (4*r,0)... will have coverage:
        #   pole at 0: covers [-r, r]
        #   pole at 2r: covers [r, 3r]
        # So coverage overlaps by exactly 0, which is fine
        spacing = 2.0 * supply_radius

        prototype = str(config["prototype"])
        footprint = tuple(int(v) for v in config.get("footprint", (1, 1)))

        # Create grid of power poles
        # Start at negative offset to ensure coverage of entities at (0,0)+
        # and offset by half-spacing to avoid typical user entity positions at integer coords
        start_offset = -spacing / 2.0 + footprint[0] / 2.0

        poles_added = 0
        x = start_offset
        while x < width:
            y = start_offset
            while y < height:
                # Convert to tile position (top-left corner)
                tile_x = int(round(x))
                tile_y = int(round(y))

                # Skip positions that would overlap with existing placements
                if not self.tile_grid.is_available((tile_x, tile_y), footprint):
                    y += spacing
                    continue

                pole_id = f"power_pole_{poles_added + 1}"

                # Convert to center position
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

                # Mark tile as occupied so layout solver avoids this position
                self.tile_grid.mark_occupied((tile_x, tile_y), footprint)

                poles_added += 1

                y += spacing
            x += spacing

        self.diagnostics.info(
            f"Added {poles_added} {pole_type} power poles in {int(width)}x{int(height)} grid "
            f"(spacing: {spacing:.1f} tiles, coverage: {supply_radius:.1f} tiles)"
        )
