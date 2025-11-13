from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from .tile_grid import TileGrid
from .layout_plan import LayoutPlan, PowerPolePlacement

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

    def _compute_entity_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Compute bounding box of all positioned entities."""
        if not self.layout_plan.entity_placements:
            return None

        positions = [
            p.position
            for p in self.layout_plan.entity_placements.values()
            if p.position is not None
        ]

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

        # Register with relay network so it can be reused for circuit routing
        if self.connection_planner and hasattr(
            self.connection_planner, "relay_network"
        ):
            # Convert tile position to center position for relay network
            center_x = snapped_position[0] + footprint[0] / 2.0
            center_y = snapped_position[1] + footprint[1] / 2.0
            center_position = (center_x, center_y)

            self.connection_planner.relay_network.add_relay_node(
                center_position, pole_id, prototype
            )

        return pole
