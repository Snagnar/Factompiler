from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from .layout_engine import LayoutEngine
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
        layout: LayoutEngine,
        layout_plan: LayoutPlan,
        diagnostics: ProgramDiagnostics,
        clusters: Optional[List[Any]] = None,
        connection_planner: Optional[Any] = None,
    ) -> None:
        self.layout = layout
        self.layout_plan = layout_plan
        self.diagnostics = diagnostics
        self.clusters = clusters or []
        self.connection_planner = connection_planner
        self._planned: List[PlannedPowerPole] = []

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def plan_power_grid(self, pole_type: str) -> List[PlannedPowerPole]:
        """Compute power pole placements using geometric corner/perimeter placement.

        This is a vastly simplified algorithm compared to the old coverage-based approach:
        1. Get pole config (footprint, prototype name)
        2. For each cluster, calculate corner/perimeter positions based on pole type
        3. Place poles at those positions (outside cluster bounds)
        4. Done! No coverage checks, no connectivity checks needed.

        Args:
            pole_type: One of "small", "medium", "big", "substation"

        Returns:
            List of PlannedPowerPole objects
        """
        if not pole_type:
            return []

        config = POWER_POLE_CONFIG.get(pole_type.lower())
        if config is None:
            self.diagnostics.warning(
                f"Unknown power pole type '{pole_type}'; skipping power grid deployment"
            )
            return []

        if not self.clusters:
            self.diagnostics.warning("No clusters available for power pole placement")
            return []

        self._planned = []
        self.layout_plan.power_poles.clear()

        prototype = config["prototype"]
        footprint = tuple(int(v) for v in config.get("footprint", (1, 1)))

        # Place poles for each cluster based on type
        pole_type_lower = pole_type.lower()

        for cluster in self.clusters:
            if not hasattr(cluster, "bounds") or cluster.bounds is None:
                continue

            x1, y1, x2, y2 = cluster.bounds

            if pole_type_lower == "small":
                positions = self._get_small_pole_positions(x1, y1, x2, y2)
            elif pole_type_lower == "medium":
                positions = self._get_medium_pole_positions(x1, y1, x2, y2)
            elif pole_type_lower == "big":
                positions = self._get_big_pole_positions(x1, y1, x2, y2)
            elif pole_type_lower == "substation":
                positions = self._get_substation_positions(x1, y1, x2, y2)
            else:
                # Fallback to corner placement
                positions = self._get_medium_pole_positions(x1, y1, x2, y2)

            # Place all poles for this cluster
            for pos in positions:
                self._place_pole(pos, prototype, footprint)

        self.diagnostics.info(
            f"Placed {len(self._planned)} {pole_type} power poles "
            f"for {len(self.clusters)} clusters"
        )

        return list(self._planned)

    # -------------------------------------------------------------------------
    # Position calculation methods (geometric formulas)
    # -------------------------------------------------------------------------

    def _get_small_pole_positions(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> List[Tuple[int, int]]:
        """Get positions for small poles (1x1 footprint, 8 poles per cluster).

        Places poles at 4 corners + 4 midpoints, all outside cluster bounds.
        """
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2

        return [
            # Corners (outside cluster by 1 tile)
            (x1 - 1, y1 - 1),  # Top-left
            (x2, y1 - 1),  # Top-right
            (x1 - 1, y2),  # Bottom-left
            (x2, y2),  # Bottom-right
            # Midpoints (outside cluster by 1 tile)
            (x1 - 1, mid_y),  # Mid-left
            (x2, mid_y),  # Mid-right
            (mid_x, y1 - 1),  # Mid-top
            (mid_x, y2),  # Mid-bottom
        ]

    def _get_medium_pole_positions(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> List[Tuple[int, int]]:
        """Get positions for medium poles (1x1 footprint, 4 poles per cluster).

        Places poles at 4 corners only, outside cluster bounds.
        """
        return [
            (x1 - 1, y1 - 1),  # Top-left
            (x2, y1 - 1),  # Top-right
            (x1 - 1, y2),  # Bottom-left
            (x2, y2),  # Bottom-right
        ]

    def _get_big_pole_positions(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> List[Tuple[int, int]]:
        """Get positions for big poles (2x2 footprint, 8 poles per cluster).

        Places poles at 4 corners + 4 midpoints, with extra space for 2x2 footprint.
        Positions are for top-left corner of the 2x2 footprint.
        """
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2

        return [
            # Corners (outside cluster by 2 tiles for 2x2 footprint)
            (x1 - 2, y1 - 2),  # Top-left
            (x2, y1 - 2),  # Top-right
            (x1 - 2, y2),  # Bottom-left
            (x2, y2),  # Bottom-right
            # Midpoints (outside cluster by 2 tiles)
            (x1 - 2, mid_y),  # Mid-left
            (x2, mid_y),  # Mid-right
            (mid_x, y1 - 2),  # Mid-top
            (mid_x, y2),  # Mid-bottom
        ]

    def _get_substation_positions(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> List[Tuple[int, int]]:
        """Get positions for substations (2x2 footprint, 1 pole per cluster).

        Places single pole at top-center, above cluster bounds.
        Position is for top-left corner of the 2x2 footprint.
        """
        mid_x = (x1 + x2) // 2

        return [
            # Top-center (-1 to center the 2x2 footprint, -2 to be outside cluster)
            (mid_x - 1, y1 - 2),
        ]

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
        # Check for duplicate at this exact position
        for existing in self._planned:
            if existing.position == position:
                return existing  # Return existing pole

        # Create new pole
        pole = PlannedPowerPole(position=position, prototype=prototype)
        self._planned.append(pole)

        # Add to layout plan
        pole_id = f"power_pole_{len(self.layout_plan.power_poles) + 1}"
        self.layout_plan.add_power_pole(
            PowerPolePlacement(
                pole_id=pole_id,
                pole_type=prototype,
                position=position,
            )
        )

        # Register with relay network so it can be reused for circuit routing
        if self.connection_planner and hasattr(
            self.connection_planner, "relay_network"
        ):
            self.connection_planner.relay_network.add_relay_node(
                position, pole_id, prototype
            )

        return pole
