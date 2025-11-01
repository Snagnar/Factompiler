"""Layout Planning Module
=========================

This package orchestrates the physical layout planning pipeline for the
Factorio Circuit DSL. It is responsible for:

1. Signal analysis – determining which logical signals require materialisation.
2. Entity placement – assigning tile coordinates to all blueprint entities.
3. Connection planning – routing circuit wires and choosing wire colours.
4. Power planning – inserting power poles to guarantee coverage and relays for
   long-span circuit links.

The resulting :class:`LayoutPlan` is consumed by the emission package to
materialise a Factorio blueprint.
"""

from .layout_engine import LayoutEngine
from .layout_plan import (
	LayoutPlan,
	EntityPlacement,
	WireConnection,
	PowerPolePlacement,
	SignalMaterialization,
)
from .signal_analyzer import (
	SignalAnalyzer,
	SignalMaterializer,
	SignalUsageEntry,
)
from .wire_router import (
	WIRE_COLORS,
	CircuitEdge,
	plan_wire_colors,
	collect_circuit_edges,
	detect_multi_source_conflicts,
	ColoringResult,
	ConflictEdge,
)
from .connection_planner import ConnectionPlanner
from .planner import LayoutPlanner
from .power_planner import PowerPlanner, PlannedPowerPole, POWER_POLE_CONFIG
from .signal_resolver import SignalResolver
from .signal_graph import SignalGraph
from .entity_placer import EntityPlacer

__all__ = [
    # Core planning
    "LayoutPlanner",
    "LayoutPlan",
    
    # Data structures
    "EntityPlacement",
    "WireConnection",
    "PowerPolePlacement",
    "SignalMaterialization",
    
    # Subsystems (for advanced use)
    "LayoutEngine",
    "EntityPlacer",
    "SignalAnalyzer",
    "SignalMaterializer",
    "SignalUsageEntry",
    "ConnectionPlanner",
    "PowerPlanner",
    "PlannedPowerPole",
    "POWER_POLE_CONFIG",
    "SignalResolver",
    "SignalGraph",
    
    # Wire routing
    "WIRE_COLORS",
    "CircuitEdge",
    "plan_wire_colors",
    "collect_circuit_edges",
    "detect_multi_source_conflicts",
    "ColoringResult",
    "ConflictEdge",
]
