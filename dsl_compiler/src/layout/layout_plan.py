from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from dsl_compiler.src.common.constants import DEFAULT_CONFIG

"""Data structures for physical layout planning."""


@dataclass
class EntityPlacement:
    """Physical placement of an IR entity."""

    ir_node_id: str
    entity_type: str  # draftsman entity type
    position: Optional[Tuple[float, float]] = (
        None  # None until positioned by force-directed layout
    )
    properties: Dict[str, Any] = field(default_factory=dict)
    role: Optional[str] = None


@dataclass
class WireConnection:
    """A physical wire connection."""

    source_entity_id: str
    sink_entity_id: str
    signal_name: str
    wire_color: str  # "red" or "green"
    source_side: Optional[str] = None  # "input" or "output" for dual-sided
    sink_side: Optional[str] = None


@dataclass
class PowerPolePlacement:
    """Physical power pole placement."""

    pole_id: str
    pole_type: str
    position: Tuple[int, int]


@dataclass
class LayoutPlan:
    """Complete physical layout plan for blueprint emission."""

    entity_placements: Dict[str, EntityPlacement] = field(default_factory=dict)

    wire_connections: List[WireConnection] = field(default_factory=list)

    power_poles: List[PowerPolePlacement] = field(default_factory=list)

    blueprint_label: str = field(
        default_factory=lambda: DEFAULT_CONFIG.default_blueprint_label
    )
    blueprint_description: str = field(
        default_factory=lambda: DEFAULT_CONFIG.default_blueprint_description
    )

    def get_placement(self, ir_node_id: str) -> Optional[EntityPlacement]:
        """Get placement for an IR node."""

        return self.entity_placements.get(ir_node_id)

    def add_placement(self, placement: EntityPlacement) -> None:
        """Add an entity placement."""

        self.entity_placements[placement.ir_node_id] = placement

    def add_wire_connection(self, connection: WireConnection) -> None:
        """Add a wire connection."""

        self.wire_connections.append(connection)

    def add_power_pole(self, pole: PowerPolePlacement) -> None:
        """Add a power pole."""

        self.power_poles.append(pole)

    def create_and_add_placement(
        self,
        ir_node_id: str,
        entity_type: str,
        position: Optional[Tuple[float, float]] = None,
        footprint: Tuple[int, int] = (1, 1),
        role: str = "combinator",
        debug_info: Optional[Dict[str, Any]] = None,
        **extra_properties: Any,
    ) -> EntityPlacement:
        """Create an EntityPlacement and add it to the plan.

        Args:
            ir_node_id: Unique identifier for this entity
            entity_type: Draftsman entity type (e.g., 'arithmetic-combinator')
            position: Optional (x, y) position (None until layout)
            footprint: (width, height) in tiles
            role: Entity role (e.g., 'combinator', 'relay', 'power_pole')
            debug_info: Debug information dictionary
            **extra_properties: Additional properties to include

        Returns:
            The created EntityPlacement
        """
        placement = EntityPlacement(
            ir_node_id=ir_node_id,
            entity_type=entity_type,
            position=position,
            properties={
                "footprint": footprint,
                "debug_info": debug_info or {},
                **extra_properties,
            },
            role=role,
        )
        self.add_placement(placement)
        return placement
