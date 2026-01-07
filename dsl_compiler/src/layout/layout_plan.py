from dataclasses import dataclass, field
from typing import Any

from dsl_compiler.src.common.constants import DEFAULT_CONFIG

"""Data structures for physical layout planning."""


@dataclass
class EntityPlacement:
    """Physical placement of an IR entity."""

    ir_node_id: str
    entity_type: str  # draftsman entity type
    position: tuple[float, float] | None = None  # None until positioned by force-directed layout
    properties: dict[str, Any] = field(default_factory=dict)
    role: str | None = None


@dataclass
class WireConnection:
    """A physical wire connection."""

    source_entity_id: str
    sink_entity_id: str
    signal_name: str
    wire_color: str  # "red" or "green"
    source_side: str | None = None  # "input" or "output" for dual-sided
    sink_side: str | None = None


@dataclass
class PowerPolePlacement:
    """Physical power pole placement."""

    pole_id: str
    pole_type: str
    position: tuple[int, int]


@dataclass
class LayoutPlan:
    """Complete physical layout plan for blueprint emission."""

    entity_placements: dict[str, EntityPlacement] = field(default_factory=dict)

    wire_connections: list[WireConnection] = field(default_factory=list)

    power_poles: list[PowerPolePlacement] = field(default_factory=list)

    blueprint_label: str = field(default_factory=lambda: DEFAULT_CONFIG.default_blueprint_label)
    blueprint_description: str = field(
        default_factory=lambda: DEFAULT_CONFIG.default_blueprint_description
    )

    def get_placement(self, ir_node_id: str) -> EntityPlacement | None:
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
        position: tuple[float, float] | None = None,
        footprint: tuple[int, int] = (1, 1),
        role: str = "combinator",
        debug_info: dict[str, Any] | None = None,
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
