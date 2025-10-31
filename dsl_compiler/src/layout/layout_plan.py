"""Data structures for physical layout planning."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

from dsl_compiler.src.ir import SignalRef


@dataclass
class EntityPlacement:
    """Physical placement of an IR entity."""

    ir_node_id: str
    entity_type: str  # draftsman entity type
    position: Tuple[int, int]
    properties: Dict[str, Any] = field(default_factory=dict)
    role: Optional[str] = None
    zone: Optional[str] = None


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
class SignalMaterialization:
    """Decision about whether/how to materialize a signal."""

    signal_id: str
    should_materialize: bool
    resolved_signal_name: Optional[str] = None
    resolved_signal_type: Optional[str] = None
    is_inlinable_constant: bool = False
    constant_value: Optional[int] = None


@dataclass
class LayoutPlan:
    """Complete physical layout plan for blueprint emission."""

    # Entity placements
    entity_placements: Dict[str, EntityPlacement] = field(default_factory=dict)

    # Wire connections
    wire_connections: List[WireConnection] = field(default_factory=list)

    # Power infrastructure
    power_poles: List[PowerPolePlacement] = field(default_factory=list)

    # Signal decisions
    signal_materializations: Dict[str, SignalMaterialization] = field(default_factory=dict)

    # Signal connectivity graph (source -> sinks)
    signal_graph: Dict[str, List[str]] = field(default_factory=dict)

    # Metadata
    blueprint_label: str = "DSL Generated"
    blueprint_description: str = ""

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
