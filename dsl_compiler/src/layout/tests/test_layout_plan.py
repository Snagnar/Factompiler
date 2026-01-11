"""
Tests for layout/layout_plan.py - Layout plan data structures.
"""

from dsl_compiler.src.layout.layout_plan import (
    EntityPlacement,
    LayoutPlan,
    PowerPolePlacement,
    WireConnection,
)


class TestEntityPlacement:
    """Tests for EntityPlacement dataclass."""

    def test_entity_placement_creation(self):
        """Test EntityPlacement stores required fields."""
        placement = EntityPlacement(
            ir_node_id="test_op_1",
            entity_type="arithmetic-combinator",
        )
        assert placement.ir_node_id == "test_op_1"
        assert placement.entity_type == "arithmetic-combinator"
        assert placement.position is None
        assert placement.properties == {}
        assert placement.role is None

    def test_entity_placement_with_position(self):
        """Test EntityPlacement with position."""
        placement = EntityPlacement(
            ir_node_id="test_op_1",
            entity_type="decider-combinator",
            position=(5.0, 10.0),
        )
        assert placement.position == (5.0, 10.0)

    def test_entity_placement_with_properties(self):
        """Test EntityPlacement with custom properties."""
        placement = EntityPlacement(
            ir_node_id="test_op_1",
            entity_type="constant-combinator",
            properties={"footprint": (1, 2), "operation": "+"},
            role="combinator",
        )
        assert placement.properties["footprint"] == (1, 2)
        assert placement.properties["operation"] == "+"
        assert placement.role == "combinator"


class TestWireConnection:
    """Tests for WireConnection dataclass."""

    def test_wire_connection_creation(self):
        """Test WireConnection stores all fields."""
        conn = WireConnection(
            source_entity_id="op_1",
            sink_entity_id="op_2",
            signal_name="signal-A",
            wire_color="red",
        )
        assert conn.source_entity_id == "op_1"
        assert conn.sink_entity_id == "op_2"
        assert conn.signal_name == "signal-A"
        assert conn.wire_color == "red"
        assert conn.source_side is None
        assert conn.sink_side is None

    def test_wire_connection_with_sides(self):
        """Test WireConnection with input/output sides."""
        conn = WireConnection(
            source_entity_id="op_1",
            sink_entity_id="op_2",
            signal_name="signal-B",
            wire_color="green",
            source_side="output",
            sink_side="input",
        )
        assert conn.source_side == "output"
        assert conn.sink_side == "input"


class TestPowerPolePlacement:
    """Tests for PowerPolePlacement dataclass."""

    def test_power_pole_placement_creation(self):
        """Test PowerPolePlacement stores fields."""
        pole = PowerPolePlacement(
            pole_id="pole_1",
            pole_type="medium-electric-pole",
            position=(10, 20),
        )
        assert pole.pole_id == "pole_1"
        assert pole.pole_type == "medium-electric-pole"
        assert pole.position == (10, 20)


class TestLayoutPlan:
    """Tests for LayoutPlan class."""

    def test_layout_plan_creation(self):
        """Test LayoutPlan initializes with empty collections."""
        plan = LayoutPlan()
        assert plan.entity_placements == {}
        assert plan.wire_connections == []
        assert plan.power_poles == []

    def test_add_placement(self):
        """Test adding entity placements."""
        plan = LayoutPlan()
        placement = EntityPlacement(
            ir_node_id="op_1",
            entity_type="arithmetic-combinator",
        )
        plan.add_placement(placement)
        assert "op_1" in plan.entity_placements
        assert plan.entity_placements["op_1"] is placement

    def test_get_placement_found(self):
        """Test getting an existing placement."""
        plan = LayoutPlan()
        placement = EntityPlacement(
            ir_node_id="op_1",
            entity_type="arithmetic-combinator",
        )
        plan.add_placement(placement)

        result = plan.get_placement("op_1")
        assert result is placement

    def test_get_placement_not_found(self):
        """Test getting a non-existent placement returns None."""
        plan = LayoutPlan()
        result = plan.get_placement("nonexistent")
        assert result is None

    def test_add_wire_connection(self):
        """Test adding wire connections."""
        plan = LayoutPlan()
        conn = WireConnection(
            source_entity_id="op_1",
            sink_entity_id="op_2",
            signal_name="signal-A",
            wire_color="red",
        )
        plan.add_wire_connection(conn)
        assert len(plan.wire_connections) == 1
        assert plan.wire_connections[0] is conn

    def test_add_power_pole(self):
        """Test adding power poles."""
        plan = LayoutPlan()
        pole = PowerPolePlacement(
            pole_id="pole_1",
            pole_type="medium-electric-pole",
            position=(10, 20),
        )
        plan.add_power_pole(pole)
        assert len(plan.power_poles) == 1
        assert plan.power_poles[0] is pole

    def test_create_and_add_placement(self):
        """Test helper method for creating and adding placements."""
        plan = LayoutPlan()
        placement = plan.create_and_add_placement(
            ir_node_id="op_1",
            entity_type="arithmetic-combinator",
            position=(5.0, 10.0),
            footprint=(2, 1),
            role="combinator",
            debug_info={"source": "test"},
            custom_prop="value",
        )

        assert placement.ir_node_id == "op_1"
        assert placement.entity_type == "arithmetic-combinator"
        assert placement.position == (5.0, 10.0)
        assert placement.role == "combinator"
        assert placement.properties["footprint"] == (2, 1)
        assert placement.properties["debug_info"]["source"] == "test"
        assert placement.properties["custom_prop"] == "value"

        # Check it was added to the plan
        assert "op_1" in plan.entity_placements

    def test_create_and_add_placement_defaults(self):
        """Test create_and_add_placement with default values."""
        plan = LayoutPlan()
        placement = plan.create_and_add_placement(
            ir_node_id="op_1",
            entity_type="constant-combinator",
        )

        assert placement.position is None
        assert placement.role == "combinator"
        assert placement.properties["footprint"] == (1, 1)
        assert placement.properties["debug_info"] == {}
