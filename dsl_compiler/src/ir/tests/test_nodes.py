"""
Tests for ir/nodes.py - IR node classes.
"""

from dsl_compiler.src.ir.builder import SignalRef
from dsl_compiler.src.ir.nodes import (
    IRArith,
    IRConst,
    IRDecider,
    IREffect,
    IREntityPropRead,
    IREntityPropWrite,
    IRMemCreate,
    IRMemRead,
    IRMemWrite,
    IRNode,
    IRPlaceEntity,
    IRValue,
)


class TestSignalRef:
    """Tests for SignalRef class."""

    def test_signal_ref_creation(self):
        """Test SignalRef stores signal_type and source_id."""
        ref = SignalRef("iron-plate", "input_1")
        assert ref.signal_type == "iron-plate"
        assert ref.source_id == "input_1"

    def test_signal_ref_string_representation(self):
        """Test SignalRef string representation."""
        ref = SignalRef("iron-plate", "input_1")
        assert str(ref) == "iron-plate@input_1"


class TestIRConst:
    """Tests for IRConst node."""

    def test_ir_const_creation(self):
        """Test IRConst stores node_id and output_type."""
        node = IRConst("const_1", "signal-A")
        assert node.node_id == "const_1"
        assert node.output_type == "signal-A"

    def test_ir_const_value_assignment(self):
        """Test IRConst value can be assigned."""
        node = IRConst("const_1", "signal-A")
        node.value = 42
        assert node.value == 42


class TestIRArith:
    """Tests for IRArith node."""

    def test_ir_arith_creation(self):
        """Test IRArith stores node_id and output_type."""
        node = IRArith("arith_1", "signal-A")
        assert node.node_id == "arith_1"
        assert node.output_type == "signal-A"

    def test_ir_arith_operand_assignment(self):
        """Test IRArith operands can be assigned."""
        node = IRArith("arith_1", "signal-A")
        node.op = "+"
        node.left_operand = SignalRef("iron-plate", "input_1")
        node.right_operand = SignalRef("copper-plate", "input_2")
        assert node.op == "+"
        assert isinstance(node.left_operand, SignalRef)
        assert isinstance(node.right_operand, SignalRef)


class TestIRDecider:
    """Tests for IRDecider node."""

    def test_ir_decider_creation(self):
        """Test IRDecider stores node_id and output_type."""
        node = IRDecider("decider_1", "signal-A")
        assert node.node_id == "decider_1"
        assert node.output_type == "signal-A"

    def test_ir_decider_condition_assignment(self):
        """Test IRDecider condition properties can be assigned."""
        node = IRDecider("decider_1", "signal-A")
        node.condition_left = SignalRef("iron-plate", "input_1")
        node.condition_op = ">"
        node.condition_right = 10
        node.output_signal = "signal-A"
        node.output_value = 1
        assert node.condition_op == ">"
        assert node.condition_right == 10


class TestIRMemRead:
    """Tests for IRMemRead node."""

    def test_ir_mem_read_creation(self):
        """Test IRMemRead stores node_id and output_type."""
        node = IRMemRead("mem_read_1", "signal-A")
        assert node.node_id == "mem_read_1"
        assert node.output_type == "signal-A"

    def test_ir_mem_read_memory_id_assignment(self):
        """Test IRMemRead memory_id can be assigned."""
        node = IRMemRead("mem_read_1", "signal-A")
        node.memory_id = "mem_counter"
        assert node.memory_id == "mem_counter"


class TestIREntityPropRead:
    """Tests for IREntityPropRead node."""

    def test_ir_entity_prop_read_creation(self):
        """Test IREntityPropRead stores node_id and output_type."""
        node = IREntityPropRead("prop_read_1", "signal-A")
        assert node.node_id == "prop_read_1"
        assert node.output_type == "signal-A"

    def test_ir_entity_prop_read_assignment(self):
        """Test IREntityPropRead properties can be assigned."""
        node = IREntityPropRead("prop_read_1", "signal-A")
        node.entity_id = "lamp_1"
        node.property_name = "enabled"
        assert node.entity_id == "lamp_1"
        assert node.property_name == "enabled"


class TestIRMemCreate:
    """Tests for IRMemCreate node."""

    def test_ir_mem_create_creation(self):
        """Test IRMemCreate stores memory_id and signal_type."""
        node = IRMemCreate("mem_counter", "signal-A")
        assert node.memory_id == "mem_counter"
        assert node.signal_type == "signal-A"


class TestIRMemWrite:
    """Tests for IRMemWrite node."""

    def test_ir_mem_write_creation(self):
        """Test IRMemWrite stores memory_id, data_signal, and write_enable."""
        value = SignalRef("signal-A", "const_1")
        condition = SignalRef("signal-B", "const_2")
        node = IRMemWrite("mem_counter", value, condition)
        assert node.memory_id == "mem_counter"
        assert node.data_signal == value
        assert node.write_enable == condition


class TestIRPlaceEntity:
    """Tests for IRPlaceEntity node."""

    def test_ir_place_entity_creation(self):
        """Test IRPlaceEntity stores all entity properties."""
        properties = {"enabled": True}
        node = IRPlaceEntity("entity_1", "small-lamp", 5, 10, properties)
        assert node.entity_id == "entity_1"
        assert node.prototype == "small-lamp"
        assert node.x == 5
        assert node.y == 10
        assert node.properties == properties


class TestIREntityPropWrite:
    """Tests for IREntityPropWrite node."""

    def test_ir_entity_prop_write_creation(self):
        """Test IREntityPropWrite stores entity_id, property_name, and value."""
        value = SignalRef("signal-A", "const_1")
        node = IREntityPropWrite("entity_1", "enabled", value)
        assert node.entity_id == "entity_1"
        assert node.property_name == "enabled"
        assert node.value == value


class TestIRNodeHierarchy:
    """Tests for IR node inheritance hierarchy."""

    def test_ir_value_nodes_inherit_from_ir_value(self):
        """All IRValue types should inherit from IRValue and IRNode."""
        values = [
            IRConst("const_1", "signal-A"),
            IRArith("arith_1", "signal-A"),
            IRDecider("decider_1", "signal-A"),
            IRMemRead("mem_read_1", "signal-A"),
            IREntityPropRead("prop_read_1", "signal-A"),
        ]

        for value in values:
            assert isinstance(value, IRValue)
            assert isinstance(value, IRNode)

    def test_ir_effect_nodes_inherit_from_ir_effect(self):
        """All IREffect types should inherit from IREffect and IRNode."""
        effects = [
            IRMemCreate("mem", "signal-A", 0),
            IRMemWrite("mem", SignalRef("signal-A", "const_1"), 1),
            IRPlaceEntity("entity_1", "small-lamp", 0, 0, {}),
            IREntityPropWrite("entity_1", "enabled", SignalRef("signal-A", "const_1")),
        ]

        for effect in effects:
            assert isinstance(effect, IREffect)
            assert isinstance(effect, IRNode)
