"""
Tests for ir/nodes.py - IR node classes.
"""

from dsl_compiler.src.ir.builder import SignalRef
from dsl_compiler.src.ir.nodes import (
    MEMORY_TYPE_RS_LATCH,
    MEMORY_TYPE_SR_LATCH,
    BundleRef,
    IRArith,
    IRConst,
    IRDecider,
    IREffect,
    IREntityOutput,
    IREntityPropRead,
    IREntityPropWrite,
    IRLatchWrite,
    IRMemCreate,
    IRMemRead,
    IRMemWrite,
    IRNode,
    IRPlaceEntity,
    IRValue,
    IRWireMerge,
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


class TestBundleRef:
    """Tests for BundleRef class."""

    def test_bundle_ref_creation(self):
        """Test BundleRef stores signal_types and source_id."""
        signal_types = {"signal-A", "signal-B", "signal-C"}
        ref = BundleRef(signal_types, "bundle_src_1")
        assert ref.signal_types == signal_types
        assert ref.source_id == "bundle_src_1"

    def test_bundle_ref_optional_params(self):
        """Test BundleRef optional parameters."""
        signal_types = {"signal-X"}
        ref = BundleRef(
            signal_types,
            "src",
            debug_label="test label",
            metadata={"key": "value"},
        )
        assert ref.debug_label == "test label"
        assert ref.debug_metadata == {"key": "value"}

    def test_bundle_ref_metadata_copy(self):
        """Test BundleRef copies metadata dict."""
        original = {"key": "value"}
        ref = BundleRef({"signal-A"}, "src", metadata=original)
        original["key"] = "modified"
        assert ref.debug_metadata["key"] == "value"  # Should not be modified


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


class TestIREntityOutput:
    """Tests for IREntityOutput node."""

    def test_ir_entity_output_creation(self):
        """Test IREntityOutput stores entity_id."""
        node = IREntityOutput("entity_out_1", "chest_1")
        assert node.node_id == "entity_out_1"
        assert node.entity_id == "chest_1"
        assert node.output_type == "bundle"

    def test_ir_entity_output_inherits_from_ir_value(self):
        """IREntityOutput should inherit from IRValue."""
        node = IREntityOutput("entity_out_1", "chest_1")
        assert isinstance(node, IRValue)
        assert isinstance(node, IRNode)


class TestIRLatchWrite:
    """Tests for IRLatchWrite node."""

    def test_ir_latch_write_rs_creation(self):
        """Test IRLatchWrite RS latch stores all fields."""
        value = SignalRef("signal-A", "const_1")
        set_signal = SignalRef("signal-B", "set_src")
        reset_signal = SignalRef("signal-C", "reset_src")
        node = IRLatchWrite("mem_1", value, set_signal, reset_signal, MEMORY_TYPE_RS_LATCH)
        assert node.memory_id == "mem_1"
        assert node.value == value
        assert node.set_signal == set_signal
        assert node.reset_signal == reset_signal
        assert node.latch_type == MEMORY_TYPE_RS_LATCH

    def test_ir_latch_write_sr_creation(self):
        """Test IRLatchWrite SR latch stores all fields."""
        value = SignalRef("signal-A", "const_1")
        set_signal = SignalRef("signal-B", "set_src")
        reset_signal = SignalRef("signal-C", "reset_src")
        node = IRLatchWrite("mem_2", value, set_signal, reset_signal, MEMORY_TYPE_SR_LATCH)
        assert node.latch_type == MEMORY_TYPE_SR_LATCH

    def test_ir_latch_write_inherits_from_ir_effect(self):
        """IRLatchWrite should inherit from IREffect."""
        value = SignalRef("signal-A", "const_1")
        set_signal = SignalRef("signal-B", "set_src")
        reset_signal = SignalRef("signal-C", "reset_src")
        node = IRLatchWrite("mem_1", value, set_signal, reset_signal, MEMORY_TYPE_RS_LATCH)
        assert isinstance(node, IREffect)
        assert isinstance(node, IRNode)


class TestIRWireMerge:
    """Tests for IRWireMerge node."""

    def test_ir_wire_merge_creation(self):
        """Test IRWireMerge stores node_id and output_type."""
        node = IRWireMerge("merge_1", "signal-A")
        assert node.node_id == "merge_1"
        assert node.output_type == "signal-A"
        assert node.sources == []

    def test_ir_wire_merge_add_source(self):
        """Test IRWireMerge can add sources."""
        node = IRWireMerge("merge_1", "signal-A")
        ref1 = SignalRef("signal-A", "src_1")
        ref2 = SignalRef("signal-A", "src_2")
        node.add_source(ref1)
        node.add_source(ref2)
        assert len(node.sources) == 2
        assert ref1 in node.sources
        assert ref2 in node.sources

    def test_ir_wire_merge_inherits_from_ir_value(self):
        """IRWireMerge should inherit from IRValue."""
        node = IRWireMerge("merge_1", "signal-A")
        assert isinstance(node, IRValue)
        assert isinstance(node, IRNode)
