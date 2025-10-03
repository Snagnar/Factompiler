"""
Tests for ir.py - Intermediate Representation classes and structures.
"""

import pytest
from dsl_compiler.src.ir import *
from dsl_compiler.src.dsl_ast import NumberLiteral, StringLiteral


class TestSignalAndValueRefs:
    """Test IR signal and value reference classes."""

    def test_signal_ref(self):
        """Test SignalRef creation and string representation."""
        ref = SignalRef("iron-plate", "input_1")
        assert ref.signal_type == "iron-plate"
        assert ref.source_id == "input_1"
        assert str(ref) == "iron-plate@input_1"

    def test_bundle_ref(self):
        """Test BundleRef creation and string representation."""
        channels = {
            "x": SignalRef("signal-A", "const_1"),
            "y": SignalRef("signal-B", "const_2")
        }
        ref = BundleRef(channels, "bundle_1")
        assert ref.channels == channels
        assert ref.bundle_id == "bundle_1"
        # String representation shows channel keys, not signal types
        assert "x" in str(ref)
        assert "y" in str(ref)


class TestIRValueNodes:
    """Test IR value-producing nodes."""

    def test_ir_const(self):
        """Test IR_Const node."""
        ast_node = NumberLiteral(42)
        node = IR_Const("const_1", "signal-A", ast_node)
        assert node.node_id == "const_1"
        assert node.output_type == "signal-A"
        assert node.source_ast == ast_node
        node.value = 42
        assert node.value == 42

    def test_ir_arith(self):
        """Test IR_Arith node."""
        node = IR_Arith("arith_1", "signal-A")
        assert node.node_id == "arith_1"
        assert node.output_type == "signal-A"
        node.op = "+"
        node.left_operand = SignalRef("iron-plate", "input_1")
        node.right_operand = SignalRef("copper-plate", "input_2")
        assert node.op == "+"
        assert isinstance(node.left_operand, SignalRef)
        assert isinstance(node.right_operand, SignalRef)

    def test_ir_decider(self):
        """Test IR_Decider node."""
        node = IR_Decider("decider_1", "signal-A")
        assert node.node_id == "decider_1"
        assert node.output_type == "signal-A"
        node.condition_left = SignalRef("iron-plate", "input_1")
        node.condition_op = ">"
        node.condition_right = 10
        node.output_signal = "signal-A"
        node.output_value = 1
        assert node.condition_op == ">"
        assert node.condition_right == 10

    def test_ir_mem_read(self):
        """Test IR_MemRead node."""
        node = IR_MemRead("mem_read_1", "signal-A")
        assert node.node_id == "mem_read_1"
        assert node.output_type == "signal-A"
        node.memory_id = "mem_counter"
        assert node.memory_id == "mem_counter"

    def test_ir_entity_prop_read(self):
        """Test IR_EntityPropRead node."""
        node = IR_EntityPropRead("prop_read_1", "signal-A")
        assert node.node_id == "prop_read_1"
        assert node.output_type == "signal-A"
        node.entity_id = "lamp_1"
        node.property_name = "enabled"
        assert node.entity_id == "lamp_1"
        assert node.property_name == "enabled"

    def test_ir_bundle(self):
        """Test IR_Bundle node."""
        components = {
            "x": SignalRef("signal-A", "const_1"),
            "y": SignalRef("signal-B", "const_2")
        }
        node = IR_Bundle("bundle_1", components)
        assert node.node_id == "bundle_1"
        assert node.inputs == components


class TestIREffectNodes:
    """Test IR effect-producing nodes."""

    def test_ir_mem_create(self):
        """Test IR_MemCreate node."""
        initial_value = SignalRef("signal-A", "const_1")
        node = IR_MemCreate("mem_counter", "signal-A", initial_value)
        assert node.memory_id == "mem_counter"
        assert node.signal_type == "signal-A"
        assert node.initial_value == initial_value

    def test_ir_mem_write(self):
        """Test IR_MemWrite node."""
        value = SignalRef("signal-A", "const_1")
        condition = SignalRef("signal-B", "const_2")
        node = IR_MemWrite("mem_counter", value, condition)
        assert node.memory_id == "mem_counter"
        assert node.data_signal == value
        assert node.write_enable == condition

    def test_ir_place_entity(self):
        """Test IR_PlaceEntity node."""
        properties = {"enabled": True}
        node = IR_PlaceEntity("entity_1", "small-lamp", 5, 10, properties)
        assert node.entity_id == "entity_1"
        assert node.prototype == "small-lamp"
        assert node.x == 5
        assert node.y == 10
        assert node.properties == properties

    def test_ir_entity_prop_write(self):
        """Test IR_EntityPropWrite node."""
        value = SignalRef("signal-A", "const_1")
        node = IR_EntityPropWrite("entity_1", "enabled", value)
        assert node.entity_id == "entity_1"
        assert node.property_name == "enabled"
        assert node.value == value

    def test_ir_connect_to_wire(self):
        """Test IR_ConnectToWire node."""
        signal = SignalRef("signal-A", "const_1")
        node = IR_ConnectToWire(signal, "red")
        assert node.signal == signal
        assert node.channel == "red"


class TestIRContainerNodes:
    """Test IR container nodes."""

    def test_ir_group(self):
        """Test IR_Group node."""
        children = [
            IR_Const("const_1", "signal-A"),
            IR_Const("const_2", "signal-B")
        ]
        node = IR_Group("group_1", children)
        assert node.node_id == "group_1"
        assert node.operations == children


class TestIRHierarchy:
    """Test IR node inheritance and type relationships."""

    def test_ir_node_base(self):
        """Test IRNode base class functionality."""
        # All IR nodes should inherit from IRNode
        assert isinstance(IR_Const("id", "type"), IRNode)
        assert isinstance(IR_MemCreate("id", "type", 0), IRNode)

    def test_ir_value_hierarchy(self):
        """Test IRValue node inheritance."""
        values = [
            IR_Const("const_1", "signal-A"),
            IR_Arith("arith_1", "signal-A"),
            IR_Decider("decider_1", "signal-A"),
            IR_MemRead("mem_read_1", "signal-A"),
            IR_EntityPropRead("prop_read_1", "signal-A"),
            IR_Bundle("bundle_1", {}),
        ]
        
        for value in values:
            assert isinstance(value, IRValue)
            assert isinstance(value, IRNode)

    def test_ir_effect_hierarchy(self):
        """Test IREffect node inheritance."""
        effects = [
            IR_MemCreate("mem", "signal-A", 0),
            IR_MemWrite("mem", SignalRef("signal-A", "const_1"), 1),
            IR_PlaceEntity("entity_1", "small-lamp", 0, 0, {}),
            IR_EntityPropWrite("entity_1", "enabled", SignalRef("signal-A", "const_1")),
            IR_ConnectToWire(SignalRef("signal-A", "const_1"), "red"),
        ]
        
        for effect in effects:
            assert isinstance(effect, IREffect)
            assert isinstance(effect, IRNode)


class TestValueRefTypes:
    """Test ValueRef type union functionality."""

    def test_value_ref_signal(self):
        """Test ValueRef with SignalRef."""
        ref = SignalRef("iron-plate", "input_1")
        # ValueRef is a Union type, just verify it can hold SignalRef
        assert isinstance(ref, SignalRef)

    def test_value_ref_bundle(self):
        """Test ValueRef with BundleRef."""
        channels = {"x": SignalRef("signal-A", "const_1")}
        ref = BundleRef(channels, "bundle_1")
        # ValueRef is a Union type, just verify it can hold BundleRef
        assert isinstance(ref, BundleRef)

    def test_value_ref_integer(self):
        """Test ValueRef with integer."""
        ref = 42
        # ValueRef is a Union type, just verify it can hold int
        assert isinstance(ref, int)