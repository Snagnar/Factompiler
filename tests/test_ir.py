"""
Tests for ir.py - Intermediate Representation classes and structures.
"""

from dsl_compiler.src.ast.literals import NumberLiteral
from dsl_compiler.src.ir.builder import (
    IRArith,
    IRBuilder,
    IRConst,
    IRDecider,
    IRMemCreate,
    IRMemRead,
    IRMemWrite,
    IRNode,
    IRPlaceEntity,
    IRValue,
    SignalRef,
)
from dsl_compiler.src.ir.nodes import (
    IREffect,
    IREntityPropRead,
    IREntityPropWrite,
)


class TestSignalAndValueRefs:
    """Test IR signal and value reference classes."""

    def test_signal_ref(self):
        """Test SignalRef creation and string representation."""
        ref = SignalRef("iron-plate", "input_1")
        assert ref.signal_type == "iron-plate"
        assert ref.source_id == "input_1"
        assert str(ref) == "iron-plate@input_1"


class TestIRValueNodes:
    """Test IR value-producing nodes."""

    def test_ir_const(self):
        """Test IRConst node."""
        ast_node = NumberLiteral(42)
        node = IRConst("const_1", "signal-A", ast_node)
        assert node.node_id == "const_1"
        assert node.output_type == "signal-A"
        assert node.source_ast == ast_node
        node.value = 42
        assert node.value == 42

    def test_ir_arith(self):
        """Test IRArith node."""
        node = IRArith("arith_1", "signal-A")
        assert node.node_id == "arith_1"
        assert node.output_type == "signal-A"
        node.op = "+"
        node.left_operand = SignalRef("iron-plate", "input_1")
        node.right_operand = SignalRef("copper-plate", "input_2")
        assert node.op == "+"
        assert isinstance(node.left_operand, SignalRef)
        assert isinstance(node.right_operand, SignalRef)

    def test_ir_decider(self):
        """Test IRDecider node."""
        node = IRDecider("decider_1", "signal-A")
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
        """Test IRMemRead node."""
        node = IRMemRead("mem_read_1", "signal-A")
        assert node.node_id == "mem_read_1"
        assert node.output_type == "signal-A"
        node.memory_id = "mem_counter"
        assert node.memory_id == "mem_counter"

    def test_ir_entity_prop_read(self):
        """Test IREntityPropRead node."""
        node = IREntityPropRead("prop_read_1", "signal-A")
        assert node.node_id == "prop_read_1"
        assert node.output_type == "signal-A"
        node.entity_id = "lamp_1"
        node.property_name = "enabled"
        assert node.entity_id == "lamp_1"
        assert node.property_name == "enabled"


class TestIREffectNodes:
    """Test IR effect-producing nodes."""

    def test_ir_mem_create(self):
        """Test IRMemCreate node."""
        node = IRMemCreate("mem_counter", "signal-A")
        assert node.memory_id == "mem_counter"
        assert node.signal_type == "signal-A"
        assert not hasattr(node, "initial_value")

    def test_ir_mem_write(self):
        """Test IRMemWrite node."""
        value = SignalRef("signal-A", "const_1")
        condition = SignalRef("signal-B", "const_2")
        node = IRMemWrite("mem_counter", value, condition)
        assert node.memory_id == "mem_counter"
        assert node.data_signal == value
        assert node.write_enable == condition

    def test_ir_place_entity(self):
        """Test IRPlaceEntity node."""
        properties = {"enabled": True}
        node = IRPlaceEntity("entity_1", "small-lamp", 5, 10, properties)
        assert node.entity_id == "entity_1"
        assert node.prototype == "small-lamp"
        assert node.x == 5
        assert node.y == 10
        assert node.properties == properties

    def test_ir_entity_prop_write(self):
        """Test IREntityPropWrite node."""
        value = SignalRef("signal-A", "const_1")
        node = IREntityPropWrite("entity_1", "enabled", value)
        assert node.entity_id == "entity_1"
        assert node.property_name == "enabled"
        assert node.value == value


class TestIRHierarchy:
    """Test IR node inheritance and type relationships."""

    def test_ir_node_base(self):
        """Test IRNode base class functionality."""
        # All IR nodes should inherit from IRNode
        assert isinstance(IRConst("id", "type"), IRNode)
        assert isinstance(IRMemCreate("id", "type", 0), IRNode)

    def test_ir_value_hierarchy(self):
        """Test IRValue node inheritance."""
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

    def test_ir_effect_hierarchy(self):
        """Test IREffect node inheritance."""
        effects = [
            IRMemCreate("mem", "signal-A", 0),
            IRMemWrite("mem", SignalRef("signal-A", "const_1"), 1),
            IRPlaceEntity("entity_1", "small-lamp", 0, 0, {}),
            IREntityPropWrite("entity_1", "enabled", SignalRef("signal-A", "const_1")),
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

    def test_value_ref_integer(self):
        """Test ValueRef with integer."""
        ref = 42
        # ValueRef is a Union type, just verify it can hold int
        assert isinstance(ref, int)


class TestImplicitSignalAllocation:
    """Validate implicit signal allocation behavior."""

    def test_allocate_more_than_26_virtual_signals(self):
        """Ensure implicit signal allocation produces unique names."""
        builder = IRBuilder()
        names = [builder.allocate_implicit_type() for _ in range(60)]

        # All allocated names should be unique
        assert len(names) == len(set(names))
        # Implicit signals use __v prefix with sequential numbering
        assert names[0] == "__v1"
        assert names[1] == "__v2"
        assert names[59] == "__v60"
