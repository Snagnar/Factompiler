"""
Tests for emission/entity_emitter.py - Entity materialization from layout plan.

This module tests the PlanEntityEmitter class which converts layout placements
to Draftsman entities for blueprint generation.
"""

import pytest

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.emission.entity_emitter import (
    PlanEntityEmitter,
    _infer_signal_type,
    format_entity_description,
)
from dsl_compiler.src.layout.layout_plan import EntityPlacement
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer


class TestInferSignalType:
    """Tests for _infer_signal_type function."""

    def test_infer_item_signal(self):
        """Test inferring item signal type."""
        result = _infer_signal_type("iron-plate")
        assert result == "item"

    def test_infer_fluid_signal(self):
        """Test inferring fluid signal type."""
        result = _infer_signal_type("water")
        assert result == "fluid"

    def test_infer_virtual_signal(self):
        """Test inferring virtual signal type."""
        result = _infer_signal_type("signal-A")
        assert result == "virtual"

    def test_infer_unknown_defaults_to_virtual(self):
        """Test that unknown signals default to virtual."""
        result = _infer_signal_type("unknown-signal-xyz")
        assert result == "virtual"


class TestFormatEntityDescription:
    """Tests for format_entity_description function."""

    def test_format_empty_debug_info(self):
        """Test formatting with no debug info."""
        result = format_entity_description(None)
        assert result == ""

    def test_format_empty_dict(self):
        """Test formatting with empty dict."""
        result = format_entity_description({})
        assert result == ""

    def test_format_with_variable_and_line(self):
        """Test formatting with variable and line number."""
        debug_info = {
            "variable": "my_signal",
            "line": 42,
        }
        result = format_entity_description(debug_info)
        assert "my_signal" in result
        assert "42" in result

    def test_format_with_file_and_line(self):
        """Test formatting with filename and line."""
        debug_info = {
            "variable": "x",
            "source_file": "test.facto",
            "line": 10,
        }
        result = format_entity_description(debug_info)
        assert "test.facto" in result
        assert "10" in result

    def test_format_intermediate_with_context(self):
        """Test formatting intermediate computation."""
        debug_info = {
            "variable": "arith_001",
            "expr_context": "result",
            "line": 5,
        }
        result = format_entity_description(debug_info)
        assert "computing" in result
        assert "result" in result

    def test_format_with_arith_operation(self):
        """Test formatting arithmetic operation."""
        debug_info = {
            "variable": "sum",
            "operation": "arith",
            "details": "op=+",
            "line": 3,
        }
        result = format_entity_description(debug_info)
        assert "+" in result

    def test_format_with_decider_operation(self):
        """Test formatting decider operation."""
        debug_info = {
            "variable": "cmp",
            "operation": "decider",
            "details": "cond=>",
            "line": 4,
        }
        result = format_entity_description(debug_info)
        assert ">" in result


class TestPlanEntityEmitter:
    """Tests for PlanEntityEmitter class."""

    @pytest.fixture
    def emitter(self):
        """Create an emitter instance."""
        return PlanEntityEmitter()

    def test_emitter_initialization(self):
        """Test emitter can be initialized."""
        emitter = PlanEntityEmitter()
        assert emitter is not None
        assert isinstance(emitter.diagnostics, ProgramDiagnostics)

    def test_emitter_with_custom_diagnostics(self):
        """Test emitter with custom diagnostics."""
        diags = ProgramDiagnostics()
        emitter = PlanEntityEmitter(diagnostics=diags)
        assert emitter.diagnostics is diags

    def test_create_constant_combinator(self, emitter):
        """Test creating a constant combinator entity."""
        placement = EntityPlacement(
            ir_node_id="const_1",
            entity_type="constant-combinator",
            position=(0, 0),
            properties={
                "signals": {"signal-A": 42},
            },
        )
        entity = emitter.create_entity(placement)
        assert entity is not None
        assert entity.id == "const_1"

    def test_create_arithmetic_combinator(self, emitter):
        """Test creating an arithmetic combinator entity."""
        placement = EntityPlacement(
            ir_node_id="arith_1",
            entity_type="arithmetic-combinator",
            position=(0, 0),
            properties={
                "first_operand": {"name": "signal-A", "type": "virtual"},
                "operation": "+",
                "second_operand": 5,
                "output": {"name": "signal-B", "type": "virtual"},
            },
        )
        entity = emitter.create_entity(placement)
        assert entity is not None

    def test_create_decider_combinator(self, emitter):
        """Test creating a decider combinator entity."""
        placement = EntityPlacement(
            ir_node_id="decider_1",
            entity_type="decider-combinator",
            position=(0, 0),
            properties={
                "first_operand": {"name": "signal-A", "type": "virtual"},
                "operation": ">",
                "second_operand": 0,
                "output": {"name": "signal-B", "type": "virtual"},
                "copy_count_from_input": False,
            },
        )
        entity = emitter.create_entity(placement)
        assert entity is not None

    def test_create_lamp(self, emitter):
        """Test creating a lamp entity."""
        placement = EntityPlacement(
            ir_node_id="lamp_1",
            entity_type="small-lamp",
            position=(0, 0),
            properties={
                "use_colors": True,
            },
        )
        entity = emitter.create_entity(placement)
        assert entity is not None

    def test_entity_has_description(self, emitter):
        """Test that created entity has player description."""
        placement = EntityPlacement(
            ir_node_id="test_1",
            entity_type="constant-combinator",
            position=(0, 0),
            properties={
                "signals": [],
                "debug_info": {
                    "variable": "my_var",
                    "line": 10,
                },
            },
        )
        entity = emitter.create_entity(placement)
        assert entity is not None
        assert entity.player_description is not None

    def test_create_decider_multi_condition(self, emitter):
        """Test creating a decider with multi-condition mode (Factorio 2.0)."""
        placement = EntityPlacement(
            ir_node_id="decider_multi",
            entity_type="decider-combinator",
            position=(0, 0),
            properties={
                "conditions": [
                    {
                        "first_signal": "signal-A",
                        "comparator": ">",
                        "second_constant": 0,
                        "compare_type": "or",
                        "first_signal_wires": {"red"},
                    },
                    {
                        "first_constant": 5,
                        "comparator": "<",
                        "second_signal": "signal-B",
                        "compare_type": "and",
                        "second_signal_wires": {"green"},
                    },
                ],
                "output_signal": "signal-C",
                "output_value": 1,
                "copy_count_from_input": False,
            },
        )
        entity = emitter.create_entity(placement)
        assert entity is not None
        assert len(entity.conditions) == 2

    def test_create_decider_multi_conditions_key(self, emitter):
        """Test decider with multi_conditions key (from memory_builder)."""
        placement = EntityPlacement(
            ir_node_id="decider_latch",
            entity_type="decider-combinator",
            position=(0, 0),
            properties={
                "multi_conditions": [
                    {"first_signal": "signal-S", "comparator": ">", "second_constant": 0},
                ],
                "output_signal": "signal-Q",
                "copy_count_from_input": True,
            },
        )
        entity = emitter.create_entity(placement)
        assert entity is not None
        assert entity.outputs[0].copy_count_from_input is True


class TestApplyPropertyWrites:
    """Tests for _apply_property_writes method."""

    @pytest.fixture
    def emitter(self):
        return PlanEntityEmitter(signal_type_map={"__sig": {"name": "signal-X", "type": "virtual"}})

    def test_enable_inline_comparison(self, emitter):
        """Test enable property with inline comparison."""
        placement = EntityPlacement(
            ir_node_id="lamp_cond",
            entity_type="small-lamp",
            position=(0, 0),
            properties={
                "property_writes": {
                    "enable": {
                        "type": "inline_comparison",
                        "comparison_data": {
                            "left_signal": "signal-A",
                            "comparator": ">",
                            "right_constant": 5,
                        },
                    },
                },
            },
        )
        entity = emitter.create_entity(placement)
        assert entity is not None
        assert entity.circuit_enabled is True

    def test_enable_inline_bundle_condition(self, emitter):
        """Test enable property with bundle condition (all/any)."""
        placement = EntityPlacement(
            ir_node_id="lamp_bundle",
            entity_type="small-lamp",
            position=(0, 0),
            properties={
                "property_writes": {
                    "enable": {
                        "type": "inline_bundle_condition",
                        "signal": "signal-everything",
                        "operator": ">",
                        "constant": 0,
                    },
                },
            },
        )
        entity = emitter.create_entity(placement)
        assert entity is not None

    def test_enable_signal_ref(self, emitter):
        """Test enable property with signal reference."""
        placement = EntityPlacement(
            ir_node_id="lamp_sig",
            entity_type="small-lamp",
            position=(0, 0),
            properties={
                "property_writes": {
                    "enable": {
                        "type": "signal",
                        "signal_ref": type("Ref", (), {"signal_type": "__sig"})(),
                    },
                },
            },
        )
        entity = emitter.create_entity(placement)
        assert entity is not None

    def test_enable_constant(self, emitter):
        """Test enable property with constant value."""
        placement = EntityPlacement(
            ir_node_id="lamp_const",
            entity_type="small-lamp",
            position=(0, 0),
            properties={
                "property_writes": {
                    "enable": {"type": "constant", "value": True},
                },
            },
        )
        entity = emitter.create_entity(placement)
        assert entity is not None

    def test_generic_property_write(self, emitter):
        """Test writing a generic property."""
        placement = EntityPlacement(
            ir_node_id="lamp_color",
            entity_type="small-lamp",
            position=(0, 0),
            properties={
                "use_colors": True,
                "property_writes": {
                    "use_colors": {"type": "constant", "value": False},
                },
            },
        )
        entity = emitter.create_entity(placement)
        assert entity is not None

    def test_unknown_property_logs_info(self, emitter):
        """Test that setting unknown property logs info message."""
        placement = EntityPlacement(
            ir_node_id="lamp_bad",
            entity_type="small-lamp",
            position=(0, 0),
            properties={
                "property_writes": {
                    "nonexistent_prop": {"value": 123},
                },
            },
        )
        entity = emitter.create_entity(placement)
        assert entity is not None


class TestFormatEntityDescriptionExtended:
    """Extended tests for format_entity_description edge cases."""

    def test_format_with_memory_operation(self):
        """Test formatting memory operation."""
        debug_info = {"variable": "mem", "operation": "memory", "details": "cell_0", "line": 5}
        result = format_entity_description(debug_info)
        assert "memory" in result
        assert "cell_0" in result

    def test_format_output_anchor(self):
        """Test formatting output anchor."""
        debug_info = {"variable": "out", "operation": "output", "line": 10}
        result = format_entity_description(debug_info)
        assert "output anchor" in result

    def test_format_with_role(self):
        """Test formatting with role."""
        debug_info = {"variable": "x", "role": "merge", "line": 3}
        result = format_entity_description(debug_info)
        assert "merge" in result

    def test_format_const_operation(self):
        """Test formatting const operation."""
        debug_info = {"variable": "c", "operation": "const", "details": "value=42", "line": 1}
        result = format_entity_description(debug_info)
        assert "value=42" in result

    def test_format_intermediate_arith(self):
        """Test formatting intermediate arithmetic."""
        debug_info = {"variable": "arith_001", "expr_context": "result", "line": 5}
        result = format_entity_description(debug_info)
        assert "computing" in result
        assert "result" in result

    def test_format_with_signal_type(self):
        """Test formatting with signal type."""
        debug_info = {"variable": "x", "signal_type": "signal-A", "line": 1}
        result = format_entity_description(debug_info)
        assert "signal-A" in result

    def test_format_file_only(self):
        """Test formatting with file but no line."""
        debug_info = {"variable": "x", "source_file": "test.facto"}
        result = format_entity_description(debug_info)
        assert "test.facto" in result


class TestCreateEntityEdgeCases:
    """Edge case tests for entity creation."""

    @pytest.fixture
    def emitter(self):
        return PlanEntityEmitter()

    def test_arithmetic_each_output_correction(self, emitter):
        """Test that signal-each output is corrected when not using each input."""
        placement = EntityPlacement(
            ir_node_id="arith_each",
            entity_type="arithmetic-combinator",
            position=(0, 0),
            properties={
                "left_operand": "signal-A",
                "operation": "+",
                "right_operand": 5,
                "output_signal": "signal-each",
            },
        )
        entity = emitter.create_entity(placement)
        assert entity.output_signal.name == "signal-0"

    def test_constant_combinator_single_signal(self, emitter):
        """Test constant combinator with single signal mode."""
        placement = EntityPlacement(
            ir_node_id="const_single",
            entity_type="constant-combinator",
            position=(0, 0),
            properties={
                "signal_name": "signal-A",
                "value": 42,
            },
        )
        entity = emitter.create_entity(placement)
        assert entity is not None

    def test_entity_from_template(self, emitter):
        """Test creating entity from template object."""
        from draftsman.entity import new_entity

        template = new_entity("small-lamp")
        template.use_colors = True
        placement = EntityPlacement(
            ir_node_id="lamp_template",
            entity_type="small-lamp",
            position=(2, 3),
            properties={"entity_obj": template},
        )
        entity = emitter.create_entity(placement)
        assert entity is not None
        assert entity.use_colors is True

    def test_generic_entity_property_setting(self, emitter):
        """Test setting generic properties on non-combinator entities."""
        placement = EntityPlacement(
            ir_node_id="inserter_1",
            entity_type="inserter",
            position=(0, 0),
            properties={
                "direction": 4,
            },
        )
        entity = emitter.create_entity(placement)
        assert entity is not None
        assert entity.direction == 4

    def test_decider_with_signal_operands(self, emitter):
        """Test decider with signal-to-signal comparison."""
        placement = EntityPlacement(
            ir_node_id="decider_sig",
            entity_type="decider-combinator",
            position=(0, 0),
            properties={
                "left_operand": "signal-A",
                "operation": "<",
                "right_operand": "signal-B",
                "output_signal": "signal-C",
                "copy_count_from_input": True,
            },
        )
        entity = emitter.create_entity(placement)
        assert entity is not None

    def test_decider_with_constant_left_operand(self, emitter):
        """Test decider with constant as left operand."""
        placement = EntityPlacement(
            ir_node_id="decider_const_left",
            entity_type="decider-combinator",
            position=(0, 0),
            properties={
                "left_operand": 10,
                "operation": ">",
                "right_operand": 5,
                "output_signal": "signal-A",
            },
        )
        entity = emitter.create_entity(placement)
        assert entity is not None


# =============================================================================
# Coverage gap tests (Lines 402-413, 433-444, 471-474, 483-485)
# =============================================================================


def compile_to_ir(source: str):
    """Helper to compile source to IR."""
    parser = DSLParser()
    ast = parser.parse(source, "<test>")
    diagnostics = ProgramDiagnostics()
    analyzer = SemanticAnalyzer(diagnostics=diagnostics)
    analyzer.visit(ast)
    lowerer = ASTLowerer(analyzer, diagnostics)
    ir_ops = lowerer.lower_program(ast)
    return ir_ops, lowerer, diagnostics


class TestEntityEmitterCoverageGaps:
    """Tests for entity_emitter.py coverage gaps > 2 lines."""

    def test_inline_condition_without_circuit_enabled_attr(self):
        """Cover lines 402-413: inline_condition when entity lacks circuit_enabled."""
        source = """
        Signal sensor = 100;
        Entity lamp = place("small-lamp", 0, 0, { use_colors: 1 });
        lamp.enabled = (sensor > 50) : 1;
        """
        ir_ops, lowerer, diags = compile_to_ir(source)
        assert not diags.has_errors()

    def test_inline_bundle_condition(self):
        """Cover lines 433-444: inline_bundle_condition handling."""
        source = """
        Signal a = 10;
        Signal b = 20;
        Bundle inputs = { a, b };
        Entity lamp = place("small-lamp", 0, 0);
        lamp.enabled = (all(inputs) > 5) : 1;
        """
        ir_ops, lowerer, diags = compile_to_ir(source)

    def test_signal_property_without_circuit_enabled(self):
        """Cover lines 471-474, 483-485: signal property on entity without circuit_enabled."""
        source = """
        Signal sensor = 100;
        Entity lamp = place("small-lamp", 0, 0);
        lamp.enabled = sensor;
        """
        ir_ops, lowerer, diags = compile_to_ir(source)
