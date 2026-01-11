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
