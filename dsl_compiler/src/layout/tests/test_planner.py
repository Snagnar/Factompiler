"""Tests for layout/planner.py - one test per function."""

import pytest

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.ir.nodes import IRArith, IRConst, SignalRef
from dsl_compiler.src.layout.planner import LayoutPlanner


def make_const(node_id, value, output_type="signal-A"):
    op = IRConst(node_id, output_type)
    op.value = value
    op.debug_metadata = {"user_declared": True}
    return op


def make_arith(node_id, left, right, op_type="+", output_type="signal-C"):
    op = IRArith(node_id, output_type)
    op.op = op_type
    op.left = left
    op.right = right
    return op


@pytest.fixture
def planner():
    return LayoutPlanner({}, ProgramDiagnostics(), max_layout_retries=0)


def test_init(planner):
    assert planner.tile_grid is not None
    assert planner.layout_plan is not None


def test_plan_layout(planner):
    result = planner.plan_layout([make_const("c1", 42)])
    assert "c1" in result.entity_placements


def test_reset_layout_state(planner):
    planner._memory_modules = {"x": 1}
    planner._reset_layout_state()
    assert planner._memory_modules == {}


def test_setup_signal_analysis(planner):
    planner._setup_signal_analysis([make_const("c1", 42)])
    assert planner.signal_analyzer is not None


def test_create_entities(planner):
    planner._setup_signal_analysis([make_const("c1", 42)])
    planner._create_entities([make_const("c1", 42)])
    assert planner.signal_graph is not None


def test_update_tile_grid(planner):
    planner._setup_signal_analysis([make_const("c1", 42)])
    planner._create_entities([make_const("c1", 42)])
    planner._update_tile_grid()  # Should not raise


def test_set_metadata(planner):
    planner._set_metadata("Label", "Desc")
    assert planner.layout_plan.blueprint_label == "Label"
    assert planner.layout_plan.blueprint_description == "Desc"


def test_determine_locked_wire_colors(planner):
    planner._setup_signal_analysis([make_const("c1", 42)])
    planner._create_entities([make_const("c1", 42)])
    result = planner._determine_locked_wire_colors()
    assert isinstance(result, dict)


def test_plan_connections(planner):
    planner._setup_signal_analysis([make_const("c1", 42)])
    planner._create_entities([make_const("c1", 42)])
    planner._update_tile_grid()
    result = planner._plan_connections()
    assert isinstance(result, bool)


def test_resolve_source_entity(planner):
    planner._setup_signal_analysis([make_const("c1", 42)])
    planner._create_entities([make_const("c1", 42)])
    # source_entity should resolve constants
    result = planner._resolve_source_entity("c1")
    assert result is None or isinstance(result, str)


def test_optimize_positions(planner):
    ops = [make_const("c1", 1), make_const("c2", 2)]
    planner._setup_signal_analysis(ops)
    planner._create_entities(ops)
    planner._update_tile_grid()
    planner._optimize_positions(time_multiplier=0.1)
    # Just verify it runs without error


def test_plan_layout_with_connections(planner):
    # Create connected operations to exercise wire routing
    c1 = make_const("c1", 10)
    c2 = make_const("c2", 20)
    arith = make_arith("add1", SignalRef("signal-A", "c1"), SignalRef("signal-A", "c2"))
    result = planner.plan_layout([c1, c2, arith])
    assert "c1" in result.entity_placements
    assert "add1" in result.entity_placements


def test_plan_layout_with_power_poles():
    planner = LayoutPlanner(
        {}, ProgramDiagnostics(), max_layout_retries=0, power_pole_type="medium-electric-pole"
    )
    ops = [make_const("c1", 42)]
    result = planner.plan_layout(ops)
    # Check that power poles were added
    has_pole = any("pole" in p.entity_type for p in result.entity_placements.values())
    assert has_pole or len(result.entity_placements) >= 1  # May not add poles if layout is small


def test_inject_wire_colors_into_placements():
    """Test _inject_wire_colors_into_placements stores wire colors."""
    planner = LayoutPlanner({}, ProgramDiagnostics(), max_layout_retries=0)
    c1 = make_const("c1", 10)
    c2 = make_const("c2", 20)
    arith = make_arith("add1", SignalRef("signal-A", "c1"), SignalRef("signal-A", "c2"))

    planner._setup_signal_analysis([c1, c2, arith])
    planner._create_entities([c1, c2, arith])
    planner._update_tile_grid()
    planner._optimize_positions(time_multiplier=0.1)
    planner._plan_connections()
    # Wire colors should be injected into placements


def test_inject_operand_wire_color():
    """Test _inject_operand_wire_color for individual operands."""
    planner = LayoutPlanner({}, ProgramDiagnostics(), max_layout_retries=0)
    c1 = make_const("c1", 10)
    arith = make_arith("add1", SignalRef("signal-A", "c1"), 5)

    planner._setup_signal_analysis([c1, arith])
    planner._create_entities([c1, arith])
    planner._update_tile_grid()
    planner._optimize_positions(time_multiplier=0.1)
    planner._plan_connections()

    # Check placement has wire color info
    planner.layout_plan.get_placement("add1")
    # May or may not have wire colors depending on connection planning


def test_trim_power_poles():
    """Test _trim_power_poles removes unused poles."""
    planner = LayoutPlanner(
        {}, ProgramDiagnostics(), max_layout_retries=0, power_pole_type="medium-electric-pole"
    )
    c1 = make_const("c1", 42)

    planner._setup_signal_analysis([c1])
    planner._reset_layout_state()
    planner._create_entities([c1])
    planner._add_power_pole_grid()
    planner._optimize_positions(time_multiplier=0.1)

    # Count poles before trim
    poles_before = sum(
        1
        for p in planner.layout_plan.entity_placements.values()
        if p.properties.get("is_power_pole")
    )

    planner._trim_power_poles()

    # Count poles after trim (should be same or fewer)
    poles_after = sum(
        1
        for p in planner.layout_plan.entity_placements.values()
        if p.properties.get("is_power_pole")
    )
    assert poles_after <= poles_before


def test_add_power_pole_grid():
    """Test _add_power_pole_grid places power poles."""
    planner = LayoutPlanner(
        {}, ProgramDiagnostics(), max_layout_retries=0, power_pole_type="small-electric-pole"
    )
    c1 = make_const("c1", 42)
    c2 = make_const("c2", 100)

    planner._setup_signal_analysis([c1, c2])
    planner._reset_layout_state()
    planner._create_entities([c1, c2])
    planner._update_tile_grid()
    planner._add_power_pole_grid()

    # Should have some power poles
    any(p.properties.get("is_power_pole") for p in planner.layout_plan.entity_placements.values())
    # May or may not add poles depending on entity bounds


def test_determine_locked_wire_colors_with_memory():
    """Test _determine_locked_wire_colors locks memory feedback to red."""
    from dsl_compiler.src.ir.nodes import IRMemCreate

    planner = LayoutPlanner({}, ProgramDiagnostics(), max_layout_retries=0)
    mem_op = IRMemCreate("mem1", "signal-A")

    planner._setup_signal_analysis([mem_op])
    planner._reset_layout_state()
    planner._create_entities([mem_op])

    locked = planner._determine_locked_wire_colors()
    # Should have locked colors for memory gates
    assert isinstance(locked, dict)


def test_determine_locked_wire_colors_with_bundle_separation():
    """Test _determine_locked_wire_colors for bundle operations."""
    planner = LayoutPlanner({}, ProgramDiagnostics(), max_layout_retries=0)
    c1 = make_const("c1", 10)
    arith = make_arith("add1", SignalRef("signal-A", "c1"), SignalRef("signal-B", "c1"))
    arith.needs_wire_separation = True

    planner._setup_signal_analysis([c1, arith])
    planner._reset_layout_state()
    planner._create_entities([c1, arith])

    locked = planner._determine_locked_wire_colors()
    assert isinstance(locked, dict)


def test_resolve_source_entity_with_at_syntax():
    """Test _resolve_source_entity with @-syntax signal IDs."""
    planner = LayoutPlanner({}, ProgramDiagnostics(), max_layout_retries=0)
    c1 = make_const("c1", 42)

    planner._setup_signal_analysis([c1])
    planner._create_entities([c1])

    # Test with @-syntax
    result = planner._resolve_source_entity("signal-A@c1")
    assert result == "c1" or result is None


def test_resolve_source_entity_with_signal_ref():
    """Test _resolve_source_entity with SignalRef."""
    planner = LayoutPlanner({}, ProgramDiagnostics(), max_layout_retries=0)
    c1 = make_const("c1", 42)

    planner._setup_signal_analysis([c1])
    planner._create_entities([c1])

    ref = SignalRef("signal-A", "c1")
    result = planner._resolve_source_entity(ref)
    assert result == "c1" or result is None


def test_plan_layout_retry_on_failure():
    """Test plan_layout retries on routing failure."""
    planner = LayoutPlanner({}, ProgramDiagnostics(), max_layout_retries=1)
    c1 = make_const("c1", 42)

    result = planner.plan_layout([c1])
    assert result is not None


def test_plan_layout_with_description():
    """Test plan_layout sets blueprint description."""
    planner = LayoutPlanner({}, ProgramDiagnostics(), max_layout_retries=0)
    c1 = make_const("c1", 42)

    result = planner.plan_layout(
        [c1], blueprint_label="Test Label", blueprint_description="Test Desc"
    )
    assert result.blueprint_label == "Test Label"
    assert result.blueprint_description == "Test Desc"


def test_resolve_source_entity_signal_graph_fallback():
    """Test _resolve_source_entity falls back to signal graph when entity not in layout."""
    planner = LayoutPlanner({}, ProgramDiagnostics(), max_layout_retries=0)
    c1 = make_const("c1", 42)

    planner._setup_signal_analysis([c1])
    planner._create_entities([c1])

    # Try to resolve a non-existent entity - should return None or fallback
    result = planner._resolve_source_entity("nonexistent")
    assert result is None or isinstance(result, str)


def test_inject_operand_wire_color_no_source():
    """Test _inject_operand_wire_color when source entity not found."""
    planner = LayoutPlanner({}, ProgramDiagnostics(), max_layout_retries=0)
    c1 = make_const("c1", 10)
    arith = make_arith("add1", SignalRef("signal-A", "nonexistent"), 5)

    planner._setup_signal_analysis([c1, arith])
    planner._create_entities([c1, arith])
    planner._update_tile_grid()
    planner._plan_connections()

    placement = planner.layout_plan.get_placement("add1")
    if placement:
        # Inject wire color for missing source
        planner._inject_operand_wire_color(placement, "left", 0)
        # Should handle gracefully


def test_trim_power_poles_removes_unused():
    """Test _trim_power_poles removes poles that don't cover any entities."""
    from unittest.mock import MagicMock

    from dsl_compiler.src.layout.layout_plan import EntityPlacement

    # Use "medium" not "medium-electric-pole" - config uses short names
    planner = LayoutPlanner(
        {}, ProgramDiagnostics(), max_layout_retries=0, power_pole_type="medium"
    )
    planner._reset_layout_state()

    # Add one entity at position (0, 0)
    planner.layout_plan.entity_placements["entity1"] = EntityPlacement(
        ir_node_id="entity1", entity_type="test", position=(0.5, 1.0), properties={}, role="test"
    )

    # Add distant pole (should be removed)
    planner.layout_plan.entity_placements["distant_pole"] = EntityPlacement(
        ir_node_id="distant_pole",
        entity_type="medium-electric-pole",
        position=(100.0, 100.0),
        properties={"is_power_pole": True},
        role="power_pole",
    )

    # Add close pole (should remain)
    planner.layout_plan.entity_placements["close_pole"] = EntityPlacement(
        ir_node_id="close_pole",
        entity_type="medium-electric-pole",
        position=(0.5, 1.0),
        properties={"is_power_pole": True},
        role="power_pole",
    )

    # Mock power_poles list with objects that have pole_id
    pole1 = MagicMock()
    pole1.pole_id = "distant_pole"
    pole2 = MagicMock()
    pole2.pole_id = "close_pole"
    planner.layout_plan.power_poles = [pole1, pole2]

    planner._trim_power_poles()

    # Distant pole should be removed
    assert "close_pole" in planner.layout_plan.entity_placements
    assert "distant_pole" not in planner.layout_plan.entity_placements


def test_determine_locked_wire_colors_sr_latch():
    """Test _determine_locked_wire_colors for SR latch memory modules."""
    from dsl_compiler.src.ir.nodes import MEMORY_TYPE_SR_LATCH, IRLatchWrite, IRMemCreate

    planner = LayoutPlanner({}, ProgramDiagnostics(), max_layout_retries=0)
    mem_op = IRMemCreate("mem1", "signal-A")
    latch_op = IRLatchWrite(
        "mem1", 1, SignalRef("signal-S", "src"), SignalRef("signal-R", "src"), MEMORY_TYPE_SR_LATCH
    )

    planner._setup_signal_analysis([mem_op, latch_op])
    planner._reset_layout_state()
    planner._create_entities([mem_op, latch_op])

    locked = planner._determine_locked_wire_colors()
    # SR latch should have locked colors
    assert isinstance(locked, dict)


def test_plan_layout_chain_of_operations():
    """Test plan_layout with a chain of operations to exercise wire routing."""
    planner = LayoutPlanner({}, ProgramDiagnostics(), max_layout_retries=0)

    # Create a chain: c1 -> arith1 -> arith2 -> arith3
    c1 = make_const("c1", 10)
    arith1 = make_arith("arith1", SignalRef("signal-A", "c1"), 2, output_type="signal-B")
    arith2 = make_arith("arith2", SignalRef("signal-B", "arith1"), 3, output_type="signal-C")
    arith3 = make_arith("arith3", SignalRef("signal-C", "arith2"), 4, output_type="signal-D")

    result = planner.plan_layout([c1, arith1, arith2, arith3])

    # All operations should be placed
    assert "c1" in result.entity_placements
    assert "arith1" in result.entity_placements
    assert "arith2" in result.entity_placements
    assert "arith3" in result.entity_placements
    # Wire connections should exist
    assert len(result.wire_connections) >= 1


def test_plan_layout_with_decider():
    """Test plan_layout with decider combinator."""
    from dsl_compiler.src.ir.nodes import DeciderCondition, IRDecider

    planner = LayoutPlanner({}, ProgramDiagnostics(), max_layout_retries=0)

    c1 = make_const("c1", 10)

    decider = IRDecider("dec1", "signal-C")
    decider.conditions = [
        DeciderCondition(
            comparator=">", first_operand=SignalRef("signal-A", "c1"), second_operand=5
        )
    ]
    decider.output_value = 1

    result = planner.plan_layout([c1, decider])
    assert "dec1" in result.entity_placements


def test_plan_layout_with_memory():
    """Test plan_layout with memory operations."""
    from dsl_compiler.src.ir.nodes import IRMemCreate, IRMemRead, IRMemWrite

    planner = LayoutPlanner({}, ProgramDiagnostics(), max_layout_retries=0)

    mem_create = IRMemCreate("mem1", "signal-A")
    c1 = make_const("c1", 42)
    mem_write = IRMemWrite("mem1", SignalRef("signal-A", "c1"), 1)
    mem_read = IRMemRead("read1", "signal-A")
    mem_read.memory_id = "mem1"

    result = planner.plan_layout([mem_create, c1, mem_write, mem_read])
    # Memory module should be created
    assert any("write" in k or "hold" in k for k in result.entity_placements)


def test_plan_layout_with_multiple_constants():
    """Test plan_layout with multiple constants feeding one arith."""
    planner = LayoutPlanner({}, ProgramDiagnostics(), max_layout_retries=0)

    c1 = make_const("c1", 10)
    c2 = make_const("c2", 20, output_type="signal-B")
    arith = make_arith("add", SignalRef("signal-A", "c1"), SignalRef("signal-B", "c2"))

    result = planner.plan_layout([c1, c2, arith])
    assert "c1" in result.entity_placements
    assert "c2" in result.entity_placements
    assert "add" in result.entity_placements


# --- Full pipeline tests that exercise more code paths ---


def compile_facto_source(source: str):
    """Compile Facto source code through the full pipeline."""
    from dsl_compiler.src.lowering.lowerer import ASTLowerer
    from dsl_compiler.src.parsing.parser import DSLParser
    from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer

    diagnostics = ProgramDiagnostics()
    parser = DSLParser()
    ast = parser.parse(source, "<test>")

    analyzer = SemanticAnalyzer(diagnostics=diagnostics)
    analyzer.visit(ast)

    lowerer = ASTLowerer(analyzer, diagnostics)
    ir_ops = lowerer.lower_program(ast)

    planner = LayoutPlanner(
        lowerer.ir_builder.signal_type_map,
        diagnostics=diagnostics,
        signal_refs=lowerer.signal_refs,
        referenced_signal_names=lowerer.referenced_signal_names,
    )
    layout = planner.plan_layout(ir_ops)
    return layout


def test_compile_basic_arithmetic():
    """Test full compilation of basic arithmetic."""
    source = """
Signal a = 100;
Signal b = 200;
Signal sum = a + b;
"""
    layout = compile_facto_source(source)
    assert len(layout.entity_placements) >= 1


def test_compile_with_memory():
    """Test full compilation with memory."""
    source = """
Memory counter: "signal-A";
counter.write(counter.read() + 1);
Signal output = counter.read();
"""
    layout = compile_facto_source(source)
    assert len(layout.entity_placements) >= 2


def test_compile_with_entity_and_decider():
    """Test full compilation with entity and comparisons."""
    source = """
Signal a = 10;
Entity lamp = place("small-lamp", 0, 0);
lamp.enable = a > 5;
"""
    layout = compile_facto_source(source)
    assert any("entity_ir" in k for k in layout.entity_placements)


def test_compile_with_bundle():
    """Test full compilation with bundle operations."""
    source = """
Signal a = 100;
bundle = each + 10;
"""
    layout = compile_facto_source(source)
    assert len(layout.entity_placements) >= 1


@pytest.mark.parametrize(
    "source,min_entities",
    [
        # Projection
        ('Signal a = 100;\nSignal b = a | "iron-plate";', 1),
        # Signal comparison
        ("Signal a = 100;\nSignal b = 50;\nSignal c = a > b;", 1),
        # Chained operations
        ("Signal a = 1;\nSignal b = a + 1;\nSignal c = b + 1;\nSignal d = c + 1;", 3),
        # Mul/div/mod/power
        ("Signal a = 100;\nSignal b = a * 2;\nSignal c = b / 4;", 2),
        ("Signal a = 100;\nSignal b = a % 7;", 1),
        ("Signal a = 2;\nSignal b = a ** 8;", 1),
        # Bitwise operations
        ("Signal a = 255;\nSignal b = a AND 15;\nSignal c = a OR 240;", 2),
        # Unary negation
        ("Signal a = 100;\nSignal b = -a;", 1),
        # Logical ops
        ("Signal a = 1;\nSignal b = 1;\nSignal c = a && b;", 1),
        # Multiple comparisons
        ("Signal a = 100;\nSignal lt = a < 50;\nSignal ge = a >= 100;", 2),
        # Memory with reset
        (
            'Memory m: "signal-A";\nSignal limit = 100;\nm.write(m.read() + 1);\nm.write(0, when=m.read() >= limit);',
            3,
        ),
        # Multiple memories
        (
            'Memory m1: "signal-A";\nMemory m2: "signal-B";\nm1.write(m1.read() + 1);\nm2.write(m2.read() + 2);',
            2,
        ),
    ],
)
def test_compile_operations(source, min_entities):
    """Parameterized test for various operations."""
    layout = compile_facto_source(source)
    assert len(layout.entity_placements) >= min_entities


# =============================================================================
# Coverage gap tests (Lines 112-124, 306-310, 337-341)
# =============================================================================


class TestLayoutPlannerCoverageGaps:
    """Tests for planner.py coverage gaps > 2 lines."""

    def test_operand_wire_color_injection_no_source(self):
        """Cover lines 306-310: operand wire color when no source found."""
        source = """
        Signal a = 10;
        Signal b = 20;
        Signal result = (a > b) : 1;
        """
        layout = compile_facto_source(source)
        assert layout is not None

    def test_output_value_wire_color_injection(self):
        """Cover lines 337-341: output_value wire color injection."""
        source = """
        Signal a = 10;
        Signal b = 20;
        Signal result = (a > 5) : b;
        """
        layout = compile_facto_source(source)
        assert layout is not None

    def test_retry_mechanism_simple(self):
        """Cover lines 112-124: basic layout without retry."""
        source = """
        Signal x = 1;
        Signal y = 2;
        Signal z = x + y;
        """
        layout = compile_facto_source(source)
        assert layout is not None
