"""
Tests for layout/entity_placer.py - Entity placement planning.
"""

import pytest

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.ir.nodes import (
    BundleRef,
    DeciderCondition,
    IRArith,
    IRConst,
    IRDecider,
    IREntityOutput,
    IREntityPropRead,
    IREntityPropWrite,
    IRMemCreate,
    IRPlaceEntity,
    IRWireMerge,
    SignalRef,
)
from dsl_compiler.src.layout.entity_placer import EntityPlacer
from dsl_compiler.src.layout.layout_plan import LayoutPlan
from dsl_compiler.src.layout.signal_analyzer import SignalAnalyzer
from dsl_compiler.src.layout.tile_grid import TileGrid

# === Helper functions ===


def make_const(node_id: str, value: int, output_type: str = "signal-A") -> IRConst:
    """Helper to create an IRConst node."""
    op = IRConst(node_id, output_type)
    op.value = value
    return op


def make_bundle_const(node_id: str, signals: dict[str, int]) -> IRConst:
    """Helper to create a bundle IRConst node."""
    op = IRConst(node_id, "signal-A")
    op.signals = signals
    return op


def make_arith(node_id: str, left, right, op: str = "+", output_type: str = "signal-C") -> IRArith:
    """Helper to create an IRArith node."""
    arith = IRArith(node_id, output_type)
    arith.op = op
    arith.left = left
    arith.right = right
    return arith


def make_decider(
    node_id: str, left, right, test_op: str = ">", output_type: str = "signal-C"
) -> IRDecider:
    """Helper to create an IRDecider node."""
    decider = IRDecider(node_id, output_type)
    decider.test_op = test_op
    decider.left = left
    decider.right = right
    decider.output_value = 1
    return decider


def make_signal_ref(source_id: str, signal_type: str = "signal-A") -> SignalRef:
    """Helper to create a SignalRef."""
    return SignalRef(signal_type, source_id)


# === Fixtures ===


@pytest.fixture
def diagnostics():
    """Create a diagnostics instance for tests."""
    return ProgramDiagnostics()


@pytest.fixture
def tile_grid():
    """Create a tile grid for tests."""
    return TileGrid()


@pytest.fixture
def layout_plan():
    """Create a layout plan for tests."""
    return LayoutPlan()


@pytest.fixture
def signal_analyzer(diagnostics):
    """Create a signal analyzer for tests."""
    return SignalAnalyzer(diagnostics, {})


@pytest.fixture
def entity_placer(tile_grid, layout_plan, signal_analyzer, diagnostics):
    """Create an entity placer for tests."""
    return EntityPlacer(tile_grid, layout_plan, signal_analyzer, diagnostics)


# === Tests for EntityPlacer.__init__ ===


class TestEntityPlacerInit:
    """Tests for EntityPlacer initialization."""

    def test_init(self, tile_grid, layout_plan, signal_analyzer, diagnostics):
        """Test EntityPlacer initialization."""
        placer = EntityPlacer(tile_grid, layout_plan, signal_analyzer, diagnostics)

        assert placer.tile_grid is tile_grid
        assert placer.plan is layout_plan
        assert placer.signal_analyzer is signal_analyzer
        assert placer.diagnostics is diagnostics
        assert placer.signal_graph is not None
        assert placer.memory_builder is not None


# === Tests for _build_debug_info ===


class TestBuildDebugInfo:
    """Tests for _build_debug_info method."""

    def test_build_debug_info_basic(self, entity_placer):
        """Test building debug info for a basic IR node."""
        op = make_const("const1", 42, "signal-A")

        debug_info = entity_placer._build_debug_info(op)

        assert debug_info["variable"] == "const1"
        assert debug_info["operation"] == "const"
        assert "value=42" in debug_info["details"]

    def test_build_debug_info_with_role_override(self, entity_placer):
        """Test building debug info with role override."""
        op = make_const("const1", 42, "signal-A")

        debug_info = entity_placer._build_debug_info(op, role_override="test_role")

        assert debug_info["role"] == "test_role"

    def test_build_debug_info_arithmetic(self, entity_placer):
        """Test building debug info for arithmetic operation."""
        op = make_arith(
            "arith1",
            make_signal_ref("const1", "signal-A"),
            make_signal_ref("const2", "signal-B"),
            "+",
            "signal-C",
        )

        debug_info = entity_placer._build_debug_info(op)

        assert debug_info["operation"] == "arith"
        assert "op=+" in debug_info["details"]

    def test_build_debug_info_decider(self, entity_placer):
        """Test building debug info for decider operation."""
        op = make_decider(
            "dec1",
            make_signal_ref("const1", "signal-A"),
            10,
            ">",
            "signal-B",
        )

        debug_info = entity_placer._build_debug_info(op)

        assert debug_info["operation"] == "decider"
        assert "cond=>" in debug_info["details"]

    def test_build_debug_info_memory_create(self, entity_placer):
        """Test building debug info for memory create operation."""
        op = IRMemCreate("mem1", "signal-A")

        debug_info = entity_placer._build_debug_info(op)

        assert debug_info["operation"] == "memory"
        assert debug_info["details"] == "decl"


# === Tests for place_ir_operation ===


class TestPlaceIROperation:
    """Tests for place_ir_operation method."""

    def test_place_ir_operation_constant(self, entity_placer, signal_analyzer):
        """Test placing a constant operation."""
        op = make_const("const1", 42, "signal-A")

        # Register in signal_usage so it materializes
        signal_analyzer.signal_usage["const1"] = type(
            "MockUsage",
            (),
            {
                "should_materialize": True,
                "resolved_signal_name": "signal-A",
                "resolved_signal_type": "virtual",
                "debug_label": None,
                "source_ast": None,
            },
        )()
        entity_placer.signal_usage = signal_analyzer.signal_usage

        entity_placer.place_ir_operation(op)

        assert "const1" in entity_placer.plan.entity_placements
        placement = entity_placer.plan.get_placement("const1")
        assert placement.entity_type == "constant-combinator"
        assert placement.role == "literal"

    def test_place_ir_operation_arithmetic(self, entity_placer, signal_analyzer):
        """Test placing an arithmetic operation."""
        # First place a constant that the arithmetic will reference
        const_op = make_const("const1", 10, "signal-A")
        signal_analyzer.signal_usage["const1"] = type(
            "MockUsage",
            (),
            {
                "should_materialize": True,
                "resolved_signal_name": "signal-A",
                "resolved_signal_type": "virtual",
                "debug_label": None,
                "source_ast": None,
                "producer": const_op,
            },
        )()
        entity_placer.signal_usage = signal_analyzer.signal_usage
        entity_placer.place_ir_operation(const_op)

        arith_op = make_arith(
            "arith1",
            make_signal_ref("const1", "signal-A"),
            5,
            "+",
            "signal-B",
        )

        entity_placer.place_ir_operation(arith_op)

        assert "arith1" in entity_placer.plan.entity_placements
        placement = entity_placer.plan.get_placement("arith1")
        assert placement.entity_type == "arithmetic-combinator"
        assert placement.role == "arithmetic"

    def test_place_ir_operation_decider(self, entity_placer, signal_analyzer):
        """Test placing a decider operation."""
        # First place a constant
        const_op = make_const("const1", 10, "signal-A")
        signal_analyzer.signal_usage["const1"] = type(
            "MockUsage",
            (),
            {
                "should_materialize": True,
                "resolved_signal_name": "signal-A",
                "resolved_signal_type": "virtual",
                "debug_label": None,
                "source_ast": None,
                "producer": const_op,
            },
        )()
        entity_placer.signal_usage = signal_analyzer.signal_usage
        entity_placer.place_ir_operation(const_op)

        decider_op = make_decider(
            "dec1",
            make_signal_ref("const1", "signal-A"),
            5,
            ">",
            "signal-B",
        )

        entity_placer.place_ir_operation(decider_op)

        assert "dec1" in entity_placer.plan.entity_placements
        placement = entity_placer.plan.get_placement("dec1")
        assert placement.entity_type == "decider-combinator"
        assert placement.role == "decider"

    def test_place_ir_operation_unknown_warns(self, entity_placer, diagnostics):
        """Test that unknown IR operations generate a warning."""

        class UnknownIRNode:
            node_id = "unknown1"

        entity_placer.place_ir_operation(UnknownIRNode())

        # Check diagnostics has a warning
        assert diagnostics._warning_count > 0


# === Tests for _place_constant ===


class TestPlaceConstant:
    """Tests for _place_constant method."""

    def test_place_constant_with_bundle_signals(self, entity_placer, signal_analyzer):
        """Test placing a bundle constant with multiple signals."""
        op = make_bundle_const("bundle1", {"signal-A": 10, "signal-B": 20})
        signal_analyzer.signal_usage["bundle1"] = type(
            "MockUsage",
            (),
            {
                "should_materialize": True,
                "resolved_signal_name": "signal-A",
                "resolved_signal_type": "virtual",
                "debug_label": None,
                "source_ast": None,
            },
        )()
        entity_placer.signal_usage = signal_analyzer.signal_usage

        entity_placer._place_constant(op)

        placement = entity_placer.plan.get_placement("bundle1")
        assert placement.role == "bundle_const"
        assert placement.properties["signals"] == {"signal-A": 10, "signal-B": 20}

    def test_place_constant_no_materialize_skips(self, entity_placer, signal_analyzer):
        """Test that constants marked as not-materialize are skipped."""
        op = make_const("const1", 42, "signal-A")
        signal_analyzer.signal_usage["const1"] = type(
            "MockUsage",
            (),
            {"should_materialize": False, "debug_label": None, "source_ast": None},
        )()
        entity_placer.signal_usage = signal_analyzer.signal_usage

        entity_placer._place_constant(op)

        assert "const1" not in entity_placer.plan.entity_placements


# === Tests for _place_arithmetic ===


class TestPlaceArithmetic:
    """Tests for _place_arithmetic method."""

    def test_place_arithmetic_with_wire_separation(self, entity_placer):
        """Test placing arithmetic with wire separation flag."""
        arith_op = make_arith(
            "arith1",
            make_signal_ref("src1", "signal-A"),
            make_signal_ref("src2", "signal-B"),
            "*",
            "signal-C",
        )
        arith_op.needs_wire_separation = True

        entity_placer._place_arithmetic(arith_op)

        placement = entity_placer.plan.get_placement("arith1")
        assert placement.properties.get("needs_wire_separation") is True


# === Tests for _place_decider ===


class TestPlaceDecider:
    """Tests for _place_decider method."""

    def test_place_single_condition_decider(self, entity_placer):
        """Test placing a single-condition decider."""
        decider_op = make_decider(
            "dec1",
            make_signal_ref("const1", "signal-A"),
            5,
            ">",
            "signal-B",
        )

        entity_placer._place_decider(decider_op)

        placement = entity_placer.plan.get_placement("dec1")
        assert placement.entity_type == "decider-combinator"
        assert placement.properties["operation"] == ">"

    def test_place_multi_condition_decider(self, entity_placer):
        """Test placing a multi-condition decider."""
        conditions = [
            DeciderCondition(
                comparator=">",
                first_operand=make_signal_ref("src1", "signal-A"),
                second_operand=5,
            ),
            DeciderCondition(
                comparator="<",
                first_operand=make_signal_ref("src2", "signal-B"),
                second_operand=10,
            ),
        ]

        decider_op = make_decider(
            "dec1",
            make_signal_ref("src1", "signal-A"),
            5,
            ">",
            "signal-C",
        )
        decider_op.conditions = conditions

        entity_placer._place_decider(decider_op)

        placement = entity_placer.plan.get_placement("dec1")
        assert placement.entity_type == "decider-combinator"
        assert "conditions" in placement.properties
        assert len(placement.properties["conditions"]) == 2


# === Tests for _place_user_entity ===


class TestPlaceUserEntity:
    """Tests for _place_user_entity method."""

    def test_place_user_entity_basic(self, entity_placer):
        """Test placing a basic user entity."""
        op = IRPlaceEntity("lamp1", "small-lamp", None, None)

        entity_placer._place_user_entity(op)

        placement = entity_placer.plan.get_placement("lamp1")
        assert placement.entity_type == "small-lamp"
        assert placement.role == "user_entity"

    def test_place_user_entity_with_position(self, entity_placer):
        """Test placing a user entity with explicit position."""
        op = IRPlaceEntity("lamp1", "small-lamp", 10, 20)

        entity_placer._place_user_entity(op)

        placement = entity_placer.plan.get_placement("lamp1")
        assert placement.position == (10, 20)
        assert placement.properties.get("user_specified_position") is True

    def test_place_user_entity_with_properties(self, entity_placer):
        """Test placing a user entity with custom properties."""
        op = IRPlaceEntity(
            "lamp1",
            "small-lamp",
            None,
            None,
            properties={"color": {"r": 1.0, "g": 0.0, "b": 0.0}},
        )

        entity_placer._place_user_entity(op)

        placement = entity_placer.plan.get_placement("lamp1")
        assert placement.properties.get("color") == {"r": 1.0, "g": 0.0, "b": 0.0}


# === Tests for _place_entity_prop_write ===


class TestPlaceEntityPropWrite:
    """Tests for _place_entity_prop_write method."""

    def test_place_entity_prop_write_constant(self, entity_placer):
        """Test writing a constant value to entity property."""
        # First place an entity
        entity_op = IRPlaceEntity("lamp1", "small-lamp", None, None)
        entity_placer._place_user_entity(entity_op)

        # Write property
        prop_op = IREntityPropWrite("lamp1", "brightness", 100)

        entity_placer._place_entity_prop_write(prop_op)

        placement = entity_placer.plan.get_placement("lamp1")
        prop_writes = placement.properties.get("property_writes", {})
        assert "brightness" in prop_writes
        assert prop_writes["brightness"]["type"] == "constant"
        assert prop_writes["brightness"]["value"] == 100

    def test_place_entity_prop_write_signal(self, entity_placer):
        """Test writing a signal value to entity property."""
        # First place an entity
        entity_op = IRPlaceEntity("lamp1", "small-lamp", None, None)
        entity_placer._place_user_entity(entity_op)

        # Write property with signal reference
        signal_ref = make_signal_ref("src1", "signal-A")
        prop_op = IREntityPropWrite("lamp1", "enable", signal_ref)

        entity_placer._place_entity_prop_write(prop_op)

        placement = entity_placer.plan.get_placement("lamp1")
        prop_writes = placement.properties.get("property_writes", {})
        assert "enable" in prop_writes
        assert prop_writes["enable"]["type"] == "signal"

    def test_place_entity_prop_write_nonexistent_entity_warns(self, entity_placer, diagnostics):
        """Test that writing to nonexistent entity generates warning."""
        prop_op = IREntityPropWrite("nonexistent", "enable", 100)

        entity_placer._place_entity_prop_write(prop_op)

        assert diagnostics._warning_count > 0


# === Tests for _place_entity_prop_read ===


class TestPlaceEntityPropRead:
    """Tests for _place_entity_prop_read method."""

    def test_place_entity_prop_read(self, entity_placer):
        """Test reading an entity property."""
        op = IREntityPropRead("read1", "signal-A")
        op.entity_id = "lamp1"
        op.property_name = "contents"

        entity_placer._place_entity_prop_read(op)

        # Check signal name is registered
        assert "read1" in entity_placer._entity_property_signals
        assert entity_placer._entity_property_signals["read1"] == "lamp1_contents"


# === Tests for _place_entity_output ===


class TestPlaceEntityOutput:
    """Tests for _place_entity_output method."""

    def test_place_entity_output(self, entity_placer):
        """Test placing an entity output read."""
        op = IREntityOutput("output1", "chest1")

        entity_placer._place_entity_output(op)

        # Check signal graph has source
        assert entity_placer.signal_graph.get_source("output1") == "chest1"


# === Tests for _place_wire_merge ===


class TestPlaceWireMerge:
    """Tests for _place_wire_merge method."""

    def test_place_wire_merge(self, entity_placer):
        """Test placing a wire merge operation."""
        sources = [
            make_signal_ref("src1", "signal-A"),
            make_signal_ref("src2", "signal-A"),
        ]
        op = IRWireMerge("merge1", "signal-A")
        op.sources = sources

        entity_placer._place_wire_merge(op)

        # Check junction is registered
        assert "merge1" in entity_placer._wire_merge_junctions
        junction = entity_placer._wire_merge_junctions["merge1"]
        assert len(junction["inputs"]) == 2

    def test_place_wire_merge_membership_tracking(self, entity_placer):
        """Test that wire merge tracks source membership."""
        # Set up signal graph for sources
        entity_placer.signal_graph.set_source("src1", "entity1")
        entity_placer.signal_graph.set_source("src2", "entity2")

        sources = [
            make_signal_ref("src1", "signal-A"),
            make_signal_ref("src2", "signal-A"),
        ]
        op = IRWireMerge("merge1", "signal-A")
        op.sources = sources

        entity_placer._place_wire_merge(op)

        # Check merge membership
        membership = entity_placer.get_merge_membership()
        assert "entity1" in membership
        assert "merge1" in membership["entity1"]


# === Tests for get_merge_membership ===


class TestGetMergeMembership:
    """Tests for get_merge_membership method."""

    def test_get_merge_membership_empty(self, entity_placer):
        """Test getting empty merge membership."""
        result = entity_placer.get_merge_membership()
        assert result == {}


# === Tests for _get_placement_position ===


class TestGetPlacementPosition:
    """Tests for _get_placement_position method."""

    def test_get_placement_position_with_signal_ref(self, entity_placer, signal_analyzer):
        """Test getting position for a SignalRef."""
        # First place an entity
        const_op = make_const("const1", 42, "signal-A")
        signal_analyzer.signal_usage["const1"] = type(
            "MockUsage",
            (),
            {
                "should_materialize": True,
                "resolved_signal_name": "signal-A",
                "resolved_signal_type": "virtual",
                "debug_label": None,
                "source_ast": None,
            },
        )()
        entity_placer.signal_usage = signal_analyzer.signal_usage
        entity_placer._place_constant(const_op)

        # Get its position
        ref = make_signal_ref("const1", "signal-A")
        pos = entity_placer._get_placement_position(ref)

        # Position should be None since we didn't specify one
        assert pos is None

    def test_get_placement_position_with_int(self, entity_placer):
        """Test getting position for an integer value."""
        result = entity_placer._get_placement_position(42)
        assert result is None


# === Tests for _add_signal_sink ===


class TestAddSignalSink:
    """Tests for _add_signal_sink method."""

    def test_add_signal_sink_with_signal_ref(self, entity_placer, signal_analyzer):
        """Test adding signal sink for SignalRef."""
        signal_analyzer.signal_usage["src1"] = type(
            "MockUsage",
            (),
            {"should_materialize": True, "debug_label": None, "source_ast": None},
        )()
        entity_placer.signal_usage = signal_analyzer.signal_usage

        ref = make_signal_ref("src1", "signal-A")
        entity_placer._add_signal_sink(ref, "consumer1")

        sinks = entity_placer.signal_graph.iter_sinks("src1")
        assert "consumer1" in sinks

    def test_add_signal_sink_with_bundle_ref(self, entity_placer):
        """Test adding signal sink for BundleRef."""
        ref = BundleRef({"signal-A", "signal-B"}, "bundle1")
        entity_placer._add_signal_sink(ref, "consumer1")

        sinks = entity_placer.signal_graph.iter_sinks("bundle1")
        assert "consumer1" in sinks

    def test_add_signal_sink_no_materialize_skips(self, entity_placer, signal_analyzer):
        """Test that non-materialized signals are skipped."""
        signal_analyzer.signal_usage["src1"] = type(
            "MockUsage",
            (),
            {"should_materialize": False, "debug_label": None, "source_ast": None},
        )()
        entity_placer.signal_usage = signal_analyzer.signal_usage

        ref = make_signal_ref("src1", "signal-A")
        entity_placer._add_signal_sink(ref, "consumer1")

        sinks = entity_placer.signal_graph.iter_sinks("src1")
        assert "consumer1" not in sinks


# === Tests for cleanup_unused_entities ===


class TestCleanupUnusedEntities:
    """Tests for cleanup_unused_entities method."""

    def test_cleanup_unused_entities_removes_inlined(
        self, entity_placer, signal_analyzer, diagnostics
    ):
        """Test that cleanup removes inlined comparison entities."""
        # Place a decider
        decider_op = make_decider(
            "dec1",
            make_signal_ref("const1", "signal-A"),
            5,
            ">",
            "signal-B",
        )
        entity_placer._place_decider(decider_op)

        # Place an entity
        entity_op = IRPlaceEntity("lamp1", "small-lamp", None, None)
        entity_placer._place_user_entity(entity_op)

        # Manually mark the decider for removal via inlined comparison
        lamp = entity_placer.plan.get_placement("lamp1")
        lamp.properties["property_writes"] = {
            "enable": {
                "type": "inline_comparison",
                "comparison_data": {"source_node_id_to_remove": "dec1"},
            }
        }

        entity_placer.cleanup_unused_entities()

        # Decider should be removed
        assert "dec1" not in entity_placer.plan.entity_placements


# === Tests for create_output_anchors ===


class TestCreateOutputAnchors:
    """Tests for create_output_anchors method."""

    def test_create_output_anchors_basic(self, entity_placer, signal_analyzer):
        """Test creating output anchors for output signals."""
        # Create a non-const producer (arithmetic) so anchors are created
        arith_op = make_arith("out1", 1, 2, "+", "signal-A")

        # Set up signal usage with is_output
        usage = type(
            "MockUsage",
            (),
            {
                "should_materialize": True,
                "resolved_signal_name": "signal-A",
                "debug_label": "output_val",
                "debug_metadata": {"is_output": True},
                "producer": arith_op,
                "output_aliases": {"alias1"},
                "signal_type": "signal-A",
                "source_ast": None,
            },
        )()
        signal_analyzer.signal_usage["out1"] = usage
        entity_placer.signal_usage = signal_analyzer.signal_usage

        entity_placer.create_output_anchors()

        # Should create anchor for the alias
        assert "out1_alias1_output_anchor" in entity_placer.plan.entity_placements

    def test_create_output_anchors_no_output_skips(self, entity_placer, signal_analyzer):
        """Test that non-output signals don't get anchors."""
        const_op = make_const("sig1", 42, "signal-A")

        usage = type(
            "MockUsage",
            (),
            {
                "should_materialize": True,
                "resolved_signal_name": "signal-A",
                "debug_label": "test_val",
                "debug_metadata": {"is_output": False},
                "producer": const_op,
                "output_aliases": set(),
                "signal_type": "signal-A",
                "source_ast": None,
            },
        )()
        signal_analyzer.signal_usage["sig1"] = usage
        entity_placer.signal_usage = signal_analyzer.signal_usage

        entity_placer.create_output_anchors()

        # No anchors should be created
        anchor_count = sum(1 for k in entity_placer.plan.entity_placements if "_output_anchor" in k)
        assert anchor_count == 0


class TestCreateDebugInfo:
    """Tests for EntityPlacer._build_debug_info method."""

    def test_build_debug_info_with_source_ast(self, entity_placer, signal_analyzer):
        """Test debug info includes source location from AST."""
        const_op = make_const("c1", 42, "signal-A")
        const_op.source_ast = type("MockAST", (), {"line": 10, "source_file": "test.facto"})()

        usage = type(
            "MockUsage",
            (),
            {
                "should_materialize": True,
                "resolved_signal_name": "signal-A",
                "resolved_signal_type": "int",
                "producer": const_op,
                "debug_label": "test",
                "source_ast": const_op.source_ast,
            },
        )()
        signal_analyzer.signal_usage["c1"] = usage
        entity_placer.signal_usage = signal_analyzer.signal_usage

        debug_info = entity_placer._build_debug_info(const_op)
        assert debug_info.get("line") == 10

    def test_build_debug_info_with_expr_context(self, entity_placer, signal_analyzer):
        """Test debug info includes expression context."""
        const_op = make_const("c1", 42, "signal-A")
        const_op.debug_metadata = {
            "user_declared": True,
            "expr_context_target": "x",
            "expr_context_line": 15,
            "expr_context_file": "test.facto",
        }

        usage = type(
            "MockUsage",
            (),
            {
                "should_materialize": True,
                "resolved_signal_name": "signal-A",
                "resolved_signal_type": "int",
                "producer": const_op,
                "debug_label": "test",
                "source_ast": None,
            },
        )()
        signal_analyzer.signal_usage["c1"] = usage
        entity_placer.signal_usage = signal_analyzer.signal_usage

        debug_info = entity_placer._build_debug_info(const_op)
        assert debug_info.get("line") == 15
        assert debug_info.get("expr_context") == "x"
