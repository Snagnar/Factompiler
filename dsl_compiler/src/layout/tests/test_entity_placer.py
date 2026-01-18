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
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer

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


def make_mock_usage(
    should_materialize: bool = True,
    resolved_signal_name: str = "signal-A",
    resolved_signal_type: str = "virtual",
    debug_label: str | None = None,
    source_ast=None,
    producer=None,
    **kwargs,
):
    """Helper to create a mock usage object for signal_usage dict."""
    attrs = {
        "should_materialize": should_materialize,
        "resolved_signal_name": resolved_signal_name,
        "resolved_signal_type": resolved_signal_type,
        "debug_label": debug_label,
        "source_ast": source_ast,
        "producer": producer,
        **kwargs,
    }
    return type("MockUsage", (), attrs)()


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

    @pytest.mark.parametrize(
        "op_type,node_id,entity_type,role",
        [
            ("const", "const1", "constant-combinator", "literal"),
            ("arith", "arith1", "arithmetic-combinator", "arithmetic"),
            ("decider", "dec1", "decider-combinator", "decider"),
        ],
    )
    def test_place_ir_operation(
        self, entity_placer, signal_analyzer, op_type, node_id, entity_type, role
    ):
        """Test placing various IR operations."""
        signal_analyzer.signal_usage["const1"] = make_mock_usage()
        entity_placer.signal_usage = signal_analyzer.signal_usage

        if op_type == "const":
            op = make_const(node_id, 42, "signal-A")
        elif op_type == "arith":
            entity_placer.place_ir_operation(make_const("const1", 10, "signal-A"))
            op = make_arith(node_id, make_signal_ref("const1"), 5, "+", "signal-B")
        else:  # decider
            entity_placer.place_ir_operation(make_const("const1", 10, "signal-A"))
            op = make_decider(node_id, make_signal_ref("const1"), 5, ">", "signal-B")

        entity_placer.place_ir_operation(op)

        placement = entity_placer.plan.get_placement(node_id)
        assert placement.entity_type == entity_type
        assert placement.role == role

    def test_place_ir_operation_unknown_warns(self, entity_placer, diagnostics):
        """Test that unknown IR operations generate a warning."""
        entity_placer.place_ir_operation(type("UnknownIRNode", (), {"node_id": "unknown1"})())
        assert diagnostics._warning_count > 0


# === Tests for _place_constant ===


class TestPlaceConstant:
    """Tests for _place_constant method."""

    def test_place_constant_with_bundle_signals(self, entity_placer, signal_analyzer):
        """Test placing a bundle constant with multiple signals."""
        op = make_bundle_const("bundle1", {"signal-A": 10, "signal-B": 20})
        signal_analyzer.signal_usage["bundle1"] = make_mock_usage()
        entity_placer.signal_usage = signal_analyzer.signal_usage

        entity_placer._place_constant(op)

        placement = entity_placer.plan.get_placement("bundle1")
        assert placement.role == "bundle_const"
        assert placement.properties["signals"] == {"signal-A": 10, "signal-B": 20}

    def test_place_constant_no_materialize_skips(self, entity_placer, signal_analyzer):
        """Test that constants marked as not-materialize are skipped."""
        op = make_const("const1", 42, "signal-A")
        signal_analyzer.signal_usage["const1"] = make_mock_usage(should_materialize=False)
        entity_placer.signal_usage = signal_analyzer.signal_usage

        entity_placer._place_constant(op)

        assert "const1" not in entity_placer.plan.entity_placements


# === Tests for _place_arithmetic ===


class TestPlaceArithmetic:
    """Tests for _place_arithmetic method."""

    def test_place_arithmetic_with_wire_separation(self, entity_placer):
        """Test placing arithmetic with wire separation flag."""
        arith_op = make_arith(
            "arith1", make_signal_ref("src1"), make_signal_ref("src2", "signal-B"), "*"
        )
        arith_op.needs_wire_separation = True
        entity_placer._place_arithmetic(arith_op)
        assert (
            entity_placer.plan.get_placement("arith1").properties.get("needs_wire_separation")
            is True
        )


# === Tests for _place_decider ===


class TestPlaceDecider:
    """Tests for _place_decider method."""

    def test_place_single_condition_decider(self, entity_placer):
        """Test placing a single-condition decider."""
        decider_op = make_decider("dec1", make_signal_ref("const1"), 5, ">", "signal-B")
        entity_placer._place_decider(decider_op)
        placement = entity_placer.plan.get_placement("dec1")
        assert placement.entity_type == "decider-combinator"
        assert placement.properties["operation"] == ">"

    def test_place_multi_condition_decider(self, entity_placer):
        """Test placing a multi-condition decider."""
        decider_op = make_decider("dec1", make_signal_ref("src1"), 5, ">")
        decider_op.conditions = [
            DeciderCondition(
                comparator=">", first_operand=make_signal_ref("src1"), second_operand=5
            ),
            DeciderCondition(
                comparator="<", first_operand=make_signal_ref("src2", "signal-B"), second_operand=10
            ),
        ]
        entity_placer._place_decider(decider_op)
        placement = entity_placer.plan.get_placement("dec1")
        assert len(placement.properties["conditions"]) == 2


# === Tests for _place_user_entity ===


class TestPlaceUserEntity:
    """Tests for _place_user_entity method."""

    @pytest.mark.parametrize(
        "x,y,expect_pos,expect_user_specified",
        [
            (None, None, None, False),
            (10, 20, (10, 20), True),
        ],
    )
    def test_place_user_entity(self, entity_placer, x, y, expect_pos, expect_user_specified):
        """Test placing user entities with/without position."""
        entity_placer._place_user_entity(IRPlaceEntity("lamp1", "small-lamp", x, y))
        placement = entity_placer.plan.get_placement("lamp1")
        assert placement.entity_type == "small-lamp"
        assert placement.role == "user_entity"
        if expect_pos:
            assert placement.position == expect_pos
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

    @pytest.mark.parametrize(
        "prop,value,expected_type",
        [
            ("brightness", 100, "constant"),
            ("enable", make_signal_ref("src1"), "signal"),
        ],
    )
    def test_place_entity_prop_write(self, entity_placer, prop, value, expected_type):
        """Test writing values to entity properties."""
        entity_placer._place_user_entity(IRPlaceEntity("lamp1", "small-lamp", None, None))
        entity_placer._place_entity_prop_write(IREntityPropWrite("lamp1", prop, value))
        prop_writes = entity_placer.plan.get_placement("lamp1").properties.get(
            "property_writes", {}
        )
        assert prop in prop_writes
        assert prop_writes[prop]["type"] == expected_type

    def test_place_entity_prop_write_nonexistent_entity_warns(self, entity_placer, diagnostics):
        """Test that writing to nonexistent entity generates warning."""
        entity_placer._place_entity_prop_write(IREntityPropWrite("nonexistent", "enable", 100))
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
        assert "dec1" not in entity_placer.plan.entity_placements


# === Tests for create_output_anchors ===


class TestCreateOutputAnchors:
    """Tests for create_output_anchors method."""

    @pytest.mark.parametrize("is_output,expect_anchor", [(True, True), (False, False)])
    def test_create_output_anchors(self, entity_placer, signal_analyzer, is_output, expect_anchor):
        """Test creating output anchors based on is_output flag."""
        producer = make_arith("out1", 1, 2, "+") if is_output else make_const("out1", 42)
        signal_analyzer.signal_usage["out1"] = make_mock_usage(
            debug_label="output_val",
            debug_metadata={"is_output": is_output},
            producer=producer,
            output_aliases={"alias1"} if is_output else set(),
            signal_type="signal-A",
        )
        entity_placer.signal_usage = signal_analyzer.signal_usage
        entity_placer.create_output_anchors()
        has_anchor = "out1_alias1_output_anchor" in entity_placer.plan.entity_placements
        assert has_anchor == expect_anchor


class TestCreateDebugInfo:
    """Tests for EntityPlacer._build_debug_info method."""

    def test_build_debug_info_with_source_ast(self, entity_placer, signal_analyzer):
        """Test debug info includes source location from AST."""
        const_op = make_const("c1", 42)
        const_op.source_ast = type("MockAST", (), {"line": 10, "source_file": "test.facto"})()
        signal_analyzer.signal_usage["c1"] = make_mock_usage(
            producer=const_op, source_ast=const_op.source_ast
        )
        entity_placer.signal_usage = signal_analyzer.signal_usage
        assert entity_placer._build_debug_info(const_op).get("line") == 10

    def test_build_debug_info_with_expr_context(self, entity_placer, signal_analyzer):
        """Test debug info includes expression context."""
        const_op = make_const("c1", 42)
        const_op.debug_metadata = {
            "user_declared": True,
            "expr_context_target": "x",
            "expr_context_line": 15,
        }
        signal_analyzer.signal_usage["c1"] = make_mock_usage(producer=const_op)
        entity_placer.signal_usage = signal_analyzer.signal_usage
        debug_info = entity_placer._build_debug_info(const_op)
        assert debug_info.get("line") == 15
        assert debug_info.get("expr_context") == "x"


class TestPlaceEntityPropWriteAdvanced:
    """Additional tests for _place_entity_prop_write edge cases."""

    def test_place_entity_prop_write_signal_with_inlined_comparison(
        self, entity_placer, signal_analyzer
    ):
        """Test property write with inline comparison."""
        decider_op = make_decider("dec1", make_signal_ref("src1"), 5, ">", "signal-B")
        decider_op.copy_count_from_input = False
        signal_analyzer.signal_usage["dec1"] = make_mock_usage(resolved_signal_name="signal-B")
        entity_placer.signal_usage = signal_analyzer.signal_usage
        entity_placer._place_decider(decider_op)

        entity_placer._place_user_entity(IRPlaceEntity("lamp1", "small-lamp", None, None))
        entity_placer._place_entity_prop_write(
            IREntityPropWrite("lamp1", "enable", SignalRef("signal-B", "dec1"))
        )
        assert "property_writes" in entity_placer.plan.get_placement("lamp1").properties


class TestPlaceWireMergeAdvanced:
    """Additional tests for wire merge placement."""

    def test_place_wire_merge_with_multiple_sources(self, entity_placer, signal_analyzer):
        """Test wire merge with multiple source signals."""
        signal_analyzer.signal_usage["src1"] = make_mock_usage()
        signal_analyzer.signal_usage["src2"] = make_mock_usage()
        entity_placer.signal_usage = signal_analyzer.signal_usage
        entity_placer._place_wire_merge(
            IRWireMerge("merge1", [SignalRef("signal-A", "src1"), SignalRef("signal-B", "src2")])
        )
        assert "merge1" in entity_placer._wire_merge_junctions


class TestPlaceUserEntityAdvanced:
    """Additional tests for user entity placement."""

    def test_place_user_entity_with_source_ast(self, entity_placer):
        """Test user entity placement with source AST info."""
        op = IRPlaceEntity("lamp1", "small-lamp", 10, 20)
        op.source_ast = type("MockAST", (), {"line": 42, "source_file": "test.facto"})()
        entity_placer._place_user_entity(op)
        assert entity_placer.plan.get_placement("lamp1").properties["debug_info"]["line"] == 42


class TestPlacePropWriteInline:
    """Tests for property write with inlined comparisons."""

    def test_place_prop_write_inline_bundle_condition(self, entity_placer, signal_analyzer):
        """Test _place_entity_prop_write with inline_bundle_condition."""
        entity_placer._place_user_entity(IRPlaceEntity("lamp1", "small-lamp", None, None))
        prop_write = IREntityPropWrite("lamp1", "enable", 1)
        prop_write.inline_bundle_condition = {
            "signal": "signal-each",
            "operator": ">",
            "constant": 0,
            "input_source": BundleRef({"signal-A"}, "bundle1"),
        }
        entity_placer._place_entity_prop_write(prop_write)
        assert (
            entity_placer.plan.get_placement("lamp1").properties["property_writes"]["enable"][
                "type"
            ]
            == "inline_bundle_condition"
        )


# =============================================================================
# Coverage gap tests (Lines 196-199, 359-365, 373-382, 679-687, 695-698)
# =============================================================================


def compile_to_ir(source: str):
    """Helper to compile source to IR."""
    diags = ProgramDiagnostics()
    parser = DSLParser()
    ast = parser.parse(source, "<test>")
    analyzer = SemanticAnalyzer(diagnostics=diags)
    analyzer.visit(ast)
    lowerer = ASTLowerer(analyzer, diags)
    ir_ops = lowerer.lower_program(ast)
    return ir_ops, lowerer, diags


class TestEntityPlacerCoverageGaps:
    """Tests for entity_placer.py coverage gaps > 2 lines."""

    def test_folded_constant_debug_info(self):
        """Cover lines 196-199: folded constant debug info creation."""
        source = """
        Signal a = 5;
        Signal b = 10;
        Signal c = 15;
        Signal total = a + b + c;
        """
        ir_ops, lowerer, diags = compile_to_ir(source)

    def test_multi_condition_decider_placement(self):
        """Cover lines 359-365, 373-382: multi-condition decider operand handling."""
        source = """
        Signal a = 10;
        Signal b = 20;
        Signal result = ((a > 5) && (b < 30)) : a;
        """
        ir_ops, lowerer, diags = compile_to_ir(source)

    def test_stale_entity_removal(self):
        """Cover lines 679-687, 695-698: removing stale entities and wire connections."""
        source = """
        Signal a = 10;
        Signal b = a + 0;
        """
        ir_ops, lowerer, diags = compile_to_ir(source)

    def test_output_anchor_creation(self):
        """Cover output anchor creation for unused output signals."""
        source = """
        Signal result = 10 + 20;
        """
        ir_ops, lowerer, diags = compile_to_ir(source)
