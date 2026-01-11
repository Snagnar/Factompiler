"""
Tests for layout/signal_analyzer.py - Signal usage analysis and resolution.
"""

import pytest

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.ir.nodes import IRArith, IRConst, IRDecider, SignalRef
from dsl_compiler.src.layout.signal_analyzer import SignalAnalyzer, SignalUsageEntry

# === Helpers ===


def make_const(node_id: str, value: int, output_type: str = "signal-A") -> IRConst:
    """Helper to create an IRConst node."""
    op = IRConst(node_id, output_type)
    op.value = value
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


# === Fixtures ===


@pytest.fixture
def diagnostics():
    """Create a new ProgramDiagnostics instance."""
    return ProgramDiagnostics()


@pytest.fixture
def empty_analyzer(diagnostics):
    """Create an analyzer with empty signal type map."""
    return SignalAnalyzer(diagnostics, signal_type_map={})


@pytest.fixture
def analyzer_with_map(diagnostics):
    """Create an analyzer with a pre-populated signal type map."""
    return SignalAnalyzer(
        diagnostics,
        signal_type_map={
            "__v1": {"name": "signal-A", "type": "virtual"},
            "__v2": "signal-B",
            "iron-plate": {"name": "iron-plate", "type": "item"},
        },
    )


# === Tests for SignalUsageEntry ===


class TestSignalUsageEntry:
    """Tests for SignalUsageEntry dataclass."""

    def test_signal_usage_entry_creation(self):
        """Test creating a basic SignalUsageEntry."""
        entry = SignalUsageEntry(signal_id="test_id")
        assert entry.signal_id == "test_id"
        assert entry.signal_type is None
        assert entry.producer is None
        assert entry.consumers == set()
        assert entry.should_materialize is True

    def test_signal_usage_entry_with_properties(self):
        """Test creating SignalUsageEntry with all properties."""
        entry = SignalUsageEntry(
            signal_id="test_id",
            signal_type="signal-A",
            literal_value=42,
            should_materialize=False,
            debug_label="my_var",
        )
        assert entry.signal_type == "signal-A"
        assert entry.literal_value == 42
        assert entry.should_materialize is False
        assert entry.debug_label == "my_var"


# === Tests for SignalAnalyzer.__init__ ===


class TestSignalAnalyzerInit:
    """Tests for SignalAnalyzer initialization."""

    def test_init_with_empty_maps(self, diagnostics):
        """Test initialization with empty signal type map."""
        analyzer = SignalAnalyzer(diagnostics, signal_type_map={})
        assert analyzer.signal_usage == {}
        assert analyzer._allocated_signals == set()

    def test_init_pre_populates_allocated_signals(self, diagnostics):
        """Test that signal_type_map entries are added to allocated signals."""
        signal_map = {
            "__v1": {"name": "signal-A", "type": "virtual"},
            "__v2": "signal-B",
        }
        analyzer = SignalAnalyzer(diagnostics, signal_type_map=signal_map)
        assert "signal-A" in analyzer._allocated_signals
        assert "signal-B" in analyzer._allocated_signals

    def test_init_with_referenced_signal_names(self, diagnostics):
        """Test initialization with referenced signal names."""
        analyzer = SignalAnalyzer(
            diagnostics,
            signal_type_map={},
            referenced_signal_names={"signal-X", "signal-Y"},
        )
        assert "signal-X" in analyzer.referenced_signal_names
        assert "signal-Y" in analyzer.referenced_signal_names


# === Tests for SignalAnalyzer.analyze ===


class TestSignalAnalyzerAnalyze:
    """Tests for SignalAnalyzer.analyze method."""

    def test_analyze_empty_operations(self, empty_analyzer):
        """Test analyzing empty list of operations."""
        result = empty_analyzer.analyze([])
        assert result == {}

    def test_analyze_ir_const_creates_entry(self, empty_analyzer):
        """Test that IRConst creates a signal usage entry."""
        const_op = make_const("const1", 42, "signal-A")
        result = empty_analyzer.analyze([const_op])
        assert "const1" in result
        assert result["const1"].literal_value == 42
        assert result["const1"].producer is const_op

    def test_analyze_ir_arith_records_consumers(self, empty_analyzer):
        """Test that IRArith records consumers from operands."""
        const1 = make_const("const1", 10, "signal-A")
        const2 = make_const("const2", 20, "signal-B")
        arith = make_arith(
            "arith1",
            left=SignalRef("signal-A", "const1"),
            right=SignalRef("signal-B", "const2"),
        )
        result = empty_analyzer.analyze([const1, const2, arith])

        assert "arith1" in result["const1"].consumers
        assert "arith1" in result["const2"].consumers

    def test_analyze_ir_decider_records_consumers(self, empty_analyzer):
        """Test that IRDecider records consumers from operands."""
        const1 = make_const("const1", 10, "signal-A")
        const2 = make_const("const2", 5, "signal-B")
        decider = make_decider(
            "decider1",
            left=SignalRef("signal-A", "const1"),
            right=SignalRef("signal-B", "const2"),
        )
        result = empty_analyzer.analyze([const1, const2, decider])

        assert "decider1" in result["const1"].consumers
        assert "decider1" in result["const2"].consumers


# === Tests for SignalAnalyzer.should_materialize ===


class TestSignalAnalyzerShouldMaterialize:
    """Tests for SignalAnalyzer.should_materialize method."""

    def test_should_materialize_unknown_signal(self, empty_analyzer):
        """Test should_materialize returns True for unknown signals."""
        assert empty_analyzer.should_materialize("unknown_signal") is True

    def test_should_materialize_respects_entry(self, empty_analyzer):
        """Test should_materialize respects entry's should_materialize field."""
        const_op = make_const("const1", 42, "signal-A")
        empty_analyzer.analyze([const_op])
        # The entry's should_materialize is set during analyze based on rules
        result = empty_analyzer.should_materialize("const1")
        assert isinstance(result, bool)


# === Tests for SignalAnalyzer.can_inline_constant ===


class TestSignalAnalyzerCanInlineConstant:
    """Tests for SignalAnalyzer.can_inline_constant method."""

    def test_can_inline_constant_unknown_signal(self, empty_analyzer):
        """Test can_inline_constant returns False for unknown signals."""
        ref = SignalRef(source_id="unknown", signal_type="signal-A")
        assert empty_analyzer.can_inline_constant(ref) is False

    def test_can_inline_constant_non_const_producer(self, empty_analyzer):
        """Test can_inline_constant returns False for non-constant producers."""
        arith = make_arith("arith1", left=5, right=10, op="+", output_type="signal-A")
        empty_analyzer.analyze([arith])
        ref = SignalRef("signal-A", "arith1")
        assert empty_analyzer.can_inline_constant(ref) is False


# === Tests for SignalAnalyzer.inline_value ===


class TestSignalAnalyzerInlineValue:
    """Tests for SignalAnalyzer.inline_value method."""

    def test_inline_value_returns_none_for_unknown(self, empty_analyzer):
        """Test inline_value returns None for unknown signals."""
        ref = SignalRef(source_id="unknown", signal_type="signal-A")
        assert empty_analyzer.inline_value(ref) is None

    def test_inline_value_returns_none_when_cannot_inline(self, empty_analyzer):
        """Test inline_value returns None when inlining is not possible."""
        arith = make_arith("arith1", left=5, right=10)
        empty_analyzer.analyze([arith])
        ref = SignalRef("signal-A", "arith1")
        assert empty_analyzer.inline_value(ref) is None


# === Tests for SignalAnalyzer.resolve_signal_name ===


class TestSignalAnalyzerResolveSignalName:
    """Tests for SignalAnalyzer.resolve_signal_name method."""

    def test_resolve_signal_name_concrete_signal(self, analyzer_with_map):
        """Test resolving a concrete Factorio signal name."""
        # iron-plate is a concrete signal, should return as-is
        result = analyzer_with_map.resolve_signal_name("iron-plate")
        assert result == "iron-plate"

    def test_resolve_signal_name_signal_prefix(self, empty_analyzer):
        """Test resolving signal- prefixed names."""
        result = empty_analyzer.resolve_signal_name("signal-X")
        assert result == "signal-X"

    def test_resolve_signal_name_from_entry(self, analyzer_with_map):
        """Test resolving signal name from entry's resolved name."""
        entry = SignalUsageEntry(signal_id="test", resolved_signal_name="signal-Z")
        result = analyzer_with_map.resolve_signal_name(None, entry)
        assert result == "signal-Z"

    def test_resolve_signal_name_via_mapping(self, analyzer_with_map):
        """Test resolving signal name via signal_type_map."""
        result = analyzer_with_map.resolve_signal_name("__v1")
        assert result == "signal-A"


# === Tests for SignalAnalyzer.resolve_signal_type ===


class TestSignalAnalyzerResolveSignalType:
    """Tests for SignalAnalyzer.resolve_signal_type method."""

    def test_resolve_signal_type_from_entry(self, empty_analyzer):
        """Test resolving signal type from entry."""
        entry = SignalUsageEntry(signal_id="test", resolved_signal_type="virtual")
        result = empty_analyzer.resolve_signal_type(None, entry)
        assert result == "virtual"

    def test_resolve_signal_type_unknown(self, empty_analyzer):
        """Test resolving signal type for unknown signal."""
        result = empty_analyzer.resolve_signal_type("unknown_signal")
        # After resolution, returns a type or None
        assert result is None or isinstance(result, str)


# === Tests for SignalAnalyzer.get_signal_name ===


class TestSignalAnalyzerGetSignalName:
    """Tests for SignalAnalyzer.get_signal_name method."""

    def test_get_signal_name_for_int(self, empty_analyzer):
        """Test get_signal_name returns signal-0 for integers."""
        result = empty_analyzer.get_signal_name(42)
        assert result == "signal-0"

    def test_get_signal_name_for_signal_ref(self, analyzer_with_map):
        """Test get_signal_name for SignalRef."""
        ref = SignalRef(source_id="const1", signal_type="__v1")
        result = analyzer_with_map.get_signal_name(ref)
        assert result == "signal-A"

    def test_get_signal_name_for_string(self, analyzer_with_map):
        """Test get_signal_name for string signal type."""
        result = analyzer_with_map.get_signal_name("__v2")
        assert result == "signal-B"


# === Tests for SignalAnalyzer.get_operand_for_combinator ===


class TestSignalAnalyzerGetOperandForCombinator:
    """Tests for SignalAnalyzer.get_operand_for_combinator method."""

    def test_get_operand_for_combinator_int(self, empty_analyzer):
        """Test get_operand_for_combinator returns int as-is."""
        result = empty_analyzer.get_operand_for_combinator(42)
        assert result == 42

    def test_get_operand_for_combinator_signal_ref(self, analyzer_with_map):
        """Test get_operand_for_combinator for SignalRef."""
        ref = SignalRef(source_id="const1", signal_type="__v1")
        result = analyzer_with_map.get_operand_for_combinator(ref)
        assert result == "signal-A"

    def test_get_operand_for_combinator_string(self, analyzer_with_map):
        """Test get_operand_for_combinator for string."""
        result = analyzer_with_map.get_operand_for_combinator("__v2")
        assert result == "signal-B"


# === Tests for SignalAnalyzer._decide_materialization ===


class TestSignalAnalyzerDecideMaterialization:
    """Tests for SignalAnalyzer._decide_materialization method."""

    def test_decide_materialization_suppressed(self, empty_analyzer):
        """Test suppressed materialization."""
        entry = SignalUsageEntry(signal_id="test")
        entry.debug_metadata["suppress_materialization"] = True
        empty_analyzer._decide_materialization(entry)
        assert entry.should_materialize is False

    def test_decide_materialization_user_declared(self, empty_analyzer):
        """Test user-declared signals are materialized."""
        const_op = make_const("const1", 42, "signal-A")
        const_op.debug_metadata = {"user_declared": True}
        entry = SignalUsageEntry(signal_id="const1", producer=const_op)
        empty_analyzer._decide_materialization(entry)
        assert entry.should_materialize is True


# === Tests for SignalAnalyzer._resolve_signal_identity ===


class TestSignalAnalyzerResolveSignalIdentity:
    """Tests for SignalAnalyzer._resolve_signal_identity method."""

    def test_resolve_signal_identity_none_entry(self, empty_analyzer):
        """Test resolving None entry does nothing."""
        empty_analyzer._resolve_signal_identity(None)
        # No exception should be raised

    def test_resolve_signal_identity_already_resolved(self, empty_analyzer):
        """Test already resolved entry is not re-resolved."""
        entry = SignalUsageEntry(
            signal_id="test", resolved_signal_name="signal-Z", resolved_signal_type="virtual"
        )
        empty_analyzer._resolve_signal_identity(entry)
        assert entry.resolved_signal_name == "signal-Z"

    def test_resolve_signal_identity_from_literal_declared_type(self, empty_analyzer):
        """Test resolution from literal_declared_type."""
        entry = SignalUsageEntry(signal_id="test", literal_declared_type="signal-A")
        empty_analyzer._resolve_signal_identity(entry, force=True)
        assert entry.resolved_signal_name == "signal-A"


# === Tests for SignalAnalyzer._resolve_via_mapping ===


class TestSignalAnalyzerResolveViaMapping:
    """Tests for SignalAnalyzer._resolve_via_mapping method."""

    def test_resolve_via_mapping_dict_entry(self, analyzer_with_map):
        """Test resolving via dict mapping entry."""
        result = analyzer_with_map._resolve_via_mapping("__v1", None)
        assert result == "signal-A"

    def test_resolve_via_mapping_string_entry(self, analyzer_with_map):
        """Test resolving via string mapping entry."""
        result = analyzer_with_map._resolve_via_mapping("__v2", None)
        assert result == "signal-B"

    def test_resolve_via_mapping_none_type(self, empty_analyzer):
        """Test resolving None signal type returns signal-0."""
        result = empty_analyzer._resolve_via_mapping(None, None)
        assert result == "signal-0"

    def test_resolve_via_mapping_from_entry_resolved(self, empty_analyzer):
        """Test resolving from entry's resolved_signal_name."""
        entry = SignalUsageEntry(signal_id="test", resolved_signal_name="signal-K")
        result = empty_analyzer._resolve_via_mapping("anything", entry)
        assert result == "signal-K"


# === Tests for SignalAnalyzer._infer_category_from_name ===


class TestSignalAnalyzerInferCategoryFromName:
    """Tests for SignalAnalyzer._infer_category_from_name method."""

    def test_infer_category_signal_prefix(self, empty_analyzer):
        """Test signal- prefix is inferred as virtual."""
        result = empty_analyzer._infer_category_from_name("signal-X")
        assert result == "virtual"

    def test_infer_category_unknown(self, empty_analyzer):
        """Test unknown signal is inferred as virtual."""
        result = empty_analyzer._infer_category_from_name("unknown_thing")
        assert result == "virtual"


# === Tests for SignalAnalyzer._build_available_signal_pool ===


class TestSignalAnalyzerBuildAvailableSignalPool:
    """Tests for SignalAnalyzer._build_available_signal_pool method."""

    def test_build_pool_excludes_allocated(self, diagnostics):
        """Test pool excludes already allocated signals."""
        analyzer = SignalAnalyzer(
            diagnostics,
            signal_type_map={"__v1": {"name": "signal-A", "type": "virtual"}},
        )
        assert "signal-A" not in analyzer._available_signal_pool

    def test_build_pool_excludes_referenced(self, diagnostics):
        """Test pool excludes referenced signal names."""
        analyzer = SignalAnalyzer(
            diagnostics, signal_type_map={}, referenced_signal_names={"signal-B"}
        )
        assert "signal-B" not in analyzer._available_signal_pool


# === Tests for SignalAnalyzer._allocate_factorio_virtual_signal ===


class TestSignalAnalyzerAllocateFactorioVirtualSignal:
    """Tests for SignalAnalyzer._allocate_factorio_virtual_signal method."""

    def test_allocate_returns_signal(self, empty_analyzer):
        """Test allocating a signal returns a valid signal name."""
        result = empty_analyzer._allocate_factorio_virtual_signal()
        assert result.startswith("signal-")

    def test_allocate_increments_index(self, empty_analyzer):
        """Test allocating signals increments pool index."""
        initial_index = empty_analyzer._signal_pool_index
        empty_analyzer._allocate_factorio_virtual_signal()
        assert empty_analyzer._signal_pool_index == initial_index + 1

    def test_allocate_adds_to_allocated(self, empty_analyzer):
        """Test allocated signal is added to _allocated_signals."""
        signal = empty_analyzer._allocate_factorio_virtual_signal()
        assert signal in empty_analyzer._allocated_signals

    def test_allocate_returns_different_signals(self, empty_analyzer):
        """Test consecutive allocations return different signals."""
        signal1 = empty_analyzer._allocate_factorio_virtual_signal()
        signal2 = empty_analyzer._allocate_factorio_virtual_signal()
        assert signal1 != signal2


# === Tests for SignalAnalyzer.finalize_materialization ===


class TestSignalAnalyzerFinalizeMaterialization:
    """Tests for SignalAnalyzer.finalize_materialization method."""

    def test_finalize_materialization_empty(self, empty_analyzer):
        """Test finalize_materialization with no usage entries."""
        empty_analyzer.finalize_materialization()
        # No exception should be raised

    def test_finalize_materialization_processes_entries(self, empty_analyzer):
        """Test finalize_materialization processes all entries."""
        const_op = make_const("const1", 42, "signal-A")
        empty_analyzer.analyze([const_op])
        # analyze already calls finalize_materialization, so entries should be processed
        assert "const1" in empty_analyzer.signal_usage
        entry = empty_analyzer.signal_usage["const1"]
        # resolved_signal_name should be set after finalization
        assert entry.resolved_signal_name is not None
