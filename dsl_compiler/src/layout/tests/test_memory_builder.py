"""Tests for layout/memory_builder.py — aligned with the archetype-based rewrite."""

import pytest

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.ir.nodes import (
    MEMORY_TYPE_RS_LATCH,
    MEMORY_TYPE_SR_LATCH,
    IRLatchWrite,
    IRMemCreate,
    IRMemRead,
    IRMemWrite,
    SignalRef,
)
from dsl_compiler.src.layout.layout_plan import LayoutPlan
from dsl_compiler.src.layout.memory_builder import MemoryBuilder, MemoryModule
from dsl_compiler.src.layout.signal_analyzer import SignalAnalyzer
from dsl_compiler.src.layout.signal_graph import SignalGraph
from dsl_compiler.src.layout.tile_grid import TileGrid
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer


@pytest.fixture
def diagnostics():
    return ProgramDiagnostics()


@pytest.fixture
def builder(diagnostics):
    return MemoryBuilder(TileGrid(), LayoutPlan(), SignalAnalyzer(diagnostics, {}), diagnostics)


# =============================================================================
# MemoryModule dataclass
# =============================================================================


def test_memory_module_defaults():
    m = MemoryModule("id", "signal-A")
    assert m.memory_id == "id"
    assert m.signal_type == "signal-A"
    assert m.archetype == "pending"
    assert m.primary is None
    assert m.secondary is None
    assert m.read_source_id is None
    assert m.latch_type is None
    assert m._feedback_signal_ids == []


# =============================================================================
# register_ir_node
# =============================================================================


def test_register_ir_node(builder):
    op = IRMemCreate("mem1", "signal-A")
    builder.register_ir_node(op)
    assert op.node_id in builder._ir_nodes


# =============================================================================
# create_memory — deferred placement
# =============================================================================


def test_create_memory_returns_pending_module(builder):
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)
    assert isinstance(module, MemoryModule)
    assert module.archetype == "pending"
    assert module.primary is None
    assert module.secondary is None


def test_create_memory_stores_in_modules(builder):
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    builder.create_memory(op, sg)
    assert "mem1" in builder._modules


# =============================================================================
# handle_read
# =============================================================================


def test_handle_read_records_source(builder):
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    builder.create_memory(op, sg)

    read = IRMemRead("read1", "signal-A")
    read.memory_id = "mem1"
    builder.handle_read(read, sg)
    assert "read1" in builder._read_sources


def test_handle_read_resolves_immediately_after_write(builder):
    """When write has already been processed, handle_read wires immediately."""
    from dsl_compiler.src.ir.nodes import IRConst

    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    builder.create_memory(op, sg)

    # Write first (always-write, no dependency → pass_through)
    const = IRConst("const1", "signal-W")
    const.value = 1
    builder.register_ir_node(const)
    write = IRMemWrite("mem1", SignalRef("signal-A", "src1"), SignalRef("signal-W", "const1"))
    builder.handle_write(write, sg)

    # Now read — should resolve immediately
    read = IRMemRead("read1", "signal-A")
    read.memory_id = "mem1"
    builder.handle_read(read, sg)
    assert "read1" in sg._sources


def test_handle_read_undefined_memory(builder):
    """Reading from undefined memory warns but does not crash."""
    sg = SignalGraph()
    read = IRMemRead("read1", "signal-A")
    read.memory_id = "nonexistent"
    builder.handle_read(read, sg)


# =============================================================================
# handle_write — archetype dispatch
# =============================================================================


def test_handle_write_undefined_memory(builder):
    """Writing to undefined memory warns but does not crash."""
    sg = SignalGraph()
    write = IRMemWrite("nonexistent", SignalRef("signal-A", "src1"), SignalRef("signal-B", "src2"))
    builder.handle_write(write, sg)


def test_handle_write_gated(builder):
    """Conditional write creates gated memory (archetype B)."""
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    data = SignalRef("signal-A", "src1")
    we = SignalRef("signal-W", "src2")
    write = IRMemWrite("mem1", data, we)
    builder.handle_write(write, sg)

    assert module.archetype == "gated"
    assert module.primary is not None  # storage
    assert module.secondary is not None  # gate
    assert module.read_source_id == "mem1_storage"


def test_handle_write_pass_through(builder):
    """Always-write without self-dependency creates pass-through (archetype A')."""
    from dsl_compiler.src.ir.nodes import IRConst

    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    const = IRConst("const1", "signal-W")
    const.value = 1
    builder.register_ir_node(const)

    data = SignalRef("signal-A", "src1")
    we = SignalRef("signal-W", "const1")
    write = IRMemWrite("mem1", data, we)
    builder.handle_write(write, sg)

    assert module.archetype == "pass_through"
    assert module.primary is not None
    assert module.secondary is None
    assert module.read_source_id == "mem1_pass_through"


def test_handle_write_accumulator(builder):
    """Always-write with self-dependency creates accumulator (archetype A)."""
    from dsl_compiler.src.ir.nodes import IRArith, IRConst

    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    # Read from memory
    read = IRMemRead("read1", "signal-A")
    read.memory_id = "mem1"
    builder.handle_read(read, sg)

    # Arithmetic: mem + 1
    arith = IRArith("arith1", "signal-A")
    arith.op = "+"
    arith.left = SignalRef("signal-A", "read1")
    arith.right = 1
    builder.register_ir_node(arith)

    # Create placement for arith (accumulator needs existing placement)
    builder.layout_plan.create_and_add_placement(
        ir_node_id="arith1",
        entity_type="arithmetic-combinator",
        position=(0, 0),
        role="arith",
        properties={"debug_info": {}},
    )

    # Always-write (const=1)
    const = IRConst("const1", "signal-W")
    const.value = 1
    builder.register_ir_node(const)

    write = IRMemWrite("mem1", SignalRef("signal-A", "arith1"), SignalRef("signal-W", "const1"))
    builder.handle_write(write, sg)

    assert module.archetype == "accumulator"
    assert module.primary is not None
    assert module.read_source_id == "arith1"


def test_handle_write_resolves_deferred_reads(builder):
    """Reads recorded before write get resolved when the write fires."""
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    builder.create_memory(op, sg)

    # Read before write
    read = IRMemRead("read1", "signal-A")
    read.memory_id = "mem1"
    builder.handle_read(read, sg)
    assert "read1" not in sg._sources  # Not resolved yet

    # Conditional write → gated
    write = IRMemWrite("mem1", SignalRef("signal-A", "src1"), SignalRef("signal-W", "src2"))
    builder.handle_write(write, sg)

    # Now read should be resolved to storage
    assert "read1" in sg._sources


# =============================================================================
# handle_latch_write — archetype C
# =============================================================================


def test_handle_latch_write_rs(builder):
    """RS latch (reset priority) creates multi-condition decider."""
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    latch = IRLatchWrite(
        "mem1",
        1,
        SignalRef("signal-S", "set_src"),
        SignalRef("signal-R", "reset_src"),
        MEMORY_TYPE_RS_LATCH,
    )
    builder.handle_latch_write(latch, sg)

    assert module.archetype == "latch"
    assert module.latch_type == MEMORY_TYPE_RS_LATCH
    assert module.primary is not None
    assert module.read_source_id == "mem1_latch"


def test_handle_latch_write_sr(builder):
    """SR latch (set priority) creates multi-condition decider."""
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    latch = IRLatchWrite(
        "mem1",
        1,
        SignalRef("signal-S", "set_src"),
        SignalRef("signal-R", "reset_src"),
        MEMORY_TYPE_SR_LATCH,
    )
    builder.handle_latch_write(latch, sg)

    assert module.archetype == "latch"
    assert module.latch_type == MEMORY_TYPE_SR_LATCH
    assert module.primary is not None


def test_handle_latch_write_with_multiplier(builder):
    """Latch value != 1 creates multiplier (secondary combinator)."""
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    latch = IRLatchWrite(
        "mem1",
        5,
        SignalRef("signal-S", "set_src"),
        SignalRef("signal-R", "reset_src"),
        MEMORY_TYPE_RS_LATCH,
    )
    builder.handle_latch_write(latch, sg)

    assert module.secondary is not None  # multiplier
    assert module.read_source_id == "mem1_multiplier"


def test_handle_latch_write_signal_value(builder):
    """Latch with SignalRef value also creates multiplier."""
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    latch = IRLatchWrite(
        "mem1",
        SignalRef("signal-V", "val_src"),
        SignalRef("signal-S", "set_src"),
        SignalRef("signal-R", "reset_src"),
        MEMORY_TYPE_RS_LATCH,
    )
    builder.handle_latch_write(latch, sg)

    assert module.secondary is not None  # multiplier


def test_handle_latch_write_value_one_no_multiplier(builder):
    """Latch with value=1 does NOT create multiplier."""
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    latch = IRLatchWrite(
        "mem1",
        1,
        SignalRef("signal-S", "set_src"),
        SignalRef("signal-R", "reset_src"),
        MEMORY_TYPE_RS_LATCH,
    )
    builder.handle_latch_write(latch, sg)

    assert module.secondary is None  # no multiplier
    assert module.read_source_id == "mem1_latch"


def test_handle_latch_write_creates_feedback_wire(builder):
    """Latch creates GREEN self-feedback wire."""
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    builder.create_memory(op, sg)

    latch = IRLatchWrite(
        "mem1",
        1,
        SignalRef("signal-S", "set_src"),
        SignalRef("signal-R", "reset_src"),
        MEMORY_TYPE_RS_LATCH,
    )
    builder.handle_latch_write(latch, sg)

    # Should have a green self-feedback wire
    green_wires = [
        w
        for w in builder.layout_plan.wire_connections
        if w.wire_color == "green"
        and w.source_entity_id == "mem1_latch"
        and w.sink_entity_id == "mem1_latch"
    ]
    assert len(green_wires) == 1


def test_handle_latch_write_feedback_signal_ids(builder):
    """Latch registers internal feedback signal ID."""
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    latch = IRLatchWrite(
        "mem1",
        1,
        SignalRef("signal-S", "set_src"),
        SignalRef("signal-R", "reset_src"),
        MEMORY_TYPE_RS_LATCH,
    )
    builder.handle_latch_write(latch, sg)

    assert len(module._feedback_signal_ids) > 0
    assert module._feedback_signal_ids[0].startswith("__feedback_")


def test_handle_latch_write_undefined_memory(builder):
    """Latch write for undefined memory warns but does not crash."""
    sg = SignalGraph()
    latch = IRLatchWrite(
        "nonexistent",
        1,
        SignalRef("signal-S", "set_src"),
        SignalRef("signal-R", "reset_src"),
        MEMORY_TYPE_RS_LATCH,
    )
    builder.handle_latch_write(latch, sg)


# =============================================================================
# Inlined latch conditions (set and reset are simple comparisons)
# =============================================================================


def test_inlined_latch_sr(builder):
    """SR inlined latch: ((L > 0) AND (HOLD_COND)) OR (SET_COND)."""
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    set_cond = (SignalRef("signal-X", "x_src"), "<", 20)
    reset_cond = (SignalRef("signal-X", "x_src"), ">=", 80)

    latch = IRLatchWrite(
        "mem1",
        1,
        SignalRef("signal-S", "set_src"),
        SignalRef("signal-R", "reset_src"),
        MEMORY_TYPE_SR_LATCH,
        set_condition=set_cond,
        reset_condition=reset_cond,
    )
    builder.handle_latch_write(latch, sg)

    assert module.archetype == "latch"
    assert module.latch_type == MEMORY_TYPE_SR_LATCH
    placement = module.primary
    conditions = placement.properties["conditions"]
    assert len(conditions) == 3
    # SR: first is L > 0, second is hold (AND), third is set (OR)
    assert conditions[0]["first_signal"] == "signal-A"
    assert conditions[1]["compare_type"] == "and"
    assert conditions[2]["compare_type"] == "or"


def test_inlined_latch_rs(builder):
    """RS inlined latch: ((SET_COND) OR (L > 0)) AND (HOLD_COND)."""
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    set_cond = (SignalRef("signal-X", "x_src"), "<", 20)
    reset_cond = (SignalRef("signal-X", "x_src"), ">=", 80)

    latch = IRLatchWrite(
        "mem1",
        1,
        SignalRef("signal-S", "set_src"),
        SignalRef("signal-R", "reset_src"),
        MEMORY_TYPE_RS_LATCH,
        set_condition=set_cond,
        reset_condition=reset_cond,
    )
    builder.handle_latch_write(latch, sg)

    assert module.archetype == "latch"
    placement = module.primary
    conditions = placement.properties["conditions"]
    assert len(conditions) == 3
    # RS: first is set condition, second is L > 0 (OR), third is hold (AND)
    assert conditions[0]["comparator"] == "<"  # set
    assert conditions[1]["compare_type"] == "or"
    assert conditions[2]["compare_type"] == "and"


# =============================================================================
# finalize
# =============================================================================


def test_finalize_warns_on_never_written(builder):
    """finalize warns about memories that were declared but never written."""
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    builder.create_memory(op, sg)

    builder.finalize(builder.layout_plan, sg)

    assert builder.diagnostics.warning_count() > 0


def test_finalize_resolves_straggler_reads(builder):
    """finalize resolves remaining deferred reads."""
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    builder.create_memory(op, sg)

    # Read and write
    read = IRMemRead("read1", "signal-A")
    read.memory_id = "mem1"
    builder.handle_read(read, sg)

    write = IRMemWrite("mem1", SignalRef("signal-A", "src1"), SignalRef("signal-W", "src2"))
    builder.handle_write(write, sg)

    # finalize should not crash
    builder.finalize(builder.layout_plan, sg)


# =============================================================================
# _operation_depends_on_memory
# =============================================================================


def test_operation_depends_on_memory(builder):
    """Transitive dependency detection from arith -> read -> memory."""
    from dsl_compiler.src.ir.nodes import IRArith

    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    builder.create_memory(op, sg)

    read = IRMemRead("read1", "signal-A")
    read.memory_id = "mem1"
    builder.handle_read(read, sg)

    arith = IRArith("arith1", "signal-B")
    arith.op = "+"
    arith.left = SignalRef("signal-A", "read1")
    arith.right = 1
    builder.register_ir_node(arith)

    assert builder._operation_depends_on_memory("arith1", "mem1")
    assert not builder._operation_depends_on_memory("nonexistent", "mem1")


# =============================================================================
# Gated memory wire connections
# =============================================================================


def test_gated_memory_creates_feedback_wires(builder):
    """Gated memory creates RED self-feedback on storage + gate->storage wire."""
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    builder.create_memory(op, sg)

    write = IRMemWrite("mem1", SignalRef("signal-A", "src1"), SignalRef("signal-W", "src2"))
    builder.handle_write(write, sg)

    # Storage self-feedback (RED)
    self_wires = [
        w
        for w in builder.layout_plan.wire_connections
        if w.wire_color == "red"
        and w.source_entity_id == "mem1_storage"
        and w.sink_entity_id == "mem1_storage"
    ]
    assert len(self_wires) == 1

    # Gate -> storage (RED)
    g2s_wires = [
        w
        for w in builder.layout_plan.wire_connections
        if w.wire_color == "red"
        and w.source_entity_id == "mem1_gate"
        and w.sink_entity_id == "mem1_storage"
    ]
    assert len(g2s_wires) == 1


def test_gated_memory_feedback_signal_ids(builder):
    """Gated memory registers internal feedback signal IDs."""
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    write = IRMemWrite("mem1", SignalRef("signal-A", "src1"), SignalRef("signal-W", "src2"))
    builder.handle_write(write, sg)

    assert len(module._feedback_signal_ids) > 0
    assert module._feedback_signal_ids[0].startswith("__feedback_")


# =============================================================================
# Comparison inversion
# =============================================================================


def test_invert_comparison():
    assert MemoryBuilder._invert_comparison("<", 10) == (">=", 10)
    assert MemoryBuilder._invert_comparison("<=", 10) == (">", 10)
    assert MemoryBuilder._invert_comparison(">", 10) == ("<=", 10)
    assert MemoryBuilder._invert_comparison(">=", 10) == ("<", 10)
    assert MemoryBuilder._invert_comparison("==", 10) == ("!=", 10)
    assert MemoryBuilder._invert_comparison("!=", 10) == ("==", 10)


# =============================================================================
# _needs_multiplier
# =============================================================================


def test_needs_multiplier():
    assert (
        MemoryBuilder._needs_multiplier(IRLatchWrite("m", 1, None, None, MEMORY_TYPE_RS_LATCH))
        is False
    )
    assert (
        MemoryBuilder._needs_multiplier(IRLatchWrite("m", 5, None, None, MEMORY_TYPE_RS_LATCH))
        is True
    )
    assert (
        MemoryBuilder._needs_multiplier(
            IRLatchWrite("m", SignalRef("sig", "src"), None, None, MEMORY_TYPE_RS_LATCH)
        )
        is True
    )


# =============================================================================
# Full-pipeline integration tests (compile_to_ir)
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


class TestMemoryBuilderCoverageGaps:
    """Integration tests via full compilation pipeline."""

    def test_optimized_latch_write_inlined(self):
        """Latch with inlined conditions (same signal for set/reset)."""
        source = """
        Memory battery: "signal-A";
        Signal level = 50;
        battery.write(1, set=level < 20, reset=level >= 80);
        Signal state = battery.read();
        """
        ir_ops, lowerer, diags = compile_to_ir(source)

    def test_latch_with_multiplier(self):
        """Latch with non-1 value requiring multiplier."""
        source = """
        Memory counter: "signal-A";
        Signal trigger = 1;
        counter.write(5, set=trigger > 0, reset=trigger < 0);
        Signal value = counter.read();
        """
        ir_ops, lowerer, diags = compile_to_ir(source)

    def test_latch_with_signal_value(self):
        """Latch where value is a signal reference."""
        source = """
        Memory store: "signal-A";
        Signal input_val = 42;
        Signal trigger = 1;
        store.write(input_val, set=trigger > 0, reset=trigger == 0);
        Signal output = store.read();
        """
        ir_ops, lowerer, diags = compile_to_ir(source)

    def test_standard_latch_fallback(self):
        """Standard (non-inlined) latch path."""
        source = """
        Memory mem: "signal-A";
        Signal set_signal = 1;
        Signal reset_signal = 0;
        mem.write(1, set=set_signal > 0, reset=reset_signal > 0);
        Signal out = mem.read();
        """
        ir_ops, lowerer, diags = compile_to_ir(source)

    def test_standard_write_setup(self):
        """Standard gated write (conditional)."""
        source = """
        Memory mem: "signal-A";
        Signal data = 100;
        Signal enable = 1;
        mem.write(data, when=enable > 0);
        Signal out = mem.read();
        """
        ir_ops, lowerer, diags = compile_to_ir(source)

    def test_memory_depends_on_memory_chain(self):
        """Chain: mem1 -> read -> write -> mem2."""
        source = """
        Memory m1: "signal-A";
        Memory m2: "signal-B";
        Signal x = 10;
        m1.write(x);
        Signal v1 = m1.read();
        m2.write(v1);
        Signal result = m2.read();
        """
        ir_ops, lowerer, diags = compile_to_ir(source)


class TestOptimizedLatchWriteInlinedConditions:
    """Tests for optimized latch write with inlined conditions."""

    def test_optimized_latch_same_signal_set_reset(self):
        """Inlined latch when same signal for set/reset."""
        source = """
        Memory battery_low: "signal-A";
        Signal battery = 50;
        battery_low.write(1, set=battery < 20, reset=battery >= 80);
        Signal is_low = battery_low.read();
        """
        ir_ops, lowerer, diags = compile_to_ir(source)
        assert not diags.has_errors()

    def test_optimized_latch_with_inverted_reset_condition(self):
        """Condition inversion: reset=battery>90 -> hold=battery<=90."""
        source = """
        Memory charging: "signal-B";
        Signal level = 70;
        charging.write(1, set=level <= 30, reset=level > 90);
        Signal is_charging = charging.read();
        """
        ir_ops, lowerer, diags = compile_to_ir(source)
        assert not diags.has_errors()

    def test_optimized_latch_with_multiplier_value(self):
        """Inlined latch with non-1 value requiring multiplier."""
        source = """
        Memory counter: "signal-C";
        Signal trigger = 100;
        counter.write(5, set=trigger > 50, reset=trigger < 10);
        Signal count = counter.read();
        """
        ir_ops, lowerer, diags = compile_to_ir(source)
        assert not diags.has_errors()

    def test_optimized_latch_with_signal_value(self):
        """Inlined latch where value is a signal reference."""
        source = """
        Memory store: "signal-D";
        Signal amount = 42;
        Signal level = 75;
        store.write(amount, set=level < 25, reset=level > 100);
        Signal stored = store.read();
        """
        ir_ops, lowerer, diags = compile_to_ir(source)
        assert not diags.has_errors()

    def test_optimized_latch_equality_comparison(self):
        """Inlined latch with == / != comparisons."""
        source = """
        Memory flag: "signal-E";
        Signal status = 1;
        flag.write(1, set=status == 1, reset=status != 1);
        Signal is_set = flag.read();
        """
        ir_ops, lowerer, diags = compile_to_ir(source)
        assert not diags.has_errors()

    def test_optimized_latch_less_than_equal(self):
        """Inlined latch with <= / > comparisons."""
        source = """
        Memory threshold: "signal-F";
        Signal value = 50;
        threshold.write(1, set=value <= 20, reset=value > 80);
        Signal below = threshold.read();
        """
        ir_ops, lowerer, diags = compile_to_ir(source)
        assert not diags.has_errors()
