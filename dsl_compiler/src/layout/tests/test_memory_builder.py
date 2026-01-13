"""Tests for layout/memory_builder.py - one test per function."""

import pytest

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.ir.nodes import (
    MEMORY_TYPE_RS_LATCH,
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


@pytest.fixture
def diagnostics():
    return ProgramDiagnostics()


@pytest.fixture
def builder(diagnostics):
    return MemoryBuilder(TileGrid(), LayoutPlan(), SignalAnalyzer(diagnostics, {}), diagnostics)


def test_register_ir_node(builder):
    op = IRMemCreate("mem1", "signal-A")
    builder.register_ir_node(op)
    assert op.node_id in builder._ir_nodes


def test_create_memory(builder):
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    result = builder.create_memory(op, sg)
    assert isinstance(result, MemoryModule)
    assert result.write_gate is not None
    assert result.hold_gate is not None


def test_handle_read(builder):
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    builder.create_memory(op, sg)

    read = IRMemRead("read1", "signal-A")
    read.memory_id = "mem1"
    builder.handle_read(read, sg)
    assert "read1" in builder._read_sources


def test_handle_write(builder):
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    data_signal = SignalRef("signal-A", "src1")
    write_enable = SignalRef("signal-B", "src2")
    write = IRMemWrite("mem1", data_signal, write_enable)
    builder.handle_write(write, sg)
    assert module._has_write


def test_handle_latch_write(builder):
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    value = SignalRef("signal-V", "val_src")
    set_signal = SignalRef("signal-S", "set_src")
    reset_signal = SignalRef("signal-R", "reset_src")
    latch = IRLatchWrite("mem1", value, set_signal, reset_signal, MEMORY_TYPE_RS_LATCH)
    builder.handle_latch_write(latch, sg)
    assert module.memory_type == MEMORY_TYPE_RS_LATCH


def test_cleanup_unused_gates(builder):
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)
    module.write_gate_unused = True
    module.hold_gate_unused = True

    builder.cleanup_unused_gates(builder.layout_plan, sg)
    # Just verify no crash


def test_memory_module_dataclass():
    m = MemoryModule("id", "signal-A")
    assert m.memory_id == "id"
    assert m.write_gate is None


def test_memory_builder_stores_memory(builder):
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    builder.create_memory(op, sg)
    assert "mem1" in builder._modules


def test_memory_module_gate_ids(builder):
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)
    # Verify gates have IDs
    assert module.write_gate is not None or module.hold_gate is not None


def test_handle_latch_write_with_multiplier(builder):
    """Test latch write with value != 1 creates multiplier."""
    from dsl_compiler.src.ir.nodes import MEMORY_TYPE_RS_LATCH

    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    # Write with value 5 (not 1) requires multiplier
    set_signal = SignalRef("signal-S", "set_src")
    reset_signal = SignalRef("signal-R", "reset_src")
    latch = IRLatchWrite("mem1", 5, set_signal, reset_signal, MEMORY_TYPE_RS_LATCH)
    builder.handle_latch_write(latch, sg)

    assert module.multiplier_combinator is not None


def test_handle_latch_write_sr_type(builder):
    """Test SR latch type creates multi-condition decider."""
    from dsl_compiler.src.ir.nodes import MEMORY_TYPE_SR_LATCH

    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    set_signal = SignalRef("signal-S", "set_src")
    reset_signal = SignalRef("signal-R", "reset_src")
    latch = IRLatchWrite("mem1", 1, set_signal, reset_signal, MEMORY_TYPE_SR_LATCH)
    builder.handle_latch_write(latch, sg)

    assert module.memory_type == MEMORY_TYPE_SR_LATCH
    assert module.latch_combinator is not None


def test_handle_latch_write_signal_value(builder):
    """Test latch write with SignalRef value creates multiplier."""
    from dsl_compiler.src.ir.nodes import MEMORY_TYPE_RS_LATCH

    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    # Write with signal value requires multiplier
    value_signal = SignalRef("signal-V", "val_src")
    set_signal = SignalRef("signal-S", "set_src")
    reset_signal = SignalRef("signal-R", "reset_src")
    latch = IRLatchWrite("mem1", value_signal, set_signal, reset_signal, MEMORY_TYPE_RS_LATCH)
    builder.handle_latch_write(latch, sg)

    assert module.multiplier_combinator is not None


def test_handle_latch_write_set_remap(builder):
    """Test latch write with set signal different from memory type creates remapper."""
    from dsl_compiler.src.ir.nodes import MEMORY_TYPE_RS_LATCH

    # Memory uses signal-A, set signal is signal-S (different)
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    # Set signal uses different type than memory signal
    set_signal = SignalRef("signal-S", "set_src")
    reset_signal = SignalRef("signal-R", "reset_src")
    latch = IRLatchWrite("mem1", 1, set_signal, reset_signal, MEMORY_TYPE_RS_LATCH)
    builder.handle_latch_write(latch, sg)

    # Should have created a set remapper
    remap_id = "mem1_set_remap"
    assert remap_id in builder.layout_plan.entity_placements


def test_handle_read_undefined_memory(builder):
    """Test reading from undefined memory warns."""
    sg = SignalGraph()
    read = IRMemRead("read1", "signal-A")
    read.memory_id = "nonexistent"

    # Should warn but not crash
    builder.handle_read(read, sg)


def test_handle_write_undefined_memory(builder):
    """Test writing to undefined memory warns."""
    sg = SignalGraph()
    data_signal = SignalRef("signal-A", "src1")
    write_enable = SignalRef("signal-B", "src2")
    write = IRMemWrite("nonexistent", data_signal, write_enable)

    # Should warn but not crash
    builder.handle_write(write, sg)


def test_handle_multiple_writes_warns(builder):
    """Test multiple writes to same memory warns."""
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    data_signal = SignalRef("signal-A", "src1")
    write_enable = SignalRef("signal-B", "src2")
    write1 = IRMemWrite("mem1", data_signal, write_enable)
    write2 = IRMemWrite("mem1", data_signal, write_enable)

    builder.handle_write(write1, sg)
    builder.handle_write(write2, sg)  # Should warn about multiple writes

    assert module._has_write


def test_setup_latch_feedback(builder):
    """Test _setup_latch_feedback creates green wire self-loop."""
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    # Upgrade to latch
    from dsl_compiler.src.ir.nodes import MEMORY_TYPE_RS_LATCH

    set_signal = SignalRef("signal-S", "set_src")
    reset_signal = SignalRef("signal-R", "reset_src")
    latch = IRLatchWrite("mem1", 1, set_signal, reset_signal, MEMORY_TYPE_RS_LATCH)
    builder.handle_latch_write(latch, sg)

    # Feedback should be set up
    assert module._feedback_connected


def test_memory_module_optimization_field():
    """Test MemoryModule optimization field."""
    m = MemoryModule("id", "signal-A", optimization="arithmetic_feedback")
    assert m.optimization == "arithmetic_feedback"


def test_handle_read_with_latch(builder):
    """Test reading from latch memory uses latch combinator."""
    from dsl_compiler.src.ir.nodes import MEMORY_TYPE_RS_LATCH

    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    # Upgrade to latch
    set_signal = SignalRef("signal-S", "set_src")
    reset_signal = SignalRef("signal-R", "reset_src")
    latch = IRLatchWrite("mem1", 1, set_signal, reset_signal, MEMORY_TYPE_RS_LATCH)
    builder.handle_latch_write(latch, sg)

    read = IRMemRead("read1", "signal-A")
    read.memory_id = "mem1"
    builder.handle_read(read, sg)

    # Read should be connected to latch output
    assert "read1" in builder._read_sources


def test_setup_standard_write(builder):
    """Test standard write sets up feedback connections."""
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    data_signal = SignalRef("signal-A", "data_src")
    write_enable = SignalRef("signal-W", "we_src")
    write = IRMemWrite("mem1", data_signal, write_enable)
    builder.handle_write(write, sg)

    # Verify feedback was set up
    assert module._feedback_signal_ids is not None
    # Verify wire connections were created for feedback
    assert any("red" in str(w) for w in builder.layout_plan.wire_connections)


def test_handle_latch_write_reset_remap(builder):
    """Test latch write remaps reset signal when it matches memory type."""
    from dsl_compiler.src.ir.nodes import MEMORY_TYPE_RS_LATCH

    # Memory uses signal-A, and reset will also be signal-A (needs remapping)
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    set_signal = SignalRef("signal-S", "set_src")
    reset_signal = SignalRef("signal-A", "reset_src")  # Same as memory signal!
    latch = IRLatchWrite("mem1", 1, set_signal, reset_signal, MEMORY_TYPE_RS_LATCH)
    builder.handle_latch_write(latch, sg)

    # Reset remapper should be created
    remap_id = "mem1_reset_remap"
    assert remap_id in builder.layout_plan.entity_placements


def test_operation_depends_on_memory(builder):
    """Test _operation_depends_on_memory detects transitive dependency."""
    from dsl_compiler.src.ir.nodes import IRArith

    # Create memory
    mem_op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    builder.create_memory(mem_op, sg)

    # Create read from memory
    read = IRMemRead("read1", "signal-A")
    read.memory_id = "mem1"
    builder.handle_read(read, sg)

    # Create arithmetic that uses the read
    arith = IRArith("arith1", "signal-B")
    arith.op = "+"
    arith.left = SignalRef("signal-A", "read1")
    arith.right = 1
    builder.register_ir_node(arith)

    # Test that arith1 depends on mem1
    assert builder._operation_depends_on_memory("arith1", "mem1")

    # Test that non-existent op doesn't depend
    assert not builder._operation_depends_on_memory("nonexistent", "mem1")


def test_optimize_to_arithmetic_feedback(builder):
    """Test arithmetic feedback optimization for always-write memory."""
    from dsl_compiler.src.ir.nodes import IRArith, IRConst

    # Create memory
    mem_op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(mem_op, sg)

    # Create a read from memory
    read = IRMemRead("read1", "signal-A")
    read.memory_id = "mem1"
    builder.handle_read(read, sg)

    # Create an arithmetic node that uses read1: mem + 1
    arith = IRArith("arith1", "signal-A")
    arith.op = "+"
    arith.left = SignalRef("signal-A", "read1")
    arith.right = 1
    builder.register_ir_node(arith)

    # Create placement for the arith node (optimizer needs this)
    builder.layout_plan.create_and_add_placement(
        ir_node_id="arith1",
        entity_type="arithmetic-combinator",
        position=(0, 0),
        role="arith",
        properties={"debug_info": {}},
    )

    # Write with always-write (write_enable=1) - value must be set after construction
    write_enable_const = IRConst("const1", "signal-W")
    write_enable_const.value = 1
    builder.register_ir_node(write_enable_const)

    data_signal = SignalRef("signal-A", "arith1")
    write_enable = SignalRef("signal-W", "const1")
    write = IRMemWrite("mem1", data_signal, write_enable)
    builder.handle_write(write, sg)

    # Should have optimized to arithmetic feedback
    assert module.optimization == "arithmetic_feedback"
    assert module.write_gate_unused
    assert module.hold_gate_unused


def test_handle_read_with_multiplier(builder):
    """Test read from latch with multiplier uses multiplier as source."""
    from dsl_compiler.src.ir.nodes import MEMORY_TYPE_RS_LATCH

    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    # Create latch with multiplier (value != 1)
    set_signal = SignalRef("signal-S", "set_src")
    reset_signal = SignalRef("signal-R", "reset_src")
    latch = IRLatchWrite("mem1", 5, set_signal, reset_signal, MEMORY_TYPE_RS_LATCH)
    builder.handle_latch_write(latch, sg)

    # Now read
    read = IRMemRead("read1", "signal-A")
    read.memory_id = "mem1"
    builder.handle_read(read, sg)

    # Read source should be set via multiplier path
    assert module.multiplier_combinator is not None


def test_handle_read_uses_hold_gate(builder):
    """Test read from standard memory uses hold gate as source."""
    op = IRMemCreate("mem1", "signal-A")
    sg = SignalGraph()
    module = builder.create_memory(op, sg)

    read = IRMemRead("read1", "signal-A")
    read.memory_id = "mem1"
    builder.handle_read(read, sg)

    # Should use hold gate as source
    assert module.hold_gate is not None
    assert "read1" in sg._sources
