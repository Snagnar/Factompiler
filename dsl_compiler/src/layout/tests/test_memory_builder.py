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
