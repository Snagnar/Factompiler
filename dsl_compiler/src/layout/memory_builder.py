"""Memory module construction for circuit-based memory cells."""

from dataclasses import dataclass
from typing import Dict, Optional, Any
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.ir.builder import (
    IRNode,
    IR_Const,
    IR_Arith,
    IR_MemCreate,
    IR_MemRead,
    IR_MemWrite,
    SignalRef,
)
from .layout_plan import LayoutPlan, EntityPlacement, WireConnection
from .signal_analyzer import SignalAnalyzer
from .signal_graph import SignalGraph
from .tile_grid import TileGrid


@dataclass
class MemoryModule:
    """Represents a memory cell's physical implementation.

    Standard memories use SR latch (2 deciders).
    Optimized memories use fewer components.
    """

    memory_id: str
    signal_type: str

    # Gate placements
    write_gate: Optional[EntityPlacement] = None
    hold_gate: Optional[EntityPlacement] = None

    # Optimization info
    optimization: Optional[str] = None  # None, 'single_gate', 'arithmetic_feedback'
    output_node_id: Optional[str] = None  # For optimized memories

    # Flags
    write_gate_unused: bool = False
    hold_gate_unused: bool = False
    _feedback_connected: bool = False
    _has_write: bool = False


class MemoryBuilder:
    """Builds memory modules from IR operations.

    Responsibilities:
    - Create SR latch gate placements
    - Detect optimization opportunities
    - Handle memory reads/writes
    - Clean up unused optimized gates
    """

    def __init__(
        self,
        tile_grid: TileGrid,
        layout_plan: LayoutPlan,
        signal_analyzer: SignalAnalyzer,
        diagnostics: ProgramDiagnostics,
    ):
        self.tile_grid = tile_grid
        self.layout_plan = layout_plan
        self.signal_analyzer = signal_analyzer
        self.diagnostics = diagnostics

        self._modules: Dict[str, MemoryModule] = {}
        self._read_sources: Dict[str, str] = {}  # read_node_id -> memory_id
        self._ir_nodes: Dict[str, IRNode] = {}  # For optimization detection

    def register_ir_node(self, node: IRNode):
        """Track IR node for optimization detection."""
        self._ir_nodes[node.node_id] = node

    def create_memory(
        self, op: IR_MemCreate, signal_graph: SignalGraph
    ) -> MemoryModule:
        """Create SR latch gates for a memory cell.

        Returns MemoryModule with write_gate and hold_gate placements.
        """
        signal_name = self.signal_analyzer.get_signal_name(op.signal_type)

        # Create write gate (position will be set by force-directed layout)
        write_id = f"{op.memory_id}_write_gate"
        write_placement = EntityPlacement(
            ir_node_id=write_id,
            entity_type="decider-combinator",
            position=None,  # Force-directed will position
            properties={
                "footprint": (1, 2),
                "operation": ">",
                "left_operand": "signal-W",
                "right_operand": 0,
                "output_signal": signal_name,
                "copy_count_from_input": True,
                "debug_info": self._make_debug_info(op, "write_gate"),
            },
            role="memory_write_gate",
        )
        self.layout_plan.add_placement(write_placement)

        # Create hold gate
        hold_id = f"{op.memory_id}_hold_gate"
        hold_placement = EntityPlacement(
            ir_node_id=hold_id,
            entity_type="decider-combinator",
            position=None,
            properties={
                "footprint": (1, 2),
                "operation": "=",
                "left_operand": "signal-W",
                "right_operand": 0,
                "output_signal": signal_name,
                "copy_count_from_input": True,
                "debug_info": self._make_debug_info(op, "hold_gate"),
            },
            role="memory_hold_gate",
        )
        self.layout_plan.add_placement(hold_placement)

        # Create module
        module = MemoryModule(
            memory_id=op.memory_id,
            signal_type=signal_name,
            write_gate=write_placement,
            hold_gate=hold_placement,
        )
        self._modules[op.memory_id] = module

        # Track signal source (memory output = hold gate)
        signal_graph.set_source(op.memory_id, hold_id)

        return module

    def handle_read(self, op: IR_MemRead, signal_graph: SignalGraph):
        """Connect read to memory output."""
        module = self._modules.get(op.memory_id)
        if not module:
            self.diagnostics.warning(f"Read from undefined memory: {op.memory_id}")
            return

        # Track read source
        self._read_sources[op.node_id] = op.memory_id

        # Handle optimized memories
        if module.optimization == "arithmetic_feedback":
            if module.output_node_id:
                signal_graph.set_source(op.node_id, module.output_node_id)
            return

        if module.optimization == "single_gate":
            if module.write_gate:
                signal_graph.set_source(op.node_id, module.write_gate.ir_node_id)
            return

        # Standard SR latch
        if module.hold_gate:
            signal_graph.set_source(op.node_id, module.hold_gate.ir_node_id)

    def handle_write(self, op: IR_MemWrite, signal_graph: SignalGraph):
        """Handle memory write with optimization detection.

        Detects:
        - Always-write optimization (when=1)
        - Arithmetic feedback optimization
        - Single-gate optimization
        """
        module = self._modules.get(op.memory_id)
        if not module:
            self.diagnostics.warning(f"Write to undefined memory: {op.memory_id}")
            return

        # Detect multiple writes (warn user)
        if module._has_write:
            self.diagnostics.warning(
                f"Multiple writes to memory '{op.memory_id}' - "
                f"only last write will be optimized"
            )
        module._has_write = True

        # Check for always-write pattern
        is_always_write = self._is_always_write(op)

        if is_always_write:
            # Try arithmetic feedback optimization
            if self._can_use_arithmetic_feedback(op, module):
                self._optimize_to_arithmetic_feedback(op, module, signal_graph)
                return

            # Fall back to single-gate optimization
            self._optimize_to_single_gate(op, module, signal_graph)
            return

        # Standard conditional write
        self._setup_standard_write(op, module, signal_graph)

    def cleanup_unused_gates(self, layout_plan: LayoutPlan, signal_graph: SignalGraph):
        """Remove gates that were optimized away."""
        to_remove = []

        for memory_id, module in self._modules.items():
            if module.write_gate_unused and module.write_gate:
                to_remove.append(module.write_gate.ir_node_id)
            if module.hold_gate_unused and module.hold_gate:
                to_remove.append(module.hold_gate.ir_node_id)

        # Remove from placements
        for entity_id in to_remove:
            layout_plan.entity_placements.pop(entity_id, None)
            self.diagnostics.info(f"Removed unused gate: {entity_id}")

        # Remove stale wire connections
        remaining = [
            conn
            for conn in layout_plan.wire_connections
            if conn.source_entity_id not in to_remove
            and conn.sink_entity_id not in to_remove
        ]
        layout_plan.wire_connections = remaining

        # Clean up signal graph
        for signal_id in list(signal_graph._sinks.keys()):
            sinks = signal_graph._sinks[signal_id]
            for removed_id in to_remove:
                if removed_id in sinks:
                    sinks.remove(removed_id)

    # Private helper methods
    def _is_always_write(self, op: IR_MemWrite) -> bool:
        """Check if write enable is constant 1."""
        if isinstance(op.write_enable, int) and op.write_enable == 1:
            return True
        if isinstance(op.write_enable, SignalRef):
            const_ir = self._ir_nodes.get(op.write_enable.source_id)
            if isinstance(const_ir, IR_Const) and const_ir.value == 1:
                return True
        return False

    def _can_use_arithmetic_feedback(
        self, op: IR_MemWrite, module: MemoryModule
    ) -> bool:
        """Detect if memory can use arithmetic self-feedback."""
        # Check if data_signal is an arithmetic operation that depends on this memory
        if not isinstance(op.data_signal, SignalRef):
            return False

        arith_node = self._ir_nodes.get(op.data_signal.source_id)
        if not isinstance(arith_node, IR_Arith):
            return False

        # Check if one operand is the memory itself
        mem_read_id = None
        other_operand = None

        if isinstance(arith_node.left, SignalRef):
            left_source = self._ir_nodes.get(arith_node.left.source_id)
            if (
                isinstance(left_source, IR_MemRead)
                and left_source.memory_id == op.memory_id
            ):
                mem_read_id = arith_node.left.source_id
                other_operand = arith_node.right

        if isinstance(arith_node.right, SignalRef) and mem_read_id is None:
            right_source = self._ir_nodes.get(arith_node.right.source_id)
            if (
                isinstance(right_source, IR_MemRead)
                and right_source.memory_id == op.memory_id
            ):
                mem_read_id = arith_node.right.source_id
                other_operand = arith_node.left

        return mem_read_id is not None

    def _optimize_to_arithmetic_feedback(
        self, op: IR_MemWrite, module: MemoryModule, signal_graph: SignalGraph
    ):
        """Convert to single arithmetic combinator with self-feedback."""
        # The arithmetic operation becomes the memory output
        arith_node_id = (
            op.data_signal.source_id if isinstance(op.data_signal, SignalRef) else None
        )

        if arith_node_id:
            module.optimization = "arithmetic_feedback"
            module.output_node_id = arith_node_id
            module.write_gate_unused = True
            module.hold_gate_unused = True

            # Update signal sources - memory reads now point to arithmetic combinator
            signal_graph.set_source(op.memory_id, arith_node_id)

    def _optimize_to_single_gate(
        self, op: IR_MemWrite, module: MemoryModule, signal_graph: SignalGraph
    ):
        """Convert to single decider gate."""
        module.optimization = "single_gate"
        module.hold_gate_unused = True

        # Update signal sources - memory reads now point to write gate
        signal_graph.set_source(op.memory_id, module.write_gate.ir_node_id)

    def _setup_standard_write(
        self, op: IR_MemWrite, module: MemoryModule, signal_graph: SignalGraph
    ):
        """Set up standard SR latch write."""
        # Standard SR latch needs connections from data/enable to write gate
        # and feedback from hold gate to itself
        # These connections will be set up by connection planner
        pass

    def _make_debug_info(self, op, role) -> Dict[str, Any]:
        """Build debug info dict for memory gates."""
        debug_info = {
            "variable": f"mem:{op.memory_id}",
            "operation": "memory",
            "details": role,
            "signal_type": self.signal_analyzer.get_signal_name(op.signal_type),
            "role": f"memory_{role}",
        }

        # Add source location if available
        if hasattr(op, "source_ast") and op.source_ast:
            if hasattr(op.source_ast, "line"):
                debug_info["line"] = op.source_ast.line
            if hasattr(op.source_ast, "source_file"):
                debug_info["source_file"] = op.source_ast.source_file

        return debug_info
