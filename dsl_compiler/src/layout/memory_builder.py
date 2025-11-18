"""Memory module construction for circuit-based memory cells."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
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

    # Internal feedback signal IDs for signal_graph (used for layout only)
    _feedback_signal_ids: List[str] = field(default_factory=list)


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

        # Standard SR latch - read from hold gate
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

        # Standard conditional write (or always-write without arithmetic feedback)
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
        """Detect if memory can use arithmetic self-feedback.

        Returns True if the write data comes from an arithmetic operation
        that (directly or indirectly) depends on reading from this same memory.
        """
        # Check if data_signal is an arithmetic operation
        if not isinstance(op.data_signal, SignalRef):
            return False

        final_node_id = op.data_signal.source_id
        arith_node = self._ir_nodes.get(final_node_id)
        if not isinstance(arith_node, IR_Arith):
            return False

        # Check if this arithmetic operation (or its inputs) reads from the memory
        return self._operation_depends_on_memory(final_node_id, op.memory_id)

    def _operation_depends_on_memory(
        self, op_id: str, memory_id: str, visited: set = None, depth: int = 0
    ) -> bool:
        """Check if an operation depends on a memory read (directly or transitively)."""
        if (
            depth > 50
        ):  # Prevent infinite recursion (increased from 10 to support longer chains)
            return False

        if visited is None:
            visited = set()

        if op_id in visited:
            return False
        visited.add(op_id)

        # Check if this operation IS a memory read from our memory
        source_memory = self._read_sources.get(op_id)
        if source_memory == memory_id:
            return True

        # Check if this is an arithmetic operation - get its inputs from IR node
        ir_node = self._ir_nodes.get(op_id)
        if isinstance(ir_node, IR_Arith):
            # Check if either operand is a SignalRef that depends on memory
            if isinstance(ir_node.left, SignalRef):
                if self._operation_depends_on_memory(
                    ir_node.left.source_id, memory_id, visited, depth + 1
                ):
                    return True
            if isinstance(ir_node.right, SignalRef):
                if self._operation_depends_on_memory(
                    ir_node.right.source_id, memory_id, visited, depth + 1
                ):
                    return True

        return False

    def _optimize_to_arithmetic_feedback(
        self, op: IR_MemWrite, module: MemoryModule, signal_graph: SignalGraph
    ):
        """Convert to arithmetic combinator feedback optimization.

        For single-operation chains: Use self-feedback on one combinator
        For multi-operation chains: Wire a feedback loop between combinators
        """
        # The arithmetic operation becomes the memory output
        arith_node_id = (
            op.data_signal.source_id if isinstance(op.data_signal, SignalRef) else None
        )

        if not arith_node_id:
            return

        # Get the final arithmetic placement
        final_placement = self.layout_plan.get_placement(arith_node_id)
        if not final_placement:
            return

        signal_name = self.signal_analyzer.get_signal_name(module.signal_type)

        # Determine if this is a single-operation or multi-operation chain
        first_consumer_id = self._find_first_memory_consumer(
            op.memory_id, arith_node_id
        )
        is_single_operation = (
            first_consumer_id == arith_node_id or first_consumer_id is None
        )

        if is_single_operation:
            # Single operation: mark for self-feedback
            final_placement.properties["has_self_feedback"] = True
            final_placement.properties["feedback_signal"] = signal_name

            # Preserve memory information in debug info for optimized combinator
            if "debug_info" in final_placement.properties:
                final_placement.properties["debug_info"]["memory_name"] = op.memory_id
                final_placement.properties["debug_info"]["details"] = (
                    f"{final_placement.properties['debug_info'].get('details', 'arith')} + memory:{op.memory_id}"
                )

            self.diagnostics.info(
                f"Optimized memory '{op.memory_id}' to single-combinator self-feedback"
            )
        else:
            # Multi-operation chain: feedback loop will be wired via signal graph
            self.diagnostics.info(
                f"Optimized memory '{op.memory_id}' to multi-combinator feedback loop"
            )

        # Mark memory gates as unused
        module.optimization = "arithmetic_feedback"
        module.output_node_id = arith_node_id
        module.write_gate_unused = True
        module.hold_gate_unused = True

        # Update signal graph: replace all memory-related sources with arithmetic combinator

        # 1. Update the memory IR node itself
        if op.memory_id in signal_graph._sources:
            old_sources = signal_graph._sources[op.memory_id]
            self.diagnostics.info(
                f"Clearing old sources for {op.memory_id}: {old_sources}"
            )
            signal_graph._sources[op.memory_id] = []
        signal_graph.set_source(op.memory_id, arith_node_id)

        # 2. Update all read operations from this memory
        for read_id, mem_id in self._read_sources.items():
            if mem_id == op.memory_id:
                # This read is from the memory we're optimizing
                if read_id in signal_graph._sources:
                    old_read_sources = signal_graph._sources[read_id]
                    self.diagnostics.info(
                        f"Updating read {read_id} sources from {old_read_sources} to [{arith_node_id}]"
                    )
                    signal_graph._sources[read_id] = []
                signal_graph.set_source(read_id, arith_node_id)

        # Remove stale signal graph references to unused gates
        if module.write_gate:
            for signal_id in list(signal_graph._sinks.keys()):
                sinks = signal_graph._sinks[signal_id]
                if module.write_gate.ir_node_id in sinks:
                    sinks.remove(module.write_gate.ir_node_id)
                    self.diagnostics.info(
                        f"Removed stale sink reference: {signal_id} -> {module.write_gate.ir_node_id} (optimized away)"
                    )

        if module.hold_gate:
            for signal_id in list(signal_graph._sinks.keys()):
                sinks = signal_graph._sinks[signal_id]
                if module.hold_gate.ir_node_id in sinks:
                    sinks.remove(module.hold_gate.ir_node_id)
                    self.diagnostics.info(
                        f"Removed stale sink reference: {signal_id} -> {module.hold_gate.ir_node_id} (optimized away)"
                    )

        # Update signal sources - memory reads now point to arithmetic combinator
        signal_graph.set_source(op.memory_id, arith_node_id)

        for read_node_id, source_memory_id in self._read_sources.items():
            if source_memory_id == op.memory_id:
                signal_graph.set_source(read_node_id, arith_node_id)

        # For multi-operation chains: register feedback edge in signal graph
        if first_consumer_id and first_consumer_id != arith_node_id:
            # The final arithmetic combinator produces a signal that feeds back to the first combinator
            # We need to register:
            # 1. The arithmetic combinator as the source of its output signal
            # 2. The first consumer as a sink of that signal
            signal_graph.set_source(arith_node_id, arith_node_id)
            signal_graph.add_sink(arith_node_id, first_consumer_id)
            self.diagnostics.info(
                f"Registered feedback loop: {arith_node_id} -> {first_consumer_id}"
            )

    def _find_first_memory_consumer(
        self, memory_id: str, final_op_id: str
    ) -> Optional[str]:
        """Find the first operation in the chain that reads from memory.

        Args:
            memory_id: Memory being optimized
            final_op_id: Final operation in the chain

        Returns:
            Entity ID of first consumer, or None
        """
        # Check all memory reads that were registered
        for read_node_id, source_memory_id in self._read_sources.items():
            if source_memory_id != memory_id:
                continue

            # Find operations that consume this read
            ir_node = self._ir_nodes.get(read_node_id)
            if not ir_node:
                continue

            # Check all IR nodes to see which ones reference this read
            for node_id, node in self._ir_nodes.items():
                if not isinstance(node, IR_Arith):
                    continue

                # Check if this arithmetic op uses the read
                left_uses = (
                    isinstance(node.left, SignalRef)
                    and node.left.source_id == read_node_id
                )
                right_uses = (
                    isinstance(node.right, SignalRef)
                    and node.right.source_id == read_node_id
                )

                if left_uses or right_uses:
                    return node_id

        return None

    def _setup_standard_write(
        self, op: IR_MemWrite, module: MemoryModule, signal_graph: SignalGraph
    ):
        """Set up standard SR latch write.

        CRITICAL: Both gates must receive ALL input signals for proper SR latch behavior:
        - Data signal must go to both gates (so hold gate has data to hold)
        - Write enable (signal-W) must go to both gates (so each gate knows when to activate)
        - Feedback loop connects both gates' outputs to both gates' inputs
        """
        if not module.write_gate or not module.hold_gate:
            self.diagnostics.warning(
                f"Cannot setup standard write for {op.memory_id}: missing gates"
            )
            return

        # ===================================================================
        # STEP 1: Connect data signal to WRITE GATE ONLY
        # ===================================================================
        if isinstance(op.data_signal, SignalRef):
            # Connect to write gate ONLY
            # Hold gate should NOT receive data - only feedback and enable
            signal_graph.add_sink(
                op.data_signal.source_id, module.write_gate.ir_node_id
            )
            self.diagnostics.info(
                f"Connected data signal {op.data_signal.source_id} → write_gate {module.write_gate.ir_node_id}"
            )

        # ===================================================================
        # STEP 2: Connect write enable (signal-W) to BOTH gates
        # ===================================================================
        if isinstance(op.write_enable, SignalRef):
            # Connect to write gate
            signal_graph.add_sink(
                op.write_enable.source_id, module.write_gate.ir_node_id
            )
            self.diagnostics.info(
                f"Connected write_enable {op.write_enable.source_id} → write_gate {module.write_gate.ir_node_id}"
            )
            # Connect to hold gate (KEY FIX #2)
            signal_graph.add_sink(
                op.write_enable.source_id, module.hold_gate.ir_node_id
            )
            self.diagnostics.info(
                f"Connected write_enable {op.write_enable.source_id} → hold_gate {module.hold_gate.ir_node_id}"
            )

        # ===================================================================
        # STEP 3: Set up unidirectional forward feedback + self-loop
        # ===================================================================
        # Write gate output → hold gate input (forward feedback, RED wire)
        # Hold gate output → hold gate input (self-loop, RED wire)
        # This topology prevents write_gate from seeing feedback (no accumulation)
        # ✅ FIX: Use UNIQUE internal signal IDs for signal_graph (for layout proximity)
        # but create DIRECT wire connections with actual signal (to avoid self-loops)

        # Create unique internal signal identifier to avoid signal_graph collisions
        # This ensures the gates are placed close together during layout optimization
        feedback_write_to_hold = f"__feedback_{op.memory_id}_w2h"

        # Add feedback edge to signal_graph for LAYOUT purposes only
        # Only forward feedback: write gate → hold gate
        # (No reverse edge since we use unidirectional topology)
        signal_graph.set_source(feedback_write_to_hold, module.write_gate.ir_node_id)
        signal_graph.add_sink(feedback_write_to_hold, module.hold_gate.ir_node_id)

        # Create DIRECT wire connections with the ACTUAL signal name
        # Use RED wire for data/feedback channel (signal-B)
        # This creates a unidirectional forward feedback + self-loop topology

        # Forward feedback: write_gate output → hold_gate input (RED)
        write_to_hold = WireConnection(
            source_entity_id=module.write_gate.ir_node_id,
            sink_entity_id=module.hold_gate.ir_node_id,
            signal_name=module.signal_type,  # Use ACTUAL signal, not internal ID
            wire_color="red",  # ✅ RED for data/feedback
            source_side="output",
            sink_side="input",
        )
        self.layout_plan.add_wire_connection(write_to_hold)

        # Self-feedback: hold_gate output → hold_gate input (RED)
        # This maintains the value when hold_gate is active
        hold_to_hold = WireConnection(
            source_entity_id=module.hold_gate.ir_node_id,
            sink_entity_id=module.hold_gate.ir_node_id,
            signal_name=module.signal_type,  # Use ACTUAL signal, not internal ID
            wire_color="red",  # ✅ RED for data/feedback
            source_side="output",
            sink_side="input",
        )
        self.layout_plan.add_wire_connection(hold_to_hold)

        # Store feedback signal IDs in module for later detection
        # Only forward feedback now (no bidirectional cross-coupling)
        module._feedback_signal_ids = [feedback_write_to_hold]

        self.diagnostics.info(
            f"Set up SR latch feedback loop for memory '{op.memory_id}': "
            f"added internal edges to signal_graph for layout, "
            f"created direct RED wire connections for actual signal '{module.signal_type}'"
        )

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
