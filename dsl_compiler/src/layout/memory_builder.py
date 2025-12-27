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
from .layout_plan import LayoutPlan, WireConnection, EntityPlacement
from .signal_analyzer import SignalAnalyzer
from .signal_graph import SignalGraph
from .tile_grid import TileGrid


@dataclass
class MemoryModule:
    """Represents a memory cell's physical implementation.

    Standard memories use a write-gated latch (2 deciders).
    Optimized memories use fewer components.
    """

    memory_id: str
    signal_type: str

    write_gate: Optional[EntityPlacement] = None
    hold_gate: Optional[EntityPlacement] = None

    optimization: Optional[str] = None  # None, 'single_gate', 'arithmetic_feedback'
    output_node_id: Optional[str] = None  # For optimized memories

    write_gate_unused: bool = False
    hold_gate_unused: bool = False
    _feedback_connected: bool = False
    _has_write: bool = False

    _feedback_signal_ids: List[str] = field(default_factory=list)
    
    # Track unique write enable signals for multi-write memories
    _write_enable_signals: List[str] = field(default_factory=list)
    _data_gate_count: int = 0


class MemoryBuilder:
    """Builds memory modules from IR operations.

    Responsibilities:
    - Create write-gated latch placements
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
        """Create write-gated latch for a memory cell.

        Returns MemoryModule with write_gate and hold_gate placements.
        """
        signal_name = self.signal_analyzer.get_signal_name(op.signal_type)

        write_id = f"{op.memory_id}_write_gate"
        write_placement = self.layout_plan.create_and_add_placement(
            ir_node_id=write_id,
            entity_type="decider-combinator",
            position=None,  # Force-directed will position
            footprint=(1, 2),
            role="memory_write_gate",
            debug_info=self._make_debug_info(op, "write_gate"),
            operation=">",
            left_operand="signal-W",
            right_operand=0,
            output_signal=signal_name,
            copy_count_from_input=True,
        )

        hold_id = f"{op.memory_id}_hold_gate"
        hold_placement = self.layout_plan.create_and_add_placement(
            ir_node_id=hold_id,
            entity_type="decider-combinator",
            position=None,
            footprint=(1, 2),
            role="memory_hold_gate",
            debug_info=self._make_debug_info(op, "hold_gate"),
            operation="=",
            left_operand="signal-W",
            right_operand=0,
            output_signal=signal_name,
            copy_count_from_input=True,
        )

        module = MemoryModule(
            memory_id=op.memory_id,
            signal_type=signal_name,
            write_gate=write_placement,
            hold_gate=hold_placement,
        )
        self._modules[op.memory_id] = module

        signal_graph.set_source(op.memory_id, hold_id)

        return module

    def handle_read(self, op: IR_MemRead, signal_graph: SignalGraph):
        """Connect read to memory output."""
        module = self._modules.get(op.memory_id)
        if not module:
            self.diagnostics.warning(
                f"Read from undefined memory '{op.memory_id}' - this may indicate a logic error"
            )
            return

        self._read_sources[op.node_id] = op.memory_id

        if module.optimization == "arithmetic_feedback":
            if module.output_node_id:
                signal_graph.set_source(op.node_id, module.output_node_id)
            return

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
            self.diagnostics.warning(
                f"Write to undefined memory '{op.memory_id}' - this may indicate a logic error"
            )
            return

        if module._has_write:
            self.diagnostics.warning(
                f"Multiple writes to memory '{op.memory_id}' detected - "
                f"only the last write will be optimized"
            )
        module._has_write = True

        is_always_write = self._is_always_write(op)

        if is_always_write:
            if self._can_use_arithmetic_feedback(op, module):
                self._optimize_to_arithmetic_feedback(op, module, signal_graph)
                return

        self._setup_standard_write(op, module, signal_graph)

    def cleanup_unused_gates(self, layout_plan: LayoutPlan, signal_graph: SignalGraph):
        """Remove gates that were optimized away."""
        to_remove = []

        for memory_id, module in self._modules.items():
            if module.write_gate_unused and module.write_gate:
                to_remove.append(module.write_gate.ir_node_id)
            if module.hold_gate_unused and module.hold_gate:
                to_remove.append(module.hold_gate.ir_node_id)

        for entity_id in to_remove:
            layout_plan.entity_placements.pop(entity_id, None)
            self.diagnostics.info(f"Removed unused gate: {entity_id}")

        remaining = [
            conn
            for conn in layout_plan.wire_connections
            if conn.source_entity_id not in to_remove
            and conn.sink_entity_id not in to_remove
        ]
        layout_plan.wire_connections = remaining

        for signal_id in list(signal_graph._sinks.keys()):
            sinks = signal_graph._sinks[signal_id]
            for removed_id in to_remove:
                if removed_id in sinks:
                    sinks.remove(removed_id)

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
        if not isinstance(op.data_signal, SignalRef):
            return False

        final_node_id = op.data_signal.source_id
        arith_node = self._ir_nodes.get(final_node_id)
        if not isinstance(arith_node, IR_Arith):
            return False

        return self._operation_depends_on_memory(final_node_id, op.memory_id)

    def _operation_depends_on_memory(
        self, op_id: str, memory_id: str, visited: Optional[set] = None
    ) -> bool:
        """Check if an operation depends on a memory read (directly or transitively)."""
        if visited is None:
            visited = set()

        if op_id in visited:
            return False
        visited.add(op_id)

        source_memory = self._read_sources.get(op_id)
        if source_memory == memory_id:
            return True

        ir_node = self._ir_nodes.get(op_id)
        if isinstance(ir_node, IR_Arith):
            if isinstance(ir_node.left, SignalRef):
                if self._operation_depends_on_memory(
                    ir_node.left.source_id, memory_id, visited
                ):
                    return True
            if isinstance(ir_node.right, SignalRef):
                if self._operation_depends_on_memory(
                    ir_node.right.source_id, memory_id, visited
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
        arith_node_id = (
            op.data_signal.source_id if isinstance(op.data_signal, SignalRef) else None
        )

        if not arith_node_id:
            return

        final_placement = self.layout_plan.get_placement(arith_node_id)
        if not final_placement:
            return

        signal_name = self.signal_analyzer.get_signal_name(module.signal_type)

        first_consumer_id = self._find_first_memory_consumer(op.memory_id)
        is_single_operation = (
            first_consumer_id == arith_node_id or first_consumer_id is None
        )

        if is_single_operation:
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
            self.diagnostics.info(
                f"Optimized memory '{op.memory_id}' to multi-combinator feedback loop"
            )

        module.optimization = "arithmetic_feedback"
        module.output_node_id = arith_node_id
        module.write_gate_unused = True
        module.hold_gate_unused = True

        if op.memory_id in signal_graph._sources:
            old_sources = signal_graph._sources[op.memory_id]
            self.diagnostics.info(
                f"Clearing old sources for {op.memory_id}: {old_sources}"
            )
            signal_graph._sources[op.memory_id] = []
        signal_graph.set_source(op.memory_id, arith_node_id)

        for read_id, mem_id in self._read_sources.items():
            if mem_id == op.memory_id:
                if read_id in signal_graph._sources:
                    old_read_sources = signal_graph._sources[read_id]
                    self.diagnostics.info(
                        f"Updating read {read_id} sources from {old_read_sources} to [{arith_node_id}]"
                    )
                    signal_graph._sources[read_id] = []
                signal_graph.set_source(read_id, arith_node_id)

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
            signal_graph.set_source(arith_node_id, arith_node_id)
            signal_graph.add_sink(arith_node_id, first_consumer_id)
            self.diagnostics.info(
                f"Registered feedback loop: {arith_node_id} -> {first_consumer_id}"
            )

    def _find_first_memory_consumer(self, memory_id: str) -> Optional[str]:
        """Find the first operation in the chain that reads from memory.

        Args:
            memory_id: Memory being optimized

        Returns:
            Entity ID of first consumer, or None
        """
        for read_node_id, source_memory_id in self._read_sources.items():
            if source_memory_id != memory_id:
                continue

            ir_node = self._ir_nodes.get(read_node_id)
            if not ir_node:
                continue

            for node_id, node in self._ir_nodes.items():
                if not isinstance(node, IR_Arith):
                    continue

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
        """Set up standard write-gated latch.

        The write-gated latch works as follows:
        - write_gate: if signal-W > 0, copy input to output (pass through data)
        - hold_gate: if signal-W = 0, copy input to output (maintain via self-feedback)
        
        For conditional writes with multiple data sources, we need data gates:
        - Each data gate receives: data signal + write enable
        - Data gate: if signal-W > 0, copy data signal to output
        - All data gates connect to write_gate input
        
        This prevents data values from accumulating when multiple writes exist.
        """
        if not module.write_gate or not module.hold_gate:
            self.diagnostics.warning(
                f"Cannot setup standard write for {op.memory_id}: missing gates"
            )
            return

        signal_name = self.signal_analyzer.get_signal_name(module.signal_type)
        
        # ===================================================================
        # STEP 1: Connect data signal to WRITE GATE
        # If this memory has multiple writes, we need a data gate to prevent
        # data values from accumulating. The data gate passes the data signal
        # only when its specific write condition is true.
        # ===================================================================
        if isinstance(op.data_signal, SignalRef):
            # Check if there are multiple writes to this memory
            write_count = sum(
                1 for ir_op in self._ir_nodes.values()
                if isinstance(ir_op, IR_MemWrite) and ir_op.memory_id == op.memory_id
            )
            
            if write_count > 1 and isinstance(op.write_enable, SignalRef):
                # Multiple writes: create a data gate that passes the data signal
                # only when this specific write condition is true.
                #
                # Each write has a UNIQUE write enable signal (signal-W, signal-X, etc.)
                # assigned at lowering time. The data gate checks its specific signal.
                gate_index = getattr(module, '_data_gate_count', 0)
                module._data_gate_count = gate_index + 1
                data_gate_id = f"{op.memory_id}_data_gate_{gate_index}"
                
                # Get the SPECIFIC write enable signal for this write
                # (assigned at lowering time: signal-W, signal-X, signal-Y, etc.)
                write_enable_signal = op.write_enable.signal_type
                
                # Create a decider that copies the data signal when THIS write_enable > 0
                self.layout_plan.create_and_add_placement(
                    ir_node_id=data_gate_id,
                    entity_type="decider-combinator",
                    position=None,
                    footprint=(1, 2),
                    role="memory_data_gate",
                    debug_info={
                        "variable": f"mem:{op.memory_id}",
                        "operation": "memory",
                        "details": f"data_gate for write #{gate_index} (checks {write_enable_signal})",
                        "signal_type": signal_name,
                        "role": "memory_data_gate",
                    },
                    operation=">",
                    left_operand=write_enable_signal,  # Check THIS write's unique signal
                    right_operand=0,
                    output_signal=signal_name,
                    copy_count_from_input=True,  # Copy the actual input value
                )
                
                # Connect data signal to data gate input
                signal_graph.add_sink(op.data_signal.source_id, data_gate_id)
                
                # Connect write_enable to data gate input (no special wire color needed)
                signal_graph.add_sink(op.write_enable.source_id, data_gate_id)
                
                # Connect data gate output to write_gate input
                signal_graph.set_source(data_gate_id, data_gate_id)
                signal_graph.add_sink(data_gate_id, module.write_gate.ir_node_id)
                
                self.diagnostics.info(
                    f"Created data gate {data_gate_id} for memory '{op.memory_id}' "
                    f"({op.data_signal.source_id} gated by {op.write_enable.source_id} [{write_enable_signal}])"
                )
            else:
                # Single write or always-write: connect data directly
                signal_graph.add_sink(
                    op.data_signal.source_id, module.write_gate.ir_node_id
                )
                self.diagnostics.info(
                    f"Connected data signal {op.data_signal.source_id} → write_gate {module.write_gate.ir_node_id}"
                )

        # ===================================================================
        # STEP 2: Connect write enable to gates
        # For multi-write memories with unique write enable signals (W, X, Y, etc.),
        # we track them for aggregation and connect to an aggregator.
        # ===================================================================
        if isinstance(op.write_enable, SignalRef):
            write_enable_signal = op.write_enable.signal_type
            
            # Check if there are multiple writes to this memory
            write_count = sum(
                1 for ir_op in self._ir_nodes.values()
                if isinstance(ir_op, IR_MemWrite) and ir_op.memory_id == op.memory_id
            )
            
            if write_count > 1:
                # Multi-write: track the write enable signal and source for aggregation
                if write_enable_signal not in module._write_enable_signals:
                    module._write_enable_signals.append(write_enable_signal)
                
                # Create a signal converter: X + 0 → W
                # This converts the unique signal to signal-W for the aggregated check
                converter_id = f"{op.memory_id}_we_conv_{len(module._write_enable_signals) - 1}"
                self.layout_plan.create_and_add_placement(
                    ir_node_id=converter_id,
                    entity_type="arithmetic-combinator",
                    position=None,
                    footprint=(1, 2),
                    role="memory_we_converter",
                    debug_info={
                        "variable": f"mem:{op.memory_id}",
                        "operation": "memory",
                        "details": f"write_enable converter: {write_enable_signal} → signal-W",
                        "role": "memory_we_converter",
                    },
                    operation="+",
                    left_operand=write_enable_signal,
                    right_operand=0,
                    output_signal="signal-W",
                )
                
                # Connect write enable source to converter input
                signal_graph.add_sink(op.write_enable.source_id, converter_id)
                
                # Connect converter output to both gates
                signal_graph.set_source(converter_id, converter_id)
                signal_graph.add_sink(converter_id, module.write_gate.ir_node_id)
                signal_graph.add_sink(converter_id, module.hold_gate.ir_node_id)
                
                self.diagnostics.info(
                    f"Created write_enable converter {converter_id}: "
                    f"{write_enable_signal} → signal-W for multi-write memory '{op.memory_id}'"
                )
            else:
                # Single write: connect directly
                signal_graph.add_sink(
                    op.write_enable.source_id, module.write_gate.ir_node_id
                )
                signal_graph.add_sink(
                    op.write_enable.source_id, module.hold_gate.ir_node_id
                )
                self.diagnostics.info(
                    f"Connected write_enable {op.write_enable.source_id} → gates"
                )

        # ===================================================================
        # STEP 3: Set up latch feedback topology
        # ===================================================================
        # Topology:
        # - write_gate output ↔ hold_gate output (same RED wire network)
        # - hold_gate output → hold_gate input (self-loop on RED)
        # - This prevents write_gate from seeing feedback (no accumulation on input)
        # - But allows hold_gate to capture write_gate's output for holding

        # Create unique internal signal identifier to avoid signal_graph collisions
        # This ensures the gates are placed close together during layout optimization
        feedback_write_to_hold = f"__feedback_{op.memory_id}_w2h"

        # Add feedback edge to signal_graph for LAYOUT purposes only
        # This helps the layout planner keep gates close together
        signal_graph.set_source(feedback_write_to_hold, module.write_gate.ir_node_id)
        signal_graph.add_sink(feedback_write_to_hold, module.hold_gate.ir_node_id)

        # Create DIRECT wire connections with the ACTUAL signal name
        # Use RED wire for data/feedback channel
        #
        # CRITICAL TOPOLOGY: For proper latching behavior, we need:
        # 1. write_gate output → hold_gate OUTPUT (shared output network)
        # 2. hold_gate output → hold_gate input (self-feedback)
        #
        # This ensures that when write_gate is active:
        # - write_gate output appears on the shared output network
        # - hold_gate's self-feedback loop sees this value
        # - When write_gate deactivates, hold_gate continues outputting via self-feedback
        #
        # Previous approach (write_gate → hold_gate INPUT) failed because:
        # - hold_gate is inactive during writes, so its self-feedback dies
        # - When hold_gate reactivates, it has no value to hold

        # Connect write_gate output to hold_gate OUTPUT side (shared output network)
        write_to_hold_output = WireConnection(
            source_entity_id=module.write_gate.ir_node_id,
            sink_entity_id=module.hold_gate.ir_node_id,
            signal_name=module.signal_type,  # Use ACTUAL signal, not internal ID
            wire_color="red",  # ✅ RED for data/feedback
            source_side="output",
            sink_side="output",  # ✅ KEY FIX: Connect to OUTPUT side, not input!
        )
        self.layout_plan.add_wire_connection(write_to_hold_output)

        # Self-feedback: hold_gate output → hold_gate input (RED)
        # Now this sees the combined output of write_gate + hold_gate
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
            f"Set up write-gated latch feedback loop for memory '{op.memory_id}': "
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

        if hasattr(op, "source_ast") and op.source_ast:
            if hasattr(op.source_ast, "line"):
                debug_info["line"] = op.source_ast.line
            if hasattr(op.source_ast, "source_file"):
                debug_info["source_file"] = op.source_ast.source_file

        return debug_info
