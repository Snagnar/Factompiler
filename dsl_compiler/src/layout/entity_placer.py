from typing import Any, Dict, Optional, Tuple
from dsl_compiler.src.common import ProgramDiagnostics
from dsl_compiler.src.common import get_entity_footprint, get_entity_alignment
from .layout_engine import LayoutEngine
from .layout_plan import LayoutPlan, EntityPlacement, WireConnection
from .signal_analyzer import SignalUsageEntry, SignalMaterializer
from .signal_resolver import SignalResolver
from .signal_graph import SignalGraph


from dsl_compiler.src.ir import (
    IRNode,
    IR_Const,
    IR_Arith,
    IR_Decider,
    IR_MemCreate,
    IR_MemRead,
    IR_MemWrite,
    IR_PlaceEntity,
    IR_EntityPropWrite,
    IR_EntityPropRead,
    IR_WireMerge,
    SignalRef,
    ValueRef,
)


class EntityPlacer:
    """Plans physical placement of IR entities without materializing them."""

    def __init__(
        self,
        layout_engine: LayoutEngine,
        layout_plan: LayoutPlan,
        signal_usage: Dict[str, SignalUsageEntry],
        materializer: SignalMaterializer,
        signal_resolver: SignalResolver,
        diagnostics: ProgramDiagnostics,
    ):
        self.layout = layout_engine
        self.plan = layout_plan
        self.signal_usage = signal_usage
        self.materializer = materializer
        self.resolver = signal_resolver
        self.diagnostics = diagnostics
        self.signal_graph = SignalGraph()

        self.next_entity_number = 1
        self._memory_modules: Dict[str, Dict[str, Any]] = {}
        self._wire_merge_junctions: Dict[str, Dict[str, Any]] = {}
        self._entity_property_signals: Dict[str, str] = {}
        self._memory_read_sources: Dict[
            str, str
        ] = {}  # Track which memory each read came from
        self._ir_nodes: Dict[str, IRNode] = {}  # Track all IR nodes by ID for lookups

    def _build_debug_info(
        self, 
        op: IRNode, 
        role_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract debug information from an IR node and its usage entry.
        
        Returns dict with keys: variable, operation, details, signal_type, 
        source_file, line, role
        """
        debug_info: Dict[str, Any] = {}
        
        # Get usage entry for this IR node (the producer)
        usage = self.signal_usage.get(op.node_id)
        
        # Extract variable name
        if usage and usage.debug_label:
            debug_info["variable"] = usage.debug_label
        elif hasattr(op, "debug_label") and op.debug_label:
            debug_info["variable"] = op.debug_label
        
        # Extract source location
        source_ast = usage.source_ast if usage else None
        if not source_ast and hasattr(op, "source_ast"):
            source_ast = op.source_ast
            
        if source_ast:
            if hasattr(source_ast, "line") and source_ast.line > 0:
                debug_info["line"] = source_ast.line
            if hasattr(source_ast, "source_file") and source_ast.source_file:
                debug_info["source_file"] = source_ast.source_file
        
        # Extract signal type
        if usage and usage.resolved_signal_name:
            debug_info["signal_type"] = usage.resolved_signal_name
        elif hasattr(op, "output_type"):
            debug_info["signal_type"] = op.output_type
        
        # Add user-declared flag
        if hasattr(op, "debug_metadata") and op.debug_metadata:
            if op.debug_metadata.get("user_declared"):
                debug_info["user_declared"] = True
                declared_name = op.debug_metadata.get("declared_name")
                if declared_name:
                    debug_info["variable"] = declared_name
        
        # Add operation-specific details
        if isinstance(op, IR_Const):
            debug_info["operation"] = "const"
            details = f"value={op.value}"
            if hasattr(op, "debug_metadata") and op.debug_metadata.get("user_declared"):
                details += " (input)"
            debug_info["details"] = details
        elif isinstance(op, IR_Arith):
            debug_info["operation"] = "arith"
            debug_info["details"] = f"op={op.op}"
        elif isinstance(op, IR_Decider):
            debug_info["operation"] = "decider"
            debug_info["details"] = f"cond={op.test_op}"
        elif isinstance(op, IR_MemCreate):
            debug_info["operation"] = "memory"
            debug_info["details"] = "decl"
        
        # Add role
        if role_override:
            debug_info["role"] = role_override
            
        return debug_info

    def place_ir_operation(self, op: IRNode) -> None:
        """Place a single IR operation."""
        # Track this IR node for later lookups
        self._ir_nodes[op.node_id] = op

        if isinstance(op, IR_Const):
            self._place_constant(op)
        elif isinstance(op, IR_Arith):
            self._place_arithmetic(op)
        elif isinstance(op, IR_Decider):
            self._place_decider(op)
        elif isinstance(op, IR_MemCreate):
            self._place_memory_create(op)
        elif isinstance(op, IR_MemRead):
            self._place_memory_read(op)
        elif isinstance(op, IR_MemWrite):
            self._place_memory_write(op)
        elif isinstance(op, IR_PlaceEntity):
            self._place_user_entity(op)
        elif isinstance(op, IR_EntityPropWrite):
            self._place_entity_prop_write(op)
        elif isinstance(op, IR_EntityPropRead):
            self._place_entity_prop_read(op)
        elif isinstance(op, IR_WireMerge):
            self._place_wire_merge(op)
        else:
            self.diagnostics.warning(f"Unknown IR operation: {type(op)}")

    def _place_constant(self, op: IR_Const) -> None:
        """Place constant combinator (if materialization required)."""
        usage = self.signal_usage.get(op.node_id)
        if not usage or not usage.should_materialize:
            return

        # Determine signal name and type
        signal_name = self.materializer.resolve_signal_name(op.output_type, usage)
        signal_type = self.materializer.resolve_signal_type(op.output_type, usage)

        # Reserve position in north literals zone
        pos = self.layout.reserve_in_zone("north_literals")

        # Build debug info
        debug_info = self._build_debug_info(op)

        # Add fold information if this is a folded constant
        if hasattr(op, "debug_metadata") and op.debug_metadata:
            folded_from = op.debug_metadata.get("folded_from")
            if folded_from:
                debug_info["details"] = f"folded from {len(folded_from)} constants: value={op.value}"
                debug_info["fold_count"] = len(folded_from)

        # Store placement in plan (NOT creating Draftsman entity yet!)
        placement = EntityPlacement(
            ir_node_id=op.node_id,
            entity_type="constant-combinator",
            position=pos,
            properties={
                "signal_name": signal_name,
                "signal_type": signal_type,
                "value": op.value,
                "footprint": (1, 1),
                "debug_info": debug_info,
            },
            role="literal",
            zone="north_literals",
        )
        self.plan.add_placement(placement)

        # Track signal source
        self.signal_graph.set_source(op.node_id, op.node_id)

    def _place_arithmetic(self, op: IR_Arith) -> None:
        """Place arithmetic combinator."""
        # Get dependency positions for smart placement
        left_pos = self._get_placement_position(op.left)
        right_pos = self._get_placement_position(op.right)

        deps = []
        if left_pos:
            deps.append(left_pos)
        if right_pos:
            deps.append(right_pos)

        # Reserve position near dependencies
        if deps:
            avg_x = sum(p[0] for p in deps) / len(deps)
            avg_y = sum(p[1] for p in deps) / len(deps)
            pos = self.layout.reserve_near((avg_x, avg_y), footprint=(1, 1))
        else:
            pos = self.layout.get_next_position(footprint=(1, 1))

        # Resolve operands and output
        usage = self.signal_usage.get(op.node_id)
        left_operand = self.resolver.get_operand_for_combinator(op.left)
        right_operand = self.resolver.get_operand_for_combinator(op.right)
        output_signal = self.materializer.resolve_signal_name(op.output_type, usage)

        # Store placement
        placement = EntityPlacement(
            ir_node_id=op.node_id,
            entity_type="arithmetic-combinator",
            position=pos,
            properties={
                "operation": op.op,
                "left_operand": left_operand,
                "right_operand": right_operand,
                "output_signal": output_signal,
                "footprint": (1, 1),
                "debug_info": self._build_debug_info(op),
            },
            role="arithmetic",
        )
        self.plan.add_placement(placement)

        # Track signals
        self.signal_graph.set_source(op.node_id, op.node_id)
        self._add_signal_sink(op.left, op.node_id)
        self._add_signal_sink(op.right, op.node_id)

    def _place_decider(self, op: IR_Decider) -> None:
        """Place decider combinator."""
        # Get dependency positions
        left_pos = self._get_placement_position(op.left)
        right_pos = self._get_placement_position(op.right)
        output_pos = self._get_placement_position(op.output_value)

        deps = []
        for p in [left_pos, right_pos, output_pos]:
            if p:
                deps.append(p)

        # Reserve position near dependencies
        if deps:
            avg_x = sum(p[0] for p in deps) / len(deps)
            avg_y = sum(p[1] for p in deps) / len(deps)
            pos = self.layout.reserve_near((avg_x, avg_y), footprint=(1, 1))
        else:
            pos = self.layout.get_next_position(footprint=(1, 1))

        # Resolve operands
        usage = self.signal_usage.get(op.node_id)
        left_operand = self.resolver.get_operand_for_combinator(op.left)
        right_operand = self.resolver.get_operand_for_combinator(op.right)
        output_signal = self.materializer.resolve_signal_name(op.output_type, usage)
        output_value = self.resolver.get_operand_for_combinator(op.output_value)

        # Determine copy_count_from_input flag
        copy_count_from_input = not isinstance(op.output_value, int)

        # Store placement
        placement = EntityPlacement(
            ir_node_id=op.node_id,
            entity_type="decider-combinator",
            position=pos,
            properties={
                "operation": op.test_op,
                "left_operand": left_operand,
                "right_operand": right_operand,
                "output_signal": output_signal,
                "output_value": output_value,
                "copy_count_from_input": copy_count_from_input,
                "footprint": (1, 1),
                "debug_info": self._build_debug_info(op),
            },
            role="decider",
        )
        self.plan.add_placement(placement)

        # Track signals
        self.signal_graph.set_source(op.node_id, op.node_id)
        self._add_signal_sink(op.left, op.node_id)
        self._add_signal_sink(op.right, op.node_id)
        if not isinstance(op.output_value, int):
            self._add_signal_sink(op.output_value, op.node_id)

    def _place_memory_create(self, op: IR_MemCreate) -> None:
        """Place memory module components."""
        # Simplified memory placement - create basic SR latch structure
        # The full memory builder will be refactored later
        signal_name = self.resolver.get_signal_name(op.signal_type)

        # Create write gate
        write_pos = self.layout.get_next_position(footprint=(1, 1))
        write_id = f"{op.memory_id}_write_gate"
        
        # Build debug info for memory write gate
        write_debug = {
            "variable": f"mem:{op.memory_id}",
            "operation": "memory",
            "details": "write_gate",
            "signal_type": signal_name,
            "role": "memory_write_gate",
        }
        if hasattr(op, "source_ast") and op.source_ast:
            if hasattr(op.source_ast, "line"):
                write_debug["line"] = op.source_ast.line
            if hasattr(op.source_ast, "source_file"):
                write_debug["source_file"] = op.source_ast.source_file
        
        write_placement = EntityPlacement(
            ir_node_id=write_id,
            entity_type="decider-combinator",
            position=write_pos,
            properties={
                "footprint": (1, 1),
                "operation": ">",
                "left_operand": "signal-W",
                "right_operand": 0,
                "output_signal": signal_name,
                "copy_count_from_input": True,
                "debug_info": write_debug,
            },
            role="memory_write_gate",
            zone="memory",
        )
        self.plan.add_placement(write_placement)

        # Create hold gate
        hold_pos = self.layout.get_next_position(footprint=(1, 1))
        hold_id = f"{op.memory_id}_hold_gate"
        
        # Build debug info for memory hold gate
        hold_debug = {
            "variable": f"mem:{op.memory_id}",
            "operation": "memory",
            "details": "hold_gate",
            "signal_type": signal_name,
            "role": "memory_hold_gate",
        }
        if hasattr(op, "source_ast") and op.source_ast:
            if hasattr(op.source_ast, "line"):
                hold_debug["line"] = op.source_ast.line
            if hasattr(op.source_ast, "source_file"):
                hold_debug["source_file"] = op.source_ast.source_file
        
        hold_placement = EntityPlacement(
            ir_node_id=hold_id,
            entity_type="decider-combinator",
            position=hold_pos,
            properties={
                "footprint": (1, 1),
                "operation": "=",
                "left_operand": "signal-W",
                "right_operand": 0,
                "output_signal": signal_name,
                "copy_count_from_input": True,
                "debug_info": hold_debug,
            },
            role="memory_hold_gate",
            zone="memory",
        )
        self.plan.add_placement(hold_placement)

        # Store memory module components
        self._memory_modules[op.memory_id] = {
            "write_gate": write_placement,
            "hold_gate": hold_placement,
            "signal_type": signal_name,
            "always_write": False,
        }

        # Track signal source - memory output comes from hold gate
        self.signal_graph.set_source(op.memory_id, hold_id)

    def _place_memory_read(self, op: IR_MemRead) -> None:
        """Memory reads are passive - they just connect to the memory's output."""
        memory_module = self._memory_modules.get(op.memory_id)
        if not memory_module:
            self.diagnostics.warning(
                f"Memory read from undefined memory: {op.memory_id}"
            )
            return

        # Track which memory this read came from
        self._memory_read_sources[op.node_id] = op.memory_id

        optimization = memory_module.get("optimization")

        # Handle optimized memories
        if optimization == "arithmetic_feedback":
            output_node_id = memory_module.get("output_node_id")
            if output_node_id:
                self.signal_graph.set_source(op.node_id, output_node_id)
                return

        if optimization == "single_gate":
            write_gate = memory_module.get("write_gate")
            if isinstance(write_gate, EntityPlacement):
                self.signal_graph.set_source(op.node_id, write_gate.ir_node_id)
                return

        # Standard SR latch
        hold_gate = memory_module.get("hold_gate")
        if not isinstance(hold_gate, EntityPlacement):
            self.diagnostics.warning(
                f"Memory module '{op.memory_id}' is missing hold gate placement"
            )
            return

        # Ensure reads source the actual hold gate so wiring resolves correctly.
        self.signal_graph.set_source(op.node_id, hold_gate.ir_node_id)

    def _place_memory_write(self, op: IR_MemWrite) -> None:
        """Place memory write circuitry and record necessary wiring."""
        
        memory_module = self._memory_modules.get(op.memory_id)
        if not memory_module:
            self.diagnostics.warning(
                f"Memory write to undefined memory: {op.memory_id}"
            )
            return
        
        # ✅ NEW: Detect multiple writes to same memory
        if memory_module.get("_has_write"):
            self.diagnostics.warning(
                f"Multiple writes to memory '{op.memory_id}' detected. "
                f"Only the last write will be optimized. "
                f"Consider using a single write with conditional logic instead."
            )
        memory_module["_has_write"] = True
        
        # Check if write_enable is a constant 1
        is_always_write = False
        if isinstance(op.write_enable, int) and op.write_enable == 1:
            is_always_write = True
        elif isinstance(op.write_enable, SignalRef):
            # Check if it's a reference to a constant 1 by looking up the IR node
            const_ir = self._ir_nodes.get(op.write_enable.source_id)
            if isinstance(const_ir, IR_Const) and const_ir.value == 1:
                is_always_write = True

        write_gate = memory_module.get("write_gate")
        hold_gate = memory_module.get("hold_gate")

        if not isinstance(write_gate, EntityPlacement) or not isinstance(
            hold_gate, EntityPlacement
        ):
            self.diagnostics.warning(
                f"Memory module '{op.memory_id}' is missing gate placements"
            )
            return

        if is_always_write:
            # Check if this can be optimized to arithmetic feedback (Fix 2)
            if self._can_use_arithmetic_feedback(op, memory_module):
                self._optimize_to_arithmetic_feedback(op, memory_module)
                return

            # Otherwise use simplified single-gate memory
            self._place_single_gate_memory(op, memory_module)
            return

        # Standard SR latch for conditional writes
        # ✅ FIX: Only connect data to write gate, NOT to hold gate!
        self._add_signal_sink(op.data_signal, write_gate.ir_node_id)
        # REMOVED: self._add_signal_sink(op.data_signal, hold_gate.ir_node_id)

        enable_source = self._ensure_constant_write_enable(op)
        if enable_source:
            self.signal_graph.add_sink(enable_source, write_gate.ir_node_id)
            self.signal_graph.add_sink(enable_source, hold_gate.ir_node_id)
        else:
            self._add_signal_sink(op.write_enable, write_gate.ir_node_id)
            self._add_signal_sink(op.write_enable, hold_gate.ir_node_id)

        if not memory_module.get("_feedback_connected"):
            signal_name = memory_module.get("signal_type")
            if signal_name:
                feedback_conn = WireConnection(
                    source_entity_id=write_gate.ir_node_id,
                    sink_entity_id=hold_gate.ir_node_id,
                    signal_name=signal_name,
                    wire_color="red",
                    source_side="output",
                    sink_side="input",
                )
                self.plan.add_wire_connection(feedback_conn)

                hold_feedback = WireConnection(
                    source_entity_id=hold_gate.ir_node_id,
                    sink_entity_id=hold_gate.ir_node_id,
                    signal_name=signal_name,
                    wire_color="red",
                    source_side="output",
                    sink_side="input",
                )
                self.plan.add_wire_connection(hold_feedback)

                memory_module["_feedback_connected"] = True

    def _can_use_arithmetic_feedback(
        self, op: IR_MemWrite, memory_module: dict
    ) -> bool:
        """Detect if always-write memory can use arithmetic self-feedback optimization.

        Returns True if:
        - Write is always-on (when=1)
        - Written value comes from arithmetic operation(s)
        - At least one operation in the chain reads from this same memory
        """
        if not isinstance(op.data_signal, SignalRef):
            return False

        # Find the operation that produces the data signal
        final_op_id = op.data_signal.source_id
        final_placement = self.plan.get_placement(final_op_id)

        if final_placement is None:
            return False

        # Must be an arithmetic combinator
        if final_placement.entity_type != "arithmetic-combinator":
            return False

        # Check if this arithmetic operation (or its inputs) reads from the memory
        return self._operation_depends_on_memory(final_op_id, op.memory_id)

    def _operation_depends_on_memory(
        self, op_id: str, memory_id: str, visited: set = None, depth: int = 0
    ) -> bool:
        """Check if an operation depends on a memory read (directly or transitively)."""
        if depth > 10:  # Prevent infinite recursion
            return False

        if visited is None:
            visited = set()

        if op_id in visited:
            return False
        visited.add(op_id)

        # Check if this operation IS a memory read from our memory
        source_memory = self._memory_read_sources.get(op_id)
        if source_memory == memory_id:
            return True

        # Check if any of the inputs to this operation depend on the memory
        placement = self.plan.get_placement(op_id)
        if not placement:
            return False

        # Check if this is an arithmetic operation - get its inputs from IR node
        if placement.entity_type == "arithmetic-combinator":
            # Look up the IR node to get the actual operands
            ir_node = self._ir_nodes.get(op_id)
            if not isinstance(ir_node, IR_Arith):
                return False

            left = ir_node.left
            right = ir_node.right

            # Check if either operand is a SignalRef that depends on memory
            if isinstance(left, SignalRef):
                if self._operation_depends_on_memory(
                    left.source_id, memory_id, visited, depth + 1
                ):
                    return True
            if isinstance(right, SignalRef):
                if self._operation_depends_on_memory(
                    right.source_id, memory_id, visited, depth + 1
                ):
                    return True

        return False

    def _find_first_memory_consumer(
        self, 
        memory_id: str, 
        final_op_id: str
    ) -> Optional[str]:
        """Find the first operation in the chain that reads from memory.
        
        Args:
            memory_id: Memory being optimized
            final_op_id: Final operation in the chain
            
        Returns:
            Entity ID of first consumer, or None
        """
        # Check all memory reads that were registered
        for read_node_id, source_memory_id in self._memory_read_sources.items():
            if source_memory_id != memory_id:
                continue
                
            # Find operations that consume this read
            consumers = list(self.signal_graph.iter_sinks(read_node_id))
            if consumers:
                # Return the first consumer (start of the chain)
                return consumers[0]
        
        return None

    def _optimize_to_arithmetic_feedback(
        self, op: IR_MemWrite, memory_module: dict
    ) -> None:
        """Optimize always-write memory to use arithmetic combinator feedback.
        
        For single-operation chains: Use self-feedback on one combinator
        For multi-operation chains: Wire a feedback loop between combinators
        """
        final_op_id = op.data_signal.source_id
        final_placement = self.plan.get_placement(final_op_id)

        if final_placement is None:
            return

        signal_name = memory_module.get("signal_type")

        # Determine if this is a single-operation or multi-operation chain
        first_consumer_id = self._find_first_memory_consumer(op.memory_id, final_op_id)
        is_single_operation = (first_consumer_id == final_op_id or first_consumer_id is None)

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
            # Do NOT add self-feedback marker
            self.diagnostics.info(
                f"Optimized memory '{op.memory_id}' to multi-combinator feedback loop"
            )

        # Mark memory gates as unused (they'll be removed in cleanup)
        memory_module["optimization"] = "arithmetic_feedback"
        memory_module["output_node_id"] = final_op_id
        memory_module["write_gate_unused"] = True
        memory_module["hold_gate_unused"] = True

        # Update signal graph: all memory reads now come from final combinator
        self.signal_graph.set_source(op.memory_id, final_op_id)
        
        for read_node_id, source_memory_id in self._memory_read_sources.items():
            if source_memory_id == op.memory_id:
                self.signal_graph.set_source(read_node_id, final_op_id)
        
        # For multi-operation chains: register feedback edge in signal graph
        if first_consumer_id and first_consumer_id != final_op_id:
            # Add edge: final_op produces signal that first_consumer needs
            self.signal_graph.add_sink(final_op_id, first_consumer_id)
            self.diagnostics.info(
                f"Registered feedback loop: {final_op_id} -> {first_consumer_id}"
            )

    def _place_single_gate_memory(self, op: IR_MemWrite, memory_module: dict) -> None:
        """Fallback: simplified single-gate always-write memory."""
        write_gate = memory_module.get("write_gate")
        if not isinstance(write_gate, EntityPlacement):
            return

        signal_name = memory_module.get("signal_type")

        # Connect data input
        self._add_signal_sink(op.data_signal, write_gate.ir_node_id)

        # Configure as always-on latch
        write_gate.properties["operation"] = ">"
        write_gate.properties["left_operand"] = signal_name
        write_gate.properties["right_operand"] = -2147483648
        write_gate.properties["copy_count_from_input"] = True

        # Add self-feedback
        if not memory_module.get("_feedback_connected"):
            self_feedback = WireConnection(
                source_entity_id=write_gate.ir_node_id,
                sink_entity_id=write_gate.ir_node_id,
                signal_name=signal_name,
                wire_color="red",
                source_side="output",
                sink_side="input",
            )
            self.plan.add_wire_connection(self_feedback)
            memory_module["_feedback_connected"] = True

        # Mark hold gate as unused
        memory_module["optimization"] = "single_gate"
        memory_module["hold_gate_unused"] = True

        self.signal_graph.set_source(op.memory_id, write_gate.ir_node_id)

    def _ensure_constant_write_enable(self, op: IR_MemWrite) -> Optional[str]:
        """Materialize or reuse a write-enable source."""
        if isinstance(op.write_enable, int):
            const_id = f"{op.memory_id}_write_enable_{self.next_entity_number}"
            self.next_entity_number += 1
            return self._create_write_enable_constant(const_id, op.write_enable)

        if isinstance(op.write_enable, SignalRef):
            source_id = op.write_enable.source_id
            entry = self.signal_usage.get(source_id)
            placement = self.plan.get_placement(source_id)
            if placement is not None:
                if entry and entry.literal_value is not None:
                    placement.properties["value"] = int(entry.literal_value)
                placement.properties["signal_name"] = "signal-W"
                placement.properties["signal_type"] = "virtual"
                self.signal_graph.set_source(source_id, placement.ir_node_id)
                if entry:
                    entry.should_materialize = True
                    entry.resolved_signal_name = "signal-W"
                    entry.resolved_signal_type = "virtual"
                return source_id

            if entry and isinstance(entry.producer, IR_Const):
                value = entry.literal_value
                if value is None:
                    value = getattr(entry.producer, "value", 0)
                created_id = self._create_write_enable_constant(source_id, int(value))
                entry.should_materialize = True
                entry.resolved_signal_name = "signal-W"
                entry.resolved_signal_type = "virtual"
                return created_id

            return source_id

        return None

    def _create_write_enable_constant(self, constant_id: str, value: int) -> str:
        position = self.layout.reserve_in_zone("north_literals")
        placement = EntityPlacement(
            ir_node_id=constant_id,
            entity_type="constant-combinator",
            position=position,
            properties={
                "signal_name": "signal-W",
                "signal_type": "virtual",
                "value": value,
                "footprint": (1, 1),
                "debug_info": {
                    "variable": "write_enable",
                    "operation": "const",
                    "details": f"value={value}",
                    "signal_type": "signal-W",
                    "role": "write_enable_constant",
                },
            },
            role="write_enable_constant",
            zone="north_literals",
        )
        self.plan.add_placement(placement)
        self.signal_graph.set_source(constant_id, placement.ir_node_id)
        return constant_id

    def _place_user_entity(self, op: IR_PlaceEntity) -> None:
        """Place user-requested entity."""
        prototype = op.prototype

        # Determine footprint dynamically from draftsman
        footprint = get_entity_footprint(prototype)

        # Determine alignment requirement dynamically
        alignment = get_entity_alignment(prototype)

        # Determine position
        if isinstance(op.x, int) and isinstance(op.y, int):
            # Explicit position provided
            desired = (int(op.x), int(op.y))
            pos = self.layout.reserve_near(
                desired,
                max_radius=4,
                footprint=footprint,
                alignment=alignment,
            )
        else:
            pos = self.layout.get_next_position(
                footprint=footprint, alignment=alignment
            )

        # Store placement
        placement = EntityPlacement(
            ir_node_id=op.entity_id,
            entity_type=prototype,
            position=pos,
            properties=op.properties or {},
            role="user_entity",
            alignment=alignment,
        )
        placement.properties["footprint"] = footprint
        
        # Add debug info for user-placed entities
        entity_debug = {
            "variable": op.entity_id,
            "operation": "place",
            "details": f"proto={prototype}",
            "role": "user_entity",
        }
        if hasattr(op, "source_ast") and op.source_ast:
            if hasattr(op.source_ast, "line"):
                entity_debug["line"] = op.source_ast.line
            if hasattr(op.source_ast, "source_file"):
                entity_debug["source_file"] = op.source_ast.source_file
        
        placement.properties["debug_info"] = entity_debug
        
        self.plan.add_placement(placement)

    def _place_entity_prop_write(self, op: IR_EntityPropWrite) -> None:
        """Handle entity property writes."""
        # Get the entity placement
        placement = self.plan.get_placement(op.entity_id)
        if placement is None:
            self.diagnostics.warning(
                f"Property write to non-existent entity: {op.entity_id}"
            )
            return

        # Store property writes for later application during entity creation
        if "property_writes" not in placement.properties:
            placement.properties["property_writes"] = {}

        # Try to inline simple comparisons
        if isinstance(op.value, SignalRef) and op.property_name == "enable":
            inline_data = self._try_inline_comparison(op.value)
            if inline_data:
                placement.properties["property_writes"][op.property_name] = {
                    "type": "inline_comparison",
                    "comparison_data": inline_data,
                }
                # Mark the comparison decider for removal
                inline_data["source_node_id_to_remove"] = op.value.source_id
                
                # Preserve debug info from the inlined comparison
                comparison_placement = self.plan.get_placement(op.value.source_id)
                if comparison_placement and "debug_info" in comparison_placement.properties:
                    comp_debug = comparison_placement.properties["debug_info"]
                    if "property_writes" in placement.properties:
                        writes = placement.properties["property_writes"]
                        if op.property_name in writes and writes[op.property_name].get("type") == "inline_comparison":
                            writes[op.property_name]["inlined_from"] = comp_debug.get("variable", "comparison")
                
                # ✅ FIX: Track that entity needs the comparison's input signal
                # The entity must read the signal being compared
                ir_node = self._ir_nodes.get(op.value.source_id)
                if isinstance(ir_node, IR_Decider):
                    # Add dependency on the left operand (the signal being tested)
                    if isinstance(ir_node.left, SignalRef):
                        # Remove the old edge from source to decider
                        self.signal_graph.remove_sink(ir_node.left.source_id, op.value.source_id)
                        # Add new edge from source to entity
                        self._add_signal_sink(ir_node.left, op.entity_id)
                        self.diagnostics.info(
                            f"Inlined comparison into {op.entity_id}.{op.property_name}, "
                            f"tracking signal dependency"
                        )
                else:
                    self.diagnostics.info(
                        f"Inlined comparison into {op.entity_id}.{op.property_name}"
                    )
                
                return

        # Resolve the value
        if isinstance(op.value, SignalRef):
            # Property is controlled by a signal - track dependency
            self._add_signal_sink(op.value, op.entity_id)
            placement.properties["property_writes"][op.property_name] = {
                "type": "signal",
                "signal_ref": op.value,
            }
        elif isinstance(op.value, int):
            # Constant value
            placement.properties["property_writes"][op.property_name] = {
                "type": "constant",
                "value": op.value,
            }
        else:
            # Other value type
            placement.properties["property_writes"][op.property_name] = {
                "type": "value",
                "value": op.value,
            }

    def _try_inline_comparison(self, signal_ref: SignalRef) -> Optional[dict]:
        """Check if a signal is a simple comparison that can be inlined."""
        source_placement = self.plan.get_placement(signal_ref.source_id)
        if not source_placement or source_placement.entity_type != "decider-combinator":
            return None

        props = source_placement.properties

        # Must be a simple comparison: signal OP constant -> 1
        left = props.get("left_operand")
        right = props.get("right_operand")
        operation = props.get("operation")
        output_value = props.get("output_value")

        # Check if it's a simple pattern
        if not (isinstance(right, int) and output_value == 1):
            return None

        # Check if this comparison is ONLY used for this property (no other consumers)
        sinks = list(self.signal_graph.iter_sinks(signal_ref.source_id))
        if len(sinks) > 1:
            # Multiple consumers - can't inline
            return None

        return {
            "left_signal": left,
            "comparator": operation,
            "right_constant": right,
            "signal_type": signal_ref.signal_type,
        }

    def _place_entity_prop_read(self, op: IR_EntityPropRead) -> None:
        """Handle entity property reads."""
        # Property reads expose entity state as a signal
        # We need to create a virtual signal source
        signal_name = f"{op.entity_id}_{op.property_name}"
        self._entity_property_signals[op.node_id] = signal_name
        self.signal_graph.set_source(op.node_id, op.entity_id)

    def _place_wire_merge(self, op: IR_WireMerge) -> None:
        """Handle wire merge operations."""
        # Wire merges don't create entities, they just affect wiring topology
        # Record the merge junction for later wire planning
        self._wire_merge_junctions[op.node_id] = {
            "inputs": list(op.sources),
            "output_id": op.node_id,
        }

        # Track signal graph: merge creates a new source from multiple inputs
        self.signal_graph.set_source(op.node_id, op.node_id)
        for input_sig in op.sources:
            self._add_signal_sink(input_sig, op.node_id)

    def _get_placement_position(self, value_ref: ValueRef) -> Optional[Tuple[int, int]]:
        """Get position of entity producing this value."""
        if isinstance(value_ref, SignalRef):
            placement = self.plan.get_placement(value_ref.source_id)
            return placement.position if placement else None
        return None

    def _add_signal_sink(self, value_ref: ValueRef, consumer_id: str) -> None:
        """Track signal consumption."""
        if isinstance(value_ref, SignalRef):
            self.signal_graph.add_sink(value_ref.source_id, consumer_id)

    def cleanup_unused_entities(self) -> None:
        """Remove entities marked as unused during optimization."""
        entities_to_remove = []

        # Remove unused memory gates
        for memory_id, memory_module in self._memory_modules.items():
            # Remove unused write gates
            if memory_module.get("write_gate_unused"):
                write_gate = memory_module.get("write_gate")
                if isinstance(write_gate, EntityPlacement):
                    entities_to_remove.append(write_gate.ir_node_id)

            # Remove unused hold gates
            if memory_module.get("hold_gate_unused"):
                hold_gate = memory_module.get("hold_gate")
                if isinstance(hold_gate, EntityPlacement):
                    entities_to_remove.append(hold_gate.ir_node_id)

        # Remove inlined comparisons
        for entity_id, placement in list(self.plan.entity_placements.items()):
            for prop_writes in placement.properties.get("property_writes", {}).values():
                if prop_writes.get("type") == "inline_comparison":
                    node_to_remove = prop_writes.get("comparison_data", {}).get(
                        "source_node_id_to_remove"
                    )
                    if node_to_remove and node_to_remove not in entities_to_remove:
                        entities_to_remove.append(node_to_remove)

        for entity_id in entities_to_remove:
            removed = self.plan.entity_placements.pop(entity_id, None)
            if removed:
                self.diagnostics.info(f"Removed unused entity: {entity_id}")
