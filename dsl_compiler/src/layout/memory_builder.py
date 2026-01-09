"""Memory module construction for circuit-based memory cells."""

from dataclasses import dataclass, field
from typing import Any

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.ir.builder import (
    IRArith,
    IRConst,
    IRMemCreate,
    IRMemRead,
    IRMemWrite,
    IRNode,
    SignalRef,
)
from dsl_compiler.src.ir.nodes import (
    MEMORY_TYPE_RS_LATCH,
    MEMORY_TYPE_SR_LATCH,
    MEMORY_TYPE_STANDARD,
    IRLatchWrite,
)

from .layout_plan import EntityPlacement, LayoutPlan, WireConnection
from .signal_analyzer import SignalAnalyzer
from .signal_graph import SignalGraph
from .tile_grid import TileGrid


@dataclass
class MemoryModule:
    """Represents a memory cell's physical implementation.

    Standard memories use a write-gated latch (2 deciders).
    RS/SR latches use a single decider combinator + optional multiplier.
    Optimized memories use fewer components.
    """

    memory_id: str
    signal_type: str
    memory_type: str = MEMORY_TYPE_STANDARD

    # Standard memory gates (write-gated latch)
    write_gate: EntityPlacement | None = None
    hold_gate: EntityPlacement | None = None

    # Latch combinator (RS/SR latches - single combinator)
    latch_combinator: EntityPlacement | None = None

    # Multiplier combinator (for latch values != 1)
    multiplier_combinator: EntityPlacement | None = None

    optimization: str | None = None  # None, 'single_gate', 'arithmetic_feedback'
    output_node_id: str | None = None  # For optimized memories

    write_gate_unused: bool = False
    hold_gate_unused: bool = False
    _feedback_connected: bool = False
    _has_write: bool = False

    _feedback_signal_ids: list[str] = field(default_factory=list)


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

        self._modules: dict[str, MemoryModule] = {}
        self._read_sources: dict[str, str] = {}  # read_node_id -> memory_id
        self._ir_nodes: dict[str, IRNode] = {}  # For optimization detection

    def register_ir_node(self, node: IRNode):
        """Track IR node for optimization detection."""
        self._ir_nodes[node.node_id] = node

    def create_memory(self, op: IRMemCreate, signal_graph: SignalGraph) -> MemoryModule:
        """Create memory cell.

        All memories start as standard write-gated latches.
        If a latch write occurs, the memory is upgraded to an RS/SR latch.

        Returns MemoryModule with appropriate placements.
        """
        return self._create_standard_memory(op, signal_graph)

    def _create_standard_memory(self, op: IRMemCreate, signal_graph: SignalGraph) -> MemoryModule:
        """Create write-gated latch for a standard memory cell.

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
            memory_type=MEMORY_TYPE_STANDARD,
            write_gate=write_placement,
            hold_gate=hold_placement,
        )
        self._modules[op.memory_id] = module

        signal_graph.set_source(op.memory_id, hold_id)

        return module

    def _setup_latch_feedback(self, module: MemoryModule, signal_graph: SignalGraph) -> None:
        """Set up green wire self-feedback for latch combinators."""
        if not module.latch_combinator:
            return

        latch_id = module.latch_combinator.ir_node_id

        # Create internal signal ID for layout purposes
        feedback_signal = f"__feedback_{module.memory_id}_latch"
        signal_graph.set_source(feedback_signal, latch_id)
        signal_graph.add_sink(feedback_signal, latch_id)

        # Create GREEN wire self-feedback connection
        feedback_conn = WireConnection(
            source_entity_id=latch_id,
            sink_entity_id=latch_id,
            signal_name=module.signal_type,
            wire_color="green",  # Green for feedback
            source_side="output",
            sink_side="input",
        )
        self.layout_plan.add_wire_connection(feedback_conn)

        module._feedback_signal_ids = [feedback_signal]
        module._feedback_connected = True

        self.diagnostics.info(
            f"Set up GREEN wire self-feedback for {module.memory_type} '{module.memory_id}'"
        )

    def handle_read(self, op: IRMemRead, signal_graph: SignalGraph):
        """Connect read to memory output.

        For standard memories: connects to hold_gate output
        For latch memories: connects to latch_combinator output
        """
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

        # For latch memories with multiplier, use the multiplier as the source
        if module.multiplier_combinator:
            signal_graph.set_source(op.node_id, module.multiplier_combinator.ir_node_id)
            return

        # For latch memories without multiplier, use the latch combinator
        if module.latch_combinator:
            signal_graph.set_source(op.node_id, module.latch_combinator.ir_node_id)
            return

        # For standard memories, use the hold gate
        if module.hold_gate:
            signal_graph.set_source(op.node_id, module.hold_gate.ir_node_id)

    def handle_write(self, op: IRMemWrite, signal_graph: SignalGraph):
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

        if is_always_write and self._can_use_arithmetic_feedback(op, module):
            self._optimize_to_arithmetic_feedback(op, module, signal_graph)
            return

        self._setup_standard_write(op, module, signal_graph)

    def handle_latch_write(self, op: IRLatchWrite, signal_graph: SignalGraph):
        """Handle latch write: creates RS/SR latch circuit.

        BINARY LATCH DESIGN:
        SR/RS latches are inherently binary (output 0 or 1). To output arbitrary
        values, we use a multiplier combinator.

        Compilation patterns:
        - write(1, set=..., reset=...) → 1 decider combinator
        - write(N, set=..., reset=...) where N ≠ 1 → 2 combinators (latch + multiplier)

        RS Latch (Reset Priority):
            Single condition: S > R
            When both S and R are active, R wins (0 > 0 is false)

        SR Latch (Set Priority):
            Multi-condition with wire filtering (Factorio 2.0):
            Row 1: S > R (read from RED wire only) [OR]
            Row 2: S > 0 (read from GREEN wire only - feedback)
            When R is active but feedback S=1 is still present, the latch stays ON.

        Wire Configuration:
            RED wire: External set (S) and reset (R) signals
            GREEN wire: Feedback from output to input (self-loop)
        """
        # Get the memory module
        module = self._modules.get(op.memory_id)
        if not module:
            self.diagnostics.warning(f"Latch write for undefined memory '{op.memory_id}'")
            return

        # Upgrade memory module to latch type
        module.memory_type = op.latch_type

        # Get signal names for set and reset from the SignalRefs
        if isinstance(op.set_signal, SignalRef):
            original_set_signal = self.signal_analyzer.get_signal_name(op.set_signal.signal_type)
            set_source_id = op.set_signal.source_id
        else:
            original_set_signal = "signal-S"
            set_source_id = None

        if isinstance(op.reset_signal, SignalRef):
            original_reset_signal = self.signal_analyzer.get_signal_name(
                op.reset_signal.signal_type
            )
            reset_source_id = op.reset_signal.source_id
        else:
            original_reset_signal = "signal-R"
            reset_source_id = None

        # The latch uses the memory's declared signal type for output AND set comparison
        memory_signal_type = module.signal_type
        latch_id = f"{op.memory_id}_latch"

        # ======================================================================
        # STEP 1: Set signal remapping (if set signal type != memory signal type)
        # ======================================================================
        # The set signal must match the memory signal type for the latch to work.
        # If they differ, add a combinator to cast: original_set → memory_signal_type
        needs_set_remap = original_set_signal != memory_signal_type
        set_remapper_id = None

        if needs_set_remap:
            self.diagnostics.warning(
                f"Latch '{op.memory_id}': casting set signal from '{original_set_signal}' "
                f"to memory type '{memory_signal_type}'. Consider using matching signal types to reduce the number of combinators.",
                stage="layout",
            )
            set_remapper_id = f"{op.memory_id}_set_remap"
            self._create_signal_remapper(
                set_remapper_id, op, original_set_signal, memory_signal_type, signal_graph
            )
            # Wire set source to set remapper (via signal graph - creates red wire)
            if set_source_id:
                signal_graph.add_sink(set_source_id, set_remapper_id)

            # Wire set remapper output to latch input via EXPLICIT red wire connection
            # (signal graph edges don't automatically create this wire because the
            # remapper output is a different signal type than the input)
            set_remap_to_latch = WireConnection(
                source_entity_id=set_remapper_id,
                sink_entity_id=latch_id,
                signal_name=memory_signal_type,  # The remapped signal
                wire_color="red",  # External inputs on RED
                source_side="output",
                sink_side="input",
            )
            self.layout_plan.add_wire_connection(set_remap_to_latch)

            # The latch now uses memory_signal_type as the set signal
            set_signal_for_latch = memory_signal_type
            # Clear set_source_id so we don't wire original source directly to latch
            set_source_id = None
        else:
            set_signal_for_latch = original_set_signal

        # ======================================================================
        # STEP 2: Reset signal remapping (if reset signal = set signal after casting)
        # ======================================================================
        # After step 1, the set signal used by the latch is memory_signal_type.
        # If reset signal = memory_signal_type, they conflict → remap reset
        needs_reset_remap = original_reset_signal == memory_signal_type
        reset_remapper_id = None
        internal_reset_signal = "signal-dot"  # Internal signal for remapped reset

        if needs_reset_remap:
            self.diagnostics.info(
                f"Latch '{op.memory_id}': reset signal '{original_reset_signal}' conflicts with "
                f"memory type. Casting to internal signal '{internal_reset_signal}'. Consider using different signal types to reduce the number of combinators.",
                stage="layout",
            )
            reset_remapper_id = f"{op.memory_id}_reset_remap"
            self._create_signal_remapper(
                reset_remapper_id, op, original_reset_signal, internal_reset_signal, signal_graph
            )
            # Wire reset source to reset remapper (via signal graph - creates red wire)
            if reset_source_id:
                signal_graph.add_sink(reset_source_id, reset_remapper_id)

            # Wire reset remapper output to latch input via EXPLICIT red wire connection
            reset_remap_to_latch = WireConnection(
                source_entity_id=reset_remapper_id,
                sink_entity_id=latch_id,
                signal_name=internal_reset_signal,  # The remapped signal
                wire_color="red",  # External inputs on RED
                source_side="output",
                sink_side="input",
            )
            self.layout_plan.add_wire_connection(reset_remap_to_latch)
            # The latch now uses internal_reset_signal as the reset signal
            reset_signal_for_latch = internal_reset_signal
            # Clear reset_source_id so we don't wire original source directly to latch
            reset_source_id = None
        else:
            reset_signal_for_latch = original_reset_signal

        # ======================================================================
        # STEP 3: Determine multiplier need
        # ======================================================================
        value_is_signal = isinstance(op.value, SignalRef)
        latch_value: int | SignalRef = (
            op.value if value_is_signal else (op.value if isinstance(op.value, int) else 1)  # type: ignore[assignment]
        )
        needs_multiplier = value_is_signal or (isinstance(latch_value, int) and latch_value != 1)

        # The latch outputs on the MEMORY's declared signal type
        latch_output_signal = memory_signal_type
        latch_output_constant = 1  # Binary latch always outputs 1

        # Mark standard gates as unused (latch replaces them)
        if module.write_gate:
            module.write_gate_unused = True
        if module.hold_gate:
            module.hold_gate_unused = True

        # Build the latch placement based on latch type
        # Both use the same signal names now: set_signal_for_latch and reset_signal_for_latch
        if op.latch_type == MEMORY_TYPE_RS_LATCH:
            # RS Latch: Single condition S > R
            latch_placement = self._create_rs_latch_placement(
                latch_id,
                op,
                set_signal_for_latch,
                reset_signal_for_latch,
                latch_output_signal,
                latch_output_constant,
            )
        else:
            # SR Latch: Multi-condition with wire filtering
            latch_placement = self._create_sr_latch_placement(
                latch_id,
                op,
                set_signal_for_latch,
                reset_signal_for_latch,
                latch_output_signal,
                latch_output_constant,
            )

        module.latch_combinator = latch_placement

        # Set up green wire self-feedback (output → input)
        self._setup_latch_feedback(module, signal_graph)

        # Connect set and reset signal sources to latch input (via RED wire)
        # Note: These may be None if they were wired to remappers instead
        if set_source_id:
            signal_graph.add_sink(set_source_id, latch_id)
        if reset_source_id:
            signal_graph.add_sink(reset_source_id, latch_id)

        # Handle multiplier pattern for values != 1 or signal values
        if needs_multiplier:
            self._create_latch_multiplier(
                op, module, latch_id, latch_output_signal, latch_value, signal_graph
            )
            # Memory reads come from the multiplier output
            multiplier_id = f"{op.memory_id}_multiplier"
            signal_graph.set_source(op.memory_id, multiplier_id)

            if isinstance(latch_value, SignalRef):
                value_str = f"signal {latch_value.signal_type}"
            else:
                value_str = str(latch_value)
            self.diagnostics.info(
                f"Created latch with multiplier for '{op.memory_id}': "
                f"latch outputs {latch_output_signal}=1, multiplier scales by {value_str}"
            )
        else:
            # No multiplier needed, latch is the memory source
            signal_graph.set_source(op.memory_id, latch_id)

            priority = (
                "SR (set priority)"
                if op.latch_type == MEMORY_TYPE_SR_LATCH
                else "RS (reset priority)"
            )
            self.diagnostics.info(
                f"Created {priority} latch '{op.memory_id}': output={latch_output_signal}=1"
            )

    def _create_latch_multiplier(
        self,
        op: IRLatchWrite,
        module: MemoryModule,
        latch_id: str,
        latch_signal: str,
        multiplier_value: int | SignalRef,
        signal_graph: SignalGraph,
    ) -> EntityPlacement:
        """Create arithmetic combinator to scale latch output.

        The latch outputs 1 on the set signal. This multiplier scales it to
        the desired value on the memory's signal type.

        For constant values: latch_signal × constant → memory_signal_type
        For signal values: latch_signal × signal_value → memory_signal_type

        Wire selection is critical for signal values:
        - Left operand (latch output): GREEN wire only (from latch feedback)
        - Right operand (signal value): RED wire only (from signal source)
        """
        multiplier_id = f"{op.memory_id}_multiplier"

        # The multiplier outputs on the memory's declared signal type
        output_signal = module.signal_type

        # Left operand ALWAYS reads from green wire only (latch output)
        left_operand_wires = {"green"}

        # Determine right operand based on value type
        if isinstance(multiplier_value, SignalRef):
            # Signal value: wire the source and use signal name
            right_operand = self.signal_analyzer.get_signal_name(multiplier_value.signal_type)
            # Right operand reads from red wire only (signal source)
            right_operand_wires = {"red"}
            # Connect signal source to multiplier input via red wire
            if multiplier_value.source_id:
                signal_graph.add_sink(multiplier_value.source_id, multiplier_id)
        else:
            # Constant value - no wire selection needed for constants
            right_operand = multiplier_value
            right_operand_wires = {"red", "green"}  # Doesn't matter for constants

        multiplier_placement = self.layout_plan.create_and_add_placement(
            ir_node_id=multiplier_id,
            entity_type="arithmetic-combinator",
            position=None,
            footprint=(1, 2),
            role="latch_multiplier",
            debug_info=self._make_multiplier_debug_info(op, multiplier_value),
            operation="*",
            left_operand=latch_signal,  # Latch output (0 or 1)
            left_operand_wires=left_operand_wires,  # Read from GREEN only
            right_operand=right_operand,  # Scale factor (constant or signal)
            right_operand_wires=right_operand_wires,  # Read from RED only (for signals)
            output_signal=output_signal,  # Memory's signal type
        )

        # Store on module so handle_read can find it
        module.multiplier_combinator = multiplier_placement

        # Connect latch output to multiplier input via GREEN wire ONLY
        # This uses the same wire as the latch feedback, avoiding double signals
        # We do NOT add this to the signal graph to avoid auto-wiring creating a red wire
        latch_to_multiplier = WireConnection(
            source_entity_id=latch_id,
            sink_entity_id=multiplier_id,
            signal_name=latch_signal,
            wire_color="green",  # Use green (same as feedback) to avoid double-counting
            source_side="output",
            sink_side="input",
        )
        self.layout_plan.add_wire_connection(latch_to_multiplier)

        # Note: We intentionally do NOT add signal graph edges here
        # because we already have the explicit wire connection above.
        # Adding edges would cause the wire router to create an additional red wire.

        return multiplier_placement

    def _make_multiplier_debug_info(
        self, op: IRLatchWrite, value: int | SignalRef
    ) -> dict[str, Any]:
        """Build debug info dict for latch multiplier combinator."""
        if isinstance(value, SignalRef):
            value_str = f"×{value.signal_type}"
        else:
            value_str = f"×{value}"

        debug_info = {
            "variable": f"mem:{op.memory_id}",
            "operation": "latch_multiplier",
            "details": value_str,
            "role": "latch_multiplier",
        }

        if hasattr(op, "source_ast") and op.source_ast:
            if hasattr(op.source_ast, "line"):
                debug_info["line"] = op.source_ast.line
            if hasattr(op.source_ast, "source_file"):
                debug_info["source_file"] = op.source_ast.source_file

        return debug_info

    def _create_signal_remapper(
        self,
        remapper_id: str,
        op: IRLatchWrite,
        input_signal: str,
        output_signal: str,
        signal_graph: SignalGraph,
    ) -> EntityPlacement:
        """Create arithmetic combinator to remap a signal to a different type.

        Used for:
        - Remapping set signal to memory signal type
        - Remapping reset signal to internal type when it conflicts with set

        The remapper simply copies the value: input × 1 → output
        """
        remapper_placement = self.layout_plan.create_and_add_placement(
            ir_node_id=remapper_id,
            entity_type="arithmetic-combinator",
            position=None,
            footprint=(1, 2),
            role="signal_remapper",
            debug_info={
                "variable": f"mem:{op.memory_id}",
                "operation": "signal_remap",
                "details": f"{input_signal}→{output_signal}",
                "role": "signal_remapper",
            },
            operation="*",
            left_operand=input_signal,
            right_operand=1,  # Multiply by 1 = passthrough
            output_signal=output_signal,
        )

        return remapper_placement

    def _create_rs_latch_placement(
        self,
        latch_id: str,
        op: IRLatchWrite,
        set_signal_name: str,
        reset_signal_name: str,
        output_signal: str,
        output_constant: int,
    ) -> EntityPlacement:
        """Create RS latch (reset priority): single condition S > R.

        RS Latch Logic (Reset Priority):
        - SET: When S > R, latch turns ON
        - HOLD: When feedback S > R (with S=1 from feedback), stays ON
        - RESET: When R >= S, latch turns OFF

        The key: set signal is now cast to memory signal type, so:
        - output_signal = set_signal_name = memory signal type
        - Feedback adds to S, so when latched ON: S(feedback) + S(external) > R

        Wire Configuration:
        - RED wire: External set (S) and reset (R) signals
        - GREEN wire: Feedback from output to input (loops back S=1)
        """
        return self.layout_plan.create_and_add_placement(
            ir_node_id=latch_id,
            entity_type="decider-combinator",
            position=None,
            footprint=(1, 2),
            role="latch",
            debug_info=self._make_latch_debug_info(op),
            # Single condition mode - classic RS latch
            operation=">",
            left_operand=set_signal_name,
            right_operand=reset_signal_name,
            output_signal=output_signal,
            copy_count_from_input=False,
            output_value=output_constant,
        )

    def _create_sr_latch_placement(
        self,
        latch_id: str,
        op: IRLatchWrite,
        set_signal_name: str,
        reset_signal_name: str,
        output_signal: str,
        output_constant: int,
    ) -> EntityPlacement:
        """Create SR latch (set priority): multi-condition with wire filtering.

        SR Latch Logic (Set Priority):
        - SET: When external S > 0, latch turns ON (regardless of R)
        - HOLD: When feedback L > 0 AND external R = 0, latch stays ON
        - RESET: When external R > 0 AND external S = 0, latch turns OFF

        Note: The feedback signal is the latch OUTPUT (L), not the set signal (S).

        Factorio 2.0 multi-condition evaluates LEFT-TO-RIGHT without operator precedence.
        So we order conditions to get: (L > 0 AND R = 0) OR S > 0

        Conditions (in this specific order):
            Row 1: L > 0 (read from GREEN wire - feedback) [first]
            Row 2: R = 0 (read from RED wire - external) [AND]
            Row 3: S > 0 (read from RED wire - external) [OR]

        This evaluates as: ((L > 0) AND (R = 0)) OR (S > 0)

        Output: L = 1 (on the output_signal type)
        """
        # Build multi-condition configuration for Factorio 2.0
        # IMPORTANT: Order matters! Factorio evaluates left-to-right.
        # We put the AND conditions first, then OR the SET condition.
        conditions = [
            {
                # Row 1: L > 0 (feedback from latch output)
                "comparator": ">",
                # First condition doesn't need compare_type
                "first_signal": output_signal,  # L - the latch OUTPUT signal
                "first_signal_wires": {"green"},  # Read L from GREEN (feedback)
                "second_constant": 0,
            },
            {
                # Row 2: AND R = 0 (reset not active)
                "comparator": "=",
                "compare_type": "and",  # AND with previous: (L > 0) AND (R = 0)
                "first_signal": reset_signal_name,
                "first_signal_wires": {"red"},  # Read R from RED (external input)
                "second_constant": 0,
            },
            {
                # Row 3: OR S > 0 (set signal active)
                "comparator": ">",
                "compare_type": "or",  # OR with previous: ((L > 0) AND (R = 0)) OR (S > 0)
                "first_signal": set_signal_name,
                "first_signal_wires": {"red"},  # Read S from RED (external input)
                "second_constant": 0,
            },
        ]

        return self.layout_plan.create_and_add_placement(
            ir_node_id=latch_id,
            entity_type="decider-combinator",
            position=None,
            footprint=(1, 2),
            role="latch",
            debug_info=self._make_latch_debug_info(op),
            # Multi-condition mode
            conditions=conditions,
            output_signal=output_signal,
            copy_count_from_input=False,
            output_value=output_constant,
        )

    def _make_latch_debug_info(self, op: IRLatchWrite) -> dict[str, Any]:
        """Build debug info dict for latch combinator."""
        latch_type = "SR" if op.latch_type == MEMORY_TYPE_SR_LATCH else "RS"
        debug_info = {
            "variable": f"mem:{op.memory_id}",
            "operation": "latch",
            "details": f"{latch_type}_latch",
            "role": "latch",
        }

        if hasattr(op, "source_ast") and op.source_ast:
            if hasattr(op.source_ast, "line"):
                debug_info["line"] = op.source_ast.line
            if hasattr(op.source_ast, "source_file"):
                debug_info["source_file"] = op.source_ast.source_file

        return debug_info

    def cleanup_unused_gates(self, layout_plan: LayoutPlan, signal_graph: SignalGraph):
        """Remove gates that were optimized away."""
        to_remove = []

        for _memory_id, module in self._modules.items():
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
            if conn.source_entity_id not in to_remove and conn.sink_entity_id not in to_remove
        ]
        layout_plan.wire_connections = remaining

        for signal_id in list(signal_graph._sinks.keys()):
            sinks = signal_graph._sinks[signal_id]
            for removed_id in to_remove:
                if removed_id in sinks:
                    sinks.remove(removed_id)

    def _is_always_write(self, op: IRMemWrite) -> bool:
        """Check if write enable is constant 1."""
        if isinstance(op.write_enable, int) and op.write_enable == 1:
            return True
        if isinstance(op.write_enable, SignalRef):
            const_ir = self._ir_nodes.get(op.write_enable.source_id)
            if isinstance(const_ir, IRConst) and const_ir.value == 1:
                return True
        return False

    def _can_use_arithmetic_feedback(self, op: IRMemWrite, module: MemoryModule) -> bool:
        """Detect if memory can use arithmetic self-feedback.

        Returns True if the write data comes from an arithmetic operation
        that (directly or indirectly) depends on reading from this same memory.
        """
        if not isinstance(op.data_signal, SignalRef):
            return False

        final_node_id = op.data_signal.source_id
        arith_node = self._ir_nodes.get(final_node_id)
        if not isinstance(arith_node, IRArith):
            return False

        return self._operation_depends_on_memory(final_node_id, op.memory_id)

    def _operation_depends_on_memory(
        self, op_id: str, memory_id: str, visited: set | None = None
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
        if isinstance(ir_node, IRArith):
            if isinstance(ir_node.left, SignalRef) and self._operation_depends_on_memory(
                ir_node.left.source_id, memory_id, visited
            ):
                return True
            if isinstance(ir_node.right, SignalRef) and self._operation_depends_on_memory(
                ir_node.right.source_id, memory_id, visited
            ):
                return True

        return False

    def _optimize_to_arithmetic_feedback(
        self, op: IRMemWrite, module: MemoryModule, signal_graph: SignalGraph
    ):
        """Convert to arithmetic combinator feedback optimization.

        For single-operation chains: Use self-feedback on one combinator
        For multi-operation chains: Wire a feedback loop between combinators
        """
        arith_node_id = op.data_signal.source_id if isinstance(op.data_signal, SignalRef) else None

        if not arith_node_id:
            return

        final_placement = self.layout_plan.get_placement(arith_node_id)
        if not final_placement:
            return

        signal_name = self.signal_analyzer.get_signal_name(module.signal_type)

        first_consumer_id = self._find_first_memory_consumer(op.memory_id)
        is_single_operation = first_consumer_id == arith_node_id or first_consumer_id is None

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
            self.diagnostics.info(f"Clearing old sources for {op.memory_id}: {old_sources}")
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

    def _find_first_memory_consumer(self, memory_id: str) -> str | None:
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
                if not isinstance(node, IRArith):
                    continue

                left_uses = isinstance(node.left, SignalRef) and node.left.source_id == read_node_id
                right_uses = (
                    isinstance(node.right, SignalRef) and node.right.source_id == read_node_id
                )

                if left_uses or right_uses:
                    return node_id

        return None

    def _setup_standard_write(
        self, op: IRMemWrite, module: MemoryModule, signal_graph: SignalGraph
    ):
        """Set up standard write-gated latch.

        CRITICAL: Both gates must receive ALL input signals for proper latch behavior:
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
            signal_graph.add_sink(op.data_signal.source_id, module.write_gate.ir_node_id)
            self.diagnostics.info(
                f"Connected data signal {op.data_signal.source_id} → write_gate {module.write_gate.ir_node_id}"
            )

        # ===================================================================
        # STEP 2: Connect write enable (signal-W) to BOTH gates
        # ===================================================================
        if isinstance(op.write_enable, SignalRef):
            signal_graph.add_sink(op.write_enable.source_id, module.write_gate.ir_node_id)
            self.diagnostics.info(
                f"Connected write_enable {op.write_enable.source_id} → write_gate {module.write_gate.ir_node_id}"
            )
            # Connect to hold gate (KEY FIX #2)
            signal_graph.add_sink(op.write_enable.source_id, module.hold_gate.ir_node_id)
            self.diagnostics.info(
                f"Connected write_enable {op.write_enable.source_id} → hold_gate {module.hold_gate.ir_node_id}"
            )

        # ===================================================================
        # STEP 3: Set up unidirectional forward feedback + self-loop
        # ===================================================================
        # Write gate output → hold gate input (forward feedback, RED wire)
        # Hold gate output → hold gate input (self-loop, RED wire)
        # This topology prevents write_gate from seeing feedback (no accumulation)

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
            f"Set up write-gated latch feedback loop for memory '{op.memory_id}': "
            f"added internal edges to signal_graph for layout, "
            f"created direct RED wire connections for actual signal '{module.signal_type}'"
        )

    def _make_debug_info(self, op, role) -> dict[str, Any]:
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
