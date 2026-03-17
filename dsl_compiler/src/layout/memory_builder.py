"""Memory module construction for circuit-based memory cells.

Every memory pattern uses exactly ONE combinator for its feedback loop.
All other logic (gating, multiplying) is done outside the feedback loop.

Archetypes:
  A: Accumulator — arithmetic combinator with self-feedback
  A': Pass-through — arithmetic combinator, no feedback (1-tick delay)
  B: Gated memory — gate + storage deciders, storage has 1-combinator feedback
  C: Set/Reset latch — single multi-condition decider with self-feedback
"""

from dataclasses import dataclass, field
from typing import Any

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.ir.builder import (
    BundleRef,
    IRArith,
    IRConst,
    IRMemCreate,
    IRMemRead,
    IRMemWrite,
    IRNode,
    SignalRef,
)
from dsl_compiler.src.ir.nodes import (
    MEMORY_TYPE_SR_LATCH,
    IRLatchWrite,
    IRResetWrite,
)

from .layout_plan import EntityPlacement, LayoutPlan, WireConnection
from .signal_analyzer import SignalAnalyzer
from .signal_graph import SignalGraph
from .tile_grid import TileGrid


@dataclass
class MemoryModule:
    """Physical implementation of a memory cell.

    Archetype is "pending" until a write determines the circuit topology.
    """

    memory_id: str
    signal_type: str

    archetype: str = "pending"
    # "accumulator", "pass_through", "gated", "latch", "pending"

    # Primary combinator (always set after archetype is determined):
    #   accumulator → the arithmetic combinator
    #   pass_through → the arithmetic combinator
    #   gated → the storage decider
    #   latch → the latch decider
    primary: EntityPlacement | None = None

    # Secondary combinator (only for some archetypes):
    #   gated → the gate decider
    #   latch → the multiplier (if value ≠ 1)
    secondary: EntityPlacement | None = None

    # Entity ID that external reads connect to
    read_source_id: str | None = None

    # Latch-specific
    latch_type: str | None = None

    # Track feedback signal IDs (for connection_planner filtering)
    _feedback_signal_ids: list[str] = field(default_factory=list)


class MemoryBuilder:
    """Builds memory circuits from IR operations.

    Dispatch:
      IRMemCreate  → register module (no placements — archetype unknown)
      IRMemRead    → record reader, resolve immediately if possible
      IRMemWrite   → determine archetype, create circuit
      IRLatchWrite → create latch circuit (Archetype C)
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
        self._read_sources: dict[str, str] = {}  # read_node_id → memory_id
        self._ir_nodes: dict[str, IRNode] = {}

    # ------------------------------------------------------------------
    # Public interface (called by EntityPlacer)
    # ------------------------------------------------------------------

    def register_ir_node(self, node: IRNode) -> None:
        """Track IR node for optimization detection."""
        self._ir_nodes[node.node_id] = node

    def create_memory(self, op: IRMemCreate, signal_graph: SignalGraph) -> MemoryModule:
        """Register a memory module. No placements yet — archetype is unknown
        until we see the write."""
        signal_name = self.signal_analyzer.get_signal_name(op.signal_type)
        module = MemoryModule(memory_id=op.memory_id, signal_type=signal_name)
        self._modules[op.memory_id] = module
        return module

    def handle_read(self, op: IRMemRead, signal_graph: SignalGraph) -> None:
        """Record a memory read. Resolve immediately if the write already happened."""
        module = self._modules.get(op.memory_id)
        if not module:
            self.diagnostics.warning(
                f"Read from undefined memory '{op.memory_id}' — this may indicate a logic error"
            )
            return

        self._read_sources[op.node_id] = op.memory_id

        # Immediate resolution when the write has already been processed
        if module.read_source_id:
            signal_graph.set_source(op.node_id, module.read_source_id)

    def handle_write(self, op: IRMemWrite, signal_graph: SignalGraph) -> None:
        """Determine archetype and create the memory circuit."""
        module = self._modules.get(op.memory_id)
        if not module:
            self.diagnostics.warning(
                f"Write to undefined memory '{op.memory_id}' — this may indicate a logic error"
            )
            return

        is_always = self._is_always_write(op)

        if is_always and self._can_use_arithmetic_feedback(op, module):
            self._create_accumulator(op, module, signal_graph)
        elif is_always:
            self._create_pass_through(op, module, signal_graph)
        else:
            self._create_gated_memory(op, module, signal_graph)

        # Resolve any reads that were recorded before this write
        self._resolve_deferred_reads(module, signal_graph)

    def handle_latch_write(self, op: IRLatchWrite, signal_graph: SignalGraph) -> None:
        """Create Archetype C: Set/Reset latch."""
        module = self._modules.get(op.memory_id)
        if not module:
            self.diagnostics.warning(f"Latch write for undefined memory '{op.memory_id}'")
            return

        self._create_latch(op, module, signal_graph)

        # Resolve any reads that were recorded before this write
        self._resolve_deferred_reads(module, signal_graph)

    def handle_reset_write(self, op: IRResetWrite, signal_graph: SignalGraph) -> None:
        """Create resettable accumulator: dispatches to 1-decider or arith+gate path."""
        module = self._modules.get(op.memory_id)
        if not module:
            self.diagnostics.warning(f"Reset write for undefined memory '{op.memory_id}'")
            return

        if not isinstance(op.data_signal, SignalRef):
            self.diagnostics.error(
                f"Memory '{op.memory_id}': write() with reset= requires an expression "
                f"value, not a constant.",
                stage="layout",
            )
            return

        has_self_dep = self._operation_depends_on_memory(op.data_signal.source_id, op.memory_id)
        if not has_self_dep:
            self.diagnostics.error(
                f"Memory '{op.memory_id}': write() with reset= requires the value "
                f"expression to depend on reading from the same memory.",
                stage="layout",
            )
            return

        # Determine which path to use
        arith_node_id = op.data_signal.source_id
        arith_node = self._ir_nodes.get(arith_node_id)
        first_consumer_id = self._find_first_memory_consumer(op.memory_id)
        is_single_op = first_consumer_id == arith_node_id or first_consumer_id is None
        is_simple_addition = (
            isinstance(arith_node, IRArith) and arith_node.op == "+" and is_single_op
        )

        if is_simple_addition:
            assert isinstance(arith_node, IRArith)  # guaranteed by is_simple_addition check
            self._create_single_decider_accumulator(op, module, signal_graph, arith_node)
        else:
            self._create_gated_chain_accumulator(op, module, signal_graph)

        self._resolve_deferred_reads(module, signal_graph)

    def finalize(self, layout_plan: LayoutPlan, signal_graph: SignalGraph) -> None:
        """Final pass: warn about never-written memories, resolve stragglers."""
        for module in self._modules.values():
            if module.archetype == "pending":
                self.diagnostics.warning(f"Memory '{module.memory_id}' declared but never written")
                continue
            # Safety: resolve any remaining deferred reads
            for read_id, mem_id in self._read_sources.items():
                if mem_id == module.memory_id and module.read_source_id:
                    signal_graph.set_source(read_id, module.read_source_id)

    # ------------------------------------------------------------------
    # Archetype A: Accumulator (arithmetic self-feedback)
    # ------------------------------------------------------------------

    def _create_accumulator(
        self, op: IRMemWrite, module: MemoryModule, signal_graph: SignalGraph
    ) -> None:
        """Single arithmetic combinator with output→input self-feedback.

        Detects the final arithmetic combinator in the expression chain and
        marks it for self-feedback. For multi-operation chains, registers a
        feedback edge from the last combinator back to the first consumer.
        """
        arith_node_id = op.data_signal.source_id if isinstance(op.data_signal, SignalRef) else None
        if not arith_node_id:
            return

        final_placement = self.layout_plan.get_placement(arith_node_id)
        if not final_placement:
            return

        signal_name = self.signal_analyzer.get_signal_name(module.signal_type)
        first_consumer_id = self._find_first_memory_consumer(op.memory_id)
        is_single_op = first_consumer_id == arith_node_id or first_consumer_id is None

        if is_single_op:
            final_placement.properties["has_self_feedback"] = True
            final_placement.properties["feedback_signal"] = signal_name
            if "debug_info" in final_placement.properties:
                final_placement.properties["debug_info"]["memory_name"] = op.memory_id
                old_details = final_placement.properties["debug_info"].get("details", "arith")
                final_placement.properties["debug_info"]["details"] = (
                    f"{old_details} + memory:{op.memory_id}"
                )
        else:
            self.diagnostics.info(
                f"Optimized memory '{op.memory_id}' to multi-combinator feedback loop"
            )

        module.archetype = "accumulator"
        module.primary = final_placement
        module.read_source_id = arith_node_id

        # Update signal graph: memory reads point to the arithmetic combinator
        signal_graph.set_source(op.memory_id, arith_node_id)
        for read_id, mem_id in self._read_sources.items():
            if mem_id == op.memory_id:
                signal_graph.set_source(read_id, arith_node_id)

        # Multi-operation chain: register feedback edge
        if first_consumer_id and first_consumer_id != arith_node_id:
            signal_graph.set_source(arith_node_id, arith_node_id)
            signal_graph.add_sink(arith_node_id, first_consumer_id)

        self.diagnostics.info(
            f"Optimized memory '{op.memory_id}' to "
            f"{'single' if is_single_op else 'multi'}-combinator arithmetic feedback"
        )

    # ------------------------------------------------------------------
    # Archetype A': Pass-through (1-tick delay, no feedback)
    # ------------------------------------------------------------------

    def _create_pass_through(
        self, op: IRMemWrite, module: MemoryModule, signal_graph: SignalGraph
    ) -> None:
        """Single arithmetic combinator: signal + 0 → signal.

        Used when every tick writes unconditionally and the value does NOT
        depend on reading from the same memory.
        """
        signal_name = self.signal_analyzer.get_signal_name(module.signal_type)
        pt_id = f"{op.memory_id}_pass_through"

        placement = self.layout_plan.create_and_add_placement(
            ir_node_id=pt_id,
            entity_type="arithmetic-combinator",
            position=None,
            footprint=(1, 2),
            role="memory_pass_through",
            debug_info=self._debug_info(op, "pass_through"),
            operation="+",
            left_operand=signal_name,
            right_operand=0,
            output_signal=signal_name,
        )

        module.archetype = "pass_through"
        module.primary = placement
        module.read_source_id = pt_id

        if isinstance(op.data_signal, SignalRef):
            signal_graph.add_sink(op.data_signal.source_id, pt_id)

        signal_graph.set_source(op.memory_id, pt_id)

        self.diagnostics.info(f"Optimized memory '{op.memory_id}' to pass-through (1-tick delay)")

    # ------------------------------------------------------------------
    # Archetype B: Gated memory (gate + storage, 1-combinator feedback)
    # ------------------------------------------------------------------

    def _create_gated_memory(
        self, op: IRMemWrite, module: MemoryModule, signal_graph: SignalGraph
    ) -> None:
        """Two deciders: gate passes data during write, storage holds via self-loop.

        Gate:    signal-W > 0 → copy data (input count)
        Storage: signal-W = 0 → copy data (input count), self-feedback on RED

        The storage decider's feedback loop is exactly 1 combinator deep.
        The gate is a sidecar that injects data — it is NOT in the feedback loop.
        """
        signal_name = self.signal_analyzer.get_signal_name(module.signal_type)
        gate_id = f"{op.memory_id}_gate"
        storage_id = f"{op.memory_id}_storage"

        gate_placement = self.layout_plan.create_and_add_placement(
            ir_node_id=gate_id,
            entity_type="decider-combinator",
            position=None,
            footprint=(1, 2),
            role="memory_gate",
            debug_info=self._debug_info(op, "gate"),
            operation=">",
            left_operand="signal-W",
            right_operand=0,
            output_signal=signal_name,
            copy_count_from_input=True,
        )

        storage_placement = self.layout_plan.create_and_add_placement(
            ir_node_id=storage_id,
            entity_type="decider-combinator",
            position=None,
            footprint=(1, 2),
            role="memory_storage",
            debug_info=self._debug_info(op, "storage"),
            operation="=",
            left_operand="signal-W",
            right_operand=0,
            output_signal=signal_name,
            copy_count_from_input=True,
        )

        module.archetype = "gated"
        module.primary = storage_placement
        module.secondary = gate_placement
        module.read_source_id = storage_id

        # --- Wiring ---

        # 1. Data source → gate only (via signal graph → will become RED)
        if isinstance(op.data_signal, SignalRef):
            signal_graph.add_sink(op.data_signal.source_id, gate_id)

        # 2. signal-W → both gate and storage (via signal graph → constrained to GREEN)
        if isinstance(op.write_enable, SignalRef):
            signal_graph.add_sink(op.write_enable.source_id, gate_id)
            signal_graph.add_sink(op.write_enable.source_id, storage_id)

        # 3. Gate output → storage input (explicit RED wire)
        self.layout_plan.add_wire_connection(
            WireConnection(
                source_entity_id=gate_id,
                sink_entity_id=storage_id,
                signal_name=signal_name,
                wire_color="red",
                source_side="output",
                sink_side="input",
            )
        )

        # 4. Storage self-feedback (explicit RED wire)
        self.layout_plan.add_wire_connection(
            WireConnection(
                source_entity_id=storage_id,
                sink_entity_id=storage_id,
                signal_name=signal_name,
                wire_color="red",
                source_side="output",
                sink_side="input",
            )
        )

        # 5. Internal signal-graph edge for layout proximity
        feedback_edge = f"__feedback_{op.memory_id}_g2s"
        signal_graph.set_source(feedback_edge, gate_id)
        signal_graph.add_sink(feedback_edge, storage_id)
        module._feedback_signal_ids = [feedback_edge]

        # Memory reads come from storage
        signal_graph.set_source(op.memory_id, storage_id)

        self.diagnostics.info(
            f"Created gated memory '{op.memory_id}': gate + storage (1-combinator feedback)"
        )

    # ------------------------------------------------------------------
    # Archetype C: Set/Reset latch
    # ------------------------------------------------------------------

    def _create_latch(
        self, op: IRLatchWrite, module: MemoryModule, signal_graph: SignalGraph
    ) -> None:
        """Single multi-condition decider with GREEN self-feedback.

        Dispatches to inlined or standard path based on whether conditions
        can be folded into the decider. Respects SR/RS priority in both paths.

        When set and reset signals share the same Factorio signal name, renaming
        combinators are inserted to cast them to unique signal names (signal-S
        for set, signal-R for reset). This is necessary because the latch
        conditions check set and reset on the same wire (RED), and summing
        identical signals makes them indistinguishable.
        """
        latch_id = f"{op.memory_id}_latch"

        set_renamer_id: str | None = None
        reset_renamer_id: str | None = None

        if op.has_inline_conditions:
            placement = self._create_inlined_latch(op, module, latch_id)
        else:
            # Determine actual signal names for set and reset
            set_signal = (
                self.signal_analyzer.get_signal_name(op.set_signal.signal_type)
                if isinstance(op.set_signal, SignalRef)
                else "signal-S"
            )
            reset_signal = (
                self.signal_analyzer.get_signal_name(op.reset_signal.signal_type)
                if isinstance(op.reset_signal, SignalRef)
                else "signal-R"
            )

            # Check if renaming is needed:
            # When set and reset use the same Factorio signal, they sum on
            # the same wire and the latch conditions can't distinguish them.
            # The connection planner's hard constraint (Fix 2) already ensures
            # all latch inputs arrive on RED, so set_signal == module.signal_type
            # is fine — per-wire filtering separates RED (inputs) from GREEN
            # (self-loop).  Only set_signal == reset_signal truly requires
            # renaming to unique signal names.
            needs_renaming = set_signal == reset_signal

            if needs_renaming:
                effective_set = "signal-S"
                effective_reset = "signal-R"

                # Create renaming combinators: original_signal + 0 → unique_signal
                if isinstance(op.set_signal, SignalRef):
                    set_renamer_id = self._create_latch_signal_caster(
                        op,
                        module,
                        latch_id,
                        set_signal,
                        effective_set,
                        "set",
                        signal_graph,
                    )
                if isinstance(op.reset_signal, SignalRef):
                    reset_renamer_id = self._create_latch_signal_caster(
                        op,
                        module,
                        latch_id,
                        reset_signal,
                        effective_reset,
                        "reset",
                        signal_graph,
                    )
            else:
                effective_set = set_signal
                effective_reset = reset_signal

            if op.latch_type == MEMORY_TYPE_SR_LATCH:
                placement = self._create_sr_latch(
                    op,
                    module,
                    latch_id,
                    set_signal_override=effective_set,
                    reset_signal_override=effective_reset,
                )
            else:
                placement = self._create_rs_latch(
                    op,
                    module,
                    latch_id,
                    set_signal_override=effective_set,
                    reset_signal_override=effective_reset,
                )

        module.archetype = "latch"
        module.latch_type = op.latch_type
        module.primary = placement
        module.read_source_id = latch_id

        # GREEN wire self-feedback
        self.layout_plan.add_wire_connection(
            WireConnection(
                source_entity_id=latch_id,
                sink_entity_id=latch_id,
                signal_name=module.signal_type,
                wire_color="green",
                source_side="output",
                sink_side="input",
            )
        )

        # Internal feedback signal for connection_planner filtering
        feedback_signal = f"__feedback_{module.memory_id}_latch"
        signal_graph.set_source(feedback_signal, latch_id)
        signal_graph.add_sink(feedback_signal, latch_id)
        module._feedback_signal_ids = [feedback_signal]

        # Connect external inputs (through renamers if they exist)
        self._connect_latch_inputs(
            op,
            latch_id,
            signal_graph,
            set_renamer_id=set_renamer_id,
            reset_renamer_id=reset_renamer_id,
        )

        # Multiplier for values ≠ 1
        if self._needs_multiplier(op):
            mult_id = self._create_multiplier(op, module, latch_id, signal_graph)
            module.secondary = self.layout_plan.get_placement(mult_id)
            module.read_source_id = mult_id

        signal_graph.set_source(op.memory_id, module.read_source_id)

        priority = (
            "SR (set priority)" if op.latch_type == MEMORY_TYPE_SR_LATCH else "RS (reset priority)"
        )
        self.diagnostics.info(f"Created {priority} latch '{op.memory_id}'")

    def _create_sr_latch(
        self,
        op: IRLatchWrite,
        module: MemoryModule,
        latch_id: str,
        *,
        set_signal_override: str | None = None,
        reset_signal_override: str | None = None,
    ) -> EntityPlacement:
        """SR latch (set priority): ((L > 0) AND (R = 0)) OR (S > 0).

        When both set and reset are active, set wins.
        """
        if set_signal_override:
            set_signal = set_signal_override
        elif isinstance(op.set_signal, SignalRef):
            set_signal = self.signal_analyzer.get_signal_name(op.set_signal.signal_type)
        else:
            set_signal = "signal-S"

        if reset_signal_override:
            reset_signal = reset_signal_override
        elif isinstance(op.reset_signal, SignalRef):
            reset_signal = self.signal_analyzer.get_signal_name(op.reset_signal.signal_type)
        else:
            reset_signal = "signal-R"

        conditions = [
            {
                "comparator": ">",
                "first_signal": module.signal_type,
                "first_signal_wires": {"green"},
                "second_constant": 0,
            },
            {
                "comparator": "=",
                "compare_type": "and",
                "first_signal": reset_signal,
                "first_signal_wires": {"red"},
                "second_constant": 0,
            },
            {
                "comparator": ">",
                "compare_type": "or",
                "first_signal": set_signal,
                "first_signal_wires": {"red"},
                "second_constant": 0,
            },
        ]

        return self.layout_plan.create_and_add_placement(
            ir_node_id=latch_id,
            entity_type="decider-combinator",
            position=None,
            footprint=(1, 2),
            role="latch",
            debug_info=self._latch_debug_info(op),
            conditions=conditions,
            output_signal=module.signal_type,
            copy_count_from_input=False,
            output_value=1,
        )

    def _create_rs_latch(
        self,
        op: IRLatchWrite,
        module: MemoryModule,
        latch_id: str,
        *,
        set_signal_override: str | None = None,
        reset_signal_override: str | None = None,
    ) -> EntityPlacement:
        """RS latch (reset priority): ((S > 0) OR (L > 0)) AND (R = 0).

        When both set and reset are active, reset wins.
        """
        if set_signal_override:
            set_signal = set_signal_override
        elif isinstance(op.set_signal, SignalRef):
            set_signal = self.signal_analyzer.get_signal_name(op.set_signal.signal_type)
        else:
            set_signal = "signal-S"

        if reset_signal_override:
            reset_signal = reset_signal_override
        elif isinstance(op.reset_signal, SignalRef):
            reset_signal = self.signal_analyzer.get_signal_name(op.reset_signal.signal_type)
        else:
            reset_signal = "signal-R"

        conditions = [
            {
                "comparator": ">",
                "first_signal": set_signal,
                "first_signal_wires": {"red"},
                "second_constant": 0,
            },
            {
                "comparator": ">",
                "compare_type": "or",
                "first_signal": module.signal_type,
                "first_signal_wires": {"green"},
                "second_constant": 0,
            },
            {
                "comparator": "=",
                "compare_type": "and",
                "first_signal": reset_signal,
                "first_signal_wires": {"red"},
                "second_constant": 0,
            },
        ]

        return self.layout_plan.create_and_add_placement(
            ir_node_id=latch_id,
            entity_type="decider-combinator",
            position=None,
            footprint=(1, 2),
            role="latch",
            debug_info=self._latch_debug_info(op),
            conditions=conditions,
            output_signal=module.signal_type,
            copy_count_from_input=False,
            output_value=1,
        )

    def _create_inlined_latch(
        self, op: IRLatchWrite, module: MemoryModule, latch_id: str
    ) -> EntityPlacement:
        """Latch with inlined set/reset comparisons on the same input signal.

        Condition ordering respects latch_type:
          RS: ((SET_COND) OR (L > 0)) AND (HOLD_COND)  — reset wins on overlap
          SR: ((L > 0) AND (HOLD_COND)) OR (SET_COND)  — set wins on overlap
        """
        assert op.set_condition is not None and op.reset_condition is not None

        set_signal_ref, set_op, set_const = op.set_condition
        reset_signal_ref, reset_op, reset_const = op.reset_condition

        assert isinstance(set_signal_ref, SignalRef)
        input_signal = self.signal_analyzer.get_signal_name(set_signal_ref.signal_type)
        hold_op, hold_const = self._invert_comparison(reset_op, reset_const)

        if op.latch_type == MEMORY_TYPE_SR_LATCH:
            # SR: ((L > 0) AND (HOLD_COND)) OR (SET_COND)
            conditions = [
                {
                    "comparator": ">",
                    "first_signal": module.signal_type,
                    "first_signal_wires": {"green"},
                    "second_constant": 0,
                },
                {
                    "comparator": hold_op,
                    "compare_type": "and",
                    "first_signal": input_signal,
                    "first_signal_wires": {"red"},
                    "second_constant": hold_const,
                },
                {
                    "comparator": set_op,
                    "compare_type": "or",
                    "first_signal": input_signal,
                    "first_signal_wires": {"red"},
                    "second_constant": set_const,
                },
            ]
        else:
            # RS: ((SET_COND) OR (L > 0)) AND (HOLD_COND)
            conditions = [
                {
                    "comparator": set_op,
                    "first_signal": input_signal,
                    "first_signal_wires": {"red"},
                    "second_constant": set_const,
                },
                {
                    "comparator": ">",
                    "compare_type": "or",
                    "first_signal": module.signal_type,
                    "first_signal_wires": {"green"},
                    "second_constant": 0,
                },
                {
                    "comparator": hold_op,
                    "compare_type": "and",
                    "first_signal": input_signal,
                    "first_signal_wires": {"red"},
                    "second_constant": hold_const,
                },
            ]

        return self.layout_plan.create_and_add_placement(
            ir_node_id=latch_id,
            entity_type="decider-combinator",
            position=None,
            footprint=(1, 2),
            role="latch",
            debug_info=self._latch_debug_info(op),
            conditions=conditions,
            output_signal=module.signal_type,
            copy_count_from_input=False,
            output_value=1,
        )

    def _create_latch_signal_caster(
        self,
        op: IRLatchWrite,
        module: MemoryModule,
        latch_id: str,
        input_signal: str,
        output_signal: str,
        role_suffix: str,
        signal_graph: SignalGraph,
    ) -> str:
        """Create an arithmetic combinator to rename a signal for latch input.

        Produces: input_signal + 0 → output_signal
        Connects the caster's output to the latch input via an explicit RED wire.

        Returns the entity ID of the caster combinator.
        """
        caster_id = f"{op.memory_id}_latch_{role_suffix}_caster"

        self.layout_plan.create_and_add_placement(
            ir_node_id=caster_id,
            entity_type="arithmetic-combinator",
            position=None,
            footprint=(1, 2),
            role="latch_signal_caster",
            debug_info={
                "variable": f"mem:{op.memory_id}",
                "operation": "latch_signal_cast",
                "details": f"cast {input_signal} → {output_signal} for latch {role_suffix}",
                "role": "latch_signal_caster",
            },
            operation="+",
            left_operand=input_signal,
            right_operand=0,
            output_signal=output_signal,
        )

        # Explicit RED wire from caster output to latch input
        self.layout_plan.add_wire_connection(
            WireConnection(
                source_entity_id=caster_id,
                sink_entity_id=latch_id,
                signal_name=output_signal,
                wire_color="red",
                source_side="output",
                sink_side="input",
            )
        )

        # Internal feedback signal so connection_planner skips this edge
        feedback_signal = f"__feedback_{module.memory_id}_{role_suffix}_caster"
        signal_graph.set_source(feedback_signal, caster_id)
        signal_graph.add_sink(feedback_signal, latch_id)
        module._feedback_signal_ids.append(feedback_signal)

        self.diagnostics.info(
            f"Created latch signal caster '{caster_id}': "
            f"{input_signal} → {output_signal} for {role_suffix}"
        )

        return caster_id

    def _connect_latch_inputs(
        self,
        op: IRLatchWrite,
        latch_id: str,
        signal_graph: SignalGraph,
        *,
        set_renamer_id: str | None = None,
        reset_renamer_id: str | None = None,
    ) -> None:
        """Wire external set/reset/condition signals to the latch input.

        When renaming combinators exist, signals are routed through them
        instead of directly to the latch. The renamers are connected to the
        latch via explicit RED wire connections (created in _create_latch_signal_caster).
        """
        if op.has_inline_conditions:
            assert op.set_condition is not None
            signal_ref = op.set_condition[0]
            if isinstance(signal_ref, SignalRef) and signal_ref.source_id:
                signal_graph.add_sink(signal_ref.source_id, latch_id)
        else:
            if isinstance(op.set_signal, SignalRef) and op.set_signal.source_id:
                target = set_renamer_id if set_renamer_id else latch_id
                signal_graph.add_sink(op.set_signal.source_id, target)
            if isinstance(op.reset_signal, SignalRef) and op.reset_signal.source_id:
                target = reset_renamer_id if reset_renamer_id else latch_id
                signal_graph.add_sink(op.reset_signal.source_id, target)

    @staticmethod
    def _needs_multiplier(op: IRLatchWrite) -> bool:
        """Check if latch needs a multiplier (value ≠ 1 or value is a signal)."""
        if isinstance(op.value, SignalRef):
            return True
        return isinstance(op.value, int) and op.value != 1

    def _create_multiplier(
        self,
        op: IRLatchWrite,
        module: MemoryModule,
        latch_id: str,
        signal_graph: SignalGraph,
    ) -> str:
        """Arithmetic combinator to scale latch output: latch_signal × value → output.

        Reads latch output from GREEN wire (same as feedback loop).
        Reads value signal from RED wire (if value is a signal).
        """
        mult_id = f"{op.memory_id}_multiplier"
        output_signal = module.signal_type

        left_wires = {"green"}

        if isinstance(op.value, SignalRef):
            right_operand: str | int = self.signal_analyzer.get_signal_name(op.value.signal_type)
            right_wires: set[str] = {"red"}
            if op.value.source_id:
                signal_graph.add_sink(op.value.source_id, mult_id)
        elif isinstance(op.value, int):
            right_operand = op.value
            right_wires = {"red", "green"}
        else:
            right_operand = 0  # BundleRef not supported in latch multiplier
            right_wires = {"red", "green"}

        self.layout_plan.create_and_add_placement(
            ir_node_id=mult_id,
            entity_type="arithmetic-combinator",
            position=None,
            footprint=(1, 2),
            role="latch_multiplier",
            debug_info=self._multiplier_debug_info(op),
            operation="*",
            left_operand=module.signal_type,
            left_operand_wires=left_wires,
            right_operand=right_operand,
            right_operand_wires=right_wires,
            output_signal=output_signal,
        )

        # Latch output → multiplier input via GREEN wire
        self.layout_plan.add_wire_connection(
            WireConnection(
                source_entity_id=latch_id,
                sink_entity_id=mult_id,
                signal_name=module.signal_type,
                wire_color="green",
                source_side="output",
                sink_side="input",
            )
        )

        return mult_id

    # ------------------------------------------------------------------
    # Archetype D: Resettable Accumulator
    # ------------------------------------------------------------------

    def _create_single_decider_accumulator(
        self,
        op: IRResetWrite,
        module: MemoryModule,
        signal_graph: SignalGraph,
        arith_node: IRArith,
    ) -> None:
        """Path 1: Replace the arith with a single decider for mem.read() + X.

        The decider accumulates via wire merging: RED(self-feedback) + GREEN(pulse)
        are summed by Factorio when they share the same signal type. Condition R=0
        gates the output for reset.

        If the pulse signal type doesn't match memory's type and we can't re-project
        it, we fall back to the multi-combinator path.
        """
        arith_node_id = arith_node.node_id
        signal_name = self.signal_analyzer.get_signal_name(module.signal_type)

        # Identify the pulse operand (the non-memory operand of the addition)
        pulse_ref = self._find_pulse_operand(arith_node, op.memory_id)
        if pulse_ref is None:
            self._create_gated_chain_accumulator(op, module, signal_graph)
            return

        # Ensure the pulse uses the memory's signal type (required for wire merging)
        if isinstance(pulse_ref, SignalRef) and pulse_ref.signal_type != module.signal_type:
            pulse_source = self._ir_nodes.get(pulse_ref.source_id)
            pulse_sinks = signal_graph.iter_sinks(pulse_ref.source_id)
            pulse_placement = self.layout_plan.get_placement(pulse_ref.source_id)

            if (
                isinstance(pulse_source, IRConst)
                and pulse_placement is not None
                and len(pulse_sinks) == 1
            ):
                # Safe to re-project: the constant only feeds the arith we're removing
                new_signal_name = self.signal_analyzer.get_signal_name(module.signal_type)
                pulse_placement.properties["signal_name"] = new_signal_name
                pulse_placement.properties["signal_type"] = module.signal_type
                if "debug_info" in pulse_placement.properties:
                    pulse_placement.properties["debug_info"]["details"] = (
                        f"re-projected to {new_signal_name} for reset accumulator"
                    )
            else:
                # Can't re-project — fall back to multi-combinator path
                self.diagnostics.info(
                    f"Memory '{op.memory_id}': pulse signal type mismatch "
                    f"('{pulse_ref.signal_type}' vs '{module.signal_type}'), "
                    f"using multi-combinator path"
                )
                self._create_gated_chain_accumulator(op, module, signal_graph)
                return

        # --- Remove the arith combinator ---
        if arith_node_id in self.layout_plan.entity_placements:
            del self.layout_plan.entity_placements[arith_node_id]

        for sink_id in signal_graph.iter_sinks(arith_node_id):
            signal_graph.remove_sink(arith_node_id, sink_id)

        # --- Determine reset signal name ---
        if isinstance(op.reset_signal, SignalRef):
            reset_signal_name = self.signal_analyzer.get_signal_name(op.reset_signal.signal_type)
        else:
            reset_signal_name = "signal-R"

        # --- Create the decider ---
        gate_id = f"{op.memory_id}_reset_decider"

        conditions = [
            {
                "comparator": "=",
                "first_signal": reset_signal_name,
                "first_signal_wires": {"green"},
                "second_constant": 0,
            },
        ]

        self.layout_plan.create_and_add_placement(
            ir_node_id=gate_id,
            entity_type="decider-combinator",
            position=None,
            footprint=(1, 2),
            role="memory_reset_decider",
            debug_info={
                "variable": f"mem:{op.memory_id}",
                "operation": "reset_accumulator",
                "details": "1-decider resettable accumulator",
                "signal_type": signal_name,
                "role": "memory_reset_decider",
                "memory_name": op.memory_id,
            },
            conditions=conditions,
            output_signal=signal_name,
            copy_count_from_input=True,
            has_self_feedback=True,
            feedback_signal=signal_name,
        )

        module.archetype = "accumulator"
        module.primary = self.layout_plan.get_placement(gate_id)
        module.read_source_id = gate_id

        # Wire: pulse source → decider input
        if isinstance(pulse_ref, SignalRef) and pulse_ref.source_id:
            signal_graph.remove_sink(pulse_ref.source_id, arith_node_id)
            signal_graph.add_sink(pulse_ref.source_id, gate_id)

        # Wire: reset signal → decider input
        if isinstance(op.reset_signal, SignalRef) and op.reset_signal.source_id:
            signal_graph.add_sink(op.reset_signal.source_id, gate_id)

        # Memory reads come from the decider output
        signal_graph.set_source(op.memory_id, gate_id)
        signal_graph.set_source(gate_id, gate_id)
        for read_id, mem_id in self._read_sources.items():
            if mem_id == op.memory_id:
                signal_graph.set_source(read_id, gate_id)

        self.diagnostics.info(
            f"Created 1-decider resettable accumulator '{op.memory_id}' "
            f"(simple addition, self-feedback)"
        )

    def _create_gated_chain_accumulator(
        self, op: IRResetWrite, module: MemoryModule, signal_graph: SignalGraph
    ) -> None:
        """Path 2: Keep arith chain, add decider reset gate in feedback path.

        Chain: arith₁ → ... → arithₙ → decider(R=0, copy input) → RED feedback to arith₁
        """
        if not isinstance(op.data_signal, SignalRef):
            return
        arith_node_id = op.data_signal.source_id
        final_placement = self.layout_plan.get_placement(arith_node_id)
        if not final_placement:
            return

        signal_name = self.signal_analyzer.get_signal_name(module.signal_type)
        first_consumer_id = self._find_first_memory_consumer(op.memory_id)

        # Determine reset signal name
        if isinstance(op.reset_signal, SignalRef):
            reset_signal_name = self.signal_analyzer.get_signal_name(op.reset_signal.signal_type)
        else:
            reset_signal_name = "signal-R"

        gate_id = f"{op.memory_id}_reset_gate"

        conditions = [
            {
                "comparator": "=",
                "first_signal": reset_signal_name,
                "second_constant": 0,
            },
        ]

        self.layout_plan.create_and_add_placement(
            ir_node_id=gate_id,
            entity_type="decider-combinator",
            position=None,
            footprint=(1, 2),
            role="memory_reset_gate",
            debug_info={
                "variable": f"mem:{op.memory_id}",
                "operation": "reset_gate",
                "details": "reset gate for multi-op accumulator",
                "signal_type": signal_name,
                "role": "memory_reset_gate",
                "memory_name": op.memory_id,
            },
            conditions=conditions,
            output_signal=signal_name,
            copy_count_from_input=True,
        )

        module.archetype = "accumulator"
        module.primary = self.layout_plan.get_placement(gate_id)
        module.read_source_id = gate_id

        # Wire: arithₙ output → gate input
        signal_graph.add_sink(arith_node_id, gate_id)

        # Wire: gate output → first consumer input (explicit RED feedback)
        target_id = first_consumer_id or arith_node_id
        self.layout_plan.add_wire_connection(
            WireConnection(
                source_entity_id=gate_id,
                sink_entity_id=target_id,
                signal_name=signal_name,
                wire_color="red",
                source_side="output",
                sink_side="input",
            )
        )

        # Internal feedback signal for connection_planner filtering
        feedback_signal = f"__feedback_{module.memory_id}_reset_gate"
        signal_graph.set_source(feedback_signal, gate_id)
        signal_graph.add_sink(feedback_signal, target_id)
        module._feedback_signal_ids = [feedback_signal]

        # Connect reset signal to gate
        if isinstance(op.reset_signal, SignalRef) and op.reset_signal.source_id:
            signal_graph.add_sink(op.reset_signal.source_id, gate_id)

        # Memory reads come from the reset gate output
        signal_graph.set_source(op.memory_id, gate_id)
        for read_id, mem_id in self._read_sources.items():
            if mem_id == op.memory_id:
                signal_graph.set_source(read_id, gate_id)

        self.diagnostics.info(
            f"Created gated resettable accumulator '{op.memory_id}' "
            f"(arith chain + 1 decider reset gate)"
        )

    def _find_pulse_operand(
        self, arith_node: IRArith, memory_id: str
    ) -> SignalRef | BundleRef | int | None:
        """Find the non-memory operand of a simple addition arith.

        For `mem.read() + X`, returns X. For `X + mem.read()`, also returns X.
        Returns None if both operands depend on the memory.
        """
        left_depends = isinstance(arith_node.left, SignalRef) and (
            self._operation_depends_on_memory(arith_node.left.source_id, memory_id)
        )
        right_depends = isinstance(arith_node.right, SignalRef) and (
            self._operation_depends_on_memory(arith_node.right.source_id, memory_id)
        )

        if left_depends and not right_depends:
            return arith_node.right
        elif right_depends and not left_depends:
            return arith_node.left
        else:
            return None

    # ------------------------------------------------------------------
    # Helpers: optimization detection
    # ------------------------------------------------------------------

    def _is_always_write(self, op: IRMemWrite) -> bool:
        """Check if write-enable is constant 1 (unconditional write)."""
        if isinstance(op.write_enable, int) and op.write_enable == 1:
            return True
        if isinstance(op.write_enable, SignalRef):
            const_ir = self._ir_nodes.get(op.write_enable.source_id)
            if isinstance(const_ir, IRConst) and const_ir.value == 1:
                return True
        return False

    def _can_use_arithmetic_feedback(self, op: IRMemWrite, module: MemoryModule) -> bool:
        """True if the write value comes from arithmetic that reads this memory."""
        if not isinstance(op.data_signal, SignalRef):
            return False
        arith_node = self._ir_nodes.get(op.data_signal.source_id)
        if not isinstance(arith_node, IRArith):
            return False
        return self._operation_depends_on_memory(op.data_signal.source_id, op.memory_id)

    def _operation_depends_on_memory(
        self, op_id: str, memory_id: str, visited: set[str] | None = None
    ) -> bool:
        """Check if an operation depends on a memory read (directly or transitively)."""
        if visited is None:
            visited = set()
        if op_id in visited:
            return False
        visited.add(op_id)

        if self._read_sources.get(op_id) == memory_id:
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

    def _find_first_memory_consumer(self, memory_id: str) -> str | None:
        """Find the first arithmetic operation that reads from this memory."""
        for read_node_id, source_memory_id in self._read_sources.items():
            if source_memory_id != memory_id:
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

    # ------------------------------------------------------------------
    # Helpers: deferred read resolution
    # ------------------------------------------------------------------

    def _resolve_deferred_reads(self, module: MemoryModule, signal_graph: SignalGraph) -> None:
        """Wire any reads that were recorded before this module's write."""
        if not module.read_source_id:
            return
        for read_id, mem_id in self._read_sources.items():
            if mem_id == module.memory_id:
                signal_graph.set_source(read_id, module.read_source_id)

    # ------------------------------------------------------------------
    # Helpers: comparison inversion
    # ------------------------------------------------------------------

    @staticmethod
    def _invert_comparison(op: str, const: int) -> tuple[str, int]:
        """Invert a comparison for hold condition: reset → hold.

        E.g. reset = battery >= 80 → hold = battery < 80.
        """
        inversions = {
            "<": ">=",
            "<=": ">",
            ">": "<=",
            ">=": "<",
            "==": "!=",
            "!=": "==",
        }
        return inversions[op], const

    # ------------------------------------------------------------------
    # Helpers: debug info
    # ------------------------------------------------------------------

    def _debug_info(self, op: IRMemWrite | IRMemCreate, role: str) -> dict[str, Any]:
        """Build debug info for memory combinators."""
        signal_type_raw = getattr(op, "signal_type", None)
        if signal_type_raw is None:
            module = self._modules.get(op.memory_id)
            signal_type_raw = module.signal_type if module else "unknown"

        info: dict[str, Any] = {
            "variable": f"mem:{op.memory_id}",
            "operation": "memory",
            "details": role,
            "signal_type": self.signal_analyzer.get_signal_name(signal_type_raw),
            "role": f"memory_{role}",
        }
        if hasattr(op, "source_ast") and op.source_ast:
            if hasattr(op.source_ast, "line"):
                info["line"] = op.source_ast.line
            if hasattr(op.source_ast, "source_file"):
                info["source_file"] = op.source_ast.source_file
        return info

    def _latch_debug_info(self, op: IRLatchWrite) -> dict[str, Any]:
        """Build debug info for latch combinator."""
        latch_type = "SR" if op.latch_type == MEMORY_TYPE_SR_LATCH else "RS"
        info: dict[str, Any] = {
            "variable": f"mem:{op.memory_id}",
            "operation": "latch",
            "details": f"{latch_type}_latch",
            "role": "latch",
        }
        if hasattr(op, "source_ast") and op.source_ast:
            if hasattr(op.source_ast, "line"):
                info["line"] = op.source_ast.line
            if hasattr(op.source_ast, "source_file"):
                info["source_file"] = op.source_ast.source_file
        return info

    def _multiplier_debug_info(self, op: IRLatchWrite) -> dict[str, Any]:
        """Build debug info for latch multiplier."""
        if isinstance(op.value, SignalRef):
            value_str = f"×{op.value.signal_type}"
        else:
            value_str = f"×{op.value}"
        info: dict[str, Any] = {
            "variable": f"mem:{op.memory_id}",
            "operation": "latch_multiplier",
            "details": value_str,
            "role": "latch_multiplier",
        }
        if hasattr(op, "source_ast") and op.source_ast:
            if hasattr(op.source_ast, "line"):
                info["line"] = op.source_ast.line
            if hasattr(op.source_ast, "source_file"):
                info["source_file"] = op.source_ast.source_file
        return info
