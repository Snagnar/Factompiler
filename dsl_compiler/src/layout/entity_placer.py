from typing import Any

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.common.entity_data import (
    get_entity_alignment,
    get_entity_footprint,
)
from dsl_compiler.src.ir.builder import (
    BundleRef,
    IR_Arith,
    IR_Const,
    IR_Decider,
    IR_MemCreate,
    IR_MemRead,
    IR_MemWrite,
    IR_PlaceEntity,
    IR_WireMerge,
    IRNode,
    SignalRef,
    ValueRef,
)
from dsl_compiler.src.ir.nodes import (
    IR_EntityOutput,
    IR_EntityPropRead,
    IR_EntityPropWrite,
    IR_LatchWrite,
)

from .layout_plan import EntityPlacement, LayoutPlan
from .memory_builder import MemoryBuilder
from .signal_analyzer import SignalAnalyzer
from .signal_graph import SignalGraph
from .tile_grid import TileGrid


class EntityPlacer:
    """Plans physical placement of IR entities without materializing them."""

    def __init__(
        self,
        tile_grid: TileGrid,
        layout_plan: LayoutPlan,
        signal_analyzer: SignalAnalyzer,
        diagnostics: ProgramDiagnostics,
    ):
        self.tile_grid = tile_grid
        self.plan = layout_plan
        self.signal_analyzer = signal_analyzer
        self.signal_usage = signal_analyzer.signal_usage
        self.diagnostics = diagnostics
        self.signal_graph = SignalGraph()

        self.memory_builder = MemoryBuilder(
            tile_grid, layout_plan, signal_analyzer, diagnostics
        )

        self._memory_modules: dict[str, dict[str, Any]] = {}
        self._wire_merge_junctions: dict[str, dict[str, Any]] = {}
        self._entity_property_signals: dict[str, str] = {}
        self._ir_nodes: dict[str, IRNode] = {}  # Track all IR nodes by ID for lookups
        self._merge_membership: dict[str, set] = {}  # source_id -> set of merge_ids

    def _build_debug_info(
        self, op: IRNode, role_override: str | None = None
    ) -> dict[str, Any]:
        """Extract debug information from an IR node and its usage entry."""
        debug_info: dict[str, Any] = {}
        usage = self.signal_usage.get(op.node_id)

        # Variable name - prefer debug_label which is the actual variable name
        debug_info["variable"] = (
            (usage.debug_label if usage else None)
            or getattr(op, "debug_label", None)
            or op.node_id
        )

        # Source location - try multiple sources
        source_ast = (usage.source_ast if usage else None) or getattr(
            op, "source_ast", None
        )

        # Get line from source_ast first
        line = None
        source_file = None
        if source_ast:
            if hasattr(source_ast, "line") and source_ast.line and source_ast.line > 0:
                line = source_ast.line
            if hasattr(source_ast, "source_file") and source_ast.source_file:
                source_file = source_ast.source_file

        # Extract expression context from debug_metadata
        expr_context_target = None
        expr_context_line = None
        expr_context_file = None
        if hasattr(op, "debug_metadata") and op.debug_metadata:
            expr_context_target = op.debug_metadata.get("expr_context_target")
            expr_context_line = op.debug_metadata.get("expr_context_line")
            expr_context_file = op.debug_metadata.get("expr_context_file")

        # Use expression context line if we don't have a line from source_ast
        if not line and expr_context_line:
            line = expr_context_line
        if not source_file and expr_context_file:
            source_file = expr_context_file

        # Store expression context target for intermediate computations
        if expr_context_target:
            debug_info["expr_context"] = expr_context_target

        if line:
            debug_info["line"] = line
        if source_file:
            debug_info["source_file"] = source_file

        # Signal type
        debug_info["signal_type"] = (
            usage.resolved_signal_name if usage else None
        ) or getattr(op, "output_type", None)

        # Check for user declaration
        if hasattr(op, "debug_metadata") and op.debug_metadata:
            if op.debug_metadata.get("user_declared"):
                debug_info["user_declared"] = True
                declared_name = op.debug_metadata.get("declared_name")
                if declared_name:
                    debug_info["variable"] = declared_name

        # Operation-specific info
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

        if role_override:
            debug_info["role"] = role_override

        return {k: v for k, v in debug_info.items() if v is not None}

    def place_ir_operation(self, op: IRNode) -> None:
        """Place a single IR operation."""
        self._ir_nodes[op.node_id] = op
        self.memory_builder.register_ir_node(op)

        if isinstance(op, IR_Const):
            self._place_constant(op)
        elif isinstance(op, IR_Arith):
            self._place_arithmetic(op)
        elif isinstance(op, IR_Decider):
            self._place_decider(op)
        elif isinstance(op, IR_MemCreate):
            self.memory_builder.create_memory(op, self.signal_graph)
        elif isinstance(op, IR_MemRead):
            self.memory_builder.handle_read(op, self.signal_graph)
        elif isinstance(op, IR_MemWrite):
            self.memory_builder.handle_write(op, self.signal_graph)
        elif isinstance(op, IR_LatchWrite):
            self.memory_builder.handle_latch_write(op, self.signal_graph)
        elif isinstance(op, IR_PlaceEntity):
            self._place_user_entity(op)
        elif isinstance(op, IR_EntityPropWrite):
            self._place_entity_prop_write(op)
        elif isinstance(op, IR_EntityPropRead):
            self._place_entity_prop_read(op)
        elif isinstance(op, IR_EntityOutput):
            self._place_entity_output(op)
        elif isinstance(op, IR_WireMerge):
            self._place_wire_merge(op)
        else:
            self.diagnostics.warning(f"Unknown IR operation: {type(op)}")

    def _place_constant(self, op: IR_Const) -> None:
        """Place constant combinator (if materialization required)."""
        usage = self.signal_usage.get(op.node_id)
        if not usage or not usage.should_materialize:
            # For bundle constants, always materialize if they have signals
            if not op.signals:
                return

        signal_name = self.signal_analyzer.resolve_signal_name(op.output_type, usage)
        signal_type = self.signal_analyzer.resolve_signal_type(op.output_type, usage)

        pos = None

        # Build debug info
        debug_info = self._build_debug_info(op)

        if hasattr(op, "debug_metadata") and op.debug_metadata:
            folded_from = op.debug_metadata.get("folded_from")
            if folded_from:
                debug_info["details"] = (
                    f"folded from {len(folded_from)} constants: value={op.value}"
                )
                debug_info["fold_count"] = len(folded_from)

        is_input = False
        if hasattr(op, "debug_metadata") and op.debug_metadata.get("user_declared"):
            is_input = True

        # Handle multi-signal constants (bundles)
        if op.signals:
            self.plan.create_and_add_placement(
                ir_node_id=op.node_id,
                entity_type="constant-combinator",
                position=pos,
                footprint=(1, 2),
                role="bundle_const",
                debug_info=debug_info,
                signals=op.signals,  # Dict of signal_name -> value
                is_input=is_input,
            )
        else:
            # Store placement in plan (NOT creating Draftsman entity yet!)
            self.plan.create_and_add_placement(
                ir_node_id=op.node_id,
                entity_type="constant-combinator",
                position=pos,
                footprint=(1, 2),
                role="literal",
                debug_info=debug_info,
                signal_name=signal_name,
                signal_type=signal_type,
                value=op.value,
                is_input=is_input,  # Mark user-declared constants as inputs
            )

        self.signal_graph.set_source(op.node_id, op.node_id)

    def _place_arithmetic(self, op: IR_Arith) -> None:
        """Place arithmetic combinator."""
        left_pos = self._get_placement_position(op.left)
        right_pos = self._get_placement_position(op.right)

        deps = []
        if left_pos:
            deps.append(left_pos)
        if right_pos:
            deps.append(right_pos)

        pos = None

        usage = self.signal_usage.get(op.node_id)
        left_operand = self.signal_analyzer.get_operand_for_combinator(op.left)
        right_operand = self.signal_analyzer.get_operand_for_combinator(op.right)
        output_signal = self.signal_analyzer.resolve_signal_name(op.output_type, usage)

        self.plan.create_and_add_placement(
            ir_node_id=op.node_id,
            entity_type="arithmetic-combinator",
            position=pos,
            footprint=(1, 2),
            role="arithmetic",
            debug_info=self._build_debug_info(op),
            operation=op.op,
            left_operand=left_operand,
            right_operand=right_operand,
            left_operand_signal_id=op.left,  # IR signal ID for wire color lookup
            right_operand_signal_id=op.right,  # IR signal ID for wire color lookup
            output_signal=output_signal,
            needs_wire_separation=op.needs_wire_separation,  # For bundle operations
        )

        self.signal_graph.set_source(op.node_id, op.node_id)
        self._add_signal_sink(op.left, op.node_id)
        self._add_signal_sink(op.right, op.node_id)

    def _place_decider(self, op: IR_Decider) -> None:
        """Place decider combinator.

        Handles both single-condition mode (legacy) and multi-condition mode
        (condition folding optimization or SR latches).
        """
        if op.conditions:
            # Multi-condition mode
            self._place_multi_condition_decider(op)
        else:
            # Legacy single-condition mode
            self._place_single_condition_decider(op)

    def _place_single_condition_decider(self, op: IR_Decider) -> None:
        """Place a single-condition decider combinator (legacy mode)."""
        left_pos = self._get_placement_position(op.left)
        right_pos = self._get_placement_position(op.right)
        output_pos = self._get_placement_position(op.output_value)

        deps = []
        for p in [left_pos, right_pos, output_pos]:
            if p:
                deps.append(p)

        pos = None

        usage = self.signal_usage.get(op.node_id)
        left_operand = self.signal_analyzer.get_operand_for_combinator(op.left)
        right_operand = self.signal_analyzer.get_operand_for_combinator(op.right)
        output_signal = self.signal_analyzer.resolve_signal_name(op.output_type, usage)
        output_value = self.signal_analyzer.get_operand_for_combinator(op.output_value)

        # Use the copy_count_from_input field from the IR node
        copy_count_from_input = op.copy_count_from_input

        self.plan.create_and_add_placement(
            ir_node_id=op.node_id,
            entity_type="decider-combinator",
            position=pos,
            footprint=(1, 2),
            role="decider",
            debug_info=self._build_debug_info(op),
            operation=op.test_op,
            left_operand=left_operand,
            right_operand=right_operand,
            left_operand_signal_id=op.left,  # IR signal ID for wire color lookup
            right_operand_signal_id=op.right,  # IR signal ID for wire color lookup
            output_signal=output_signal,
            output_value=output_value,
            copy_count_from_input=copy_count_from_input,
        )

        self.signal_graph.set_source(op.node_id, op.node_id)
        self._add_signal_sink(op.left, op.node_id)
        self._add_signal_sink(op.right, op.node_id)
        if not isinstance(op.output_value, int):
            self._add_signal_sink(op.output_value, op.node_id)

    def _place_multi_condition_decider(self, op: IR_Decider) -> None:
        """Place a multi-condition decider combinator.

        Handles conditions from:
        1. IR-time construction (condition folding): Uses first_operand/second_operand ValueRefs
        2. Layout-time construction (SR latches): Uses first_signal/second_signal strings
        """

        # Track all input operands for signal graph
        all_operands = []

        # Build conditions list for the placement properties
        conditions_list = []
        for cond in op.conditions:
            cond_dict = {
                "comparator": cond.comparator,
                "compare_type": cond.compare_type,
            }

            # Handle first operand - check ValueRef first, then string fallback
            if cond.first_operand is not None:
                # IR-time: ValueRef needs resolution
                if isinstance(cond.first_operand, int):
                    cond_dict["first_constant"] = cond.first_operand
                else:
                    first_op = self.signal_analyzer.get_operand_for_combinator(
                        cond.first_operand
                    )
                    cond_dict["first_signal"] = first_op
                    all_operands.append(cond.first_operand)
            elif cond.first_signal:
                # Layout-time: string already resolved
                cond_dict["first_signal"] = cond.first_signal
                if cond.first_signal_wires:
                    cond_dict["first_signal_wires"] = cond.first_signal_wires
            elif cond.first_constant is not None:
                cond_dict["first_constant"] = cond.first_constant

            # Handle second operand
            if cond.second_operand is not None:
                # IR-time: ValueRef needs resolution
                if isinstance(cond.second_operand, int):
                    cond_dict["second_constant"] = cond.second_operand
                else:
                    second_op = self.signal_analyzer.get_operand_for_combinator(
                        cond.second_operand
                    )
                    cond_dict["second_signal"] = second_op
                    all_operands.append(cond.second_operand)
            elif cond.second_signal:
                # Layout-time: string already resolved
                cond_dict["second_signal"] = cond.second_signal
                if cond.second_signal_wires:
                    cond_dict["second_signal_wires"] = cond.second_signal_wires
            elif cond.second_constant is not None:
                cond_dict["second_constant"] = cond.second_constant

            conditions_list.append(cond_dict)

        # Resolve output
        usage = self.signal_usage.get(op.node_id)
        output_signal = self.signal_analyzer.resolve_signal_name(op.output_type, usage)
        output_value = self.signal_analyzer.get_operand_for_combinator(op.output_value)

        self.plan.create_and_add_placement(
            ir_node_id=op.node_id,
            entity_type="decider-combinator",
            position=None,
            footprint=(1, 2),
            role="decider",
            debug_info=self._build_debug_info(op),
            conditions=conditions_list,
            output_signal=output_signal,
            output_value=output_value,
            copy_count_from_input=op.copy_count_from_input,
        )

        # Signal graph: this node is source of its output
        self.signal_graph.set_source(op.node_id, op.node_id)

        # All operands from conditions are sinks
        for operand in all_operands:
            self._add_signal_sink(operand, op.node_id)

        # Output value if it's a signal
        if not isinstance(op.output_value, int):
            self._add_signal_sink(op.output_value, op.node_id)

    def _place_user_entity(self, op: IR_PlaceEntity) -> None:
        """Place user-requested entity."""
        prototype = op.prototype

        footprint = get_entity_footprint(prototype)

        alignment = get_entity_alignment(prototype)

        user_specified = isinstance(op.x, int) and isinstance(op.y, int)

        if user_specified:
            desired = (int(op.x), int(op.y))
            pos = desired
        else:
            pos = None

        placement = EntityPlacement(
            ir_node_id=op.entity_id,
            entity_type=prototype,
            position=pos,
            properties=op.properties or {},
            role="user_entity",
        )
        placement.properties["footprint"] = footprint
        placement.properties["alignment"] = (
            alignment  # Store alignment in properties if needed
        )

        if user_specified:
            placement.properties["user_specified_position"] = True

        entity_debug = {
            "variable": op.entity_id,
            "operation": "place",
            "details": f"proto={prototype}",
            "role": "user_entity",
        }
        if hasattr(op, "source_ast") and op.source_ast:
            line = getattr(op.source_ast, "line", None)
            if line and line > 0:
                entity_debug["line"] = line
            source_file = getattr(op.source_ast, "source_file", None)
            if source_file:
                entity_debug["source_file"] = source_file

        placement.properties["debug_info"] = entity_debug

        self.plan.add_placement(placement)

    def _place_entity_prop_write(self, op: IR_EntityPropWrite) -> None:
        """Handle entity property writes."""
        placement = self.plan.get_placement(op.entity_id)
        if placement is None:
            self.diagnostics.warning(
                f"Property write to non-existent entity: {op.entity_id}"
            )
            return

        if "property_writes" not in placement.properties:
            placement.properties["property_writes"] = {}

        # Handle inline bundle condition (all()/any() inlining)
        if hasattr(op, "inline_bundle_condition") and op.inline_bundle_condition:
            cond = op.inline_bundle_condition
            placement.properties["property_writes"][op.property_name] = {
                "type": "inline_bundle_condition",
                "signal": cond["signal"],
                "operator": cond["operator"],
                "constant": cond["constant"],
            }
            # Track wire connection from bundle source to entity
            input_source = cond.get("input_source")
            if isinstance(input_source, (SignalRef, BundleRef)):
                self._add_signal_sink(input_source, op.entity_id)
            self.diagnostics.info(
                f"Inlined bundle condition ({cond['signal']}) into "
                f"{op.entity_id}.{op.property_name}"
            )
            return

        # Try to inline simple comparisons
        if isinstance(op.value, SignalRef) and op.property_name == "enable":
            inline_data = self._try_inline_comparison(op.value)
            if inline_data:
                placement.properties["property_writes"][op.property_name] = {
                    "type": "inline_comparison",
                    "comparison_data": inline_data,
                }
                inline_data["source_node_id_to_remove"] = op.value.source_id

                # Preserve debug info from the inlined comparison
                comparison_placement = self.plan.get_placement(op.value.source_id)
                if (
                    comparison_placement
                    and "debug_info" in comparison_placement.properties
                ):
                    comp_debug = comparison_placement.properties["debug_info"]
                    if "property_writes" in placement.properties:
                        writes = placement.properties["property_writes"]
                        if (
                            op.property_name in writes
                            and writes[op.property_name].get("type")
                            == "inline_comparison"
                        ):
                            writes[op.property_name]["inlined_from"] = comp_debug.get(
                                "variable", "comparison"
                            )

                # ✅ FIX: Track that entity needs the comparison's input signal
                # The entity must read the signal being compared
                ir_node = self._ir_nodes.get(op.value.source_id)
                if isinstance(ir_node, IR_Decider):
                    if isinstance(ir_node.left, SignalRef):
                        self.signal_graph.remove_sink(
                            ir_node.left.source_id, op.value.source_id
                        )
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

        if isinstance(op.value, SignalRef):
            self._add_signal_sink(op.value, op.entity_id)
            placement.properties["property_writes"][op.property_name] = {
                "type": "signal",
                "signal_ref": op.value,
            }
        elif isinstance(op.value, int):
            placement.properties["property_writes"][op.property_name] = {
                "type": "constant",
                "value": op.value,
            }
        else:
            placement.properties["property_writes"][op.property_name] = {
                "type": "value",
                "value": op.value,
            }

    def _try_inline_comparison(self, signal_ref: SignalRef) -> dict | None:
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

        if not (isinstance(right, int) and output_value == 1):
            return None

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

    def _place_entity_output(self, op: "IR_EntityOutput") -> None:
        """Handle entity output reads (e.g., chest.output, tank.output).

        Entity outputs expose the circuit network output of entities like chests
        (item counts) or tanks (fluid levels) as a Bundle. This doesn't create
        any new entities - it just establishes that the IR node reads from
        the entity's circuit output.
        """
        # Track that this node reads from the entity's circuit output
        self.signal_graph.set_source(op.node_id, op.entity_id)

    def _place_wire_merge(self, op: IR_WireMerge) -> None:
        """Handle wire merge operations with membership tracking."""
        # Wire merges don't create entities, they just affect wiring topology
        # Record the merge junction for later wire planning
        self._wire_merge_junctions[op.node_id] = {
            "inputs": list(op.sources),
            "output_id": op.node_id,
        }

        # Track which sources belong to this merge for wire color conflict detection
        # IMPORTANT: Track by ACTUAL entity ID, not IR node ID, to detect when the
        # same physical entity's output is used in multiple different wire merges.
        # This is essential for balanced loader patterns where a chest output is used
        # both in a "total" merge (all chests together) and individual "diff" merges.
        for source in op.sources:
            if isinstance(source, (SignalRef, BundleRef)):
                source_id = source.source_id
                # Resolve IR node ID to actual entity ID using signal graph
                actual_entity_id = self.signal_graph.get_source(source_id)
                if actual_entity_id is None:
                    actual_entity_id = source_id

                if actual_entity_id not in self._merge_membership:
                    self._merge_membership[actual_entity_id] = set()
                self._merge_membership[actual_entity_id].add(op.node_id)

        # Track signal graph: merge creates a new source from multiple inputs
        self.signal_graph.set_source(op.node_id, op.node_id)
        for input_sig in op.sources:
            self._add_signal_sink(input_sig, op.node_id)

    def get_merge_membership(self) -> dict[str, set]:
        """Return merge membership info for wire color conflict detection."""
        return self._merge_membership

    def _get_placement_position(self, value_ref: ValueRef) -> tuple[int, int] | None:
        """Get position of entity producing this value."""
        if isinstance(value_ref, SignalRef):
            placement = self.plan.get_placement(value_ref.source_id)
            return placement.position if placement else None
        return None

    def _add_signal_sink(self, value_ref: ValueRef, consumer_id: str) -> None:
        """Track signal consumption."""
        if isinstance(value_ref, SignalRef):
            source_usage = self.signal_usage.get(value_ref.source_id)
            if source_usage and not source_usage.should_materialize:
                return

            self.signal_graph.add_sink(value_ref.source_id, consumer_id)
        elif isinstance(value_ref, BundleRef):
            # For bundles, connect from the bundle's source (usually a wire merge)
            self.signal_graph.add_sink(value_ref.source_id, consumer_id)

    def cleanup_unused_entities(self) -> None:
        """Remove entities marked as unused during optimization."""
        self.memory_builder.cleanup_unused_gates(self.plan, self.signal_graph)

        self._memory_modules = self.memory_builder._modules

        entities_to_remove = []
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

        # Also remove any planned wire connections that reference removed entities
        if entities_to_remove:
            remaining_connections = []
            for conn in self.plan.wire_connections:
                if (
                    conn.source_entity_id in entities_to_remove
                    or conn.sink_entity_id in entities_to_remove
                ):
                    self.diagnostics.info(
                        f"Removed stale wire connection referencing removed entity: {conn.signal_name} ({conn.source_entity_id} -> {conn.sink_entity_id})"
                    )
                    continue
                remaining_connections.append(conn)

            self.plan.wire_connections = remaining_connections

        # create new signal graph, iterate through iter_source_sink_pairs of old graph and only add entries not involving removed entities
        new_signal_graph = SignalGraph()
        for signal_id, source_id, sink_id in self.signal_graph.iter_source_sink_pairs():
            if source_id in entities_to_remove or sink_id in entities_to_remove:
                self.diagnostics.info(
                    f"Removed stale signal graph edge referencing removed entity: {signal_id} ({source_id} -> {sink_id})"
                )
                continue
            new_signal_graph.add_sink(signal_id, sink_id)
            new_signal_graph.set_source(signal_id, source_id)
        self.signal_graph = new_signal_graph

    def create_output_anchors(self) -> None:
        """Create empty constant combinators for output signals with no consumers.

        For each signal marked as is_output (which already implies it has a
        variable name and no consumers), create an empty constant combinator
        as an anchor point for viewing output values.

        If a signal has output_aliases (variable names that alias to it but are
        not consumed), create one anchor for each alias.
        """
        for signal_id, entry in self.signal_usage.items():
            # is_output already checks: has debug_label, no consumers, named signal
            if not entry.debug_metadata.get("is_output"):
                continue

            if not entry.producer:
                continue

            # Get output aliases if any
            output_aliases = entry.output_aliases

            # For IR_Const producers: only create anchors for aliases, not the original
            # The original constant combinator already exists and serves as output
            if isinstance(entry.producer, IR_Const):
                if not output_aliases:
                    # No aliases - the constant combinator itself is the output
                    continue
                # Remove the ORIGINAL declared name from aliases (it's already materialized)
                # Use declared_name from debug_metadata which is preserved as the original
                original_name = entry.debug_metadata.get("declared_name")
                if original_name:
                    output_aliases = output_aliases - {original_name}
                if not output_aliases:
                    continue

            # For non-const producers: use output_aliases if any, otherwise use debug_label
            if not output_aliases:
                output_aliases = {entry.debug_label} if entry.debug_label else set()

            for alias_name in output_aliases:
                anchor_id = f"{signal_id}_{alias_name}_output_anchor"

                # Build debug info for the anchor
                debug_info = {
                    "variable": alias_name,
                    "operation": "output",
                    "details": "anchor",
                    "signal_type": entry.resolved_signal_name or entry.signal_type,
                }

                if "location" in entry.debug_metadata:
                    location = entry.debug_metadata["location"]
                    if ":" in location:
                        file_part, line_part = location.rsplit(":", 1)
                        debug_info["source_file"] = file_part
                        try:
                            debug_info["line"] = int(line_part)
                        except ValueError:
                            pass

                self.plan.create_and_add_placement(
                    ir_node_id=anchor_id,
                    entity_type="constant-combinator",
                    position=None,  # Will be set by layout optimizer
                    footprint=(1, 1),
                    role="output_anchor",
                    debug_info=debug_info,
                    signals=[],  # Empty constant combinator
                    is_output=True,  # Mark output anchors as outputs
                )

                # Wire the producer's output to this anchor
                # The anchor acts as a sink for the signal
                self.signal_graph.add_sink(signal_id, anchor_id)

                self.diagnostics.info(
                    f"Created output anchor '{anchor_id}' for signal '{alias_name}'"
                )
