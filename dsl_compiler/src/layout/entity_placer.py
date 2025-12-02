from typing import Any, Dict, Optional, Tuple
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.common.entity_data import (
    get_entity_footprint,
    get_entity_alignment,
)
from .layout_plan import LayoutPlan, EntityPlacement, WireConnection
from .signal_analyzer import SignalAnalyzer, SignalUsageEntry
from .signal_graph import SignalGraph
from .tile_grid import TileGrid
from .memory_builder import MemoryBuilder


from dsl_compiler.src.ir.builder import (
    IRNode,
    IR_Const,
    IR_Arith,
    IR_Decider,
    IR_MemCreate,
    IR_MemRead,
    IR_MemWrite,
    IR_PlaceEntity,
    IR_WireMerge,
    SignalRef,
    ValueRef,
)
from dsl_compiler.src.ir.nodes import (
    IR_EntityPropWrite,
    IR_EntityPropRead,
)


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

        # Create memory builder
        self.memory_builder = MemoryBuilder(
            tile_grid, layout_plan, signal_analyzer, diagnostics
        )

        self._memory_modules: Dict[str, Dict[str, Any]] = {}
        self._wire_merge_junctions: Dict[str, Dict[str, Any]] = {}
        self._entity_property_signals: Dict[str, str] = {}
        self._ir_nodes: Dict[str, IRNode] = {}  # Track all IR nodes by ID for lookups

    def _build_debug_info(
        self, op: IRNode, role_override: Optional[str] = None
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
        # Track this IR node for later lookups and for memory builder
        self._ir_nodes[op.node_id] = op
        self.memory_builder.register_ir_node(op)

        if isinstance(op, IR_Const):
            self._place_constant(op)
        elif isinstance(op, IR_Arith):
            self._place_arithmetic(op)
        elif isinstance(op, IR_Decider):
            self._place_decider(op)
        elif isinstance(op, IR_MemCreate):
            # Delegate to memory builder
            self.memory_builder.create_memory(op, self.signal_graph)
        elif isinstance(op, IR_MemRead):
            # Delegate to memory builder
            self.memory_builder.handle_read(op, self.signal_graph)
        elif isinstance(op, IR_MemWrite):
            # Delegate to memory builder
            self.memory_builder.handle_write(op, self.signal_graph)
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
        signal_name = self.signal_analyzer.resolve_signal_name(op.output_type, usage)
        signal_type = self.signal_analyzer.resolve_signal_type(op.output_type, usage)

        # Position will be set by force-directed layout
        pos = None

        # Build debug info
        debug_info = self._build_debug_info(op)

        # Add fold information if this is a folded constant
        if hasattr(op, "debug_metadata") and op.debug_metadata:
            folded_from = op.debug_metadata.get("folded_from")
            if folded_from:
                debug_info["details"] = (
                    f"folded from {len(folded_from)} constants: value={op.value}"
                )
                debug_info["fold_count"] = len(folded_from)

        # Check if this is a user-declared input constant
        is_input = False
        if hasattr(op, "debug_metadata") and op.debug_metadata.get("user_declared"):
            is_input = True

        # Store placement in plan (NOT creating Draftsman entity yet!)
        placement = EntityPlacement(
            ir_node_id=op.node_id,
            entity_type="constant-combinator",
            position=pos,
            properties={
                "signal_name": signal_name,
                "signal_type": signal_type,
                "value": op.value,
                "footprint": (1, 2),
                "debug_info": debug_info,
                "is_input": is_input,  # Mark user-declared constants as inputs
            },
            role="literal",
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

        # Position will be set by force-directed layout
        pos = None

        # Resolve operands and output
        usage = self.signal_usage.get(op.node_id)
        left_operand = self.signal_analyzer.get_operand_for_combinator(op.left)
        right_operand = self.signal_analyzer.get_operand_for_combinator(op.right)
        output_signal = self.signal_analyzer.resolve_signal_name(op.output_type, usage)

        # Store placement
        placement = EntityPlacement(
            ir_node_id=op.node_id,
            entity_type="arithmetic-combinator",
            position=pos,
            properties={
                "operation": op.op,
                "left_operand": left_operand,
                "right_operand": right_operand,
                "left_operand_signal_id": op.left,  # IR signal ID for wire color lookup
                "right_operand_signal_id": op.right,  # IR signal ID for wire color lookup
                "output_signal": output_signal,
                "footprint": (1, 2),
                "debug_info": self._build_debug_info(op),
            },
            role="arithmetic",
        )
        self.plan.add_placement(placement)

        # Track signals
        self.signal_graph.set_source(op.node_id, op.node_id)
        self._add_signal_sink(op.left, op.node_id)
        self._add_signal_sink(op.right, op.node_id)
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

        # Position will be set by force-directed layout
        pos = None

        # Resolve operands
        usage = self.signal_usage.get(op.node_id)
        left_operand = self.signal_analyzer.get_operand_for_combinator(op.left)
        right_operand = self.signal_analyzer.get_operand_for_combinator(op.right)
        output_signal = self.signal_analyzer.resolve_signal_name(op.output_type, usage)
        output_value = self.signal_analyzer.get_operand_for_combinator(op.output_value)

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
                "left_operand_signal_id": op.left,  # IR signal ID for wire color lookup
                "right_operand_signal_id": op.right,  # IR signal ID for wire color lookup
                "output_signal": output_signal,
                "output_value": output_value,
                "copy_count_from_input": copy_count_from_input,
                "footprint": (1, 2),
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

    def _place_user_entity(self, op: IR_PlaceEntity) -> None:
        """Place user-requested entity."""
        prototype = op.prototype

        # Determine footprint dynamically from draftsman
        footprint = get_entity_footprint(prototype)

        # Determine alignment requirement dynamically
        alignment = get_entity_alignment(prototype)

        # Check if coordinates are constants (user-specified position)
        user_specified = isinstance(op.x, int) and isinstance(op.y, int)

        # Determine position
        if user_specified:
            # Explicit position provided - use it directly
            # (force-directed layout will respect user-specified positions)
            desired = (int(op.x), int(op.y))
            pos = desired
        else:
            # Position will be set by force-directed layout
            pos = None

        # Store placement
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

        # Mark user-specified positions
        if user_specified:
            placement.properties["user_specified_position"] = True

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
                    # Add dependency on the left operand (the signal being tested)
                    if isinstance(ir_node.left, SignalRef):
                        # Remove the old edge from source to decider
                        self.signal_graph.remove_sink(
                            ir_node.left.source_id, op.value.source_id
                        )
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
            # Skip non-materialized constants - they're inlined and don't need wiring
            source_usage = self.signal_usage.get(value_ref.source_id)
            if source_usage and not source_usage.should_materialize:
                return

            self.signal_graph.add_sink(value_ref.source_id, consumer_id)

    def cleanup_unused_entities(self) -> None:
        """Remove entities marked as unused during optimization."""
        # Delegate memory cleanup to memory builder
        self.memory_builder.cleanup_unused_gates(self.plan, self.signal_graph)

        # ✅ Copy memory modules from builder for later use in wiring
        self._memory_modules = self.memory_builder._modules

        # Remove inlined comparisons
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
        """
        for signal_id, entry in self.signal_usage.items():
            # is_output already checks: has debug_label, no consumers, named signal
            if not entry.debug_metadata.get("is_output"):
                continue

            # Get the producer node
            if not entry.producer:
                continue

            # Skip if producer is already a constant combinator
            if isinstance(entry.producer, IR_Const):
                continue

            # Create empty constant combinator as output anchor
            anchor_id = f"{signal_id}_output_anchor"

            # Build debug info for the anchor
            debug_info = {
                "variable": entry.debug_label,
                "operation": "output",
                "details": "anchor",
                "signal_type": entry.resolved_signal_name or entry.signal_type,
            }

            # Add location info from metadata
            if "location" in entry.debug_metadata:
                location = entry.debug_metadata["location"]
                if ":" in location:
                    file_part, line_part = location.rsplit(":", 1)
                    debug_info["source_file"] = file_part
                    try:
                        debug_info["line"] = int(line_part)
                    except ValueError:
                        pass

            # Create the placement
            placement = EntityPlacement(
                ir_node_id=anchor_id,
                entity_type="constant-combinator",
                position=None,  # Will be set by layout optimizer
                properties={
                    "signals": [],  # Empty constant combinator
                    "footprint": (1, 1),
                    "debug_info": debug_info,
                    "is_output": True,  # Mark output anchors as outputs
                },
                role="output_anchor",
            )
            self.plan.add_placement(placement)

            # Wire the producer's output to this anchor
            # The anchor acts as a sink for the signal
            self.signal_graph.add_sink(signal_id, anchor_id)

            self.diagnostics.info(
                f"Created output anchor '{anchor_id}' for signal '{entry.debug_label}'"
            )
