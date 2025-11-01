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

    def place_ir_operation(self, op: IRNode) -> None:
        """Place a single IR operation."""
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
            },
            role="memory_write_gate",
            zone="memory",
        )
        self.plan.add_placement(write_placement)

        # Create hold gate
        hold_pos = self.layout.get_next_position(footprint=(1, 1))
        hold_id = f"{op.memory_id}_hold_gate"
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

        write_gate = memory_module.get("write_gate")
        hold_gate = memory_module.get("hold_gate")

        if not isinstance(write_gate, EntityPlacement) or not isinstance(
            hold_gate, EntityPlacement
        ):
            self.diagnostics.warning(
                f"Memory module '{op.memory_id}' is missing gate placements"
            )
            return

        is_always_write = isinstance(op.write_enable, int) and op.write_enable == 1

        if is_always_write:
            self._add_signal_sink(op.data_signal, write_gate.ir_node_id)

            write_gate.properties["operation"] = ">"
            output_signal = write_gate.properties.get("output_signal", "signal-A")
            write_gate.properties["left_operand"] = output_signal
            write_gate.properties["right_operand"] = -2147483648
            write_gate.properties["copy_count_from_input"] = True

            signal_name = memory_module.get("signal_type")
            if signal_name and not memory_module.get("_feedback_connected"):
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

            self.signal_graph.set_source(op.memory_id, write_gate.ir_node_id)
            memory_module["always_write"] = True
        else:
            self._add_signal_sink(op.data_signal, write_gate.ir_node_id)
            self._add_signal_sink(op.data_signal, hold_gate.ir_node_id)

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
