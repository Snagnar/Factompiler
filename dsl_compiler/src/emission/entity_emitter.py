"""Entity emission helpers for the blueprint emitter."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

from draftsman.data import signals as signal_data  # type: ignore[import-not-found]
from draftsman.entity import (  # type: ignore[import-not-found]
    ConstantCombinator,
    DeciderCombinator,
    new_entity,
)
from draftsman.signatures import SignalID  # type: ignore[import-not-found]

from ..ir import (
    IR_Arith,
    IR_Const,
    IR_Decider,
    IR_EntityPropRead,
    IR_EntityPropWrite,
    IR_MemCreate,
    IR_MemRead,
    IR_MemWrite,
    IR_PlaceEntity,
    IR_WireMerge,
    SignalRef,
    ValueRef,
)
from ..semantic import render_source_location
from .signals import EntityPlacement

if TYPE_CHECKING:  # pragma: no cover - type checking aid
    from .emitter import BlueprintEmitter
    from .signal_resolver import SignalResolver


class EntityEmitter:
    """Emit Factorio entities from lowered IR operations."""

    def __init__(self, parent: "BlueprintEmitter") -> None:  # pragma: no cover - thin wrapper
        self._parent = parent

    def __getattr__(self, name):  # pragma: no cover - delegation helper
        return getattr(self._parent, name)

    def _require_resolver(self) -> "SignalResolver":
        resolver = self.signal_resolver
        if resolver is None:
            raise RuntimeError("SignalResolver not initialised; call prepare() first")
        return resolver

    def emit_constant(self, op: IR_Const):
        """Emit constant combinator for IR_Const, only if materialization is required."""
        resolver = self._require_resolver()

        usage_entry = self.signal_usage.get(op.node_id)
        if self.materializer and usage_entry and not usage_entry.should_materialize:
            print(
                f"SKIP constant: {op.node_id} (should_materialize={usage_entry.should_materialize}, is_typed_literal={getattr(usage_entry, 'is_typed_literal', None)})"
            )
            return

        combinator = new_entity("constant-combinator")
        pos = self._place_entity_in_zone(combinator, "north_literals")
        section = combinator.add_section()
        if usage_entry and usage_entry.is_typed_literal:
            base_signal_key = usage_entry.literal_declared_type or op.output_type
            if self.materializer:
                signal_name = self.materializer.resolve_signal_name(
                    base_signal_key, usage_entry
                )
            else:
                signal_name = resolver.get_signal_name(base_signal_key)
            signal_type = usage_entry.resolved_signal_type or (
                self.materializer.resolve_signal_type(op.output_type, usage_entry)
                if self.materializer
                else None
            )
            if not signal_type:
                if signal_name in self.signal_type_map:
                    mapped = self.signal_type_map[signal_name]
                    if isinstance(mapped, dict):
                        signal_type = mapped.get("type")
                if not signal_type:
                    if signal_data is not None and signal_name in signal_data.raw:
                        proto_type = signal_data.raw[signal_name].get("type", "virtual")
                        signal_type = (
                            "virtual" if proto_type == "virtual-signal" else proto_type
                        )
                    elif signal_name.startswith("signal-"):
                        signal_type = "virtual"
                    else:
                        signal_type = "item"
            value = (
                usage_entry.literal_value
                if usage_entry.literal_value is not None
                else op.value
            )
        else:
            signal_name = (
                self.materializer.resolve_signal_name(op.output_type, usage_entry)
                if self.materializer
                else resolver.get_signal_name(op.output_type)
            )
            signal_type = usage_entry.resolved_signal_type if usage_entry else None
            value = op.value
        print(
            f"EMIT constant: {op.node_id} name={signal_name} type={signal_type} value={value}"
        )
        if usage_entry and usage_entry.is_typed_literal:
            if value == 0:
                print(
                    f"EMIT anchor for explicit typed literal: {op.node_id} (value=0, name={signal_name})"
                )
            elif op.value != usage_entry.literal_value:
                print(
                    f"SKIP non-literal for explicit typed literal: {op.node_id} (value={op.value}, expected={usage_entry.literal_value}, name={signal_name})"
                )
                return
            print(
                f"DEBUG explicit typed literal: node_id={op.node_id} declared_type={usage_entry.literal_declared_type} signal_name={signal_name} output_type={op.output_type}"
            )
            print(
                f"EMIT explicit typed literal: {op.node_id} name={signal_name} value={value}"
            )
            if value != 0:
                try:
                    if signal_type:
                        section.set_signal(
                            index=0,
                            signal=SignalID(signal_name, type=signal_type),
                            count=value,
                        )
                    else:
                        section.set_signal(index=0, name=signal_name, count=value)
                except Exception:
                    section.set_signal(index=0, name=signal_name, count=value)
        else:
            try:
                if signal_type:
                    section.set_signal(
                        index=0,
                        signal=SignalID(signal_name, type=signal_type),
                        count=value,
                    )
                else:
                    section.set_signal(
                        index=0,
                        signal=SignalID(signal_name),
                        count=value,
                    )
            except Exception:
                section.set_signal(index=0, name=signal_name, count=value)
        combinator = self._add_entity(combinator)
        debug_info = self._compose_debug_info(usage_entry, fallback_name=signal_name)
        if debug_info:
            self.annotate_entity_description(combinator, debug_info)
        placement = EntityPlacement(
            entity=combinator,
            entity_id=op.node_id,
            position=pos,
            output_signals={signal_name: "red"},
            input_signals={},
            role="literal",
            zone="north_literals",
        )
        self.entities[op.node_id] = placement
        self.signal_graph.set_source(op.node_id, op.node_id)
        self._track_signal_source(op.node_id, op.node_id)
        for sec in getattr(combinator, "sections", []):
            print(
                f"DEBUG combinator {op.node_id} filters: {getattr(sec, 'filters', None)}"
            )

    def emit_arithmetic(self, op: IR_Arith):
        """Emit arithmetic combinator for IR_Arith."""

        resolver = self._require_resolver()
        combinator = new_entity("arithmetic-combinator")
        pos = self._place_entity(
            combinator, dependencies=(op.left, op.right)
        )

        left_operand = resolver.get_operand_for_combinator(op.left)
        right_operand = resolver.get_operand_for_combinator(op.right)
        output_signal = (
            self.materializer.resolve_signal_name(
                op.output_type, self.signal_usage.get(op.node_id)
            )
            if self.materializer
            else resolver.get_signal_name(op.output_type)
        )

        combinator.first_operand = left_operand
        combinator.second_operand = right_operand
        combinator.operation = op.op
        combinator.output_signal = output_signal

        combinator = self._add_entity(combinator)

        label_candidate = (
            output_signal
            if isinstance(output_signal, str)
            else getattr(output_signal, "name", None)
        )
        usage_entry = self.signal_usage.get(op.node_id)
        debug_info = self._compose_debug_info(
            usage_entry, fallback_name=label_candidate
        )
        if debug_info:
            self.annotate_entity_description(combinator, debug_info)

        placement = EntityPlacement(
            entity=combinator,
            entity_id=op.node_id,
            position=pos,
            output_signals={output_signal: "red"},
            input_signals={},
        )
        self.entities[op.node_id] = placement
        self.signal_graph.set_source(op.node_id, op.node_id)
        self._track_signal_source(op.node_id, op.node_id)

        self._add_signal_sink(op.left, op.node_id)
        self._add_signal_sink(op.right, op.node_id)

    def emit_wire_merge(self, op: IR_WireMerge) -> None:
        """Register a virtual wire merge junction for later wiring."""

        usage_entry = self.signal_usage.get(op.node_id)
        if usage_entry and not usage_entry.debug_label:
            usage_entry.debug_label = op.node_id

        self.signal_graph.set_source(op.node_id, op.node_id)

        sources: List[SignalRef] = [
            source for source in op.sources if isinstance(source, SignalRef)
        ]

        self._wire_merge_junctions[op.node_id] = {
            "sources": sources,
            "output_type": op.output_type,
            "source_ast": op.source_ast,
        }

    def emit_decider(self, op: IR_Decider):
        """Emit decider combinator for IR_Decider."""

        resolver = self._require_resolver()
        combinator = new_entity("decider-combinator")
        pos = self._place_entity(
            combinator, dependencies=(op.left, op.right)
        )

        left_operand = resolver.get_operand_for_combinator(op.left)
        right_operand = resolver.get_operand_value(op.right)
        output_signal = (
            self.materializer.resolve_signal_name(
                op.output_type, self.signal_usage.get(op.node_id)
            )
            if self.materializer
            else resolver.get_signal_name(op.output_type)
        )

        condition_kwargs = {"comparator": op.test_op}
        if isinstance(left_operand, int):
            condition_kwargs["first_signal"] = "signal-0"
            condition_kwargs["constant"] = left_operand
        else:
            condition_kwargs["first_signal"] = left_operand

        if isinstance(right_operand, int):
            condition_kwargs["constant"] = right_operand
        else:
            condition_kwargs["second_signal"] = right_operand

        condition = DeciderCombinator.Condition(**condition_kwargs)
        combinator.conditions = [condition]

        copy_from_input = op.output_value == "input"
        output_kwargs = {
            "signal": output_signal,
            "copy_count_from_input": copy_from_input,
        }
        if not copy_from_input:
            output_value = op.output_value if isinstance(op.output_value, int) else 1
            output_kwargs["constant"] = output_value

        output = DeciderCombinator.Output(**output_kwargs)
        combinator.outputs = [output]

        combinator = self._add_entity(combinator)

        label_candidate = (
            output_signal
            if isinstance(output_signal, str)
            else getattr(output_signal, "name", None)
        )
        usage_entry = self.signal_usage.get(op.node_id)
        debug_info = self._compose_debug_info(
            usage_entry, fallback_name=label_candidate
        )
        if debug_info:
            self.annotate_entity_description(combinator, debug_info)

        placement = EntityPlacement(
            entity=combinator,
            entity_id=op.node_id,
            position=pos,
            output_signals={output_signal: "red"},
            input_signals={},
        )
        self.entities[op.node_id] = placement
        self.signal_graph.set_source(op.node_id, op.node_id)
        self._track_signal_source(op.node_id, op.node_id)

        self._add_signal_sink(op.left, op.node_id)
        if not isinstance(op.right, int):
            self._add_signal_sink(op.right, op.node_id)

    def emit_memory_create(self, op: IR_MemCreate):
        """Emit memory module creation using simplified 2-combinator cell."""
        resolver = self._require_resolver()
        signal_type = resolver.get_signal_name(op.signal_type)

        memory_components = self.memory_builder.build_sr_latch(
            op.memory_id, signal_type
        )
        memory_components["signal_type"] = signal_type

        memory_variable = op.memory_id
        if memory_variable.startswith("mem_"):
            memory_variable = memory_variable[4:]
        source_location = (
            render_source_location(op.source_ast) if op.source_ast else None
        )

        component_labels = {
            "write_gate": "memory write gate",
            "hold_gate": "memory latch",
        }

        for component_name, placement in memory_components.items():
            if not isinstance(placement, EntityPlacement):
                continue

            self.entities[placement.entity_id] = placement

            output_signal = next(iter(placement.output_signals), None)
            debug_info = {
                "name": memory_variable,
                "label": component_labels.get(
                    component_name, component_name.replace("_", " ")
                ),
                "resolved_signal": output_signal,
                "declared_type": "Memory",
                "location": source_location,
            }
            debug_info = {k: v for k, v in debug_info.items() if v is not None}
            self.annotate_entity_description(placement.entity, debug_info)

        hold_gate = memory_components.get("hold_gate")
        if hold_gate:
            self.signal_graph.set_source(op.memory_id, hold_gate.entity_id)
            self._track_signal_source(op.memory_id, hold_gate.entity_id)

    def emit_memory_read(self, op: IR_MemRead):
        """Emit memory read operation from the 3-combinator memory cell."""
        if op.memory_id in self.memory_builder.memory_modules:
            memory_components = self.memory_builder.memory_modules[op.memory_id]
            hold_gate = memory_components.get("hold_gate")
            if hold_gate:
                self.signal_graph.set_source(op.node_id, hold_gate.entity_id)
                self._track_signal_source(op.node_id, hold_gate.entity_id)

                if not hasattr(self, "memory_read_signals"):
                    self.memory_read_signals = {}
                declared_type = memory_components.get("signal_type", op.output_type)
                self.memory_read_signals[op.node_id] = declared_type
            else:
                self.diagnostics.error(f"Memory hold gate not found in {op.memory_id}")
        else:
            self.diagnostics.error(
                f"Memory {op.memory_id} not found for read operation"
            )

    def _resolve_literal_value(self, value: ValueRef) -> Optional[int]:
        """Attempt to resolve a ValueRef to a literal integer at emit time."""

        if isinstance(value, int):
            return value

        if isinstance(value, SignalRef):
            if self.materializer:
                inlined = self.materializer.inline_value(value)
                if inlined is not None:
                    return inlined

            source_op = self._prepared_operation_index.get(value.source_id)
            if isinstance(source_op, IR_Const):
                return getattr(source_op, "value", None)

        return None

    def _determine_write_strategy(self, op: IR_MemWrite) -> str:
        """Determine the appropriate emission strategy for a memory write."""

        memory_id = op.memory_id
        if memory_id not in self.memory_builder.memory_modules:
            return "SR_LATCH"

        enable_literal = self._resolve_literal_value(op.write_enable)
        if enable_literal is not None and enable_literal != 0:
            if self._write_references_same_memory(op.data_signal, memory_id):
                return "FEEDBACK_LOOP"

        if getattr(op, "is_one_shot", False):
            return "PULSE_GATE"

        return "SR_LATCH"

    def _write_references_same_memory(
        self, value_signal: ValueRef, memory_id: str
    ) -> bool:
        """Check if a value expression ultimately depends on the same memory."""

        visited: Set[str] = set()

        def visit(value: ValueRef) -> bool:
            if isinstance(value, SignalRef):
                source_id = value.source_id
                if source_id in visited:
                    return False
                visited.add(source_id)

                source_op = self._prepared_operation_index.get(source_id)
                if isinstance(source_op, IR_MemRead):
                    return source_op.memory_id == memory_id
                if isinstance(source_op, IR_Arith):
                    return visit(source_op.left) or visit(source_op.right)
                if isinstance(source_op, IR_Decider):
                    return (
                        visit(source_op.left)
                        or visit(source_op.right)
                        or visit(source_op.output_value)
                    )
                return False

            if isinstance(value, (list, tuple)):
                for item in value:
                    if visit(item):
                        return True

            return False

        return visit(value_signal)

    def emit_memory_write(self, op: IR_MemWrite):
        """Emit memory write operation using the most suitable strategy."""
        memory_id = op.memory_id

        if memory_id not in self.memory_builder.memory_modules:
            self.diagnostics.error(f"Memory {memory_id} not found for write operation")
            return

        strategy = self._determine_write_strategy(op)

        if strategy == "FEEDBACK_LOOP":
            self._emit_feedback_loop_write(op)
            return
        if strategy == "PULSE_GATE":
            self._emit_pulse_gate_write(op)
            return

        self._emit_sr_latch_write(op)

    def _emit_feedback_loop_write(self, op: IR_MemWrite) -> None:
        """Emit optimized feedback loop for unconditional self-referential writes."""

        resolver = self._require_resolver()
        memory_id = op.memory_id
        memory_module = self.memory_builder.memory_modules.get(memory_id)
        if not memory_module:
            self._emit_sr_latch_write(op)
            return

        if not isinstance(op.data_signal, SignalRef):
            self._emit_sr_latch_write(op)
            return

        source_op = self._prepared_operation_index.get(op.data_signal.source_id)
        if not isinstance(source_op, IR_Arith):
            self._emit_sr_latch_write(op)
            return

        if not self._is_simple_feedback_candidate(source_op, memory_id):
            self._emit_sr_latch_write(op)
            return

        placement = self.entities.get(source_op.node_id)
        if placement is None:
            self._emit_sr_latch_write(op)
            return

        declared_signal = memory_module.get("signal_type")
        if not declared_signal:
            declared_signal = resolver.get_signal_name(op.data_signal.signal_type)

        placement.metadata["feedback_loop"] = True
        placement.metadata["feedback_signal"] = declared_signal
        placement.metadata["memory_id"] = memory_id

        for read_node_id in self._memory_reads_by_memory.get(memory_id, []):
            self.signal_graph.set_source(read_node_id, placement.entity_id)
            self._track_signal_source(read_node_id, placement.entity_id)

        self.signal_graph.set_source(memory_id, placement.entity_id)
        self._track_signal_source(memory_id, placement.entity_id)
        self._track_signal_source(op.node_id, placement.entity_id)

    def _is_simple_feedback_candidate(
        self, arith_op: IR_Arith, memory_id: str
    ) -> bool:
        """Return True when the arithmetic op models a direct memory increment."""

        def is_memory_read(value: ValueRef) -> bool:
            if not isinstance(value, SignalRef):
                return False
            source = self._prepared_operation_index.get(value.source_id)
            return isinstance(source, IR_MemRead) and source.memory_id == memory_id

        def is_immediate(value: ValueRef) -> bool:
            if isinstance(value, int):
                return True
            if isinstance(value, SignalRef):
                source = self._prepared_operation_index.get(value.source_id)
                return isinstance(source, IR_Const)
            return False

        if arith_op.op not in {"+", "-"}:
            return False

        left_mem = is_memory_read(arith_op.left)
        right_mem = is_memory_read(arith_op.right)

        if left_mem and not right_mem and is_immediate(arith_op.right):
            return True
        if right_mem and not left_mem and is_immediate(arith_op.left):
            return True

        return False

    def _emit_pulse_gate_write(self, op: IR_MemWrite) -> None:
        """Emit one-shot memory write. Currently delegates to SR latch implementation."""

        self._emit_sr_latch_write(op)

    def _emit_sr_latch_write(self, op: IR_MemWrite) -> None:
        """Emit traditional SR latch write for conditional writes."""

        resolver = self._require_resolver()
        memory_id = op.memory_id
        memory_module = self.memory_builder.memory_modules.get(memory_id)
        if not memory_module:
            self.diagnostics.error(
                f"Memory {memory_id} not found for write operation"
            )
            return

        write_gate_placement = memory_module.get("write_gate")
        hold_gate_placement = memory_module.get("hold_gate")

        if not (write_gate_placement and hold_gate_placement):
            self.diagnostics.error(
                f"Incomplete memory module for {memory_id}; expected write and hold gates"
            )
            return

        try:
            write_gate = write_gate_placement.entity
            hold_gate = hold_gate_placement.entity

            neighbor_positions: List[Tuple[int, int]] = []
            if write_gate_placement:
                neighbor_positions.append(write_gate_placement.position)
            if hold_gate_placement:
                neighbor_positions.append(hold_gate_placement.position)

            desired_location: Optional[Tuple[int, int]] = None
            if neighbor_positions:
                avg_x = sum(pos[0] for pos in neighbor_positions) / len(
                    neighbor_positions
                )
                avg_y = sum(pos[1] for pos in neighbor_positions) / len(
                    neighbor_positions
                )
                desired_location = (int(round(avg_x)), int(round(avg_y)))

            declared_signal = memory_module.get("signal_type")
            if not declared_signal:
                base_signal = (
                    getattr(op.data_signal, "signal_type", None)
                    if isinstance(op.data_signal, SignalRef)
                    else op.memory_id
                )
                declared_signal = resolver.get_signal_name(base_signal)

            enable_literal = self._resolve_literal_value(op.write_enable)

            enable_combinator = DeciderCombinator()
            enable_pos = self._place_entity(
                enable_combinator,
                desired=desired_location,
                max_radius=8,
            )

            if enable_literal is not None:
                comparator = "="
                constant = 0 if enable_literal != 0 else 1
                condition = DeciderCombinator.Condition(
                    first_signal="signal-0",
                    comparator=comparator,
                    constant=constant,
                )
                pulse_constant = 1 if enable_literal != 0 else 0
            else:
                enable_signal = resolver.get_signal_name(op.write_enable)
                condition = DeciderCombinator.Condition(
                    first_signal=enable_signal,
                    comparator="!=",
                    constant=0,
                )
                pulse_constant = 1

            enable_combinator.conditions = [condition]

            enable_output = DeciderCombinator.Output(
                signal="signal-W",
                copy_count_from_input=False,
                networks={"green": True},
            )
            enable_output.constant = pulse_constant
            enable_combinator.outputs.append(enable_output)
            enable_combinator = self._add_entity(enable_combinator)

            enable_entity_id = f"{memory_id}_write_enable_{self.next_entity_number}"
            self.next_entity_number += 1
            enable_placement = EntityPlacement(
                entity=enable_combinator,
                entity_id=enable_entity_id,
                position=enable_pos,
                output_signals={"signal-W": "green"},
                input_signals={},
            )
            self.entities[enable_entity_id] = enable_placement
            self.signal_graph.set_source(enable_entity_id, enable_entity_id)
            self._track_signal_source(enable_entity_id, enable_entity_id)

            if enable_literal is None:
                self._add_signal_sink(op.write_enable, enable_entity_id)

            self.blueprint.add_circuit_connection(
                "green",
                enable_combinator,
                write_gate,
                side_1="output",
                side_2="input",
            )
            self.blueprint.add_circuit_connection(
                "green",
                enable_combinator,
                hold_gate,
                side_1="output",
                side_2="input",
            )

            injector = DeciderCombinator()
            injector_pos = self._place_entity(
                injector,
                desired=desired_location,
                max_radius=8,
            )

            injector_condition = DeciderCombinator.Condition(
                first_signal="signal-W",
                comparator=">",
                constant=0,
                first_signal_networks={"green": True},
            )
            injector_output = DeciderCombinator.Output(
                signal=declared_signal,
                copy_count_from_input=True,
                networks={"red": True},
            )
            injector.conditions = [injector_condition]
            injector.outputs = [injector_output]
            injector = self._add_entity(injector)

            injector_entity_id = f"{memory_id}_write_data_{self.next_entity_number}"
            self.next_entity_number += 1
            injector_placement = EntityPlacement(
                entity=injector,
                entity_id=injector_entity_id,
                position=injector_pos,
                output_signals={declared_signal: "red"},
                input_signals={declared_signal: "red", "signal-W": "green"},
            )
            self.entities[injector_entity_id] = injector_placement
            self._track_signal_source(op.node_id, injector_entity_id)

            self._add_signal_sink(op.data_signal, injector_entity_id)

            self.blueprint.add_circuit_connection(
                "green",
                enable_combinator,
                injector,
                side_1="output",
                side_2="input",
            )

            self.blueprint.add_circuit_connection(
                "red",
                injector,
                write_gate,
                side_1="output",
                side_2="input",
            )

        except Exception as exc:
            self.diagnostics.warning(
                f"Could not configure memory write combinator for {memory_id}: {exc}"
            )

    def emit_place_entity(self, op: IR_PlaceEntity):
        """Emit entity placement using the entity factory."""
        try:
            entity = new_entity(op.prototype)
            footprint = self._entity_footprint(entity)

            if isinstance(op.x, int) and isinstance(op.y, int):
                desired = (int(op.x), int(op.y))
                max_radius = 0
                if desired in self.layout.used_positions:
                    max_radius = 4
                pos = self.layout.reserve_near(
                    desired,
                    max_radius=max_radius,
                    footprint=footprint,
                )
            else:
                pos = self.layout.get_next_position(footprint=footprint)

            entity.tile_position = pos

            if op.properties:
                for prop_name, prop_value in op.properties.items():
                    if hasattr(entity, prop_name):
                        setattr(entity, prop_name, prop_value)
                    else:
                        self.diagnostics.error(
                            f"Unknown property '{prop_name}' for entity '{op.prototype}'"
                        )

            entity = self._add_entity(entity)

            placement = EntityPlacement(
                entity=entity,
                entity_id=op.entity_id,
                position=pos,
                output_signals={},
                input_signals={},
            )
            self.entities[op.entity_id] = placement

        except ValueError as exc:
            self.diagnostics.error(f"Failed to create entity: {exc}")
        except Exception as exc:
            self.diagnostics.error(
                f"Unexpected error creating entity '{op.prototype}': {exc}"
            )

    def emit_entity_prop_write(self, op: IR_EntityPropWrite):
        """Emit entity property write (circuit network connection)."""
        resolver = self._require_resolver()
        if op.entity_id not in self.entities:
            self.diagnostics.error(
                f"Entity {op.entity_id} not found for property write"
            )
            return

        entity_placement = self.entities[op.entity_id]
        entity = entity_placement.entity
        property_key = (op.entity_id, op.property_name)
        self.entity_property_signals[property_key] = op.value

        if isinstance(op.value, int):
            if hasattr(entity, op.property_name):
                setattr(entity, op.property_name, op.value)
            else:
                self.diagnostics.warning(
                    f"Entity '{op.entity_id}' has no property '{op.property_name}' for static assignment"
                )
            return

        if isinstance(op.value, str):
            if hasattr(entity, op.property_name):
                setattr(entity, op.property_name, op.value)
            else:
                self.diagnostics.warning(
                    f"Entity '{op.entity_id}' has no property '{op.property_name}' for static assignment"
                )
            return

        signal_name = resolver.get_signal_name(op.value)
        entity_placement.input_signals[signal_name] = "red"
        self._add_signal_sink(op.value, op.entity_id)

        if op.property_name == "enable":
            try:
                try:
                    signal_id = SignalID(signal_name)
                except Exception:
                    if signal_data is not None and signal_name not in signal_data.raw:
                        signal_data.add_signal(signal_name, "virtual")
                    signal_id = SignalID(signal_name)

                if hasattr(entity, "circuit_enabled"):
                    entity.circuit_enabled = True

                if hasattr(entity, "set_circuit_condition"):
                    entity.set_circuit_condition(
                        first_operand=signal_id, comparator=">", second_operand=0
                    )
                else:
                    behavior = getattr(entity, "control_behavior", {}) or {}
                    behavior["circuit_enable_disable"] = True
                    behavior["circuit_condition"] = {
                        "first_signal": {
                            "name": signal_id.name,
                            "type": signal_id.type,
                        },
                        "comparator": ">",
                        "constant": 0,
                    }
                    entity.control_behavior = behavior
            except Exception as exc:
                self.diagnostics.warning(
                    f"Failed to configure circuit enable for entity '{op.entity_id}': {exc}"
                )
        else:
            self.diagnostics.warning(
                f"Dynamic property '{op.property_name}' not yet supported for entity '{op.entity_id}'"
            )

    def emit_entity_prop_read(self, op: IR_EntityPropRead):
        """Emit entity property read (circuit network connection)."""
        resolver = self._require_resolver()
        property_key = (op.entity_id, op.property_name)
        stored_value = self.entity_property_signals.get(property_key)

        if isinstance(stored_value, SignalRef):
            self.signal_graph.set_source(op.node_id, stored_value.source_id)
            self._track_signal_source(op.node_id, stored_value.source_id)
        elif isinstance(stored_value, int):
            combinator = ConstantCombinator()
            pos = self._place_entity(combinator)
            section = combinator.add_section()
            signal_name = resolver.get_signal_name(op.output_type)
            section.set_signal(index=0, name=signal_name, count=stored_value)
            combinator = self._add_entity(combinator)

            placement = EntityPlacement(
                entity=combinator,
                entity_id=op.node_id,
                position=pos,
                output_signals={signal_name: "red"},
                input_signals={},
            )
            self.entities[op.node_id] = placement
            self.signal_graph.set_source(op.node_id, op.node_id)
            self._track_signal_source(op.node_id, op.node_id)
        else:
            if op.entity_id in self.entities:
                self.signal_graph.set_source(op.node_id, op.entity_id)
                self._track_signal_source(op.node_id, op.entity_id)
