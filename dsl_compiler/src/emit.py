# emit.py
"""
Blueprint emission module for the Factorio Circuit DSL.

This module converts IR operations into actual Factorio combinators and entities
using the factorio-draftsman library to generate blueprint JSON.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set

# Add draftsman to path
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent / "factorio-draftsman")
)

from draftsman.blueprintable import Blueprint
from draftsman.entity import new_entity  # Use draftsman's factory
from draftsman.entity import *  # Import all entities
from draftsman.classes.entity import Entity
from draftsman.constants import Direction
from draftsman.data import entities as entity_data
from draftsman.data import signals as signal_data
from draftsman.signatures import SignalID


from .ir import *
from .dsl_ast import SignalLiteral
from .semantic import DiagnosticCollector, render_source_location
from .emission.signals import EntityPlacement, SignalUsageEntry, SignalGraph, SignalMaterializer
from .emission.memory import MemoryCircuitBuilder
from .emission.layout import LayoutEngine
from .emission.debug_format import format_entity_description


# =============================================================================
# Entity Factory using draftsman's catalog
# =============================================================================





# =============================================================================
# Memory Circuit Builder
# =============================================================================






class BlueprintEmitter:
    def allocate_shared_position(self, producer_ids: List[str]) -> Tuple[int, int]:
        """Allocate a layout position for shared producers, avoiding duplication."""
        positions = [self.entities[pid].position for pid in producer_ids if pid in self.entities]
        if positions:
            avg_x = sum(pos[0] for pos in positions) / len(positions)
            avg_y = sum(pos[1] for pos in positions) / len(positions)
            return self.layout.reserve_near((avg_x, avg_y))
        return self.layout.get_next_position()

    def get_wire_color(self, signal_type: Optional[str] = None) -> str:
        """Attach correct wire color based on signal type/category."""
        if signal_type == "memory":
            return "green"
        return "red"

    def annotate_entity_description(self, entity: Entity, debug_info: Optional[dict] = None) -> None:
        """Attach compiler metadata to the entity description without clobbering user tags."""

        if not self.enable_metadata_annotations or not debug_info:
            return

        # Ensure we have a human-friendly description string.
        description = format_entity_description(debug_info)
        if not description:
            return

        # Prefer Draftsman's player description support when available so that
        # the text shows up in Factorio's UI.
        player_description_applied = False
        if hasattr(entity, "player_description"):
            existing = getattr(entity, "player_description", "") or ""
            if not existing:
                entity.player_description = description
            elif description not in existing:
                entity.player_description = f"{existing}; {description}"[:500]
            player_description_applied = True

        container: Optional[Dict[str, Any]] = None
        if hasattr(entity, "tags"):
            if getattr(entity, "tags") is None:
                entity.tags = {}
            container = entity.tags
        elif hasattr(entity, "extra_keys"):
            if getattr(entity, "extra_keys") is None:
                entity.extra_keys = {}
            container = entity.extra_keys

        if container is None and not player_description_applied:
            return

        if container is not None:
            existing = container.get("description")
            if existing:
                existing_entries = [entry.strip() for entry in existing.split(";") if entry.strip()]
                if description in existing_entries:
                    return
                existing_entries.append(description)
                container["description"] = "; ".join(existing_entries)
            else:
                container["description"] = description
    """Converts IR operations to Factorio blueprint using Draftsman."""

    def __init__(
        self,
        signal_type_map: Dict[str, str] = None,
        *,
        enable_metadata_annotations: bool = True,
    ):
        # Persistent configuration
        self.signal_type_map = signal_type_map or {}
        self.enable_metadata_annotations = enable_metadata_annotations

        # Runtime placeholders
        self._reset_for_emit()

    def set_metadata_annotation_mode(self, enabled: bool) -> None:
        """Toggle compiler-provided metadata descriptions at runtime."""

        self.enable_metadata_annotations = enabled

    def _reset_for_emit(self) -> None:
        """Reset transient state so the emitter can be reused safely."""

        self.blueprint = Blueprint()
        self.blueprint.label = "DSL Generated Blueprint"
        self.blueprint.version = (2, 0)

        self.layout = LayoutEngine()
        self.diagnostics = DiagnosticCollector()
        self.memory_builder = MemoryCircuitBuilder(self.layout, self.blueprint)

        # Entity tracking
        self.entities = {}
        self.next_entity_number = 1

        # Signal connectivity graph
        self.signal_graph = SignalGraph()

        # Entity property signal tracking (for reads)
        self.entity_property_signals = {}

        # Signal usage/materializer cache
        self.signal_usage = {}
        self.materializer = None
        self._prepared_operations = []

    def prepare(self, ir_operations: List[IRNode]) -> None:
        """Initialize emission state and analyze the IR prior to realization."""

        self._reset_for_emit()
        self._prepared_operations = ir_operations

        self.signal_usage = self._analyze_signal_usage(ir_operations)
        self.materializer = SignalMaterializer(
            self.signal_usage, self.signal_type_map, self.diagnostics
        )
        self.materializer.finalize()

    def _extract_dependency_positions(self, value_ref) -> List[Tuple[int, int]]:
        """Find positions of entities that feed a value reference."""
        positions: List[Tuple[int, int]] = []

        if isinstance(value_ref, SignalRef):
            placement = self.entities.get(value_ref.source_id)
            if placement:
                positions.append(placement.position)
        elif isinstance(value_ref, str):
            placement = self.entities.get(value_ref)
            if placement:
                positions.append(placement.position)

        return positions

    def _allocate_position(self, *value_refs) -> Tuple[int, int]:
        """Allocate a placement near the average of dependency positions if possible."""
        positions: List[Tuple[int, int]] = []
        for ref in value_refs:
            positions.extend(self._extract_dependency_positions(ref))

        if positions:
            avg_x = sum(pos[0] for pos in positions) / len(positions)
            avg_y = sum(pos[1] for pos in positions) / len(positions)
            return self.layout.reserve_near((avg_x, avg_y))

        return self.layout.get_next_position()

    def _add_entity(self, entity: Entity) -> Entity:
        """Add an entity to the blueprint and return the stored reference."""
        return self.blueprint.entities.append(entity, copy=False)

    def _compose_debug_info(
        self, usage_entry: Optional[SignalUsageEntry], fallback_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Create a debug info dictionary from signal usage metadata."""

        if usage_entry is None and fallback_name is None:
            return None

        debug_info: Dict[str, Any] = {}

        if usage_entry is not None:
            metadata = getattr(usage_entry, "debug_metadata", {}) or {}
            debug_info.update(metadata)

            if usage_entry.debug_label and "label" not in debug_info:
                debug_info["label"] = usage_entry.debug_label

            if "name" not in debug_info:
                debug_info["name"] = (
                    metadata.get("name")
                    or usage_entry.debug_label
                    or fallback_name
                )

            if usage_entry.resolved_signal_name and "factorio_signal" not in debug_info:
                debug_info["factorio_signal"] = usage_entry.resolved_signal_name

            if usage_entry.resolved_signal_type and "category" not in debug_info:
                debug_info["category"] = usage_entry.resolved_signal_type

            if "location" not in debug_info and usage_entry.source_ast:
                location = render_source_location(usage_entry.source_ast)
                if location:
                    debug_info["location"] = location

            if usage_entry.source_ast is not None:
                debug_info.setdefault("source_ast", usage_entry.source_ast)

        elif fallback_name is not None:
            debug_info["name"] = fallback_name

        return debug_info or None

    def _analyze_signal_usage(self, ir_operations: List[IRNode]) -> Dict[str, SignalUsageEntry]:
        """Construct a signal usage index from the IR prior to emission."""

        usage: Dict[str, SignalUsageEntry] = {}

        def ensure_entry(signal_id: str, signal_type: Optional[str] = None) -> SignalUsageEntry:
            entry = usage.get(signal_id)
            if entry is None:
                entry = SignalUsageEntry(signal_id=signal_id)
                usage[signal_id] = entry
            if signal_type and not entry.signal_type:
                entry.signal_type = signal_type
            return entry

        def record_consumer(ref: Any, consumer_id: str) -> None:
            if isinstance(ref, SignalRef):
                entry = ensure_entry(ref.source_id, ref.signal_type)
                entry.consumers.add(consumer_id)
            elif isinstance(ref, (list, tuple)):
                for item in ref:
                    record_consumer(item, consumer_id)

        def record_export(ref: Any, export_label: str) -> None:
            if isinstance(ref, SignalRef):
                entry = ensure_entry(ref.source_id, ref.signal_type)
                entry.export_targets.add(export_label)

        for op in ir_operations:
            if isinstance(op, IRValue):
                entry = ensure_entry(op.node_id, getattr(op, "output_type", None))
                entry.producer = op
                if op.source_ast and not entry.source_ast:
                    entry.source_ast = op.source_ast
                if getattr(op, "debug_label", None):
                    entry.debug_label = op.debug_label
                elif not entry.debug_label:
                    entry.debug_label = op.node_id
                if getattr(op, "debug_metadata", None):
                    entry.debug_metadata.update(op.debug_metadata)

                if isinstance(op, IR_Const):
                    entry.literal_value = op.value
                    if isinstance(op.source_ast, SignalLiteral):
                        declared_type = getattr(op.source_ast, "signal_type", None)
                        if declared_type:
                            entry.is_typed_literal = True
                            entry.literal_declared_type = declared_type
                        else:
                            entry.literal_declared_type = getattr(op, "output_type", None)

            if isinstance(op, IR_Arith):
                record_consumer(op.left, op.node_id)
                record_consumer(op.right, op.node_id)
            elif isinstance(op, IR_Decider):
                record_consumer(op.left, op.node_id)
                record_consumer(op.right, op.node_id)
                record_consumer(op.output_value, op.node_id)
            elif isinstance(op, IR_MemCreate):
                record_consumer(op.initial_value, op.node_id)
            elif isinstance(op, IR_MemWrite):
                record_consumer(op.data_signal, op.node_id)
                record_consumer(op.write_enable, op.node_id)
            elif isinstance(op, IR_PlaceEntity):
                record_consumer(op.x, op.node_id)
                record_consumer(op.y, op.node_id)
                for prop_value in op.properties.values():
                    record_consumer(prop_value, op.node_id)
            elif isinstance(op, IR_EntityPropWrite):
                record_consumer(op.value, op.node_id)
                record_export(op.value, f"entity:{op.entity_id}.{op.property_name}")
            elif isinstance(op, IR_ConnectToWire):
                record_export(op.signal, f"wire:{op.channel}")

        return usage

    def _ensure_export_anchors(self):
        """Create anchor entities for externally visible or dangling signals."""

        for signal_id, usage_entry in self.signal_usage.items():
            has_external_targets = bool(usage_entry.export_targets)
            is_dangling_output = (
                not usage_entry.export_targets
                and usage_entry.producer is not None
                and not usage_entry.consumers
                and not isinstance(usage_entry.producer, IR_Const)
            )

            if not (has_external_targets or is_dangling_output):
                continue
            if usage_entry.export_anchor_id:
                continue
            self._create_export_anchor(signal_id, usage_entry)

    def _create_export_anchor(self, signal_id: str, usage_entry: SignalUsageEntry):
        anchor_id = f"{signal_id}_export_anchor"
        if anchor_id in self.entities:
            usage_entry.export_anchor_id = anchor_id
            return

        pos = self.layout.get_next_position()
        combinator = new_entity('constant-combinator', tile_position=pos)
        self._add_entity(combinator)
        placement = EntityPlacement(
            entity=combinator,
            entity_id=anchor_id,
            position=pos,
            output_signals={},
            input_signals={},
        )
        self.entities[anchor_id] = placement
        usage_entry.export_anchor_id = anchor_id

        debug_info = self._compose_debug_info(usage_entry)
        if debug_info:
            self.annotate_entity_description(combinator, debug_info)

        # Register as sink so standard wiring connects it to the producer
        self.signal_graph.add_sink(signal_id, anchor_id)

    def emit_blueprint(self, ir_operations: List[IRNode]) -> Blueprint:
        """Convert IR operations to blueprint."""
        import warnings

        # Preserve caller-provided blueprint metadata across resets
        previous_label = getattr(self.blueprint, "label", "DSL Generated Blueprint")

        # Initialize state and perform IR pre-analysis
        self.prepare(ir_operations)
        self.blueprint.label = previous_label

        # Collect draftsman warnings during blueprint construction
        captured_warnings = []
        
        def warning_handler(message, category, filename, lineno, file=None, line=None):
            captured_warnings.append((message, category, filename, lineno))
        
        # Set up warning capture
        old_showwarning = warnings.showwarning
        warnings.showwarning = warning_handler
        
        try:
            # Process all IR operations
            for op in self._prepared_operations:
                self.emit_ir_operation(op)

            # Create export anchors for signals with external targets but no consumers
            self._ensure_export_anchors()

            # Add wiring connections
            self.create_circuit_connections()
            
            # Validate the final blueprint and process captured warnings
            self._validate_blueprint_with_warnings(captured_warnings)

            return self.blueprint

        except Exception as e:
            self.diagnostics.error(f"Blueprint emission failed: {e}")
            raise
        finally:
            # Restore original warning handler
            warnings.showwarning = old_showwarning

    def _validate_blueprint_with_warnings(self, captured_warnings):
        """Perform validation on the generated blueprint and process draftsman warnings."""
        
        # Process captured draftsman warnings
        critical_warning_types = {
            'OverlappingObjectsWarning',
            'UnknownEntityWarning', 
            'InvalidEntityError',
            'InvalidConnectorError'
        }
        
        for message, category, filename, lineno in captured_warnings:
            warning_name = category.__name__
            
            # Convert critical warnings to errors
            if warning_name in critical_warning_types:
                self.diagnostics.error(f"Blueprint validation failed - {warning_name}: {message}")
            else:
                # Keep other warnings as warnings
                self.diagnostics.warning(f"{warning_name}: {message}")
        
        # Basic blueprint validation
        if len(self.blueprint.entities) == 0:
            self.diagnostics.warning("Generated blueprint is empty")
            
        # Check for entities with invalid positions
        for entity in self.blueprint.entities:
            if hasattr(entity, 'tile_position'):
                x, y = entity.tile_position
                if x < -1000 or x > 1000 or y < -1000 or y > 1000:
                    self.diagnostics.warning(f"Entity at extreme position: ({x}, {y})")
        
        # Count entity types for debugging
        entity_counts = {}
        for entity in self.blueprint.entities:
            entity_type = type(entity).__name__
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        self.diagnostics.info(f"Generated blueprint with {len(self.blueprint.entities)} entities: {entity_counts}")

    def emit_ir_operation(self, op: IRNode):
        """Emit a single IR operation, consulting signal usage index for materialization rules."""
        usage_entry = self.signal_usage.get(getattr(op, 'node_id', None))
        if isinstance(op, IR_Const):
            # Only materialize if required by usage index
            if self.materializer and usage_entry and not usage_entry.should_materialize:
                return
            self.emit_constant(op)
        elif isinstance(op, IR_Arith):
            self.emit_arithmetic(op)
        elif isinstance(op, IR_Decider):
            self.emit_decider(op)
        elif isinstance(op, IR_MemCreate):
            self.emit_memory_create(op)
        elif isinstance(op, IR_MemRead):
            self.emit_memory_read(op)
        elif isinstance(op, IR_MemWrite):
            self.emit_memory_write(op)
        elif isinstance(op, IR_PlaceEntity):
            self.emit_place_entity(op)
        elif isinstance(op, IR_EntityPropWrite):
            self.emit_entity_prop_write(op)
        elif isinstance(op, IR_EntityPropRead):
            self.emit_entity_prop_read(op)
        else:
            self.diagnostics.error(f"Unknown IR operation: {type(op)}")

    def emit_constant(self, op: IR_Const):
        """Emit constant combinator for IR_Const, only if materialization is required."""
        usage_entry = self.signal_usage.get(op.node_id)
        if self.materializer and usage_entry and not usage_entry.should_materialize:
            print(f"SKIP constant: {op.node_id} (should_materialize={usage_entry.should_materialize}, is_typed_literal={getattr(usage_entry, 'is_typed_literal', None)})")
            return
        pos = self.layout.get_next_position()
        combinator = new_entity('constant-combinator', tile_position=pos)
        section = combinator.add_section()
        # For explicit typed literals, always use the declared type as the signal name
        if usage_entry and usage_entry.is_typed_literal:
            base_signal_key = usage_entry.literal_declared_type or op.output_type
            # Resolve the physical signal name, honoring implicit mappings
            if self.materializer:
                signal_name = self.materializer.resolve_signal_name(
                    base_signal_key, usage_entry
                )
            else:
                signal_name = self._get_signal_name(base_signal_key)
            # Resolve the correct Factorio signal category (item/virtual/fluid)
            signal_type = (
                usage_entry.resolved_signal_type
                or (
                    self.materializer.resolve_signal_type(op.output_type, usage_entry)
                    if self.materializer
                    else None
                )
            )
            if not signal_type:
                if signal_name in self.signal_type_map:
                    mapped = self.signal_type_map[signal_name]
                    if isinstance(mapped, dict):
                        signal_type = mapped.get("type")
                if not signal_type:
                    if signal_data is not None and signal_name in signal_data.raw:
                        proto_type = signal_data.raw[signal_name].get("type", "virtual")
                        signal_type = "virtual" if proto_type == "virtual-signal" else proto_type
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
                else self._get_signal_name(op.output_type)
            )
            signal_type = (
                usage_entry.resolved_signal_type if usage_entry else None
            )
            value = op.value
        print(f"EMIT constant: {op.node_id} name={signal_name} type={signal_type} value={value}")
        # For explicit typed literals, emit only if op.value is nonzero and matches literal_value, skip anchor combinators
        if usage_entry and usage_entry.is_typed_literal:
            if value == 0:
                print(
                    f"EMIT anchor for explicit typed literal: {op.node_id} (value=0, name={signal_name})"
                )
            elif op.value != usage_entry.literal_value:
                print(f"SKIP non-literal for explicit typed literal: {op.node_id} (value={op.value}, expected={usage_entry.literal_value}, name={signal_name})")
                return
            print(f"DEBUG explicit typed literal: node_id={op.node_id} declared_type={usage_entry.literal_declared_type} signal_name={signal_name} output_type={op.output_type}")
            print(f"EMIT explicit typed literal: {op.node_id} name={signal_name} value={value}")
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
        )
        self.entities[op.node_id] = placement
        self.signal_graph.set_source(op.node_id, op.node_id)
        # Debug: Print filters for this constant combinator
        for sec in getattr(combinator, "sections", []):
            print(f"DEBUG combinator {op.node_id} filters: {getattr(sec, 'filters', None)}")

    def emit_arithmetic(self, op: IR_Arith):
        """Emit arithmetic combinator for IR_Arith."""

        pos = self._allocate_position(op.left, op.right)

        combinator = new_entity('arithmetic-combinator', tile_position=pos)

        # Configure arithmetic operation with proper signal handling
        left_operand = self._get_operand_for_combinator(op.left)
        right_operand = self._get_operand_for_combinator(op.right)
        output_signal = (
            self.materializer.resolve_signal_name(
                op.output_type, self.signal_usage.get(op.node_id)
            )
            if self.materializer
            else self._get_signal_name(op.output_type)
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
        debug_info = self._compose_debug_info(usage_entry, fallback_name=label_candidate)
        if debug_info:
            self.annotate_entity_description(combinator, debug_info)

        # Track entity for wiring
        placement = EntityPlacement(
            entity=combinator,
            entity_id=op.node_id,
            position=pos,
            output_signals={output_signal: "red"},
            input_signals={},  # Will be populated when wiring
        )
        self.entities[op.node_id] = placement
        self.signal_graph.set_source(op.node_id, op.node_id)

        # Track signal dependencies
        self._add_signal_sink(op.left, op.node_id)
        self._add_signal_sink(op.right, op.node_id)

    def emit_decider(self, op: IR_Decider):
        """Emit decider combinator for IR_Decider."""
        pos = self._allocate_position(op.left, op.right)

        combinator = new_entity('decider-combinator', tile_position=pos)

        # Configure decider operation
        left_operand = self._get_operand_for_combinator(op.left)
        right_operand = self._get_operand_value(op.right)
        output_signal = (
            self.materializer.resolve_signal_name(
                op.output_type, self.signal_usage.get(op.node_id)
            )
            if self.materializer
            else self._get_signal_name(op.output_type)
        )
        
        combinator.first_operand = left_operand
        combinator.second_operand = right_operand
        combinator.comparator = op.test_op
        combinator.output_signal = output_signal
        combinator.copy_count_from_input = op.output_value == "input"
        if not combinator.copy_count_from_input:
            combinator.constant = op.output_value if isinstance(op.output_value, int) else 1

        combinator = self._add_entity(combinator)

        label_candidate = (
            output_signal
            if isinstance(output_signal, str)
            else getattr(output_signal, "name", None)
        )
        usage_entry = self.signal_usage.get(op.node_id)
        debug_info = self._compose_debug_info(usage_entry, fallback_name=label_candidate)
        if debug_info:
            self.annotate_entity_description(combinator, debug_info)

        # Track entity for wiring
        placement = EntityPlacement(
            entity=combinator,
            entity_id=op.node_id,
            position=pos,
            output_signals={output_signal: "red"},
            input_signals={},
        )
        self.entities[op.node_id] = placement
        self.signal_graph.set_source(op.node_id, op.node_id)

        # Track signal dependencies
        self._add_signal_sink(op.left, op.node_id)
        if not isinstance(op.right, int):  # Signal reference
            self._add_signal_sink(op.right, op.node_id)

    def emit_memory_create(self, op: IR_MemCreate):
        """Emit memory module creation using SR latch circuit."""
        # Get initial value
        initial_value = 0
        if isinstance(op.initial_value, int):
            initial_value = op.initial_value
        elif isinstance(op.initial_value, SignalRef):
            source_id = op.initial_value.source_id
            placement = self.entities.get(source_id)
            if placement and hasattr(placement.entity, "sections"):
                try:
                    sections = placement.entity.sections
                    if sections:
                        signals = sections[0].signals
                        if signals:
                            initial_value = signals[0].get("count", 0)
                except Exception:
                    self.diagnostics.warning(
                        f"Unable to extract initial value for memory {op.memory_id} from source {source_id}; defaulting to 0"
                    )
        elif hasattr(op.initial_value, 'value'):
            initial_value = getattr(op.initial_value, 'value', 0)
        
        # Validate and resolve signal type
        signal_type = self._get_signal_name(op.signal_type)

        # Build the SR latch circuit
        memory_components = self.memory_builder.build_sr_latch(
            op.memory_id, signal_type, initial_value
        )
        memory_components['signal_type'] = signal_type
        
        # Register all components as entities
        for component_name, placement in memory_components.items():
            if isinstance(placement, EntityPlacement):
                self.entities[placement.entity_id] = placement
        
        # The converter is the main interface for this memory
        converter_placement = memory_components.get('output_converter')
        if converter_placement:
            self.signal_graph.set_source(op.memory_id, converter_placement.entity_id)

    def emit_memory_read(self, op: IR_MemRead):
        """Emit memory read operation from the 3-combinator memory cell."""
        # Memory read connects to the output of the memory combinator
        if op.memory_id in self.memory_builder.memory_modules:
            memory_components = self.memory_builder.memory_modules[op.memory_id]
            converter = memory_components.get('output_converter')
            if converter:
                self.signal_graph.set_source(op.node_id, converter.entity_id)

                if not hasattr(self, 'memory_read_signals'):
                    self.memory_read_signals = {}
                declared_type = memory_components.get('signal_type', op.output_type)
                self.memory_read_signals[op.node_id] = declared_type
            else:
                self.diagnostics.error(f"Memory converter not found in {op.memory_id}")
        else:
            self.diagnostics.error(f"Memory {op.memory_id} not found for read operation")

    def emit_memory_write(self, op: IR_MemWrite):
        """Emit memory write operation to the 3-combinator memory cell."""
        memory_id = op.memory_id
        
        if memory_id not in self.memory_builder.memory_modules:
            self.diagnostics.error(f"Memory {memory_id} not found for write operation")
            return
        
        # Get the memory module components
        memory_module = self.memory_builder.memory_modules[memory_id]
        input_combinator_placement = memory_module['input_combinator']
        output_combinator_placement = memory_module['output_combinator']

        from draftsman.entity import ArithmeticCombinator, DeciderCombinator

        try:
            # Data writer: normalizes incoming signal to the memory's declared channel
            data_signal_name = memory_module.get('signal_type', self._get_signal_name(op.data_signal))
            write_pos = self.layout.get_next_position()
            data_combinator = ArithmeticCombinator(tile_position=write_pos)
            data_combinator.first_operand = self._get_operand_for_combinator(op.data_signal)
            data_combinator.second_operand = 1
            data_combinator.operation = "*"
            data_combinator.output_signal = data_signal_name
            data_combinator = self._add_entity(data_combinator)

            data_entity_id = f"{memory_id}_write_data_{self.next_entity_number}"
            self.next_entity_number += 1
            data_placement = EntityPlacement(
                entity=data_combinator,
                entity_id=data_entity_id,
                position=write_pos,
                output_signals={data_signal_name: "green"},
                input_signals={},
            )
            self.entities[data_entity_id] = data_placement
            self.signal_graph.set_source(data_entity_id, data_entity_id)

            # Ensure upstream signals connect to the writer
            self._add_signal_sink(op.data_signal, data_entity_id)

            # Connect writer output to memory components
            input_combinator = input_combinator_placement.entity
            output_combinator = output_combinator_placement.entity

            self.blueprint.add_circuit_connection(
                "green",
                data_combinator,
                input_combinator,
                side_1="output",
                side_2="input",
            )
            self.blueprint.add_circuit_connection(
                "red",
                data_combinator,
                output_combinator,
                side_1="output",
                side_2="input",
            )

            # Enable writer: produce signal-R pulse based on write enable expression
            enable_pos = self.layout.get_next_position()
            enable_combinator = DeciderCombinator(tile_position=enable_pos)
            enable_signal = self._get_signal_name(op.write_enable)
            condition = DeciderCombinator.Condition(
                first_signal=enable_signal,
                comparator="!=",
                constant=0,
            )
            enable_combinator.conditions = [condition]

            enable_output = DeciderCombinator.Output(
                signal="signal-R",
                copy_count_from_input=False,
                constant=1,
            )
            enable_combinator.outputs.append(enable_output)
            enable_combinator = self._add_entity(enable_combinator)

            enable_entity_id = f"{memory_id}_write_enable_{self.next_entity_number}"
            self.next_entity_number += 1
            enable_placement = EntityPlacement(
                entity=enable_combinator,
                entity_id=enable_entity_id,
                position=enable_pos,
                output_signals={"signal-R": "red"},
                input_signals={},
            )
            self.entities[enable_entity_id] = enable_placement
            self.signal_graph.set_source(enable_entity_id, enable_entity_id)

            self._add_signal_sink(op.write_enable, enable_entity_id)

            self.blueprint.add_circuit_connection(
                "red",
                enable_combinator,
                input_combinator,
                side_1="output",
                side_2="input",
            )

        except Exception as e:
            self.diagnostics.warning(
                f"Could not configure memory write combinator for {memory_id}: {e}"
            )

    def emit_place_entity(self, op: IR_PlaceEntity):
        """Emit entity placement using the entity factory."""
        # Handle both constant and variable coordinates
        if isinstance(op.x, int) and isinstance(op.y, int):
            # Use specified position with offset to avoid combinators
            base_x = op.x + 20  # Offset entities to the right of combinators
            base_y = op.y
            pos = (int(base_x), int(base_y))  # Ensure grid alignment
            
            # Mark this position as used in the layout manager to prevent overlaps
            self.layout.used_positions.add(pos)
        else:
            # Default to layout engine for variable coordinates or fallback
            pos = self.layout.get_next_position()

        try:
            entity = new_entity(op.prototype, tile_position=pos)
            
            # Apply any additional properties
            if op.properties:
                for prop_name, prop_value in op.properties.items():
                    if hasattr(entity, prop_name):
                        setattr(entity, prop_name, prop_value)
                    else:
                        self.diagnostics.error(f"Unknown property '{prop_name}' for entity '{op.prototype}'")

            entity = self._add_entity(entity)

            placement = EntityPlacement(
                entity=entity,
                entity_id=op.entity_id,
                position=pos,
                output_signals={},
                input_signals={},
            )
            self.entities[op.entity_id] = placement
            
        except ValueError as e:
            self.diagnostics.error(f"Failed to create entity: {e}")
        except Exception as e:
            self.diagnostics.error(f"Unexpected error creating entity '{op.prototype}': {e}")

    def emit_entity_prop_write(self, op: IR_EntityPropWrite):
        """Emit entity property write (circuit network connection)."""
        if op.entity_id not in self.entities:
            self.diagnostics.error(f"Entity {op.entity_id} not found for property write")
            return

        entity_placement = self.entities[op.entity_id]
        entity = entity_placement.entity
        property_key = (op.entity_id, op.property_name)
        self.entity_property_signals[property_key] = op.value

        # Static assignments (ints/strings) are applied directly when possible
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

        signal_name = self._get_signal_name(op.value)
        entity_placement.input_signals[signal_name] = "red"
        self._add_signal_sink(op.value, op.entity_id)

        # Only enable/disable is currently supported dynamically
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
                    entity.set_circuit_condition(first_operand=signal_id, comparator=">", second_operand=0)
                else:
                    behavior = getattr(entity, "control_behavior", {}) or {}
                    behavior["circuit_enable_disable"] = True
                    behavior["circuit_condition"] = {
                        "first_signal": {"name": signal_id.name, "type": signal_id.type},
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
        property_key = (op.entity_id, op.property_name)
        stored_value = self.entity_property_signals.get(property_key)

        if isinstance(stored_value, SignalRef):
            self.signal_graph.set_source(op.node_id, stored_value.source_id)
        elif isinstance(stored_value, int):
            from draftsman.entity import ConstantCombinator

            pos = self.layout.get_next_position()
            combinator = ConstantCombinator(tile_position=pos)
            section = combinator.add_section()
            signal_name = self._get_signal_name(op.output_type)
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
        else:
            # Fallback: expose controlling entity output directly
            if op.entity_id in self.entities:
                self.signal_graph.set_source(op.node_id, op.entity_id)

    def create_circuit_connections(self):
        """Create circuit wire connections between entities."""
        # First, wire up all memory circuits
        for memory_id in self.memory_builder.memory_modules:
            try:
                self.memory_builder.wire_sr_latch(memory_id)
            except Exception as e:
                self.diagnostics.error(f"Failed to wire memory {memory_id}: {e}")

        # Create connections between signal sources and sinks
        for signal_id, source_entity_id, sink_entities in self.signal_graph.iter_edges():
            if source_entity_id:
                if source_entity_id in self.entities:
                    source_placement = self.entities[source_entity_id]

                    for sink_entity_id in sink_entities:
                        if sink_entity_id in self.entities:
                            sink_placement = self.entities[sink_entity_id]

                            # Determine wire color based on signal type
                            wire_color = self._get_wire_color(source_placement, sink_placement)

                            # Determine connection sides based on entity capabilities
                            source_entity = source_placement.entity
                            sink_entity = sink_placement.entity
                            source_dual = getattr(
                                source_entity, "dual_circuit_connectable", False
                            )
                            sink_dual = getattr(
                                sink_entity, "dual_circuit_connectable", False
                            )

                            connection_kwargs: Dict[str, Any] = dict(
                                color=wire_color,
                                entity_1=source_entity,
                                entity_2=sink_entity,
                            )

                            if source_dual:
                                connection_kwargs["side_1"] = "output"
                            if sink_dual:
                                connection_kwargs["side_2"] = "input"

                            # Add circuit connection
                            try:
                                self.blueprint.add_circuit_connection(**connection_kwargs)
                            except Exception as e:
                                self.diagnostics.error(
                                    f"Failed to connect {source_entity_id} -> {sink_entity_id}: {e}"
                                )
                else:
                    self.diagnostics.error(
                        f"Source entity {source_entity_id} not found for signal {signal_id}"
                    )
            else:
                self.diagnostics.error(f"No source found for signal {signal_id}")

    def _get_wire_color(self, source: EntityPlacement, sink: EntityPlacement) -> str:
        """Determine appropriate wire color for connection."""
        # Use red as default, green for memory outputs to avoid conflicts
        if "memory" in source.entity_id and "output" in source.entity_id:
            return "green"
        return "red"

    def _get_signal_name(self, operand) -> str:
        """Get signal name from operand - much simpler approach."""
        if self.materializer:
            entry = None
            operand_key = operand
            if isinstance(operand, SignalRef):
                entry = self.signal_usage.get(operand.source_id)
                operand_key = operand.signal_type
            if isinstance(operand_key, str):
                return self.materializer.resolve_signal_name(operand_key, entry)

        # Handle SignalRef objects
        if hasattr(operand, 'signal_type'):
            operand_str = operand.signal_type
        else:
            operand_str = str(operand)

        # Clean up signal reference strings
        clean_name = operand_str.split("@")[0]  # Remove anything after @

        # Handle integer constants
        if isinstance(operand, int):
            return "signal-0"

        # Handle bundle references
        if clean_name.startswith("Bundle["):
            return "signal-each"  # Special signal for bundle operations

        # Check signal mapping first
        if clean_name in self.signal_type_map:
            mapped_signal = self.signal_type_map[clean_name]
            if isinstance(mapped_signal, dict):
                signal_name = mapped_signal.get("name", clean_name)
                signal_type = mapped_signal.get("type", "virtual")
                if signal_data is not None and signal_name not in signal_data.raw:
                    try:
                        signal_data.add_signal(signal_name, signal_type)
                    except Exception as e:
                        self.diagnostics.warning(
                            f"Could not register custom signal '{signal_name}': {e}"
                        )
                return signal_name
            if signal_data is not None and mapped_signal not in signal_data.raw:
                try:
                    inferred_type = "virtual"
                    signal_data.add_signal(mapped_signal, inferred_type)
                except Exception as e:
                    self.diagnostics.warning(
                        f"Could not register signal '{mapped_signal}' as virtual: {e}"
                    )
            return mapped_signal

        if signal_data is not None and clean_name in signal_data.raw:
            return clean_name

        # Handle implicit signals (__v1, __v2, etc.) by mapping to virtual signals
        if clean_name.startswith("__v"):
            try:
                num = int(clean_name[3:])
                # Map to signal-A through signal-Z, cycling if needed
                letter_index = (num - 1) % 26
                virtual_signal = f"signal-{chr(ord('A') + letter_index)}"
                return virtual_signal
            except ValueError:
                pass

        # Default to registering as virtual signal for custom identifiers
        if signal_data is not None and clean_name not in signal_data.raw:
            try:
                signal_data.add_signal(clean_name, "virtual")
            except Exception as e:
                self.diagnostics.warning(
                    f"Could not register signal '{clean_name}' as virtual: {e}"
                )
        return clean_name

    def _get_operand_for_combinator(self, operand) -> Union[str, int]:
        """Get operand for arithmetic/decider combinator (can be signal name or constant)."""
        if isinstance(operand, int):
            return operand
        elif isinstance(operand, SignalRef):
            if self.materializer:
                inlined = self.materializer.inline_value(operand)
                if inlined is not None:
                    return inlined
                usage_entry = self.signal_usage.get(operand.source_id)
                return self.materializer.resolve_signal_name(
                    operand.signal_type, usage_entry
                )
            return self._get_signal_name(operand.signal_type)
        elif isinstance(operand, str):
            return self._get_signal_name(operand)
        else:
            return self._get_signal_name(str(operand))

    def _get_operand_value(self, operand):
        """Get operand value for decider combinator."""
        if isinstance(operand, int):
            return operand
        elif isinstance(operand, str):
            return self._get_signal_name(operand)
        else:
            return str(operand)

    def _add_signal_sink(self, signal_ref, entity_id: str):
        """Track that an entity needs this signal as input."""
        signal_ids: List[str] = []

        if isinstance(signal_ref, SignalRef):
            if self.materializer and self.materializer.can_inline_constant(signal_ref):
                return
            signal_ids.append(signal_ref.source_id)
        elif isinstance(signal_ref, (list, tuple)):
            for item in signal_ref:
                self._add_signal_sink(item, entity_id)
            return
        elif isinstance(signal_ref, dict):
            for item in signal_ref.values():
                self._add_signal_sink(item, entity_id)
            return
        elif isinstance(signal_ref, str):
            signal_ids.append(signal_ref)

        for signal_id in signal_ids:
            self.signal_graph.add_sink(signal_id, entity_id)


# =============================================================================
# Public API
# =============================================================================


def emit_blueprint(
    ir_operations: List[IRNode],
    label: str = "DSL Generated",
    signal_type_map: Dict[str, str] = None,
) -> Tuple[Blueprint, DiagnosticCollector]:
    """Convert IR operations to Factorio blueprint."""
    emitter = BlueprintEmitter(signal_type_map)
    emitter.blueprint.label = label

    try:
        blueprint = emitter.emit_blueprint(ir_operations)
        return blueprint, emitter.diagnostics
    except Exception as e:
        emitter.diagnostics.error(f"Blueprint emission failed: {e}")
        return emitter.blueprint, emitter.diagnostics


def emit_blueprint_string(
    ir_operations: List[IRNode],
    label: str = "DSL Generated",
    signal_type_map: Dict[str, str] = None,
) -> Tuple[str, DiagnosticCollector]:
    """Convert IR operations to Factorio blueprint string."""
    blueprint, diagnostics = emit_blueprint(ir_operations, label, signal_type_map)

    try:
        blueprint_string = blueprint.to_string()
        return blueprint_string, diagnostics
    except Exception as e:
        diagnostics.error(f"Blueprint string generation failed: {e}")
        return "", diagnostics
