# emit.py
"""
Blueprint emission module for the Factorio Circuit DSL.

This module converts IR operations into actual Factorio combinators and entities
using the factorio-draftsman library to generate blueprint JSON.
"""

import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set, Literal

# Add draftsman to path
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent / "factorio-draftsman")
)

from draftsman.blueprintable import Blueprint  # type: ignore[import-not-found]
from draftsman.entity import new_entity  # Use draftsman's factory  # type: ignore[import-not-found]
from draftsman.entity import *  # Import all entities  # type: ignore[import-not-found]
from draftsman.entity import (  # type: ignore[import-not-found]
    DeciderCombinator,
    ConstantCombinator,
)
from draftsman.classes.entity import Entity  # type: ignore[import-not-found]
from draftsman.data import signals as signal_data  # type: ignore[import-not-found]
from draftsman.signatures import SignalID  # type: ignore[import-not-found]


from .ir import *
from .dsl_ast import SignalLiteral
from .semantic import DiagnosticCollector, render_source_location
from .emission.signals import (
    EntityPlacement,
    SignalUsageEntry,
    SignalGraph,
    SignalMaterializer,
)
from .emission.wiring import (
    WIRE_COLORS,
    collect_circuit_edges,
    plan_wire_colors,
)
MAX_CIRCUIT_WIRE_SPAN = 9.0
WIRE_RELAY_ENTITY = "medium-electric-pole"
EDGE_LAYOUT_NOTE = (
    "Edge layout: literal constants are placed along the north boundary; "
    "export anchors align along the south boundary."
)


@dataclass(frozen=True)
class WireRelayOptions:
    """Configuration flags controlling automatic wire relay insertion."""

    enabled: bool = True
    max_span: float = MAX_CIRCUIT_WIRE_SPAN
    placement_strategy: Literal["euclidean", "manhattan"] = "euclidean"
    max_relays: Optional[int] = None

    def normalized_span(self) -> float:
        span = self.max_span if self.max_span and self.max_span > 0 else MAX_CIRCUIT_WIRE_SPAN
        return max(1.0, float(span))
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
        positions = [
            self.entities[pid].position for pid in producer_ids if pid in self.entities
        ]
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

    def annotate_entity_description(
        self, entity: Entity, debug_info: Optional[dict] = None
    ) -> None:
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
                existing_entries = [
                    entry.strip() for entry in existing.split(";") if entry.strip()
                ]
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
        wire_relay_options: Optional[WireRelayOptions] = None,
    ):
        # Persistent configuration
        self.signal_type_map = signal_type_map or {}
        self.enable_metadata_annotations = enable_metadata_annotations
        options = wire_relay_options or WireRelayOptions()
        if options.placement_strategy not in {"euclidean", "manhattan"}:
            options = replace(options, placement_strategy="euclidean")
        if options.max_span is not None and options.max_span <= 0:
            options = replace(options, max_span=MAX_CIRCUIT_WIRE_SPAN)

        # Clone options to decouple from caller mutations
        self.wire_relay_options = replace(options)

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
        self._circuit_edges = []
        self._node_color_assignments = {}
        self._edge_color_map = {}
        self._coloring_conflicts = []
        self._coloring_success = True
        self._wire_relay_counter = 0

        self._ensure_signal_map_registered()

    def _ensure_signal_map_registered(self) -> None:
        """Pre-register implicit signal aliases with Draftsman."""

        if signal_data is None:
            return

        for entry in self.signal_type_map.values():
            if isinstance(entry, dict):
                name = entry.get("name")
                signal_type = entry.get("type") or "virtual"
            else:
                name = entry
                signal_type = "virtual"

            if not name:
                continue

            existing = signal_data.raw.get(name)
            if existing is not None:
                continue

            try:
                signal_data.add_signal(name, signal_type)
            except Exception as exc:
                self.diagnostics.warning(
                    f"Could not pre-register signal '{name}' as {signal_type}: {exc}"
                )

    def _entity_footprint(self, entity: Entity) -> Tuple[int, int]:
        """Return the integer tile footprint for a draftsman entity."""

        width = getattr(entity, "tile_width", 1) or 1
        height = getattr(entity, "tile_height", 1) or 1
        return (max(1, math.ceil(width)), max(1, math.ceil(height)))

    def _place_entity(
        self,
        entity: Entity,
        *,
        dependencies: Tuple[Any, ...] = (),
        desired: Optional[Tuple[int, int]] = None,
        max_radius: int = 12,
        padding: int = 0,
    ) -> Tuple[int, int]:
        """Reserve a layout slot for ``entity`` and assign its tile position."""

        footprint = self._entity_footprint(entity)

        if desired is not None:
            pos = self.layout.reserve_near(
                desired,
                max_radius=max_radius,
                footprint=footprint,
                padding=padding,
            )
        elif dependencies:
            pos = self._allocate_position(
                *dependencies, footprint=footprint, padding=padding
            )
        else:
            pos = self.layout.get_next_position(
                footprint=footprint, padding=padding
            )

        entity.tile_position = pos
        return pos

    def _place_entity_in_zone(
        self,
        entity: Entity,
        zone: str,
        *,
        padding: int = 0,
    ) -> Tuple[int, int]:
        footprint = self._entity_footprint(entity)
        pos = self.layout.reserve_in_zone(zone, footprint=footprint, padding=padding)
        entity.tile_position = pos
        return pos

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

    def _allocate_position(
        self,
        *value_refs,
        footprint: Tuple[int, int] = (1, 1),
        padding: int = 0,
    ) -> Tuple[int, int]:
        """Allocate a placement near the average of dependency positions if possible."""
        positions: List[Tuple[int, int]] = []
        for ref in value_refs:
            positions.extend(self._extract_dependency_positions(ref))

        if positions:
            avg_x = sum(pos[0] for pos in positions) / len(positions)
            avg_y = sum(pos[1] for pos in positions) / len(positions)
            search_radius = max(6, len(positions) * 4)
            return self.layout.reserve_near(
                (avg_x, avg_y),
                max_radius=search_radius,
                footprint=footprint,
                padding=padding,
            )

        return self.layout.get_next_position(footprint=footprint, padding=padding)

    def _add_entity(self, entity: Entity) -> Entity:
        """Add an entity to the blueprint and return the stored reference."""
        return self.blueprint.entities.append(entity, copy=False)

    def _compose_debug_info(
        self,
        usage_entry: Optional[SignalUsageEntry],
        fallback_name: Optional[str] = None,
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
                    metadata.get("name") or usage_entry.debug_label or fallback_name
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

    def _apply_blueprint_metadata(self, previous_description: str) -> None:
        """Ensure blueprint metadata advertises edge placement conventions."""

        if not hasattr(self.blueprint, "description"):
            return

        note = EDGE_LAYOUT_NOTE
        description = previous_description or ""

        if note in description:
            self.blueprint.description = description
            return

        if description:
            if not description.endswith("\n"):
                description += "\n"
            self.blueprint.description = description + note
        else:
            self.blueprint.description = note

    def _analyze_signal_usage(
        self, ir_operations: List[IRNode]
    ) -> Dict[str, SignalUsageEntry]:
        """Construct a signal usage index from the IR prior to emission."""

        usage: Dict[str, SignalUsageEntry] = {}

        def ensure_entry(
            signal_id: str, signal_type: Optional[str] = None
        ) -> SignalUsageEntry:
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
                            entry.literal_declared_type = getattr(
                                op, "output_type", None
                            )

            if isinstance(op, IR_Arith):
                record_consumer(op.left, op.node_id)
                record_consumer(op.right, op.node_id)
            elif isinstance(op, IR_Decider):
                record_consumer(op.left, op.node_id)
                record_consumer(op.right, op.node_id)
                record_consumer(op.output_value, op.node_id)
            elif isinstance(op, IR_MemCreate):
                if hasattr(op, "initial_value") and op.initial_value is not None:
                    record_consumer(op.initial_value, op.node_id)
            elif isinstance(op, IR_MemWrite):
                entry = ensure_entry(op.node_id)
                if not entry.debug_label:
                    entry.debug_label = op.memory_id
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

            if has_external_targets and self.signal_graph.iter_sinks(signal_id):
                continue

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

        combinator = new_entity("constant-combinator")
        pos = self._place_entity_in_zone(combinator, "south_exports")
        self._add_entity(combinator)
        placement = EntityPlacement(
            entity=combinator,
            entity_id=anchor_id,
            position=pos,
            output_signals={},
            input_signals={},
            role="export_anchor",
            zone="south_exports",
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
        previous_description = getattr(self.blueprint, "description", "")

        # Initialize state and perform IR pre-analysis
        self.prepare(ir_operations)
        self.blueprint.label = previous_label
        self._apply_blueprint_metadata(previous_description)

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

            # Analyze wiring edges prior to creating physical connections
            self._prepare_wiring_plan()

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
            "OverlappingObjectsWarning",
            "UnknownEntityWarning",
            "InvalidEntityError",
            "InvalidConnectorError",
        }

        for message, category, filename, lineno in captured_warnings:
            warning_name = category.__name__

            # Convert critical warnings to errors
            if warning_name in critical_warning_types:
                self.diagnostics.error(
                    f"Blueprint validation failed - {warning_name}: {message}"
                )
            else:
                # Keep other warnings as warnings
                self.diagnostics.warning(f"{warning_name}: {message}")

        # Basic blueprint validation
        if len(self.blueprint.entities) == 0:
            self.diagnostics.warning("Generated blueprint is empty")

        # Check for entities with invalid positions
        for entity in self.blueprint.entities:
            if hasattr(entity, "tile_position"):
                x, y = entity.tile_position
                if x < -1000 or x > 1000 or y < -1000 or y > 1000:
                    self.diagnostics.warning(f"Entity at extreme position: ({x}, {y})")

        # Count entity types for debugging
        entity_counts = {}
        for entity in self.blueprint.entities:
            entity_type = type(entity).__name__
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

        self.diagnostics.info(
            f"Generated blueprint with {len(self.blueprint.entities)} entities: {entity_counts}"
        )

    def emit_ir_operation(self, op: IRNode):
        """Emit a single IR operation, consulting signal usage index for materialization rules."""
        usage_entry = self.signal_usage.get(getattr(op, "node_id", None))
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
            print(
                f"SKIP constant: {op.node_id} (should_materialize={usage_entry.should_materialize}, is_typed_literal={getattr(usage_entry, 'is_typed_literal', None)})"
            )
            return

        combinator = new_entity("constant-combinator")
        pos = self._place_entity_in_zone(combinator, "north_literals")
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
                else self._get_signal_name(op.output_type)
            )
            signal_type = usage_entry.resolved_signal_type if usage_entry else None
            value = op.value
        print(
            f"EMIT constant: {op.node_id} name={signal_name} type={signal_type} value={value}"
        )
        # For explicit typed literals, emit only if op.value is nonzero and matches literal_value, skip anchor combinators
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
        # Debug: Print filters for this constant combinator
        for sec in getattr(combinator, "sections", []):
            print(
                f"DEBUG combinator {op.node_id} filters: {getattr(sec, 'filters', None)}"
            )

    def emit_arithmetic(self, op: IR_Arith):
        """Emit arithmetic combinator for IR_Arith."""

        combinator = new_entity("arithmetic-combinator")
        pos = self._place_entity(
            combinator, dependencies=(op.left, op.right)
        )

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
        debug_info = self._compose_debug_info(
            usage_entry, fallback_name=label_candidate
        )
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
        self._track_signal_source(op.node_id, op.node_id)

        # Track signal dependencies
        self._add_signal_sink(op.left, op.node_id)
        self._add_signal_sink(op.right, op.node_id)

    def emit_decider(self, op: IR_Decider):
        """Emit decider combinator for IR_Decider."""

        combinator = new_entity("decider-combinator")
        pos = self._place_entity(
            combinator, dependencies=(op.left, op.right)
        )

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

        condition_kwargs = {"comparator": op.test_op}
        if isinstance(left_operand, int):
            # Factorio deciders require a signal on the left; fallback to zero signal.
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
        self._track_signal_source(op.node_id, op.node_id)

        # Track signal dependencies
        self._add_signal_sink(op.left, op.node_id)
        if not isinstance(op.right, int):  # Signal reference
            self._add_signal_sink(op.right, op.node_id)

    def emit_memory_create(self, op: IR_MemCreate):
        """Emit memory module creation using simplified 2-combinator cell."""
        signal_type = self._get_signal_name(op.signal_type)

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
        # Memory read connects to the output of the memory combinator
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

    def emit_memory_write(self, op: IR_MemWrite):
        """Emit memory write operation to the 3-combinator memory cell."""
        memory_id = op.memory_id

        if memory_id not in self.memory_builder.memory_modules:
            self.diagnostics.error(f"Memory {memory_id} not found for write operation")
            return

        # Get the memory module components
        memory_module = self.memory_builder.memory_modules[memory_id]
        write_gate_placement = memory_module["write_gate"]
        hold_gate_placement = memory_module["hold_gate"]

        try:
            write_gate = write_gate_placement.entity
            hold_gate = hold_gate_placement.entity

            neighbor_positions = []
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

            # Memory data channel is provided via a dedicated injector combinator per write.
            declared_signal = memory_module.get("signal_type")
            if not declared_signal:
                base_signal = (
                    getattr(op.data_signal, "signal_type", None)
                    if isinstance(op.data_signal, SignalRef)
                    else op.memory_id
                )
                declared_signal = self._get_signal_name(base_signal)

            # Enable writer: produce signal-W pulse based on write enable expression
            enable_combinator = DeciderCombinator()
            enable_pos = self._place_entity(
                enable_combinator,
                desired=desired_location,
                max_radius=8,
            )

            enable_literal: Optional[int] = None
            if isinstance(op.write_enable, int):
                enable_literal = op.write_enable
            elif isinstance(op.write_enable, SignalRef) and self.materializer:
                enable_literal = self.materializer.inline_value(op.write_enable)

            if enable_literal is not None:
                # Inline literal enables create a self-sustaining pulse generator so we
                # don't need a dedicated constant combinator. A positive literal means
                # "always write".
                comparator = "="
                constant = 0 if enable_literal != 0 else 1
                condition = DeciderCombinator.Condition(
                    first_signal="signal-0",
                    comparator=comparator,
                    constant=constant,
                )
                pulse_constant = 1 if enable_literal != 0 else 0
            else:
                enable_signal = self._get_signal_name(op.write_enable)
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

            # Build a write injector that gates the data signal with signal-W pulses.
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

            # Route the value signal into the injector instead of the latch directly.
            self._add_signal_sink(op.data_signal, injector_entity_id)

            # Feed signal-W into the injector to gate writes.
            self.blueprint.add_circuit_connection(
                "green",
                enable_combinator,
                injector,
                side_1="output",
                side_2="input",
            )

            # Connect injector output to the memory write gate on the red network.
            self.blueprint.add_circuit_connection(
                "red",
                injector,
                write_gate,
                side_1="output",
                side_2="input",
            )

        except Exception as e:
            self.diagnostics.warning(
                f"Could not configure memory write combinator for {memory_id}: {e}"
            )

    def emit_place_entity(self, op: IR_PlaceEntity):
        """Emit entity placement using the entity factory."""
        try:
            entity = new_entity(op.prototype)
            footprint = self._entity_footprint(entity)

            # Handle both constant and variable coordinates
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
                # Default to layout engine for variable coordinates or fallback
                pos = self.layout.get_next_position(footprint=footprint)

            entity.tile_position = pos

            # Apply any additional properties
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

        except ValueError as e:
            self.diagnostics.error(f"Failed to create entity: {e}")
        except Exception as e:
            self.diagnostics.error(
                f"Unexpected error creating entity '{op.prototype}': {e}"
            )

    def emit_entity_prop_write(self, op: IR_EntityPropWrite):
        """Emit entity property write (circuit network connection)."""
        if op.entity_id not in self.entities:
            self.diagnostics.error(
                f"Entity {op.entity_id} not found for property write"
            )
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
        property_key = (op.entity_id, op.property_name)
        stored_value = self.entity_property_signals.get(property_key)

        if isinstance(stored_value, SignalRef):
            self.signal_graph.set_source(op.node_id, stored_value.source_id)
            self._track_signal_source(op.node_id, stored_value.source_id)
        elif isinstance(stored_value, int):
            combinator = ConstantCombinator()
            pos = self._place_entity(combinator)
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
            self._track_signal_source(op.node_id, op.node_id)
        else:
            # Fallback: expose controlling entity output directly
            if op.entity_id in self.entities:
                self.signal_graph.set_source(op.node_id, op.entity_id)
                self._track_signal_source(op.node_id, op.entity_id)

    def _prepare_wiring_plan(self) -> None:
        """Gather circuit edges and emit early diagnostics for potential conflicts."""

        self._circuit_edges = collect_circuit_edges(
            self.signal_graph, self.signal_usage, self.entities
        )

        sink_conflicts: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )

        for edge in self._circuit_edges:
            if not edge.source_entity_id:
                continue
            sink_conflicts[edge.sink_entity_id][edge.resolved_signal_name].add(
                edge.source_entity_id
            )

        for sink_id, conflict_map in sink_conflicts.items():
            for resolved_signal, sources in conflict_map.items():
                if len(sources) <= 1:
                    continue

                source_labels = []
                for source_entity_id in sorted(sources):
                    placement = self.entities.get(source_entity_id)
                    if placement:
                        source_labels.append(placement.entity_id)
                    else:
                        source_labels.append(source_entity_id)

                sink_label = sink_id
                placement = self.entities.get(sink_id)
                if placement:
                    sink_label = placement.entity_id

                source_desc = ", ".join(source_labels)

                self.diagnostics.warning(
                    "Detected multiple producers for signal "
                    f"'{resolved_signal}' feeding sink '{sink_label}'; attempting wire coloring to isolate networks (sources: {source_desc})."
                )

        locked_colors = self._determine_locked_wire_colors()
        coloring_result = plan_wire_colors(self._circuit_edges, locked_colors)

        self._node_color_assignments = coloring_result.assignments
        self._coloring_conflicts = coloring_result.conflicts
        self._coloring_success = coloring_result.is_bipartite

        # Map each source→sink edge to a concrete color decision
        edge_color_map: Dict[Tuple[str, str, str], str] = {}
        for edge in self._circuit_edges:
            if not edge.source_entity_id:
                continue
            node_key = (edge.source_entity_id, edge.resolved_signal_name)
            color = self._node_color_assignments.get(node_key, "red")
            edge_color_map[
                (edge.source_entity_id, edge.sink_entity_id, edge.resolved_signal_name)
            ] = color

        self._edge_color_map = edge_color_map

        if self._edge_color_map:
            color_counts = Counter(self._edge_color_map.values())
            summaries = []
            for color in WIRE_COLORS:
                count = color_counts.get(color, 0)
                if count:
                    summaries.append(f"{count} {color}")
            if summaries:
                self.diagnostics.info(
                    "Wire color planner assignments: " + ", ".join(summaries)
                )

        if not self._coloring_success and self._coloring_conflicts:
            for conflict in self._coloring_conflicts:
                resolved_signal = conflict.nodes[0][1]
                source_desc = ", ".join(
                    sorted({node_id for node_id, _ in conflict.nodes})
                )
                sink_desc = (
                    ", ".join(sorted(conflict.sinks))
                    if conflict.sinks
                    else "unknown sinks"
                )
                self.diagnostics.error(
                    "Two-color routing could not isolate signal "
                    f"'{resolved_signal}' across sinks [{sink_desc}]; falling back to single-channel wiring for involved entities ({source_desc})."
                )
            # Fallback to legacy behavior when coloring fails
            self._edge_color_map = {}

    def _determine_locked_wire_colors(self) -> Dict[Tuple[str, str], str]:
        """Collect wire color locks for structures that must retain manual wiring."""

        locked: Dict[Tuple[str, str], str] = {}

        for module in self.memory_builder.memory_modules.values():
            for placement in module.values():
                if not isinstance(placement, EntityPlacement):
                    continue
                for signal_name, color in placement.output_signals.items():
                    if color == "green":
                        locked[(placement.entity_id, signal_name)] = color

        return locked

    def create_circuit_connections(self):
        """Create circuit wire connections between entities."""
        # First, wire up all memory circuits
        for memory_id in self.memory_builder.memory_modules:
            try:
                self.memory_builder.wire_sr_latch(memory_id)
            except Exception as e:
                self.diagnostics.error(f"Failed to wire memory {memory_id}: {e}")

        # Create connections between signal sources and sinks
        for (
            signal_id,
            source_entity_id,
            sink_entities,
        ) in self.signal_graph.iter_edges():
            if source_entity_id:
                if source_entity_id in self.entities:
                    source_placement = self.entities[source_entity_id]

                    for sink_entity_id in sink_entities:
                        if sink_entity_id in self.entities:
                            sink_placement = self.entities[sink_entity_id]

                            # Determine wire color using precomputed assignments with fallback
                            resolved_signal = signal_id
                            usage_entry = self.signal_usage.get(signal_id)
                            if usage_entry and usage_entry.resolved_signal_name:
                                resolved_signal = usage_entry.resolved_signal_name

                            wire_color = self._edge_color_map.get(
                                (source_entity_id, sink_entity_id, resolved_signal)
                            )

                            if not wire_color:
                                wire_color = self._get_wire_color(
                                    source_placement,
                                    sink_placement,
                                    resolved_signal,
                                )

                            self._connect_with_wire_path(
                                source_placement,
                                sink_placement,
                                wire_color,
                            )
                else:
                    self.diagnostics.error(
                        f"Source entity {source_entity_id} not found for signal {signal_id}"
                    )
            else:
                self.diagnostics.error(f"No source found for signal {signal_id}")

    def _get_wire_color(
        self,
        source: EntityPlacement,
        sink: EntityPlacement,
        resolved_signal: Optional[str] = None,
    ) -> str:
        """Determine appropriate wire color for connection."""
        if resolved_signal:
            desired = sink.input_signals.get(resolved_signal)
            if desired:
                return desired
            if resolved_signal != "signal-W":
                desired = sink.input_signals.get("signal-each")
                if desired:
                    return desired
            desired = sink.input_signals.get("signal-W")
            if desired and resolved_signal == "signal-W":
                return desired

        # Use red as default, green for memory outputs to avoid conflicts
        if "memory" in source.entity_id and "output" in source.entity_id:
            return "green"
        return "red"

    # ------------------------------------------------------------------
    # Wiring helpers
    # ------------------------------------------------------------------

    def _compute_wire_distance(
        self, source: EntityPlacement, sink: EntityPlacement
    ) -> float:
        """Return Euclidean distance between two placements in tile units."""

        sx, sy = source.position
        tx, ty = sink.position
        if self.wire_relay_options.placement_strategy == "manhattan":
            return abs(tx - sx) + abs(ty - sy)
        return math.dist((sx, sy), (tx, ty))

    def _connect_with_wire_path(
        self,
        source: EntityPlacement,
        sink: EntityPlacement,
        wire_color: str,
    ) -> None:
        """Wire entities, inserting relay poles when range limits are exceeded."""

        source_entity = source.entity
        sink_entity = sink.entity
        source_dual = getattr(source_entity, "dual_circuit_connectable", False)
        sink_dual = getattr(sink_entity, "dual_circuit_connectable", False)

        span_limit = self.wire_relay_options.normalized_span()
        total_distance = self._compute_wire_distance(source, sink)

        path = [source]
        relays = self._insert_wire_relays_if_needed(source, sink)
        if relays:
            path.extend(relays)
        path.append(sink)

        if not relays and total_distance > span_limit:
            self.diagnostics.warning(
                "Connection %s -> %s spans %.1f tiles which exceeds configured reach %.1f; proceeding without relays."
                % (source.entity_id, sink.entity_id, total_distance, span_limit)
            )

        for idx in range(len(path) - 1):
            first_pos = path[idx].position
            second_pos = path[idx + 1].position
            segment_distance = math.dist(first_pos, second_pos)
            if segment_distance > span_limit + 1e-6:
                self.diagnostics.warning(
                    "Segment %s -> %s spans %.1f tiles (limit %.1f)."
                    % (
                        path[idx].entity_id,
                        path[idx + 1].entity_id,
                        segment_distance,
                        span_limit,
                    )
                )
        
        path_length = len(path)

        for idx in range(path_length - 1):
            first = path[idx]
            second = path[idx + 1]

            connection_kwargs: Dict[str, Any] = dict(
                color=wire_color,
                entity_1=first.entity,
                entity_2=second.entity,
            )

            if idx == 0 and source_dual:
                connection_kwargs["side_1"] = "output"
            if idx == path_length - 2 and sink_dual:
                connection_kwargs["side_2"] = "input"

            try:
                self.blueprint.add_circuit_connection(**connection_kwargs)
            except Exception as exc:
                self.diagnostics.error(
                    f"Failed to connect {first.entity_id} -> {second.entity_id}: {exc}"
                )

    def _insert_wire_relays_if_needed(
        self,
        source: EntityPlacement,
        sink: EntityPlacement,
    ) -> List[EntityPlacement]:
        """Insert medium poles when two endpoints exceed wire reach."""

        options = self.wire_relay_options
        if not options.enabled:
            return []

        span_limit = options.normalized_span()
        distance = self._compute_wire_distance(source, sink)
        if distance <= span_limit:
            return []

        segments = max(1, math.ceil(distance / span_limit))
        required_relays = segments - 1

        relays: List[EntityPlacement] = []
        if required_relays <= 0:
            return relays

        if options.max_relays is not None and required_relays > options.max_relays:
            self.diagnostics.warning(
                "Connection %s -> %s requires %d relay poles but max_relays=%d; skipping automatic relay placement."
                % (
                    source.entity_id,
                    sink.entity_id,
                    required_relays,
                    options.max_relays,
                )
            )
            return relays

        self.diagnostics.info(
            "Inserting %d wire relay(s) (strategy=%s, span=%.1f) to bridge %.1f tiles between %s and %s."
            % (
                required_relays,
                options.placement_strategy,
                span_limit,
                distance,
                source.entity_id,
                sink.entity_id,
            )
        )

        for idx in range(1, segments):
            ratio = idx / segments

            try:
                pole_entity = new_entity(WIRE_RELAY_ENTITY)
            except Exception as exc:
                self.diagnostics.error(
                    f"Failed to instantiate relay pole for {source.entity_id}->{sink.entity_id}: {exc}"
                )
                break

            footprint = self._entity_footprint(pole_entity)
            pos = self.layout.reserve_along_path(
                source.position,
                sink.position,
                ratio,
                strategy=options.placement_strategy,
                max_radius=12,
                footprint=footprint,
                padding=0,
            )

            pole_entity.tile_position = pos
            pole_entity = self._add_entity(pole_entity)

            relay_id = f"__wire_relay_{self._wire_relay_counter}"
            self._wire_relay_counter += 1

            placement = EntityPlacement(
                entity=pole_entity,
                entity_id=relay_id,
                position=pos,
                output_signals={},
                input_signals={},
                role="wire_relay",
                zone="infrastructure",
            )

            self.entities[relay_id] = placement
            relays.append(placement)

        return relays

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
        if hasattr(operand, "signal_type"):
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
        elif isinstance(operand, SignalRef):
            if self.materializer:
                inlined = self.materializer.inline_value(operand)
                if inlined is not None:
                    return inlined
                usage_entry = self.signal_usage.get(operand.source_id)
                resolved = self.materializer.resolve_signal_name(
                    operand.signal_type, usage_entry
                )
                if resolved:
                    return resolved
            if hasattr(operand, "signal_type"):
                return self._get_signal_name(operand.signal_type)
            return str(operand)
        elif isinstance(operand, str):
            return self._get_signal_name(operand)
        else:
            return str(operand)

    def _track_signal_source(self, signal_id: str, entity_id: str) -> None:
        """Record that a logical signal is produced by a specific entity."""

        usage_entry = self.signal_usage.get(signal_id)
        if not usage_entry:
            return

        usage_entry.output_entities.add(entity_id)

        resolved_name = usage_entry.resolved_signal_name
        if resolved_name:
            usage_entry.resolved_outputs[entity_id] = resolved_name

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
