# emit.py
"""
Blueprint emission module for the Factorio Circuit DSL.

This module converts IR operations into actual Factorio combinators and entities
using the factorio-draftsman library to generate blueprint JSON.
"""

import math
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Literal, Set, Iterable

# Add draftsman to path
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent.parent / "factorio-draftsman")
)

from draftsman.blueprintable import Blueprint  # type: ignore[import-not-found]
from draftsman.entity import (
    new_entity,
)  # Use draftsman's factory  # type: ignore[import-not-found]
from draftsman.entity import *  # Import all entities  # type: ignore[import-not-found]
from draftsman.entity import (  # type: ignore[import-not-found]
    DeciderCombinator,
    ConstantCombinator,
)
from draftsman.classes.entity import Entity  # type: ignore[import-not-found]
from draftsman.data import signals as signal_data  # type: ignore[import-not-found]
from draftsman.signatures import SignalID  # type: ignore[import-not-found]


from ..ir import *
from ..ast import SignalLiteral
from ..semantic import DiagnosticCollector, render_source_location
from .signals import (
    EntityPlacement,
    SignalUsageEntry,
    SignalGraph,
    SignalMaterializer,
)
from .wiring import (
    WIRE_COLORS,
    CircuitEdge,
    collect_circuit_edges,
    plan_wire_colors,
)

MAX_CIRCUIT_WIRE_SPAN = 9.0
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
        span = (
            self.max_span
            if self.max_span and self.max_span > 0
            else MAX_CIRCUIT_WIRE_SPAN
        )
        return max(1.0, float(span))


from .memory import MemoryCircuitBuilder
from .layout import LayoutEngine
from .debug_format import format_entity_description
from .entity_emitter import EntityEmitter
from .connection_builder import ConnectionBuilder
from .signal_resolver import SignalResolver


POWER_POLE_CONFIG = {
    "small": {
        "prototype": "small-electric-pole",
        "footprint": (1, 1),
        "supply_radius": 2,
        "padding": 0,
        "wire_reach": 9,
    },
    "medium": {
        "prototype": "medium-electric-pole",
        "footprint": (1, 1),
        "supply_radius": 3,
        "padding": 0,
        "wire_reach": 9,
    },
    "big": {
        "prototype": "big-electric-pole",
        "footprint": (2, 2),
        "supply_radius": 5,
        "padding": 1,
        "wire_reach": 30,
    },
    "substation": {
        "prototype": "substation",
        "footprint": (2, 2),
        "supply_radius": 9,
        "padding": 1,
        "wire_reach": 18,
    },
}


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
        power_pole_type: Optional[str] = None,
    ):
        # Persistent configuration
        self.signal_type_map = signal_type_map or {}
        self.enable_metadata_annotations = enable_metadata_annotations
        self.power_pole_type = power_pole_type.lower() if power_pole_type else None
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
        self._prepared_operation_index: Dict[str, IRNode] = {}
        self._memory_reads_by_memory: Dict[str, List[str]] = defaultdict(list)
        self._memory_write_strategies: Dict[str, Set[str]] = defaultdict(set)
        self._wire_merge_junctions: Dict[str, Dict[str, Any]] = {}
        self.power_poles: List[EntityPlacement] = []

        # Helper components constructed during prepare()
        self.signal_resolver: Optional[SignalResolver] = None
        self.entity_emitter: Optional[EntityEmitter] = None
        self.connection_builder: Optional[ConnectionBuilder] = None

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
            pos = self.layout.get_next_position(footprint=footprint, padding=padding)

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
        self._prepared_operation_index = {}
        self._memory_reads_by_memory = defaultdict(list)
        self._memory_write_strategies = defaultdict(set)

        for op in self._prepared_operations:
            node_id = getattr(op, "node_id", None)
            if node_id:
                self._prepared_operation_index[node_id] = op
            if isinstance(op, IR_MemRead):
                self._memory_reads_by_memory[op.memory_id].append(node_id)
            if isinstance(op, IR_MemWrite):
                strategy = self._analyze_write_strategy_early(op)
                self._memory_write_strategies[op.memory_id].add(strategy)

        self.signal_usage = self._analyze_signal_usage(ir_operations)
        self.materializer = SignalMaterializer(
            self.signal_usage, self.signal_type_map, self.diagnostics
        )
        self.materializer.finalize()

        if self.signal_resolver is None:
            self.signal_resolver = SignalResolver(
                self.signal_type_map,
                self.diagnostics,
                materializer=self.materializer,
                signal_usage=self.signal_usage,
            )
        else:
            self.signal_resolver.update_context(
                materializer=self.materializer, signal_usage=self.signal_usage
            )

        self.entity_emitter = EntityEmitter(self)
        self.connection_builder = ConnectionBuilder(self)

    def _analyze_write_strategy_early(self, op: IR_MemWrite) -> str:
        """Determine memory write strategy before entities are emitted."""

        enable_literal: Optional[int] = None
        if isinstance(op.write_enable, int):
            enable_literal = op.write_enable
        elif isinstance(op.write_enable, SignalRef):
            enable_op = self._prepared_operation_index.get(op.write_enable.source_id)
            if isinstance(enable_op, IR_Const):
                enable_literal = enable_op.value

        if enable_literal is not None and enable_literal != 0:
            if self._write_data_references_memory(op.data_signal, op.memory_id):
                return "FEEDBACK_LOOP"

        return "SR_LATCH"

    def _write_data_references_memory(
        self, value_signal: ValueRef, memory_id: str, depth: int = 0
    ) -> bool:
        """Recursively check if a value expression reads from a specific memory."""

        if depth > 10:
            return False

        if isinstance(value_signal, SignalRef):
            source_op = self._prepared_operation_index.get(value_signal.source_id)
            if isinstance(source_op, IR_MemRead):
                return source_op.memory_id == memory_id
            if isinstance(source_op, IR_Arith):
                return self._write_data_references_memory(
                    source_op.left, memory_id, depth + 1
                ) or self._write_data_references_memory(
                    source_op.right, memory_id, depth + 1
                )
            if isinstance(source_op, IR_Decider):
                return (
                    self._write_data_references_memory(
                        source_op.left, memory_id, depth + 1
                    )
                    or self._write_data_references_memory(
                        source_op.right, memory_id, depth + 1
                    )
                    or self._write_data_references_memory(
                        source_op.output_value, memory_id, depth + 1
                    )
                )

        if isinstance(value_signal, (list, tuple)):
            for item in value_signal:
                if self._write_data_references_memory(item, memory_id, depth + 1):
                    return True

        return False

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
            elif isinstance(op, IR_WireMerge):
                record_consumer(op.sources, op.node_id)

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

            if self.connection_builder is None:
                raise RuntimeError(
                    "ConnectionBuilder not initialised; call prepare() first"
                )

            # Analyze wiring edges prior to creating physical connections
            self.connection_builder.prepare_wiring_plan()

            # Add wiring connections
            self.connection_builder.create_circuit_connections()

            if self.power_pole_type:
                self._deploy_power_poles()

            if self.power_poles:
                self._connect_power_grid()

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
        if self.entity_emitter is None:
            raise RuntimeError("EntityEmitter not initialised; call prepare() first")

        emitter = self.entity_emitter
        usage_entry = self.signal_usage.get(getattr(op, "node_id", None))
        if isinstance(op, IR_Const):
            # Only materialize if required by usage index
            if self.materializer and usage_entry and not usage_entry.should_materialize:
                return
            emitter.emit_constant(op)
        elif isinstance(op, IR_Arith):
            emitter.emit_arithmetic(op)
        elif isinstance(op, IR_Decider):
            emitter.emit_decider(op)
        elif isinstance(op, IR_MemCreate):
            emitter.emit_memory_create(op)
        elif isinstance(op, IR_MemRead):
            emitter.emit_memory_read(op)
        elif isinstance(op, IR_MemWrite):
            emitter.emit_memory_write(op)
        elif isinstance(op, IR_PlaceEntity):
            emitter.emit_place_entity(op)
        elif isinstance(op, IR_EntityPropWrite):
            emitter.emit_entity_prop_write(op)
        elif isinstance(op, IR_EntityPropRead):
            emitter.emit_entity_prop_read(op)
        elif isinstance(op, IR_WireMerge):
            emitter.emit_wire_merge(op)
        else:
            self.diagnostics.error(f"Unknown IR operation: {type(op)}")

    # Entity emission methods are implemented in entity_emitter.EntityEmitter

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

    def _iter_entity_tiles(self, placement: EntityPlacement) -> Iterable[Tuple[int, int]]:
        """Yield every tile coordinate occupied by an entity placement."""

        footprint = self._entity_footprint(placement.entity)
        base_x, base_y = placement.position
        for dx in range(footprint[0]):
            for dy in range(footprint[1]):
                yield (base_x + dx, base_y + dy)

    def _position_center(
        self,
        pos: Tuple[int, int],
        footprint: Tuple[int, int],
    ) -> Tuple[float, float]:
        """Return the geometric centre of an entity footprint at ``pos``."""

        return (
            pos[0] + footprint[0] / 2.0,
            pos[1] + footprint[1] / 2.0,
        )

    def _distance_tile_to_position(
        self,
        tile: Tuple[int, int],
        pos: Tuple[int, int],
        footprint: Tuple[int, int],
    ) -> float:
        tile_center = (tile[0] + 0.5, tile[1] + 0.5)
        position_center = self._position_center(pos, footprint)
        return math.hypot(
            tile_center[0] - position_center[0],
            tile_center[1] - position_center[1],
        )

    def _tiles_covered_by_position(
        self,
        pos: Tuple[int, int],
        tiles: Iterable[Tuple[int, int]],
        supply_radius: float,
        footprint: Tuple[int, int],
    ) -> Set[Tuple[int, int]]:
        """Return tiles covered by a power pole at ``pos`` with the given radius."""

        cx, cy = self._position_center(pos, footprint)
        radius = float(supply_radius) + 0.45
        radius_sq = radius * radius
        covered: Set[Tuple[int, int]] = set()
        for tile in tiles:
            tx = tile[0] + 0.5
            ty = tile[1] + 0.5
            if (tx - cx) ** 2 + (ty - cy) ** 2 <= radius_sq:
                covered.add(tile)
        return covered

    def _candidate_power_pole_positions(
        self,
        target_tile: Tuple[int, int],
        footprint: Tuple[int, int],
        padding: int,
        supply_radius: float,
        bounds: Tuple[int, int, int, int],
    ) -> Iterable[Tuple[int, int]]:
        """Enumerate feasible candidate slots near a target tile."""

        tx, ty = target_tile
        min_x, min_y, max_x, max_y = bounds
        search_radius = max(1, math.ceil(supply_radius) + max(footprint))
        seen: Set[Tuple[int, int]] = set()

        for dx in range(-search_radius, search_radius + max(footprint) + 1):
            for dy in range(-search_radius, search_radius + max(footprint) + 1):
                raw_x = tx + dx
                raw_y = ty + dy
                if raw_x < min_x or raw_x > max_x or raw_y < min_y or raw_y > max_y:
                    continue
                candidate = self.layout.snap_to_grid((raw_x, raw_y))
                if candidate in seen:
                    continue
                seen.add(candidate)
                if not self.layout.can_reserve(
                    candidate, footprint=footprint, padding=padding
                ):
                    continue
                yield candidate

    def _spawn_power_pole(
        self, config: Dict[str, Any], position: Tuple[int, int]
    ) -> EntityPlacement:
        """Create and register a power pole entity at ``position``."""

        pole_entity = new_entity(config["prototype"])
        pole_entity.tile_position = position
        pole_entity = self._add_entity(pole_entity)
        pole_id = f"power_pole_{self.next_entity_number}"
        self.next_entity_number += 1
        placement = EntityPlacement(
            entity=pole_entity,
            entity_id=pole_id,
            position=position,
            output_signals={},
            input_signals={},
            role="power",
        )
        self.entities[pole_id] = placement
        self.power_poles.append(placement)
        return placement

    def _power_pole_center(self, placement: EntityPlacement) -> Tuple[float, float]:
        footprint = self._entity_footprint(placement.entity)
        return self._position_center(placement.position, footprint)

    def _compute_power_components(
        self, wire_reach: float
    ) -> List[List[EntityPlacement]]:
        if not self.power_poles:
            return []

        reach_sq = float(wire_reach) * float(wire_reach)
        centers = [self._power_pole_center(p) for p in self.power_poles]
        components: List[List[EntityPlacement]] = []
        visited: Set[int] = set()

        for idx in range(len(self.power_poles)):
            if idx in visited:
                continue
            stack = [idx]
            component_indices: List[int] = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component_indices.append(current)
                cx, cy = centers[current]
                for other in range(len(self.power_poles)):
                    if other == current or other in visited:
                        continue
                    ox, oy = centers[other]
                    if (ox - cx) ** 2 + (oy - cy) ** 2 <= reach_sq:
                        stack.append(other)
            components.append([self.power_poles[i] for i in component_indices])

        return components

    def _ensure_power_pole_connectivity(self, config: Dict[str, Any]) -> None:
        wire_reach = float(config.get("wire_reach", 0) or 0)
        if wire_reach <= 0 or len(self.power_poles) < 2:
            return

        max_attempts = len(self.power_poles) + 8
        attempts = 0

        while attempts < max_attempts:
            components = self._compute_power_components(wire_reach)
            if len(components) <= 1:
                break

            base_component = components[0]
            best_distance = float("inf")
            closest_pair: Optional[Tuple[EntityPlacement, EntityPlacement]] = None

            for candidate_component in components[1:]:
                for pole_a in base_component:
                    center_a = self._power_pole_center(pole_a)
                    for pole_b in candidate_component:
                        center_b = self._power_pole_center(pole_b)
                        distance = math.dist(center_a, center_b)
                        if distance < best_distance:
                            best_distance = distance
                            closest_pair = (pole_a, pole_b)

            if not closest_pair:
                break

            if best_distance <= wire_reach:
                attempts += 1
                continue

            pole_a, pole_b = closest_pair
            needed = max(1, math.ceil(best_distance / wire_reach) - 1)
            center_a = self._power_pole_center(pole_a)
            center_b = self._power_pole_center(pole_b)
            footprint = config["footprint"]
            padding = config["padding"]

            for index in range(1, needed + 1):
                ratio = index / (needed + 1)
                target_center = (
                    center_a[0] + (center_b[0] - center_a[0]) * ratio,
                    center_a[1] + (center_b[1] - center_a[1]) * ratio,
                )
                approx_top_left = (
                    target_center[0] - footprint[0] / 2.0,
                    target_center[1] - footprint[1] / 2.0,
                )
                placement_pos = self.layout.reserve_near(
                    approx_top_left,
                    max_radius=max(6, math.ceil(wire_reach)),
                    footprint=footprint,
                    padding=padding,
                )
                if placement_pos is None:
                    placement_pos = self.layout.get_next_position(
                        footprint=footprint,
                        padding=padding,
                    )
                self._spawn_power_pole(config, placement_pos)

            attempts += 1

    def _tiles_missing_power(
        self,
        tiles: Iterable[Tuple[int, int]],
        supply_radius: float,
    ) -> Set[Tuple[int, int]]:
        radius_sq = (float(supply_radius) + 0.5) ** 2
        centers = [self._power_pole_center(pole) for pole in self.power_poles]
        if not centers:
            return set(tiles)

        uncovered: Set[Tuple[int, int]] = set()
        for tile in tiles:
            tx = tile[0] + 0.5
            ty = tile[1] + 0.5
            if not any((tx - cx) ** 2 + (ty - cy) ** 2 <= radius_sq for cx, cy in centers):
                uncovered.add(tile)
        return uncovered

    def _deploy_power_poles(self) -> None:
        """Place power poles of the configured type to cover all circuit entities."""

        if not self.power_pole_type:
            return

        config = POWER_POLE_CONFIG.get(self.power_pole_type)
        if not config:
            self.diagnostics.warning(
                f"Unknown power pole type '{self.power_pole_type}'; skipping power grid deployment"
            )
            return

        footprint: Tuple[int, int] = tuple(config["footprint"])
        padding = int(config["padding"])
        supply_radius = float(config["supply_radius"])

        placements = [
            placement
            for placement in self.entities.values()
            if placement.role != "power"
        ]

        if not placements:
            return

        coverage_tiles: Set[Tuple[int, int]] = set()
        for placement in placements:
            coverage_tiles.update(self._iter_entity_tiles(placement))

        if not coverage_tiles:
            return

        margin = max(
            2,
            math.ceil(supply_radius) + max(footprint) + padding,
        )
        min_x = min(tile[0] for tile in coverage_tiles) - margin
        min_y = min(tile[1] for tile in coverage_tiles) - margin
        max_x = max(tile[0] for tile in coverage_tiles) + margin
        max_y = max(tile[1] for tile in coverage_tiles) + margin
        bounds = (min_x, min_y, max_x, max_y)

        uncovered_tiles = set(coverage_tiles)
        coverage_cache: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}

        while uncovered_tiles:
            target_tile = next(iter(uncovered_tiles))
            best_pos: Optional[Tuple[int, int]] = None
            best_total_cover: Set[Tuple[int, int]] = set()
            best_cover: Set[Tuple[int, int]] = set()
            best_distance = float("inf")

            for candidate in self._candidate_power_pole_positions(
                target_tile,
                footprint,
                padding,
                supply_radius,
                bounds,
            ):
                total_cover = coverage_cache.get(candidate)
                if total_cover is None:
                    total_cover = self._tiles_covered_by_position(
                        candidate,
                        coverage_tiles,
                        supply_radius,
                        footprint,
                    )
                    coverage_cache[candidate] = total_cover
                new_cover = total_cover & uncovered_tiles
                if not new_cover:
                    continue

                distance = self._distance_tile_to_position(
                    target_tile,
                    candidate,
                    footprint,
                )

                if len(new_cover) > len(best_cover) or (
                    len(new_cover) == len(best_cover) and distance < best_distance
                ):
                    best_pos = candidate
                    best_total_cover = total_cover
                    best_cover = new_cover
                    best_distance = distance

            if best_pos is not None:
                claimed = self.layout.reserve_exact(
                    best_pos,
                    footprint=footprint,
                    padding=padding,
                )
                if claimed is None:
                    claimed = self.layout.reserve_near(
                        best_pos,
                        max_radius=max(6, math.ceil(supply_radius) + max(footprint)),
                        footprint=footprint,
                        padding=padding,
                    )
                if claimed is None:
                    claimed = self.layout.get_next_position(
                        footprint=footprint,
                        padding=padding,
                    )
                placement = self._spawn_power_pole(config, claimed)
                if claimed == best_pos and best_total_cover:
                    coverage = best_total_cover
                else:
                    coverage = self._tiles_covered_by_position(
                        placement.position,
                        coverage_tiles,
                        supply_radius,
                        footprint,
                    )
            else:
                claimed = self.layout.reserve_near(
                    target_tile,
                    max_radius=max(6, math.ceil(supply_radius) + max(footprint)),
                    footprint=footprint,
                    padding=padding,
                )
                if claimed is None:
                    claimed = self.layout.get_next_position(
                        footprint=footprint,
                        padding=padding,
                    )
                placement = self._spawn_power_pole(config, claimed)
                coverage = self._tiles_covered_by_position(
                    placement.position,
                    coverage_tiles,
                    supply_radius,
                    footprint,
                )

            if not coverage:
                coverage = {target_tile}

            coverage_cache[placement.position] = coverage
            uncovered_tiles.difference_update(coverage)

        residual_uncovered = self._tiles_missing_power(coverage_tiles, supply_radius)
        safety_guard = 0
        max_attempts = len(coverage_tiles) * 2 + 8

        while residual_uncovered and safety_guard < max_attempts:
            target_tile = residual_uncovered.pop()
            placement_pos = self.layout.reserve_near(
                target_tile,
                max_radius=max(
                    8,
                    math.ceil(supply_radius) + max(footprint) + padding,
                ),
                footprint=footprint,
                padding=padding,
            )
            if placement_pos is None:
                placement_pos = self.layout.get_next_position(
                    footprint=footprint,
                    padding=padding,
                )
            self._spawn_power_pole(config, placement_pos)
            residual_uncovered = self._tiles_missing_power(
                coverage_tiles,
                supply_radius,
            )
            safety_guard += 1

        if residual_uncovered:
            self.diagnostics.warning(
                "Unable to guarantee power coverage for all tiles after fallback placement"
            )

        if not self.power_poles:
            fallback = self.layout.get_next_position(
                footprint=footprint,
                padding=padding,
            )
            self._spawn_power_pole(config, fallback)

        self._ensure_power_pole_connectivity(config)

    def _connect_power_grid(self) -> None:
        """Ensure all placed power poles are connected via copper wires."""

        if not self.power_poles:
            return

        try:
            self.blueprint.generate_power_connections()
        except Exception as exc:
            self.diagnostics.warning(
                f"Failed to auto-generate power connections: {exc}"
            )


# =============================================================================
# Public API
# =============================================================================


def emit_blueprint(
    ir_operations: List[IRNode],
    label: str = "DSL Generated",
    signal_type_map: Dict[str, str] = None,
    *,
    power_pole_type: Optional[str] = None,
) -> Tuple[Blueprint, DiagnosticCollector]:
    """Convert IR operations to Factorio blueprint."""
    emitter = BlueprintEmitter(
        signal_type_map,
        power_pole_type=power_pole_type,
    )
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
    *,
    power_pole_type: Optional[str] = None,
) -> Tuple[str, DiagnosticCollector]:
    """Convert IR operations to Factorio blueprint string."""
    blueprint, diagnostics = emit_blueprint(
        ir_operations,
        label,
        signal_type_map,
        power_pole_type=power_pole_type,
    )

    try:
        blueprint_string = blueprint.to_string()
        return blueprint_string, diagnostics
    except Exception as e:
        diagnostics.error(f"Blueprint string generation failed: {e}")
        return "", diagnostics
