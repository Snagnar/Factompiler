# emit.py
"""
Blueprint emission module for the Factorio Circuit DSL.

This module converts IR operations into actual Factorio combinators and entities
using the factorio-draftsman library to generate blueprint JSON.
"""

from __future__ import annotations

import copy
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Literal, Set, Iterable, TYPE_CHECKING

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
from dsl_compiler.src.layout.layout_plan import EntityPlacement as EmittedEntityPlacement

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


WIRE_RELAY_ENTITY = "medium-electric-pole"


from dsl_compiler.src.layout.memory import MemoryCircuitBuilder
from dsl_compiler.src.layout.layout_engine import LayoutEngine
from dsl_compiler.src.layout.layout_plan import (
    LayoutPlan,
    EntityPlacement as LayoutEntityPlacement,
    PowerPolePlacement,
)
from dsl_compiler.src.layout.legacy_signals import (
    EntityPlacement as EmittedEntityPlacement,
)
from dsl_compiler.src.layout.signal_graph import SignalGraph
from dsl_compiler.src.layout.connection_planner import ConnectionPlanner
from dsl_compiler.src.layout.power_planner import PowerPlanner, PlannedPowerPole
from dsl_compiler.src.layout.signal_analyzer import (
    SignalAnalyzer,
    SignalMaterializer,
    SignalUsageEntry,
)
from dsl_compiler.src.layout.debug import format_entity_description
from .entity_emitter import PlanEntityEmitter
from dsl_compiler.src.layout.legacy_entity_emitter import EntityEmitter
from dsl_compiler.src.layout.legacy_signal_resolver import SignalResolver

if TYPE_CHECKING:  # pragma: no cover - type checking helper
    from dsl_compiler.src.layout.planner import LayoutPlanner


# =============================================================================
# Entity Factory using draftsman's catalog
# =============================================================================


# =============================================================================
# Memory Circuit Builder
# =============================================================================


class _PlanOnlyBlueprintProxy:
    """Lightweight proxy that ignores circuit wiring during plan-only runs."""

    def __init__(self, blueprint: Blueprint) -> None:
        object.__setattr__(self, "_blueprint", blueprint)

    def add_circuit_connection(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_blueprint"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_blueprint"), name, value)


class LayoutBuilder:
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
        plan_only_mode: bool = False,
    ):
        # Persistent configuration
        self.signal_type_map = signal_type_map or {}
        self.enable_metadata_annotations = enable_metadata_annotations
        self.power_pole_type = power_pole_type.lower() if power_pole_type else None
        self._plan_only_mode = plan_only_mode
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

        raw_blueprint = Blueprint()
        self.blueprint = (
            _PlanOnlyBlueprintProxy(raw_blueprint)
            if self._plan_only_mode
            else raw_blueprint
        )
        self.blueprint.label = "DSL Generated Blueprint"
        self.blueprint.version = (2, 0)

        self.layout = LayoutEngine()
        self.layout_plan = LayoutPlan()
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
        self.signal_analyzer = None
        self.materializer = None
        self._prepared_operations = []
        self._prepared_operation_index: Dict[str, IRNode] = {}
        self._memory_reads_by_memory: Dict[str, List[str]] = defaultdict(list)
        self._memory_write_strategies: Dict[str, Set[str]] = defaultdict(set)
        self._wire_merge_junctions: Dict[str, Dict[str, Any]] = {}
        self.power_poles: List[EmittedEntityPlacement] = []
        self._wire_relay_counter = 0

        # Helper components constructed during prepare()
        self.signal_resolver: Optional[SignalResolver] = None
        self.entity_emitter: Optional[EntityEmitter] = None
        self.connection_planner: Optional[ConnectionPlanner] = None

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

        signal_analyzer = SignalAnalyzer(self.diagnostics)
        self.signal_usage = signal_analyzer.analyze(ir_operations)
        self.signal_analyzer = signal_analyzer
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
        self.connection_planner = ConnectionPlanner(
            self.layout_plan,
            self.signal_usage,
            self.diagnostics,
            self.layout,
            max_wire_span=self.wire_relay_options.normalized_span(),
        )

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

    def _record_entity_placement(self, placement: EmittedEntityPlacement) -> None:
        """Record placement in emission lookup and mirrored layout plan."""

        self.entities[placement.entity_id] = placement

        entity = placement.entity
        entity_type: str = getattr(entity, "name", None) or type(entity).__name__
        metadata: Dict[str, Any] = dict(getattr(placement, "metadata", {}) or {})
        metadata.setdefault("footprint", self._entity_footprint(entity))
        metadata.setdefault("entity_obj", entity)
        layout_entry = LayoutEntityPlacement(
            ir_node_id=placement.entity_id,
            entity_type=entity_type,
            position=placement.position,
            properties=metadata,
            role=placement.role,
            zone=placement.zone,
        )
        self.layout_plan.add_placement(layout_entry)

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
        placement = EmittedEntityPlacement(
            entity=combinator,
            entity_id=anchor_id,
            position=pos,
            output_signals={},
            input_signals={},
            role="export_anchor",
            zone="south_exports",
        )
        self._record_entity_placement(placement)
        usage_entry.export_anchor_id = anchor_id

        debug_info = self._compose_debug_info(usage_entry)
        if debug_info:
            self.annotate_entity_description(combinator, debug_info)

        # Register as sink so standard wiring connects it to the producer
        self.signal_graph.add_sink(signal_id, anchor_id)

    def emit_blueprint(self, ir_operations: List[IRNode]) -> Blueprint:
        """Convert IR operations to blueprint."""
        import warnings

        from dsl_compiler.src.layout.planner import LayoutPlanner

        # Preserve caller-provided blueprint metadata across resets
        previous_label = getattr(self.blueprint, "label", "DSL Generated Blueprint")
        previous_description = getattr(self.blueprint, "description", "")

        # Reset state for a fresh emission run
        self._reset_for_emit()
        self.blueprint.label = previous_label

        planner = LayoutPlanner(
            self.signal_type_map,
            diagnostics=self.diagnostics,
            power_pole_type=self.power_pole_type,
            max_wire_span=self.wire_relay_options.normalized_span(),
        )

        layout_plan = planner.plan_layout(
            ir_operations,
            blueprint_label=previous_label,
            blueprint_description=previous_description,
        )

        # Synchronise internal state with the planner output
        self.layout = planner.layout_engine
        self.layout_plan = layout_plan
        self.signal_analyzer = planner.signal_analyzer
        self.signal_usage = planner.signal_usage
        self.materializer = planner.materializer
        self.connection_planner = planner.connection_planner
        self.signal_graph = planner.signal_graph
        self._wire_merge_junctions = dict(
            getattr(getattr(planner, "_emitter", None), "_wire_merge_junctions", {})
        )
        self.memory_builder = MemoryCircuitBuilder(self.layout, self.blueprint)
        self._prepared_operations = list(ir_operations)
        self._prepared_operation_index = {}
        self._memory_reads_by_memory = defaultdict(list)
        self._memory_write_strategies = defaultdict(set)

        source_emitter = getattr(planner, "_emitter", None)
        if source_emitter is not None:
            source_reads = getattr(source_emitter, "_memory_reads_by_memory", None)
            if source_reads:
                for memory_id, reads in source_reads.items():
                    self._memory_reads_by_memory[memory_id].extend(reads)

            source_strategies = getattr(
                source_emitter, "_memory_write_strategies", None
            )
            if source_strategies:
                for memory_id, strategies in source_strategies.items():
                    self._memory_write_strategies[memory_id].update(strategies)

        self.blueprint.label = layout_plan.blueprint_label
        self._apply_blueprint_metadata(layout_plan.blueprint_description or previous_description)

        # Collect draftsman warnings during blueprint construction
        captured_warnings = []

        def warning_handler(message, category, filename, lineno, file=None, line=None):
            captured_warnings.append((message, category, filename, lineno))

        # Set up warning capture
        old_showwarning = warnings.showwarning
        warnings.showwarning = warning_handler

        try:
            self._materialize_entities_from_plan(planner)
            self._realize_circuit_connections()
            self._materialize_power_grid()

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

    def _materialize_entities_from_plan(self, planner: LayoutPlanner) -> None:
        """Instantiate blueprint entities from a completed layout plan."""

        emitter_state = getattr(planner, "_emitter", None)
        source_entities = planner.entities or {}

        self.entities = {}
        self.power_poles = []

        for entity_id, placement in source_entities.items():
            entity_copy = copy.deepcopy(placement.entity)
            entity_copy.id = entity_id
            entity_copy.tile_position = placement.position
            entity_copy = self._add_entity(entity_copy)

            cloned = replace(placement, entity=entity_copy)
            self.entities[entity_id] = cloned

            layout_entry = self.layout_plan.entity_placements.get(entity_id)
            if layout_entry is not None:
                layout_entry.properties["entity_obj"] = entity_copy

        self.next_entity_number = len(self.entities) + 1

        # Reconstruct memory module bookkeeping for SR latch wiring
        self.memory_builder.memory_modules.clear()
        source_builder = getattr(emitter_state, "memory_builder", None)
        source_modules = getattr(source_builder, "memory_modules", {}) or {}
        for memory_id, components in source_modules.items():
            new_components = {}
            for component_name, component_placement in components.items():
                if isinstance(component_placement, EmittedEntityPlacement):
                    cloned = self.entities.get(component_placement.entity_id)
                    if cloned is not None:
                        new_components[component_name] = cloned
                else:
                    new_components[component_name] = component_placement
            if new_components:
                self.memory_builder.memory_modules[memory_id] = new_components

    def _materialize_power_grid(self) -> None:
        """Instantiate planned power poles and connect the power grid."""

        self.power_poles = []
        if not self.layout_plan.power_poles:
            return

        for pole in self.layout_plan.power_poles:
            pole_entity = new_entity(pole.pole_type)
            pole_entity.tile_position = pole.position
            pole_entity = self._add_entity(pole_entity)

            pole_id = pole.pole_id or f"power_pole_{self.next_entity_number}"
            placement = EmittedEntityPlacement(
                entity=pole_entity,
                entity_id=pole_id,
                position=pole.position,
                output_signals={},
                input_signals={},
                role="power",
            )
            self.power_poles.append(placement)

        if self.power_poles:
            self._connect_power_grid()

        self.next_entity_number = len(self.blueprint.entities) + 1

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

    def _determine_locked_wire_colors(self) -> Dict[Tuple[str, str], str]:
        """Collect wire color locks for memory modules requiring green outputs."""

        locked: Dict[Tuple[str, str], str] = {}

        for module in self.memory_builder.memory_modules.values():
            if not isinstance(module, dict):
                continue
            for placement in module.values():
                if not isinstance(placement, EmittedEntityPlacement):
                    continue
                for signal_name, color in placement.output_signals.items():
                    if color == "green":
                        locked[(placement.entity_id, signal_name)] = color

        return locked

    def _resolve_wire_color(
        self,
        source: EmittedEntityPlacement,
        sink: EmittedEntityPlacement,
        resolved_signal: Optional[str] = None,
    ) -> str:
        """Determine preferred wire color based on sink expectations."""

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

        if "memory" in source.entity_id and "output" in source.entity_id:
            return "green"
        return "red"

    def _compute_wire_distance(
        self, source: EmittedEntityPlacement, sink: EmittedEntityPlacement
    ) -> float:
        """Return distance between two placements honoring configured metric."""

        sx, sy = source.position
        tx, ty = sink.position
        if self.wire_relay_options.placement_strategy == "manhattan":
            return abs(tx - sx) + abs(ty - sy)
        return math.dist((sx, sy), (tx, ty))

    def _connect_with_wire_path(
        self,
        source: EmittedEntityPlacement,
        sink: EmittedEntityPlacement,
        wire_color: str,
        resolved_signal: Optional[str] = None,
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

        if self._plan_only_mode:
            return

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
        source: EmittedEntityPlacement,
        sink: EmittedEntityPlacement,
    ) -> List[EmittedEntityPlacement]:
        """Insert relay poles when two endpoints exceed wire reach."""

        if self._plan_only_mode:
            return []

        options = self.wire_relay_options
        if not options.enabled:
            return []

        span_limit = options.normalized_span()
        distance = self._compute_wire_distance(source, sink)
        if distance <= span_limit:
            return []

        segments = max(1, math.ceil(distance / span_limit))
        required_relays = segments - 1

        relays: List[EmittedEntityPlacement] = []
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

            placement = EmittedEntityPlacement(
                entity=pole_entity,
                entity_id=relay_id,
                position=pos,
                output_signals={},
                input_signals={},
                role="wire_relay",
                zone="infrastructure",
            )

            self._record_entity_placement(placement)
            relays.append(placement)

        return relays

    def _plan_connections(self) -> None:
        """Run the connection planner to populate the layout wire plan."""

        if self.connection_planner is None:
            raise RuntimeError(
                "ConnectionPlanner not initialised; call prepare() first"
            )

        locked_colors = self._determine_locked_wire_colors()
        self.connection_planner.plan_connections(
            self.signal_graph,
            self.entities,
            wire_merge_junctions=self._wire_merge_junctions,
            locked_colors=locked_colors,
        )

    def _realize_circuit_connections(self) -> None:
        """Convert planned wire connections into blueprint wiring."""

        for memory_id in self.memory_builder.memory_modules:
            try:
                self.memory_builder.wire_sr_latch(memory_id)
            except Exception as exc:
                self.diagnostics.error(f"Failed to wire memory {memory_id}: {exc}")

        if self.connection_planner is None:
            raise RuntimeError(
                "ConnectionPlanner not initialised; call prepare() first"
            )

        for connection in self.layout_plan.wire_connections:
            source = self.entities.get(connection.source_entity_id)
            sink = self.entities.get(connection.sink_entity_id)

            if source is None:
                self.diagnostics.error(
                    f"Source entity {connection.source_entity_id} missing for signal {connection.signal_name}"
                )
                continue

            if sink is None:
                self.diagnostics.error(
                    f"Sink entity {connection.sink_entity_id} missing for signal {connection.signal_name}"
                )
                continue

            wire_color = connection.wire_color or self._resolve_wire_color(
                source,
                sink,
                connection.signal_name,
            )

            self._connect_with_wire_path(
                source,
                sink,
                wire_color,
                resolved_signal=connection.signal_name,
            )

    def _instantiate_power_pole(self, plan: PlannedPowerPole) -> EmittedEntityPlacement:
        """Create and register a power pole entity from a planner result."""

        pole_entity = new_entity(plan.prototype)
        pole_entity.tile_position = plan.position
        pole_entity = self._add_entity(pole_entity)
        pole_id = f"power_pole_{self.next_entity_number}"
        self.next_entity_number += 1
        placement = EmittedEntityPlacement(
            entity=pole_entity,
            entity_id=pole_id,
            position=plan.position,
            output_signals={},
            input_signals={},
            role="power",
        )
        self._record_entity_placement(placement)
        self.power_poles.append(placement)
        self.layout_plan.add_power_pole(
            PowerPolePlacement(
                pole_id=pole_id,
                pole_type=plan.prototype,
                position=plan.position,
            )
        )
        return placement

    def _deploy_power_poles(self) -> None:
        """Delegate power pole placement to the layout power planner."""

        if not self.power_pole_type:
            return

        planner = PowerPlanner(self.layout, self.layout_plan, self.diagnostics)
        planned_poles = planner.plan_power_grid(self.power_pole_type)

        for plan in planned_poles:
            self._instantiate_power_pole(plan)

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


class BlueprintEmitter:
    """Materialize a :class:`LayoutPlan` into a Factorio blueprint."""

    def __init__(
        self,
        signal_type_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.signal_type_map = signal_type_map or {}
        self.diagnostics = DiagnosticCollector()
        self.blueprint = Blueprint()
        self._ensure_signal_map_registered()
        self.entity_factory = PlanEntityEmitter(self.diagnostics)

    def emit_from_plan(self, layout_plan: LayoutPlan) -> Blueprint:
        """Emit a blueprint from a completed layout plan."""

        self.blueprint = Blueprint()
        self.blueprint.label = layout_plan.blueprint_label or "DSL Generated"
        self.blueprint.description = layout_plan.blueprint_description or ""
        self.blueprint.version = (2, 0)

        entity_map: Dict[str, Entity] = {}

        for placement in layout_plan.entity_placements.values():
            entity = self.entity_factory.create_entity(placement)
            if entity is None:
                continue
            entity_map[placement.ir_node_id] = entity
            self.blueprint.entities.append(entity, copy=False)

        self._materialize_power_grid(layout_plan, entity_map)
        self._materialize_connections(layout_plan, entity_map)
        self._apply_blueprint_metadata(self.blueprint.description)

        return self.blueprint

    # ------------------------------------------------------------------
    # Connection materialisation
    # ------------------------------------------------------------------

    def _materialize_connections(
        self,
        layout_plan: LayoutPlan,
        entity_map: Dict[str, Entity],
    ) -> None:
        for connection in layout_plan.wire_connections:
            source = entity_map.get(connection.source_entity_id)
            sink = entity_map.get(connection.sink_entity_id)

            if source is None or sink is None:
                missing_id = (
                    connection.source_entity_id
                    if source is None
                    else connection.sink_entity_id
                )
                self.diagnostics.warning(
                    f"Skipped wire for '{connection.signal_name}' due to missing entity '{missing_id}'."
                )
                continue

            kwargs: Dict[str, Any] = {
                "color": connection.wire_color,
                "entity_1": source,
                "entity_2": sink,
            }
            if connection.source_side:
                kwargs["side_1"] = connection.source_side
            if connection.sink_side:
                kwargs["side_2"] = connection.sink_side

            try:
                self.blueprint.add_circuit_connection(**kwargs)
            except Exception as exc:  # pragma: no cover - draftsman errors
                self.diagnostics.error(
                    f"Failed to wire {source.id} -> {sink.id} ({connection.signal_name}): {exc}"
                )

    def _materialize_power_grid(
        self,
        layout_plan: LayoutPlan,
        entity_map: Dict[str, Entity],
    ) -> None:
        if not layout_plan.power_poles:
            return

        for pole in layout_plan.power_poles:
            try:
                entity = new_entity(pole.pole_type)
            except Exception as exc:  # pragma: no cover - draftsman errors
                self.diagnostics.error(
                    f"Failed to instantiate power pole '{pole.pole_type}': {exc}"
                )
                continue

            entity.id = pole.pole_id or entity.id
            entity.tile_position = pole.position
            self.blueprint.entities.append(entity, copy=False)
            entity_map[entity.id] = entity

        try:
            self.blueprint.generate_power_connections()
        except Exception as exc:  # pragma: no cover - draftsman warnings
            self.diagnostics.warning(
                f"Failed to auto-generate power connections: {exc}"
            )

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _ensure_signal_map_registered(self) -> None:
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
            except Exception as exc:  # pragma: no cover - draftsman errors
                self.diagnostics.warning(
                    f"Could not register signal '{name}' as {signal_type}: {exc}"
                )

    def _apply_blueprint_metadata(self, previous_description: str) -> None:
        if not hasattr(self.blueprint, "description"):
            return

        description = previous_description or ""
        note = EDGE_LAYOUT_NOTE

        if note in description:
            self.blueprint.description = description
            return

        if description:
            if not description.endswith("\n"):
                description += "\n"
            self.blueprint.description = description + note
        else:
            self.blueprint.description = note


# Transitional: expose legacy planning emitter until layout pipeline fully migrates.
LegacyBlueprintEmitter = LayoutBuilder


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
    signal_type_map = signal_type_map or {}

    emitter = BlueprintEmitter(signal_type_map)

    planner_diagnostics = DiagnosticCollector()
    from dsl_compiler.src.layout.planner import LayoutPlanner

    planner = LayoutPlanner(
        signal_type_map,
        diagnostics=planner_diagnostics,
        power_pole_type=power_pole_type,
        max_wire_span=MAX_CIRCUIT_WIRE_SPAN,
    )

    layout_plan = planner.plan_layout(
        ir_operations,
        blueprint_label=label,
        blueprint_description="",
    )

    combined_diagnostics = DiagnosticCollector()
    combined_diagnostics.diagnostics.extend(planner.diagnostics.diagnostics)

    if planner.diagnostics.has_errors():
        return Blueprint(), combined_diagnostics

    blueprint = emitter.emit_from_plan(layout_plan)

    combined_diagnostics.diagnostics.extend(emitter.diagnostics.diagnostics)

    return blueprint, combined_diagnostics


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
