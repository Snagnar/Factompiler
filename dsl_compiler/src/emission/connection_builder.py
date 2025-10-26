"""Connection planning helpers for blueprint emission."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from draftsman.entity import new_entity  # type: ignore[import-not-found]

from ..ir import SignalRef
from .signals import EntityPlacement
from .wiring import CircuitEdge, WIRE_COLORS, collect_circuit_edges, plan_wire_colors

if TYPE_CHECKING:  # pragma: no cover - type checking aid
    from .emitter import BlueprintEmitter


WIRE_RELAY_ENTITY = "medium-electric-pole"


class ConnectionBuilder:
    """Responsible for assigning wire colors and drawing circuit connections."""

    def __init__(self, parent: "BlueprintEmitter") -> None:
        self._parent = parent
        self._wire_relay_counter = 0
        self._circuit_edges: List[CircuitEdge] = []
        self._node_color_assignments: Dict[Tuple[str, str], str] = {}
        self._edge_color_map: Dict[Tuple[str, str, str], str] = {}
        self._coloring_conflicts = []
        self._coloring_success = True

    def __getattr__(self, name):  # pragma: no cover - delegation helper
        return getattr(self._parent, name)

    # ------------------------------------------------------------------
    # Planning pipeline
    # ------------------------------------------------------------------

    def prepare_wiring_plan(self) -> None:
        """Gather connectivity and decide wire colors ahead of emission."""

        self._circuit_edges = []
        self._node_color_assignments = {}
        self._edge_color_map = {}
        self._coloring_conflicts = []
        self._coloring_success = True
        self._wire_relay_counter = 0

        base_edges = collect_circuit_edges(
            self.signal_graph, self.signal_usage, self.entities
        )

        expanded_edges: List[CircuitEdge] = []
        for edge in base_edges:
            merge_info = self._wire_merge_junctions.get(edge.source_entity_id or "")
            if merge_info:
                for source_ref in merge_info.get("sources", []):
                    if not isinstance(source_ref, SignalRef):
                        continue
                    actual_source_id = source_ref.source_id
                    source_entity_type = None
                    placement = self.entities.get(actual_source_id)
                    if placement:
                        source_entity_type = type(placement.entity).__name__
                    expanded_edges.append(
                        CircuitEdge(
                            logical_signal_id=edge.logical_signal_id,
                            resolved_signal_name=edge.resolved_signal_name,
                            source_entity_id=actual_source_id,
                            sink_entity_id=edge.sink_entity_id,
                            source_entity_type=source_entity_type,
                            sink_entity_type=edge.sink_entity_type,
                            sink_role=edge.sink_role,
                        )
                    )
            else:
                expanded_edges.append(edge)

        self._circuit_edges = expanded_edges

        sink_conflicts: Dict[str, Dict[str, set[str]]] = defaultdict(
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
            self._edge_color_map = {}

    def create_circuit_connections(self) -> None:
        """Draw circuit wires according to the planned assignments."""

        for memory_id in self.memory_builder.memory_modules:
            try:
                self.memory_builder.wire_sr_latch(memory_id)
            except Exception as exc:
                self.diagnostics.error(f"Failed to wire memory {memory_id}: {exc}")

        for (
            signal_id,
            source_entity_id,
            sink_entities,
        ) in self.signal_graph.iter_edges():
            if source_entity_id in self._wire_merge_junctions:
                merge_info = self._wire_merge_junctions[source_entity_id]
                usage_entry = self.signal_usage.get(signal_id)
                resolved_signal = (
                    usage_entry.resolved_signal_name
                    if usage_entry and usage_entry.resolved_signal_name
                    else signal_id
                )

                for sink_entity_id in sink_entities:
                    sink_placement = self.entities.get(sink_entity_id)
                    if not sink_placement:
                        continue

                    for source_ref in merge_info.get("sources", []):
                        if not isinstance(source_ref, SignalRef):
                            continue

                        actual_source_id = source_ref.source_id
                        source_placement = self.entities.get(actual_source_id)
                        if not source_placement:
                            continue

                        wire_color = self._edge_color_map.get(
                            (actual_source_id, sink_entity_id, resolved_signal)
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
                continue

            if source_entity_id:
                if source_entity_id in self.entities:
                    source_placement = self.entities[source_entity_id]

                    for sink_entity_id in sink_entities:
                        if sink_entity_id in self.entities:
                            sink_placement = self.entities[sink_entity_id]

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

    # ------------------------------------------------------------------
    # Wiring helpers
    # ------------------------------------------------------------------

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

        if "memory" in source.entity_id and "output" in source.entity_id:
            return "green"
        return "red"

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
