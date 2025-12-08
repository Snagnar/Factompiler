"""
Blueprint emission module for the Factorio Circuit DSL.

This module converts IR operations into actual Factorio combinators and entities
using the factorio-draftsman library to generate blueprint JSON.
"""

from __future__ import annotations

from typing import Dict, Optional, Any

from draftsman.blueprintable import Blueprint
from draftsman.entity import (
    new_entity,
)  # Use draftsman's factory
from draftsman.classes.entity import Entity
from draftsman.data import signals as signal_data

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.common.constants import DEFAULT_CONFIG
from dsl_compiler.src.layout.layout_plan import (
    LayoutPlan,
)
from .entity_emitter import PlanEntityEmitter

EDGE_LAYOUT_NOTE = (
    "Edge layout: literal constants are placed along the north boundary; "
    "export anchors align along the south boundary."
)


class BlueprintEmitter:
    """Materialize a :class:`LayoutPlan` into a Factorio blueprint."""

    def __init__(
        self,
        diagnostics: ProgramDiagnostics,
        signal_type_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.signal_type_map = signal_type_map or {}
        self.diagnostics = diagnostics
        self.diagnostics.default_stage = "emission"
        self.blueprint = Blueprint()
        self._ensure_signal_map_registered()
        self.entity_factory = PlanEntityEmitter(self.diagnostics, self.signal_type_map)

    def emit_from_plan(self, layout_plan: LayoutPlan) -> Blueprint:
        """Emit a blueprint from a completed layout plan."""

        self.blueprint = Blueprint()
        self.blueprint.label = layout_plan.blueprint_label or DEFAULT_CONFIG.default_blueprint_label
        self.blueprint.description = layout_plan.blueprint_description or ""
        self.blueprint.version = (2, 0)

        entity_map: Dict[str, Entity] = {}

        # Sort placements by ID for deterministic entity ordering
        for ir_node_id in sorted(layout_plan.entity_placements.keys()):
            placement = layout_plan.entity_placements[ir_node_id]
            entity = self.entity_factory.create_entity(placement)
            if entity is None:
                self.diagnostics.error(
                    f"Failed to create entity for placement ID '{placement.ir_node_id}'."
                )
                continue

            entity_map[placement.ir_node_id] = entity
            self.blueprint.entities.append(entity, copy=False)

        self._materialize_power_grid(layout_plan, entity_map)
        self._materialize_connections(layout_plan, entity_map)
        self._apply_blueprint_metadata(self.blueprint.description)

        return self.blueprint

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
            except Exception as e:
                self.diagnostics.error(
                    f"Failed to add wire connection {connection.source_entity_id} -> {connection.sink_entity_id}: {e}"
                )

    def _materialize_power_grid(
        self,
        layout_plan: LayoutPlan,
        entity_map: Dict[str, Entity],
    ) -> None:
        """Materialize power grid and generate connections between poles.

        Note: Power poles are now added as entity_placements, so they're already
        created by the entity factory. This method just handles legacy power_poles
        list and generates the copper wire connections.
        """
        for pole in layout_plan.power_poles:
            if pole.pole_id in entity_map:
                continue  # Already created via entity_placements

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
            self.blueprint.generate_power_connections(prefer_axis=True, only_axis=True)
        except Exception as exc:  # pragma: no cover - draftsman warnings
            self.diagnostics.warning(
                f"Failed to auto-generate power connections: {exc}"
            )

    def _ensure_signal_map_registered(self) -> None:
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
