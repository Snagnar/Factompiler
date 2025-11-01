# emit.py
"""
Blueprint emission module for the Factorio Circuit DSL.

This module converts IR operations into actual Factorio combinators and entities
using the factorio-draftsman library to generate blueprint JSON.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Literal, TYPE_CHECKING

# Add draftsman to path
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent.parent / "factorio-draftsman")
)

from draftsman.blueprintable import Blueprint  # type: ignore[import-not-found]
from draftsman.entity import (
    new_entity,
)  # Use draftsman's factory  # type: ignore[import-not-found]
from draftsman.entity import *  # Import all entities  # type: ignore[import-not-found]
from draftsman.classes.entity import Entity  # type: ignore[import-not-found]
from draftsman.data import signals as signal_data  # type: ignore[import-not-found]


from ..ir import *
from ..semantic import DiagnosticCollector

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


from dsl_compiler.src.layout.layout_plan import (
    LayoutPlan,
)
from .entity_emitter import PlanEntityEmitter

if TYPE_CHECKING:  # pragma: no cover - type checking helper
    pass


# =============================================================================
# Debug Formatting
# =============================================================================


def format_entity_description(debug_info: Optional[dict]) -> str:
    """Format entity debug information into a human-readable description.

    This is a minimal stub - the full implementation was removed as legacy code.
    """
    if not debug_info:
        return ""

    # Simple format: just show the operation type and variable if available
    parts = []
    if "operation" in debug_info:
        parts.append(f"op={debug_info['operation']}")
    if "variable" in debug_info:
        parts.append(f"var={debug_info['variable']}")

    return ", ".join(parts) if parts else ""


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


# LayoutBuilder class removed (dead code)


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
