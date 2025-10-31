"""Blueprint entity materialization helpers built on LayoutPlan."""

from __future__ import annotations

from typing import Dict, Optional

import copy

from draftsman.classes.entity import Entity  # type: ignore[import-not-found]
from draftsman.entity import new_entity  # type: ignore[import-not-found]

from dsl_compiler.src.layout.layout_plan import LayoutPlan
from dsl_compiler.src.semantic import DiagnosticCollector


class PlanEntityEmitter:
    """Instantiate Draftsman entities from LayoutPlan placements."""

    def __init__(self, diagnostics: Optional[DiagnosticCollector] = None) -> None:
        self.diagnostics = diagnostics or DiagnosticCollector()

    def create_entity(self, placement: LayoutPlan.EntityPlacement) -> Optional[Entity]:
        """Create a Draftsman entity matching ``placement``.

        Returns the instantiated entity or ``None`` if instantiation fails.
        """

        template = placement.properties.get("entity_obj")
        entity: Optional[Entity]

        if template is not None:
            # Existing Draftsman entity provided by planner; clone to avoid mutation.
            entity = copy.deepcopy(template)
        else:
            try:
                entity = new_entity(placement.entity_type)
            except Exception as exc:  # pragma: no cover - draftsman errors
                self.diagnostics.error(
                    f"Failed to instantiate entity '{placement.entity_type}': {exc}"
                )
                return None

            for key, value in placement.properties.items():
                if key in {"entity_obj", "footprint"}:
                    continue
                if hasattr(entity, key):
                    try:
                        setattr(entity, key, value)
                    except Exception:
                        # Preserve diagnostics for consistency with legacy emitter.
                        self.diagnostics.warning(
                            f"Could not set property '{key}' on '{placement.entity_type}'."
                        )

        entity.id = placement.ir_node_id
        entity.tile_position = placement.position
        return entity

    def create_entity_map(
        self, layout_plan: LayoutPlan
    ) -> Dict[str, Entity]:
        """Instantiate entities for the entire layout plan."""

        entities: Dict[str, Entity] = {}
        for placement in layout_plan.entity_placements.values():
            entity = self.create_entity(placement)
            if entity is not None:
                entities[placement.ir_node_id] = entity
        return entities


__all__ = ["PlanEntityEmitter"]
