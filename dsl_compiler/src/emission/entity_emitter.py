"""Blueprint entity materialization helpers built on LayoutPlan."""

from __future__ import annotations

from typing import Dict, Optional, Any

import copy

from draftsman.classes.entity import Entity  # type: ignore[import-not-found]
from draftsman.entity import new_entity  # type: ignore[import-not-found]
from draftsman.entity import DeciderCombinator, ArithmeticCombinator, ConstantCombinator  # type: ignore[import-not-found]

from dsl_compiler.src.layout.layout_plan import EntityPlacement
from dsl_compiler.src.ir import SignalRef
from dsl_compiler.src.semantic import DiagnosticCollector


class PlanEntityEmitter:
    """Instantiate Draftsman entities from LayoutPlan placements."""

    def __init__(
        self,
        diagnostics: Optional[DiagnosticCollector] = None,
        signal_type_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.diagnostics = diagnostics or DiagnosticCollector()
        self.signal_type_map = signal_type_map or {}

    def create_entity(self, placement: EntityPlacement) -> Optional[Entity]:
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

            # Handle special entity types that need structured configuration
            if placement.entity_type == "decider-combinator":
                self._configure_decider(entity, placement.properties)
            elif placement.entity_type == "arithmetic-combinator":
                self._configure_arithmetic(entity, placement.properties)
            elif placement.entity_type == "constant-combinator":
                self._configure_constant(entity, placement.properties)
            else:
                # Generic property setting for other entity types
                for key, value in placement.properties.items():
                    if key in {"entity_obj", "footprint", "property_writes"}:
                        continue
                    if hasattr(entity, key):
                        try:
                            setattr(entity, key, value)
                        except Exception:
                            # Preserve diagnostics for consistency with legacy emitter.
                            self.diagnostics.warning(
                                f"Could not set property '{key}' on '{placement.entity_type}'."
                            )

        # Apply property writes (e.g., entity.enable = signal)
        property_writes = placement.properties.get("property_writes", {})
        if property_writes:
            self._apply_property_writes(entity, property_writes, placement)

        entity.id = placement.ir_node_id
        entity.tile_position = placement.position
        return entity

    def _configure_decider(self, entity: DeciderCombinator, props: Dict[str, Any]) -> None:
        """Configure a decider combinator from placement properties."""
        operation = props.get("operation", "=")
        left_operand = props.get("left_operand")
        right_operand = props.get("right_operand")
        output_signal = props.get("output_signal")
        output_value = props.get("output_value", 1)
        copy_count = props.get("copy_count_from_input", False)

        # Build condition
        condition_kwargs = {"comparator": operation}
        if isinstance(left_operand, int):
            condition_kwargs["first_signal"] = "signal-0"
            condition_kwargs["constant"] = left_operand
        else:
            condition_kwargs["first_signal"] = left_operand

        if isinstance(right_operand, int):
            condition_kwargs["constant"] = right_operand
        else:
            condition_kwargs["second_signal"] = right_operand

        entity.conditions = [DeciderCombinator.Condition(**condition_kwargs)]

        # Build output
        output_kwargs = {
            "signal": output_signal,
            "copy_count_from_input": copy_count,
        }
        if not copy_count and isinstance(output_value, int):
            output_kwargs["constant"] = output_value

        entity.outputs = [DeciderCombinator.Output(**output_kwargs)]

    def _configure_arithmetic(self, entity: ArithmeticCombinator, props: Dict[str, Any]) -> None:
        """Configure an arithmetic combinator from placement properties."""
        operation = props.get("operation", "+")
        left_operand = props.get("left_operand")
        right_operand = props.get("right_operand")
        output_signal = props.get("output_signal")

        # Validate signal-each usage per Draftsman requirements
        if output_signal == "signal-each":
            # signal-each can only be output if at least one input is signal-each
            if left_operand != "signal-each" and right_operand != "signal-each":
                # Fallback to signal-0 to avoid Draftsman warning
                output_signal = "signal-0"

        entity.first_operand = left_operand
        entity.second_operand = right_operand
        entity.operation = operation
        entity.output_signal = output_signal

    def _configure_constant(self, entity: ConstantCombinator, props: Dict[str, Any]) -> None:
        """Configure a constant combinator from placement properties."""
        signal_name = props.get("signal_name")
        signal_type = props.get("signal_type")
        value = props.get("value", 0)

        if signal_name:
            section = entity.add_section()
            section.set_signal(0, signal_name, value)

    def _apply_property_writes(self, entity: Entity, property_writes: Dict[str, Any], placement: EntityPlacement) -> None:
        """Apply property writes to entity (e.g., entity.enable = signal)."""
        for prop_name, prop_data in property_writes.items():
            if prop_name == "enable":
                # Circuit enable condition
                if prop_data["type"] == "signal":
                    # Signal-controlled enable
                    signal_ref = prop_data["signal_ref"]
                    
                    # Resolve signal name
                    signal_type = signal_ref.signal_type
                    signal_name = self.signal_type_map.get(signal_type, signal_type)
                    
                    # Determine signal type (virtual, item, fluid, etc.)
                    # For now, assume virtual signals (most common case)
                    signal_dict = {
                        "name": signal_name,
                        "type": "virtual"
                    }
                    
                    # Check if entity supports circuit_enabled
                    if hasattr(entity, 'circuit_enabled'):
                        entity.circuit_enabled = True
                        # Set the circuit condition
                        if hasattr(entity, 'set_circuit_condition'):
                            entity.set_circuit_condition(
                                signal_dict,
                                ">",
                                0
                            )
                    else:
                        # Fallback to control_behavior dict
                        if not hasattr(entity, 'control_behavior'):
                            entity.control_behavior = {}
                        entity.control_behavior["circuit_enabled"] = True
                        entity.control_behavior["circuit_condition"] = {
                            "first_signal": signal_dict,
                            "comparator": ">",
                            "constant": 0
                        }
                elif prop_data["type"] == "constant":
                    # Constant enable/disable
                    if hasattr(entity, 'circuit_enabled'):
                        entity.circuit_enabled = bool(prop_data["value"])
                    else:
                        if not hasattr(entity, 'control_behavior'):
                            entity.control_behavior = {}
                        entity.control_behavior["circuit_enabled"] = bool(prop_data["value"])
            else:
                # Other properties - try direct assignment
                try:
                    setattr(entity, prop_name, prop_data.get("value"))
                except Exception:
                    self.diagnostics.warning(
                        f"Could not set property '{prop_name}' on '{placement.entity_type}'."
                    )


    def create_entity_map(
        self, layout_plan: Any
    ) -> Dict[str, Entity]:
        """Instantiate entities for the entire layout plan."""

        entities: Dict[str, Entity] = {}
        for placement in layout_plan.entity_placements.values():
            entity = self.create_entity(placement)
            if entity is not None:
                entities[placement.ir_node_id] = entity
        return entities


__all__ = ["PlanEntityEmitter"]
