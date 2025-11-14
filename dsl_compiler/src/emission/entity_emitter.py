from __future__ import annotations
from typing import Dict, Optional, Any
import copy
from draftsman.classes.entity import Entity  # type: ignore[import-not-found]
from draftsman.entity import new_entity  # type: ignore[import-not-found]
from draftsman.entity import DeciderCombinator, ArithmeticCombinator, ConstantCombinator  # type: ignore[import-not-found]
from draftsman.data import items, fluids  # type: ignore[import-not-found]
from dsl_compiler.src.layout.layout_plan import EntityPlacement
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics

"""Blueprint entity materialization helpers built on LayoutPlan."""


def _infer_signal_type(signal_name: str) -> str:
    """Infer the type of a signal from its name by checking game data.

    Returns 'item', 'fluid', or 'virtual' based on what's found in Draftsman data.
    """
    # Check if it's an item
    if items.raw and signal_name in items.raw:
        return "item"

    # Check if it's a fluid
    if fluids.raw and signal_name in fluids.raw:
        return "fluid"

    # Default to virtual for signals like signal-A, signal-B, etc.
    return "virtual"


def format_entity_description(debug_info: Optional[dict]) -> str:
    """Format entity debug information into a human-readable description.

    Format: "variable_name in file.fcdsl at line X (operation details)"
    Or: "file.fcdsl:X (operation details)" if no variable name

    Args:
        debug_info: Dict with keys: variable, operation, details, signal_type,
                    source_file, line, role

    Returns:
        Formatted description string
    """
    if not debug_info:
        return ""

    parts = []

    # Build location string
    location_parts = []
    if debug_info.get("variable"):
        location_parts.append(debug_info["variable"])

    if debug_info.get("source_file"):
        file_name = debug_info["source_file"]
        # Extract just filename if it's a path
        if "/" in file_name or "\\" in file_name:
            file_name = file_name.split("/")[-1].split("\\")[-1]

        if debug_info.get("line"):
            location_parts.append(f"in {file_name} at line {debug_info['line']}")
        else:
            location_parts.append(f"in {file_name}")
    elif debug_info.get("line"):
        location_parts.append(f"at line {debug_info['line']}")

    if location_parts:
        parts.append(" ".join(location_parts))

    # Build operation description
    op_parts = []
    if debug_info.get("operation"):
        op_parts.append(debug_info["operation"])

    if debug_info.get("details"):
        op_parts.append(debug_info["details"])

    if debug_info.get("signal_type"):
        op_parts.append(f"type={debug_info['signal_type']}")

    if op_parts:
        parts.append(f"({', '.join(op_parts)})")

    return " ".join(parts)


class PlanEntityEmitter:
    """Instantiate Draftsman entities from LayoutPlan placements."""

    def __init__(
        self,
        diagnostics: Optional[ProgramDiagnostics] = None,
        signal_type_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.diagnostics = diagnostics or ProgramDiagnostics()
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

        # Set entity description from debug info
        # Set entity description from debug info
        debug_info = placement.properties["debug_info"]
        description = format_entity_description(debug_info)
        entity.player_description = description
        # if not hasattr(entity, "tags") or entity.tags is None:
        #     entity.tags = {}
        # entity.tags["description"] = description
        self.diagnostics.info(
            f"Set description tag for {placement.entity_type} '{placement.ir_node_id}'"
        )

        # Log when description is successfully set (only in debug mode)
        if self.diagnostics.log_level == "debug":
            self.diagnostics.info(
                f"Entity description: {description[:80]}{'...' if len(description) > 80 else ''}"
            )
        # debug_info = placement.properties.get("debug_info")
        #     description = format_entity_description(debug_info)
        #                 if not hasattr(entity, "tags"):
        #                     entity.tags = {}
        #                 elif entity.tags is None:
        #                     entity.tags = {}

        #                 entity.tags["description"] = description
        #                 description_set = True
        #                 self.diagnostics.info(
        #                     f"Set description tag for {placement.entity_type} '{placement.ir_node_id}'"
        #                 )
        #             except Exception as fallback_e:
        #                 self.diagnostics.warning(
        #                     f"Could not set description for {placement.entity_type} '{placement.ir_node_id}': "
        #                     f"direct set failed ({e}), tags fallback also failed ({fallback_e})"
        #                 )

        #         # Log when description is successfully set (only in debug mode)
        #         if description_set and self.diagnostics.log_level == "debug":
        #             self.diagnostics.info(
        #                 f"Entity description: {description[:80]}{'...' if len(description) > 80 else ''}"
        #             )

        entity.id = placement.ir_node_id

        # Set entity position from placement
        # placement.position is already in CENTER coordinates (set by ClusterPacker)
        # which matches draftsman's entity.position
        if placement.position is not None:
            entity.position = placement.position

        return entity

    def _configure_decider(
        self, entity: DeciderCombinator, props: Dict[str, Any]
    ) -> None:
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

    def _configure_arithmetic(
        self, entity: ArithmeticCombinator, props: Dict[str, Any]
    ) -> None:
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

    def _configure_constant(
        self, entity: ConstantCombinator, props: Dict[str, Any]
    ) -> None:
        """Configure a constant combinator from placement properties."""
        signal_name = props.get("signal_name")
        props.get("signal_type")
        value = props.get("value", 0)

        if signal_name:
            section = entity.add_section()
            section.set_signal(0, signal_name, value)

    def _apply_property_writes(
        self,
        entity: Entity,
        property_writes: Dict[str, Any],
        placement: EntityPlacement,
    ) -> None:
        """Apply property writes to entity (e.g., entity.enable = signal)."""
        for prop_name, prop_data in property_writes.items():
            if prop_name == "enable":
                prop_type = prop_data.get("type")

                # ✅ Handle inline comparisons
                if prop_type == "inline_comparison":
                    comp_data = prop_data.get("comparison_data", {})
                    left_signal = comp_data.get("left_signal")
                    comparator = comp_data.get("comparator")
                    right_constant = comp_data.get("right_constant")

                    # Resolve signal dict - infer type from the signal name itself
                    if isinstance(left_signal, str):
                        # Look up the signal type from game data, not from DSL signal types
                        signal_category = _infer_signal_type(left_signal)
                        signal_dict = {"name": left_signal, "type": signal_category}
                    else:
                        # Shouldn't happen, but handle it
                        signal_dict = {"name": "signal-0", "type": "virtual"}

                    # Apply circuit condition
                    if hasattr(entity, "circuit_enabled"):
                        entity.circuit_enabled = True
                        if hasattr(entity, "set_circuit_condition"):
                            entity.set_circuit_condition(
                                signal_dict, comparator, right_constant
                            )
                        else:
                            # Fallback to control_behavior dict
                            if not hasattr(entity, "control_behavior"):
                                entity.control_behavior = {}
                            entity.control_behavior["circuit_condition"] = {
                                "first_signal": signal_dict,
                                "comparator": comparator,
                                "constant": right_constant,
                            }
                    else:
                        if not hasattr(entity, "control_behavior"):
                            entity.control_behavior = {}
                        entity.control_behavior["circuit_enabled"] = True
                        entity.control_behavior["circuit_condition"] = {
                            "first_signal": signal_dict,
                            "comparator": comparator,
                            "constant": right_constant,
                        }
                    continue

                # Circuit enable condition
                if prop_type == "signal":
                    # Signal-controlled enable
                    signal_ref = prop_data["signal_ref"]

                    # Resolve signal name from the DSL signal identifier
                    signal_type_key = signal_ref.signal_type
                    signal_info = self.signal_type_map.get(signal_type_key)

                    if signal_info and isinstance(signal_info, dict):
                        signal_name = signal_info.get("name", signal_type_key)
                        signal_category = signal_info.get("type", "virtual")
                    else:
                        # Fallback for old string format or missing entry
                        signal_name = signal_info if signal_info else signal_type_key
                        signal_category = "virtual"

                    signal_dict = {"name": signal_name, "type": signal_category}

                    # Check if entity supports circuit_enabled
                    if hasattr(entity, "circuit_enabled"):
                        entity.circuit_enabled = True
                        # Set the circuit condition
                        if hasattr(entity, "set_circuit_condition"):
                            entity.set_circuit_condition(signal_dict, ">", 0)
                    else:
                        # Fallback to control_behavior dict
                        if not hasattr(entity, "control_behavior"):
                            entity.control_behavior = {}
                        entity.control_behavior["circuit_enabled"] = True
                        entity.control_behavior["circuit_condition"] = {
                            "first_signal": signal_dict,
                            "comparator": ">",
                            "constant": 0,
                        }
                elif prop_data["type"] == "constant":
                    # Constant enable/disable
                    if hasattr(entity, "circuit_enabled"):
                        entity.circuit_enabled = bool(prop_data["value"])
                    else:
                        if not hasattr(entity, "control_behavior"):
                            entity.control_behavior = {}
                        entity.control_behavior["circuit_enabled"] = bool(
                            prop_data["value"]
                        )
            else:
                # Other properties - try direct assignment
                try:
                    setattr(entity, prop_name, prop_data.get("value"))
                except Exception:
                    self.diagnostics.warning(
                        f"Could not set property '{prop_name}' on '{placement.entity_type}'."
                    )


__all__ = ["PlanEntityEmitter"]
