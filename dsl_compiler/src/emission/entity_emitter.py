from __future__ import annotations
from typing import Dict, Optional, Any, List, Set
import copy
from draftsman.classes.entity import Entity  # type: ignore[import-not-found]
from draftsman.entity import new_entity  # type: ignore[import-not-found]
from draftsman.entity import DeciderCombinator, ArithmeticCombinator, ConstantCombinator  # type: ignore[import-not-found]
from draftsman.signatures import CircuitNetworkSelection  # type: ignore[import-not-found]
from draftsman.data import items, fluids  # type: ignore[import-not-found]
from dsl_compiler.src.layout.layout_plan import EntityPlacement
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics

"""Blueprint entity materialization helpers built on LayoutPlan."""


def _infer_signal_type(signal_name: str) -> str:
    """Infer the type of a signal from its name by checking game data.

    Returns 'item', 'fluid', or 'virtual' based on what's found in Draftsman data.
    """
    if items.raw and signal_name in items.raw:
        return "item"

    if fluids.raw and signal_name in fluids.raw:
        return "fluid"

    return "virtual"


def format_entity_description(debug_info: Optional[dict]) -> str:
    """Format entity debug information into a human-readable description.

    Generates user-friendly descriptions that always include line numbers
    when available, and provide context about what computation the entity
    is part of.
    """
    if not debug_info:
        return ""

    parts = []

    # Extract key information
    var = debug_info.get("variable", "")
    expr_context = debug_info.get("expr_context")  # What variable this computes
    file_name = debug_info.get("source_file", "")
    if "/" in file_name or "\\" in file_name:
        file_name = file_name.split("/")[-1].split("\\")[-1]
    line = debug_info.get("line")
    operation = debug_info.get("operation")
    details = debug_info.get("details")
    signal_type = debug_info.get("signal_type")
    role = debug_info.get("role")

    # Build primary identifier
    # For intermediates, show context: "computing <target_var>"
    # For final results, show the variable name
    is_intermediate = var and var.startswith(
        ("arith_", "decider_", "const_", "wire_merge_")
    )

    if is_intermediate and expr_context:
        # This is an intermediate computation for a named variable
        primary_name = f"computing {expr_context}"
    elif var:
        primary_name = var
    else:
        primary_name = "entity"

    # Build location string - ALWAYS include line if available
    location_str = ""
    if line:
        if file_name:
            location_str = f"[{file_name}:{line}]"
        else:
            location_str = f"[line {line}]"
    elif file_name:
        location_str = f"[{file_name}]"

    # Build operation description
    op_desc = ""
    if operation == "arith" and details:
        # Extract operation from details like "op=+"
        op_match = details.replace("op=", "")
        op_desc = f"({op_match})"
    elif operation == "decider" and details:
        # Extract condition from details like "cond=>"
        cond_match = details.replace("cond=", "")
        op_desc = f"({cond_match})"
    elif operation == "const" and details:
        op_desc = f"({details})"
    elif operation == "memory" and details:
        op_desc = f"({operation}: {details})"
    elif operation == "output":
        op_desc = "(output anchor)"
    elif role:
        op_desc = f"({role})"
    elif operation:
        op_desc = f"({operation})"

    # Combine parts - prioritize readability
    # Format: "[file:line] name (operation) -> signal_type"
    result_parts = []

    # Always lead with location for easy identification
    if location_str:
        result_parts.append(location_str)

    result_parts.append(primary_name)

    if op_desc:
        result_parts.append(op_desc)

    if signal_type:
        result_parts.append(f"-> {signal_type}")

    return " ".join(result_parts)


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
            entity = copy.deepcopy(template)
        else:
            try:
                entity = new_entity(placement.entity_type)
            except Exception as exc:  # pragma: no cover - draftsman errors
                self.diagnostics.error(
                    f"Failed to instantiate entity '{placement.entity_type}': {exc}"
                )
                return None

            if placement.entity_type == "decider-combinator":
                self._configure_decider(entity, placement.properties)
            elif placement.entity_type == "arithmetic-combinator":
                self._configure_arithmetic(entity, placement.properties)
            elif placement.entity_type == "constant-combinator":
                self._configure_constant(entity, placement.properties)
            else:
                for key, value in placement.properties.items():
                    if key in {"entity_obj", "footprint", "property_writes"}:
                        continue
                    if hasattr(entity, key):
                        try:
                            if isinstance(getattr(entity, key), bool):
                                setattr(entity, key, bool(value))
                            else:
                                setattr(entity, key, value)
                        except Exception as e:
                            self.diagnostics.info(
                                f"Could not set property '{key}' on '{placement.entity_type}': {e}"
                            )

        property_writes = placement.properties.get("property_writes", {})
        if property_writes:
            self._apply_property_writes(entity, property_writes, placement)

        # Set entity description from debug info
        debug_info = placement.properties.get("debug_info", {})
        description = format_entity_description(debug_info)
        entity.player_description = description

        entity.id = placement.ir_node_id

        if placement.position is not None:
            entity.position = placement.position

        return entity

    def _configure_decider(
        self, entity: DeciderCombinator, props: Dict[str, Any]
    ) -> None:
        """Configure a decider combinator from placement properties.

        Supports both legacy single-condition mode and Factorio 2.0 multi-condition mode.
        Multi-condition mode is used when 'conditions' or 'multi_conditions' key is present in props.
        """
        # Check for multi-condition mode (Factorio 2.0)
        # 'conditions' is from IR layer, 'multi_conditions' is from memory_builder for SR latches
        conditions_list = props.get("conditions") or props.get("multi_conditions")
        if conditions_list:
            self._configure_decider_multi_condition(entity, props, conditions_list)
            return

        # Legacy single-condition mode
        operation = props.get("operation", "=")
        left_operand = props.get("left_operand")
        right_operand = props.get("right_operand")
        output_signal = props.get("output_signal")
        output_value = props.get("output_value", 1)
        copy_count = props.get("copy_count_from_input", False)

        left_operand_wires = props.get("left_operand_wires", {"red", "green"})
        right_operand_wires = props.get("right_operand_wires", {"red", "green"})

        condition_kwargs = {"comparator": operation}
        if isinstance(left_operand, int):
            condition_kwargs["first_signal"] = "signal-0"
            condition_kwargs["constant"] = left_operand
        else:
            condition_kwargs["first_signal"] = left_operand
            condition_kwargs["first_signal_networks"] = left_operand_wires

        if isinstance(right_operand, int):
            condition_kwargs["constant"] = right_operand
        else:
            condition_kwargs["second_signal"] = right_operand
            condition_kwargs["second_signal_networks"] = right_operand_wires

        entity.conditions = [DeciderCombinator.Condition(**condition_kwargs)]

        output_kwargs = {
            "signal": output_signal,
            "copy_count_from_input": copy_count,
        }
        if not copy_count and isinstance(output_value, int):
            output_kwargs["constant"] = output_value

        entity.outputs = [DeciderCombinator.Output(**output_kwargs)]

    def _configure_decider_multi_condition(
        self,
        entity: DeciderCombinator,
        props: Dict[str, Any],
        conditions_list: List[Dict[str, Any]],
    ) -> None:
        """Configure a decider combinator with multiple conditions (Factorio 2.0).

        Each condition dict can have:
        - comparator: Comparison operator (default: ">")
        - first_signal: Left-hand signal name
        - first_constant: Left-hand constant (alternative to first_signal)
        - first_signal_wires: Set of wire colors to read first_signal from
        - second_signal: Right-hand signal name
        - second_constant: Right-hand constant (alternative to second_signal)
        - second_signal_wires: Set of wire colors to read second_signal from
        - compare_type: "or" or "and" (how to combine with previous condition)
        """
        draftsman_conditions = []

        for cond in conditions_list:
            cond_kwargs: Dict[str, Any] = {
                "comparator": cond.get("comparator", ">"),
                "compare_type": cond.get("compare_type", "or"),
            }

            # Handle first operand (left side)
            first_signal = cond.get("first_signal")
            first_constant = cond.get("first_constant")
            first_wires = cond.get("first_signal_wires")

            if first_signal:
                cond_kwargs["first_signal"] = first_signal
                if first_wires:
                    cond_kwargs["first_signal_networks"] = self._wires_to_network_selection(first_wires)
            elif first_constant is not None:
                cond_kwargs["first_signal"] = "signal-0"
                cond_kwargs["constant"] = first_constant

            # Handle second operand (right side)
            second_signal = cond.get("second_signal")
            second_constant = cond.get("second_constant")
            second_wires = cond.get("second_signal_wires")

            if second_signal:
                cond_kwargs["second_signal"] = second_signal
                if second_wires:
                    cond_kwargs["second_signal_networks"] = self._wires_to_network_selection(second_wires)
            elif second_constant is not None:
                cond_kwargs["constant"] = second_constant

            draftsman_conditions.append(DeciderCombinator.Condition(**cond_kwargs))

        entity.conditions = draftsman_conditions

        # Configure output
        output_signal = props.get("output_signal")
        output_value = props.get("output_value", 1)
        copy_count = props.get("copy_count_from_input", False)

        output_kwargs: Dict[str, Any] = {
            "signal": output_signal,
            "copy_count_from_input": copy_count,
        }
        if not copy_count and isinstance(output_value, int):
            output_kwargs["constant"] = output_value

        entity.outputs = [DeciderCombinator.Output(**output_kwargs)]

    def _wires_to_network_selection(self, wires: Set[str]) -> CircuitNetworkSelection:
        """Convert a set of wire colors to a CircuitNetworkSelection."""
        return CircuitNetworkSelection(
            red="red" in wires,
            green="green" in wires,
        )

    def _configure_arithmetic(
        self, entity: ArithmeticCombinator, props: Dict[str, Any]
    ) -> None:
        """Configure an arithmetic combinator from placement properties."""
        operation = props.get("operation", "+")
        left_operand = props.get("left_operand")
        right_operand = props.get("right_operand")
        output_signal = props.get("output_signal")

        left_operand_wires = props.get("left_operand_wires", {"red", "green"})
        right_operand_wires = props.get("right_operand_wires", {"red", "green"})

        if output_signal == "signal-each":
            if left_operand != "signal-each" and right_operand != "signal-each":
                output_signal = "signal-0"

        entity.first_operand = left_operand
        entity.second_operand = right_operand
        entity.operation = operation
        entity.output_signal = output_signal

        # Convert wire sets to CircuitNetworkSelection
        entity.first_operand_wires = self._wires_to_network_selection(left_operand_wires)
        entity.second_operand_wires = self._wires_to_network_selection(right_operand_wires)

    def _configure_constant(
        self, entity: ConstantCombinator, props: Dict[str, Any]
    ) -> None:
        """Configure a constant combinator from placement properties."""
        # Handle multi-signal constants (bundles)
        signals = props.get("signals")
        if signals:
            section = entity.add_section()
            for idx, (signal_name, value) in enumerate(signals.items()):
                section.set_signal(idx, signal_name, value)
            return

        # Single signal constant
        signal_name = props.get("signal_name")
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

                if prop_type == "inline_comparison":
                    comp_data = prop_data.get("comparison_data", {})
                    left_signal = comp_data.get("left_signal")
                    comparator = comp_data.get("comparator")
                    right_constant = comp_data.get("right_constant")

                    if isinstance(left_signal, str):
                        signal_category = _infer_signal_type(left_signal)
                        signal_dict = {"name": left_signal, "type": signal_category}
                    else:
                        signal_dict = {"name": "signal-0", "type": "virtual"}

                    if hasattr(entity, "circuit_enabled"):
                        entity.circuit_enabled = True
                        if hasattr(entity, "set_circuit_condition"):
                            entity.set_circuit_condition(
                                signal_dict, comparator, right_constant
                            )
                        else:
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

                if prop_type == "signal":
                    signal_ref = prop_data["signal_ref"]

                    signal_type_key = signal_ref.signal_type
                    signal_info = self.signal_type_map.get(signal_type_key)

                    if signal_info and isinstance(signal_info, dict):
                        signal_name = signal_info.get("name", signal_type_key)
                        signal_category = signal_info.get("type", "virtual")
                    else:
                        signal_name = signal_info if signal_info else signal_type_key
                        signal_category = "virtual"

                    signal_dict = {"name": signal_name, "type": signal_category}

                    if hasattr(entity, "circuit_enabled"):
                        entity.circuit_enabled = True
                        if hasattr(entity, "set_circuit_condition"):
                            entity.set_circuit_condition(signal_dict, ">", 0)
                    else:
                        if not hasattr(entity, "control_behavior"):
                            entity.control_behavior = {}
                        entity.control_behavior["circuit_enabled"] = True
                        entity.control_behavior["circuit_condition"] = {
                            "first_signal": signal_dict,
                            "comparator": ">",
                            "constant": 0,
                        }
                elif prop_data["type"] == "constant":
                    if hasattr(entity, "circuit_enabled"):
                        entity.circuit_enabled = bool(prop_data["value"])
                    else:
                        if not hasattr(entity, "control_behavior"):
                            entity.control_behavior = {}
                        entity.control_behavior["circuit_enabled"] = bool(
                            prop_data["value"]
                        )
            else:
                try:
                    setattr(entity, prop_name, prop_data.get("value"))
                except Exception:
                    self.diagnostics.info(
                        f"Could not set property '{prop_name}' on '{placement.entity_type}'."
                    )


__all__ = ["PlanEntityEmitter"]
