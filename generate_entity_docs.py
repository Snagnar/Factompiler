#!/usr/bin/env python3
"""
Generate comprehensive entity documentation for the Facto.

This script creates educational documentation that helps users understand:
1. What properties can be set on each entity (both at placement and dynamically)
2. How to set enum values (with the actual integer values)
3. What circuit input/output signals each entity supports
4. DSL syntax examples for each entity type

All information is extracted dynamically from the Draftsman library.

Usage:
    python generate_entity_docs.py -o ENTITY_REFERENCE_DSL.md
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any

import attrs

# Import draftsman
try:
    import draftsman
    from draftsman import constants
    from draftsman import entity as entity_module
    from draftsman.classes.entity import Entity
except ImportError as e:
    print("Error: Could not import draftsman.")
    print(f"Details: {e}")
    sys.exit(1)


@dataclass
class EnumInfo:
    name: str
    members: dict[str, Any]
    description: str = ""


@dataclass
class PropertyInfo:
    name: str
    type_name: str  # Original Python type
    dsl_type: str  # DSL-friendly type description
    python_type: Any
    default_value: Any
    description: str
    is_enum: bool = False
    enum_info: EnumInfo | None = None
    example_value: str = ""
    is_dsl_supported: bool = True  # Whether this type is directly settable in DSL
    is_signal_property: bool = False  # Whether this is a signal I/O property


@dataclass
class SignalIOEntry:
    """Represents a single signal input or output."""

    property_name: str
    direction: str  # "input" or "output"
    signal_type: str
    description: str
    enable_property: str | None = None


@dataclass
class CircuitIOInfo:
    has_circuit_connection: bool = True
    has_dual_connection: bool = False
    signal_inputs: list[SignalIOEntry] = field(default_factory=list)
    signal_outputs: list[SignalIOEntry] = field(default_factory=list)
    content_outputs: list[str] = field(default_factory=list)  # Generic outputs like "item contents"


@dataclass
class EntityInfo:
    class_name: str
    cls: type
    description: str
    prototypes: list[str]
    mixins: list[str]
    properties: list[PropertyInfo]
    circuit_io: CircuitIOInfo
    dsl_examples: list[str] = field(default_factory=list)


def get_all_enums() -> dict[str, EnumInfo]:
    enums = {}
    for name in dir(constants):
        if name.startswith("_"):
            continue
        obj = getattr(constants, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, (IntEnum, Enum))
            and hasattr(obj, "__members__")
        ):
            members = {}
            for member_name, member in obj.__members__.items():
                try:
                    val = member.value
                    if hasattr(val, "__int__"):
                        members[member_name] = int(val)
                    else:
                        members[member_name] = str(val)  # type: ignore[assignment]
                except (ValueError, TypeError):
                    members[member_name] = str(member.value)  # type: ignore[assignment]
            enums[name] = EnumInfo(name=name, members=members)
    return enums


ALL_ENUMS = get_all_enums()


def find_enum_for_type(type_hint: Any) -> EnumInfo | None:
    type_str = str(type_hint) if type_hint else ""
    for enum_name, enum_info in ALL_ENUMS.items():
        if enum_name in type_str:
            return enum_info
    return None


SKIP_PROPS = {
    "_parent",
    "_connections",
    "_entity_number",
    "extra_keys",
    "position",
    "tile_position",
    "id",
    "tags",
    "mirror",
}

# Detailed signal I/O definitions
# Format: property_name -> (direction, signal_type, description, enable_property)
# direction: "input" or "output"
# signal_type: what kind of signal value is read/written
# description: what this signal does
# enable_property: the boolean property that enables this (or more complex string)
SIGNAL_IO_DEFINITIONS = {
    # === COMMON INPUT SIGNALS ===
    "circuit_condition": (
        "input",
        "Any signal",
        "Signal used in enable/disable condition",
        "circuit_enabled: 1",
    ),
    "logistic_condition": (
        "input",
        "Any signal",
        "Logistic network signal for enable/disable",
        "connect_to_logistic_network: 1",
    ),
    "stack_size_control_signal": (
        "input",
        "Integer signal",
        "Sets inserter stack size from signal value",
        "circuit_set_stack_size: 1",
    ),
    # === LAMP SIGNALS ===
    # For COMPONENTS mode (color_mode: 1), use red_signal/green_signal/blue_signal
    # For PACKED_RGB mode (color_mode: 2), use rgb_signal
    "red_signal": (
        "input",
        "Integer (0-255)",
        "Red color component (COMPONENTS mode)",
        "use_colors: 1, color_mode: 1",
    ),
    "green_signal": (
        "input",
        "Integer (0-255)",
        "Green color component (COMPONENTS mode)",
        "use_colors: 1, color_mode: 1",
    ),
    "blue_signal": (
        "input",
        "Integer (0-255)",
        "Blue color component (COMPONENTS mode)",
        "use_colors: 1, color_mode: 1",
    ),
    "rgb_signal": (
        "input",
        "Packed RGB integer",
        "Combined RGB value (PACKED_RGB mode)",
        "use_colors: 1, color_mode: 2",
    ),
    # === RAIL SIGNAL OUTPUTS ===
    "red_output_signal": (
        "output",
        "1 when red",
        "Outputs 1 when rail signal shows red",
        "read_signal: 1",
    ),
    "yellow_output_signal": (
        "output",
        "1 when yellow",
        "Outputs 1 when rail signal shows yellow",
        "read_signal: 1",
    ),
    "green_output_signal": (
        "output",
        "1 when green",
        "Outputs 1 when rail signal shows green",
        "read_signal: 1",
    ),
    "blue_output_signal": ("output", "1 when blue", "Outputs 1 when chain signal reserved", None),
    # === TRAIN STOP SIGNALS ===
    "train_stopped_signal": (
        "output",
        "Train ID",
        "Outputs ID of stopped train",
        "read_stopped_train: 1",
    ),
    "trains_limit_signal": (
        "input",
        "Integer",
        "Sets train limit from signal value",
        "signal_limits_trains: 1",
    ),
    "trains_count_signal": (
        "output",
        "Integer",
        "Outputs count of trains en route",
        "read_trains_count: 1",
    ),
    "priority_signal": (
        "input",
        "Integer",
        "Sets station priority from signal value",
        "set_priority: 1",
    ),
    # === CRAFTING MACHINE SIGNALS ===
    "recipe_finished_signal": (
        "output",
        "Pulse (1)",
        "Pulses when recipe completes",
        "read_recipe_finished: 1",
    ),
    "working_signal": (
        "output",
        "1 when working",
        "Outputs 1 while machine is crafting",
        "read_working: 1",
    ),
    # === COMBINATOR SIGNALS ===
    "output_signal": ("output", "Result value", "Signal to output the combinator result on", None),
    "index_signal": ("input", "Integer", "Signal for selector index input", None),
    "count_signal": ("output", "Integer", "Signal to output the count result", None),
    "quality_source_signal": ("input", "Any signal", "Signal to read quality from", None),
    "quality_destination_signal": (
        "output",
        "Quality level",
        "Signal to output quality result",
        None,
    ),
    # === ROBOPORT SIGNALS ===
    "available_logistic_robots_signal": (
        "output",
        "Integer",
        "Count of idle logistic robots",
        "read_logistics: 1",
    ),
    "total_logistic_robots_signal": (
        "output",
        "Integer",
        "Total logistic robots in network",
        "read_logistics: 1",
    ),
    "available_construction_robots_signal": (
        "output",
        "Integer",
        "Count of idle construction robots",
        "read_logistics: 1",
    ),
    "total_construction_robots_signal": (
        "output",
        "Integer",
        "Total construction robots",
        "read_logistics: 1",
    ),
    # === ACCUMULATOR SIGNALS ===
    # Note: accumulator uses "output_signal" but that's also used by combinators
    # === SPEAKER SIGNALS ===
    "signal_value_is_pitch": ("input", "Pitch value", "Signal value controls note pitch", None),
    # === MINING DRILL SIGNALS ===
    # read_resources outputs resource signals based on what's under the drill
    # === ASTEROID COLLECTOR SIGNALS ===
    "status_signal": ("output", "Status code", "Current collector status", "read_status"),
    "storage_signal": ("output", "Item contents", "Items stored in collector", "read_contents"),
}

# Properties that enable signal reading/writing (bool properties)
SIGNAL_ENABLE_PROPERTIES = {
    "read_contents": "Enables outputting contents to circuit network",
    "read_hand_contents": "Enables outputting items in inserter hand",
    "read_resources": "Enables outputting resource amounts under entity",
    "read_signal": "Enables outputting rail signal state",
    "read_from_train": "Enables reading train cargo contents",
    "send_to_train": "Enables sending signals to train for schedule control",
    "read_stopped_train": "Enables outputting stopped train ID",
    "read_trains_count": "Enables outputting count of trains en route",
    "read_recipe_finished": "Enables recipe finished pulse signal",
    "read_working": "Enables outputting working status",
    "read_logistics": "Enables outputting robot counts",
    "read_robot_stats": "Enables outputting robot statistics",
    "circuit_enabled": "Enables circuit condition control",
    "connect_to_logistic_network": "Enables logistic network condition control",
    "circuit_set_filters": "Enables setting filters from circuit signals",
    "circuit_set_stack_size": "Enables setting stack size from signal",
    "circuit_set_recipe": "Enables setting recipe from circuit signals",
    "signal_limits_trains": "Enables setting train limit from signal",
    "set_priority": "Enables setting station priority from signal",
    "use_colors": "Enables color control from circuit signals",
    "read_status": "Enables outputting entity status",
    "read_ammo": "Enables outputting ammo count",
}

# Set of property names that are signal I/O and should be excluded from settable properties
SIGNAL_PROPERTY_NAMES = set(SIGNAL_IO_DEFINITIONS.keys())


# DSL-friendly type mappings
# Map Python/Draftsman types to user-friendly DSL types
def get_dsl_type(type_str: str, prop_name: str) -> tuple[str, bool]:
    """
    Convert a Python type string to a DSL-friendly type description.
    Returns (dsl_type, is_supported) tuple.
    """
    type_lower = type_str.lower()

    # Boolean types
    if "bool" in type_lower:
        return "Boolean (0/1)", True

    # String types
    if prop_name in ("station", "player_description", "text"):
        return "String", True
    if prop_name == "name":
        return "String (entity prototype name)", True
    if prop_name == "recipe":
        return "String (recipe name)", True

    # Integer types
    if "int" in type_lower and "annotated" in type_lower:
        return "Integer", True
    if prop_name in ("priority", "index_constant", "random_update_interval"):
        return "Integer", True

    # Direction enum
    if "direction" in type_lower:
        return "Integer (0-15, see Direction enum)", True

    # Color type
    if "color" in type_lower and "mode" not in prop_name:
        return "Color {r: 0-255, g: 0-255, b: 0-255}", True

    # Signal ID types
    if "signalid" in type_lower:
        return "String (signal name)", True

    # Quality
    if "quality" in type_lower and "literal" in type_lower:
        return "String (normal/uncommon/rare/epic/legendary)", True

    # Literal string types (extract values)
    if "literal[" in type_lower:
        # Try to extract the literal values
        import re

        match = re.search(r"literal\[([^\]]+)\]", type_str, re.IGNORECASE)
        if match:
            values = match.group(1).replace("'", "").replace('"', "")
            if len(values) < 60:
                return f"One of: {values}", True
        return "String (see type for valid values)", True

    # Enum types
    if "<enum" in type_lower:
        return "Integer (see enum reference)", True

    # Lists and complex types - generally not directly settable in DSL
    if "list[" in type_lower:
        return "List (complex)", False
    if "condition" in type_lower:
        return "Condition (set via .enable)", False
    if "vector" in type_lower:
        return "Vector {x, y}", False
    if "factory" in type_lower or "annotated[" in type_lower:
        # Try to determine the base type
        if "int" in type_lower:
            return "Integer", True
        if "str" in type_lower:
            return "String", True

    # Orientation for trains
    if "orientation" in type_lower:
        return "Float (0.0-1.0)", True

    # Default - unknown/complex
    return "Complex (see draftsman docs)", False


def get_docstrings_from_source(cls: type) -> dict[str, str]:
    docstrings = {}
    for klass in cls.__mro__:
        if klass in (object,) or not hasattr(klass, "__module__"):
            continue
        try:
            source = inspect.getsource(klass)
            patterns = [
                r'^\s*(\w+)\s*:\s*[^=]+?=\s*attrs\.field\([\s\S]*?\)\s*\n\s*"""([\s\S]*?)"""',
                r'^\s*(\w+)\s*=\s*attrs_reuse\([\s\S]*?\)\s*\n\s*"""([\s\S]*?)"""',
                r'^\s*(\w+)\s*:\s*[^=]+?=\s*[^\n]+\n\s*"""([\s\S]*?)"""',
            ]
            for pattern in patterns:
                for match in re.finditer(pattern, source, re.MULTILINE):
                    attr_name = match.group(1)
                    docstring = match.group(2).strip()
                    if attr_name not in docstrings:
                        docstring = re.sub(
                            r"\.\.\s*\w+::\s*\n.*?(?=\n\n|\Z)", "", docstring, flags=re.DOTALL
                        )
                        docstring = re.sub(r":py:\w+:`[~.]?([^`]+)`", r"\1", docstring)
                        docstring = re.sub(r"\s+", " ", docstring).strip()
                        docstrings[attr_name] = (
                            docstring[:200] if len(docstring) > 200 else docstring
                        )
        except (OSError, TypeError):
            pass
    return docstrings


def get_example_value(prop_name: str, prop_type: Any, enum_info: EnumInfo | None) -> str:
    if enum_info:
        first_member = next(iter(enum_info.members.items()))
        return f"{first_member[1]}  # {first_member[0]}"
    type_str = str(prop_type).lower() if prop_type else ""
    if "bool" in type_str:
        return "1"
    elif "signal" in prop_name.lower():
        return '"signal-A"'
    elif "color" in prop_name.lower() and "mode" not in prop_name.lower():
        return "{r: 255, g: 0, b: 0}"
    elif prop_name == "direction":
        return "0  # NORTH"
    elif prop_name == "station":
        return '"Station Name"'
    elif prop_name == "recipe":
        return '"iron-gear-wheel"'
    return ""


def get_entity_properties(cls: type) -> list[PropertyInfo]:
    properties: list[PropertyInfo] = []
    if not attrs.has(cls):
        return properties
    docstrings = get_docstrings_from_source(cls)
    for fld in attrs.fields(cls):
        name = fld.name
        if name in SKIP_PROPS or name.startswith("_"):
            continue
        type_str = str(fld.type) if fld.type else "Any"
        type_str = re.sub(r"typing\.", "", type_str)
        type_str = re.sub(r"draftsman\.\w+\.", "", type_str)
        type_str = re.sub(r"<class '([^']+)'>", r"\1", type_str)
        if len(type_str) > 80:
            type_str = type_str[:77] + "..."

        # Get DSL-friendly type
        dsl_type, is_supported = get_dsl_type(type_str, name)

        # Check if this is a signal property
        is_signal_prop = name in SIGNAL_PROPERTY_NAMES

        enum_info = find_enum_for_type(fld.type)
        is_enum = enum_info is not None

        # Override DSL type for enums
        if is_enum and enum_info:
            dsl_type = f"Integer ([{enum_info.name}](#{enum_info.name.lower()}))"
            is_supported = True

        if fld.default is attrs.NOTHING:
            default = "required"
        elif isinstance(fld.default, attrs.Factory):  # type: ignore[arg-type]
            default = "(factory)"
        elif fld.default is None:
            default = "None"
        elif isinstance(fld.default, bool):
            default = "true" if fld.default else "false"
        elif isinstance(fld.default, str):
            default = f'"{fld.default}"'
        else:
            default = str(fld.default)
        prop = PropertyInfo(
            name=name,
            type_name=type_str,
            dsl_type=dsl_type,
            python_type=fld.type,
            default_value=default,
            description=docstrings.get(name, ""),
            is_enum=is_enum,
            enum_info=enum_info,
            example_value=get_example_value(name, fld.type, enum_info),
            is_dsl_supported=is_supported,
            is_signal_property=is_signal_prop,
        )
        properties.append(prop)
    # Sort alphabetically for consistency
    properties.sort(key=lambda p: p.name)
    return properties


def get_circuit_io_info(cls: type) -> CircuitIOInfo:
    """Extract circuit signal I/O information from an entity class."""
    info = CircuitIOInfo()
    mixin_names = [c.__name__ for c in cls.__mro__ if "Mixin" in c.__name__]

    # Check for dual connection (combinators)
    if cls.__name__ in ("ArithmeticCombinator", "DeciderCombinator", "SelectorCombinator"):
        info.has_dual_connection = True

    # Get all property names from the entity
    prop_names = set()
    if attrs.has(cls):
        for fld in attrs.fields(cls):
            prop_names.add(fld.name)

    # Check for specific signal properties defined in SIGNAL_IO_DEFINITIONS
    for prop_name, (
        direction,
        signal_type,
        description,
        enable_prop,
    ) in SIGNAL_IO_DEFINITIONS.items():
        if prop_name in prop_names:
            entry = SignalIOEntry(
                property_name=prop_name,
                direction=direction,
                signal_type=signal_type,
                description=description,
                enable_property=enable_prop,
            )
            if direction == "input":
                info.signal_inputs.append(entry)
            else:
                info.signal_outputs.append(entry)

    # Check for generic content output capabilities based on mixins
    if "CircuitReadContentsMixin" in mixin_names and "read_contents" in prop_names:
        info.content_outputs.append(
            "Item contents (all items in entity, enable with `read_contents: 1`)"
        )
    if "CircuitReadHandMixin" in mixin_names and "read_hand_contents" in prop_names:
        info.content_outputs.append(
            "Inserter hand contents (items being moved, enable with `read_hand_contents: 1`)"
        )
    if "CircuitReadResourceMixin" in mixin_names and "read_resources" in prop_names:
        info.content_outputs.append(
            "Resource amounts (resources under entity, enable with `read_resources: 1`)"
        )

    return info


def get_all_entity_classes() -> dict[str, type]:
    classes = {}
    for name in dir(entity_module):
        obj = getattr(entity_module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, Entity)
            and obj is not Entity
            and not name.startswith("_")
        ):
            classes[name] = obj
    return classes


def get_entity_prototypes(cls: type) -> list[str]:
    """Get ALL prototypes for this entity class."""
    try:
        instance = cls()
        if hasattr(instance, "similar_entities"):
            return list(instance.similar_entities)  # No limit!
        return [instance.name] if hasattr(instance, "name") and instance.name else []
    except Exception:
        return []


def get_mixins(cls: type) -> list[str]:
    return [c.__name__ for c in cls.__mro__ if "Mixin" in c.__name__]


def get_entity_description(cls: type) -> str:
    doc = cls.__doc__ or ""
    lines = doc.strip().split("\n")
    result: list[str] = []
    for line in lines:
        line = line.strip()
        if not line and result:
            break
        if line:
            result.append(line)
    desc = " ".join(result) if result else "No description available."
    # Clean up sphinx directives
    desc = re.sub(r"\.\.\s+\w+::\s*\S*", "", desc)
    return desc.strip()


def get_dsl_examples(cls: type, class_name: str) -> list[str]:
    examples = {
        "Lamp": [
            "# Basic lamp controlled by circuit",
            'Entity lamp = place("small-lamp", 0, 0);',
            "lamp.enable = signal > 0;",
            "",
            "# RGB colored lamp",
            'Entity rgb_lamp = place("small-lamp", 2, 0, {use_colors: 1, color_mode: 1});',
            "rgb_lamp.r = red_signal;",
            "rgb_lamp.g = green_signal;",
            "rgb_lamp.b = blue_signal;",
        ],
        "Inserter": [
            "# Inserter that enables when chest has items",
            'Entity inserter = place("inserter", 0, 0, {direction: 4});',
            "inserter.enable = chest_contents > 50;",
        ],
        "TransportBelt": [
            "# Belt that stops when storage is full",
            'Entity belt = place("transport-belt", 0, 0, {direction: 4});',
            "belt.enable = storage_count < 1000;",
        ],
        "TrainStop": [
            "# Train station with circuit control",
            'Entity station = place("train-stop", 0, 0, {station: "Iron Pickup"});',
            "station.enable = has_cargo > 0;",
        ],
        "AssemblingMachine": [
            "# Assembler controlled by circuit",
            'Entity assembler = place("assembling-machine-1", 0, 0, {recipe: "iron-gear-wheel"});',
            "assembler.enable = iron_count > 100;",
        ],
        "ArithmeticCombinator": [
            "# Note: Combinators are typically generated by the compiler",
            "Signal result = input * 2 + offset;  # Creates ArithmeticCombinator(s)",
        ],
        "DeciderCombinator": [
            "# Note: Combinators are typically generated by the compiler",
            "Signal flag = (count > 100) : 1;  # Creates DeciderCombinator",
        ],
        "ConstantCombinator": [
            "# Note: Constant combinators are typically generated by the compiler",
            "Signal constant = 42;  # Creates ConstantCombinator",
        ],
    }
    return examples.get(class_name, [])


def gather_entity_info(cls: type, class_name: str) -> EntityInfo:
    return EntityInfo(
        class_name=class_name,
        cls=cls,
        description=get_entity_description(cls),
        prototypes=get_entity_prototypes(cls),
        mixins=get_mixins(cls),
        properties=get_entity_properties(cls),
        circuit_io=get_circuit_io_info(cls),
        dsl_examples=get_dsl_examples(cls, class_name),
    )


def generate_enum_reference() -> list[str]:
    lines = [
        "## Enum Reference",
        "",
        "When setting enum properties in the DSL, use the **integer value**.",
        "This section lists all enums used by entity properties.",
        "",
    ]

    # List ALL enums that are actually used by entities
    relevant_enums = [
        "LampColorMode",
        "Direction",
        "InserterModeOfOperation",
        "InserterReadMode",
        "BeltReadMode",
        "FilterMode",
        "LogisticModeOfOperation",
        "MiningDrillReadMode",
        "SiloReadMode",
        "SpaceConnectionReadMode",
        "AsteroidCollectorStatus",
    ]

    for enum_name in relevant_enums:
        if enum_name not in ALL_ENUMS:
            continue
        enum_info = ALL_ENUMS[enum_name]
        lines.extend(
            [
                f'### <a id="{enum_name.lower()}"></a>{enum_name}',
                "",
                "| DSL Value | Enum Name |",
                "|-----------|-----------|",
            ]
        )
        for member_name, value in enum_info.members.items():
            lines.append(f"| `{value}` | {member_name} |")
        lines.append("")
    return lines


def format_type_with_enum_link(prop: PropertyInfo) -> str:
    """Format type string with enum link if applicable."""
    if prop.is_enum and prop.enum_info:
        return f"{prop.type_name} ([{prop.enum_info.name}](#{prop.enum_info.name.lower()}))"
    return prop.type_name


# Track unsupported properties across all entities
UNSUPPORTED_PROPERTIES: set[tuple[str, str]] = set()  # (property_name, dsl_type)


def generate_entity_section(entity: EntityInfo) -> list[str]:
    lines = [f"### {entity.class_name}", ""]

    # Description
    lines.append(f"**Description:** {entity.description}")
    lines.append("")

    # ALL prototypes
    if entity.prototypes:
        proto_str = ", ".join(f'`"{p}"`' for p in entity.prototypes)
        lines.extend([f"**Prototypes:** {proto_str}", ""])

    # Circuit connection type
    io = entity.circuit_io
    if io.has_dual_connection:
        lines.append(
            "**Connection Type:** Dual circuit connection (separate input and output sides)"
        )
    else:
        lines.append("**Connection Type:** Single circuit connection")
    lines.append("")

    # === Circuit Signal I/O Section ===
    has_signal_io = io.signal_inputs or io.signal_outputs or io.content_outputs
    if has_signal_io:
        lines.append("#### Circuit Signal I/O")
        lines.append("")

        # Signal Inputs Table
        if io.signal_inputs:
            lines.append("**Signal Inputs:**")
            lines.append("")
            lines.append("| Signal Property | Signal Type | Description | Enable With |")
            lines.append("|-----------------|-------------|-------------|-------------|")
            for sig in io.signal_inputs:
                enable = f"`{sig.enable_property}`" if sig.enable_property else "Always active"
                lines.append(
                    f"| `{sig.property_name}` | {sig.signal_type} | {sig.description} | {enable} |"
                )
            lines.append("")

        # Signal Outputs Table
        if io.signal_outputs:
            lines.append("**Signal Outputs:**")
            lines.append("")
            lines.append("| Signal Property | Signal Type | Description | Enable With |")
            lines.append("|-----------------|-------------|-------------|-------------|")
            for sig in io.signal_outputs:
                enable = f"`{sig.enable_property}`" if sig.enable_property else "Always active"
                lines.append(
                    f"| `{sig.property_name}` | {sig.signal_type} | {sig.description} | {enable} |"
                )
            lines.append("")

        # Content outputs (generic item/resource outputs)
        if io.content_outputs:
            lines.append("**Content Outputs:**")
            lines.append("")
            for content in io.content_outputs:
                lines.append(f"- {content}")
            lines.append("")

    # DSL Examples
    if entity.dsl_examples:
        lines.extend(["#### DSL Examples", "", "```facto"])
        lines.extend(entity.dsl_examples)
        lines.extend(["```", ""])

    # === Settable Properties Section ===
    # Filter out signal properties (they're in the Circuit Signal I/O section)
    settable_props = [p for p in entity.properties if not p.is_signal_property]

    if settable_props:
        lines.append("#### Settable Properties")
        lines.append("")
        lines.append(
            'Set at placement: `place("name", x, y, {prop: value})` or after: `entity.prop = value`'
        )
        lines.append("")
        lines.append("| Property | Type | Default | Example |")
        lines.append("|----------|------|---------|---------|")
        for prop in settable_props:
            # Use DSL type, not Python type
            type_str = prop.dsl_type
            example = f"`{prop.example_value}`" if prop.example_value else ""

            # Track unsupported properties
            if not prop.is_dsl_supported:
                UNSUPPORTED_PROPERTIES.add((prop.name, prop.dsl_type))
                type_str = f"{prop.dsl_type} ⚠️"  # Mark as potentially unsupported

            lines.append(f"| `{prop.name}` | {type_str} | {prop.default_value} | {example} |")
        lines.append("")

    lines.extend(["---", ""])
    return lines


# Entity categorization - will add uncategorized entities to "Other" automatically
ENTITY_CATEGORIES = {
    "Combinators": [
        "ArithmeticCombinator",
        "DeciderCombinator",
        "ConstantCombinator",
        "SelectorCombinator",
    ],
    "Lamps & Displays": ["Lamp", "DisplayPanel"],
    "Inserters": ["Inserter"],
    "Belts & Logistics": [
        "TransportBelt",
        "UndergroundBelt",
        "Splitter",
        "Loader",
        "LinkedBelt",
        "LaneLaneSplitter",
    ],
    "Train System": [
        "TrainStop",
        "RailSignal",
        "RailChainSignal",
        "Locomotive",
        "CargoWagon",
        "FluidWagon",
        "ArtilleryWagon",
        "StraightRail",
        "CurvedRailA",
        "CurvedRailB",
        "HalfDiagonalRail",
        "RailRamp",
        "RailSupport",
        "ElevatedStraightRail",
        "ElevatedCurvedRailA",
        "ElevatedCurvedRailB",
        "ElevatedHalfDiagonalRail",
        "LegacyStraightRail",
        "LegacyCurvedRail",
    ],
    "Production": [
        "AssemblingMachine",
        "Furnace",
        "MiningDrill",
        "Lab",
        "RocketSilo",
        "Beacon",
        "Boiler",
        "Generator",
        "BurnerGenerator",
        "Reactor",
        "FusionReactor",
        "FusionGenerator",
        "LightningAttractor",
        "LightningRod",
        "AgriculturalTower",
    ],
    "Storage": [
        "Container",
        "LogisticPassiveContainer",
        "LogisticActiveContainer",
        "LogisticStorageContainer",
        "LogisticRequestContainer",
        "LogisticBufferContainer",
        "LinkedContainer",
    ],
    "Power": [
        "ElectricPole",
        "PowerSwitch",
        "Accumulator",
        "SolarPanel",
        "ElectricEnergyInterface",
    ],
    "Fluids": ["Pump", "StorageTank", "OffshorePump", "Pipe", "PipeToGround", "InfinityPipe"],
    "Combat": [
        "Radar",
        "ArtilleryTurret",
        "AmmoTurret",
        "ElectricTurret",
        "FluidTurret",
        "Wall",
        "Gate",
        "Landmine",
    ],
    "Robots & Logistics": ["Roboport", "ConstructionRobot", "LogisticRobot"],
    "Space": ["SpacePlatformHub", "CargoLandingPad", "AsteroidCollector", "CargoBay", "Thruster"],
    "Misc": [
        "ProgrammableSpeaker",
        "Car",
        "SpiderVehicle",
        "HeatPipe",
        "HeatInterface",
        "SimpleEntityWithOwner",
        "SimpleEntityWithForce",
    ],
}


def generate_documentation() -> str:
    lines = [
        "# Entity Reference for Facto",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}",
        f"**Draftsman version:** {draftsman.__version__}",
        "",
        "This is the **complete reference** for all entities available in the DSL.",
        "Each entity lists its prototypes, circuit I/O capabilities, and all settable properties.",
        "",
        "## Table of Contents",
        "",
        "- [Using Entities in the DSL](#using-entities-in-the-dsl)",
        "- [Enum Reference](#enum-reference)",
    ]

    # Add category links
    for category in ENTITY_CATEGORIES:
        anchor = category.lower().replace(" ", "-").replace("&", "").replace("  ", "-")
        lines.append(f"- [{category}](#{anchor})")
    lines.append("- [Uncategorized Entities](#uncategorized-entities)")
    lines.append("")

    # Usage section
    lines.extend(
        [
            "## Using Entities in the DSL",
            "",
            "### Placement Syntax",
            "",
            "```facto",
            'Entity name = place("prototype-name", x, y, {prop1: value1, prop2: value2});',
            "```",
            "",
            "### Setting Properties",
            "",
            "**At placement time** (in the property dictionary):",
            "```facto",
            'Entity lamp = place("small-lamp", 0, 0, {use_colors: 1, color_mode: 1});',
            "```",
            "",
            "**After placement** (for circuit-controlled values):",
            "```facto",
            "lamp.enable = signal > 0;  # Control based on circuit signal",
            "lamp.r = red_value;        # Dynamic RGB control",
            "```",
            "",
            "### Enum Properties",
            "",
            "Enum properties accept **integer values**. See the [Enum Reference](#enum-reference) for all values.",
            "",
            "```facto",
            'Entity lamp = place("small-lamp", 0, 0, {color_mode: 1});  # 1 = COMPONENTS',
            "```",
            "",
            "### Boolean Properties",
            "",
            "Boolean properties accept `1` (true) or `0` (false):",
            "```facto",
            'Entity lamp = place("small-lamp", 0, 0, {use_colors: 1, always_on: 1});',
            "```",
            "",
        ]
    )

    # Enum reference
    lines.extend(generate_enum_reference())

    # Get all entity classes
    entity_classes = get_all_entity_classes()

    # Track which entities are categorized
    categorized = set()

    # Generate sections by category
    for category, class_names in ENTITY_CATEGORIES.items():
        # Only include category if it has any entities
        category_entities = [name for name in class_names if name in entity_classes]
        if not category_entities:
            continue

        lines.extend([f"## {category}", ""])

        for class_name in class_names:
            if class_name not in entity_classes:
                continue
            categorized.add(class_name)
            cls = entity_classes[class_name]
            entity = gather_entity_info(cls, class_name)
            lines.extend(generate_entity_section(entity))

    # Generate sections for ALL uncategorized entities (no truncation!)
    uncategorized = [name for name in sorted(entity_classes.keys()) if name not in categorized]
    if uncategorized:
        lines.extend(
            [
                "## Uncategorized Entities",
                "",
                "The following entities are available but not yet categorized.",
                "They still have full documentation below.",
                "",
            ]
        )

        for class_name in uncategorized:
            cls = entity_classes[class_name]
            entity = gather_entity_info(cls, class_name)
            lines.extend(generate_entity_section(entity))

    # Add notes section about unsupported/complex types
    if UNSUPPORTED_PROPERTIES:
        lines.extend(
            [
                "## Notes on Complex Property Types",
                "",
                "Some properties marked with ⚠️ have complex types that may not be directly settable",
                "in the current DSL syntax. These typically include:",
                "",
                "| Type | Description | Workaround |",
                "|------|-------------|------------|",
                "| List | Arrays of items/filters | May require special syntax |",
                "| Condition | Circuit conditions | Use `.enable = signal > value` syntax |",
                "| Vector | Position offsets | Use `{x: value, y: value}` |",
                "| Complex | Other structured data | See draftsman documentation |",
                "",
                "For full details on complex types, refer to the ",
                "[Draftsman documentation](https://factorio-draftsman.readthedocs.io/en/latest/).",
                "",
            ]
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive entity documentation for the Facto"
    )
    parser.add_argument("--output", "-o", help="Output file path (default: stdout)")
    args = parser.parse_args()

    print("Gathering entity information...", file=sys.stderr)
    content = generate_documentation()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Documentation written to: {args.output}", file=sys.stderr)
    else:
        print(content)


if __name__ == "__main__":
    main()
