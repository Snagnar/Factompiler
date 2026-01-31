#!/usr/bin/env python3
"""
Generate comprehensive entity documentation for the Facto.

This script creates educational documentation that helps users understand:
1. What properties can be set on each entity (both at placement and dynamically)
2. How to set enum/literal values (with all valid options)
3. What circuit outputs each entity can produce via `.output`
4. DSL syntax examples for each entity type

All information is extracted dynamically from the Draftsman library.

Usage:
    # Generate fresh documentation
    python doc/generate_entity_docs.py -o doc/ENTITY_REFERENCE.md

    # Update existing documentation (targeted changes only)
    python doc/generate_entity_docs.py --update doc/ENTITY_REFERENCE.md
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Literal, get_args, get_origin

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


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EnumInfo:
    """Information about an enum or Literal type."""

    name: str
    members: dict[str, Any]  # member_name -> value
    description: str = ""
    is_literal: bool = False  # True for Literal[] types, False for IntEnum


@dataclass
class PropertyInfo:
    """Information about an entity property."""

    name: str
    type_name: str  # Original Python type
    dsl_type: str  # DSL-friendly type description
    python_type: Any
    default_value: Any
    description: str
    is_enum: bool = False
    enum_info: EnumInfo | None = None
    literal_values: tuple | None = None  # For Literal types
    example_value: str = ""
    is_dsl_supported: bool = True
    is_signal_property: bool = False  # e.g. count_signal, output_signal


@dataclass
class EntityOutputInfo:
    """Information about entity circuit outputs via .output property."""

    supports_output: bool = False
    output_type: str = "Bundle"  # What .output returns
    output_signals: list[str] = field(default_factory=list)  # What signals it can output
    enable_properties: dict[str, str] = field(default_factory=dict)  # prop -> description
    description: str = ""


@dataclass
class CircuitIOInfo:
    """Information about entity circuit connections."""

    has_circuit_connection: bool = True
    has_dual_connection: bool = False
    output_info: EntityOutputInfo = field(default_factory=EntityOutputInfo)


@dataclass
class EntityInfo:
    """Complete information about an entity class."""

    class_name: str
    cls: type
    description: str
    prototypes: list[str]
    mixins: list[str]
    properties: list[PropertyInfo]
    circuit_io: CircuitIOInfo
    dsl_examples: list[str] = field(default_factory=list)


# =============================================================================
# Enum and Literal Type Detection
# =============================================================================


def get_all_int_enums() -> dict[str, EnumInfo]:
    """Get all IntEnum types from draftsman.constants."""
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
                        members[member_name] = str(val)
                except (ValueError, TypeError):
                    members[member_name] = str(member.value)
            enums[name] = EnumInfo(name=name, members=members, is_literal=False)
    return enums


def get_literal_values_from_type(type_hint: Any) -> tuple | None:
    """Extract values from a Literal type annotation."""
    origin = get_origin(type_hint)
    if origin is Literal:
        return get_args(type_hint)
    # Handle Optional[Literal[...]] and similar
    if origin is type(None) or str(origin) == "typing.Union":
        for arg in get_args(type_hint):
            if get_origin(arg) is Literal:
                return get_args(arg)
    return None


def get_all_literal_types() -> dict[str, EnumInfo]:
    """Get all unique Literal types used in entity properties."""
    literal_types = {}

    # Well-known Literal types to name them nicely
    KNOWN_LITERALS = {
        ("normal", "uncommon", "rare", "epic", "legendary", "quality-unknown"): "QualityID",
        ("*", "/", "+", "-", "%", "^", "<<", ">>", "AND", "OR", "XOR"): "ArithmeticOperation",
        (
            "select",
            "count",
            "random",
            "stack-size",
            "rocket-capacity",
            "quality-filter",
            "quality-transfer",
        ): "SelectorOperation",
        ("at-least", "at-most", "exactly", "add", "remove"): "InfinityMode",
        ("whitelist", "blacklist"): "FilterMode",
        ("spoiled-first", "fresh-first"): "SpoilPriority",
        ("input", "output"): "IOType",
        ("local", "surface", "global"): "PlaybackMode",
        ("left", "none", "right"): "SplitterPriority",
    }

    for name in dir(entity_module):
        obj = getattr(entity_module, name)
        if not isinstance(obj, type) or not issubclass(obj, Entity) or obj is Entity:
            continue
        if not attrs.has(obj):
            continue

        for fld in attrs.fields(obj):
            if fld.type is None:
                continue
            literal_vals = get_literal_values_from_type(fld.type)
            if literal_vals:
                # Filter out None values
                filtered_vals = tuple(v for v in literal_vals if v is not None)
                if not filtered_vals:
                    continue

                # Look for a known name
                known_name = KNOWN_LITERALS.get(filtered_vals)
                if known_name and known_name not in literal_types:
                    members = {str(v): v for v in filtered_vals}
                    literal_types[known_name] = EnumInfo(
                        name=known_name, members=members, is_literal=True
                    )

    return literal_types


ALL_INT_ENUMS = get_all_int_enums()
ALL_LITERAL_TYPES = get_all_literal_types()


def find_enum_for_type(type_hint: Any) -> EnumInfo | None:
    """Find an IntEnum that matches this type hint."""
    type_str = str(type_hint) if type_hint else ""
    for enum_name, enum_info in ALL_INT_ENUMS.items():
        if enum_name in type_str:
            return enum_info
    return None


def find_literal_for_type(type_hint: Any) -> tuple[tuple | None, EnumInfo | None]:
    """Find Literal values and corresponding EnumInfo for a type."""
    literal_vals = get_literal_values_from_type(type_hint)
    if not literal_vals:
        return None, None

    filtered_vals = tuple(v for v in literal_vals if v is not None)
    if not filtered_vals:
        return None, None

    # Look for matching EnumInfo
    for enum_info in ALL_LITERAL_TYPES.values():
        if set(enum_info.members.keys()) == {str(v) for v in filtered_vals}:
            return filtered_vals, enum_info

    return filtered_vals, None


# =============================================================================
# Skip lists and signal definitions
# =============================================================================


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

# Properties that indicate signal I/O (usually for circuit configuration)
SIGNAL_IO_PROPERTIES = {
    # Combinator signals
    "output_signal",
    "index_signal",
    "count_signal",
    "quality_source_signal",
    "quality_destination_signal",
    # Rail signal outputs
    "red_output_signal",
    "yellow_output_signal",
    "green_output_signal",
    "blue_output_signal",
    # Train stop
    "train_stopped_signal",
    "trains_limit_signal",
    "trains_count_signal",
    "priority_signal",
    # Recipe/working signals
    "recipe_finished_signal",
    "working_signal",
    # Roboport
    "available_logistic_robots_signal",
    "total_logistic_robots_signal",
    "available_construction_robots_signal",
    "total_construction_robots_signal",
    # Lamp RGB
    "red_signal",
    "green_signal",
    "blue_signal",
    "rgb_signal",
    # Circuit condition
    "circuit_condition",
    "logistic_condition",
    # Stack control
    "stack_size_control_signal",
    # Others
    "status_signal",
    "storage_signal",
}

# Entity class name -> output description mapping
ENTITY_OUTPUT_DESCRIPTIONS = {
    "Container": {
        "supports_output": True,
        "description": "Item contents of the container",
        "enable_properties": {"read_contents": "Enable reading container contents"},
    },
    "StorageTank": {
        "supports_output": True,
        "description": "Fluid level in the tank",
        "enable_properties": {},  # Always outputs
    },
    "Inserter": {
        "supports_output": True,
        "description": "Items in hand or filter status",
        "enable_properties": {
            "read_hand_contents": "Read items in hand",
            "circuit_set_filters": "Control via filters",
        },
    },
    "MiningDrill": {
        "supports_output": True,
        "description": "Resources under the drill",
        "enable_properties": {"read_resources": "Read resource amounts"},
    },
    "TransportBelt": {
        "supports_output": True,
        "description": "Items on the belt",
        "enable_properties": {"read_contents": "Read belt contents"},
    },
    "ArithmeticCombinator": {
        "supports_output": True,
        "description": "Computed arithmetic result",
        "enable_properties": {},  # Always outputs
    },
    "DeciderCombinator": {
        "supports_output": True,
        "description": "Conditional output signals",
        "enable_properties": {},  # Always outputs
    },
    "SelectorCombinator": {
        "supports_output": True,
        "description": "Selected/filtered signals",
        "enable_properties": {},  # Always outputs
    },
    "ConstantCombinator": {
        "supports_output": True,
        "description": "Constant signal values",
        "enable_properties": {},  # Always outputs
    },
    "Accumulator": {
        "supports_output": True,
        "description": "Charge level percentage",
        "enable_properties": {},
    },
    "Roboport": {
        "supports_output": True,
        "description": "Robot counts and logistics info",
        "enable_properties": {
            "read_logistics": "Read logistic robot counts",
            "read_robot_stats": "Read robot statistics",
        },
    },
    "TrainStop": {
        "supports_output": True,
        "description": "Train ID, count, and cargo",
        "enable_properties": {
            "read_stopped_train": "Read stopped train ID",
            "read_trains_count": "Read incoming trains count",
            "read_from_train": "Read train cargo",
        },
    },
    "RailSignal": {
        "supports_output": True,
        "description": "Signal state (red/yellow/green)",
        "enable_properties": {"read_signal": "Read signal state"},
    },
    "RailChainSignal": {
        "supports_output": True,
        "description": "Signal state (red/yellow/green/blue)",
        "enable_properties": {"read_signal": "Read signal state"},
    },
    "Lamp": {
        "supports_output": False,
        "description": "",
        "enable_properties": {},
    },
    "AssemblingMachine": {
        "supports_output": True,
        "description": "Recipe finished pulse, working status",
        "enable_properties": {
            "read_recipe_finished": "Pulse when recipe completes",
            "read_working": "Output working status",
        },
    },
}


# =============================================================================
# Type conversion helpers
# =============================================================================


def get_dsl_type(
    type_str: str, prop_name: str, literal_vals: tuple | None, enum_info: EnumInfo | None
) -> tuple[str, bool]:
    """Convert Python type to DSL-friendly description. Returns (type_str, is_supported)."""
    type_lower = type_str.lower()

    # Literal types - show possible values
    if literal_vals:
        if enum_info:
            values_str = ", ".join(f'`"{v}"`' for v in literal_vals)
            return (
                f"One of: {values_str} ([{enum_info.name}](#{enum_info.name.lower()}))",
                True,
            )
        values_str = ", ".join(f'`"{v}"`' for v in literal_vals)
        return f"One of: {values_str}", True

    # Boolean types
    if "bool" in type_lower:
        return "Boolean (`0` or `1`)", True

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
    if prop_name in ("priority", "index_constant", "random_update_interval", "bar"):
        return "Integer", True

    # Direction enum
    if "direction" in type_lower:
        return "Integer (`0`-`15`, see [Direction](#direction))", True

    # Color type
    if "color" in type_lower and "mode" not in prop_name:
        return "Object `{r, g, b}` (0-255 each)", True

    # Signal ID types - important for entity configuration
    if "signalid" in type_lower:
        return 'String (signal name, e.g. `"signal-A"`)', True

    # Quality
    if "quality" in type_lower and "literal" in type_lower:
        return "String ([QualityID](#qualityid))", True

    # Enum types
    if "<enum" in type_lower:
        return "Integer (see enum reference)", True

    # Float/Double
    if "float" in type_lower or "double" in type_lower:
        return "Number", True

    # Lists and complex types - not directly settable
    if "list[" in type_lower:
        return "List (complex)", False
    if "condition" in type_lower:
        return "Condition (use `.enable = expr`)", False
    if "vector" in type_lower:
        return "Vector `{x, y}`", False

    return "Complex (see draftsman docs)", False


def get_docstrings_from_source(cls: type) -> dict[str, str]:
    """Extract property docstrings from class source code."""
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


def get_example_value(
    prop_name: str, prop_type: Any, enum_info: EnumInfo | None, literal_vals: tuple | None
) -> str:
    """Generate an example value for a property."""
    if literal_vals and len(literal_vals) > 0:
        return f'"{literal_vals[0]}"'
    if enum_info:
        first_member = next(iter(enum_info.members.items()))
        return f"{first_member[1]}  # {first_member[0]}"

    type_str = str(prop_type).lower() if prop_type else ""

    if "bool" in type_str:
        return "1"
    elif "signal" in prop_name.lower() and "signalid" in type_str:
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


# =============================================================================
# Entity information gathering
# =============================================================================


def get_entity_properties(cls: type) -> list[PropertyInfo]:
    """Extract all settable properties from an entity class."""
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

        # Check for enums and literals
        enum_info = find_enum_for_type(fld.type)
        literal_vals, literal_enum = find_literal_for_type(fld.type)

        # Use literal enum if available
        if literal_enum:
            enum_info = literal_enum

        # Get DSL type
        dsl_type, is_supported = get_dsl_type(type_str, name, literal_vals, enum_info)

        # Override for enums
        if enum_info and not enum_info.is_literal:
            dsl_type = f"Integer ([{enum_info.name}](#{enum_info.name.lower()}))"
            is_supported = True

        # Default value
        if fld.default is attrs.NOTHING:
            default = "required"
        elif isinstance(fld.default, attrs.Factory):
            default = "(factory)"
        elif fld.default is None:
            default = "None"
        elif isinstance(fld.default, bool):
            default = "1" if fld.default else "0"
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
            is_enum=enum_info is not None,
            enum_info=enum_info,
            literal_values=literal_vals,
            example_value=get_example_value(name, fld.type, enum_info, literal_vals),
            is_dsl_supported=is_supported,
            is_signal_property=name in SIGNAL_IO_PROPERTIES,
        )
        properties.append(prop)

    properties.sort(key=lambda p: p.name)
    return properties


def get_circuit_io_info(cls: type) -> CircuitIOInfo:
    """Extract circuit I/O information from an entity class."""
    info = CircuitIOInfo()

    # Check for dual connection (combinators)
    if cls.__name__ in ("ArithmeticCombinator", "DeciderCombinator", "SelectorCombinator"):
        info.has_dual_connection = True

    # Get output info from class name
    class_name = cls.__name__
    if class_name in ENTITY_OUTPUT_DESCRIPTIONS:
        output_desc = ENTITY_OUTPUT_DESCRIPTIONS[class_name]
        info.output_info = EntityOutputInfo(
            supports_output=output_desc["supports_output"],
            description=output_desc["description"],
            enable_properties=output_desc.get("enable_properties", {}),
        )
    else:
        # Check mixins for output capability
        mixin_names = [c.__name__ for c in cls.__mro__ if "Mixin" in c.__name__]
        if any("CircuitRead" in m or "CircuitConnect" in m for m in mixin_names):
            info.output_info = EntityOutputInfo(
                supports_output=True,
                description="Circuit network signals",
                enable_properties={},
            )

    return info


def get_all_entity_classes() -> dict[str, type]:
    """Get all entity classes from draftsman."""
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
    """Get all prototypes for an entity class."""
    try:
        instance = cls()
        if hasattr(instance, "similar_entities"):
            return list(instance.similar_entities)
        return [instance.name] if hasattr(instance, "name") and instance.name else []
    except Exception:
        return []


def get_mixins(cls: type) -> list[str]:
    """Get mixin class names."""
    return [c.__name__ for c in cls.__mro__ if "Mixin" in c.__name__]


def get_entity_description(cls: type) -> str:
    """Get entity class description."""
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
    desc = re.sub(r"\.\.\s+\w+::\s*\S*", "", desc)
    return desc.strip()


def get_dsl_examples(cls: type, class_name: str) -> list[str]:
    """Get DSL code examples for an entity."""
    examples = {
        "Lamp": [
            "# Basic lamp controlled by circuit",
            'Entity lamp = place("small-lamp", 0, 0);',
            "lamp.enable = signal > 0;",
            "",
            "# RGB colored lamp (color_mode: 1 = COMPONENTS)",
            'Entity rgb_lamp = place("small-lamp", 2, 0, {use_colors: 1, color_mode: 1});',
            "rgb_lamp.r = red_value;",
            "rgb_lamp.g = green_value;",
            "rgb_lamp.b = blue_value;",
        ],
        "SelectorCombinator": [
            "# Selector in count mode",
            'Entity counter = place("selector-combinator", 0, 0, {',
            '    operation: "count",',
            '    count_signal: "signal-C"',
            "});",
            "# Reading output",
            "Bundle result = counter.output;",
        ],
        "Inserter": [
            "# Inserter that enables when chest has items",
            'Entity inserter = place("inserter", 0, 0, {direction: 4});',
            "inserter.enable = chest.output > 50;",
        ],
        "Container": [
            "# Read chest contents",
            'Entity chest = place("iron-chest", 0, 0);',
            "Bundle contents = chest.output;",
            'Signal iron = contents["iron-plate"];',
        ],
        "ArithmeticCombinator": [
            "# Note: Arithmetic combinators are usually auto-generated",
            "Signal result = input * 2 + offset;  # Creates combinator(s)",
            "",
            "# Manual placement if needed",
            'Entity arith = place("arithmetic-combinator", 0, 0, {',
            '    operation: "+"',
            "});",
        ],
        "DeciderCombinator": [
            "# Note: Decider combinators are usually auto-generated",
            "Signal flag = (count > 100) : 1;  # Creates decider",
        ],
        "ConstantCombinator": [
            "# Note: Constants are usually auto-generated",
            "Signal constant = 42;  # Creates constant combinator",
        ],
        "TrainStop": [
            "# Train station with circuit control",
            'Entity station = place("train-stop", 0, 0, {station: "Iron Pickup"});',
            "station.enable = cargo.output > 0;",
            "# Read train info",
            "Bundle train_info = station.output;",
        ],
    }
    return examples.get(class_name, [])


def get_draftsman_source_link(cls: type) -> str:
    """Generate a GitHub link to the draftsman source file for this entity."""
    try:
        source_file = inspect.getfile(cls)
        # Extract relative path from draftsman package
        if "draftsman" in source_file:
            # Get path after 'draftsman/'
            parts = source_file.split("draftsman/")
            if len(parts) > 1:
                rel_path = "draftsman/" + parts[-1]
                # Link to forked repo
                github_url = f"https://github.com/redruin1/factorio-draftsman/blob/main/{rel_path}"
                return github_url
    except (TypeError, OSError):
        pass
    return ""


def gather_entity_info(cls: type, class_name: str) -> EntityInfo:
    """Gather all information about an entity class."""
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


# =============================================================================
# Document generation
# =============================================================================


def generate_enum_reference() -> list[str]:
    """Generate the enum reference section."""
    lines = [
        "## Enum Reference",
        "",
        "When setting enum properties in the DSL, use the **integer value** for IntEnums,",
        "or the **string value** for Literal types.",
        "",
    ]

    # IntEnums - only include relevant ones
    relevant_int_enums = [
        "LampColorMode",
        "Direction",
        "InserterModeOfOperation",
        "InserterReadMode",
        "BeltReadMode",
        "FilterMode",
        "LogisticModeOfOperation",
        "MiningDrillReadMode",
        "SiloReadMode",
    ]

    lines.append("### Integer Enums")
    lines.append("")

    for enum_name in relevant_int_enums:
        if enum_name not in ALL_INT_ENUMS:
            continue
        enum_info = ALL_INT_ENUMS[enum_name]
        lines.extend(
            [
                f'#### <a id="{enum_name.lower()}"></a>{enum_name}',
                "",
                "| DSL Value | Enum Name |",
                "|-----------|-----------|",
            ]
        )
        for member_name, value in enum_info.members.items():
            lines.append(f"| `{value}` | {member_name} |")
        lines.append("")

    # Literal types
    lines.append("### String Enums (Literal Types)")
    lines.append("")
    lines.append("These properties accept string values. Use the exact string shown.")
    lines.append("")

    for enum_name, enum_info in sorted(ALL_LITERAL_TYPES.items()):
        lines.extend(
            [
                f'#### <a id="{enum_name.lower()}"></a>{enum_name}',
                "",
                "| Valid Values |",
                "|-------------|",
            ]
        )
        for value in enum_info.members:
            lines.append(f'| `"{value}"` |')
        lines.append("")

    return lines


def generate_entity_section(entity: EntityInfo) -> list[str]:
    """Generate documentation section for a single entity."""
    lines = [f"### {entity.class_name}", ""]

    # Description
    lines.append(f"**Description:** {entity.description}")
    lines.append("")

    # Draftsman source link
    source_link = get_draftsman_source_link(entity.cls)
    if source_link:
        lines.append(f"**Draftsman Source:** [{entity.class_name} class]({source_link})")
        lines.append("")

    # Prototypes
    if entity.prototypes:
        proto_str = ", ".join(f'`"{p}"`' for p in entity.prototypes[:10])
        if len(entity.prototypes) > 10:
            proto_str += f", ... ({len(entity.prototypes)} total)"
        lines.extend([f"**Prototypes:** {proto_str}", ""])

    # Circuit connection info
    io = entity.circuit_io
    if io.has_dual_connection:
        lines.append("**Connection Type:** Dual circuit (separate input/output sides)")
    else:
        lines.append("**Connection Type:** Single circuit connection")
    lines.append("")

    # === Entity Output Section ===
    output_info = io.output_info
    if output_info.supports_output:
        lines.append("#### Reading Entity Output")
        lines.append("")
        lines.append(f"Use `entity.output` to read: **{output_info.description}**")
        lines.append("")
        lines.append("```facto")
        lines.append(
            f'Entity e = place("{entity.prototypes[0] if entity.prototypes else entity.class_name.lower()}", 0, 0);'
        )
        lines.append("Bundle signals = e.output;  # Returns all output signals")
        lines.append("```")
        lines.append("")

        if output_info.enable_properties:
            lines.append("**Enable properties:**")
            lines.append("")
            for prop, desc in output_info.enable_properties.items():
                lines.append(f"- `{prop}`: {desc}")
            lines.append("")

    # DSL Examples
    if entity.dsl_examples:
        lines.extend(["#### DSL Examples", "", "```facto"])
        lines.extend(entity.dsl_examples)
        lines.extend(["```", ""])

    # === Settable Properties Section ===
    settable_props = [p for p in entity.properties if not p.is_signal_property]

    if settable_props:
        lines.append("#### Settable Properties")
        lines.append("")
        lines.append('Set at placement: `place("name", x, y, {prop: value})`')
        lines.append("")
        lines.append("| Property | Type | Default | Example |")
        lines.append("|----------|------|---------|---------|")

        for prop in settable_props:
            type_str = prop.dsl_type
            if not prop.is_dsl_supported:
                type_str = f"{prop.dsl_type} ⚠️"
            example = f"`{prop.example_value}`" if prop.example_value else ""
            lines.append(f"| `{prop.name}` | {type_str} | {prop.default_value} | {example} |")
        lines.append("")

    # Signal configuration properties
    signal_props = [p for p in entity.properties if p.is_signal_property]
    if signal_props:
        lines.append("#### Signal Configuration")
        lines.append("")
        lines.append("Properties for configuring which signals the entity uses:")
        lines.append("")
        lines.append("| Property | Type | Description |")
        lines.append("|----------|------|-------------|")
        for prop in signal_props:
            lines.append(
                f"| `{prop.name}` | {prop.dsl_type} | {prop.description[:60]}{'...' if len(prop.description) > 60 else ''} |"
            )
        lines.append("")

    lines.extend(["---", ""])
    return lines


# Entity categorization
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
    ],
    "Train System": [
        "TrainStop",
        "RailSignal",
        "RailChainSignal",
        "Locomotive",
        "CargoWagon",
        "FluidWagon",
        "ArtilleryWagon",
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
        "Reactor",
    ],
    "Storage": [
        "Container",
        "LogisticPassiveContainer",
        "LogisticActiveContainer",
        "LogisticStorageContainer",
        "LogisticRequestContainer",
        "LogisticBufferContainer",
    ],
    "Power": [
        "ElectricPole",
        "PowerSwitch",
        "Accumulator",
        "SolarPanel",
    ],
    "Fluids": ["Pump", "StorageTank", "OffshorePump", "Pipe", "PipeToGround"],
    "Combat": [
        "Radar",
        "ArtilleryTurret",
        "AmmoTurret",
        "ElectricTurret",
        "FluidTurret",
        "Wall",
        "Gate",
        "LandMine",
    ],
    "Robots & Logistics": ["Roboport"],
    "Space": ["SpacePlatformHub", "CargoLandingPad", "AsteroidCollector", "CargoBay", "Thruster"],
    "Misc": [
        "ProgrammableSpeaker",
        "Car",
        "SpiderVehicle",
        "HeatPipe",
        "HeatInterface",
    ],
}


def generate_documentation() -> str:
    """Generate complete documentation."""
    lines = [
        "# Entity Reference for Facto",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}",
        f"**Draftsman version:** {draftsman.__version__}",
        "",
        "This is the **complete reference** for all entities available in the DSL.",
        "",
        "## Table of Contents",
        "",
        "- [Using Entities in the DSL](#using-entities-in-the-dsl)",
        "- [Reading Entity Outputs](#reading-entity-outputs)",
        "- [Enum Reference](#enum-reference)",
    ]

    # Category links
    for category in ENTITY_CATEGORIES:
        anchor = category.lower().replace(" ", "-").replace("&", "").replace("  ", "-")
        lines.append(f"- [{category}](#{anchor})")
    lines.append("- [Other Entities](#other-entities)")
    lines.append("")

    # Usage section
    lines.extend(
        [
            "## Using Entities in the DSL",
            "",
            "### How the Compiler Handles Entities",
            "",
            "When you use `place()` in Facto, the compiler creates a corresponding",
            "[Draftsman](https://github.com/Snagnar/factorio-draftsman) entity object.",
            "Properties specified in the placement object (the `{...}` part) are passed directly",
            "to Draftsman as Python attributes during entity construction. The compiler validates",
            "that property names and types match what Draftsman expects for that entity class.",
            "",
            "Circuit-controlled properties (like `entity.enable = expression`) are handled differently:",
            "the compiler generates the necessary combinator logic and wire connections to implement",
            "the circuit behavior, then sets the appropriate control properties on the entity.",
            "",
            "### Placement Syntax",
            "",
            "```facto",
            'Entity name = place("prototype-name", x, y, {prop1: value1, prop2: value2});',
            "```",
            "",
            "### Setting Properties",
            "",
            "**At placement time:**",
            "```facto",
            'Entity lamp = place("small-lamp", 0, 0, {use_colors: 1, color_mode: 1});',
            "```",
            "",
            "**After placement (circuit-controlled):**",
            "```facto",
            "lamp.enable = signal > 0;",
            "lamp.r = red_value;",
            "```",
            "",
        ]
    )

    # Reading Entity Outputs - NEW SECTION
    lines.extend(
        [
            "## Reading Entity Outputs",
            "",
            "Most entities can output circuit signals. Access them using `.output`:",
            "",
            "```facto",
            "# Read all signals from a container",
            'Entity chest = place("iron-chest", 0, 0, {read_contents: 1});',
            "Bundle contents = chest.output;",
            "",
            "# Extract a specific signal",
            'Signal iron_count = contents["iron-plate"];',
            "",
            "# Use in calculations",
            "Signal need_more = (iron_count < 100) : 1;",
            "```",
            "",
            "### Output Types by Entity",
            "",
            "| Entity Type | What `.output` Returns | Enable Property |",
            "|-------------|------------------------|-----------------|",
            "| Combinators | Computed result signals | (always active) |",
            "| Containers | Item counts | `read_contents: 1` |",
            "| Storage Tanks | Fluid level | (always active) |",
            "| Inserters | Hand contents | `read_hand_contents: 1` |",
            "| Belts | Belt contents | `read_contents: 1` |",
            "| Mining Drills | Resource amounts | `read_resources: 1` |",
            "| Train Stops | Train ID, count | `read_stopped_train: 1`, etc. |",
            "| Rail Signals | Signal state | `read_signal: 1` |",
            "| Roboports | Robot counts | `read_logistics: 1` |",
            "",
            "### Note on Combinators",
            "",
            "For combinators (arithmetic, decider, selector), the `.output` reads from the",
            "**output side** of the combinator, which is the result of its computation.",
            "",
        ]
    )

    # Enum reference
    lines.extend(generate_enum_reference())

    # Get all entity classes
    entity_classes = get_all_entity_classes()
    categorized = set()

    # Generate sections by category
    for category, class_names in ENTITY_CATEGORIES.items():
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

    # Uncategorized entities
    uncategorized = [name for name in sorted(entity_classes.keys()) if name not in categorized]
    if uncategorized:
        lines.extend(
            [
                "## Other Entities",
                "",
                "Additional entities not in the main categories:",
                "",
            ]
        )
        for class_name in uncategorized:
            cls = entity_classes[class_name]
            entity = gather_entity_info(cls, class_name)
            lines.extend(generate_entity_section(entity))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate entity documentation for the Facto")
    parser.add_argument("--output", "-o", help="Output file path (default: stdout)")
    parser.add_argument("--update", help="Update existing file (targeted changes)")
    args = parser.parse_args()

    print("Gathering entity information...", file=sys.stderr)
    content = generate_documentation()

    if args.update:
        # TODO: Implement targeted update mode
        # For now, just regenerate
        print("Note: Full regeneration mode (targeted update not yet implemented)", file=sys.stderr)
        with open(args.update, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Documentation written to: {args.update}", file=sys.stderr)
    elif args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Documentation written to: {args.output}", file=sys.stderr)
    else:
        print(content)


if __name__ == "__main__":
    main()
