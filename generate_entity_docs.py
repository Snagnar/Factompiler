#!/usr/bin/env python3
"""
Generate Markdown documentation for all Draftsman entity types and their properties.

This script dynamically introspects the draftsman library to create a comprehensive
reference of all blueprintable entities and their settable properties.

All information is extracted dynamically from draftsman's internal structure:
- Entity classes and their inheritance hierarchy
- Attributes defined via attrs
- Export mappings registered with draftsman_converters
- Docstrings parsed from source code
- Mixin composition

Usage:
    python generate_entity_docs.py > ENTITY_REFERENCE.md

Or with output file:
    python generate_entity_docs.py --output docs/ENTITY_REFERENCE.md
"""

from __future__ import annotations
import argparse
import attrs
import inspect
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass, field

# Attempt to import draftsman
try:
    import draftsman
    from draftsman.serialization import draftsman_converters
    from draftsman.classes.entity import Entity
    from draftsman.classes.exportable import Exportable
    from draftsman import entity as entity_module
    from draftsman.data import entities as entity_data
except ImportError as e:
    print(
        f"Error: Could not import draftsman. Install it with: pip install factorio-draftsman"
    )
    print(f"Details: {e}")
    sys.exit(1)


@dataclass
class PropertyInfo:
    """Information about a single property/attribute."""

    name: str
    type_name: str
    is_exported: bool
    json_path: Optional[
        Tuple[str, ...]
    ]  # Path in blueprint JSON, e.g. ("control_behavior", "circuit_enabled")
    default_value: Any
    description: str
    defined_in: str  # Class where this attribute is defined
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityInfo:
    """Information about an entity class."""

    class_name: str
    cls: type
    description: str
    entity_names: List[str]
    mixins: List[str]
    properties: List[PropertyInfo]
    rtd_url: str


def get_all_entity_classes() -> Dict[str, type]:
    """Get all entity classes from draftsman.entity module."""
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


def get_export_mapping(
    cls: type, version: Tuple[int, ...] = (2, 0)
) -> Dict[str, Tuple[str, ...]]:
    """
    Get the export mapping for an entity class.

    Returns a dict mapping attribute names to their JSON paths in the blueprint.
    """
    try:
        version_data = draftsman_converters.get_version(version)
        conv = version_data.get_converter()
        struct_dict = version_data.get_structure_dict(cls, conv)

        # Convert the structure dict to attr_name -> json_path mapping
        mapping = {}
        for json_path, attr_info in struct_dict.items():
            if attr_info is None:
                continue

            # Handle different formats of attr_info
            if isinstance(attr_info, str):
                attr_name = attr_info
            elif isinstance(attr_info, tuple):
                attr_name = (
                    attr_info[0].name
                    if hasattr(attr_info[0], "name")
                    else str(attr_info[0])
                )
            elif isinstance(attr_info, dict):
                attr_name = attr_info.get("name", str(attr_info))
            elif hasattr(attr_info, "name"):
                attr_name = attr_info.name
            else:
                continue

            mapping[attr_name] = json_path

        return mapping
    except Exception as e:
        return {}


def get_class_docstrings(cls: type) -> Dict[str, str]:
    """
    Parse docstrings from the class source code for each attribute.
    Returns a dict mapping attribute names to their docstrings.
    """
    docstrings = {}

    try:
        # Get the source for this class and all its base classes
        sources_to_check = []
        for klass in cls.__mro__:
            if klass in (object,) or not hasattr(klass, "__module__"):
                continue
            try:
                source = inspect.getsource(klass)
                sources_to_check.append((klass.__name__, source))
            except (OSError, TypeError):
                pass

        for class_name, source in sources_to_check:
            # Pattern 1: attr_name: Type = attrs.field(...) followed by """docstring"""
            # This handles multi-line attrs.field() definitions with balanced parens
            # We need to handle the case where attrs.field() spans multiple lines
            attr_pattern = re.compile(
                r'^\s*(\w+)\s*:\s*[^=]+?=\s*attrs\.field\([\s\S]*?\)\s*\n\s*"""([\s\S]*?)"""',
                re.MULTILINE,
            )

            for match in attr_pattern.finditer(source):
                attr_name = match.group(1)
                docstring = match.group(2).strip()
                if attr_name not in docstrings:
                    docstrings[attr_name] = docstring

            # Pattern 2: Handle attrs_reuse pattern
            reuse_pattern = re.compile(
                r'^\s*(\w+)\s*=\s*attrs_reuse\([\s\S]*?\)\s*\n\s*"""([\s\S]*?)"""',
                re.MULTILINE,
            )

            for match in reuse_pattern.finditer(source):
                attr_name = match.group(1)
                docstring = match.group(2).strip()
                if attr_name not in docstrings:
                    docstrings[attr_name] = docstring

            # Pattern 3: Simple attribute with docstring (attr: Type = value\n"""doc""")
            simple_pattern = re.compile(
                r'^\s*(\w+)\s*:\s*[^=]+?=\s*[^\n]+\n\s*"""([\s\S]*?)"""', re.MULTILINE
            )

            for match in simple_pattern.finditer(source):
                attr_name = match.group(1)
                docstring = match.group(2).strip()
                if attr_name not in docstrings:
                    docstrings[attr_name] = docstring

    except Exception as e:
        pass

    return docstrings


def extract_description_from_docstring(docstring: str) -> str:
    """Extract a clean description from a docstring, removing RST directives."""
    if not docstring:
        return ""

    # Remove the .. serialized:: directive and its content
    docstring = re.sub(
        r"\.\.\s*serialized::\s*\n\s*This attribute is imported/exported from blueprint strings\.?\s*",
        "",
        docstring,
    )

    # Remove other RST directives like .. NOTE::, .. WARNING::, etc.
    docstring = re.sub(
        r"\.\.\s*\w+::\s*\n\s*(.*?)(?=\n\n|\Z)", "", docstring, flags=re.DOTALL
    )

    # Remove :py:attr: and similar RST roles, keeping just the text
    docstring = re.sub(r":py:\w+:`[~.]?([^`]+)`", r"\1", docstring)

    # Clean up multiple newlines and whitespace
    docstring = re.sub(r"\n\s*\n", "\n", docstring)
    docstring = re.sub(r"\s+", " ", docstring)

    return docstring.strip()


def is_serialized(docstring: str) -> bool:
    """Check if a docstring contains the .. serialized:: directive."""
    return ".. serialized::" in docstring if docstring else False


def get_mixins(cls: type) -> List[str]:
    """Get the list of mixin classes that this entity inherits from."""
    mixins = []
    for base in cls.__mro__:
        if base in (object, Entity, Exportable, cls):
            continue
        name = base.__name__
        if "Mixin" in name:
            mixins.append(name)
    return mixins


def get_defining_class(cls: type, attr_name: str) -> str:
    """Find which class in the MRO defines this attribute."""
    for klass in cls.__mro__:
        if klass in (object,):
            continue
        try:
            klass_fields = attrs.fields(klass)
            for field in klass_fields:
                if field.name == attr_name:
                    # Check if this class actually defines it (not inherited)
                    if hasattr(klass, "__attrs_own_attrs__"):
                        if attr_name in [f.name for f in klass.__attrs_own_attrs__]:
                            return klass.__name__
        except attrs.exceptions.NotAnAttrsClassError:
            continue
    return cls.__name__


def format_type_name(type_hint: Any) -> str:
    """Format a type hint into a readable string."""
    if type_hint is None:
        return "Any"

    type_str = str(type_hint)

    # Clean up common patterns
    type_str = re.sub(r"typing\.", "", type_str)
    type_str = re.sub(r"draftsman\.signatures\.", "", type_str)
    type_str = re.sub(r"draftsman\.classes\.vector\.", "", type_str)
    type_str = re.sub(r"draftsman\.classes\.entity\._PosVector", "Vector", type_str)
    type_str = re.sub(
        r"draftsman\.classes\.entity\._TileVector", "TileVector", type_str
    )
    type_str = re.sub(r"draftsman\.constants\.", "", type_str)
    type_str = re.sub(r"draftsman\.prototypes\.[^.]+\.", "", type_str)
    type_str = re.sub(r"<class '([^']+)'>", r"\1", type_str)
    type_str = re.sub(r"Annotated\[([^,]+),.*?\]", r"\1", type_str)
    type_str = re.sub(r"<enum '([^']+)'>", r"\1", type_str)

    # Simplify long nested types
    if len(type_str) > 60:
        # Truncate very long type strings
        type_str = type_str[:57] + "..."

    return type_str


def format_default(default: Any) -> str:
    """Format a default value into a readable string."""
    if default is attrs.NOTHING:
        return "required"
    if isinstance(default, attrs.Factory):
        return "factory"
    if default is None:
        return "None"
    if isinstance(default, str):
        return f'"{default}"'
    return str(default)


def get_entity_properties(cls: type) -> List[PropertyInfo]:
    """
    Get all properties for an entity class, dynamically determined.
    """
    properties = []

    # Attributes to skip (internal/non-user-facing)
    SKIP_ATTRS = {
        "_parent",
        "_connections",
        "_entity_number",
        "extra_keys",  # Internal for unknown keys
    }

    try:
        # Get the export mapping for this class
        export_mapping = get_export_mapping(cls)

        # Get all attrs fields
        all_fields = attrs.fields(cls)

        # Get docstrings from source
        docstrings = get_class_docstrings(cls)

        for field in all_fields:
            name = field.name

            # Skip internal attributes
            if name in SKIP_ATTRS or name.startswith("_"):
                continue

            # Check if this attribute is explicitly omitted
            omit = field.metadata.get("omit", None)

            # Determine if it's exported to blueprints
            is_exported = name in export_mapping
            if omit is False:  # Explicitly not omitted = definitely exported
                is_exported = True
            elif omit is True:  # Explicitly omitted = not exported
                is_exported = False

            # Get JSON path if exported
            json_path = export_mapping.get(name) if is_exported else None

            # Get docstring
            docstring = docstrings.get(name, "")

            # Also check if serialized marker is in docstring
            if is_serialized(docstring):
                is_exported = True

            # Get defining class
            defined_in = get_defining_class(cls, name)

            prop_info = PropertyInfo(
                name=name,
                type_name=format_type_name(field.type),
                is_exported=is_exported,
                json_path=json_path,
                default_value=format_default(field.default),
                description=extract_description_from_docstring(docstring),
                defined_in=defined_in,
                metadata=dict(field.metadata),
            )
            properties.append(prop_info)

    except Exception as e:
        pass

    # Sort: exported first, then by name
    properties.sort(key=lambda p: (not p.is_exported, p.name))

    return properties


def get_entity_names(cls: type) -> List[str]:
    """Get all Factorio entity names that map to this Draftsman class."""
    try:
        instance = cls()
        return (
            list(instance.similar_entities)
            if hasattr(instance, "similar_entities")
            else []
        )
    except Exception:
        return []


def get_class_description(cls: type) -> str:
    """Get the class docstring, cleaned up."""
    doc = cls.__doc__ or ""
    lines = doc.strip().split("\n")
    result = []
    for line in lines:
        line = line.strip()
        if not line and result:
            break
        if line:
            result.append(line)
    return " ".join(result) if result else "No description available."


def get_rtd_url(class_name: str) -> str:
    """Generate ReadTheDocs URL for a class."""
    # Convert CamelCase to snake_case
    rtd_name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
    rtd_name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", rtd_name).lower()
    return f"https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/{rtd_name}.html"


def gather_all_entity_info() -> List[EntityInfo]:
    """Gather information about all entity classes."""
    entity_classes = get_all_entity_classes()
    entities_info = []

    for class_name, cls in sorted(entity_classes.items()):
        info = EntityInfo(
            class_name=class_name,
            cls=cls,
            description=get_class_description(cls),
            entity_names=get_entity_names(cls),
            mixins=get_mixins(cls),
            properties=get_entity_properties(cls),
            rtd_url=get_rtd_url(class_name),
        )
        entities_info.append(info)

    return entities_info


def generate_markdown(
    entities: List[EntityInfo], output_file: Optional[str] = None
) -> str:
    """Generate complete markdown documentation."""
    lines = []

    # Header
    lines.append("# Draftsman Entity Reference")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Draftsman version:** {draftsman.__version__}")
    lines.append(f"**Total entity classes:** {len(entities)}")
    lines.append("")
    lines.append(
        "This document lists all blueprintable entity types supported by Draftsman "
    )
    lines.append("and the properties that can be set on each entity type.")
    lines.append("")
    lines.append(
        "**Note:** All information in this document is dynamically extracted from "
    )
    lines.append(
        "the Draftsman library internals, ensuring it stays up-to-date with new versions."
    )
    lines.append("")
    lines.append("## Legend")
    lines.append("")
    lines.append(
        "- **✓ Blueprint** = Property is exported to/imported from blueprint strings"
    )
    lines.append(
        "- **JSON Path** = Location in the blueprint JSON structure (e.g., `control_behavior.circuit_enabled`)"
    )
    lines.append("- Properties are sorted with blueprint-relevant ones first")
    lines.append("")
    lines.append("## External Resources")
    lines.append("")
    lines.append(
        "- [Draftsman Documentation](https://factorio-draftsman.readthedocs.io/en/latest/)"
    )
    lines.append("- [Draftsman GitHub](https://github.com/redruin1/factorio-draftsman)")
    lines.append(
        "- [Factorio Wiki - Blueprint String Format](https://wiki.factorio.com/Blueprint_string_format)"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Table of Contents
    lines.append("## Table of Contents")
    lines.append("")
    for entity in entities:
        anchor = entity.class_name.lower().replace(" ", "-")
        lines.append(f"- [{entity.class_name}](#{anchor})")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Generate each entity section
    for entity in entities:
        lines.append(f"## {entity.class_name}")
        lines.append("")

        # Description
        lines.append(f"**Description:** {entity.description}")
        lines.append("")

        # Entity names (prototypes)
        if entity.entity_names:
            if len(entity.entity_names) <= 5:
                names_str = ", ".join(f"`{n}`" for n in entity.entity_names)
            else:
                names_str = (
                    ", ".join(f"`{n}`" for n in entity.entity_names[:5])
                    + f" ... ({len(entity.entity_names)} total)"
                )
            lines.append(f"**Factorio prototypes:** {names_str}")
            lines.append("")

        # Mixins (behaviors)
        if entity.mixins:
            mixins_str = ", ".join(entity.mixins[:8])
            if len(entity.mixins) > 8:
                mixins_str += f" ... (+{len(entity.mixins) - 8} more)"
            lines.append(f"**Behaviors (Mixins):** {mixins_str}")
            lines.append("")

        # ReadTheDocs link
        lines.append(f"**Docs:** [ReadTheDocs]({entity.rtd_url})")
        lines.append("")

        # Properties
        if entity.properties:
            # Separate exported and non-exported
            exported_props = [p for p in entity.properties if p.is_exported]
            other_props = [p for p in entity.properties if not p.is_exported]

            if exported_props:
                lines.append("### Blueprint Properties")
                lines.append("")
                lines.append(
                    "These properties are saved to/loaded from blueprint strings."
                )
                lines.append("")
                lines.append("| Property | Type | JSON Path | Default | Description |")
                lines.append("|----------|------|-----------|---------|-------------|")

                for prop in exported_props:
                    json_path = ".".join(prop.json_path) if prop.json_path else "—"
                    desc = (
                        prop.description[:80] + "..."
                        if len(prop.description) > 80
                        else prop.description
                    )
                    lines.append(
                        f"| `{prop.name}` | {prop.type_name} | `{json_path}` | {prop.default_value} | {desc} |"
                    )
                lines.append("")

            if other_props:
                lines.append("### Other Properties")
                lines.append("")
                lines.append(
                    "These properties are computed, internal, or not part of the blueprint format."
                )
                lines.append("")
                lines.append("| Property | Type | Default | Defined In |")
                lines.append("|----------|------|---------|------------|")

                for prop in other_props:
                    lines.append(
                        f"| `{prop.name}` | {prop.type_name} | {prop.default_value} | {prop.defined_in} |"
                    )
                lines.append("")
        else:
            lines.append("*No settable properties found.*")
            lines.append("")

        lines.append("---")
        lines.append("")

    content = "\n".join(lines)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Documentation written to: {output_file}", file=sys.stderr)

    return content


def generate_json(entities: List[EntityInfo]) -> str:
    """Generate JSON output for programmatic use."""
    import json

    result = {}
    for entity in entities:
        result[entity.class_name] = {
            "class_name": entity.class_name,
            "description": entity.description,
            "entity_names": entity.entity_names,
            "mixins": entity.mixins,
            "rtd_url": entity.rtd_url,
            "properties": [
                {
                    "name": p.name,
                    "type": p.type_name,
                    "is_exported": p.is_exported,
                    "json_path": list(p.json_path) if p.json_path else None,
                    "default": p.default_value,
                    "description": p.description,
                    "defined_in": p.defined_in,
                }
                for p in entity.properties
            ],
        }

    return json.dumps(result, indent=2)


def generate_dsl_docs(entities: List[EntityInfo]) -> str:
    """Generate documentation focused on circuit DSL usage patterns."""
    lines = []

    lines.append("# Entity Properties for Circuit DSL")
    lines.append("")
    lines.append(f"Generated for Draftsman {draftsman.__version__}")
    lines.append("")
    lines.append(
        "This reference shows entity properties relevant for circuit control in your DSL."
    )
    lines.append(
        "All information is dynamically extracted from Draftsman's internal structure."
    )
    lines.append("")

    # Categories of entities most relevant for circuit DSL
    categories = {
        "Combinators": [
            "ArithmeticCombinator",
            "DeciderCombinator",
            "ConstantCombinator",
            "SelectorCombinator",
        ],
        "Lamps & Displays": ["Lamp", "DisplayPanel"],
        "Inserters": ["Inserter"],
        "Belts": [
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
        ],
        "Production": [
            "AssemblingMachine",
            "Furnace",
            "MiningDrill",
            "Lab",
            "RocketSilo",
            "Beacon",
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
        "Power": ["ElectricPole", "PowerSwitch", "Accumulator"],
        "Fluid": ["Pump", "StorageTank", "OffshorePump", "InfinityPipe", "Valve"],
        "Other": ["ProgrammableSpeaker", "Roboport", "Radar"],
    }

    entity_map = {e.class_name: e for e in entities}

    for category, class_names in categories.items():
        lines.append(f"## {category}")
        lines.append("")

        for class_name in class_names:
            if class_name not in entity_map:
                continue

            entity = entity_map[class_name]
            lines.append(f"### {class_name}")
            lines.append("")

            # Entity names
            if entity.entity_names:
                if len(entity.entity_names) <= 3:
                    examples = ", ".join(f'`"{n}"`' for n in entity.entity_names)
                else:
                    examples = (
                        ", ".join(f'`"{n}"`' for n in entity.entity_names[:3])
                        + f" ({len(entity.entity_names)} total)"
                    )
                lines.append(f"**Prototypes:** {examples}")
                lines.append("")

            # Filter for circuit-related properties
            circuit_keywords = {
                "circuit",
                "enable",
                "signal",
                "condition",
                "read",
                "send",
                "logistic",
                "mode",
                "filter",
                "priority",
                "limit",
                "train",
                "color",
            }

            circuit_props = []
            static_props = []

            for prop in entity.properties:
                if not prop.is_exported:
                    continue

                prop_lower = prop.name.lower()
                if any(kw in prop_lower for kw in circuit_keywords):
                    circuit_props.append(prop)
                else:
                    static_props.append(prop)

            if circuit_props:
                lines.append("**Circuit/Dynamic Properties:**")
                lines.append("")
                lines.append("| Property | Type | JSON Path |")
                lines.append("|----------|------|-----------|")
                for prop in circuit_props:
                    json_path = ".".join(prop.json_path) if prop.json_path else "—"
                    lines.append(
                        f"| `{prop.name}` | {prop.type_name} | `{json_path}` |"
                    )
                lines.append("")

            if static_props:
                lines.append("**Static Properties:**")
                lines.append("")
                display_props = static_props[:6]
                for prop in display_props:
                    lines.append(f"- `{prop.name}` ({prop.type_name})")
                if len(static_props) > 6:
                    lines.append(f"- ... and {len(static_props) - 6} more")
                lines.append("")

        lines.append("")

    # Add uncategorized entities
    categorized = set()
    for class_names in categories.values():
        categorized.update(class_names)

    uncategorized = [e for e in entities if e.class_name not in categorized]
    if uncategorized:
        lines.append("---")
        lines.append("")
        lines.append("## All Other Entities")
        lines.append("")
        for entity in uncategorized:
            lines.append(f"- [{entity.class_name}]({entity.rtd_url})")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Draftsman entity documentation dynamically extracted from the library"
    )
    parser.add_argument("--output", "-o", help="Output file path (default: stdout)")
    parser.add_argument(
        "--format",
        "-f",
        choices=["markdown", "json", "dsl"],
        default="markdown",
        help="Output format: 'markdown' (full), 'json' (programmatic), 'dsl' (circuit-focused)",
    )

    args = parser.parse_args()

    # Gather all entity information
    print("Gathering entity information...", file=sys.stderr)
    entities = gather_all_entity_info()
    print(f"Found {len(entities)} entity classes", file=sys.stderr)

    # Generate output
    if args.format == "markdown":
        content = generate_markdown(entities, args.output)
    elif args.format == "json":
        content = generate_json(entities)
    elif args.format == "dsl":
        content = generate_dsl_docs(entities)

    if not args.output:
        print(content)
    elif args.format != "markdown":  # markdown already writes to file
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Output written to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
