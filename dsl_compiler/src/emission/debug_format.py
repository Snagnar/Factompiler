"""Utilities for formatting debug metadata into blueprint-friendly annotations."""

from __future__ import annotations

from typing import Any, Dict, Optional


def _normalize_location(location: Any) -> Optional[str]:
    if location is None:
        return None
    if isinstance(location, str):
        return location
    if hasattr(location, "__str__"):
        value = str(location)
        return value if value else None
    return None


def format_entity_description(debug_info: Optional[Dict[str, Any]]) -> Optional[str]:
    """Compose a human-readable description string from debug metadata.

    The formatter prefers the logical signal name when available, appends the
    resolved physical Factorio signal in parentheses when it differs, and
    optionally suffixes the source location (``file:line``) for quick lookup in
    the original DSL program.
    """

    if not debug_info:
        return None

    name = debug_info.get("name")
    label = debug_info.get("label")
    resolved_signal = debug_info.get("factorio_signal") or debug_info.get("resolved_signal")
    declared_type = debug_info.get("declared_type")
    location = _normalize_location(debug_info.get("location"))

    primary_label = name or label or resolved_signal

    description_parts: list[str] = []
    if primary_label:
        entry = f"signal {primary_label}"
        if resolved_signal and resolved_signal != primary_label:
            entry += f" ({resolved_signal})"
        description_parts.append(entry)
    elif resolved_signal:
        description_parts.append(f"signal {resolved_signal}")

    if label and label != primary_label:
        description_parts.append(f"as {label}")

    if declared_type and declared_type not in {"Signal", "Memory", "variable", "memory"}:
        description_parts.append(f"type {declared_type}")

    if location:
        description_parts.append(f"@ {location}")

    if not description_parts:
        return None

    return " ".join(description_parts)


__all__ = ["format_entity_description"]
