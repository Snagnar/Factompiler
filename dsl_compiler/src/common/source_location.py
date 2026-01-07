from dataclasses import dataclass
from pathlib import Path
from typing import Any

"""Source location utilities for tracking code positions."""


@dataclass
class SourceLocation:
    """Represents a location in source code."""

    file: str | None = None
    line: int = 0
    column: int = 0

    def __str__(self) -> str:
        """Format as file:line:col."""
        parts = []
        if self.file:
            parts.append(Path(self.file).name)
        if self.line > 0:
            parts.append(str(self.line))
            if self.column > 0:
                parts.append(str(self.column))
        return ":".join(parts) if parts else "unknown"

    @staticmethod
    def render(node: Any | None, default_file: str | None = None) -> str | None:
        """Format a human-friendly file:line string for an AST node.

        This is a compatibility method for code that uses render_source_location.
        """
        if node is None:
            return None

        filename = getattr(node, "source_file", None) or default_file
        line = getattr(node, "line", 0) or 0

        if not filename and line <= 0:
            return None

        if filename and line > 0:
            return f"{Path(filename).name}:{line}"

        if filename:
            return Path(filename).name

        if line > 0:
            return f"?:{line}"

        return None
