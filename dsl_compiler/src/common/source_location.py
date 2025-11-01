"""Source location utilities for tracking code positions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any


@dataclass
class SourceLocation:
    """Represents a location in source code."""

    file: Optional[str] = None
    line: int = 0
    column: int = 0

    @classmethod
    def from_node(cls, node: Any) -> "SourceLocation":
        """Create a SourceLocation from an AST node."""
        return cls(
            file=getattr(node, "source_file", None),
            line=getattr(node, "line", 0),
            column=getattr(node, "column", 0),
        )

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

    def is_valid(self) -> bool:
        """Check if this location has meaningful information."""
        return self.file is not None or self.line > 0
