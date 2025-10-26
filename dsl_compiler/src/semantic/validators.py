"""Helper utilities for semantic analysis diagnostics."""

from pathlib import Path
from typing import Optional

from dsl_compiler.src.ast import ASTNode


EXPLAIN_MODE = False


def render_source_location(
    node: Optional[ASTNode], default_file: Optional[str] = None
) -> Optional[str]:
    """Format a human-friendly ``file:line`` string for an AST node."""

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
