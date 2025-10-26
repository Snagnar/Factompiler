"""Import preprocessing utilities for the Factorio Circuit DSL."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Set


def preprocess_imports(
    source_code: str,
    base_path: Optional[Path] = None,
    processed_files: Optional[Set[Path]] = None,
) -> str:
    """Inline imports by recursively expanding `import "...";` statements."""
    if processed_files is None:
        processed_files = set()

    if base_path is None:
        base_path = Path("tests/sample_programs").resolve()
    else:
        base_path = base_path.resolve()

    lines = source_code.split("\n")
    processed_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('import "') and stripped.endswith('";'):
            raw_import_path = stripped[8:-2]
            import_path = Path(raw_import_path)
            if import_path.suffix != ".fcdsl":
                import_path = import_path.with_suffix(".fcdsl")

            if not import_path.is_absolute():
                file_path = (base_path / import_path).resolve()
            else:
                file_path = import_path

            if file_path in processed_files:
                processed_lines.append(f"# Skipped circular import: {import_path}")
                continue

            try:
                if file_path.exists():
                    processed_files.add(file_path)
                    with open(file_path, "r", encoding="utf-8") as handle:
                        imported_content = handle.read()

                    imported_content = preprocess_imports(
                        imported_content,
                        base_path=file_path.parent,
                        processed_files=processed_files,
                    )

                    display_path = (
                        str(import_path)
                        if import_path.is_absolute()
                        else str(import_path)
                    )
                    processed_lines.append(f"# --- Imported from {display_path} ---")
                    processed_lines.append(imported_content)
                    processed_lines.append(f"# --- End import {display_path} ---")
                else:
                    processed_lines.append(line)
            except Exception:
                processed_lines.append(line)
        else:
            processed_lines.append(line)

    return "\n".join(processed_lines)
