from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Set

"""Import preprocessing utilities for the Factorio Circuit DSL."""

if "FACTORIO_IMPORT_PATH" not in os.environ:
    os.environ["FACTORIO_IMPORT_PATH"] = ".;tests/sample_programs" + str(
        Path(__file__).resolve().parent.parent.parent
    )


def resolve_import_path(import_path: str, base_path: Optional[Path] = None) -> Path:
    """Resolve the full path of an import statement."""
    import_path_obj = Path(import_path)
    base_paths = os.environ["FACTORIO_IMPORT_PATH"].split(";")
    if base_path is not None:
        base_paths.insert(0, str(base_path.resolve()))
    for base in base_paths:
        candidate_path = Path(base) / import_path_obj
        if candidate_path.exists():
            return candidate_path.resolve()
    raise FileNotFoundError(f"Import file not found: {import_path}")


def preprocess_imports(
    source_code: str,
    base_path: Optional[Path] = None,
    processed_files: Optional[Set[Path]] = None,
) -> str:
    """Inline imports by recursively expanding `import "...";` statements."""
    if processed_files is None:
        processed_files = set()

    lines = source_code.split("\n")
    processed_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('import "') and stripped.endswith('";'):
            raw_import_path = stripped[8:-2]
            import_path = Path(raw_import_path)
            if import_path.suffix != ".fcdsl":
                import_path = import_path.with_suffix(".fcdsl")

            file_path = resolve_import_path(import_path, base_path)
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

                    processed_lines.append(f"# --- Imported from {file_path:!s} ---")
                    processed_lines.append(imported_content)
                    processed_lines.append(f"# --- End import {file_path:!s} ---")
                else:
                    processed_lines.append(line)
            except Exception:
                processed_lines.append(line)
        else:
            processed_lines.append(line)

    return "\n".join(processed_lines)
