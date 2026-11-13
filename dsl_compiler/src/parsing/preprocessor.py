from __future__ import annotations

import os
from pathlib import Path

"""Import preprocessing utilities for the Facto language."""

# The package directory (``dsl_compiler/``). The standard library ships inside
# it as ``dsl_compiler/lib/*.facto``, so ``import "lib/math.facto";`` resolves
# both from a source checkout and from a pip-installed wheel.
_PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent

if "FACTORIO_IMPORT_PATH" not in os.environ:
    os.environ["FACTORIO_IMPORT_PATH"] = ".;example_programs"


def _search_paths(base_path: Path | None) -> list[str]:
    """Directories searched for imports, in priority order.

    1. The directory of the importing file (if known)
    2. Entries of ``FACTORIO_IMPORT_PATH`` (``;``-separated)
    3. The package directory, which contains the bundled ``lib/`` folder,
       and ``lib/`` itself (so ``import "math.facto";`` also works)
    """
    paths = [p for p in os.environ.get("FACTORIO_IMPORT_PATH", "").split(";") if p]
    if base_path is not None:
        paths.insert(0, str(base_path.resolve()))
    paths.append(str(_PACKAGE_DIR))
    paths.append(str(_PACKAGE_DIR / "lib"))
    return paths


def resolve_import_path(import_path: str, base_path: Path | None = None) -> Path:
    """Resolve the full path of an import statement."""
    import_path_obj = Path(import_path)
    base_paths = _search_paths(base_path)
    for base in base_paths:
        candidate_path = Path(base) / import_path_obj
        if candidate_path.exists():
            return candidate_path.resolve()
    raise FileNotFoundError(
        f"Import file not found: {import_path} (searched: {', '.join(base_paths)})"
    )


# One entry per line of preprocessed output: (source file or None, 1-based line)
LineMap = list[tuple[str | None, int]]


def preprocess_imports(
    source_code: str,
    base_path: Path | None = None,
    processed_files: set[Path] | None = None,
) -> str:
    """Inline imports by recursively expanding `import "...";` statements."""
    code, _ = preprocess_imports_with_map(source_code, base_path, processed_files)
    return code


def preprocess_imports_with_map(
    source_code: str,
    base_path: Path | None = None,
    processed_files: set[Path] | None = None,
    source_name: str | None = None,
) -> tuple[str, LineMap]:
    """Inline imports and record where every output line came from.

    Imports are expanded textually, which shifts line numbers. The returned
    line map translates a line of the expanded code back to its original
    file and line, so diagnostics can point at the right place.
    """
    if processed_files is None:
        processed_files = set()

    processed_lines: list[str] = []
    line_map: LineMap = []

    def emit(text: str, file: str | None, lineno: int) -> None:
        processed_lines.append(text)
        line_map.append((file, lineno))

    for lineno, line in enumerate(source_code.split("\n"), start=1):
        stripped = line.strip()

        if stripped.startswith('import "') and stripped.endswith('";'):
            raw_import_path = stripped[8:-2]
            import_path = Path(raw_import_path)
            if import_path.suffix != ".facto":
                import_path = import_path.with_suffix(".facto")

            try:
                file_path = resolve_import_path(str(import_path), base_path)
            except FileNotFoundError as exc:
                where = Path(source_name).name if source_name else "<input>"
                raise FileNotFoundError(f"{where}:{lineno}: {exc}") from exc
            if file_path in processed_files:
                emit(f"# Skipped circular import: {import_path}", source_name, lineno)
                continue

            processed_files.add(file_path)
            with open(file_path, encoding="utf-8") as handle:
                imported_content = handle.read()

            sub_code, sub_map = preprocess_imports_with_map(
                imported_content,
                base_path=file_path.parent,
                processed_files=processed_files,
                source_name=str(file_path),
            )

            emit(f"# --- Imported from {file_path} ---", source_name, lineno)
            processed_lines.extend(sub_code.split("\n"))
            line_map.extend(sub_map)
            emit(f"# --- End import {file_path} ---", source_name, lineno)
        else:
            emit(line, source_name, lineno)

    return "\n".join(processed_lines), line_map
