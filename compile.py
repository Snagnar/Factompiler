#!/usr/bin/env python3
"""
Factompiler - Facto Language Compiler (source-checkout entry point)

This is a thin wrapper around ``dsl_compiler.cli`` (the ``factompile``
command installed by pip), so both behave identically.

Usage:
    python compile.py input.facto                     # Print blueprint to stdout
    python compile.py input.facto -o output.blueprint # Save blueprint to file
    python compile.py input.facto --power-poles       # Add medium power poles
    python compile.py input.facto --power-poles big   # Add big power poles
    python compile.py --input "Signal a = 5;"         # Compile a string
"""

from pathlib import Path

from dsl_compiler.cli import compile_dsl_source, main, setup_logging, validate_power_poles
from dsl_compiler.src.common.constants import DEFAULT_CONFIG, CompilerConfig

__all__ = [
    "compile_dsl_file",
    "compile_dsl_source",
    "main",
    "setup_logging",
    "validate_power_poles",
]


def compile_dsl_file(
    input_path: Path,
    program_name: str | None = None,
    optimize: bool = True,
    log_level: str = "error",
    power_pole_type: str | None = None,
    use_json: bool = False,
    config: CompilerConfig = DEFAULT_CONFIG,
    max_layout_retries: int = 3,
) -> tuple[bool, str, list]:
    """Compile a Facto source file (see ``compile_dsl_source``)."""
    if not input_path.exists():
        return False, f"Input file '{input_path}' does not exist", []
    try:
        source = input_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Failed to read input file: {e}", []
    if program_name is None:
        program_name = input_path.stem.replace("_", " ").title()
    return compile_dsl_source(
        source,
        source_name=str(input_path.resolve()),
        program_name=program_name,
        optimize=optimize,
        log_level=log_level,
        power_pole_type=power_pole_type,
        use_json=use_json,
        config=config,
        max_layout_retries=max_layout_retries,
    )


if __name__ == "__main__":
    main()
