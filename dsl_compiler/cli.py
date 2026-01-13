#!/usr/bin/env python3
"""
Factompile CLI - Command-line interface for the Facto compiler.

This module provides the entry point for the 'factompile' command installed via pip.

Usage:
    factompile input.facto                      # Compile from file
    factompile --input "Signal a = 5;"         # Compile from string
    factompile input.facto -o output.blueprint  # Save blueprint to file
    factompile input.facto --power-poles        # Add medium power poles
    factompile input.facto --power-poles big    # Add big power poles
"""

import logging
import sys
from pathlib import Path

import click

from dsl_compiler.src.common.constants import DEFAULT_CONFIG, CompilerConfig
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.emission.emitter import BlueprintEmitter
from dsl_compiler.src.ir.optimizer import ConstantPropagationOptimizer, CSEOptimizer
from dsl_compiler.src.layout.planner import LayoutPlanner
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer


def validate_power_poles(ctx, param, value):
    """Validate power pole type."""
    if value is None:
        return None

    valid_types = ["small", "medium", "big", "substation"]
    if value.lower() not in valid_types:
        raise click.BadParameter(f"must be one of: {', '.join(valid_types)}")

    return value.lower()


def compile_dsl_source(
    source_code: str,
    source_name: str = "<string>",
    program_name: str | None = None,
    optimize: bool = True,
    log_level: str = "error",
    power_pole_type: str | None = None,
    use_json: bool = False,
    config: CompilerConfig = DEFAULT_CONFIG,
    max_layout_retries: int = 3,
) -> tuple[bool, str, list]:
    """
    Compile Facto source code to blueprint string.

    Args:
        source_code: The Facto source code to compile
        source_name: Name of the source (for error messages)
        program_name: Name for the blueprint (default: derived from source_name)
        optimize: Enable IR optimizations and MST wire optimization
        log_level: Logging verbosity level
        power_pole_type: Type of power poles to add (or None for no power poles)
        use_json: If True, return JSON dict instead of compressed blueprint string
        config: Compiler configuration settings
        max_layout_retries: Maximum retry attempts for layout planning on routing failures

    Returns:
        (success: bool, result: str, diagnostics: list)
    """
    if program_name is None:
        if source_name == "<string>":
            program_name = "Facto Circuit"
        else:
            # Extract name from path-like string
            program_name = Path(source_name).stem.replace("_", " ").title()

    # Create unified diagnostics collector
    diagnostics = ProgramDiagnostics(log_level=log_level, raise_errors=True)

    # Parse
    parser = DSLParser()
    program = parser.parse(source_code.strip(), source_name)
    if diagnostics.has_errors():
        return False, "Parsing failed", diagnostics.get_messages()

    # Semantic analysis
    analyzer = SemanticAnalyzer(diagnostics=diagnostics)
    analyzer.visit(program)
    if diagnostics.has_errors():
        return False, "Semantic analysis failed", diagnostics.get_messages()

    # IR generation
    lowerer = ASTLowerer(analyzer, diagnostics)
    ir_operations = lowerer.lower_program(program)

    if diagnostics.has_errors():
        return False, "IR lowering failed", diagnostics.get_messages()

    # Optimize
    if optimize:
        # First: constant propagation and folding
        ir_operations = ConstantPropagationOptimizer().optimize(ir_operations)
        # Second: common subexpression elimination
        ir_operations = CSEOptimizer().optimize(ir_operations)

    # Layout planning
    planner = LayoutPlanner(
        lowerer.ir_builder.signal_type_map,
        diagnostics=diagnostics,
        signal_refs=lowerer.signal_refs,
        referenced_signal_names=lowerer.referenced_signal_names,
        power_pole_type=power_pole_type,
        config=config,
        use_mst_optimization=optimize,
        max_layout_retries=max_layout_retries,
    )

    layout_plan = planner.plan_layout(
        ir_operations,
        blueprint_label=f"{program_name} Blueprint"
        if program_name
        else config.default_blueprint_label,
        blueprint_description=config.default_blueprint_description,
    )

    if planner.diagnostics.has_errors():
        return False, "Layout planning failed", diagnostics.get_messages()

    # Blueprint emission
    emitter = BlueprintEmitter(diagnostics, lowerer.ir_builder.signal_type_map)
    blueprint = emitter.emit_from_plan(layout_plan)

    if diagnostics.has_errors():
        return False, "Blueprint emission failed", diagnostics.get_messages()

    # Return JSON dict or compressed blueprint string
    if use_json:
        import json

        blueprint_result = json.dumps(blueprint.to_dict())
    else:
        blueprint_result = blueprint.to_string()

    return True, blueprint_result, diagnostics.get_messages()


def setup_logging(level: str) -> None:
    """Setup logging configuration."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(level=numeric_level, format="%(levelname)s: %(message)s")


@click.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "-i",
    "--input",
    "input_string",
    type=str,
    help="Compile from string instead of file",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file for the blueprint (default: stdout)",
)
@click.option("--name", type=str, help="Blueprint name (default: derived from input filename)")
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    default="warning",
    help="Set the logging level",
)
@click.option("--no-optimize", is_flag=True, help="Disable IR optimizations")
@click.option(
    "--power-poles",
    is_flag=False,
    flag_value=DEFAULT_CONFIG.default_power_pole_type,
    default=None,
    callback=validate_power_poles,
    help=f"Add power poles (small/medium/big/substation, defaults to {DEFAULT_CONFIG.default_power_pole_type} if no value)",
)
@click.option(
    "--json",
    is_flag=True,
    help="Output blueprint in JSON format instead of compressed string format",
)
@click.option(
    "--layout-retries",
    type=int,
    default=3,
    help="Maximum retries for layout planning on routing failures (default: 3)",
)
def main(
    input_file,
    input_string,
    output,
    name,
    log_level,
    no_optimize,
    power_poles,
    json,
    layout_retries,
):
    """Compile Facto source files or strings to Factorio blueprint format."""
    setup_logging(log_level)

    # Validate input source
    if input_file and input_string:
        click.echo("Error: Cannot specify both input file and --input string", err=True)
        sys.exit(1)

    if not input_file and not input_string:
        click.echo("Error: Must specify either an input file or --input string", err=True)
        sys.exit(1)

    # Read source code
    if input_string:
        source_code = input_string
        source_name = "<string>"
        if log_level in ["debug", "info"]:
            click.echo("Compiling from string input...")
    else:
        # Read from file
        try:
            source_code = input_file.read_text(encoding="utf-8")
            source_name = str(input_file.resolve())
            if log_level in ["debug", "info"]:
                click.echo(f"Compiling {input_file}...")
        except Exception as e:
            click.echo(f"Failed to read input file: {e}", err=True)
            sys.exit(1)

    # Compile
    success, result, diagnostic_messages = compile_dsl_source(
        source_code,
        source_name=source_name,
        program_name=name,
        optimize=not no_optimize,
        power_pole_type=power_poles,
        log_level=log_level,
        use_json=json,
        max_layout_retries=layout_retries,
    )
    verbose = log_level in ["debug", "info"]

    if not success:
        click.echo(f"Compilation failed: {result}", err=True)
        sys.exit(1)

    # Output blueprint
    if output:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(result, encoding="utf-8")
            if verbose:
                click.echo(f"Blueprint saved to {output}")
        except Exception as e:
            click.echo(f"Failed to write output file: {e}", err=True)
            sys.exit(1)
    else:
        click.echo(result)

    if verbose:
        msg_count = len(diagnostic_messages) if diagnostic_messages else 0
        msg = (
            f"Compilation completed with {msg_count} diagnostic(s)."
            if msg_count
            else "Compilation completed successfully."
        )
        click.echo(msg, err=True)


if __name__ == "__main__":
    main()
