#!/usr/bin/env python3
"""
Factorio Circuit DSL Compiler

Main compilation script that takes DSL source files and produces Factorio blueprints.

Usage:
    python compile.py input.fcdsl                    # Print blueprint to stdout
    python compile.py input.fcdsl -o output.blueprint # Save blueprint to file
    python compile.py input.fcdsl --strict           # Enable strict type checking
"""

import sys
import click
from pathlib import Path

# Add the compiler to the path
sys.path.insert(0, str(Path(__file__).parent))

from dsl_compiler.src.parser import DSLParser
from dsl_compiler.src.semantic import analyze_program, SemanticAnalyzer
from dsl_compiler.src.lowerer import lower_program
from dsl_compiler.src.emit import emit_blueprint_string


def compile_dsl_file(
    input_path: Path, 
    strict_types: bool = False, 
    program_name: str = None
) -> tuple[bool, str, list]:
    """
    Compile a DSL file to blueprint string.
    
    Returns:
        (success: bool, result: str, diagnostics: list)
    """
    if not input_path.exists():
        return False, f"Input file '{input_path}' does not exist", []
    
    # Read the DSL source
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            dsl_code = f.read()
    except Exception as e:
        return False, f"Failed to read input file: {e}", []
    
    if program_name is None:
        program_name = input_path.stem.replace('_', ' ').title()
    
    all_diagnostics = []
    
    try:
        # Parse
        parser = DSLParser()
        program = parser.parse(dsl_code.strip(), str(input_path.resolve()))
        
        # Semantic analysis
        analyzer = SemanticAnalyzer(strict_types=strict_types)
        semantic_diagnostics = analyze_program(
            program, 
            strict_types=strict_types, 
            analyzer=analyzer,
            file_path=str(input_path)
        )
        
        if semantic_diagnostics.has_errors():
            error_msgs = semantic_diagnostics.get_messages()
            all_diagnostics.extend(error_msgs)
            return False, f"Semantic analysis failed:\n" + "\n".join(error_msgs), all_diagnostics
        
        # Collect warnings from semantic analysis
        if semantic_diagnostics.warning_count > 0:
            all_diagnostics.extend(semantic_diagnostics.get_messages())
        
        # IR generation
        ir_operations, lowering_diagnostics, signal_type_map = lower_program(
            program, analyzer
        )
        
        if lowering_diagnostics.has_errors():
            error_msgs = lowering_diagnostics.get_messages()
            all_diagnostics.extend(error_msgs)
            return False, f"IR lowering failed:\n" + "\n".join(error_msgs), all_diagnostics
        
        # Collect warnings from lowering
        if lowering_diagnostics.warning_count > 0:
            all_diagnostics.extend(lowering_diagnostics.get_messages())
        
        # Blueprint generation
        blueprint_string, emit_diagnostics = emit_blueprint_string(
            ir_operations, f"{program_name} Blueprint", signal_type_map
        )
        
        if emit_diagnostics.has_errors():
            error_msgs = emit_diagnostics.get_messages()
            all_diagnostics.extend(error_msgs)
            return False, f"Blueprint emission failed:\n" + "\n".join(error_msgs), all_diagnostics
        
        # Collect warnings from emission
        if emit_diagnostics.warning_count > 0:
            all_diagnostics.extend(emit_diagnostics.get_messages())
        
        # Validate blueprint string
        if not blueprint_string or len(blueprint_string) < 10 or not blueprint_string.startswith("0eN"):
            return False, "Invalid blueprint string generated", all_diagnostics
        
        return True, blueprint_string, all_diagnostics
        
    except Exception as e:
        return False, f"Compilation failed with unexpected error: {e}", all_diagnostics


@click.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option(
    '-o', '--output', 
    type=click.Path(path_type=Path),
    help='Output file for the blueprint (default: print to stdout)'
)
@click.option(
    '--strict', 
    is_flag=True,
    help='Enable strict type checking (warnings become errors)'
)
@click.option(
    '--name',
    type=str,
    help='Blueprint name (default: derived from input filename)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Show detailed diagnostic messages'
)
def main(input_file: Path, output: Path, strict: bool, name: str, verbose: bool):
    """
    Compile Factorio Circuit DSL files to blueprint format.
    
    INPUT_FILE: Path to the .fcdsl source file to compile
    """
    if verbose:
        click.echo(f"Compiling {input_file}...")
        if strict:
            click.echo("Using strict type checking mode.")
    
    # Compile the file
    success, result, diagnostics = compile_dsl_file(
        input_file, 
        strict_types=strict, 
        program_name=name
    )
    
    # Print diagnostics if verbose or if there are warnings
    if diagnostics and (verbose or not success):
        click.echo("Diagnostics:", err=True)
        for msg in diagnostics:
            click.echo(f"  {msg}", err=True)
        if diagnostics:
            click.echo("", err=True)  # Empty line after diagnostics
    
    if not success:
        click.echo(f"Compilation failed: {result}", err=True)
        sys.exit(1)
    
    # Output the blueprint
    if output:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, 'w', encoding='utf-8') as f:
                f.write(result)
            if verbose:
                click.echo(f"Blueprint saved to {output}")
        except Exception as e:
            click.echo(f"Failed to write output file: {e}", err=True)
            sys.exit(1)
    else:
        # Print to stdout
        click.echo(result)
    
    if verbose and diagnostics:
        click.echo(f"Compilation completed with {len(diagnostics)} diagnostic message(s).", err=True)
    elif verbose:
        click.echo("Compilation completed successfully.", err=True)


if __name__ == "__main__":
    main()