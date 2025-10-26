#!/usr/bin/env python3
"""
Factorio Circuit DSL Compiler

Usage:
    python compile.py input.fcdsl                    # Print blueprint to stdout
    python compile.py input.fcdsl -o output.blueprint # Save blueprint to file
    python compile.py input.fcdsl --strict           # Enable strict type checking
    python compile.py input.fcdsl --power-poles      # Add medium power poles
    python compile.py input.fcdsl --power-poles big  # Add big power poles
"""

import sys
import click
from pathlib import Path

# Add the compiler to the path
sys.path.insert(0, str(Path(__file__).parent))

from dsl_compiler.src.parsing import DSLParser
from dsl_compiler.src.semantic import analyze_program, SemanticAnalyzer
from dsl_compiler.src.lowering import lower_program
from dsl_compiler.src.emission import emit_blueprint_string
from dsl_compiler.src.ir import CSEOptimizer
import dsl_compiler.src.semantic as semantic_module


def validate_power_poles(ctx, param, value):
    """Validate power pole type."""
    if value is None:
        return None
    
    valid_types = ['small', 'medium', 'big', 'substation']
    if value.lower() not in valid_types:
        raise click.BadParameter(f"must be one of: {', '.join(valid_types)}")
    
    return value.lower()


def compile_dsl_file(
    input_path: Path,
    strict_types: bool = False,
    program_name: str = None,
    optimize: bool = True,
    explain: bool = False,
    power_pole_type: str | None = None,
) -> tuple[bool, str, list]:
    """
    Compile a DSL file to blueprint string.
    
    Returns:
        (success: bool, result: str, diagnostics: list)
    """
    if not input_path.exists():
        return False, f"Input file '{input_path}' does not exist", []
    
    # Read source file
    try:
        dsl_code = input_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Failed to read input file: {e}", []
    
    if program_name is None:
        program_name = input_path.stem.replace("_", " ").title()
    
    all_diagnostics = []
    previous_explain = semantic_module.EXPLAIN_MODE
    semantic_module.EXPLAIN_MODE = explain
    
    try:
        # Parse
        parser = DSLParser()
        program = parser.parse(dsl_code.strip(), str(input_path.resolve()))
        
        # Semantic analysis
        analyzer = SemanticAnalyzer(strict_types=strict_types)
        semantic_diagnostics = analyze_program(
            program, strict_types=strict_types, analyzer=analyzer, file_path=str(input_path)
        )
        
        if semantic_diagnostics.has_errors():
            all_diagnostics.extend(semantic_diagnostics.get_messages())
            return False, "Semantic analysis failed", all_diagnostics
        
        all_diagnostics.extend(semantic_diagnostics.get_messages())
        
        # IR generation
        ir_operations, lowering_diagnostics, signal_type_map = lower_program(program, analyzer)
        
        if lowering_diagnostics.has_errors():
            all_diagnostics.extend(lowering_diagnostics.get_messages())
            return False, "IR lowering failed", all_diagnostics
        
        all_diagnostics.extend(lowering_diagnostics.get_messages())
        
        # Optimize
        if optimize:
            ir_operations = CSEOptimizer().optimize(ir_operations)
        
        # Emit blueprint
        blueprint_string, emit_diagnostics = emit_blueprint_string(
            ir_operations,
            f"{program_name} Blueprint",
            signal_type_map,
            power_pole_type=power_pole_type,
        )
        
        if emit_diagnostics.has_errors():
            all_diagnostics.extend(emit_diagnostics.get_messages())
            return False, "Blueprint emission failed", all_diagnostics
        
        all_diagnostics.extend(emit_diagnostics.get_messages())
        
        # Validate blueprint
        if not blueprint_string or len(blueprint_string) < 10 or not blueprint_string.startswith("0eN"):
            return False, "Invalid blueprint string generated", all_diagnostics
        
        return True, blueprint_string, all_diagnostics
    
    except Exception as e:
        return False, f"Compilation failed: {e}", all_diagnostics
    finally:
        semantic_module.EXPLAIN_MODE = previous_explain


@click.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('-o', '--output', type=click.Path(path_type=Path),
              help='Output file for the blueprint (default: stdout)')
@click.option('--strict', is_flag=True,
              help='Enable strict type checking')
@click.option('--name', type=str,
              help='Blueprint name (default: derived from input filename)')
@click.option('-v', '--verbose', is_flag=True,
              help='Show detailed diagnostic messages')
@click.option('--no-optimize', is_flag=True,
              help='Disable IR optimizations')
@click.option('--explain', is_flag=True,
              help='Add extended explanations to diagnostics')
@click.option('--power-poles', is_flag=False, flag_value='medium', default=None,
              callback=validate_power_poles,
              help='Add power poles (small/medium/big/substation, defaults to medium if no value)')
def main(input_file, output, strict, name, verbose, no_optimize, explain, power_poles):
    """Compile Factorio Circuit DSL files to blueprint format."""
    
    if verbose:
        click.echo(f"Compiling {input_file}...")
        if strict:
            click.echo("Using strict type checking mode.")
    
    # Compile
    success, result, diagnostics = compile_dsl_file(
        input_file,
        strict_types=strict,
        program_name=name,
        optimize=not no_optimize,
        explain=explain,
        power_pole_type=power_poles,
    )
    
    # Print diagnostics if needed
    if diagnostics and (verbose or not success):
        click.echo("Diagnostics:", err=True)
        for msg in diagnostics:
            click.echo(f"  {msg}", err=True)
        click.echo("", err=True)
    
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
        msg = f"Compilation completed with {len(diagnostics)} diagnostic(s)." if diagnostics else "Compilation completed successfully."
        click.echo(msg, err=True)


if __name__ == "__main__":
    main()