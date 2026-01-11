# General Agent Guidelines

# Compiler Architecture Overview

The compiler transforms Facto source code into Factorio blueprint strings through a 5-stage pipeline:

```
Source (.facto) → Parser → Semantic Analyzer → AST Lowerer → Layout Planner → Blueprint Emitter → Blueprint
```

## Directory Structure

```
dsl_compiler/
├── grammar/
│   └── facto.lark              # Lark grammar definition for Facto
├── src/
│   ├── ast/                    # AST node definitions
│   │   ├── base.py             # Base AST node class with source location
│   │   ├── expressions.py      # Expression nodes (binary ops, function calls, etc.)
│   │   ├── literals.py         # Literal nodes (numbers, signals, etc.)
│   │   └── statements.py       # Statement nodes (assignments, memory, etc.)
│   │
│   ├── parsing/                # Stage 1: Source → AST
│   │   ├── parser.py           # Main parser using Lark, entry: DSLParser.parse()
│   │   ├── preprocessor.py     # Import resolution and preprocessing
│   │   └── transformer.py      # Lark tree → AST node transformation
│   │
│   ├── semantic/               # Stage 2: Type checking & validation
│   │   ├── analyzer.py         # Main analyzer, entry: SemanticAnalyzer.visit()
│   │   ├── symbol_table.py     # Symbol table for variable tracking
│   │   └── type_system.py      # Type definitions and type checking logic
│   │
│   ├── lowering/               # Stage 3: AST → IR
│   │   ├── lowerer.py          # Main lowerer, entry: ASTLowerer.lower_program()
│   │   ├── expression_lowerer.py  # Expression to IR translation
│   │   ├── statement_lowerer.py   # Statement to IR translation
│   │   ├── memory_lowerer.py   # Memory operations to IR
│   │   └── constant_folder.py  # Compile-time constant evaluation
│   │
│   ├── ir/                     # Intermediate Representation
│   │   ├── nodes.py            # IR node definitions (IROperation, etc.)
│   │   ├── builder.py          # IRBuilder for constructing IR
│   │   └── optimizer.py        # CSE and constant propagation optimizers
│   │
│   ├── layout/                 # Stage 4: IR → Physical Layout
│   │   ├── planner.py          # Main planner, entry: LayoutPlanner.plan_layout()
│   │   ├── layout_plan.py      # LayoutPlan data structure
│   │   ├── entity_placer.py    # Entity placement algorithms
│   │   ├── connection_planner.py  # Wire connection planning
│   │   ├── memory_builder.py   # Memory cell layout construction
│   │   ├── wire_router.py      # Wire routing and MST optimization
│   │   ├── signal_analyzer.py  # Signal flow analysis
│   │   ├── signal_graph.py     # Signal dependency graph
│   │   ├── power_planner.py    # Power pole placement
│   │   └── tile_grid.py        # 2D grid management
│   │
│   ├── emission/               # Stage 5: Layout → Blueprint
│   │   ├── emitter.py          # Main emitter, entry: BlueprintEmitter.emit_from_plan()
│   │   └── entity_emitter.py   # Entity-specific emission logic
│   │
│   └── common/                 # Shared utilities
│       ├── constants.py        # Compiler configuration constants
│       ├── diagnostics.py      # Error/warning collection
│       ├── entity_data.py      # Factorio entity definitions
│       ├── signal_registry.py  # Signal type registry
│       ├── signals.py          # Signal utilities
│       └── source_location.py  # Source code location tracking
```

## Key Entry Points

- **compile.py** - CLI tool, orchestrates the full pipeline
- **DSLParser.parse(source, filename)** - Parse source to AST
- **SemanticAnalyzer.visit(ast)** - Type check and validate
- **ASTLowerer.lower_program(ast)** - Lower AST to IR
- **LayoutPlanner.plan_layout(ir_ops)** - Plan physical layout
- **BlueprintEmitter.emit_from_plan(plan)** - Emit blueprint

## Minimal Pipeline Example

```python
#!/usr/bin/env python3
"""Minimal example of running the compilation pipeline."""
from pathlib import Path
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.layout.planner import LayoutPlanner
from dsl_compiler.src.emission.emitter import BlueprintEmitter
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics

def compile_source(source: str) -> str:
    """Compile Facto source code to blueprint string.""""
    diagnostics = ProgramDiagnostics()
    
    # Stage 1: Parse
    parser = DSLParser()
    ast = parser.parse(source, "<string>")
    
    # Stage 2: Semantic Analysis
    analyzer = SemanticAnalyzer(diagnostics=diagnostics)
    analyzer.visit(ast)
    
    # Stage 3: Lower to IR
    lowerer = ASTLowerer(analyzer, diagnostics)
    ir_ops = lowerer.lower_program(ast)
    
    # Stage 4: Plan Layout
    planner = LayoutPlanner(
        lowerer.ir_builder.signal_type_map,
        diagnostics=diagnostics,
        signal_refs=lowerer.signal_refs,
        referenced_signal_names=lowerer.referenced_signal_names,
    )
    layout = planner.plan_layout(ir_ops)
    
    # Stage 5: Emit Blueprint
    emitter = BlueprintEmitter(diagnostics, lowerer.ir_builder.signal_type_map)
    blueprint = emitter.emit_from_plan(layout)
    
    return blueprint.to_string()

# Example usage:
# source = Path("example_programs/01_basic_arithmetic.facto").read_text()
# print(compile_source(source))
```

---

# running bash programs

When running programs, be aware that there is a venv in the factompiler directory that needs to be activated first to access the required dependencies. You can activate it by running once:

```bash
source /path/to/factompiler/venv/bin/activate
```

# draftsman

The compiler emits draftsman code for the designs. Draftsman is a Python library for creating DXF files. You can find more information about Draftsman and its usage in the [Draftsman documentation](https://draftsman.readthedocs.io/en/latest/).

Its source code is available in the directory ../draftsman relative to the factompiler directory.
But be aware that this is just for reference, and you should use the installed version in the venv.

# sample programs

There is a collection of sample programs available in the `example_programs` directory within the factompiler repository. These programs denote what we want to be able to compile. You can refer to these samples for guidance on how to structure your own programs. They should only be changed if we remove or add new features to the language.


# language spec

The `Language_SPEC.md` file in the factompiler repository contains the complete specification of the Facto programming language. This document outlines the syntax, semantics, and features of the language. It serves as a reference for understanding how to write programs in Facto and how the compiler should interpret them.

# coding guidelines

THIS PACKAGE IS NOT RELEASED YET. Ergo, you may never ever need to keep backward compability in mind and in the code. If I tell you to change something, change it and do not regard anything that was there before.

When writing code, please follow these guidelines:
- Write clear and concise code with appropriate comments.
- Follow the existing coding style and conventions used in the project.
- Write unit tests for new features or bug fixes to ensure code quality.
- Do not use defensive programming unless explicitly told to do so.
- Letting things fail/throw exceptions is often better than trying to handle every possible error case.

# Testing stuff:

- please make yourself aware of the compile script arguments:
Usage: compile.py [OPTIONS] INPUT_FILE

  Compile Facto source files to blueprint format.

Options:
  -o, --output PATH               Output file for the blueprint (default:
                                  stdout)
  --name TEXT                     Blueprint name (default: derived from input
                                  filename)
  --log-level [debug|info|warning|error]
                                  Set the logging level
  --no-optimize                   Disable IR optimizations
  --power-poles TEXT              Add power poles
                                  (small/medium/big/substation, defaults to
                                  medium if no value)
  --json                          Output blueprint in JSON format
  --help                          Show this message and exit.

- also, when you want to test things, do not execute python in a cli command with python -c! Instead, create a temporary script file and run that. This way, the venv dependencies will be correctly picked up and you can iterate on that file.
- all temporary scripts and files you create should be placed into the temp/ directory in the factompiler repo root. This directory is gitignored, so you don't have to worry about cleaning up after yourself.

Also when running pytest, always append -n auto to make use of the xdist extension and run the tests in parallel.