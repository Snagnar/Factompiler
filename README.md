# Factompiler

A compiled language for Factorio Blueprints.

## Quick Start

Compile a DSL file to a Factorio blueprint:

```bash
# Compile and print blueprint to stdout
python compile.py tests/sample_programs/01_basic_arithmetic.fcdsl

# Save blueprint to file
python compile.py tests/sample_programs/01_basic_arithmetic.fcdsl -o output.blueprint

# Use strict type checking
python compile.py tests/sample_programs/01_basic_arithmetic.fcdsl --strict

# Verbose output with diagnostics
python compile.py tests/sample_programs/01_basic_arithmetic.fcdsl --verbose

# Custom blueprint name
python compile.py tests/sample_programs/01_basic_arithmetic.fcdsl --name "My Circuit"
```

## Installation

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Ensure `factorio-draftsman` is available (included in workspace)

## Usage Examples

### Basic Compilation
```bash
# Compile a simple arithmetic circuit
python compile.py tests/sample_programs/01_basic_arithmetic.fcdsl

# Output: 0eNrFld9ugjAUxl/FnGRXww0QnJDsQudbLAspWmcTaFlbjIbw7jvgf4XFJkzDBdCeno...
```

### Save to File
```bash
# Create a blueprint file
python compile.py tests/sample_programs/05_entities.fcdsl -o my_entities.blueprint
```

### Strict Mode
```bash
# Enable strict type checking (warnings become errors)
python compile.py tests/sample_programs/01_basic_arithmetic.fcdsl --strict
# Output: Compilation failed due to type mismatches
```

### Verbose Output
```bash
# See detailed diagnostic information
python compile.py tests/sample_programs/04_memory.fcdsl --verbose
# Shows: warnings, compilation stages, output location
```

## DSL Language Features

The Factorio Circuit DSL supports:

- **Signals**: `Signal a = ("iron-plate", 0);`
- **Arithmetic**: `Signal result = a + b * 2;`
- **Memory**: `Memory counter = memory(0);`
- **Entities**: `Entity lamp = place("small-lamp", 10, 5);`
- **Functions**: Reusable circuit components
- **Type Safety**: Optional strict type checking

See `tests/sample_programs/` for complete examples.

## Command Line Options

```
Usage: compile.py [OPTIONS] INPUT_FILE

Options:
  -o, --output PATH  Output file for the blueprint (default: print to stdout)
  --strict           Enable strict type checking (warnings become errors)
  --name TEXT        Blueprint name (default: derived from input filename)
  -v, --verbose      Show detailed diagnostic messages
  --help             Show this message and exit.
```

## Blueprint Output

The compiler generates Factorio-compatible blueprint strings that can be:
- Imported directly into Factorio
- Shared with other players
- Modified with blueprint editing tools

All blueprints are generated using the `factorio-draftsman` library for maximum compatibility.
