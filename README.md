# Factompiler

A compiler for the **Facto** programming language, which compiles to Factorio blueprints.

## Quick Start

Compile a Facto source file to a Factorio blueprint:

```bash
# Compile and print blueprint to stdout
python compile.py tests/sample_programs/01_basic_arithmetic.facto

# Save blueprint to file
python compile.py tests/sample_programs/01_basic_arithmetic.facto -o output.blueprint

# Verbose output with diagnostics
python compile.py tests/sample_programs/01_basic_arithmetic.facto --verbose

# Custom blueprint name
python compile.py tests/sample_programs/01_basic_arithmetic.facto --name "My Circuit"
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
python compile.py tests/sample_programs/01_basic_arithmetic.facto

# Output: 0eNrFld9ugjAUxl/FnGRXww0QnJDsQudbLAspWmcTaFlbjIbw7jvgf4XFJkzDBdCeno...
```

### Save to File
```bash
# Create a blueprint file
python compile.py tests/sample_programs/05_entities.facto -o my_entities.blueprint
```

### Verbose Output
```bash
# See detailed diagnostic information
python compile.py tests/sample_programs/04_memory.facto --verbose
# Shows: warnings, compilation stages, output location
```

## Facto Language Features

The Facto language supports:

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
  --name TEXT        Blueprint name (default: derived from input filename)
  -v, --verbose      Show detailed diagnostic messages
  --help             Show this message and exit.
```

## Advanced Configuration

Programmatic consumers can customize the emitter when embedding the compiler. The `BlueprintEmitter` accepts `WireRelayOptions` to adjust automatic relay pole insertion (enable/disable, Euclidean vs. Manhattan planning, relay caps):

```python
from dsl_compiler.src.emission import BlueprintEmitter
from dsl_compiler.src.emission.emitter import WireRelayOptions

relay_options = WireRelayOptions(placement_strategy="manhattan", max_relays=4)
emitter = BlueprintEmitter(signal_type_map=signal_map, wire_relay_options=relay_options)
```
```

## Blueprint Output

The compiler generates Factorio-compatible blueprint strings that can be:
- Imported directly into Factorio
- Shared with other players
- Modified with blueprint editing tools

All blueprints are generated using the `factorio-draftsman` library for maximum compatibility.
