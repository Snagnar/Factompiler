# Factompiler

[![CI Pipeline](https://github.com/Snagnar/Factompiler/actions/workflows/ci.yml/badge.svg)](https://github.com/Snagnar/Factompiler/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Snagnar/Factompiler/branch/main/graph/badge.svg)](https://codecov.io/gh/Snagnar/Factompiler)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A compiler for the **Facto** programming language, which compiles to Factorio blueprints.

## Quick Start

Compile a Facto source file to a Factorio blueprint:

```bash
# Compile and print blueprint to stdout
python compile.py example_programs/01_basic_arithmetic.facto

# Save blueprint to file
python compile.py example_programs/01_basic_arithmetic.facto -o output.blueprint

# Verbose output with diagnostics
python compile.py example_programs/01_basic_arithmetic.facto --verbose

# Custom blueprint name
python compile.py example_programs/01_basic_arithmetic.facto --name "My Circuit"
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
python compile.py example_programs/01_basic_arithmetic.facto

# Output: 0eNrFld9ugjAUxl/FnGRXww0QnJDsQudbLAspWmcTaFlbjIbw7jvgf4XFJkzDBdCeno...
```

### Save to File
```bash
# Create a blueprint file
python compile.py example_programs/05_entities.facto -o my_entities.blueprint
```

### Verbose Output
```bash
# See detailed diagnostic information
python compile.py example_programs/04_memory.facto --verbose
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

See `example_programs/` for complete examples.

## Development & Testing

### Running Tests

The project uses pytest with separate markers for unit and end-to-end tests:

```bash
# Run all tests
pytest

# Run only unit tests (faster, excludes end-to-end integration tests)
pytest -m "not end2end"

# Run only end-to-end tests
pytest -m "end2end"

# Run with coverage report
pytest -m "not end2end" --cov=dsl_compiler --cov-report=html
```

### Code Quality

The project uses modern Python tooling:

- **Ruff**: Fast linting and code formatting
- **mypy**: Static type checking
- **pytest**: Test framework with coverage reporting

```bash
# Run linting
ruff check .

# Format code
ruff format .

# Type checking
mypy dsl_compiler/ compile.py --ignore-missing-imports
```

### CI Pipeline

The GitHub Actions CI pipeline runs on every push and pull request:

1. **Linting**: Ruff check, format validation, and mypy type checking
2. **Unit Tests**: Fast tests on Python 3.11 and 3.12 with coverage reporting
3. **End-to-End Tests**: Full compilation tests of all sample programs

Coverage reports are automatically uploaded to Codecov and available as artifacts in GitHub Actions.

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
