# Common Module

## Purpose
Provides shared utilities, constants, and data structures used across all compilation stages.

## Files

### diagnostics.py
**Error and Warning Collection**
- `Diagnostic`: Single error/warning message
  - `severity`: ERROR, WARNING, INFO
  - `message`: Human-readable description
  - `stage`: Which compiler stage (parsing, semantic, lowering, etc.)
  - `location`: Source file, line, column
  - `node`: Associated AST node (optional)

- `ProgramDiagnostics`: Collection of all diagnostics
  - `error(message, stage, node)`: Add error
  - `warning(message, stage, node)`: Add warning
  - `has_errors()`: Check if any errors
  - `get_messages()`: Get formatted messages

**Usage**: Thread diagnostics through all compiler stages to collect errors/warnings.

### constants.py
**Compiler Configuration Constants**
- Entity spacing defaults
- Grid sizes
- Wire distance limits
- Signal type categories
- Operator definitions
- Default entity settings

Examples:
```python
ENTITY_SPACING = 2  # Tiles between entities
MAX_WIRE_DISTANCE = 7  # Max wire reach
ARITHMETIC_OPS = ['+', '-', '*', '/', '%']
COMPARISON_OPS = ['==', '!=', '<', '>', '<=', '>=']
```

### entity_data.py
**Factorio Entity Definitions**
- Entity prototypes and properties
- Valid entity types
- Property access validation
- Entity bounding boxes
- Connection point positions

Defines:
- Which entities can be placed
- Which properties are readable/writable
- Entity dimensions
- Circuit connection points

### signal_registry.py
**Signal Type Management**
- `SignalRegistry`: Central registry of all Factorio signal types
  - Items (iron-plate, copper-cable, etc.)
  - Fluids (water, crude-oil, etc.)
  - Virtual signals (signal-A through signal-Z, signal-0 through signal-9)
  - Special signals (signal-each, signal-anything, signal-everything)

Methods:
- `is_valid_signal(name)`: Check if signal type exists
- `get_signal_category(name)`: Get category (item/fluid/virtual)
- `get_signal_icon(name)`: Get icon info

### signals.py
**Signal Utility Functions**
- Signal type validation
- Signal name normalization
- Signal type inference helpers
- Common signal type constants

### source_location.py
**Source Code Location Tracking**
- `SourceLocation`: File, line, column information
  - Attached to AST nodes
  - Used in error messages
  - Tracks original source text

Enables precise error messages:
```
ERROR [semantic:example.facto:15:23]: Type mismatch
Signal b = "text" + 5;
                    ^
```

## Information Flow

```
All Compiler Stages
    ↓
├─ [diagnostics.py] - Collect errors/warnings
│   ↓
│   Error messages with locations
│
├─ [entity_data.py] - Validate entities
│   ↓
│   Entity configuration
│
├─ [signal_registry.py] - Validate signals
│   ↓
│   Signal type info
│
├─ [constants.py] - Configuration
│   ↓
│   Spacing, limits, defaults
│
└─ [source_location.py] - Track positions
    ↓
    Error location context
```

## Key Features

### Diagnostic Collection
Errors/warnings accumulated across stages:
```python
diagnostics = ProgramDiagnostics()

# Semantic stage adds error
analyzer = SemanticAnalyzer(diagnostics)
analyzer.visit(ast)

# Lowering stage adds warning
lowerer = ASTLowerer(analyzer, diagnostics)
ir_ops = lowerer.lower_program(ast)

# Check results
if diagnostics.has_errors():
    for msg in diagnostics.get_messages():
        print(msg)
```

### Signal Validation
All signal types validated against registry:
```python
from common.signal_registry import SignalRegistry

registry = SignalRegistry()

# Valid signals
assert registry.is_valid_signal("iron-plate")
assert registry.is_valid_signal("signal-A")

# Invalid signals
assert not registry.is_valid_signal("invalid-signal")
```

### Entity Validation
Entity properties validated:
```python
from common.entity_data import EntityData

entity_data = EntityData()

# Check if entity type exists
assert entity_data.is_valid_entity("steel-chest")

# Check if property is readable
assert entity_data.can_read_property("steel-chest", "output")

# Get bounding box
bbox = entity_data.get_bounding_box("arithmetic-combinator")
# bbox = (width, height) in tiles
```

### Source Location Formatting
Pretty error messages with context:
```python
from common.source_location import SourceLocation

loc = SourceLocation(
    filename="test.facto",
    line=10,
    column=15,
    source_line="Signal x = a + b;",
    context_lines=[...]
)

print(loc.format_error("Undefined variable 'a'"))
# Output:
# ERROR [test.facto:10:15]: Undefined variable 'a'
# Signal x = a + b;
#            ^
```

## Usage Example

```python
# Initialize diagnostics
from common.diagnostics import ProgramDiagnostics

diagnostics = ProgramDiagnostics()

# Use constants
from common.constants import ENTITY_SPACING, MAX_WIRE_DISTANCE

spacing = ENTITY_SPACING  # 2 tiles

# Validate signals
from common.signal_registry import SignalRegistry

registry = SignalRegistry()
if not registry.is_valid_signal("unknown-signal"):
    diagnostics.error("Unknown signal type", stage="semantic")

# Check results
if diagnostics.has_errors():
    print("Compilation failed:")
    for msg in diagnostics.get_messages():
        print(f"  {msg}")
```

## Constants Reference

### Operator Categories
- `ARITHMETIC_OPS`: +, -, *, /, %, ^
- `COMPARISON_OPS`: ==, !=, <, >, <=, >=
- `LOGICAL_OPS`: &&, ||, and, or
- `BITWISE_OPS`: AND, OR, XOR
- `SHIFT_OPS`: <<, >>

### Entity Types
- Combinators: arithmetic-combinator, decider-combinator, constant-combinator
- Power: small-electric-pole, medium-electric-pole, big-electric-pole, substation
- Entities: steel-chest, accumulator, small-lamp, etc.

### Signal Categories
- Items: Iron, copper, circuits, etc. (1000+ types)
- Fluids: Water, oil, gas, etc. (20+ types)
- Virtual: signal-A through signal-Z, signal-0 through signal-9
- Special: signal-each, signal-anything, signal-everything

## Testing

Tests verify:
- Diagnostic collection and formatting
- Signal registry completeness
- Entity data accuracy
- Constant correctness
- Source location formatting

## Performance

- Signal registry uses hash maps for O(1) lookup
- Entity data cached
- Diagnostics collected in list (O(1) append)
- Minimal overhead

## Future Extensions

- More detailed signal metadata (icons, descriptions)
- Extended entity property definitions
- Configurable constants
- Custom signal types
- Better error recovery suggestions
