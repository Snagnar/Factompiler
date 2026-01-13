# Emission Module

## Purpose
Converts `LayoutPlan` into Factorio blueprint strings. Final stage of compilation that generates importable blueprints.

## Files

### emitter.py
**Main Blueprint Emitter**
- `BlueprintEmitter`: Converts layout to blueprint
  - Entry: `emit_from_plan(layout_plan)` → Blueprint object
  - Coordinates entity emission
  - Generates blueprint metadata
  - Handles blueprint book creation
  - Encodes to blueprint string

**Blueprint Format**:
- JSON structure defined by Factorio
- Base64 encoded with zlib compression
- Version 0 prefix for blueprint strings

### entity_emitter.py
**Entity-Specific Emission**
- `EntityEmitter`: Handles individual entity types
  - Arithmetic combinators: operation, input/output config
  - Decider combinators: condition, output mode
  - Constant combinators: signal filters
  - Other entities: lamps, power poles, etc.
  - Wire connections: red/green wire endpoints

**Entity Configuration**:
- Control behavior (circuit/logistic conditions)
- Filters (constant combinator signals, signal selection)
- Colors (lamp RGB values)
- Other entity-specific settings

## Information Flow

```
LayoutPlan (from layout)
    ↓
[emitter.py] - Main emission coordination
    ↓
├─ Create Blueprint object (draftsman)
├─ Set metadata (label, icons)
│
├─ [entity_emitter.py] - Emit each entity
│   ├─ Add arithmetic combinators
│   ├─ Add decider combinators
│   ├─ Add constant combinators
│   ├─ Add memory cells (decider pairs)
│   ├─ Add placed entities (lamps, poles, etc.)
│   └─ Add wire connections
│
└─ Encode blueprint to string
    ↓
Blueprint string (ready for Factorio import)
```

## Key Features

### Draftsman Integration
Uses `draftsman` library for blueprint generation:
- Provides Factorio blueprint data structures
- Handles encoding/decoding
- Validates entity configurations
- Manages signal types and filters

### Entity Emission Rules

**Arithmetic Combinator**:
```json
{
  "name": "arithmetic-combinator",
  "position": {"x": 0, "y": 0},
  "control_behavior": {
    "arithmetic_conditions": {
      "operation": "*",
      "first_signal": {"type": "virtual", "name": "signal-A"},
      "second_constant": 255,
      "output_signal": {"type": "item", "name": "iron-plate"}
    }
  }
}
```

**Decider Combinator**:
```json
{
  "name": "decider-combinator",
  "control_behavior": {
    "decider_conditions": {
      "comparator": ">",
      "first_signal": {"type": "virtual", "name": "signal-A"},
      "constant": 10,
      "output_signal": {"type": "virtual", "name": "signal-X"},
      "copy_count_from_input": false
    }
  }
}
```

**Constant Combinator**:
```json
{
  "name": "constant-combinator",
  "control_behavior": {
    "filters": [
      {"signal": {"type": "item", "name": "iron-plate"}, "count": 100, "index": 1}
    ]
  }
}
```

### Wire Connections
Wires connect entity connection points:
```python
# Red wire from entity 1 output to entity 2 input
{
    "entity_id_1": 1,
    "entity_id_2": 2,
    "circuit_id_1": 1,  # Output side
    "circuit_id_2": 2,  # Input side
    "wire": "red"
}
```

### Signal Type Mapping
Converts internal signal types to Factorio format:
- "iron-plate" → `{"type": "item", "name": "iron-plate"}`
- "signal-A" → `{"type": "virtual", "name": "signal-A"}`
- "water" → `{"type": "fluid", "name": "water"}`

### Blueprint Metadata
- Blueprint label (from code or auto-generated)
- Icons (first 4 signal types used)
- Version information
- Description (optional)

## Usage Example

```python
from emission.emitter import BlueprintEmitter
from layout.planner import LayoutPlanner
# ... (parse, analyze, lower, plan layout)

# Emit blueprint
emitter = BlueprintEmitter(diagnostics, signal_type_map)
blueprint = emitter.emit_from_plan(layout_plan)

# Get blueprint string
bp_string = blueprint.to_string()

# Import into Factorio:
# 1. Copy bp_string to clipboard
# 2. In Factorio, press Ctrl+V
# 3. Blueprint appears in hand
```

## Blueprint String Format

Factorio blueprint strings:
```
0eNqrVkpJTc7PKylWsoquViotTi0qyi8tUYrJLC7OrI/JTM...
^ 
|__ Version 0 marker
    Followed by base64(zlib(json_data))
```

Decoding process:
1. Remove version prefix ("0")
2. Base64 decode
3. Zlib decompress
4. Parse JSON

## Testing

Tests verify:
- Blueprint string generation
- Entity configuration correctness
- Wire connection validity
- Signal type mapping
- Metadata generation
- Round-trip (emit → import → emit)

## Draftsman API

Key draftsman classes used:
- `Blueprint`: Main blueprint container
- `ArithmeticCombinator`: Arithmetic combinator entity
- `DeciderCombinator`: Decider combinator entity
- `ConstantCombinator`: Constant combinator entity
- `Association`: Wire connection between entities

## Error Handling

Emission can fail with:
- Invalid entity configurations
- Unsupported signal types
- Invalid wire connections
- Draftsman validation errors

Errors collected in `diagnostics`.

## Performance

- Emission typically <50ms for moderate blueprints
- Most time spent in:
  - JSON serialization
  - Zlib compression
  - Base64 encoding

## Future Extensions

- Blueprint book support (multiple blueprints)
- Blueprint snapping and alignment
- Entity settings (recipes, modules)
- More entity types (trains, logistics)
- Blueprint optimization (layout minimization)
