# Entities and Control

Entities are the physical objects in Factorio – lamps, inserters, train stops, assemblers, and more. Factompiler lets you place entities in your blueprint and control them with circuit signals.

## Placing Entities

Use the `place()` function to add an entity to your blueprint:

```fcdsl
Entity name = place("prototype", x, y);
Entity name = place("prototype", x, y, {property: value, ...});
```

Parameters:
- **prototype** – The Factorio entity name (e.g., `"small-lamp"`, `"inserter"`)
- **x, y** – Position coordinates
- **properties** (optional) – Dictionary of initial settings

### Basic Placement

```fcdsl
Entity lamp = place("small-lamp", 0, 0);
Entity inserter = place("inserter", 5, 0);
Entity belt = place("transport-belt", 10, 0);
```

> **[IMAGE PLACEHOLDER]**: Screenshot showing a lamp, inserter, and belt placed in a row.

### Placement with Properties

Configure entity settings at placement time:

```fcdsl
Entity colored_lamp = place("small-lamp", 0, 0, {
    use_colors: 1,
    always_on: 1,
    color_mode: 1
});

Entity named_stop = place("train-stop", 10, 0, {
    station: "Iron Pickup"
});

Entity fast_inserter = place("fast-inserter", 5, 0, {
    direction: 4  # East
});
```

## Entity Coordinates

### Fixed Positions

When you use integer constants, the entity is placed at exactly those coordinates:

```fcdsl
Entity lamp1 = place("small-lamp", 0, 0);   # At origin
Entity lamp2 = place("small-lamp", 5, 0);   # 5 tiles right
Entity lamp3 = place("small-lamp", 0, 5);   # 5 tiles down
```

Factorio uses a coordinate system where:
- **X** increases to the right
- **Y** increases downward

### Layout-Optimized Positions

For functional circuits (not visual displays), you can let the compiler optimize positions:

```fcdsl
Signal x_pos = some_signal;
Entity lamp = place("small-lamp", x_pos, 0);  # Position chosen by layout engine
```

When coordinates are signals or expressions (not constants), the layout engine places entities for optimal wire connections.

**Tip:** For most circuits, using fixed positions (0, 0), (0, 2), etc. works well. The compiler handles wire routing automatically.

## Controlling Entities with Circuits

### The `enable` Property

Most entities have an `enable` property that turns them on or off based on a circuit signal:

```fcdsl
Entity lamp = place("small-lamp", 0, 0);
lamp.enable = count > 50;  # Lamp turns on when count exceeds 50
```

This is the most common way to control entities.

### Property Assignment Syntax

Use dot notation to set properties:

```fcdsl
entity.property = value;
```

The `value` can be:
- A signal expression: `count > 50`
- A constant: `1`
- Another signal: `enable_signal`

## Common Entity Types

### Lamps

Lamps are the simplest way to visualize circuit output.

**Basic lamp:**
```fcdsl
Entity lamp = place("small-lamp", 0, 0);
lamp.enable = signal > 0;
```

**Colored lamp (RGB):**
```fcdsl
Entity rgb_lamp = place("small-lamp", 0, 0, {
    use_colors: 1,      # Enable color mode
    always_on: 1,       # Stay on regardless of enable
    color_mode: 1       # Use RGB component signals
});

# Control color with signals (0-255 each)
rgb_lamp.r = red_value;
rgb_lamp.g = green_value;
rgb_lamp.b = blue_value;
```

> **[IMAGE PLACEHOLDER]**: Screenshot showing three colored lamps – one red, one green, one blue.

**Color mode values:**
| Value | Mode | Description |
|-------|------|-------------|
| 0 | COLOR_MAPPING | Uses signal color palette |
| 1 | COMPONENTS | Uses r, g, b signal inputs |
| 2 | PACKED_RGB | Uses single 0xRRGGBB value |

### Inserters

Control inserter operation with circuit conditions:

```fcdsl
Entity inserter = place("inserter", 0, 0);
inserter.enable = chest_count < 100;  # Only insert when chest has < 100 items
```

**Inserter variants:**
- `"burner-inserter"` – Uses fuel
- `"inserter"` – Basic electric
- `"long-handed-inserter"` – Longer reach
- `"fast-inserter"` – Faster operation
- `"bulk-inserter"` – Moves multiple items
- `"stack-inserter"` – Maximum throughput

**Direction property:**
```fcdsl
Entity inserter = place("inserter", 0, 0, {direction: 4});  # Facing East
```

Direction values:
| Value | Direction |
|-------|-----------|
| 0 | North |
| 4 | East |
| 8 | South |
| 12 | West |

### Transport Belts

Control belt movement:

```fcdsl
Entity belt = place("transport-belt", 0, 0);
belt.enable = should_run > 0;
```

**Belt variants:**
- `"transport-belt"` – Basic (yellow)
- `"fast-transport-belt"` – Medium (red)
- `"express-transport-belt"` – Fast (blue)

### Train Stops

Train stops have many circuit-controllable properties:

```fcdsl
Entity station = place("train-stop", 0, 0, {
    station: "Iron Pickup",     # Station name
    color: {r: 255, g: 0, b: 0} # Display color
});

station.enable = iron_available > 0;    # Enable/disable station
station.send_to_train = 1;               # Send signals to stopped train
station.read_from_train = 1;             # Read train contents
```

### Assembling Machines

Control production:

```fcdsl
Entity assembler = place("assembling-machine-1", 0, 0);
assembler.enable = has_materials > 0;
```

**Variants:**
- `"assembling-machine-1"` – Slow, 2 ingredient slots
- `"assembling-machine-2"` – Medium
- `"assembling-machine-3"` – Fast

### Power Poles

Power poles are circuit connection points:

```fcdsl
Entity pole = place("medium-electric-pole", 0, 0);
```

The compiler automatically adds relay poles when wire distances exceed 9 tiles, but you can place them explicitly for specific layouts.

**Pole variants:**
- `"small-electric-pole"` – Short range, cheap
- `"medium-electric-pole"` – Standard range
- `"big-electric-pole"` – Long range, large area
- `"substation"` – Maximum coverage

## Reading Entity Properties

Some entity properties can be read back as signals:

```fcdsl
Entity lamp = place("small-lamp", 0, 0);
lamp.enable = some_condition;

Signal lamp_status = lamp.enable;  # Read back the enable state
```

This is useful for creating feedback loops or monitoring entity state.

## Entity Property Inlining

When you assign a simple comparison to `enable`, the compiler **inlines** it into the entity's circuit condition:

```fcdsl
Entity lamp = place("small-lamp", 0, 0);
lamp.enable = count > 10;  # No extra combinator! Uses lamp's built-in condition.
```

Instead of creating a decider combinator that outputs 1/0 and connecting it to the lamp, the compiler configures the lamp's internal condition to `count > 10`. This saves entities and reduces tick delay.

**This optimization applies when:**
- The property is `enable`
- The value is a simple comparison (`signal OP constant` or `signal OP signal`)
- The comparison result isn't used elsewhere

If the comparison is complex or reused, a decider combinator is created:

```fcdsl
Signal is_high = count > 10;
lamp1.enable = is_high;  # Uses the signal
lamp2.enable = is_high;  # Reuses the same signal
# Here a decider is created because is_high is used by multiple entities
```

## Working with Multiple Entities

### Placement Loops

Use `for` loops to place multiple entities efficiently:

```fcdsl
# Place 8 lamps in a row
for i in 0..8 {
    Entity lamp = place("small-lamp", i * 2, 0);
    lamp.enable = count > 0;
}

# Place lamps at specific positions
for pos in [0, 5, 12, 20] {
    Entity lamp = place("small-lamp", pos, 0);
    lamp.enable = active > 0;
}
```

For loops are unrolled at compile time, so each iteration creates independent entities.

### Entity Factory Functions

You can also use functions to encapsulate entity creation:

```fcdsl
func place_lamp(int x, int y) {
    Entity lamp = place("small-lamp", x, y);
    return lamp;
}

Entity lamp0 = place_lamp(0, 0);
Entity lamp1 = place_lamp(2, 0);
Entity lamp2 = place_lamp(4, 0);
Entity lamp3 = place_lamp(6, 0);
```

### Coordinated Control

Control multiple entities with the same logic:

```fcdsl
Memory counter: "signal-A";
counter.write((counter.read() + 1) % 20);

Signal phase = counter.read();

Entity lamp1 = place("small-lamp", 0, 0);
Entity lamp2 = place("small-lamp", 2, 0);
Entity lamp3 = place("small-lamp", 4, 0);
Entity lamp4 = place("small-lamp", 6, 0);

# Sequential activation (knight rider effect)
lamp1.enable = phase < 5;
lamp2.enable = (phase >= 5) && (phase < 10);
lamp3.enable = (phase >= 10) && (phase < 15);
lamp4.enable = phase >= 15;
```

> **[IMAGE PLACEHOLDER]**: Animated GIF or series of screenshots showing the knight rider effect on 4 lamps.

### Chaser Pattern

```fcdsl
Memory counter: "signal-A";
counter.write((counter.read() + 1) % 8);
Signal pos = counter.read();

# Create 8 lamps in a row using a for loop
for i in 0..8 {
    Entity lamp = place("small-lamp", i * 2, 0);
    lamp.enable = pos == i;
}
```

## Complete Example: RGB Color Cycling Lamp

This example creates a lamp that smoothly cycles through all colors:

```fcdsl
# HSV color cycling (simplified)
Memory hue: "signal-H";
hue.write((hue.read() + 1) % 1530);  # 1530 = 6 * 255

Signal h = hue.read();

# Calculate sector (0-5) and position within sector
Signal sector = h / 255;
Signal position = h % 255;
Signal rising = position;
Signal falling = 255 - position;

# Sector detection
Signal in_s0 = sector == 0;
Signal in_s1 = sector == 1;
Signal in_s2 = sector == 2;
Signal in_s3 = sector == 3;
Signal in_s4 = sector == 4;
Signal in_s5 = sector == 5;

# Calculate RGB
Signal red_val = (in_s0 * 255) + (in_s1 * falling) + (in_s4 * rising) + (in_s5 * 255);
Signal green_val = (in_s0 * rising) + (in_s1 * 255) + (in_s2 * 255) + (in_s3 * falling);
Signal blue_val = (in_s2 * rising) + (in_s3 * 255) + (in_s4 * 255) + (in_s5 * falling);

# Place the lamp
Entity lamp = place("small-lamp", 0, 0, {
    use_colors: 1,
    always_on: 1,
    color_mode: 1
});

lamp.r = red_val;
lamp.g = green_val;
lamp.b = blue_val;
```

> **[IMAGE PLACEHOLDER]**: Screenshot or animated GIF of the color cycling lamp showing different colors.

## Complete Example: Production Monitor

A practical circuit that monitors production and controls machines:

```fcdsl
# Input signals (wire from storage chests)
Signal iron_stock = ("iron-plate", 0);
Signal copper_stock = ("copper-plate", 0);
Signal circuit_stock = ("electronic-circuit", 0);

# Thresholds
int low_iron = 100;
int low_copper = 100;
int target_circuits = 500;

# Status calculations
Signal need_iron = iron_stock < low_iron;
Signal need_copper = copper_stock < low_copper;
Signal circuits_low = circuit_stock < target_circuits;

# Control assembler: only make circuits if we have materials and need them
Signal should_produce = (!need_iron) && (!need_copper) && circuits_low;

Entity assembler = place("assembling-machine-1", 0, 0);
assembler.enable = should_produce;

# Status lamps
Entity lamp_iron = place("small-lamp", 5, 0);
Entity lamp_copper = place("small-lamp", 7, 0);
Entity lamp_running = place("small-lamp", 9, 0);

lamp_iron.enable = need_iron;       # On = need more iron
lamp_copper.enable = need_copper;   # On = need more copper
lamp_running.enable = should_produce; # On = producing
```

## Entity Reference

For a complete list of all entities, their prototypes, and properties, see the **[Entity Reference](../ENTITY_REFERENCE_DSL.md)**.

This reference includes:
- All entity prototype names
- Available properties for each entity type
- Enum values for direction, mode settings, etc.
- Circuit signal I/O capabilities
- Example DSL code for each entity

## Tips and Best Practices

### Use Meaningful Positions

For displays (lamp arrays, status indicators), use evenly spaced positions:

```fcdsl
Entity lamp1 = place("small-lamp", 0, 0);
Entity lamp2 = place("small-lamp", 2, 0);  # 2-tile spacing
Entity lamp3 = place("small-lamp", 4, 0);
```

### Group Related Entities

Place related entities near each other:

```fcdsl
# Status display at y=0
Entity status_lamp = place("small-lamp", 0, 0);

# Control logic at y=5
Entity inserter = place("inserter", 0, 5);
Entity assembler = place("assembling-machine-1", 3, 5);
```

### Use Inline Conditions

Let the compiler optimize simple conditions:

```fcdsl
# Good - inlines into lamp's condition
lamp.enable = count > 100;

# Less optimal - creates extra combinator
Signal enable = count > 100;
lamp.enable = enable;
```

### Name Your Entities Meaningfully

```fcdsl
Entity output_indicator = place("small-lamp", 0, 0);
Entity material_warning = place("small-lamp", 2, 0);
Entity production_status = place("small-lamp", 4, 0);
```

---

## Summary

- Place entities with `place("prototype", x, y, {properties})`
- Control entities with `entity.property = signal_expression`
- Most entities support `enable` for on/off control
- Lamps support RGB color control with `use_colors` mode
- The compiler inlines simple comparisons into entity conditions
- See the **[Entity Reference](../ENTITY_REFERENCE_DSL.md)** for all entity types

---

**← [Memory](04_memory.md)** | **[Functions →](06_functions.md)**
