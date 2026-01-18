# Entities and Control

Entities are physical objects in Factorio — lamps, inserters, train stops, assemblers, and more. Facto lets you place entities in blueprints and control them with circuit signals.

---

## Placing Entities

Use `place()` to add an entity to your blueprint:

```facto
Entity name = place("prototype", x, y);
Entity name = place("prototype", x, y, {property: value, ...});
```

| Parameter | Description |
|-----------|-------------|
| `prototype` | Factorio entity name (e.g., `"small-lamp"`, `"inserter"`) |
| `x, y` | Position coordinates |
| `properties` | Optional dictionary of initial settings |

### Basic Placement

```facto
Entity lamp = place("small-lamp", 0, 0);
Entity inserter = place("inserter", 5, 0);
Entity belt = place("transport-belt", 10, 0);
```

### Placement with Properties

```facto
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

---

## Entity Coordinates

Factorio's coordinate system:
- **X** increases to the right
- **Y** increases downward

### Fixed Positions

Use integer constants for precise placement:

```facto
Entity lamp1 = place("small-lamp", 0, 0);   # At origin
Entity lamp2 = place("small-lamp", 5, 0);   # 5 tiles right
Entity lamp3 = place("small-lamp", 0, 5);   # 5 tiles down
```

### Layout-Optimized Positions

For functional circuits, let the compiler optimize:

```facto
Signal x_pos = some_signal;
Entity lamp = place("small-lamp", x_pos, 0);  # Compiler chooses position
```

When coordinates are signals or expressions, the layout engine places entities for optimal wire connections.

---

## Controlling Entities

### The `enable` Property

Most entities have `enable` for on/off control:

```facto
Entity lamp = place("small-lamp", 0, 0);
lamp.enable = count > 50;  # Lamp on when count exceeds 50
```

### Property Assignment

Use dot notation:

```facto
entity.property = value;
```

The value can be:
- A signal expression: `count > 50`
- A constant: `1`
- Another signal: `enable_signal`

---

## Reading Entity Properties

### Enable State

Read back what you set:

```facto
Entity lamp = place("small-lamp", 0, 0);
lamp.enable = some_condition;

Signal lamp_status = lamp.enable;  # Read the enable state
```

### Entity Contents: `.output`

Entities like chests and tanks output their contents as circuit signals. Access with `.output`:

```facto
Entity chest = place("steel-chest", 0, 0, {read_contents: 1});
Bundle contents = chest.output;  # All item signals from chest
```

The `.output` property returns a **Bundle** — all signals the entity outputs. For chests, this is item counts. Signal types are dynamic, determined at runtime.

**Common uses:**
- Monitoring inventory levels
- Balancing item distribution
- Controlling production based on storage

### Balanced Loader Example

```facto
Entity c1 = place("steel-chest", 0, 0, {read_contents: 1});
Entity c2 = place("steel-chest", 1, 0, {read_contents: 1});
Entity c3 = place("steel-chest", 2, 0, {read_contents: 1});

# Sum contents and compute negative average
Bundle total = {c1.output, c2.output, c3.output};
Bundle neg_avg = total / -3;

# Each inserter compares its chest to average
Entity i1 = place("fast-inserter", 0, 1);
Bundle in1 = {neg_avg, c1.output};
i1.enable = in1 < 0;  # Enable if below average
```

---

## Entity Property Inlining

When you assign a simple comparison to `enable`, the compiler **inlines** it:

```facto
Entity lamp = place("small-lamp", 0, 0);
lamp.enable = count > 10;  # Uses lamp's built-in condition — no extra combinator!
```

Instead of creating a decider combinator, the compiler configures the lamp's internal circuit condition. This saves entities and reduces tick delay.

**Inlining applies when:**
- Property is `enable`
- Value is simple comparison (`signal OP constant` or `signal OP signal`)
- Result isn't used elsewhere

If comparison is complex or reused, a decider is created:

```facto
Signal is_high = count > 10;
lamp1.enable = is_high;  # Shared signal
lamp2.enable = is_high;
# A decider is created because is_high is used by multiple entities
```

---

## Common Entity Types

### Lamps

**Basic lamp:**
```facto
Entity lamp = place("small-lamp", 0, 0);
lamp.enable = signal > 0;
```

**Colored lamp (RGB):**
```facto
Entity rgb_lamp = place("small-lamp", 0, 0, {
    use_colors: 1,      # Enable color mode
    always_on: 1,       # Stay lit regardless of enable
    color_mode: 1       # Use RGB component signals
});

rgb_lamp.r = red_value;     # 0-255
rgb_lamp.g = green_value;
rgb_lamp.b = blue_value;
```

**Color mode values:**

| Value | Mode | Description |
|-------|------|-------------|
| 0 | COLOR_MAPPING | Uses signal color palette |
| 1 | COMPONENTS | Uses r, g, b signal inputs |
| 2 | PACKED_RGB | Uses single 0xRRGGBB value |

### Inserters

```facto
Entity inserter = place("inserter", 0, 0);
inserter.enable = chest_count < 100;
```

**Variants:**
- `"burner-inserter"` — Uses fuel
- `"inserter"` — Basic electric
- `"long-handed-inserter"` — Longer reach
- `"fast-inserter"` — Faster operation
- `"bulk-inserter"` — Multiple items
- `"stack-inserter"` — Maximum throughput

**Direction values:**

| Value | Direction |
|-------|-----------|
| 0 | North |
| 4 | East |
| 8 | South |
| 12 | West |

### Transport Belts

```facto
Entity belt = place("transport-belt", 0, 0);
belt.enable = should_run > 0;
```

**Variants:** `"transport-belt"`, `"fast-transport-belt"`, `"express-transport-belt"`

### Train Stops

```facto
Entity station = place("train-stop", 0, 0, {
    station: "Iron Pickup",
    color: {r: 255, g: 0, b: 0}
});

station.enable = iron_available > 0;
station.send_to_train = 1;
station.read_from_train = 1;
```

### Assembling Machines

```facto
Entity assembler = place("assembling-machine-1", 0, 0);
assembler.enable = has_materials > 0;
```

**Variants:** `"assembling-machine-1"`, `"assembling-machine-2"`, `"assembling-machine-3"`

### Power Poles

```facto
Entity pole = place("medium-electric-pole", 0, 0);
```

The compiler auto-adds relay poles when wire distances exceed 9 tiles.

**Variants:** `"small-electric-pole"`, `"medium-electric-pole"`, `"big-electric-pole"`, `"substation"`

---

## Working with Multiple Entities

### Placement Loops

```facto
# 8 lamps in a row
for i in 0..8 {
    Entity lamp = place("small-lamp", i * 2, 0);
    lamp.enable = count > 0;
}

# Specific positions
for pos in [0, 5, 12, 20] {
    Entity lamp = place("small-lamp", pos, 0);
    lamp.enable = active > 0;
}
```

Loops unroll at compile time — each iteration creates independent entities.

### Entity Factory Functions

```facto
func place_lamp(int x, int y) {
    Entity lamp = place("small-lamp", x, y);
    return lamp;
}

Entity lamp0 = place_lamp(0, 0);
Entity lamp1 = place_lamp(2, 0);
```

### Coordinated Control

```facto
Memory counter: "signal-A";
counter.write((counter.read() + 1) % 20);
Signal phase = counter.read();

Entity lamp1 = place("small-lamp", 0, 0);
Entity lamp2 = place("small-lamp", 2, 0);
Entity lamp3 = place("small-lamp", 4, 0);
Entity lamp4 = place("small-lamp", 6, 0);

# Sequential activation using conditional values
lamp1.enable = phase < 5;
lamp2.enable = (phase >= 5) && (phase < 10);
lamp3.enable = (phase >= 10) && (phase < 15);
lamp4.enable = phase >= 15;
```

### Chaser Pattern

```facto
Memory counter: "signal-A";
counter.write((counter.read() + 1) % 8);
Signal pos = counter.read();

for i in 0..8 {
    Entity lamp = place("small-lamp", i * 2, 0);
    lamp.enable = pos == i;
}
```

---

## Complete Example: RGB Color Cycling

A lamp that smoothly cycles through all colors using conditional values:

```facto
Memory hue: "signal-H";
hue.write((hue.read() + 1) % 1530);  # 1530 = 6 * 255

Signal h = hue.read();
Signal sector = h / 255;
Signal position = h % 255;
Signal rising = position;
Signal falling = 255 - position;

# Calculate RGB using conditional values
Signal red_val = 
    (sector == 0) : 255 +
    (sector == 1) : falling +
    (sector == 4) : rising +
    (sector == 5) : 255;

Signal green_val = 
    (sector == 0) : rising +
    (sector == 1) : 255 +
    (sector == 2) : 255 +
    (sector == 3) : falling;

Signal blue_val = 
    (sector == 2) : rising +
    (sector == 3) : 255 +
    (sector == 4) : 255 +
    (sector == 5) : falling;

Entity lamp = place("small-lamp", 0, 0, {
    use_colors: 1,
    always_on: 1,
    color_mode: 1
});

lamp.r = red_val;
lamp.g = green_val;
lamp.b = blue_val;
```

---

## Complete Example: Production Monitor

Monitor production and control machines:

```facto
# Inputs (wire from storage chests)
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

# Control: produce if we have materials and need circuits
Signal should_produce = (!need_iron) && (!need_copper) && circuits_low;

Entity assembler = place("assembling-machine-1", 0, 0);
assembler.enable = should_produce;

# Status lamps
Entity lamp_iron = place("small-lamp", 5, 0);
Entity lamp_copper = place("small-lamp", 7, 0);
Entity lamp_running = place("small-lamp", 9, 0);

lamp_iron.enable = need_iron;
lamp_copper.enable = need_copper;
lamp_running.enable = should_produce;
```

---

## Tips and Best Practices

**Use meaningful positions for displays:**
```facto
Entity lamp1 = place("small-lamp", 0, 0);
Entity lamp2 = place("small-lamp", 2, 0);  # 2-tile spacing
Entity lamp3 = place("small-lamp", 4, 0);
```

**Group related entities:**
```facto
# Status display at y=0
Entity status_lamp = place("small-lamp", 0, 0);

# Control logic at y=5
Entity inserter = place("inserter", 0, 5);
```

**Let the compiler inline conditions:**
```facto
# Good — inlines into lamp's condition
lamp.enable = count > 100;

# Less optimal — creates extra combinator
Signal enable = count > 100;
lamp.enable = enable;
```

---

## Entity Reference

For a complete list of all entities, prototypes, and properties, see the **[Entity Reference](ENTITY_REFERENCE.md)**.

---

## Summary

| Action | Syntax |
|--------|--------|
| Place entity | `place("prototype", x, y, {props})` |
| Control entity | `entity.enable = condition;` |
| Read contents | `Bundle contents = chest.output;` |
| RGB lamp colors | `lamp.r`, `lamp.g`, `lamp.b` (0-255) |

**Key points:**
- The compiler inlines simple conditions into entity circuit conditions
- Use `.output` to read entity contents as bundles
- For loops unroll at compile time for entity placement
- See Entity Reference for all supported entity types

---

**[← Memory](04_memory.md)** | **[Functions →](06_functions.md)**
