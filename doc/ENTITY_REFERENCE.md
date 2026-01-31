# Entity Reference for Facto

**Generated:** 2026-01-29
**Draftsman version:** 3.2.0

This is the **complete reference** for all entities available in the DSL.
Each entity lists its prototypes, circuit I/O capabilities, and all settable properties.

## Table of Contents

- [Using Entities in the DSL](#using-entities-in-the-dsl)
- [Reading Entity Outputs](#reading-entity-outputs)
- [Enum Reference](#enum-reference)
- [Combinators](#combinators)
- [Lamps & Displays](#lamps--displays)
- [Inserters](#inserters)
- [Belts & Logistics](#belts--logistics)
- [Train System](#train-system)
- [Production](#production)
- [Storage](#storage)
- [Power](#power)
- [Fluids](#fluids)
- [Combat](#combat)
- [Robots & Logistics](#robots--logistics)
- [Space](#space)
- [Misc](#misc)
- [Other Entities](#other-entities)

## Using Entities in the DSL

### How the Compiler Handles Entities

When you use `place()` in Facto, the compiler creates a corresponding
[Draftsman](https://github.com/redruin1/factorio-draftsman) entity object.
Properties specified in the placement object (the `{...}` part) are passed directly
to Draftsman as Python attributes during entity construction. The compiler validates
that property names and types match what Draftsman expects for that entity class.

Circuit-controlled properties (like `entity.enable = expression`) are handled differently:
the compiler generates the necessary combinator logic and wire connections to implement
the circuit behavior, then sets the appropriate control properties on the entity.

### Placement Syntax

```facto
Entity name = place("prototype-name", x, y, {prop1: value1, prop2: value2});
```

### Setting Properties

**At placement time:**
```facto
Entity lamp = place("small-lamp", 0, 0, {use_colors: 1, color_mode: 1});
```

**After placement (circuit-controlled):**
```facto
lamp.enable = signal > 0;
lamp.r = red_value;
```

## Reading Entity Outputs

Most entities can output circuit signals. Access them using `.output`:

```facto
Entity combinator = place("arithmetic-combinator", 0, 0, {...});
Signal result = combinator.output;  # Read the combinator's output signal
```

This is particularly useful for reading values from combinators, sensors, and other entities that produce circuit signals.

## Wiring Signals to Entity Inputs

To wire signals to an entity's circuit input, use `.input`:

```facto
Bundle resources = { ("iron-plate", 100), ("copper-plate", 80) };
Entity selector = place("selector-combinator", 0, 0, {
    operation: "count",
    count_signal: "signal-C"
});
selector.input = resources;  # Wire resources to selector's input
```

The `.input` property creates a wire connection from a signal source to the entity:

- **For dual-connector entities** (arithmetic, decider, selector combinators): Wires connect to the **input side** of the combinator
- **For single-connector entities** (lamps, chests, inserters): Wires connect to the entity's circuit connection

This is useful for:
- Providing input signals to combinators
- Connecting signal sources to entities for circuit control
- Building circuits where explicit wiring is needed

### Enum Properties

Enum properties accept **integer values**. See the [Enum Reference](#enum-reference) for all values.

```facto
Entity lamp = place("small-lamp", 0, 0, {color_mode: 1});  # 1 = COMPONENTS
```

### Boolean Properties

Boolean properties accept `1` (true) or `0` (false):
```facto
Entity lamp = place("small-lamp", 0, 0, {use_colors: 1, always_on: 1});
```

## Enum Reference

When setting enum properties in the DSL, use the **integer value** for IntEnums,
or the **string value** for Literal types.

### Integer Enums

#### <a id="lampcolormode"></a>LampColorMode

| DSL Value | Enum Name |
|-----------|-----------|
| `0` | COLOR_MAPPING |
| `1` | COMPONENTS |
| `2` | PACKED_RGB |

#### <a id="direction"></a>Direction

| DSL Value | Enum Name |
|-----------|-----------|
| `0` | NORTH |
| `1` | NORTHNORTHEAST |
| `2` | NORTHEAST |
| `3` | EASTNORTHEAST |
| `4` | EAST |
| `5` | EASTSOUTHEAST |
| `6` | SOUTHEAST |
| `7` | SOUTHSOUTHEAST |
| `8` | SOUTH |
| `9` | SOUTHSOUTHWEST |
| `10` | SOUTHWEST |
| `11` | WESTSOUTHWEST |
| `12` | WEST |
| `13` | WESTNORTHWEST |
| `14` | NORTHWEST |
| `15` | NORTHNORTHWEST |

#### <a id="insertermodeofoperation"></a>InserterModeOfOperation

| DSL Value | Enum Name |
|-----------|-----------|
| `0` | ENABLE_DISABLE |
| `1` | SET_FILTERS |
| `2` | READ_HAND_CONTENTS |
| `3` | NONE |
| `4` | SET_STACK_SIZE |

#### <a id="inserterreadmode"></a>InserterReadMode

| DSL Value | Enum Name |
|-----------|-----------|
| `0` | PULSE |
| `1` | HOLD |

#### <a id="beltreadmode"></a>BeltReadMode

| DSL Value | Enum Name |
|-----------|-----------|
| `0` | PULSE |
| `1` | HOLD |
| `2` | HOLD_ALL_BELTS |

#### <a id="filtermode"></a>FilterMode

| DSL Value | Enum Name |
|-----------|-----------|
| `0` | WHITELIST |
| `1` | BLACKLIST |

#### <a id="logisticmodeofoperation"></a>LogisticModeOfOperation

| DSL Value | Enum Name |
|-----------|-----------|
| `0` | SEND_CONTENTS |
| `1` | SET_REQUESTS |
| `2` | NONE |

#### <a id="miningdrillreadmode"></a>MiningDrillReadMode

| DSL Value | Enum Name |
|-----------|-----------|
| `0` | UNDER_DRILL |
| `1` | TOTAL_PATCH |

#### <a id="siloreadmode"></a>SiloReadMode

| DSL Value | Enum Name |
|-----------|-----------|
| `0` | NONE |
| `1` | READ_CONTENTS |
| `2` | READ_ORBITAL_REQUESTS |

### <a id="readitemsmode"></a>ReadItemsMode

| DSL Value | Enum Name |
|-----------|-----------|
| `0` | NONE |
| `1` | LOGISTICS |
| `2` | MISSING_REQUESTS |

### String Enums (Literal Types)

These properties accept string values. Use the exact string shown.

#### <a id="arithmeticoperation"></a>ArithmeticOperation

| Valid Values |
|-------------|
| `"*"` |
| `"/"` |
| `"+"` |
| `"-"` |
| `"%"` |
| `"^"` |
| `"<<"` |
| `">>"` |
| `"AND"` |
| `"OR"` |
| `"XOR"` |

#### <a id="filtermode"></a>FilterMode

| Valid Values |
|-------------|
| `"whitelist"` |
| `"blacklist"` |

#### <a id="iotype"></a>IOType

| Valid Values |
|-------------|
| `"input"` |
| `"output"` |

#### <a id="infinitymode"></a>InfinityMode

| Valid Values |
|-------------|
| `"at-least"` |
| `"at-most"` |
| `"exactly"` |
| `"add"` |
| `"remove"` |

#### <a id="playbackmode"></a>PlaybackMode

| Valid Values |
|-------------|
| `"local"` |
| `"surface"` |
| `"global"` |

#### <a id="qualityid"></a>QualityID

| Valid Values |
|-------------|
| `"normal"` |
| `"uncommon"` |
| `"rare"` |
| `"epic"` |
| `"legendary"` |
| `"quality-unknown"` |

#### <a id="selectoroperation"></a>SelectorOperation

| Valid Values |
|-------------|
| `"select"` |
| `"count"` |
| `"random"` |
| `"stack-size"` |
| `"rocket-capacity"` |
| `"quality-filter"` |
| `"quality-transfer"` |

#### <a id="splitterpriority"></a>SplitterPriority

| Valid Values |
|-------------|
| `"left"` |
| `"none"` |
| `"right"` |

#### <a id="spoilpriority"></a>SpoilPriority

| Valid Values |
|-------------|
| `"spoiled-first"` |
| `"fresh-first"` |

## Combinators

### ArithmeticCombinator

**Description:** An arithmetic combinator. Peforms a mathematical or bitwise operation on circuit signals.

**Draftsman Source:** [ArithmeticCombinator class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/arithmetic_combinator.py)

**Prototypes:** `"arithmetic-combinator"`

**Connection Type:** Dual circuit (separate input/output sides)

#### Reading Entity Output

Use `entity.output` to read: **Computed arithmetic result**

```facto
Entity e = place("arithmetic-combinator", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### DSL Examples

```facto
# Note: Arithmetic combinators are usually auto-generated
Signal result = input * 2 + offset;  # Creates combinator(s)

# Manual placement if needed
Entity arith = place("arithmetic-combinator", 0, 0, {
    operation: "+"
});
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `first_operand` | Integer | None |  |
| `first_operand_wires` | Complex (see draftsman docs) ⚠️ | (factory) |  |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `operation` | One of: `"*"`, `"/"`, `"+"`, `"-"`, `"%"`, `"^"`, `"<<"`, `">>"`, `"AND"`, `"OR"`, `"XOR"` ([ArithmeticOperation](#arithmeticoperation)) | "*" | `"*"` |
| `player_description` | String | "" |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `second_operand` | Integer | 0 |  |
| `second_operand_wires` | Complex (see draftsman docs) ⚠️ | (factory) |  |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `output_signal` | String (signal name, e.g. `"signal-A"`) | The output signal of the ``ArithmeticCombinator``. Cannot be... |

---

### DeciderCombinator

**Description:** A decider combinator. Makes comparisons based on circuit network inputs.

**Draftsman Source:** [DeciderCombinator class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/decider_combinator.py)

**Prototypes:** `"decider-combinator"`

**Connection Type:** Dual circuit (separate input/output sides)

#### Reading Entity Output

Use `entity.output` to read: **Conditional output signals**

```facto
Entity e = place("decider-combinator", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### DSL Examples

```facto
# Note: Decider combinators are usually auto-generated
Signal flag = (count > 100) : 1;  # Creates decider
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `conditions` | List (complex) ⚠️ | (factory) |  |
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `outputs` | List (complex) ⚠️ | (factory) |  |
| `player_description` | String | "" |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### ConstantCombinator

**Description:** A combinator that holds a number of constant signals that can be output to the circuit network.

**Draftsman Source:** [ConstantCombinator class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/constant_combinator.py)

**Prototypes:** `"constant-combinator"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Constant signal values**

```facto
Entity e = place("constant-combinator", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### DSL Examples

```facto
# Note: Constants are usually auto-generated
Signal constant = 42;  # Creates constant combinator
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `enabled` | Boolean (`0` or `1`) | 1 | `1` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `player_description` | String | "" |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `sections` | List (complex) ⚠️ | (factory) |  |

---

### SelectorCombinator

**Description:** (Factorio 2.0)

**Draftsman Source:** [SelectorCombinator class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/selector_combinator.py)

**Prototypes:** `"selector-combinator"`

**Connection Type:** Dual circuit (separate input/output sides)

#### Reading Entity Output

Use `entity.output` to read: **Selected/filtered signals**

```facto
Entity e = place("selector-combinator", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### DSL Examples

```facto
# Selector in count mode
Entity counter = place("selector-combinator", 0, 0, {
    operation: "count",
    count_signal: "signal-C"
});
# Reading output
Bundle result = counter.output;
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `index_constant` | Integer | 0 |  |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `operation` | One of: `"select"`, `"count"`, `"random"`, `"stack-size"`, `"rocket-capacity"`, `"quality-filter"`, `"quality-transfer"` ([SelectorOperation](#selectoroperation)) | "select" | `"select"` |
| `player_description` | String | "" |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `quality_filter` | Complex (see draftsman docs) ⚠️ | (factory) |  |
| `quality_source_static` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `random_update_interval` | Integer | 0 |  |
| `select_max` | Boolean (`0` or `1`) | 1 | `1` |
| `select_quality_from_signal` | Boolean (`0` or `1`) | 0 | `1` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `count_signal` | String (signal name, e.g. `"signal-A"`) | What signal to output the sum total number of unique signals... |
| `index_signal` | String (signal name, e.g. `"signal-A"`) | Which input signal to pull the index value from in order to ... |
| `quality_destination_signal` | String (signal name, e.g. `"signal-A"`) | The destination signal(s) to output with the read quality va... |
| `quality_source_signal` | String (signal name, e.g. `"signal-A"`) | The input signal type to pull the quality from dynamically, ... |

---

## Lamps & Displays

### Lamp

**Description:** An entity that illuminates an area.

**Draftsman Source:** [Lamp class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/lamp.py)

**Prototypes:** `"small-lamp"`

**Connection Type:** Single circuit connection

#### DSL Examples

```facto
# Basic lamp controlled by circuit
Entity lamp = place("small-lamp", 0, 0);
lamp.enable = signal > 0;

# RGB colored lamp (color_mode: 1 = COMPONENTS)
Entity rgb_lamp = place("small-lamp", 2, 0, {use_colors: 1, color_mode: 1});
rgb_lamp.r = red_value;
rgb_lamp.g = green_value;
rgb_lamp.b = blue_value;
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `always_on` | Boolean (`0` or `1`) | 0 | `1` |
| `circuit_enabled` | Boolean (`0` or `1`) | 0 | `1` |
| `color` | Object `{r, g, b}` (0-255 each) | (factory) | `{r: 255, g: 0, b: 0}` |
| `color_mode` | Integer ([LampColorMode](#lampcolormode)) | 0 | `0  # COLOR_MAPPING` |
| `connect_to_logistic_network` | Boolean (`0` or `1`) | 0 | `1` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `use_colors` | Boolean (`0` or `1`) | 0 | `1` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `blue_signal` | String (signal name, e.g. `"signal-A"`) | The signal to pull the blue color component from, if color_m... |
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |
| `green_signal` | String (signal name, e.g. `"signal-A"`) | The signal to pull the green color component from, if color_... |
| `logistic_condition` | Condition (use `.enable = expr`) | The logistic condition that must be passed in order for this... |
| `red_signal` | String (signal name, e.g. `"signal-A"`) |  |
| `rgb_signal` | String (signal name, e.g. `"signal-A"`) | .. versionadded:: 3.0.0 (Factorio 2.0) |

---

### DisplayPanel

**Description:** (Factorio 2.0)

**Draftsman Source:** [DisplayPanel class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/display_panel.py)

**Prototypes:** `"display-panel"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("display-panel", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `always_show_in_alt_mode` | Boolean (`0` or `1`) | 0 | `1` |
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `icon` | String (signal name, e.g. `"signal-A"`) | None |  |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `messages` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `player_description` | String | "" |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `show_in_chart` | Boolean (`0` or `1`) | 0 | `1` |
| `text` | String | "" |  |

---

## Inserters

### Inserter

**Description:** An entity with a swinging arm that can move items between machines.

**Draftsman Source:** [Inserter class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/inserter.py)

**Prototypes:** `"bulk-inserter"`, `"fast-inserter"`, `"inserter"`, `"burner-inserter"`, `"long-handed-inserter"`, `"stack-inserter"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Items in hand or filter status**

```facto
Entity e = place("bulk-inserter", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

**Enable properties:**

- `read_hand_contents`: Read items in hand
- `circuit_set_filters`: Control via filters

#### DSL Examples

```facto
# Inserter that enables when chest has items
Entity inserter = place("inserter", 0, 0, {direction: 4});
inserter.enable = chest.output > 50;
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `circuit_enabled` | Boolean (`0` or `1`) | 0 | `1` |
| `circuit_set_filters` | Boolean (`0` or `1`) | 0 | `1` |
| `circuit_set_stack_size` | Boolean (`0` or `1`) | 0 | `1` |
| `connect_to_logistic_network` | Boolean (`0` or `1`) | 0 | `1` |
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `drop_position_offset` | Vector `{x, y}` ⚠️ | (factory) |  |
| `filter_mode` | One of: `"whitelist"`, `"blacklist"` ([FilterMode](#filtermode)) | "whitelist" | `"whitelist"` |
| `filters` | List (complex) ⚠️ | (factory) |  |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `mode_of_operation` | Integer ([InserterModeOfOperation](#insertermodeofoperation)) | 0 | `0  # ENABLE_DISABLE` |
| `name` | String (entity prototype name) | (factory) |  |
| `override_stack_size` | Integer | None |  |
| `pickup_position_offset` | Vector `{x, y}` ⚠️ | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_hand_contents` | Boolean (`0` or `1`) | 0 | `1` |
| `read_mode` | Integer ([InserterReadMode](#inserterreadmode)) | 0 | `0  # PULSE` |
| `spoil_priority` | One of: `"spoiled-first"`, `"fresh-first"` ([SpoilPriority](#spoilpriority)) | None | `"spoiled-first"` |
| `use_filters` | Boolean (`0` or `1`) | 0 | `1` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |
| `logistic_condition` | Condition (use `.enable = expr`) | The logistic condition that must be passed in order for this... |
| `stack_size_control_signal` | String (signal name, e.g. `"signal-A"`) | What circuit network signal should indicate the current stac... |

---

## Belts & Logistics

### TransportBelt

**Description:** An entity that transports items.

**Draftsman Source:** [TransportBelt class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/transport_belt.py)

**Prototypes:** `"express-transport-belt"`, `"transport-belt"`, `"fast-transport-belt"`, `"turbo-transport-belt"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Items on the belt**

```facto
Entity e = place("express-transport-belt", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

**Enable properties:**

- `read_contents`: Read belt contents

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `circuit_enabled` | Boolean (`0` or `1`) | 0 | `1` |
| `connect_to_logistic_network` | Boolean (`0` or `1`) | 0 | `1` |
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_contents` | Boolean (`0` or `1`) | 0 | `1` |
| `read_mode` | Integer ([BeltReadMode](#beltreadmode)) | 0 | `0  # PULSE` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |
| `logistic_condition` | Condition (use `.enable = expr`) | The logistic condition that must be passed in order for this... |

---

### UndergroundBelt

**Description:** A transport belt that transfers items underneath other entities.

**Draftsman Source:** [UndergroundBelt class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/underground_belt.py)

**Prototypes:** `"underground-belt"`, `"turbo-underground-belt"`, `"fast-underground-belt"`, `"express-underground-belt"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `io_type` | One of: `"input"`, `"output"` ([IOType](#iotype)) | "input" | `"input"` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### Splitter

**Description:** An entity that evenly splits a set of input belts between a set of output belts.

**Draftsman Source:** [Splitter class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/splitter.py)

**Prototypes:** `"turbo-splitter"`, `"express-splitter"`, `"fast-splitter"`, `"splitter"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("turbo-splitter", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `filter` | String (signal name, e.g. `"signal-A"`) | None |  |
| `input_left_condition` | Condition (use `.enable = expr`) ⚠️ | (factory) |  |
| `input_priority` | One of: `"left"`, `"none"`, `"right"` ([SplitterPriority](#splitterpriority)) | "none" | `"left"` |
| `input_right_condition` | Condition (use `.enable = expr`) ⚠️ | (factory) |  |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `output_left_condition` | Condition (use `.enable = expr`) ⚠️ | (factory) |  |
| `output_priority` | One of: `"left"`, `"none"`, `"right"` ([SplitterPriority](#splitterpriority)) | "none" | `"left"` |
| `output_right_condition` | Condition (use `.enable = expr`) ⚠️ | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `set_filter` | Boolean (`0` or `1`) | 0 | `1` |
| `set_input_side` | Boolean (`0` or `1`) | 0 | `1` |
| `set_output_side` | Boolean (`0` or `1`) | 0 | `1` |

---

### Loader

**Description:** An entity that inserts items directly from a belt to an inventory or vise-versa.

**Draftsman Source:** [Loader class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/loader.py)

**Prototypes:** `"loader"`, `"fast-loader"`, `"turbo-loader"`, `"express-loader"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `filters` | List (complex) ⚠️ | (factory) |  |
| `io_type` | One of: `"input"`, `"output"` ([IOType](#iotype)) | "input" | `"input"` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `use_filters` | Boolean (`0` or `1`) | 0 | `1` |

---

### LinkedBelt

**Description:** A belt object that can transfer items over any distance, regardless of constraint, as long as the two are paired together.

**Draftsman Source:** [LinkedBelt class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/linked_belt.py)

**Prototypes:** `"linked-belt"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

## Train System

### TrainStop

**Description:** A stop for making train schedules for locomotives.

**Draftsman Source:** [TrainStop class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/train_stop.py)

**Prototypes:** `"train-stop"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Train ID, count, and cargo**

```facto
Entity e = place("train-stop", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

**Enable properties:**

- `read_stopped_train`: Read stopped train ID
- `read_trains_count`: Read incoming trains count
- `read_from_train`: Read train cargo

#### DSL Examples

```facto
# Train station with circuit control
Entity station = place("train-stop", 0, 0, {station: "Iron Pickup"});
station.enable = cargo.output > 0;
# Read train info
Bundle train_info = station.output;
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `circuit_enabled` | Boolean (`0` or `1`) | 0 | `1` |
| `color` | Object `{r, g, b}` (0-255 each) | (factory) | `{r: 255, g: 0, b: 0}` |
| `connect_to_logistic_network` | Boolean (`0` or `1`) | 0 | `1` |
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `manual_trains_limit` | Integer | None |  |
| `name` | String (entity prototype name) | (factory) |  |
| `priority` | Integer | 50 |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_from_train` | Boolean (`0` or `1`) | 0 | `1` |
| `read_stopped_train` | Boolean (`0` or `1`) | 0 | `1` |
| `read_trains_count` | Boolean (`0` or `1`) | 0 | `1` |
| `send_to_train` | Boolean (`0` or `1`) | 1 | `1` |
| `set_priority` | Boolean (`0` or `1`) | 0 | `1` |
| `signal_limits_trains` | Boolean (`0` or `1`) | 0 | `1` |
| `station` | String | "" | `"Station Name"` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |
| `logistic_condition` | Condition (use `.enable = expr`) | The logistic condition that must be passed in order for this... |
| `priority_signal` | String (signal name, e.g. `"signal-A"`) | Which signal to read the dynamic priority of this train stop... |
| `train_stopped_signal` | String (signal name, e.g. `"signal-A"`) | What signal to output the unique train ID if a train is curr... |
| `trains_count_signal` | String (signal name, e.g. `"signal-A"`) | What signal to use to output the current number of trains th... |
| `trains_limit_signal` | String (signal name, e.g. `"signal-A"`) | What signal to read to limit the number of trains that can u... |

---

### RailSignal

**Description:** A rail signal that determines whether or not trains can pass along their rail block.

**Draftsman Source:** [RailSignal class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/rail_signal.py)

**Prototypes:** `"rail-signal"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Signal state (red/yellow/green)**

```facto
Entity e = place("rail-signal", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

**Enable properties:**

- `read_signal`: Read signal state

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `circuit_enabled` | Boolean (`0` or `1`) | 0 | `1` |
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_signal` | Boolean (`0` or `1`) | 1 | `1` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |
| `green_output_signal` | String (signal name, e.g. `"signal-A"`) | The green output signal. Sent with a unit value when the rai... |
| `red_output_signal` | String (signal name, e.g. `"signal-A"`) | The red output signal. Sent with a unit value when the rail ... |
| `yellow_output_signal` | String (signal name, e.g. `"signal-A"`) | The yellow output signal. Sent with a unit value when the ra... |

---

### RailChainSignal

**Description:** A rail signal that determines access of a current rail block based on a forward rail block.

**Draftsman Source:** [RailChainSignal class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/rail_chain_signal.py)

**Prototypes:** `"rail-chain-signal"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Signal state (red/yellow/green/blue)**

```facto
Entity e = place("rail-chain-signal", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

**Enable properties:**

- `read_signal`: Read signal state

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `blue_output_signal` | String (signal name, e.g. `"signal-A"`) | The blue output signal. Sent with a unit value when the rail... |
| `green_output_signal` | String (signal name, e.g. `"signal-A"`) | The green output signal. Sent with a unit value when the rai... |
| `red_output_signal` | String (signal name, e.g. `"signal-A"`) | The red output signal. Sent with a unit value when the rail ... |
| `yellow_output_signal` | String (signal name, e.g. `"signal-A"`) | The yellow output signal. Sent with a unit value when the ra... |

---

### Locomotive

**Description:** A train car that moves other wagons around using a fuel.

**Draftsman Source:** [Locomotive class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/locomotive.py)

**Prototypes:** `"locomotive"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `color` | Object `{r, g, b}` (0-255 each) | Color(r=0.9176470588235294, g=0.06666666666666667, b=0, a=0.4980392156862745) | `{r: 255, g: 0, b: 0}` |
| `enable_logistics_while_moving` | Boolean (`0` or `1`) | 1 | `1` |
| `equipment` | List (complex) ⚠️ | (factory) |  |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `orientation` | Complex (see draftsman docs) ⚠️ | <Orientation.NORTH: 0.0> |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### CargoWagon

**Description:** A train wagon that holds items as cargo.

**Draftsman Source:** [CargoWagon class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/cargo_wagon.py)

**Prototypes:** `"cargo-wagon"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `enable_logistics_while_moving` | Boolean (`0` or `1`) | 1 | `1` |
| `equipment` | List (complex) ⚠️ | (factory) |  |
| `inventory` | Complex (see draftsman docs) ⚠️ | (factory) |  |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `orientation` | Complex (see draftsman docs) ⚠️ | <Orientation.NORTH: 0.0> |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### FluidWagon

**Description:** A train wagon that holds a fluid as cargo.

**Draftsman Source:** [FluidWagon class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/fluid_wagon.py)

**Prototypes:** `"fluid-wagon"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `enable_logistics_while_moving` | Boolean (`0` or `1`) | 1 | `1` |
| `equipment` | List (complex) ⚠️ | (factory) |  |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `orientation` | Complex (see draftsman docs) ⚠️ | <Orientation.NORTH: 0.0> |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### ArtilleryWagon

**Description:** An artillery train car.

**Draftsman Source:** [ArtilleryWagon class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/artillery_wagon.py)

**Prototypes:** `"artillery-wagon"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `auto_target` | Boolean (`0` or `1`) | 1 | `1` |
| `enable_logistics_while_moving` | Boolean (`0` or `1`) | 1 | `1` |
| `equipment` | List (complex) ⚠️ | (factory) |  |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `orientation` | Complex (see draftsman docs) ⚠️ | <Orientation.NORTH: 0.0> |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

## Production

### AssemblingMachine

**Description:** A machine that takes input items and produces output items. Includes assembling machines, chemical plants, oil refineries, and centrifuges, but does not include :py:class:`.RocketSilo`.

**Draftsman Source:** [AssemblingMachine class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/assembling_machine.py)

**Prototypes:** `"centrifuge"`, `"biochamber"`, `"oil-refinery"`, `"foundry"`, `"captive-biter-spawner"`, `"assembling-machine-1"`, `"electromagnetic-plant"`, `"crusher"`, `"cryogenic-plant"`, `"assembling-machine-2"`, ... (12 total)

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Recipe finished pulse, working status**

```facto
Entity e = place("centrifuge", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

**Enable properties:**

- `read_recipe_finished`: Pulse when recipe completes
- `read_working`: Output working status

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `circuit_enabled` | Boolean (`0` or `1`) | 0 | `1` |
| `circuit_set_recipe` | Boolean (`0` or `1`) | 0 | `1` |
| `connect_to_logistic_network` | Boolean (`0` or `1`) | 0 | `1` |
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `include_in_crafting` | Boolean (`0` or `1`) | 1 | `1` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_contents` | Boolean (`0` or `1`) | 0 | `1` |
| `read_recipe_finished` | Boolean (`0` or `1`) | 0 | `1` |
| `read_working` | Boolean (`0` or `1`) | 0 | `1` |
| `recipe` | String (recipe name) | None | `"iron-gear-wheel"` |
| `recipe_quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |
| `logistic_condition` | Condition (use `.enable = expr`) | The logistic condition that must be passed in order for this... |
| `recipe_finished_signal` | String (signal name, e.g. `"signal-A"`) | What signal to pulse when the crafting cycle completes. Only... |
| `working_signal` | String (signal name, e.g. `"signal-A"`) | .. versionadded:: 3.0.0 (Factorio 2.0) |

---

### Furnace

**Description:** An entity that automatically determines it's recipe from it's input items. Obviously includes regular furnaces, but can also include other machines like recyclers.

**Draftsman Source:** [Furnace class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/furnace.py)

**Prototypes:** `"recycler"`, `"stone-furnace"`, `"steel-furnace"`, `"electric-furnace"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("recycler", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `circuit_enabled` | Boolean (`0` or `1`) | 0 | `1` |
| `circuit_set_recipe` | Boolean (`0` or `1`) | 0 | `1` |
| `connect_to_logistic_network` | Boolean (`0` or `1`) | 0 | `1` |
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `include_in_crafting` | Boolean (`0` or `1`) | 1 | `1` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_contents` | Boolean (`0` or `1`) | 0 | `1` |
| `read_recipe_finished` | Boolean (`0` or `1`) | 0 | `1` |
| `read_working` | Boolean (`0` or `1`) | 0 | `1` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |
| `logistic_condition` | Condition (use `.enable = expr`) | The logistic condition that must be passed in order for this... |
| `recipe_finished_signal` | String (signal name, e.g. `"signal-A"`) | What signal to pulse when the crafting cycle completes. Only... |
| `working_signal` | String (signal name, e.g. `"signal-A"`) | .. versionadded:: 3.0.0 (Factorio 2.0) |

---

### MiningDrill

**Description:** An entity that extracts resources (item or fluid) from the environment.

**Draftsman Source:** [MiningDrill class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/mining_drill.py)

**Prototypes:** `"electric-mining-drill"`, `"big-mining-drill"`, `"pumpjack"`, `"burner-mining-drill"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Resources under the drill**

```facto
Entity e = place("electric-mining-drill", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

**Enable properties:**

- `read_resources`: Read resource amounts

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `circuit_enabled` | Boolean (`0` or `1`) | 0 | `1` |
| `connect_to_logistic_network` | Boolean (`0` or `1`) | 0 | `1` |
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_mode` | Integer ([MiningDrillReadMode](#miningdrillreadmode)) | 0 | `0  # UNDER_DRILL` |
| `read_resources` | Boolean (`0` or `1`) | 1 | `1` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |
| `logistic_condition` | Condition (use `.enable = expr`) | The logistic condition that must be passed in order for this... |

---

### Lab

**Description:** An entity that consumes items and produces research.

**Draftsman Source:** [Lab class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/lab.py)

**Prototypes:** `"lab"`, `"biolab"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### RocketSilo

**Description:** An entity that launches rockets, usually to move items between surfaces or space platforms.

**Draftsman Source:** [RocketSilo class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/rocket_silo.py)

**Prototypes:** `"rocket-silo"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("rocket-silo", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `auto_launch` | Boolean (`0` or `1`) | 0 | `1` |
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_items_mode` | Integer ([SiloReadMode](#siloreadmode)) | 1 | `0  # NONE` |
| `recipe` | String (recipe name) | "rocket-part" | `"iron-gear-wheel"` |
| `recipe_quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `transitional_request_index` | Integer | 0 |  |
| `use_transitional_requests` | Boolean (`0` or `1`) | 0 | `1` |

---

### Beacon

**Description:** An entity designed to apply module effects to other machines in its radius.

**Draftsman Source:** [Beacon class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/beacon.py)

**Prototypes:** `"beacon"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### Boiler

**Description:** An entity that uses a fuel to convert a fluid (usually water) to another fluid (usually steam).

**Draftsman Source:** [Boiler class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/boiler.py)

**Prototypes:** `"boiler"`, `"heat-exchanger"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### Generator

**Description:** An entity that converts a fluid (usually steam) to electricity.

**Draftsman Source:** [Generator class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/generator.py)

**Prototypes:** `"steam-engine"`, `"steam-turbine"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### Reactor

**Description:** An entity that converts a fuel into thermal energy.

**Draftsman Source:** [Reactor class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/reactor.py)

**Prototypes:** `"heating-tower"`, `"nuclear-reactor"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("heating-tower", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_burner_fuel` | Boolean (`0` or `1`) | 0 | `1` |
| `read_temperature` | Boolean (`0` or `1`) | 0 | `1` |
| `temperature_signal` | String (signal name, e.g. `"signal-A"`) | (factory) | `"signal-A"` |

---

## Storage

### Container

**Description:** An entity that holds items.

**Draftsman Source:** [Container class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/container.py)

**Prototypes:** `"factorio-logo-22tiles"`, `"bottomless-chest"`, `"red-chest"`, `"crash-site-chest-2"`, `"factorio-logo-16tiles"`, `"iron-chest"`, `"blue-chest"`, `"crash-site-chest-1"`, `"steel-chest"`, `"factorio-logo-11tiles"`, ... (11 total)

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Item contents of the container**

```facto
Entity e = place("factorio-logo-22tiles", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

**Enable properties:**

- `read_contents`: Enable reading container contents

#### DSL Examples

```facto
# Read chest contents
Entity chest = place("iron-chest", 0, 0);
Bundle contents = chest.output;
Signal iron = contents["iron-plate"];
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `bar` | Integer | None |  |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### LogisticPassiveContainer

**Description:** A logistics container that provides it's contents to the logistic network when needed by the network.

**Draftsman Source:** [LogisticPassiveContainer class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/logistic_passive_container.py)

**Prototypes:** `"passive-provider-chest"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("passive-provider-chest", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `bar` | Integer | None |  |
| `circuit_enabled` | Boolean (`0` or `1`) | 0 | `1` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_contents` | Boolean (`0` or `1`) | 1 | `1` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |

---

### LogisticActiveContainer

**Description:** A logistics container that immediately provides it's contents to the logistic network.

**Draftsman Source:** [LogisticActiveContainer class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/logistic_active_container.py)

**Prototypes:** `"active-provider-chest"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("active-provider-chest", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `bar` | Integer | None |  |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_contents` | Boolean (`0` or `1`) | 1 | `1` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |

---

### LogisticStorageContainer

**Description:** A logistics container that stores items not currently being used in the logistic network.

**Draftsman Source:** [LogisticStorageContainer class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/logistic_storage_container.py)

**Prototypes:** `"storage-chest"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("storage-chest", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `bar` | Integer | None |  |
| `circuit_enabled` | Boolean (`0` or `1`) | 0 | `1` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_contents` | Boolean (`0` or `1`) | 1 | `1` |
| `request_from_buffers` | Boolean (`0` or `1`) | 1 | `1` |
| `requests_enabled` | Boolean (`0` or `1`) | 1 | `1` |
| `sections` | List (complex) ⚠️ | (factory) |  |
| `trash_not_requested` | Boolean (`0` or `1`) | 0 | `1` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |

---

### LogisticRequestContainer

**Description:** A logistics container that requests items with a primary priority.

**Draftsman Source:** [LogisticRequestContainer class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/logistic_request_container.py)

**Prototypes:** `"requester-chest"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("requester-chest", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `bar` | Integer | None |  |
| `circuit_enabled` | Boolean (`0` or `1`) | 0 | `1` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `mode_of_operation` | Integer ([LogisticModeOfOperation](#logisticmodeofoperation)) | 0 | `0  # SEND_CONTENTS` |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `request_from_buffers` | Boolean (`0` or `1`) | 0 | `1` |
| `requests_enabled` | Boolean (`0` or `1`) | 1 | `1` |
| `sections` | List (complex) ⚠️ | (factory) |  |
| `trash_not_requested` | Boolean (`0` or `1`) | 0 | `1` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |

---

### LogisticBufferContainer

**Description:** A logistics container that requests items on a secondary priority.

**Draftsman Source:** [LogisticBufferContainer class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/logistic_buffer_container.py)

**Prototypes:** `"buffer-chest"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("buffer-chest", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `bar` | Integer | None |  |
| `circuit_enabled` | Boolean (`0` or `1`) | 0 | `1` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `mode_of_operation` | Integer ([LogisticModeOfOperation](#logisticmodeofoperation)) | 0 | `0  # SEND_CONTENTS` |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `request_from_buffers` | Boolean (`0` or `1`) | 1 | `1` |
| `requests_enabled` | Boolean (`0` or `1`) | 1 | `1` |
| `sections` | List (complex) ⚠️ | (factory) |  |
| `trash_not_requested` | Boolean (`0` or `1`) | 0 | `1` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |

---

## Power

### ElectricPole

**Description:** An entity used to distribute electrical energy as a network.

**Draftsman Source:** [ElectricPole class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/electric_pole.py)

**Prototypes:** `"substation"`, `"medium-electric-pole"`, `"big-electric-pole"`, `"small-electric-pole"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("substation", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### PowerSwitch

**Description:** An entity that connects or disconnects a power network.

**Draftsman Source:** [PowerSwitch class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/power_switch.py)

**Prototypes:** `"power-switch"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("power-switch", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `connect_to_logistic_network` | Boolean (`0` or `1`) | 0 | `1` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `switch_state` | Boolean (`0` or `1`) | 0 | `1` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |
| `logistic_condition` | Condition (use `.enable = expr`) | The logistic condition that must be passed in order for this... |

---

### Accumulator

**Description:** An entity that stores electricity for periods of high demand.

**Draftsman Source:** [Accumulator class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/accumulator.py)

**Prototypes:** `"accumulator"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Charge level percentage**

```facto
Entity e = place("accumulator", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `output_signal` | String (signal name, e.g. `"signal-A"`) | The signal used to output this accumulator's charge level, i... |

---

### SolarPanel

**Description:** An entity that produces electricity depending on the presence of the sun.

**Draftsman Source:** [SolarPanel class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/solar_panel.py)

**Prototypes:** `"solar-panel"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

## Fluids

### Pump

**Description:** An entity that aids fluid transfer through pipes.

**Draftsman Source:** [Pump class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/pump.py)

**Prototypes:** `"pump"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("pump", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `connect_to_logistic_network` | Boolean (`0` or `1`) | 0 | `1` |
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |
| `logistic_condition` | Condition (use `.enable = expr`) | The logistic condition that must be passed in order for this... |

---

### StorageTank

**Description:** An entity that stores a fluid.

**Draftsman Source:** [StorageTank class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/storage_tank.py)

**Prototypes:** `"storage-tank"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Fluid level in the tank**

```facto
Entity e = place("storage-tank", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### OffshorePump

**Description:** An entity that pumps a fluid from the environment.

**Draftsman Source:** [OffshorePump class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/offshore_pump.py)

**Prototypes:** `"offshore-pump"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("offshore-pump", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `connect_to_logistic_network` | Boolean (`0` or `1`) | 0 | `1` |
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |
| `logistic_condition` | Condition (use `.enable = expr`) | The logistic condition that must be passed in order for this... |

---

### Pipe

**Description:** A structure that transports a fluid across a surface.

**Draftsman Source:** [Pipe class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/pipe.py)

**Prototypes:** `"pipe"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

## Combat

### Radar

**Description:** An entity that reveals and scans neighbouring chunks.

**Draftsman Source:** [Radar class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/radar.py)

**Prototypes:** `"radar"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("radar", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### ArtilleryTurret

**Description:** A turret which can only target enemy structures and uses artillery ammunition.

**Draftsman Source:** [ArtilleryTurret class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/artillery_turret.py)

**Prototypes:** `"artillery-turret"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("artillery-turret", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `auto_target` | Boolean (`0` or `1`) | 1 | `1` |
| `circuit_enabled` | Boolean (`0` or `1`) | 0 | `1` |
| `connect_to_logistic_network` | Boolean (`0` or `1`) | 0 | `1` |
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_ammo` | Boolean (`0` or `1`) | 1 | `1` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |
| `logistic_condition` | Condition (use `.enable = expr`) | The logistic condition that must be passed in order for this... |

---

### AmmoTurret

**Description:** An entity that automatically targets and attacks other forces within range. Consumes item-based ammunition.

**Draftsman Source:** [AmmoTurret class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/ammo_turret.py)

**Prototypes:** `"rocket-turret"`, `"gun-turret"`, `"railgun-turret"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("rocket-turret", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `circuit_enabled` | Boolean (`0` or `1`) | 0 | `1` |
| `connect_to_logistic_network` | Boolean (`0` or `1`) | 0 | `1` |
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `ignore_unlisted_targets_condition` | Condition (use `.enable = expr`) ⚠️ | (factory) |  |
| `ignore_unprioritized` | Boolean (`0` or `1`) | 0 | `1` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `priority_list` | List (complex) ⚠️ | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_ammo` | Boolean (`0` or `1`) | 1 | `1` |
| `set_ignore_unprioritized` | Boolean (`0` or `1`) | 0 | `1` |
| `set_priority_list` | Boolean (`0` or `1`) | 0 | `1` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |
| `logistic_condition` | Condition (use `.enable = expr`) | The logistic condition that must be passed in order for this... |

---

### ElectricTurret

**Description:** An entity that automatically targets and attacks other forces within range. Uses electricity as ammunition.

**Draftsman Source:** [ElectricTurret class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/electric_turret.py)

**Prototypes:** `"tesla-turret"`, `"laser-turret"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("tesla-turret", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `circuit_enabled` | Boolean (`0` or `1`) | 0 | `1` |
| `connect_to_logistic_network` | Boolean (`0` or `1`) | 0 | `1` |
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `ignore_unlisted_targets_condition` | Condition (use `.enable = expr`) ⚠️ | (factory) |  |
| `ignore_unprioritized` | Boolean (`0` or `1`) | 0 | `1` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `priority_list` | List (complex) ⚠️ | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_ammo` | Boolean (`0` or `1`) | 1 | `1` |
| `set_ignore_unprioritized` | Boolean (`0` or `1`) | 0 | `1` |
| `set_priority_list` | Boolean (`0` or `1`) | 0 | `1` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |
| `logistic_condition` | Condition (use `.enable = expr`) | The logistic condition that must be passed in order for this... |

---

### FluidTurret

**Description:** An entity that automatically targets and attacks other forces within range. Uses fluids as ammunition.

**Draftsman Source:** [FluidTurret class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/fluid_turret.py)

**Prototypes:** `"flamethrower-turret"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("flamethrower-turret", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `circuit_enabled` | Boolean (`0` or `1`) | 0 | `1` |
| `connect_to_logistic_network` | Boolean (`0` or `1`) | 0 | `1` |
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `ignore_unlisted_targets_condition` | Condition (use `.enable = expr`) ⚠️ | (factory) |  |
| `ignore_unprioritized` | Boolean (`0` or `1`) | 0 | `1` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `priority_list` | List (complex) ⚠️ | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_ammo` | Boolean (`0` or `1`) | 1 | `1` |
| `set_ignore_unprioritized` | Boolean (`0` or `1`) | 0 | `1` |
| `set_priority_list` | Boolean (`0` or `1`) | 0 | `1` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |
| `logistic_condition` | Condition (use `.enable = expr`) | The logistic condition that must be passed in order for this... |

---

### Wall

**Description:** A static barrier that acts as protection for structures.

**Draftsman Source:** [Wall class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/wall.py)

**Prototypes:** `"stone-wall"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("stone-wall", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `circuit_enabled` | Boolean (`0` or `1`) | 1 | `1` |
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_gate` | Boolean (`0` or `1`) | 0 | `1` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |
| `output_signal` | String (signal name, e.g. `"signal-A"`) |  |

---

### Gate

**Description:** A wall that opens near the player.

**Draftsman Source:** [Gate class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/gate.py)

**Prototypes:** `"gate"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### LandMine

**Description:** An entity that explodes when in proximity to another force.

**Draftsman Source:** [LandMine class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/land_mine.py)

**Prototypes:** `"land-mine"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

## Robots & Logistics

### Roboport

**Description:** An entity that acts as a node in a logistics network.

**Draftsman Source:** [Roboport class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/roboport.py)

**Prototypes:** `"roboport"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Robot counts and logistics info**

```facto
Entity e = place("roboport", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

**Enable properties:**

- `read_logistics`: Read logistic robot counts
- `read_robot_stats`: Read robot statistics

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `available_construction_signal` | String (signal name, e.g. `"signal-A"`) | (factory) | `"signal-A"` |
| `available_logistic_signal` | String (signal name, e.g. `"signal-A"`) | (factory) | `"signal-A"` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_items_mode` | Integer (see enum reference) | 1 |  |
| `read_robot_stats` | Boolean (`0` or `1`) | 0 | `1` |
| `request_from_buffers` | Boolean (`0` or `1`) | 1 | `1` |
| `requests_enabled` | Boolean (`0` or `1`) | 1 | `1` |
| `roboport_count_signal` | String (signal name, e.g. `"signal-A"`) | (factory) | `"signal-A"` |
| `sections` | List (complex) ⚠️ | (factory) |  |
| `total_construction_signal` | String (signal name, e.g. `"signal-A"`) | (factory) | `"signal-A"` |
| `total_logistic_signal` | String (signal name, e.g. `"signal-A"`) | (factory) | `"signal-A"` |
| `trash_not_requested` | Boolean (`0` or `1`) | 0 | `1` |

---

## Space

### SpacePlatformHub

**Description:** (Factorio 2.0)

**Draftsman Source:** [SpacePlatformHub class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/space_platform_hub.py)

**Prototypes:** `"space-platform-hub"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("space-platform-hub", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `damage_taken_signal` | String (signal name, e.g. `"signal-A"`) | (factory) | `"signal-A"` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_contents` | Boolean (`0` or `1`) | 1 | `1` |
| `read_damage_taken` | Boolean (`0` or `1`) | 0 | `1` |
| `read_moving_from` | Boolean (`0` or `1`) | 0 | `1` |
| `read_moving_to` | Boolean (`0` or `1`) | 0 | `1` |
| `read_speed` | Boolean (`0` or `1`) | 0 | `1` |
| `request_from_buffers` | Boolean (`0` or `1`) | 1 | `1` |
| `request_missing_construction_materials` | Boolean (`0` or `1`) | 1 | `1` |
| `requests_enabled` | Boolean (`0` or `1`) | 1 | `1` |
| `sections` | List (complex) ⚠️ | (factory) |  |
| `send_to_platform` | Boolean (`0` or `1`) | 1 | `1` |
| `speed_signal` | String (signal name, e.g. `"signal-A"`) | (factory) | `"signal-A"` |
| `trash_not_requested` | Boolean (`0` or `1`) | 0 | `1` |

---

### CargoLandingPad

**Description:** (Factorio 2.0)

**Draftsman Source:** [CargoLandingPad class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/cargo_landing_pad.py)

**Prototypes:** `"cargo-landing-pad"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("cargo-landing-pad", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `mode_of_operation` | Integer ([LogisticModeOfOperation](#logisticmodeofoperation)) | 0 | `0  # SEND_CONTENTS` |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `request_from_buffers` | Boolean (`0` or `1`) | 1 | `1` |
| `requests_enabled` | Boolean (`0` or `1`) | 1 | `1` |
| `sections` | List (complex) ⚠️ | (factory) |  |
| `trash_not_requested` | Boolean (`0` or `1`) | 0 | `1` |

---

### AsteroidCollector

**Description:** (Factorio 2.0)

**Draftsman Source:** [AsteroidCollector class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/asteroid_collector.py)

**Prototypes:** `"asteroid-collector"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("asteroid-collector", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `chunk_filter` | List (complex) ⚠️ | (factory) |  |
| `circuit_enabled` | Boolean (`0` or `1`) | 0 | `1` |
| `circuit_set_filters` | Boolean (`0` or `1`) | 0 | `1` |
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_contents` | Boolean (`0` or `1`) | 0 | `1` |
| `read_hands` | Boolean (`0` or `1`) | 1 | `1` |
| `result_inventory` | Complex (see draftsman docs) ⚠️ | (factory) |  |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |

---

### CargoBay

**Description:** (Factorio 2.0)

**Draftsman Source:** [CargoBay class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/cargo_bay.py)

**Prototypes:** `"cargo-bay"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### Thruster

**Description:** (Factorio 2.0)

**Draftsman Source:** [Thruster class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/thruster.py)

**Prototypes:** `"thruster"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

## Misc

### ProgrammableSpeaker

**Description:** An entity that makes sounds that can be controlled by circuit network signals.

**Draftsman Source:** [ProgrammableSpeaker class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/programmable_speaker.py)

**Prototypes:** `"programmable-speaker"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("programmable-speaker", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `alert_icon` | String (signal name, e.g. `"signal-A"`) | None |  |
| `alert_message` | Complex (see draftsman docs) ⚠️ | "" |  |
| `allow_polyphony` | Boolean (`0` or `1`) | 0 | `1` |
| `circuit_enabled` | Boolean (`0` or `1`) | 0 | `1` |
| `instrument_id` | Integer | 0 |  |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `note_id` | Integer | 0 |  |
| `playback_mode` | One of: `"local"`, `"surface"`, `"global"` ([PlaybackMode](#playbackmode)) | "local" | `"local"` |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `show_alert` | Boolean (`0` or `1`) | 0 | `1` |
| `show_alert_on_map` | Boolean (`0` or `1`) | 1 | `1` |
| `signal_value_is_pitch` | Boolean (`0` or `1`) | 0 | `1` |
| `stop_playing_sounds` | Boolean (`0` or `1`) | 0 | `1` |
| `volume` | Number | 1.0 |  |
| `volume_controlled_by_signal` | Boolean (`0` or `1`) | 0 | `1` |
| `volume_signal` | String (signal name, e.g. `"signal-A"`) | None | `"signal-A"` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |

---

### Car

**Description:** (Factorio 2.0)

**Draftsman Source:** [Car class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/car.py)

**Prototypes:** `"car"`, `"tank"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `ammo_inventory` | Complex (see draftsman docs) ⚠️ | (factory) |  |
| `driver_is_main_gunner` | Boolean (`0` or `1`) | 0 | `1` |
| `enable_logistics_while_moving` | Boolean (`0` or `1`) | 1 | `1` |
| `equipment` | List (complex) ⚠️ | (factory) |  |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `orientation` | Complex (see draftsman docs) ⚠️ | <Orientation.NORTH: 0.0> |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `selected_gun_index` | Integer | 0 |  |
| `trunk_inventory` | Complex (see draftsman docs) ⚠️ | (factory) |  |

---

### SpiderVehicle

**Description:** (Factorio 2.0)

**Draftsman Source:** [SpiderVehicle class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/spider_vehicle.py)

**Prototypes:** `"spidertron"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `ammo_inventory` | Complex (see draftsman docs) ⚠️ | (factory) |  |
| `auto_target_with_gunner` | Boolean (`0` or `1`) | 0 | `1` |
| `auto_target_without_gunner` | Boolean (`0` or `1`) | 1 | `1` |
| `color` | Object `{r, g, b}` (0-255 each) | (factory) | `{r: 255, g: 0, b: 0}` |
| `driver_is_main_gunner` | Boolean (`0` or `1`) | 0 | `1` |
| `enable_logistics_while_moving` | Boolean (`0` or `1`) | 1 | `1` |
| `equipment` | List (complex) ⚠️ | (factory) |  |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `request_from_buffers` | Boolean (`0` or `1`) | 1 | `1` |
| `requests_enabled` | Boolean (`0` or `1`) | 1 | `1` |
| `sections` | List (complex) ⚠️ | (factory) |  |
| `selected_gun_index` | Integer | 0 |  |
| `trash_not_requested` | Boolean (`0` or `1`) | 0 | `1` |
| `trunk_inventory` | Complex (see draftsman docs) ⚠️ | (factory) |  |

---

### HeatPipe

**Description:** An entity used to transfer thermal energy.

**Draftsman Source:** [HeatPipe class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/heat_pipe.py)

**Prototypes:** `"heat-pipe"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### HeatInterface

**Description:** An entity that interacts with a heat network.

**Draftsman Source:** [HeatInterface class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/heat_interface.py)

**Prototypes:** `"heat-interface"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `mode` | One of: `"at-least"`, `"at-most"`, `"exactly"`, `"add"`, `"remove"` ([InfinityMode](#infinitymode)) | "at-least" | `"at-least"` |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `temperature` | Number | 0.0 |  |

---

## Other Entities

Additional entities not in the main categories:

### AgriculturalTower

**Description:** (Factorio 2.0)

**Draftsman Source:** [AgriculturalTower class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/agricultural_tower.py)

**Prototypes:** `"agricultural-tower"`

**Connection Type:** Single circuit connection

#### Reading Entity Output

Use `entity.output` to read: **Circuit network signals**

```facto
Entity e = place("agricultural-tower", 0, 0);
Bundle signals = e.output;  # Returns all output signals
```

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `circuit_enabled` | Boolean (`0` or `1`) | 0 | `1` |
| `connect_to_logistic_network` | Boolean (`0` or `1`) | 0 | `1` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `read_contents` | Boolean (`0` or `1`) | 0 | `1` |

#### Signal Configuration

Properties for configuring which signals the entity uses:

| Property | Type | Description |
|----------|------|-------------|
| `circuit_condition` | Condition (use `.enable = expr`) | The circuit condition that must be passed in order for this ... |
| `logistic_condition` | Condition (use `.enable = expr`) | The logistic condition that must be passed in order for this... |

---

### BurnerGenerator

**Description:** A electrical generator that turns burnable fuel directly into electrical energy.

**Draftsman Source:** [BurnerGenerator class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/burner_generator.py)

**Prototypes:** `"burner-generator"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### CurvedRailA

**Description:** Curved rails which connect straight rails to half-diagonal rails.

**Draftsman Source:** [CurvedRailA class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/curved_rail_a.py)

**Prototypes:** `"curved-rail-a"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### CurvedRailB

**Description:** Curved rails which connect half-diagonal rails to diagonal rails.

**Draftsman Source:** [CurvedRailB class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/curved_rail_b.py)

**Prototypes:** `"curved-rail-b"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### ElectricEnergyInterface

**Description:** An entity that interfaces with an electrical grid.

**Draftsman Source:** [ElectricEnergyInterface class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/electric_energy_interface.py)

**Prototypes:** `"electric-energy-interface"`, `"hidden-electric-energy-interface"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `buffer_size` | Number | (factory) |  |
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `power_production` | Number | (factory) |  |
| `power_usage` | Number | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### ElevatedCurvedRailA

**Description:** (Factorio 2.0)

**Draftsman Source:** [ElevatedCurvedRailA class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/elevated_curved_rail_a.py)

**Prototypes:** `"dummy-elevated-curved-rail-a"`, `"elevated-curved-rail-a"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### ElevatedCurvedRailB

**Description:** (Factorio 2.0)

**Draftsman Source:** [ElevatedCurvedRailB class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/elevated_curved_rail_b.py)

**Prototypes:** `"dummy-elevated-curved-rail-b"`, `"elevated-curved-rail-b"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### ElevatedHalfDiagonalRail

**Description:** (Factorio 2.0)

**Draftsman Source:** [ElevatedHalfDiagonalRail class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/elevated_half_diagonal_rail.py)

**Prototypes:** `"dummy-elevated-half-diagonal-rail"`, `"elevated-half-diagonal-rail"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### ElevatedStraightRail

**Description:** (Factorio 2.0)

**Draftsman Source:** [ElevatedStraightRail class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/elevated_straight_rail.py)

**Prototypes:** `"dummy-elevated-straight-rail"`, `"elevated-straight-rail"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### FusionGenerator

**Description:** (Factorio 2.0)

**Draftsman Source:** [FusionGenerator class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/fusion_generator.py)

**Prototypes:** `"fusion-generator"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### FusionReactor

**Description:** (Factorio 2.0)

**Draftsman Source:** [FusionReactor class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/fusion_reactor.py)

**Prototypes:** `"fusion-reactor"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### HalfDiagonalRail

**Description:** (Factorio 2.0)

**Draftsman Source:** [HalfDiagonalRail class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/half_diagonal_rail.py)

**Prototypes:** `"half-diagonal-rail"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### InfinityContainer

**Description:** An entity used to create an infinite amount of any item.

**Draftsman Source:** [InfinityContainer class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/infinity_container.py)

**Prototypes:** `"infinity-chest"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `bar` | Integer | None |  |
| `filters` | List (complex) ⚠️ | (factory) |  |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `remove_unfiltered_items` | Boolean (`0` or `1`) | 0 | `1` |

---

### InfinityPipe

**Description:** An entity used to create an infinite amount of any fluid at any temperature.

**Draftsman Source:** [InfinityPipe class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/infinity_pipe.py)

**Prototypes:** `"infinity-pipe"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `fluid_name` | Condition (use `.enable = expr`) ⚠️ | None |  |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `mode` | One of: `"at-least"`, `"at-most"`, `"exactly"`, `"add"`, `"remove"` ([InfinityMode](#infinitymode)) | "at-least" | `"at-least"` |
| `name` | String (entity prototype name) | (factory) |  |
| `percentage` | Number | 0.0 |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `temperature` | Integer | 0 |  |

---

### LegacyCurvedRail

**Description:** An old, Factorio 1.0 curved rail entity.

**Draftsman Source:** [LegacyCurvedRail class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/legacy_curved_rail.py)

**Prototypes:** `"legacy-curved-rail"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### LegacyStraightRail

**Description:** An old, Factorio 1.0 straight rail entity.

**Draftsman Source:** [LegacyStraightRail class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/legacy_straight_rail.py)

**Prototypes:** `"legacy-straight-rail"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### LightningAttractor

**Description:** (Factorio 2.0)

**Draftsman Source:** [LightningAttractor class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/lightning_attractor.py)

**Prototypes:** `"lightning-rod"`, `"fulgoran-ruin-attractor"`, `"lightning-collector"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### LinkedContainer

**Description:** An entity that allows sharing it's contents with any other ``LinkedContainer`` with the same ``link_id``.

**Draftsman Source:** [LinkedContainer class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/linked_container.py)

**Prototypes:** `"linked-chest"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `bar` | Integer | None |  |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `link_id` | Integer | 0 |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### PlayerPort

**Description:** A constructable respawn point typically used in scenarios.

**Draftsman Source:** [PlayerPort class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/player_port.py)

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### RailRamp

**Description:** (Factorio 2.0)

**Draftsman Source:** [RailRamp class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/rail_ramp.py)

**Prototypes:** `"rail-ramp"`, `"dummy-rail-ramp"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### RailSupport

**Description:** (Factorio 2.0)

**Draftsman Source:** [RailSupport class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/rail_support.py)

**Prototypes:** `"dummy-rail-support"`, `"rail-support"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### SimpleEntityWithForce

**Description:** A generic entity associated with friends or foes.

**Draftsman Source:** [SimpleEntityWithForce class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/simple_entity_with_force.py)

**Prototypes:** `"simple-entity-with-force"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `variation` | Integer | 1 |  |

---

### SimpleEntityWithOwner

**Description:** A generic entity owned by some other entity.

**Draftsman Source:** [SimpleEntityWithOwner class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/simple_entity_with_owner.py)

**Prototypes:** `"simple-entity-with-owner"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |
| `variation` | Integer | 1 |  |

---

### StraightRail

**Description:** A piece of rail track that moves in the 8 cardinal directions.

**Draftsman Source:** [StraightRail class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/straight_rail.py)

**Prototypes:** `"straight-rail"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### UndergroundPipe

**Description:** A pipe that transports fluids underneath other entities.

**Draftsman Source:** [UndergroundPipe class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/underground_pipe.py)

**Prototypes:** `"pipe-to-ground"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---

### Valve

**Description:** A pipe that may or may not admit fluid to pass through it based on some threshold.

**Draftsman Source:** [Valve class](https://github.com/redruin1/factorio-draftsman/blob/main/draftsman/prototypes/valve.py)

**Prototypes:** `"top-up-valve"`, `"overflow-valve"`, `"one-way-valve"`

**Connection Type:** Single circuit connection

#### Settable Properties

Set at placement: `place("name", x, y, {prop: value})`

| Property | Type | Default | Example |
|----------|------|---------|---------|
| `direction` | Integer ([Direction](#direction)) | 0 | `0  # NORTH` |
| `item_requests` | List (complex) ⚠️ | (factory) |  |
| `name` | String (entity prototype name) | (factory) |  |
| `quality` | One of: `"normal"`, `"uncommon"`, `"rare"`, `"epic"`, `"legendary"`, `"quality-unknown"` ([QualityID](#qualityid)) | "normal" | `"normal"` |

---
