# Quick Start Guide

Get your first Facto circuit running in under 5 minutes!

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install factompile
```

That's it! You're ready to compile your first program.

### Option 2: Install from Source

For contributors or those wanting the latest development version:

```bash
git clone https://github.com/Snagnar/Factompiler.git
cd Factompiler
pip install -e .
```

### Verify Installation

```bash
factompile --help
```

You should see:

```
Usage: factompile [OPTIONS] [INPUT_FILE]

  Compile Facto source files or strings to Factorio blueprint format.

Options:
  -i, --input TEXT                Compile from string instead of file
  -o, --output PATH               Output file for the blueprint
  --name TEXT                     Blueprint name
  --log-level [debug|info|warning|error]
  --no-optimize                   Disable IR optimizations
  --power-poles TEXT              Add power poles
  --json                          Output blueprint in JSON format
  --help                          Show this message and exit.
```

---

## Your First Program: A Blinking Lamp

Let's build a classic circuit — a lamp that blinks on and off.

### Step 1: Create the Source File

Create `blink.facto`:

```facto
# A simple blinking lamp

# Memory stores a value that persists across game ticks
Memory counter: "signal-A";

# Increment counter each tick, wrapping at 20
counter.write((counter.read() + 1) % 20);

# Lamp is ON for the first half of each cycle
Signal blink = counter.read() < 10;

# Place a lamp and connect it to our signal
Entity lamp = place("small-lamp", 0, 0);
lamp.enable = blink;
```

### Step 2: Compile

```bash
factompile blink.facto
```

The compiler outputs a blueprint string.

### Step 3: Import into Factorio

1. Copy the blueprint string from your terminal
2. In Factorio, press `B` to open the blueprint library
3. Click "Import string" and paste
4. Place the blueprint in your world

<table>
<tr>
<td width="50%">

**Your lamp is now blinking!**

The compiled circuit consists of:
- One constant combinator (the memory cell)
- One decider combinator (the comparison)
- One lamp (the output)

All automatically wired together.

</td>
<td>
<img src="img/placeholder_blink.gif" width="250" alt="Blinking lamp circuit in Factorio"/>
</td>
</tr>
</table>

### Understanding the Code

```facto
Memory counter: "signal-A";
```
Creates a **memory cell** — a circuit that remembers a value between ticks. The `"signal-A"` specifies what signal type to use.

```facto
counter.write((counter.read() + 1) % 20);
```
Each tick: read the current value, add 1, mod 20 (so it cycles 0→19→0...), and write it back.

```facto
Signal blink = counter.read() < 10;
```
Creates a signal that's `1` (true) when counter is 0-9, and `0` (false) when it's 10-19.

```facto
Entity lamp = place("small-lamp", 0, 0);
lamp.enable = blink;
```
Places a lamp at position (0,0) and wires it to the blink signal.

---

## Key Concept: Conditional Values

A frequently-used pattern in Facto is **conditional value syntax**:

```facto
Signal result = condition : value;
```

This outputs `value` when condition is true, `0` when false. It compiles to a decider combinator using its native "copy input" mode, avoiding the extra combinator that multiplication would require.

### Example: Clamping a Value

```facto
Signal speed = ("signal-S", 150);

# Limit speed to maximum of 100
Signal capped = (speed > 100) : 100;     # 100 if over limit, else 0
Signal passed = (speed <= 100) : speed;  # speed if within, else 0
Signal safe_speed = capped + passed;     # Combined: max 100
```

### Example: Selection (If-Then-Else)

```facto
Signal flag = ("signal-F", 1);
Signal value_a = ("signal-A", 100);
Signal value_b = ("signal-B", 200);

# Choose based on flag
Signal result = (flag > 0) : value_a + (flag == 0) : value_b;
```

You'll use this pattern constantly. It's the foundation of conditional logic in Facto.

---

## Saving to Files

Instead of copying from the terminal:

```bash
factompile blink.facto -o blink.blueprint
```

---

## Practical Example: Smart Power Backup

Here's a real-world circuit — backup steam power that activates when batteries are low and stays on until they're charged:

```facto
# Wire this signal from your accumulator
Signal battery = ("signal-A", 0);

# Latch: ON when battery < 20%, OFF when battery >= 80%
Memory steam_enabled: "signal-S";
steam_enabled.write(1, set=battery < 20, reset=battery >= 80);

# Control the power switch
Entity steam_switch = place("power-switch", 0, 0);
steam_switch.enable = steam_enabled.read() > 0;
```

This uses a **latch** — memory that toggles between states. The `set=` condition turns it on, `reset=` turns it off. The gap between 20% and 80% prevents flickering when power hovers near a threshold.

---

## Practical Example: Controlled Inserter

An inserter that only runs when there's enough in the source chest:

```facto
# This signal would come from reading a chest
Signal chest = ("iron-plate", 0);

# Only run when chest has > 100 items
Entity inserter = place("inserter", 0, 0);
inserter.enable = chest > 100;
```

---

## Practical Example: Priority Selection

Using conditional values to create a priority system:

```facto
Signal input = ("signal-I", 75);

# Output different values based on input ranges
Signal output = 
    (input >= 100) : 3 +    # High priority
    (input >= 50 && input < 100) : 2 +   # Medium priority
    (input < 50) : 1;       # Low priority
```

---

## Practical Example: Resource Monitor with Bundles

**Bundles** let you work with multiple signals at once. Here's a simple resource warning system:

```facto
# Bundle of resources (wire these from your storage)
Bundle resources = { 
    ("iron-plate", 0), 
    ("copper-plate", 0), 
    ("coal", 0) 
};

# Light warning lamp if ANY resource drops below 100
Signal any_low = any(resources) < 100;

Entity warning = place("small-lamp", 0, 0);
warning.enable = any_low > 0;
```

The `any()` function checks if *any* signal in the bundle meets the condition. There's also `all()` to check if *all* signals meet a condition.

---

## Compiler Options

### Quick Experiments Without Files
Test ideas instantly from the command line:
```bash
factompile -i 'Signal x = 42;'
factompile --input 'Memory m: "signal-A"; m.write(m.read() + 1);'
```

### Debug Mode
See what the compiler is doing:
```bash
factompile blink.facto --log-level debug
```

### Add Power Poles
```bash
factompile blink.facto --power-poles medium
```
Options: `small`, `medium`, `big`, `substation`

### Custom Blueprint Name
```bash
factompile blink.facto --name "My Awesome Blinker"
```

### Raw JSON Output
```bash
factompile blink.facto --json
```

---

## Common Questions

**What Factorio version is required?**
Facto targets Factorio 2.0+. Multi-condition deciders and other features require the latest version.

**My circuit doesn't work!**
Check: (1) nearby power pole, (2) external signals wired in, (3) give it a tick to initialize.

**What's the difference between `int` and `Signal`?**
- `int` is a compile-time constant — no combinator, just a number in calculations
- `Signal` is a Factorio signal — creates a constant combinator outputting a value

**"signal-W is reserved"?**
`signal-W` is used internally for memory operations. Use a different signal.

---

## Next Steps

Now that you understand the basics:

- **[Signals and Types](03_signals_and_types.md)** — The type system, bundles, and all operations
- **[Memory](04_memory.md)** — Counters, latches, state machines
- **[Entities](05_entities.md)** — Control lamps, inserters, trains, and more
- **[Functions](06_functions.md)** — Reusable components and imports
- **[Library Reference](LIBRARY_REFERENCE.md)** — Standard library functions

---

**[Signals and Types →](03_signals_and_types.md)**
