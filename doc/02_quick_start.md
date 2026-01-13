# Quick Start Guide

Get your first Facto circuit running in under 5 minutes!

## Installation

### Option 1: Install from PyPI (Recommended)

The easiest way to get started is to install Factompile from PyPI:

```bash
pip install factompile
```

That's it! You're ready to compile your first program.

### Option 2: Install from Source

If you want to contribute or use the latest development version:

```bash
git clone https://github.com/Snagnar/Factompiler.git
cd Factompiler
pip install -e .
```

### Verify Installation

```bash
factompile --help
```

You should see the compiler's help output:

```
Usage: factompile [OPTIONS] INPUT_FILE

  Compile Facto source files to blueprint format.

Options:
  -o, --output PATH               Output file for the blueprint
  --name TEXT                     Blueprint name
  --log-level [debug|info|warning|error]
  --no-optimize                   Disable IR optimizations
  --power-poles TEXT              Add power poles
  --json                          Output blueprint in JSON format
  --help                          Show this message and exit.
```

## Your First Program: A Blinking Lamp

Let's create a classic circuit – a lamp that blinks on and off every few ticks.

### Step 1: Create the Source File

Create a new file called `blink.facto`:

```facto
# blink.facto - A simple blinking lamp

# Create a memory cell to count ticks
Memory counter: "signal-A";

# Increment the counter each tick, wrapping at 20
counter.write((counter.read() + 1) % 20);

# Lamp is ON when counter is less than 10 (first half of cycle)
Signal blink = counter.read() < 10;

# Place a lamp and control it with our blink signal
Entity lamp = place("small-lamp", 0, 0);
lamp.enable = blink;
```

### Step 2: Compile It

```bash
factompile blink.facto
```

The compiler outputs a base64-encoded blueprint string to your terminal.

### Step 3: Import into Factorio

1. Copy the entire blueprint string from your terminal
2. Open Factorio and enter your world
3. Press `B` to open the blueprint library
4. Click "Import string" 
5. Paste the blueprint and click "Import"
6. Place the blueprint in your world

> **[IMAGE PLACEHOLDER]**: Screenshot of the imported blinking lamp blueprint placed in Factorio, showing the combinator arrangement.

**Congratulations!** Your lamp is now blinking!

### Understanding What Happened

Let's break down what each line does:

```facto
Memory counter: "signal-A";
```
This creates a **memory cell** – a circuit that remembers a value between ticks. It stores a signal of type `signal-A`.

```facto
counter.write((counter.read() + 1) % 20);
```
Every tick, this:
1. Reads the current value from `counter`
2. Adds 1 to it
3. Takes the remainder when divided by 20 (so it cycles 0→19→0→19...)
4. Writes the result back to `counter`

```facto
Signal blink = counter.read() < 10;
```
Creates a signal that is `1` (true) when the counter is 0-9, and `0` (false) when it's 10-19.

```facto
Entity lamp = place("small-lamp", 0, 0);
lamp.enable = blink;
```
Places a lamp at coordinates (0, 0) and connects its enable condition to our blink signal.

## Saving Blueprints to Files

Instead of copying from the terminal, you can save directly to a file:

```bash
factompile blink.facto -o blink.blueprint
```

Then open `blink.blueprint` in any text editor to copy the string, or use the file directly with blueprint sharing tools.

## Example: Simple Arithmetic

Let's try something with visible outputs – a circuit that does math:

```facto
# arithmetic.facto - Basic arithmetic operations

# Input signals (you'd wire these from your factory)
Signal input_a = ("signal-A", 100);
Signal input_b = ("signal-B", 50);

# Arithmetic operations
Signal sum = input_a + input_b;           # 150
Signal difference = input_a - input_b;    # 50
Signal product = input_a * input_b;       # 5000
Signal quotient = input_a / input_b;      # 2

# Label outputs with meaningful signal types
Signal output_sum = sum | "signal-1";
Signal output_diff = difference | "signal-2";
Signal output_prod = product | "signal-3";
Signal output_quot = quotient | "signal-4";
```

Compile and import:

```bash
factompile arithmetic.facto -o arithmetic.blueprint
```

> **[IMAGE PLACEHOLDER]**: Screenshot of the arithmetic circuit in Factorio, with a view showing the constant combinators and arithmetic combinators.

## Example: Controlled Inserter

Here's a practical example – an inserter that only runs when there are items to move:

```facto
# controlled_inserter.facto

# This signal comes from reading a chest (wire it in-game)
Signal chest_contents = ("iron-plate", 0);

# Only enable when chest has at least 100 items
Signal should_run = chest_contents > 100;

# Place and control the inserter
Entity inserter = place("inserter", 0, 0);
inserter.enable = should_run;
```

> **[IMAGE PLACEHOLDER]**: Screenshot showing an inserter connected to a chest via circuit wire.

## Example: Smart Power Backup (Hysteresis)

One of the most practical circuits: backup power that turns on when accumulators are low, and stays on until they're full – without flickering on and off.

```facto
# steam_backup.facto

# Signal from accumulator (0-100%)
Signal battery = ("signal-A", 0);  # Wire from your accumulator

# Latch: turns ON when battery < 20%, OFF when battery >= 80%
Memory steam_enabled: "signal-S";
steam_enabled.write(1, set=battery < 20, reset=battery >= 80);

# Control a power switch
Entity steam_switch = place("power-switch", 0, 0);
steam_switch.enable = steam_enabled.read() > 0;
```

This uses a **latch** – a special kind of memory that remembers its on/off state. The `set=` condition turns it on, and `reset=` turns it off. The gap between 20% and 80% prevents flickering.

See [Memory](04_memory.md) for more on latches and hysteresis patterns.

## Useful Compiler Options

### See What's Happening (Debug Mode)

```bash
factompile blink.facto --log-level debug
```

This shows you the compilation stages, optimizations applied, and what the compiler is doing internally.

### Add Power Poles

```bash
factompile blink.facto --power-poles medium
```

Automatically adds power poles to your blueprint so the combinators have power. Options: `small`, `medium`, `big`, `substation`.

### Custom Blueprint Name

```bash
factompile blink.facto --name "My Awesome Blinker"
```

The blueprint will have this name when imported into Factorio's blueprint library.

### View as JSON

```bash
factompile blink.facto --json
```

Outputs the raw blueprint JSON instead of the encoded string. Useful for debugging, inspection, or integration with other tools.

---

## Common Questions

**What version of Factorio does this work with?**
Facto generates blueprints for Factorio 2.0 and later. The multi-condition decider combinators and other features require the latest version.

**What's the difference between `int` and `Signal`?**
- `int` is a compile-time constant – just a number, no combinator created
- `Signal` is a Factorio signal – creates a constant combinator outputting a value

**My circuit doesn't work when I paste it!**
Check: (1) power pole nearby, (2) external signals wired in, (3) give it a tick to initialize.

**The compiler says "signal-W is reserved"**
`signal-W` is used internally for memory. Use a different signal.

**Can I use loops?**
Yes! `for` loops are unrolled at compile time:

```facto
for i in 0..5 {
    Entity lamp = place("small-lamp", i * 2, 0);
    lamp.enable = count > 0;
}
```

---

## Next Steps

You've got the basics down! Here's where to go next:

- **[Signals and Types](03_signals_and_types.md)** – Understand the type system, bundles, and for loops
- **[Memory](04_memory.md)** – Counters, latches, hysteresis patterns
- **[Entities](05_entities.md)** – Control lamps, inserters, trains, and more
- **[Functions](06_functions.md)** – Reusable circuit components and imports

---

**← [Introduction](01_introduction.md)** | **[Signals and Types →](03_signals_and_types.md)**
