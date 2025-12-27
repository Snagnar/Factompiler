# Quick Start Guide

Get your first Factompiler circuit running in under 5 minutes!

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Snagnar/Factompiler.git
cd Factompiler
```

### 2. Set Up the Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python compile.py --help
```

You should see the compiler's help output:

```
Usage: compile.py [OPTIONS] INPUT_FILE

  Compile Factorio Circuit DSL files to blueprint format.

Options:
  -o, --output PATH               Output file for the blueprint
  --strict                        Enable strict type checking
  --name TEXT                     Blueprint name
  --log-level [debug|info|warning|error]
  --no-optimize                   Disable IR optimizations
  --explain                       Add extended explanations to diagnostics
  --power-poles TEXT              Add power poles
  --json                          Output blueprint in JSON format
  --help                          Show this message and exit.
```

## Your First Program: A Blinking Lamp

Let's create a classic circuit – a lamp that blinks on and off every few ticks.

### Step 1: Create the Source File

Create a new file called `blink.fcdsl`:

```fcdsl
# blink.fcdsl - A simple blinking lamp

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
python compile.py blink.fcdsl
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

```fcdsl
Memory counter: "signal-A";
```
This creates a **memory cell** – a circuit that remembers a value between ticks. It stores a signal of type `signal-A`.

```fcdsl
counter.write((counter.read() + 1) % 20);
```
Every tick, this:
1. Reads the current value from `counter`
2. Adds 1 to it
3. Takes the remainder when divided by 20 (so it cycles 0→19→0→19...)
4. Writes the result back to `counter`

```fcdsl
Signal blink = counter.read() < 10;
```
Creates a signal that is `1` (true) when the counter is 0-9, and `0` (false) when it's 10-19.

```fcdsl
Entity lamp = place("small-lamp", 0, 0);
lamp.enable = blink;
```
Places a lamp at coordinates (0, 0) and connects its enable condition to our blink signal.

## Saving Blueprints to Files

Instead of copying from the terminal, you can save directly to a file:

```bash
python compile.py blink.fcdsl -o blink.blueprint
```

Then open `blink.blueprint` in any text editor to copy the string.

## Example: Simple Arithmetic

Let's try something with visible outputs – a circuit that does math:

```fcdsl
# arithmetic.fcdsl - Basic arithmetic operations

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
python compile.py arithmetic.fcdsl -o arithmetic.blueprint
```

> **[IMAGE PLACEHOLDER]**: Screenshot of the arithmetic circuit in Factorio, with a view showing the constant combinators and arithmetic combinators.

## Example: Controlled Inserter

Here's a practical example – an inserter that only runs when there are items to move:

```fcdsl
# controlled_inserter.fcdsl

# This signal comes from reading a chest (wire it in-game)
Signal chest_contents = ("iron-plate", 0);

# Only enable when chest has at least 100 items
Signal should_run = chest_contents > 100;

# Place and control the inserter
Entity inserter = place("inserter", 0, 0);
inserter.enable = should_run;
```

> **[IMAGE PLACEHOLDER]**: Screenshot showing an inserter connected to a chest via circuit wire.

## Useful Compiler Options

### See What's Happening (Debug Mode)

```bash
python compile.py blink.fcdsl --log-level debug
```

This shows you the compilation stages and what the compiler is doing.

### Strict Type Checking

```bash
python compile.py blink.fcdsl --strict
```

Turns all warnings into errors. Great for catching subtle bugs.

### Add Power Poles

```bash
python compile.py blink.fcdsl --power-poles medium
```

Automatically adds power poles to your blueprint. Options: `small`, `medium`, `big`, `substation`.

### Custom Blueprint Name

```bash
python compile.py blink.fcdsl --name "My Awesome Blinker"
```

The blueprint will have this name when imported.

### View as JSON

```bash
python compile.py blink.fcdsl --json
```

Outputs the raw blueprint JSON instead of the encoded string. Useful for debugging or integration with other tools.

---

## Frequently Asked Questions

### General Questions

**Q: What version of Factorio does this work with?**

A: Factompiler generates blueprints compatible with Factorio 2.0 and later. Some entity features (like selector combinators) are 2.0-specific.

**Q: Can I use this with mods?**

A: The compiler uses Factorio's base game entities. Modded entities are not directly supported, though you might be able to place them manually and wire them to compiled circuits.

**Q: What file extension should I use?**

A: We use `.fcdsl` (Factorio Circuit DSL), but any text file will work.

### Troubleshooting

**Q: I get "Unknown signal type" warnings**

A: Make sure you're using valid Factorio signal names. Common ones include:
- Virtual signals: `signal-A` through `signal-Z`, `signal-0` through `signal-9`
- Item signals: `iron-plate`, `copper-plate`, `electronic-circuit`, etc.
- Fluid signals: `water`, `petroleum-gas`, etc.

**Q: My circuit doesn't work when I paste it**

A: Check for:
1. Power connections – place a power pole nearby
2. Missing input signals – some examples assume you'll wire in external signals
3. Tick timing – some circuits need a moment to initialize

**Q: The compiler says "signal-W is reserved"**

A: The signal `signal-W` is used internally for memory write-enable logic. Use a different signal name for your code.

**Q: Why is my blueprint so large?**

A: The compiler generates all necessary combinators. Complex expressions create more combinators. You can use `--no-optimize` to see the unoptimized output, but the optimized version is usually smaller.

### Language Questions

**Q: Can I use loops?**

A: Yes! Factompiler supports `for` loops that are unrolled at compile time. This is perfect for creating arrays of entities:

```fcdsl
# Create 5 lamps in a row
for i in 0..5 {
    Entity lamp = place("small-lamp", i * 2, 0);
    lamp.enable = count > 0;
}
```

You can also use step values and list iteration:

```fcdsl
# Every other position
for j in 0..10 step 2 {
    Entity lamp = place("small-lamp", j, 0);
}

# Specific values
for x in [1, 3, 7, 15] {
    Entity lamp = place("small-lamp", x, 0);
}
```

See [Signals and Types](03_signals_and_types.md) for full for loop documentation.

**Q: What's the difference between `int` and `Signal`?**

A: 
- `int` is a plain compile-time number. It doesn't exist in the circuit.
- `Signal` is a Factorio signal that flows through circuit networks.

For example: `int count = 5` just means "5" wherever you use `count`. `Signal count = 5` creates a constant combinator outputting a signal with value 5.

**Q: How do I make a signal wait for another signal?**

A: Use memory with a conditional write:

```fcdsl
Memory buffer: "signal-A";
buffer.write(input_signal, when=trigger > 0);
Signal output = buffer.read();
```

**Q: Can I create multiple blueprints from one file?**

A: Not currently. Each file compiles to one blueprint.

---

## Next Steps

You've got the basics down! Here's where to go next:

- **[Signals and Types](03_signals_and_types.md)** – Understand the type system and signal routing
- **[Memory](04_memory.md)** – Build stateful circuits with counters, buffers, and state machines
- **[Entities](05_entities.md)** – Control all kinds of Factorio entities
- **[Functions](06_functions.md)** – Write reusable circuit components

---

**← [Introduction](01_introduction.md)** | **[Signals and Types →](03_signals_and_types.md)**
