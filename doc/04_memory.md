# Working with Memory

Most Factorio circuits need to remember something: a counter, a previous state, a buffered value. Memory lets you store data that persists across game ticks.

## Understanding Memory in Factorio

In Factorio's circuit network, signals exist only during the tick they're produced. If a combinator outputs "signal-A: 50" on tick 100, that signal is gone by tick 101 unless something preserves it.

Memory creates a **feedback loop** – a circuit that feeds its output back to its input, effectively remembering values indefinitely.

## Declaring Memory

Declare a memory cell with the `Memory` keyword:

```fcdsl
Memory counter: "signal-A";    # Explicit type
Memory buffer;                 # Type inferred from first write
```

### Explicit vs. Implicit Types

**Explicit types** (recommended) make your code clear:

```fcdsl
Memory counter: "signal-A";
Memory item_count: "iron-plate";
Memory state: "signal-S";
```

**Implicit types** are inferred from the first write:

```fcdsl
Memory buffer;  # Type unknown until first write

Signal iron = ("iron-plate", 50);
write(iron, buffer);  # Now buffer uses iron-plate
```

While convenient, implicit types can be confusing. We recommend always specifying the type.

## Reading and Writing Memory

### Reading: The `read()` Function

Read the current value stored in memory:

```fcdsl
Memory counter: "signal-A";

Signal current_value = read(counter);
```

`read()` returns a signal with the memory's type and its stored value.

### Writing: The `write()` Function

Store a new value in memory:

```fcdsl
write(value, memory_name, when=condition);
```

Parameters:
- **value** – The value to store (must match memory's signal type)
- **memory_name** – Which memory cell to write to
- **when** (optional) – Condition for when to write (default: always)

### Unconditional Writes

Write every tick:

```fcdsl
Memory counter: "signal-A";
write(read(counter) + 1, counter);  # Increment every tick
```

When `when` is omitted, the write happens every single tick.

### Conditional Writes

Write only when a condition is met:

```fcdsl
Memory buffer: "signal-A";
Signal trigger = ("signal-T", 0);  # Wire this from your factory

write(new_value, buffer, when=trigger > 0);
```

The value is written only when `trigger > 0`. Otherwise, the previous value is held.

## Common Memory Patterns

### Counter

The classic increment-every-tick counter:

```fcdsl
Memory counter: "signal-A";
write(read(counter) + 1, counter);
```

> **[IMAGE PLACEHOLDER]**: Screenshot of a simple counter circuit in Factorio, showing an arithmetic combinator with feedback.

**How it works:**
- Tick 0: counter is 0, writes 0+1=1
- Tick 1: counter is 1, writes 1+1=2
- Tick 2: counter is 2, writes 2+1=3
- ...and so on

### Wrapping Counter

Count up to a limit, then reset:

```fcdsl
Memory counter: "signal-A";
int max_value = 60;  # Reset after reaching 60

write((read(counter) + 1) % max_value, counter);
# Counts 0, 1, 2, ..., 59, 0, 1, 2, ...
```

Use this for:
- Blinking lights (cycle every N ticks)
- Timed sequences
- Animation frames

### Sample and Hold

Capture a value when triggered:

```fcdsl
Memory captured: "signal-A";
Signal input = ("signal-input", 0);     # External signal
Signal trigger = ("signal-trigger", 0); # Capture when > 0

write(input, captured, when=trigger > 0);

# Output the captured value
Signal output = read(captured);
```

When `trigger` is 0, the memory holds its previous value. When `trigger` becomes positive, it captures the current `input` value.

> **[IMAGE PLACEHOLDER]**: Diagram showing sample-and-hold behavior: input changing while output stays constant until trigger.

### Toggle (Flip-Flop)

Switch between 0 and 1:

```fcdsl
Memory toggle: "signal-A";
Signal button = ("signal-button", 0);  # Pulse input

# Toggle on button press
Signal new_state = 1 - read(toggle);  # 0→1 or 1→0
write(new_state, toggle, when=button > 0);

Signal output = read(toggle);
```

Press once to turn on, press again to turn off.

**Caution:** If `button` stays high for multiple ticks, the toggle will flip every tick. For real buttons, you typically need edge detection (see Advanced Concepts).

### Accumulator

Sum values over time:

```fcdsl
Memory total: "signal-A";
Signal incoming = ("iron-plate", 0);  # Items passing by

write(read(total) + incoming, total);

Signal running_total = read(total);
```

Each tick, the incoming value is added to the total.

### Conditional Accumulator

Only accumulate when a condition is met:

```fcdsl
Memory total: "iron-plate";
Signal incoming = ("iron-plate", 0);
Signal should_count = ("signal-enable", 0);

write(read(total) + incoming, total, when=should_count > 0);
```

### Maximum Tracker

Remember the highest value seen:

```fcdsl
Memory maximum: "signal-A";
Signal input = ("signal-input", 0);

Signal current_max = read(maximum);
Signal new_max = (input > current_max) * input 
               + (input <= current_max) * current_max;

write(new_max, maximum);
```

### Minimum Tracker (with initialization)

```fcdsl
Memory minimum: "signal-A";
Memory initialized: "signal-I";
Signal input = ("signal-input", 0);

Signal current_min = read(minimum);
Signal is_first = read(initialized) == 0;

# On first tick, use input as minimum
# After that, take smaller of current and input
Signal new_min = is_first * input
               + (!is_first) * ((input < current_min) * input
                              + (input >= current_min) * current_min);

write(new_min, minimum);
write(1, initialized, when=is_first);
```

## Multiple Memories

You can declare as many memory cells as you need:

```fcdsl
Memory counter: "signal-A";
Memory previous: "signal-B";
Memory accumulator: "signal-C";

# Counter increments
write(read(counter) + 1, counter);

# Previous holds the old counter value
write(read(counter), previous);

# Accumulator sums all counter values
write(read(accumulator) + read(counter), accumulator);
```

Each memory cell is independent and can hold different signal types.

## Memory Type Rules

### Rule: All Writes Must Match

Every write to a memory cell must use the same signal type:

```fcdsl
Memory buffer: "iron-plate";

Signal iron = ("iron-plate", 50);
Signal copper = ("copper-plate", 30);

write(iron, buffer);    # OK - types match
write(copper, buffer);  # ERROR - type mismatch!
```

If you need to store different types, use projection:

```fcdsl
Memory buffer: "iron-plate";

Signal copper = ("copper-plate", 30);
write(copper | "iron-plate", buffer);  # OK - projected to correct type
```

### Reserved Signal: signal-W

The signal `signal-W` is used internally for memory write-enable logic. You cannot use it in your code:

```fcdsl
Signal bad = ("signal-W", 5);  # COMPILE ERROR!
```

This is the only reserved signal – all others are available for your use.

## How Memory Works Under the Hood

Understanding the implementation helps with debugging and optimization.

### SR Latch Circuit

For conditional writes, the compiler generates an **SR latch** using two decider combinators:

```
                ┌─────────────────┐
Data ─────────► │  Write Gate    │ ──┬──► Output
                │  if W > 0      │   │
                └─────────────────┘   │
                                      │
                ┌─────────────────┐   │
                │  Hold Gate     │ ◄─┘
         ┌────► │  if W == 0     │ ──┬──► (feedback)
         │      └─────────────────┘   │
         │                            │
         └────────────────────────────┘
```

- **Write Gate**: Passes data when write-enable (signal-W) is positive
- **Hold Gate**: Recirculates its output when write-enable is zero
- **Wire Colors**: Data flows on red wires, control (signal-W) flows on green wires

> **[IMAGE PLACEHOLDER]**: Screenshot of an SR latch in Factorio showing the two decider combinators and their connections.

### Arithmetic Feedback Optimization

For unconditional writes (no `when` parameter), the compiler can optimize to a simpler form:

```fcdsl
Memory counter: "signal-A";
write(read(counter) + 1, counter);
```

This becomes a single arithmetic combinator with a feedback wire:

```
     ┌─────────────────────────────┐
     │                             │
     ▼                             │
┌─────────────────┐               │
│ Arithmetic (+1) │ ──────────────┘
└─────────────────┘
       │
       ▼
     Output
```

The combinator's output feeds back to its own input, creating a self-incrementing loop.

**This optimization applies when:**
- No `when` condition (or `when=1`)
- The write expression references the same memory being written

## Common Mistakes

### Mistake 1: Forgetting to Read

```fcdsl
# WRONG - this just stores 1 forever
Memory counter: "signal-A";
write(1, counter);
```

```fcdsl
# CORRECT - increment based on current value
Memory counter: "signal-A";
write(read(counter) + 1, counter);
```

### Mistake 2: Type Mismatch

```fcdsl
Memory buffer: "signal-A";
Signal value = ("signal-B", 50);  # Different type!

write(value, buffer);  # Warning or error
```

Fix with projection:
```fcdsl
write(value | "signal-A", buffer);  # OK
```

### Mistake 3: Using signal-W

```fcdsl
Memory write_count: "signal-W";  # ERROR - reserved!
```

Use a different signal:
```fcdsl
Memory write_count: "signal-V";  # OK
```

### Mistake 4: Complex Conditional Logic

```fcdsl
# This works but creates many combinators
Memory value: "signal-A";
write(some_complex_expression, value, when=(a > b) && (c < d) || (e == f));
```

Sometimes it's cleaner to pre-compute the condition:
```fcdsl
Signal should_write = (a > b) && (c < d) || (e == f);
write(some_complex_expression, value, when=should_write);
```

## Practical Example: Binary Clock

A clock that cycles through binary patterns:

```fcdsl
# Binary counter for 4 lamps (0-15)
Memory counter: "signal-A";
write((read(counter) + 1) % 16, counter);

Signal bits = read(counter);

# Extract individual bits
Signal bit0 = (bits >> 0) AND 1;  # Fastest (toggles every tick)
Signal bit1 = (bits >> 1) AND 1;
Signal bit2 = (bits >> 2) AND 1;
Signal bit3 = (bits >> 3) AND 1;  # Slowest (toggles every 8 ticks)

# Create 4 lamps showing binary count
Entity lamp0 = place("small-lamp", 0, 0);
Entity lamp1 = place("small-lamp", 2, 0);
Entity lamp2 = place("small-lamp", 4, 0);
Entity lamp3 = place("small-lamp", 6, 0);

lamp0.enable = bit0 > 0;
lamp1.enable = bit1 > 0;
lamp2.enable = bit2 > 0;
lamp3.enable = bit3 > 0;
```

> **[IMAGE PLACEHOLDER]**: Screenshot of the binary clock showing 4 lamps in a row, some lit, some dark, representing a binary number.

## Practical Example: Moving Average

Smooth out a noisy signal with a 4-sample moving average:

```fcdsl
# Store the last 4 values
Memory sample1: "signal-A";
Memory sample2: "signal-A";
Memory sample3: "signal-A";
Memory sample4: "signal-A";

Signal input = ("signal-input", 0);  # Wire from your sensor

# Shift samples (oldest falls off)
write(read(sample3), sample4);
write(read(sample2), sample3);
write(read(sample1), sample2);
write(input, sample1);

# Calculate average
Signal sum = read(sample1) + read(sample2) + read(sample3) + read(sample4);
Signal average = sum / 4;

Signal output = average | "signal-output";
```

---

## Summary

- **Memory** stores values that persist across ticks
- Declare with `Memory name: "type";`
- Use `read(memory)` to get the current value
- Use `write(value, memory, when=condition)` to store values
- All writes to a memory must use the same signal type
- The `signal-W` signal is reserved for internal use
- The compiler optimizes common patterns (counters, feedback loops)

---

**← [Signals and Types](03_signals_and_types.md)** | **[Entities →](05_entities.md)**
