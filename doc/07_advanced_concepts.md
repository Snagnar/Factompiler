# Advanced Concepts

This chapter covers optimization, design patterns, debugging techniques, and advanced features for building sophisticated circuits.

## Understanding the Compiler

### The Compilation Pipeline

Your code passes through several stages:

```
Source Code (.facto)
       │
       ▼
    ┌──────────────────┐
    │  Preprocessing   │ ─── Inline imports
    └────────┬─────────┘
             ▼
    ┌──────────────────┐
    │  Parsing         │ ─── Create abstract syntax tree (AST)
    └────────┬─────────┘
             ▼
    ┌──────────────────┐
    │  Semantic        │ ─── Type checking, validation
    │  Analysis        │
    └────────┬─────────┘
             ▼
    ┌──────────────────┐
    │  IR Generation   │ ─── Lower to intermediate representation
    └────────┬─────────┘
             ▼
    ┌──────────────────┐
    │  Optimization    │ ─── CSE, constant folding, etc.
    └────────┬─────────┘
             ▼
    ┌──────────────────┐
    │  Layout          │ ─── Position entities on the grid
    └────────┬─────────┘
             ▼
    ┌──────────────────┐
    │  Wire Routing    │ ─── Connect entities with wires
    └────────┬─────────┘
             ▼
    ┌──────────────────┐
    │  Emission        │ ─── Generate Factorio blueprint JSON
    └────────┬─────────┘
             ▼
    Blueprint String
```

Understanding this helps you predict what the compiler will generate and debug issues.

### Intermediate Representation (IR)

The compiler converts your code to IR nodes representing Factorio entities:

| IR Node | Factorio Entity | Purpose |
|---------|-----------------|---------|
| `IR_Const` | Constant Combinator | Output fixed signal values |
| `IR_Arith` | Arithmetic Combinator | Math operations (+, -, *, /, %) |
| `IR_Decider` | Decider Combinator | Comparisons and conditionals |
| `IR_WireMerge` | (virtual) | Wire-only signal addition |
| `IR_MemCreate` | Decider pair (write-gated latch) | Memory cell |
| `IR_PlaceEntity` | Various | User-placed entities |

## Compiler Optimizations

### Common Subexpression Elimination (CSE)

The compiler detects identical expressions and reuses them:

```facto
Signal x = a + b;
Signal y = a + b;  # Reuses x's combinator
Signal z = a + b;  # Also reuses x's combinator
```

Only **one** arithmetic combinator is created, and all three variables reference its output.

**When CSE applies:**
- Expressions must be structurally identical
- Same operands, same operator
- Applied during IR generation

### Condition Folding

Logical chains of comparisons are folded into single multi-condition decider combinators:

```facto
Signal a = ("signal-A", 0);
Signal b = ("signal-B", 0);
Signal c = ("signal-C", 0);

# Before optimization: 2 deciders + 1 arithmetic combinator
# After optimization: 1 multi-condition decider
Signal result = (a > 5) && (b < 10) && (c == 3);
```

This takes advantage of Factorio 2.0's multi-condition decider combinators.

**When condition folding applies:**
- Two or more comparisons chained with `&&` (AND)
- Two or more comparisons chained with `||` (OR)
- Comparison operands are "simple" (signals or constants)
- All comparisons use the same logical operator (all AND or all OR)

**When it doesn't apply:**
- Mixed `&&` and `||`: `(a > 0) && (b > 0) || (c > 0)`
- Complex operands: `((a + b) > 0) && (c > 0)`
- Non-comparison operands: `a && b` (booleanizes a and b separately)

### Constant Folding


Arithmetic in literals is evaluated at compile time:

```facto
Signal step = ("signal-A", 5 * 2 - 3);  # Becomes ("signal-A", 7)
int threshold = 100 / 4 + 25;            # Becomes 50
```

No combinators are created for these calculations – they're resolved before the circuit is built.

### Wire Merge Optimization

When adding signals of the same type that come from "simple" sources, the compiler skips creating arithmetic combinators:

```facto
Signal iron_a = ("iron-plate", 100);  # Constant combinator
Signal iron_b = ("iron-plate", 200);  # Constant combinator
Signal iron_c = ("iron-plate", 50);   # Constant combinator

Signal total = iron_a + iron_b + iron_c;  # Just wires, no arithmetic combinator!
```

The signals are wired together directly, and Factorio's circuit network sums them automatically.

**Wire merge applies when:**
- All operands are "simple" (constants, entity reads, or other wire merges)
- All operands have the same signal type
- Operator is `+`
- No operand appears twice (e.g., `a + a` still needs a combinator)

### Projection Elimination

Same-type projections are removed:

```facto
Signal iron = ("iron-plate", 100);
Signal same = iron | "iron-plate";  # No combinator, same is just iron
```

### Memory Optimization

For unconditional memory writes with feedback, the compiler generates optimized arithmetic loops:

```facto
# This:
Memory counter: "signal-A";
counter.write(counter.read() + 1);

# Becomes a single arithmetic combinator with self-feedback
# Instead of the two-decider latch
```

**Multi-step optimization:**

```facto
Memory state: "signal-A";
Signal s1 = state.read() + 1;
Signal s2 = s1 * 3;
Signal s3 = s2 % 100;
state.write(s3);
```

The compiler creates a chain of arithmetic combinators with feedback from the last back to the first, rather than a write-gated latch.

### Entity Property Inlining

Simple comparisons in `enable` assignments are inlined:

```facto
lamp.enable = count > 10;
# The lamp's circuit condition is set to "signal-A > 10"
# No decider combinator created
```

## Wire Color Strategy

Factorio has two wire colors: red and green. Understanding how the compiler uses them helps you debug.

### General Rule

The wire router assigns colors to avoid conflicts when multiple sources produce the same signal type:

```facto
Signal a = ("signal-A", 10);
Signal b = ("signal-A", 20);
Signal c = a * b;  # Multiplication combinator
```

Both `a` and `b` are `signal-A`. If they arrived at the multiplier on the same wire color, they'd sum to 30. The compiler assigns:
- `a` → red wire to multiplier
- `b` → green wire to multiplier

The multiplier sees both values separately.

### Memory Wire Colors

For memory circuits:
- **Red wires** carry data signals and feedback
- **Green wires** carry the control signal (`signal-W`)

This separation is critical for correct latch operation.

### Transitive Merge Conflicts

A more complex wire conflict occurs in patterns like the **MadZuri balanced loader**, where the same source participates in multiple merges that eventually connect to the same sink.

**The Pattern:**

```facto
# 6 chests storing items
Bundle chests = {c1.output, c2.output, c3.output, c4.output, c5.output, c6.output};

# Total of all chests → combinator computes negative average
Signal total = chests;
Bundle neg_avg = total / -6;

# Each inserter sees: its own chest + negative average
# Inserter 1's input: c1.output + neg_avg
Bundle in1 = {neg_avg, c1.output};
# ... similar for inserters 2-6
```

**The Problem:**

The signal from `c1.output` arrives at inserter1 via **two paths**:
1. **Direct path:** `c1.output` → inserter1 (individual chest content)
2. **Indirect path:** `c1.output` → combinator → `neg_avg` → inserter1 (via the average)

If both paths used the same wire color, `c1.output` would be **double-counted** at the inserter.

**The Solution:**

The compiler detects these **transitive conflicts** by tracking:
- Which sources participate in which merges (merge membership)
- Which merge sinks become sources for other merges (transitive paths)

When a source appears in multiple merges with a transitive relationship, the compiler assigns different wire colors:
- `c1.output` → combinator: **RED wire** (contributes to average)
- `c1.output` → inserter1: **GREEN wire** (individual content)

This ensures each chest's content is counted exactly once in the inserter's comparison.

### Debugging Wire Issues

If your circuit behaves unexpectedly, check for:
1. **Signal type collisions** – Two signals of the same type on the same wire
2. **Unintended summation** – Values adding when they shouldn't
3. **Missing connections** – A wire didn't get connected

Use `--json` to inspect the raw blueprint and see wire connections:

```bash
factompile program.facto --json | jq '.blueprint.entities[].connections'
```

## Wire Distance and Relay Poles

Factorio circuit wires have a **maximum span of 9 tiles**. The compiler automatically inserts relay poles for longer distances.

### Automatic Relay Insertion

If you place entities far apart:

```facto
Entity lamp1 = place("small-lamp", 0, 0);
Entity lamp2 = place("small-lamp", 50, 0);

Signal x = ("signal-A", 100);
lamp1.enable = x > 50;
lamp2.enable = x > 50;
```

The compiler inserts medium electric poles between them to relay the wire connections.

### Manual Power Pole Placement

You can also add power poles explicitly:

```facto
Entity pole1 = place("medium-electric-pole", 0, 5);
Entity pole2 = place("medium-electric-pole", 9, 5);
Entity pole3 = place("medium-electric-pole", 18, 5);
```

### Power Pole Options

Add power poles to the entire blueprint:

```bash
factompile program.facto --power-poles medium
```

Options: `small`, `medium`, `big`, `substation`

## Design Patterns

### Edge Detection

Detect when a signal changes (pulse on transition):

```facto
Memory previous: "signal-A";
Signal current = input_signal;
Signal changed = current != previous.read();
previous.write(current);

# changed is 1 for one tick when input changes
```

**Rising edge only:**

```facto
Memory previous: "signal-A";
Signal current = input_signal;
Signal rising = (current > 0) && (previous.read() == 0);
previous.write(current);
```

### Debouncing

Ignore rapid toggles (require signal to be stable):

```facto
Memory stable_count: "signal-C";
Memory last_input: "signal-L";
Memory output: "signal-O";

Signal input = ("signal-input", 0);
int stability_threshold = 3;

Signal input_changed = input != last_input.read();
Signal count = input_changed * 0 + (!input_changed) * (stable_count.read() + 1);
stable_count.write(count);
last_input.write(input);

# Only update output when input has been stable
Signal should_update = count >= stability_threshold;
output.write(input, when=should_update);

Signal stable_output = output.read();
```

### State Machine

Implement a finite state machine:

```facto
Memory state: "signal-S";
Signal current = state.read();

# Inputs
Signal start = ("signal-start", 0);
Signal stop = ("signal-stop", 0);
Signal reset = ("signal-reset", 0);

# States: 0=IDLE, 1=RUNNING, 2=STOPPED

# Transition logic
Signal to_running = (current == 0) && (start > 0);
Signal to_stopped = (current == 1) && (stop > 0);
Signal to_idle = (current == 2) && (reset > 0);
Signal stay = (!to_running) && (!to_stopped) && (!to_idle);

Signal next = to_running * 1 + to_stopped * 2 + stay * current;
state.write(next);

# Outputs based on state
Signal is_idle = current == 0;
Signal is_running = current == 1;
Signal is_stopped = current == 2;
```

> **[IMAGE PLACEHOLDER]**: State machine diagram showing IDLE → RUNNING → STOPPED → IDLE transitions.

### Priority Encoder

Select the highest-priority active input:

```facto
Signal priority0 = ("signal-0", 0);  # Highest priority
Signal priority1 = ("signal-1", 0);
Signal priority2 = ("signal-2", 0);
Signal priority3 = ("signal-3", 0);  # Lowest priority

# Output which priority is active (highest wins)
Signal selected = (priority0 > 0) * 0
                + ((priority0 == 0) && (priority1 > 0)) * 1
                + ((priority0 == 0) && (priority1 == 0) && (priority2 > 0)) * 2
                + ((priority0 == 0) && (priority1 == 0) && (priority2 == 0) && (priority3 > 0)) * 3;
```

### Multiplexer (Signal Selection)

Choose one of several inputs based on a selector:

```facto
Signal selector = ("signal-select", 0);  # 0, 1, 2, or 3
Signal input0 = ("signal-A", 0);
Signal input1 = ("signal-B", 0);
Signal input2 = ("signal-C", 0);
Signal input3 = ("signal-D", 0);

Signal output = (selector == 0) * input0
              + (selector == 1) * input1
              + (selector == 2) * input2
              + (selector == 3) * input3;
```

### Timer

Count time and trigger actions:

```facto
Memory ticks: "signal-T";
int duration = 60;  # 60 ticks = 1 second at 60 UPS

Signal running = ("signal-run", 0);
Signal count = ticks.read();

# Only count when running
ticks.write((running > 0) * (count + 1), when=running > 0);
ticks.write(0, when=running == 0);  # Reset when not running

Signal elapsed = count >= duration;
Signal progress = (count * 100) / duration;  # 0-100%
```

### Pulse Generator

Generate regular pulses:

```facto
Memory counter: "signal-A";
int period = 60;
int pulse_width = 5;

counter.write((counter.read() + 1) % period);
Signal pulse = counter.read() < pulse_width;
# pulse is 1 for 5 ticks, then 0 for 55 ticks
```

## Bundle Patterns

Bundles enable powerful parallel processing patterns using Factorio's "each" signal.

### Resource Monitoring

Monitor multiple resources with a single operation:

```facto
# Create a bundle of resource levels
Bundle resources = { 
    ("iron-plate", 0),    # Wire from chests
    ("copper-plate", 0),
    ("coal", 0),
    ("steel-plate", 0)
};

# Check all resources at once
Signal anyLow = any(resources) < 100;      # Warning if any resource is low
Signal allStocked = all(resources) > 500;  # All-clear if everything is stocked

# Create warning indicators
Entity warningLamp = place("small-lamp", 0, 0);
warningLamp.enable = anyLow > 0;

Entity allClearLamp = place("small-lamp", 2, 0);
allClearLamp.enable = allStocked > 0;
```

### Parallel Scaling

Apply the same transformation to multiple signals:

```facto
# Input signals from sensors
Bundle sensors = { 
    ("signal-1", 0),
    ("signal-2", 0),
    ("signal-3", 0),
    ("signal-4", 0)
};

# Normalize to percentage (assuming max value is 1000)
Bundle normalized = (sensors * 100) / 1000;

# Apply threshold
Signal anyHigh = any(normalized) > 80;
```

### Bundle with Selection

Extract and process individual signals from a bundle:

```facto
Bundle data = { ("signal-A", 100), ("signal-B", 200), ("signal-C", 50) };

# Get specific values
Signal a = data["signal-A"];
Signal b = data["signal-B"];
Signal c = data["signal-C"];

# Compute weighted average
Signal weighted = (a * 3 + b * 2 + c * 1) / 6;
```

## For Loop Patterns

For loops enable powerful entity generation and repetitive logic.

### LED Bar Graph

Create a bar graph display:

```facto
Signal level = ("signal-L", 0);  # Input: 0-7

for i in 0..8 {
    Entity lamp = place("small-lamp", i, 0);
    lamp.enable = level > i;  # Light up if level exceeds position
}
```

### Binary Display

Show a number in binary:

```facto
Memory value: "signal-V";
value.write((value.read() + 1) % 256);  # 8-bit counter

for bit in 0..8 {
    Entity lamp = place("small-lamp", bit, 0);
    # Extract each bit
    lamp.enable = ((value.read() >> bit) AND 1) > 0;
}
```

### Grid Placement

Create a 2D grid using nested positioning:

```facto
Signal active = ("signal-A", 1);

# 4x4 grid of lamps
for row in 0..4 {
    for col in 0..4 {
        Entity lamp = place("small-lamp", col * 2, row * 2);
        lamp.enable = active > 0;
    }
}
```

### Sequencer

Create a step sequencer using loops:

```facto
Memory step: "signal-S";
step.write((step.read() + 1) % 16);
Signal current = step.read();

# 16-step indicator
for i in 0..16 {
    Entity lamp = place("small-lamp", i, 0);
    lamp.enable = current == i;
}

# Define pattern using list iteration
# Steps where output is active
for active_step in [0, 4, 8, 12] {
    # This creates lamps that light up on beat
    Entity beatLamp = place("small-lamp", active_step, 2);
    beatLamp.enable = current == active_step;
}
```

### Dynamic Type Propagation with `.type`

Use `.type` in loops to maintain type consistency:

```facto
Signal reference = ("iron-plate", 100);

for i in 0..4 {
    # Create signals with same type as reference
    Signal offset = (i * 10) | reference.type;
    Entity lamp = place("small-lamp", i, 0);
    lamp.enable = reference > offset;
}
```

## Debugging Techniques

### Using `--log-level debug`

See detailed compilation information:

```bash
factompile program.facto --log-level debug
```

This shows:
- Parse tree structure
- Type inference decisions
- IR nodes generated
- Optimization passes applied
- Layout decisions

### Using `--json`

Inspect the raw blueprint:

```bash
factompile program.facto --json > blueprint.json
```

Then examine with `jq` or a JSON viewer:

```bash
# List all entities
jq '.blueprint.entities[] | .name' blueprint.json

# See wire connections
jq '.blueprint.entities[] | {name, connections}' blueprint.json
```

### Using `--no-optimize`

See the unoptimized output:

```bash
factompile program.facto --no-optimize
```

Useful for understanding what combinators each expression creates.

### Common Error Messages

**"Undefined variable 'x'"**
- You're using a variable that hasn't been declared
- Check for typos in variable names

**"Type mismatch: Memory expects 'iron-plate' but write provides 'copper-plate'"**
- All writes to a memory must use the same signal type
- Use projection to convert: `mem.write(copper | "iron-plate")`

**"Signal 'signal-W' is reserved"**
- `signal-W` is used internally for memory
- Choose a different signal name

**"Mixed signal types in binary operation"**
- You're adding/multiplying signals of different types
- Use projection to align types, or acknowledge the warning

## Performance Considerations

### Minimize Combinator Count

Each combinator adds processing overhead. Reduce count by:

1. **Reuse expressions** – CSE handles this automatically
2. **Use wire merge** – Add constants directly when possible
3. **Simplify logic** – Combine conditions when possible

### Tick Delay Awareness

Each combinator adds 1 tick of delay. Deep expression chains create latency:

```facto
# This has 3 ticks of delay
Signal result = ((a + b) * c) / d;
# tick 1: a + b
# tick 2: * c
# tick 3: / d
```

For real-time control, minimize expression depth.

### Memory vs. Computation Trade-off

Sometimes storing a value is better than recomputing it:

```facto
# Recomputes expensive expression every tick
Signal expensive = ((a * b) + (c * d)) / ((e * f) - (g * h));
lamp.enable = expensive > threshold;

# If expensive changes rarely, cache it
Memory cached: "signal-C";
cached.write(((a * b) + (c * d)) / ((e * f) - (g * h)), when=inputs_changed);
lamp.enable = cached.read() > threshold;
```

## Advanced Examples

### Shift Register

Pass values through a chain of memory cells:

```facto
Memory stage0: "signal-A";
Memory stage1: "signal-A";
Memory stage2: "signal-A";
Memory stage3: "signal-A";

Signal input = ("signal-input", 0);
Signal clock = ("signal-clock", 0);  # Pulse to shift

# Shift on clock pulse
stage3.write(stage2.read(), when=clock > 0);
stage2.write(stage1.read(), when=clock > 0);
stage1.write(stage0.read(), when=clock > 0);
stage0.write(input, when=clock > 0);

# Outputs
Signal out0 = stage0.read();
Signal out1 = stage1.read();
Signal out2 = stage2.read();
Signal out3 = stage3.read();
```

### Ring Counter

One-hot counter that cycles through positions:

```facto
Memory position: "signal-A";
int num_positions = 8;

# Increment position each tick, wrapping
position.write((position.read() + 1) % num_positions);

# One-hot outputs
Signal pos0 = position.read() == 0;
Signal pos1 = position.read() == 1;
Signal pos2 = position.read() == 2;
Signal pos3 = position.read() == 3;
Signal pos4 = position.read() == 4;
Signal pos5 = position.read() == 5;
Signal pos6 = position.read() == 6;
Signal pos7 = position.read() == 7;
```

### PID Controller (Simplified)

A basic proportional-integral-derivative controller:

```facto
Signal setpoint = ("signal-setpoint", 100);
Signal actual = ("signal-actual", 0);  # Wire from sensor

# Tuning constants
int kp = 10;  # Proportional gain
int ki = 1;   # Integral gain
int kd = 5;   # Derivative gain

# Error
Signal error = setpoint - actual;

# Integral (accumulated error)
Memory integral: "signal-I";
integral.write(integral.read() + error);

# Derivative (rate of change)
Memory prev_error: "signal-P";
Signal derivative = error - prev_error.read();
prev_error.write(error);

# PID output
Signal p_term = error * kp;
Signal i_term = integral.read() * ki;
Signal d_term = derivative * kd;

Signal output = (p_term + i_term + d_term) / 100;
Signal control = output | "signal-control";
```

### Binary to BCD Converter

Convert binary to binary-coded decimal for display:

```facto
Signal binary = ("signal-input", 0);  # 0-999

# Extract digits using division and modulo
Signal ones = binary % 10;
Signal tens = (binary / 10) % 10;
Signal hundreds = (binary / 100) % 10;

# Output on separate signals
Signal digit_ones = ones | "signal-1";
Signal digit_tens = tens | "signal-2";
Signal digit_hundreds = hundreds | "signal-3";
```

## Integration with the Game

### Input Signals

To get signals into your compiled circuit:
1. Place the blueprint in your world
2. Connect a wire from your factory to the circuit's input area
3. Ensure the signal types match what your code expects

### Output Signals

To use signals from your circuit:
1. Find the output combinators (usually at the bottom of the blueprint)
2. Connect wires from them to your factory equipment

### Power Requirements

Compiled blueprints include combinators that need power:
- Ensure power poles reach all combinators
- Use `--power-poles` option to add poles automatically

### Testing in Creative Mode

1. Import your blueprint
2. Use constant combinators to simulate inputs
3. Check lamp states to verify outputs
4. Use the circuit network overlay (Alt mode) to see wire connections

---

## Summary

- **Understand the pipeline** to predict compiler behavior
- **Optimizations are automatic**: CSE, condition folding, constant folding, wire merge
- **Wire colors** prevent signal collisions; memory uses red for data, green for control
- **Design patterns**: edge detection, state machines, timers, multiplexers
- **Bundle patterns**: parallel scaling, resource monitoring, threshold checking with `any()`/`all()`
- **For loop patterns**: LED displays, binary outputs, grids, sequencers
- **Debug with**: `--log-level debug`, `--json`
- **Performance**: minimize combinators, be aware of tick delay

---

## Reference Documentation

For complete details, see:
- **[Language Specification](../LANGUAGE_SPEC.md)** – Complete syntax and semantics
- **[Entity Reference](ENTITY_REFERENCE.md)** – All entities and their properties

---

**← [Functions](06_functions.md)** | **[Back to Introduction](01_introduction.md)**
