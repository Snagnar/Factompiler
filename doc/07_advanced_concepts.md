# Advanced Concepts

This chapter covers optimization, design patterns, debugging techniques, and advanced features for building sophisticated circuits.

> **🔮 Looking for future features?** See [08_missing_features.md](08_missing_features.md) for planned features and current limitations.

---

## Understanding the Compiler

### The Compilation Pipeline

```
Source Code (.facto)
       │
       ▼
    Preprocessing     ─── Inline imports
       │
       ▼
    Parsing           ─── Create abstract syntax tree (AST)
       │
       ▼
    Semantic Analysis ─── Type checking, validation
       │
       ▼
    IR Generation     ─── Lower to intermediate representation
       │
       ▼
    Optimization      ─── CSE, constant folding, etc.
       │
       ▼
    Layout            ─── Position entities on the grid
       │
       ▼
    Wire Routing      ─── Connect entities with wires
       │
       ▼
    Emission          ─── Generate Factorio blueprint JSON
       │
       ▼
    Blueprint String
```

### Intermediate Representation (IR)

| IR Node | Factorio Entity | Purpose |
|---------|-----------------|---------|
| `IR_Const` | Constant Combinator | Output fixed signal values |
| `IR_Arith` | Arithmetic Combinator | Math operations |
| `IR_Decider` | Decider Combinator | Comparisons and conditionals |
| `IR_WireMerge` | (virtual) | Wire-only signal addition |
| `IR_MemCreate` | Decider pair | Memory cell (write-gated latch) |
| `IR_PlaceEntity` | Various | User-placed entities |

---

## Compiler Optimizations

### Common Subexpression Elimination (CSE)

Identical expressions are computed once:

```facto
Signal x = a + b;
Signal y = a + b;  # Reuses x's combinator
Signal z = a + b;  # Also reuses x's combinator
```

Only **one** arithmetic combinator is created.

### Condition Folding

Logical chains become single multi-condition deciders (Factorio 2.0):

```facto
Signal result = (a > 5) && (b < 10) && (c == 3);
# Before optimization: 2 deciders + 1 arithmetic
# After optimization: 1 multi-condition decider
```

**Applies when:**
- Two or more comparisons with same logical operator (`&&` or `||`)
- Simple operands (signals or constants)

**Doesn't apply:**
- Mixed `&&` and `||`: `(a > 0) && (b > 0) || (c > 0)`
- Complex operands: `((a + b) > 0) && (c > 0)`

### Constant Folding

Arithmetic in literals is evaluated at compile time:

```facto
Signal step = ("signal-A", 5 * 2 - 3);  # Becomes ("signal-A", 7)
int threshold = 100 / 4 + 25;            # Becomes 50
```

### Wire Merge Optimization

Same-type signals from simple sources are wired together directly:

```facto
Signal iron_a = ("iron-plate", 100);
Signal iron_b = ("iron-plate", 200);
Signal iron_c = ("iron-plate", 50);

Signal total = iron_a + iron_b + iron_c;  # Just wires, no arithmetic!
```

**Applies when:**
- All operands are "simple" (constants, entity reads, other wire merges)
- Same signal type
- Operator is `+`
- No operand appears twice

### Projection Elimination

Same-type projections are removed:

```facto
Signal iron = ("iron-plate", 100);
Signal same = iron | "iron-plate";  # No combinator created
```

### Memory Optimization

Unconditional memory writes with feedback become optimized arithmetic loops:

```facto
Memory counter: "signal-A";
counter.write(counter.read() + 1);
# Single arithmetic combinator with self-feedback
```

### Entity Property Inlining

Simple comparisons in `enable` are inlined:

```facto
lamp.enable = count > 10;
# Lamp's circuit condition set to "count > 10"
# No decider combinator created
```

---

## Optimization Patterns: Conditional Values

The `:` operator (conditional value) is more efficient than multiplication for conditional logic:

| Pattern | Result | Efficiency |
|---------|--------|------------|
| `(cond) : value` | 1 decider (copy input) | ✓ Best |
| `(cond) * value` | 1 decider + 1 arithmetic | ✗ Slower |

### Converting Old Patterns

**Selection (if-then-else):**

```facto
# Old (multiplication) — 4 combinators
Signal result = (x > 0) * a + (x <= 0) * b;

# New (conditional values) — 2 combinators
Signal result = ((x > 0) : a) + ((x <= 0) : b);
```

**Clamping:**

```facto
# Old
Signal clamped = (x < min) * min + (x > max) * max + (x >= min && x <= max) * x;

# New  
Signal clamped = ((x < min) : min) + ((x > max) : max) + ((x >= min && x <= max) : x);
```

**State machines:**

```facto
# Old
Signal next = (state == 0 && start) * 1 + (state == 1 && stop) * 2 + (!change) * state;

# New
Signal next = ((state == 0 && start) : 1) + ((state == 1 && stop) : 2) + ((!change) : state);
```

---

## Wire Color Strategy

### General Rule

The compiler assigns colors to avoid conflicts:

```facto
Signal a = ("signal-A", 10);
Signal b = ("signal-A", 20);
Signal c = a * b;  # Both inputs are signal-A!
```

Compiler assigns:
- `a` → red wire to multiplier
- `b` → green wire to multiplier

### Memory Wire Colors

- **Red wires:** data signals and feedback
- **Green wires:** control signal (`signal-W`)

### Transitive Merge Conflicts

The compiler handles complex patterns like balanced loaders:

```facto
Entity c1 = place("steel-chest", 0, 0, {read_contents: 1});
Bundle total = {c1.output, c2.output, c3.output};
Bundle neg_avg = total / -3;

Bundle in1 = {neg_avg, c1.output};  # c1.output appears in TWO paths
```

The compiler detects that `c1.output` arrives via two paths and assigns different colors.

### Debugging Wire Issues

```bash
factompile program.facto --json | jq '.blueprint.entities[].connections'
```

---

## Wire Distance and Relay Poles

Circuit wires have a **maximum span of 9 tiles**. The compiler auto-inserts relay poles.

### Power Pole Options

```bash
factompile program.facto --power-poles medium
```

Options: `small`, `medium`, `big`, `substation`

---

## Design Patterns

### Edge Detection

Detect signal changes (pulse on transition):

```facto
Memory previous: "signal-A";
Signal current = input_signal;
Signal changed = current != previous.read();
previous.write(current);
```

**Rising edge only:**

```facto
Memory previous: "signal-A";
Signal current = input_signal;
Signal rising = (current > 0) && (previous.read() == 0);
previous.write(current);
```

### Debouncing

Require signal to be stable before accepting:

```facto
Memory stable_count: "signal-C";
Memory last_input: "signal-L";
Memory output: "signal-O";

Signal input = ("signal-input", 0);
int stability_threshold = 3;

Signal input_changed = input != last_input.read();
Signal count = ((input_changed != 0) : 0) + ((input_changed == 0) : (stable_count.read() + 1));
stable_count.write(count);
last_input.write(input);

output.write(input, when=count >= stability_threshold);
Signal stable_output = output.read();
```

### State Machine

Implement a finite state machine using conditional values:

```facto
Memory state: "signal-S";
Signal current = state.read();

Signal start = ("signal-start", 0);
Signal stop = ("signal-stop", 0);
Signal reset = ("signal-reset", 0);

# States: 0=IDLE, 1=RUNNING, 2=STOPPED

# Transitions using conditional values
Signal to_running = (current == 0) && (start > 0);
Signal to_stopped = (current == 1) && (stop > 0);
Signal to_idle = (current == 2) && (reset > 0);
Signal stay = (!to_running) && (!to_stopped) && (!to_idle);

Signal next = ((to_running) : 1) + ((to_stopped) : 2) + ((stay) : current);
state.write(next);

# Outputs
Signal is_idle = current == 0;
Signal is_running = current == 1;
Signal is_stopped = current == 2;
```

### Priority Encoder

Select highest-priority active input:

```facto
Signal priority0 = ("signal-0", 0);  # Highest
Signal priority1 = ("signal-1", 0);
Signal priority2 = ("signal-2", 0);
Signal priority3 = ("signal-3", 0);  # Lowest

Signal selected = 
    ((priority0 > 0) : 0) +
    ((priority0 == 0 && priority1 > 0) : 1) +
    ((priority0 == 0 && priority1 == 0 && priority2 > 0) : 2) +
    ((priority0 == 0 && priority1 == 0 && priority2 == 0 && priority3 > 0) : 3);
```

### Multiplexer

Choose input based on selector:

```facto
Signal selector = ("signal-select", 0);
Signal input0 = ("signal-A", 0);
Signal input1 = ("signal-B", 0);
Signal input2 = ("signal-C", 0);
Signal input3 = ("signal-D", 0);

Signal output = ((selector == 0) : input0)
              + ((selector == 1) : input1)
              + ((selector == 2) : input2)
              + ((selector == 3) : input3);
```

### Timer

Count time and trigger actions:

```facto
Memory ticks: "signal-T";
int duration = 60;  # 1 second at 60 UPS

Signal running = ("signal-run", 0);
Signal count = ticks.read();

ticks.write((running > 0) : (count + 1), when=running > 0);
ticks.write(0, when=running == 0);

Signal elapsed = count >= duration;
Signal progress = (count * 100) / duration;
```

### Pulse Generator

Regular pulses:

```facto
Memory counter: "signal-A";
int period = 60;
int pulse_width = 5;

counter.write((counter.read() + 1) % period);
Signal pulse = counter.read() < pulse_width;
```

---

## Bundle Patterns

### Resource Monitoring

```facto
Bundle resources = { 
    ("iron-plate", 0),
    ("copper-plate", 0),
    ("coal", 0),
    ("steel-plate", 0)
};

Signal anyLow = any(resources) < 100;
Signal allStocked = all(resources) > 500;

Entity warningLamp = place("small-lamp", 0, 0);
warningLamp.enable = anyLow > 0;
```

### Parallel Scaling

```facto
Bundle sensors = { 
    ("signal-1", 0),
    ("signal-2", 0),
    ("signal-3", 0)
};

Bundle normalized = (sensors * 100) / 1000;
Signal anyHigh = any(normalized) > 80;
```

### Bundle Selection

```facto
Bundle data = { ("signal-A", 100), ("signal-B", 200), ("signal-C", 50) };

Signal a = data["signal-A"];
Signal b = data["signal-B"];
Signal c = data["signal-C"];

Signal weighted = (a * 3 + b * 2 + c * 1) / 6;
```

---

## For Loop Patterns

### LED Bar Graph

```facto
Signal level = ("signal-L", 0);

for i in 0..8 {
    Entity lamp = place("small-lamp", i, 0);
    lamp.enable = level > i;
}
```

### Binary Display

```facto
Memory value: "signal-V";
value.write((value.read() + 1) % 256);

for bit in 0..8 {
    Entity lamp = place("small-lamp", bit, 0);
    lamp.enable = ((value.read() >> bit) AND 1) > 0;
}
```

### Grid Placement

```facto
for row in 0..4 {
    for col in 0..4 {
        Entity lamp = place("small-lamp", col * 2, row * 2);
        lamp.enable = active > 0;
    }
}
```

### Sequencer

```facto
Memory step: "signal-S";
step.write((step.read() + 1) % 16);
Signal current = step.read();

for i in 0..16 {
    Entity lamp = place("small-lamp", i, 0);
    lamp.enable = current == i;
}
```

---

## Debugging Techniques

### Debug Logging

```bash
factompile program.facto --log-level debug
```

Shows: parse tree, type inference, IR nodes, optimization passes, layout decisions.

### JSON Inspection

```bash
factompile program.facto --json > blueprint.json
jq '.blueprint.entities[] | {name, connections}' blueprint.json
```

### Disable Optimizations

```bash
factompile program.facto --no-optimize
```

Useful for understanding what each expression creates.

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| "Undefined variable 'x'" | Variable not declared | Check spelling |
| "Type mismatch" | Wrong signal type for memory | Use projection: `value \| "type"` |
| "Signal 'signal-W' is reserved" | Using internal signal | Choose different signal |
| "Mixed signal types" | Different types in operation | Align types with projection |

---

## Performance Considerations

### Minimize Combinator Count

1. **Reuse expressions** — CSE handles automatically
2. **Use wire merge** — Add constants directly
3. **Use conditional values** — More efficient than multiplication

### Tick Delay Awareness

Each combinator adds 1 tick delay:

```facto
Signal result = ((a + b) * c) / d;
# tick 1: a + b
# tick 2: * c
# tick 3: / d
```

For real-time control, minimize expression depth.

### Memory vs. Computation

Cache expensive calculations:

```facto
Memory cached: "signal-C";
cached.write(expensive_expression, when=inputs_changed);
lamp.enable = cached.read() > threshold;
```

---

## Advanced Examples

### Shift Register

```facto
Memory stage0: "signal-A";
Memory stage1: "signal-A";
Memory stage2: "signal-A";
Memory stage3: "signal-A";

Signal input = ("signal-input", 0);
Signal clock = ("signal-clock", 0);

stage3.write(stage2.read(), when=clock > 0);
stage2.write(stage1.read(), when=clock > 0);
stage1.write(stage0.read(), when=clock > 0);
stage0.write(input, when=clock > 0);
```

### Ring Counter

```facto
Memory position: "signal-A";
int num_positions = 8;

position.write((position.read() + 1) % num_positions);

for i in 0..8 {
    Entity lamp = place("small-lamp", i, 0);
    lamp.enable = position.read() == i;
}
```

### PID Controller

```facto
Signal setpoint = ("signal-setpoint", 100);
Signal actual = ("signal-actual", 0);

int kp = 10;  # Proportional
int ki = 1;   # Integral
int kd = 5;   # Derivative

Signal error = setpoint - actual;

Memory integral: "signal-I";
integral.write(integral.read() + error);

Memory prev_error: "signal-P";
Signal derivative = error - prev_error.read();
prev_error.write(error);

Signal p_term = error * kp;
Signal i_term = integral.read() * ki;
Signal d_term = derivative * kd;

Signal output = (p_term + i_term + d_term) / 100;
```

### Binary to BCD

```facto
Signal binary = ("signal-input", 0);  # 0-999

Signal ones = binary % 10;
Signal tens = (binary / 10) % 10;
Signal hundreds = (binary / 100) % 10;

Signal digit_ones = ones | "signal-1";
Signal digit_tens = tens | "signal-2";
Signal digit_hundreds = hundreds | "signal-3";
```

---

## Integration with Factorio

### Input Signals

1. Place the blueprint
2. Wire from your factory to circuit's input area
3. Match signal types to what code expects

### Output Signals

1. Find output combinators (usually at bottom)
2. Wire to factory equipment

### Power Requirements

Use `--power-poles` to auto-add poles:

```bash
factompile program.facto --power-poles medium
```

### Testing in Creative Mode

1. Import blueprint
2. Use constant combinators for inputs
3. Check lamp states for outputs
4. Use circuit overlay (Alt mode) to see wires

---

## Summary

| Topic | Key Point |
|-------|-----------|
| **Optimizations** | CSE, condition folding, wire merge — automatic |
| **Conditional values** | Use `:` instead of `*` for efficiency |
| **Wire colors** | Compiler prevents conflicts automatically |
| **Design patterns** | Edge detection, state machines, timers, muxes |
| **Bundle patterns** | Parallel operations with `any()`/`all()` |
| **For loops** | Entity generation, displays, sequencers |
| **Debugging** | `--log-level debug`, `--json`, `--no-optimize` |
| **Performance** | Minimize combinators, watch tick delay |

---

## Reference Documentation

- **[Language Specification](../LANGUAGE_SPEC.md)** — Complete syntax and semantics
- **[Entity Reference](ENTITY_REFERENCE.md)** — All entities and properties
- **[Library Reference](LIBRARY_REFERENCE.md)** — Standard library functions
- **[Missing Features](08_missing_features.md)** — Planned features and current limitations

---

**[← Functions](06_functions.md)** | **[Back to Quick Start](02_quick_start.md)**
