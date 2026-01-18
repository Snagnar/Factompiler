# Functions and Modules

As your circuits grow more complex, you'll want to reuse common patterns. Functions let you define reusable circuit templates, and imports let you organize code across multiple files.

---

## Defining Functions

Use `func` to define a function:

```facto
func function_name(Type1 param1, Type2 param2) {
    # Function body
    return expression;
}
```

### A Simple Example

```facto
func double(Signal x) {
    return x * 2;
}

Signal input = 50;
Signal result = double(input);  # result = 100
```

### Function Parameters

| Type | Description | Example |
|------|-------------|---------|
| `Signal` | A circuit signal value | `Signal x` |
| `int` | A plain integer constant | `int threshold` |
| `Entity` | A reference to a placed entity | `Entity lamp` |

**Note:** `Memory` cannot be a parameter type — memory is stateful and doesn't work with function inlining.

### Return Values

Functions can return signals or entities:

```facto
# Returns a Signal
func clamp(Signal value, int min_val, int max_val) {
    return (value < min_val) : min_val
         + (value > max_val) : max_val
         + (value >= min_val && value <= max_val) : value;
}

# Returns an Entity
func make_lamp(int x, int y) {
    Entity lamp = place("small-lamp", x, y);
    return lamp;
}
```

---

## How Functions Work: Inlining

**Important:** Functions in Facto are **always inlined** at their call sites. They're not "called" — their code is copied and substituted wherever used.

```facto
func double(Signal x) {
    return x * 2;
}

Signal a = double(10);  # Becomes: Signal a = 10 * 2;
Signal b = double(20);  # Becomes: Signal b = 20 * 2;
```

Each call site gets its own copy. This means:
- Recursive functions are not supported
- Each call creates its own combinators
- Memory in functions creates separate instances per call

---

## Common Function Patterns

### Clamping Values

Restrict a value to a range using conditional values:

```facto
func clamp(Signal value, int min_val, int max_val) {
    return (value < min_val) : min_val 
         + (value > max_val) : max_val 
         + (value >= min_val && value <= max_val) : value;
}

Signal raw_input = ("signal-input", 0);
Signal safe_input = clamp(raw_input, 0, 100);
```

### Selection (If-Then-Else)

Choose between values based on condition:

```facto
func select(Signal condition, Signal if_true, Signal if_false) {
    return (condition != 0) : if_true + (condition == 0) : if_false;
}

Signal temperature = ("signal-T", 0);
Signal status = select(temperature > 100, 1, 0);
```

### Absolute Value

```facto
func abs(Signal value) {
    return (value >= 0) : value + (value < 0) : (0 - value);
}

Signal diff = value1 - value2;
Signal distance = abs(diff);
```

### Sign Function

Returns -1, 0, or 1:

```facto
func sign(Signal value) {
    return (value > 0) : 1 + (value < 0) : (-1);
    # Returns 0 when value == 0 (both conditions false)
}
```

### Minimum and Maximum

```facto
func min(Signal a, Signal b) {
    return (a < b) : a + (a >= b) : b;
}

func max(Signal a, Signal b) {
    return (a > b) : a + (a <= b) : b;
}

Signal low = min(sensor1, sensor2);
Signal high = max(sensor1, sensor2);
```

### Type-Preserving Functions

Use `.type` for functions that preserve input signal types:

```facto
func add_offset(Signal value, int offset) {
    Signal typed_offset = offset | value.type;
    return value + typed_offset;
}

Signal iron = ("iron-plate", 100);
Signal more_iron = add_offset(iron, 50);  # Result is still iron-plate
```

---

## Entity Factory Functions

Functions can create and configure entities:

```facto
func place_colored_lamp(int x, int y, Signal r, Signal g, Signal b) {
    Entity lamp = place("small-lamp", x, y, {
        use_colors: 1,
        always_on: 1,
        color_mode: 1
    });
    lamp.r = r;
    lamp.g = g;
    lamp.b = b;
    return lamp;
}

Entity red_lamp = place_colored_lamp(0, 0, 255, 0, 0);
Entity green_lamp = place_colored_lamp(2, 0, 0, 255, 0);
Entity blue_lamp = place_colored_lamp(4, 0, 0, 0, 255);
```

### Configuring Existing Entities

```facto
func configure_status_lamp(Entity lamp, Signal status) {
    lamp.r = (status == 0) : 255;  # Red when 0
    lamp.g = (status == 1) : 255;  # Green when 1
    lamp.b = (status == 2) : 255;  # Blue when 2
}

Entity lamp1 = place("small-lamp", 0, 0, {use_colors: 1, always_on: 1, color_mode: 1});
configure_status_lamp(lamp1, machine1_status);
```

---

## Memory in Functions

Functions can declare local memory, but **each call creates a separate instance**:

```facto
func counter() {
    Memory count: "signal-A";
    count.write(count.read() + 1);
    return count.read();
}

Signal counter1 = counter();  # Independent counter
Signal counter2 = counter();  # Another independent counter
```

### When This Is Useful

Creating multiple independent timers:

```facto
func make_blinker(int period) {
    Memory tick: "signal-T";
    tick.write((tick.read() + 1) % period);
    return tick.read() < (period / 2);
}

Signal fast_blink = make_blinker(10);   # Fast
Signal slow_blink = make_blinker(60);   # Slow

Entity lamp1 = place("small-lamp", 0, 0);
Entity lamp2 = place("small-lamp", 2, 0);
lamp1.enable = fast_blink;
lamp2.enable = slow_blink;
```

### When to Be Careful

If you want **shared** state, declare memory outside the function:

```facto
# WRONG - each call has separate memory
func get_shared_counter() {
    Memory shared: "signal-A";
    return shared.read();
}

# RIGHT - declare memory once, outside
Memory shared_counter: "signal-A";
shared_counter.write(shared_counter.read() + 1);
Signal counter1 = shared_counter.read();
Signal counter2 = shared_counter.read();
```

---

## Local Variables

Variables inside functions are local:

```facto
func calculate(Signal input) {
    Signal doubled = input * 2;      # Local
    Signal adjusted = doubled + 10;  # Local
    return adjusted;
}

Signal result = calculate(50);
# doubled and adjusted don't exist outside
```

---

## Importing Modules

Split code across files with `import`:

```facto
import "path/to/other_file.facto";
```

### How Imports Work

Imports use **textual inclusion** (like C's `#include`). The imported file's content is inserted at the import location.

**main.facto:**
```facto
import "utils.facto";

Signal input = ("signal-input", 0);
Signal safe_input = clamp(input, 0, 100);  # From utils
```

**utils.facto:**
```facto
func clamp(Signal value, int min_val, int max_val) {
    return (value < min_val) : min_val
         + (value > max_val) : max_val
         + (value >= min_val && value <= max_val) : value;
}

func abs(Signal value) {
    return (value >= 0) : value + (value < 0) : (0 - value);
}
```

### Import Paths

Paths are relative to the file containing the import:

```
project/
  main.facto           # import "lib/utils.facto"
  lib/
    utils.facto        # ← imported file
```

### Circular Import Protection

The compiler detects and prevents circular imports:

**a.facto:**
```facto
import "b.facto";  # Imports b
```

**b.facto:**
```facto
import "a.facto";  # Would create cycle - skipped
```

### Standard Library

The `lib/` directory contains reusable utilities:

```facto
import "lib/math.facto";
import "lib/memory_patterns.facto";
```

See the [Library Reference](LIBRARY_REFERENCE.md) for all available functions.

---

## Best Practices

**Keep functions focused:**
```facto
func is_in_range(Signal value, int min_val, int max_val) {
    return (value >= min_val) && (value <= max_val);
}

func clamp(Signal value, int min_val, int max_val) {
    return (value < min_val) : min_val
         + (value > max_val) : max_val
         + (is_in_range(value, min_val, max_val)) : value;
}
```

**Use descriptive names:**
```facto
# Good
func calculate_production_rate(Signal items, Signal ticks) { ... }

# Not so good
func calc(Signal a, Signal b) { ... }
```

**Use conditional values (`:`) over multiplication:**
```facto
# Efficient — one decider combinator
func select(Signal cond, Signal a, Signal b) {
    return (cond != 0) : a + (cond == 0) : b;
}

# Less efficient — extra arithmetic
func select_old(Signal cond, Signal a, Signal b) {
    return (cond != 0) * a + (cond == 0) * b;
}
```

**Avoid deep nesting:**
```facto
# Hard to read
Signal result = process(transform(filter(clamp(input, 0, 100), threshold)));

# Better
Signal clamped = clamp(input, 0, 100);
Signal filtered = filter(clamped, threshold);
Signal result = process(transform(filtered));
```

---

## Practical Example: Display Library

**display_lib.facto:**
```facto
# Status indicator: green=OK, yellow=warning, red=error
func make_status_indicator(int x, int y, Signal status) {
    Entity lamp = place("small-lamp", x, y, {
        use_colors: 1,
        always_on: 1,
        color_mode: 1
    });
    
    # Using conditional values for color selection
    lamp.r = (status == 1) : 255 + (status == 2) : 255;  # yellow or red
    lamp.g = (status == 0) : 255 + (status == 1) : 255;  # green or yellow
    lamp.b = 0;
    
    return lamp;
}
```

**main.facto:**
```facto
import "display_lib.facto";

Signal system_status = ("signal-status", 0);
Entity indicator = make_status_indicator(0, 0, system_status);
```

---

## Practical Example: Signal Processing

**signal_processing.facto:**
```facto
# Deadband filter — ignores small changes
func deadband(Signal input, int threshold) {
    Memory last_output: "signal-L";
    Signal current = last_output.read();
    Signal diff = input - current;
    Signal abs_diff = (diff >= 0) : diff + (diff < 0) : (0 - diff);
    
    Signal should_update = abs_diff > threshold;
    Signal new_output = (should_update) : input + (!should_update) : current;
    last_output.write(new_output);
    
    return new_output;
}

# Rate limiter — limits how fast a value can change
func rate_limit(Signal input, int max_change) {
    Memory last: "signal-L";
    Signal current = last.read();
    Signal diff = input - current;
    
    # Clamp the change using conditional values
    Signal clamped_diff = (diff > max_change) : max_change
                        + (diff < -max_change) : (-max_change)
                        + (diff >= -max_change && diff <= max_change) : diff;
    
    Signal new_value = current + clamped_diff;
    last.write(new_value);
    
    return new_value;
}

# Hysteresis — different thresholds for on/off
func hysteresis(Signal input, int low_threshold, int high_threshold) {
    Memory state: "signal-S";
    state.write(1, 
        set=input >= high_threshold,
        reset=input < low_threshold
    );
    return state.read();
}
```

---

## Summary

| Concept | Syntax |
|---------|--------|
| Define function | `func name(Type param) { return value; }` |
| Parameter types | `Signal`, `int`, `Entity` (not `Memory`) |
| Import module | `import "path/to/file.facto";` |

**Key points:**
- Functions are **inlined** — no true function calls
- Each call creates its own combinator instances
- Memory in functions creates **separate instances** per call
- Use conditional values (`:`) for efficient conditional logic
- Use `.type` to preserve signal types in generic functions
- See [Library Reference](LIBRARY_REFERENCE.md) for standard library functions

---

**[← Entities](05_entities.md)** | **[Advanced Concepts →](07_advanced_concepts.md)**
