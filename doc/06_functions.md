# Functions and Modules

As your circuits grow more complex, you'll want to reuse common patterns. Functions let you define reusable circuit templates, and imports let you organize code across multiple files.

## Defining Functions

Use the `func` keyword to define a function:

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

Functions can take three types of parameters:

| Type | Description | Example |
|------|-------------|---------|
| `Signal` | A circuit signal value | `Signal x` |
| `int` | A plain integer constant | `int threshold` |
| `Entity` | A reference to a placed entity | `Entity lamp` |

**Note:** `Memory` cannot be a parameter type – memory is stateful and doesn't work with function inlining.

### Return Values

Functions can return signals or entities:

```facto
# Returns a Signal
func clamp(Signal value, int min_val, int max_val) {
    Signal result = (value < min_val) * min_val
                  + (value > max_val) * max_val
                  + ((value >= min_val) && (value <= max_val)) * value;
    return result;
}

# Returns an Entity
func make_lamp(int x, int y) {
    Entity lamp = place("small-lamp", x, y);
    return lamp;
}
```

## How Functions Work: Inlining

**Important:** Functions in Facto are **always inlined** at their call sites. They're not "called" in the traditional sense – their code is copied and substituted wherever they're used.

This means:

```facto
func double(Signal x) {
    return x * 2;
}

Signal a = double(10);  # Becomes: Signal a = 10 * 2;
Signal b = double(20);  # Becomes: Signal b = 20 * 2;
```

Each call site gets its own copy of the function's logic. This is great for circuit generation but means:
- Recursive functions are not supported
- Each call creates its own combinators
- Memory declared in functions creates separate instances per call

## Common Function Patterns

### Clamping Values

Restrict a value to a range:

```facto
func clamp(Signal value, int min_val, int max_val) {
    Signal too_low = value < min_val;
    Signal too_high = value > max_val;
    Signal in_range = (!too_low) && (!too_high);
    
    return too_low * min_val 
         + too_high * max_val 
         + in_range * value;
}

Signal raw_input = ("signal-input", 0);
Signal safe_input = clamp(raw_input, 0, 100);  # Keep in 0-100 range
```

### Signal Selection (If-Then-Else)

Choose between two values based on a condition:

```facto
func select(Signal condition, Signal if_true, Signal if_false) {
    return (condition != 0) * if_true + (condition == 0) * if_false;
}

Signal temperature = ("signal-T", 0);
Signal threshold = 100;
Signal status = select(temperature > threshold, 1, 0);
```

### Absolute Value

```facto
func abs(Signal value) {
    Signal is_negative = value < 0;
    return is_negative * (0 - value) + (!is_negative) * value;
}

Signal diff = value1 - value2;
Signal distance = abs(diff);
```

### Type-Preserving Functions with `.type`

Use `.type` to write functions that preserve input signal types:

```facto
func add_offset(Signal value, int offset) {
    # Create offset with same type as value
    Signal typed_offset = offset | value.type;
    return value + typed_offset;
}

Signal iron = ("iron-plate", 100);
Signal more_iron = add_offset(iron, 50);  # Result is still iron-plate
```

This is particularly useful for generic utility functions that should work with any signal type.

### Sign Function

Returns -1, 0, or 1 based on sign:

```facto
func sign(Signal value) {
    Signal positive = (value > 0) * 1;
    Signal negative = (value < 0) * -1;
    return positive + negative;  # 0 when value == 0
}
```

### Minimum and Maximum

```facto
func min(Signal a, Signal b) {
    Signal a_smaller = a < b;
    return a_smaller * a + (!a_smaller) * b;
}

func max(Signal a, Signal b) {
    Signal a_larger = a > b;
    return a_larger * a + (!a_larger) * b;
}

Signal low = min(sensor1, sensor2);
Signal high = max(sensor1, sensor2);
```

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

# Create a row of colored lamps
Entity red_lamp = place_colored_lamp(0, 0, 255, 0, 0);
Entity green_lamp = place_colored_lamp(2, 0, 0, 255, 0);
Entity blue_lamp = place_colored_lamp(4, 0, 0, 0, 255);
```

### Configuring Existing Entities

Functions can also modify entities passed as parameters:

```facto
func configure_status_lamp(Entity lamp, Signal status) {
    lamp.r = (status == 0) * 255;   # Red when 0
    lamp.g = (status == 1) * 255;   # Green when 1
    lamp.b = (status == 2) * 255;   # Blue when 2
}

Entity lamp1 = place("small-lamp", 0, 0, {use_colors: 1, always_on: 1, color_mode: 1});
Entity lamp2 = place("small-lamp", 2, 0, {use_colors: 1, always_on: 1, color_mode: 1});

configure_status_lamp(lamp1, machine1_status);
configure_status_lamp(lamp2, machine2_status);
```

## Memory in Functions

Functions can declare local memory, but remember: **each call creates a separate instance**.

```facto
func counter() {
    Memory count: "signal-A";
    count.write(count.read() + 1);
    return count.read();
}

Signal counter1 = counter();  # Independent counter
Signal counter2 = counter();  # Another independent counter
```

This is useful when you want multiple independent instances of the same stateful logic.

### When This Is Useful

Creating multiple independent counters:

```facto
func make_blinker(int period) {
    Memory tick: "signal-T";
    tick.write((tick.read() + 1) % period);
    return tick.read() < (period / 2);
}

# Two blinkers with different periods
Signal fast_blink = make_blinker(10);   # Fast blinking
Signal slow_blink = make_blinker(60);   # Slow blinking

Entity lamp1 = place("small-lamp", 0, 0);
Entity lamp2 = place("small-lamp", 2, 0);
lamp1.enable = fast_blink;
lamp2.enable = slow_blink;
```

### When to Be Careful

If you want **shared** state across multiple uses, don't put memory in a function:

```facto
# WRONG - each call has separate memory
func get_shared_counter() {
    Memory shared: "signal-A";
    return shared.read();
}
# counter1 and counter2 read from DIFFERENT memories!
Signal counter1 = get_shared_counter();
Signal counter2 = get_shared_counter();

# RIGHT - declare memory once, outside the function
Memory shared_counter: "signal-A";
shared_counter.write(shared_counter.read() + 1);

Signal counter1 = shared_counter.read();  # Same memory
Signal counter2 = shared_counter.read();  # Same memory
```

## Type Coercion in Parameters

Parameters accept compatible types:

```facto
func process(Signal s, int n) {
    return s + n;
}

Signal sig = ("iron-plate", 100);
int num = 50;

# All of these work:
Signal r1 = process(sig, num);   # Signal, int
Signal r2 = process(100, 50);    # int → Signal, int
Signal r3 = process(sig, sig);   # Signal, Signal → int
```

The compiler automatically coerces types as needed.

## Local Variables

Variables declared inside functions are local to that function:

```facto
func calculate(Signal input) {
    Signal doubled = input * 2;      # Local variable
    Signal adjusted = doubled + 10;  # Another local
    return adjusted;
}

# doubled and adjusted don't exist outside the function
Signal result = calculate(50);
```

This helps prevent naming conflicts and keeps your code organized.

## Importing Modules

For larger projects, split your code across multiple files and use `import`:

```facto
import "path/to/other_file.facto";
```

### How Imports Work

Imports use **textual inclusion** (like C's `#include`). The imported file's content is inserted at the import location before parsing.

**main.facto:**
```facto
import "utils.facto";

Signal input = ("signal-input", 0);
Signal safe_input = clamp(input, 0, 100);  # Uses function from utils
```

**utils.facto:**
```facto
func clamp(Signal value, int min_val, int max_val) {
    Signal too_low = value < min_val;
    Signal too_high = value > max_val;
    Signal in_range = (!too_low) && (!too_high);
    return too_low * min_val + too_high * max_val + in_range * value;
}

func abs(Signal value) {
    Signal is_negative = value < 0;
    return is_negative * (0 - value) + (!is_negative) * value;
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
import "a.facto";  # Would create a cycle - skipped
```

The second import is silently skipped to break the cycle.

### Standard Library

The sample programs include a `stdlib/` directory with reusable utilities. You can create your own library of common functions:

```facto
import "stdlib/math.facto";
import "stdlib/display.facto";
```

## Function Best Practices

### Keep Functions Focused

Each function should do one thing well:

```facto
# Good - focused functions
func is_in_range(Signal value, int min_val, int max_val) {
    return (value >= min_val) && (value <= max_val);
}

func clamp(Signal value, int min_val, int max_val) {
    Signal below = value < min_val;
    Signal above = value > max_val;
    return below * min_val + above * max_val + is_in_range(value, min_val, max_val) * value;
}
```

### Use Descriptive Names

```facto
# Good
func calculate_production_rate(Signal items, Signal ticks) { ... }

# Not so good
func calc(Signal a, Signal b) { ... }
```

### Document Complex Functions

```facto
# Smooth filter: averages the last N samples
# Parameters:
#   input - current sample value
#   window - number of samples to average (memory created per call)
# Returns: smoothed value
func smooth_filter(Signal input, int window) {
    Memory sum: "signal-sum";
    Memory count: "signal-count";
    # ... implementation
}
```

### Avoid Deep Nesting

Instead of deeply nested function calls, use intermediate variables:

```facto
# Hard to read
Signal result = process(transform(filter(clamp(input, 0, 100), threshold)));

# Better
Signal clamped = clamp(input, 0, 100);
Signal filtered = filter(clamped, threshold);
Signal transformed = transform(filtered);
Signal result = process(transformed);
```

## Practical Example: Display Library

Create a reusable display library for common patterns:

**display_lib.facto:**
```facto
# Creates a bar graph display using lamps
# Returns the leftmost lamp entity
func make_bar_graph(int x, int y, int width, Signal value, int max_val) {
    # For a full implementation, you'd need to place multiple lamps
    # This is simplified to show the concept
    Entity lamp = place("small-lamp", x, y);
    lamp.enable = value > 0;
    return lamp;
}

# Creates a status indicator that changes color
# green = OK, yellow = warning, red = error
func make_status_indicator(int x, int y, Signal status) {
    Entity lamp = place("small-lamp", x, y, {
        use_colors: 1,
        always_on: 1,
        color_mode: 1
    });
    
    Signal is_ok = status == 0;
    Signal is_warning = status == 1;
    Signal is_error = status == 2;
    
    lamp.r = (is_warning * 255) + (is_error * 255);
    lamp.g = (is_ok * 255) + (is_warning * 255);
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

## Practical Example: Signal Processing

Common signal processing functions:

**signal_processing.facto:**
```facto
# Deadband filter - ignores small changes
func deadband(Signal input, int threshold) {
    Memory last_output: "signal-L";
    Signal diff = input - last_output.read();
    Signal abs_diff = (diff < 0) * (0 - diff) + (diff >= 0) * diff;
    
    # Only update if change exceeds threshold
    Signal should_update = abs_diff > threshold;
    Signal new_output = should_update * input + (!should_update) * last_output.read();
    last_output.write(new_output);
    
    return new_output;
}

# Rate limiter - limits how fast a value can change
func rate_limit(Signal input, int max_change) {
    Memory last: "signal-L";
    Signal current = last.read();
    Signal diff = input - current;
    
    # Clamp the change
    Signal clamped_up = (diff > max_change) * max_change;
    Signal clamped_down = (diff < -max_change) * -max_change;
    Signal clamped_same = ((diff >= -max_change) && (diff <= max_change)) * diff;
    Signal clamped_diff = clamped_up + clamped_down + clamped_same;
    
    Signal new_value = current + clamped_diff;
    last.write(new_value);
    
    return new_value;
}

# Hysteresis - different thresholds for on/off
func hysteresis(Signal input, int low_threshold, int high_threshold) {
    Memory state: "signal-S";
    Signal current_state = state.read();
    
    # Turn on when above high, off when below low
    Signal turn_on = (current_state == 0) && (input >= high_threshold);
    Signal turn_off = (current_state == 1) && (input < low_threshold);
    Signal stay_same = (!turn_on) && (!turn_off);
    
    Signal new_state = turn_on * 1 + stay_same * current_state;
    state.write(new_state);
    
    return new_state;
}
```

---

## Summary

- **Functions** are defined with `func name(Type param) { ... return value; }`
- Functions are **inlined** at call sites – no true "function calls"
- Parameter types: `Signal`, `int`, `Entity` (not `Memory`)
- Each function call creates its own combinator instances
- Memory in functions creates **independent instances** per call
- Use `import "file.facto"` to include other files
- Imports are textual – content is inserted at the import point
- Create libraries of reusable functions for common patterns

---

**← [Entities](05_entities.md)** | **[Advanced Concepts →](07_advanced_concepts.md)**
