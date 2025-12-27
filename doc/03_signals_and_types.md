# Signals and Types

Signals are the foundation of everything in Factompiler. Understanding how they work will help you build effective circuits.

## What is a Signal?

In Factorio, signals are data that flows through circuit networks. Each signal has two parts:

1. **Type** – What kind of signal it is (e.g., `iron-plate`, `signal-A`)
2. **Value** – An integer count

When you read a circuit network, you see values like "iron-plate: 50, copper-plate: 30". That's two signals of different types with their respective values.

In Factompiler, we represent signals like this:

```fcdsl
Signal my_signal = ("iron-plate", 50);
```

This creates a signal of type `iron-plate` with value `50`.

## Signal Types in Factorio

Factorio has several categories of signals:

### Virtual Signals
These are abstract signals used purely for circuit logic:
- **Letters**: `signal-A` through `signal-Z`
- **Numbers**: `signal-0` through `signal-9`
- **Colors**: `signal-red`, `signal-green`, `signal-blue`, `signal-yellow`, `signal-cyan`, `signal-magenta`, `signal-white`, `signal-grey`, `signal-black`, `signal-pink`
- **Special**: `signal-everything`, `signal-anything`, `signal-each`

### Item Signals
Every item in Factorio is also a signal type:
- `iron-plate`, `copper-plate`, `steel-plate`
- `electronic-circuit`, `advanced-circuit`, `processing-unit`
- `iron-ore`, `copper-ore`, `coal`, `stone`
- And many more...

### Fluid Signals
Fluids can be used as signals too:
- `water`, `steam`
- `crude-oil`, `petroleum-gas`, `light-oil`, `heavy-oil`
- `lubricant`, `sulfuric-acid`

## Declaring Signals

### Explicit Types

The most common way to declare a signal with a specific type:

```fcdsl
Signal iron = ("iron-plate", 100);
Signal counter = ("signal-A", 0);
Signal temperature = ("steam", 500);
```

### Implicit Types (Auto-Allocation)

If you don't need a specific type, let the compiler pick one:

```fcdsl
Signal x = 5;    # Compiler assigns signal-A
Signal y = 10;   # Compiler assigns signal-B
Signal z = 15;   # Compiler assigns signal-C
```

The compiler automatically allocates virtual signals (`signal-A`, `signal-B`, etc.) in order. This is convenient for intermediate calculations where the specific type doesn't matter.

### The `int` Type

For pure compile-time constants that shouldn't become signals:

```fcdsl
int threshold = 100;
int multiplier = 5;
Signal result = input * multiplier;  # multiplier is just the number 5
```

The key difference:
- `Signal x = 5` creates a constant combinator outputting `signal-A: 5`
- `int x = 5` is just the number 5 embedded in expressions

Use `int` when you need a constant for calculations but don't want it to exist as a separate signal in your circuit.

## Arithmetic Operations

Signals support all standard arithmetic:

```fcdsl
Signal a = ("signal-A", 100);
Signal b = ("signal-B", 30);

Signal sum = a + b;           # Addition: 130
Signal difference = a - b;    # Subtraction: 70
Signal product = a * b;       # Multiplication: 3000
Signal quotient = a / b;      # Division: 3 (integer division)
Signal remainder = a % b;     # Modulo: 10
Signal power = a ** 2;        # Exponentiation: 10000
```

Each operation typically creates an **arithmetic combinator** in your blueprint.

### Bitwise Operations

For bit manipulation:

```fcdsl
Signal flags = ("signal-A", 0b11110000);
Signal mask = ("signal-B", 0b00001111);

Signal and_result = flags AND mask;    # Bitwise AND: 0
Signal or_result = flags OR mask;      # Bitwise OR: 0b11111111 (255)
Signal xor_result = flags XOR mask;    # Bitwise XOR: 0b11111111 (255)
Signal shifted = flags << 4;           # Left shift: 0b111100000000
Signal right_shifted = flags >> 4;     # Right shift: 0b00001111
```

Note: Bitwise operators are uppercase (`AND`, `OR`, `XOR`) to distinguish them from logical operators.

### Number Literals

Factompiler supports multiple number formats:

```fcdsl
Signal decimal = 255;         # Decimal
Signal binary = 0b11111111;   # Binary
Signal octal = 0o377;         # Octal
Signal hex = 0xFF;            # Hexadecimal

# All four represent the same value: 255
```

This is particularly useful for bitwise operations and color values:

```fcdsl
Signal red_color = 0xFF0000;     # Pure red in RGB
Signal permissions = 0b00000111; # Three flags set
```

## Comparison Operations

Comparisons produce signals with value `1` (true) or `0` (false):

```fcdsl
Signal a = ("signal-A", 50);
Signal threshold = ("signal-T", 100);

Signal is_above = a > threshold;    # 0 (false)
Signal is_below = a < threshold;    # 1 (true)
Signal is_equal = a == threshold;   # 0 (false)
Signal is_not_equal = a != threshold;  # 1 (true)
Signal is_at_least = a >= threshold;   # 0 (false)
Signal is_at_most = a <= threshold;    # 1 (true)
```

Comparisons create **decider combinators** in your blueprint.

## Logical Operations

Combine boolean conditions:

```fcdsl
Signal temp = ("signal-T", 75);
Signal pressure = ("signal-P", 50);

# Logical AND - both conditions must be true
Signal safe = (temp < 100) && (pressure < 80);    # 1 (true)
Signal safe2 = (temp < 100) and (pressure < 80);  # Same thing

# Logical OR - at least one condition must be true  
Signal warning = (temp > 90) || (pressure > 70);    # 0 (false)
Signal warning2 = (temp > 90) or (pressure > 70);   # Same thing

# Logical NOT - invert a boolean
Signal not_safe = !(temp < 100);  # 0 (false)
```

You can use either symbolic operators (`&&`, `||`) or word operators (`and`, `or`) – they're equivalent.

## Type Inference and Coercion

Understanding how types flow through operations is important.

### Rule 1: Explicit Types Win

When you specify a type, it's used:

```fcdsl
Signal iron = ("iron-plate", 100);  # Type: iron-plate
```

### Rule 2: Left Operand Determines Result Type

In binary operations, the left operand's type is used for the result:

```fcdsl
Signal iron = ("iron-plate", 100);
Signal copper = ("copper-plate", 50);

Signal mixed = iron + copper;  # Result type: iron-plate, value: 150
# Warning: Mixed signal types in binary operation
```

### Rule 3: Signals Absorb Integers

When a signal meets an integer, the integer adopts the signal's type:

```fcdsl
Signal iron = ("iron-plate", 100);
Signal more_iron = iron + 50;  # Result: iron-plate, value: 150
```

### Avoiding Type Warnings

Mixed type operations work but generate warnings. To avoid them:

```fcdsl
Signal iron = ("iron-plate", 100);
Signal copper = ("copper-plate", 50);

# Option 1: Project to same type before operating
Signal aligned = (copper | "iron-plate") + iron;  # Both iron-plate, no warning

# Option 2: Use the result type you want
Signal total = (iron | "signal-T") + (copper | "signal-T");  # Both signal-T
```

## The Projection Operator (`|`)

The projection operator changes a signal's type without changing its value:

```fcdsl
Signal iron = ("iron-plate", 100);
Signal as_copper = iron | "copper-plate";   # Type: copper-plate, value: 100
Signal as_virtual = iron | "signal-A";      # Type: signal-A, value: 100
```

### Common Projection Patterns

#### Aggregating Multiple Signal Types

When you want to combine values from different signal types:

```fcdsl
Signal iron = ("iron-plate", 100);
Signal copper = ("copper-plate", 80);
Signal coal = ("coal", 50);

# Sum all onto one output channel
Signal total = (iron | "signal-total")
             + (copper | "signal-total")  
             + (coal | "signal-total");
# Result: signal-total with value 230
```

> **[IMAGE PLACEHOLDER]**: Diagram showing three signals being projected to the same type and summed.

#### Labeling Outputs

Give meaningful names to output signals:

```fcdsl
Signal calculated_value = complex_expression;
Signal output = calculated_value | "signal-output";
```

#### Type Annotation Shorthand

The projection operator can also be used in declarations:

```fcdsl
# These are equivalent:
Signal iron = ("iron-plate", 100);
Signal iron = 100 | "iron-plate";
```

The second form is especially useful when adding types to expressions:

```fcdsl
Signal result = (a + b * 2) | "signal-R";
```

### Same-Type Projections

Projecting to the same type is a no-op – the compiler removes it:

```fcdsl
Signal iron = ("iron-plate", 100);
Signal same = iron | "iron-plate";  # No combinator created, just uses iron
```

## Signal Type Access (`.type`)

You can access a signal's type at compile time using the `.type` property. This allows you to create signals that inherit their type from another signal dynamically.

### Basic Usage

```fcdsl
Signal a = ("iron-plate", 60);
Signal b = 50 | a.type;   # b is projected to iron-plate (same type as a)
```

### In Signal Literals

Use `.type` in signal literal syntax:

```fcdsl
Signal iron = ("iron-plate", 100);
Signal derived = (iron.type, 42);  # Creates iron-plate signal with value 42
```

### Practical Use Cases

**Type Propagation:** Keep signal types consistent without repeating type names:

```fcdsl
Signal input = ("signal-A", 0);       # Input from circuit
Signal doubled = input * 2;            # Inherits signal-A type
Signal offset = 10 | input.type;       # Explicit same type as input
Signal result = doubled + offset;      # No type mismatch warning
```

**Dynamic Type Matching in Functions:** Match the type of a parameter:

```fcdsl
func add_offset(Signal value, int offset) {
    Signal typed_offset = offset | value.type;
    return value + typed_offset;
}
```

This is particularly useful when writing generic functions that should preserve the signal type of their inputs.

## Bundles

Bundles are **unordered collections of signals** that can be operated on as a unit. When used in arithmetic operations, bundles compile to Factorio's "each" signal combinators. When used in comparisons, bundles use "anything" or "everything" virtual signals.

### Bundle Creation

Create bundles using curly brace syntax with signal elements:

```fcdsl
# Bundle from signal literals
Bundle resources = { ("iron-plate", 100), ("copper-plate", 80), ("coal", 50) };

# Bundle from existing signals
Signal x = ("signal-X", 10);
Signal y = ("signal-Y", 20);
Bundle pair = { x, y };

# Empty bundle
Bundle empty = {};
```

**Bundle Flattening:** Bundles can contain other bundles, which are automatically flattened:

```fcdsl
Bundle more = { ("signal-Z", 30) };
Bundle all = { pair, more };  # Contains signal-X, signal-Y, signal-Z
```

**Important:** Each signal type within a bundle must be unique. Duplicate signal types are a compile-time error.

### Bundle Element Selection

Extract a specific signal from a bundle using bracket notation:

```fcdsl
Bundle resources = { ("iron-plate", 100), ("copper-plate", 80) };

Signal iron = resources["iron-plate"];   # Value: 100
Signal copper = resources["copper-plate"];  # Value: 80

# Use directly in expressions
Signal doubled = resources["iron-plate"] * 2;
```

### Bundle Arithmetic

Apply arithmetic operations to **all signals** in a bundle simultaneously:

```fcdsl
Bundle resources = { ("iron-plate", 100), ("copper-plate", 80), ("coal", 50) };

Bundle doubled = resources * 2;      # All values doubled
Bundle incremented = resources + 10; # 10 added to all values
Bundle shifted = resources >> 1;     # All values right-shifted
```

**Factorio Output:** Each bundle arithmetic operation compiles to exactly one arithmetic combinator using the `signal-each` virtual signal:

```
Input: signal-each
Operation: * (or +, -, etc.)
Output: signal-each
```

**Supported operators:** `+`, `-`, `*`, `/`, `%`, `**`, `<<`, `>>`, `AND`, `OR`, `XOR`

**Important:** Bundle arithmetic requires a **scalar operand** (Signal or int). Bundle + Bundle is not supported.

### Bundle Comparisons with `any()` and `all()`

Use `any()` and `all()` functions for bundle-wide comparisons:

```fcdsl
Bundle levels = { ("water", 500), ("oil", 100), ("steam", 0) };

# any() - True if at least one signal matches
Signal anyLow = any(levels) < 50;      # True (steam=0 matches)
Signal anyHigh = any(levels) > 400;    # True (water=500 matches)

# all() - True if every signal matches
Signal allPresent = all(levels) > 0;   # False (steam=0 fails)
Signal allBelow1000 = all(levels) < 1000;  # True (all values match)
```

**Factorio Output:**
- `any()` compiles to a decider combinator using `signal-anything`
- `all()` compiles to a decider combinator using `signal-everything`

### Bundle Use Cases

Bundles are ideal for:
- **Parallel processing**: Apply the same operation to multiple signals
- **Signal aggregation**: Combine multiple signals for bulk operations
- **Conditional logic**: Check conditions across multiple signals at once

```fcdsl
# Scale all resource values for display
Bundle resources = { ("iron-plate", 100), ("copper-plate", 80) };
Bundle scaled = resources * 10;

# Check if any resource is critically low
Signal anyLow = any(resources) < 20;
Entity warning_lamp = place("small-lamp", 0, 0);
warning_lamp.enable = anyLow > 0;
```

## For Loops

For loops allow you to repeat code with an iterator variable. They are **unrolled at compile time**, meaning each iteration generates separate entities and IR operations.

### Range Iteration

Iterate over a range of numbers:

```fcdsl
# Basic range: 0, 1, 2, 3, 4 (excludes end value)
for i in 0..5 {
    Entity lamp = place("small-lamp", i, 0);
    lamp.enable = count > 0;
}

# Range with step: 0, 2, 4, 6, 8
for j in 0..10 step 2 {
    Entity lamp = place("small-lamp", j, 0);
}

# Counting down: 10, 8, 6, 4, 2 (excludes end value)
for k in 10..0 step -2 {
    Entity lamp = place("small-lamp", k, 0);
}
```

**Range Syntax:** `start..end [step value]`
- **start**: Starting value (inclusive)
- **end**: Ending value (exclusive)
- **step**: Optional increment/decrement (default: 1)

### List Iteration

Iterate over an explicit list of values:

```fcdsl
for value in [1, 3, 5, 7, 9] {
    Entity lamp = place("small-lamp", value, 0);
    lamp.enable = count >= value;
}
```

### Using Iterator Variables

The iterator variable can be used in expressions:

```fcdsl
Signal base = ("signal-B", 10);

for x in 0..4 {
    Entity lamp = place("small-lamp", x, 2);
    # Use iterator in comparisons
    lamp.enable = base > x * 2;
}
```

**Important:** Iterator variables are **compile-time constants** – they are substituted directly into expressions during loop unrolling.

### Compile-Time Unrolling

For loops are fully unrolled at compile time. This means:

```fcdsl
for i in 0..3 {
    Entity lamp = place("small-lamp", i, 0);
}
```

Is equivalent to writing:

```fcdsl
Entity lamp_0 = place("small-lamp", 0, 0);
Entity lamp_1 = place("small-lamp", 1, 0);
Entity lamp_2 = place("small-lamp", 2, 0);
```

### Practical Examples

**Creating a lamp array:**

```fcdsl
Memory counter: "signal-A";
counter.write((counter.read() + 1) % 8);
Signal pos = counter.read();

# Create 8 lamps in a row - one lights up at a time (chaser effect)
for i in 0..8 {
    Entity lamp = place("small-lamp", i * 2, 0);
    lamp.enable = pos == i;
}
```

**Binary display:**

```fcdsl
Memory bits: "signal-A";
bits.write((bits.read() + 1) % 16);

# 4-bit binary display
for bit in 0..4 {
    Entity lamp = place("small-lamp", bit, 0);
    lamp.enable = ((bits.read() >> bit) AND 1) > 0;
}
```

## The Output Specifier (`:`)

The output specifier is used with decider combinators to pass through values conditionally:

```fcdsl
Signal count = ("signal-A", 50);
Signal data = ("signal-B", 1000);

# Only output data when count > 10
Signal filtered = (count > 10) : data;
```

This is different from multiplication:
- `condition * data` – outputs `condition_value × data_value`
- `condition : data` – outputs `data_value` when condition is true

The output specifier is particularly useful for signal gating:

```fcdsl
# Gate pattern: only pass signal when enabled
Signal gate = (enable > 0) : input_signal;
```

## Operator Precedence

From highest (evaluated first) to lowest:

1. Parentheses `()`
2. Unary operators: `+`, `-`, `!`
3. Power: `**`
4. Multiplicative: `*`, `/`, `%`
5. Additive: `+`, `-`
6. Shift: `<<`, `>>`
7. Bitwise AND: `AND`
8. Bitwise XOR: `XOR`
9. Bitwise OR: `OR`
10. Projection: `|`
11. Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
12. Output specifier: `:`
13. Logical AND: `&&`, `and`
14. Logical OR: `||`, `or`

### Examples

```fcdsl
Signal a = 2 + 3 * 4;        # 14 (multiplication first)
Signal b = (2 + 3) * 4;      # 20 (parentheses override)
Signal c = 2 ** 3 ** 2;      # 512 (** is right-associative: 2^(3^2) = 2^9)
Signal d = x > 5 && y < 10;  # Comparison before logical AND
```

## Practical Examples

### Temperature Monitor

```fcdsl
# Input from temperature sensor (wire from heat exchanger)
Signal temp = ("signal-T", 0);

# Define thresholds
int too_cold = 100;
int too_hot = 500;
int critical = 700;

# Compute status flags
Signal normal = (temp >= too_cold) && (temp <= too_hot);
Signal warning = (temp > too_hot) && (temp < critical);
Signal danger = temp >= critical;

# Output labeled status signals
Signal status_normal = normal | "signal-green";
Signal status_warning = warning | "signal-yellow";
Signal status_danger = danger | "signal-red";
```

### Resource Ratio Calculator

```fcdsl
# Inputs from logistics network (wire from roboport)
Signal iron = ("iron-plate", 0);
Signal copper = ("copper-plate", 0);
Signal circuits = ("electronic-circuit", 0);

# Calculate how many circuits we could make
# (1 circuit = 1 iron + 3 copper)
Signal iron_limited = iron;
Signal copper_limited = copper / 3;

# Minimum determines actual capacity
Signal can_make = (iron_limited < copper_limited) * iron_limited
                + (copper_limited <= iron_limited) * copper_limited;

# Output the result
Signal output = can_make | "signal-C";
```

### RGB Color Mixer

```fcdsl
# Individual color channels (0-255)
Signal red = ("signal-R", 128);
Signal green = ("signal-G", 64);
Signal blue = ("signal-B", 255);

# Pack into single value for packed RGB mode
# Format: 0xRRGGBB
Signal packed = (red << 16) + (green << 8) + blue;
Signal rgb_output = packed | "signal-black";  # Use any signal
```

---

## Summary

- **Signals** have a **type** and a **value**
- Use **explicit types** when the signal type matters
- Use **implicit types** for intermediate calculations
- **Projections** (`|`) change signal types
- **`.type`** accesses a signal's type for dynamic type propagation
- **Bundles** group multiple signals for parallel operations using `signal-each`
- Use **`any()`** and **`all()`** for bundle-wide comparisons
- **For loops** unroll at compile time to create multiple entities or expressions
- **Comparisons** produce boolean signals (0 or 1)
- **Logical operators** combine boolean signals
- The **output specifier** (`:`) gates signal values based on conditions

---

**← [Quick Start](02_quick_start.md)** | **[Memory →](04_memory.md)**
