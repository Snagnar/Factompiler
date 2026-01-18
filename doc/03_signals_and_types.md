# Signals and Types

Signals are the foundation of everything in Facto. Understanding how they work unlocks the full power of Factorio's circuit network.

---

## What is a Signal?

In Factorio, signals are data flowing through circuit networks. Each signal has two parts:

| Component | Description |
|-----------|-------------|
| **Type** | What kind of signal — `iron-plate`, `signal-A`, `water`, etc. |
| **Value** | An integer count |

When you read a circuit network in-game, you see entries like "iron-plate: 50, copper-plate: 30" — two signals with different types and values.

In Facto:

```facto
Signal my_signal = ("iron-plate", 50);
```

This creates a signal of type `iron-plate` with value `50`.

---

## Signal Types in Factorio

### Virtual Signals

Abstract signals for pure logic — they don't represent physical items:

| Category | Examples |
|----------|----------|
| Letters | `signal-A` through `signal-Z` |
| Numbers | `signal-0` through `signal-9` |
| Colors | `signal-red`, `signal-green`, `signal-blue`, `signal-yellow`, `signal-cyan`, `signal-magenta`, `signal-white`, `signal-grey`, `signal-black`, `signal-pink` |
| Special | `signal-everything`, `signal-anything`, `signal-each` |

### Item Signals

Every Factorio item is also a signal type:
- Plates: `iron-plate`, `copper-plate`, `steel-plate`
- Circuits: `electronic-circuit`, `advanced-circuit`, `processing-unit`
- Ores: `iron-ore`, `copper-ore`, `coal`, `stone`
- And many more...

### Fluid Signals

Fluids work as signals too:
- `water`, `steam`
- `crude-oil`, `petroleum-gas`, `light-oil`, `heavy-oil`
- `lubricant`, `sulfuric-acid`

---

## Declaring Signals

### Explicit Types

Specify exactly what signal type you need:

```facto
Signal iron = ("iron-plate", 100);
Signal counter = ("signal-A", 0);
Signal temperature = ("steam", 500);
```

### Auto-Allocated Types

Let the compiler choose a type when you don't care:

```facto
Signal x = 5;    # Compiler assigns signal-A
Signal y = 10;   # Compiler assigns signal-B
Signal z = 15;   # Compiler assigns signal-C
```

The compiler allocates virtual signals (`signal-A`, `signal-B`, etc.) automatically. Perfect for intermediate calculations.

### The `int` Type

For compile-time constants that shouldn't become signals:

```facto
int threshold = 100;
int multiplier = 5;
Signal result = input * multiplier;  # multiplier is just the number 5
```

**Key difference:**
- `Signal x = 5` → creates a constant combinator outputting `signal-A: 5`
- `int x = 5` → just the number 5, no combinator

Use `int` for calculation constants that don't need their own signal.

---

## Number Literals

Facto supports multiple number formats:

```facto
Signal decimal = 255;         # Decimal
Signal binary = 0b11111111;   # Binary
Signal octal = 0o377;         # Octal
Signal hex = 0xFF;            # Hexadecimal

# All four represent the same value: 255
```

Especially useful for bitwise operations and colors:

```facto
Signal red_color = 0xFF0000;     # Pure red in RGB
Signal permissions = 0b00000111; # Three flags set
```

---

## Arithmetic Operations

All standard arithmetic works on signals:

```facto
Signal a = ("signal-A", 100);
Signal b = ("signal-B", 30);

Signal sum = a + b;           # Addition: 130
Signal difference = a - b;    # Subtraction: 70
Signal product = a * b;       # Multiplication: 3000
Signal quotient = a / b;      # Division: 3 (integer)
Signal remainder = a % b;     # Modulo: 10
Signal power = a ** 2;        # Exponentiation: 10000
```

Each operation creates an **arithmetic combinator** in your blueprint.

### Bitwise Operations

For bit manipulation (note: operators are UPPERCASE):

```facto
Signal flags = ("signal-A", 0b11110000);
Signal mask = ("signal-B", 0b00001111);

Signal and_result = flags AND mask;    # Bitwise AND: 0
Signal or_result = flags OR mask;      # Bitwise OR: 255
Signal xor_result = flags XOR mask;    # Bitwise XOR: 255
Signal shifted = flags << 4;           # Left shift
Signal right_shifted = flags >> 4;     # Right shift
```

---

## Comparison Operations

Comparisons produce `1` (true) or `0` (false):

```facto
Signal a = ("signal-A", 50);
Signal threshold = ("signal-T", 100);

Signal is_above = a > threshold;       # 0 (false)
Signal is_below = a < threshold;       # 1 (true)
Signal is_equal = a == threshold;      # 0 (false)
Signal is_not_equal = a != threshold;  # 1 (true)
Signal is_at_least = a >= threshold;   # 0 (false)
Signal is_at_most = a <= threshold;    # 1 (true)
```

Comparisons create **decider combinators** in your blueprint.

---

## Conditional Values (The `:` Operator)

This is often a very useful feature. The output specifier creates efficient conditional logic:

```facto
Signal result = condition : value;
```

When `condition` is true, output `value`. When false, output `0`.

### Why Conditional Values Matter

The `:` operator compiles to a decider combinator's "copy input" mode, which is more efficient than multiplication:

| Pattern | Compiles To | Efficiency |
|---------|-------------|------------|
| `(x > 0) : y` | 1 decider (copy input) | ✓ Best |
| `(x > 0) * y` | 1 decider + 1 arithmetic | ✗ Slower |

### Basic Examples

```facto
Signal count = ("signal-A", 50);
Signal data = ("signal-B", 1000);

# Only output data when count > 10
Signal filtered = (count > 10) : data;
```

### Building If-Then-Else Logic

Since `:` outputs 0 when false, you can add multiple conditions:

```facto
Signal x = ("signal-X", 75);

# If x > 100: output 3
# Else if x > 50: output 2
# Else: output 1
Signal result = 
    ((x > 100) : 3) +
    ((x > 50 && x <= 100) : 2) +
    ((x <= 50) : 1);
```

### Signal Gating

A common pattern — only pass a signal when enabled:

```facto
Signal enable = ("signal-E", 1);
Signal input = ("signal-I", 500);

Signal gated = (enable > 0) : input;  # Passes input only when enabled
```

### Clamping Values

```facto
Signal speed = ("signal-S", 150);

# Cap speed at 100
Signal capped = ((speed > 100) : 100) + ((speed <= 100) : speed);
```

### Selection (Choose A or B)

```facto
Signal flag = ("signal-F", 1);
Signal value_a = ("signal-A", 100);
Signal value_b = ("signal-B", 200);

Signal result = ((flag > 0) : value_a) + ((flag == 0) : value_b);
```

---

## Logical Operations

Combine boolean conditions:

```facto
Signal temp = ("signal-T", 75);
Signal pressure = ("signal-P", 50);

# Logical AND — both must be true
Signal safe = (temp < 100) && (pressure < 80);     # 1 (true)
Signal safe2 = (temp < 100) and (pressure < 80);   # Same thing

# Logical OR — at least one must be true
Signal warning = (temp > 90) || (pressure > 70);   # 0 (false)
Signal warning2 = (temp > 90) or (pressure > 70);  # Same thing

# Logical NOT — invert a boolean
Signal not_safe = !(temp < 100);  # 0 (false)
```

Both symbolic (`&&`, `||`) and word (`and`, `or`) operators work identically.

---

## Type Inference and Coercion

### Rule 1: Explicit Types Win

When you specify a type, it's used:

```facto
Signal iron = ("iron-plate", 100);  # Type: iron-plate
```

### Rule 2: Left Operand Determines Result Type

In binary operations, the result inherits the left operand's type:

```facto
Signal iron = ("iron-plate", 100);
Signal copper = ("copper-plate", 50);

Signal mixed = iron + copper;  # Type: iron-plate, value: 150
# Warning: Mixed signal types
```

### Rule 3: Signals Absorb Integers

Integers adopt the signal's type:

```facto
Signal iron = ("iron-plate", 100);
Signal more_iron = iron + 50;  # Type: iron-plate, value: 150
```

---

## The Projection Operator (`|`)

Change a signal's type without changing its value:

```facto
Signal iron = ("iron-plate", 100);
Signal as_copper = iron | "copper-plate";   # Type: copper-plate, value: 100
Signal as_virtual = iron | "signal-A";      # Type: signal-A, value: 100
```

### Aggregating Multiple Signal Types

Combine different signals onto one output channel:

```facto
Signal iron = ("iron-plate", 100);
Signal copper = ("copper-plate", 80);
Signal coal = ("coal", 50);

Signal total = (iron | "signal-T")
             + (copper | "signal-T")  
             + (coal | "signal-T");
# Result: signal-T with value 230
```

### Type Annotation Shorthand

Projection works in declarations too:

```facto
Signal result = (a + b * 2) | "signal-R";
```

---

## Signal Type Access (`.type`)

Access a signal's type at compile time:

```facto
Signal a = ("iron-plate", 60);
Signal b = 50 | a.type;   # b is iron-plate (same type as a)
```

### In Signal Literals

```facto
Signal iron = ("iron-plate", 100);
Signal derived = (iron.type, 42);  # Creates iron-plate: 42
```

### Type Propagation

Keep types consistent without repeating names:

```facto
Signal input = ("signal-A", 0);
Signal offset = 10 | input.type;       # Matches input's type
Signal result = input + offset;        # No type mismatch warning
```

---

## Bundles

Bundles are **unordered collections of signals** that operate as a unit. They compile to Factorio's "each" signal combinators.

### Creating Bundles

```facto
# From signal literals
Bundle resources = { ("iron-plate", 100), ("copper-plate", 80), ("coal", 50) };

# From existing signals
Signal x = ("signal-X", 10);
Signal y = ("signal-Y", 20);
Bundle pair = { x, y };

# Empty bundle
Bundle empty = {};
```

Bundles automatically flatten nested bundles.

**Important:** Each signal type in a bundle must be unique.

### Element Selection

Extract specific signals with bracket notation:

```facto
Bundle resources = { ("iron-plate", 100), ("copper-plate", 80) };

Signal iron = resources["iron-plate"];     # Value: 100
Signal copper = resources["copper-plate"]; # Value: 80
```

### Bundle Arithmetic

Apply operations to **all signals** at once:

```facto
Bundle resources = { ("iron-plate", 100), ("copper-plate", 80), ("coal", 50) };

Bundle doubled = resources * 2;      # All values doubled
Bundle incremented = resources + 10; # 10 added to each
Bundle shifted = resources >> 1;     # All values shifted
```

Each bundle arithmetic operation compiles to one arithmetic combinator using `signal-each`.

**Supported operators:** `+`, `-`, `*`, `/`, `%`, `**`, `<<`, `>>`, `AND`, `OR`, `XOR`

**Note:** Bundle arithmetic requires a scalar operand. Bundle + Bundle is not supported.

### Bundle Comparisons: `any()` and `all()`

```facto
Bundle levels = { ("water", 500), ("oil", 100), ("steam", 0) };

# any() — true if ANY signal matches
Signal anyLow = any(levels) < 50;      # True (steam=0 matches)
Signal anyHigh = any(levels) > 400;    # True (water=500 matches)

# all() — true if EVERY signal matches
Signal allPresent = all(levels) > 0;   # False (steam=0 fails)
Signal allBelow1000 = all(levels) < 1000;  # True
```

- `any()` compiles to `signal-anything`
- `all()` compiles to `signal-everything`

---

## For Loops

For loops **unroll at compile time** — each iteration generates separate entities.

### Range Iteration

```facto
# 0, 1, 2, 3, 4 (end value excluded)
for i in 0..5 {
    Entity lamp = place("small-lamp", i, 0);
}

# 0, 2, 4, 6, 8 (with step)
for j in 0..10 step 2 {
    Entity lamp = place("small-lamp", j, 0);
}

# Counting down: 10, 8, 6, 4, 2
for k in 10..0 step -2 {
    Entity lamp = place("small-lamp", k, 0);
}
```

### List Iteration

```facto
for value in [1, 3, 5, 7, 9] {
    Entity lamp = place("small-lamp", value, 0);
}
```

### Using Iterator Variables

Iterator variables are compile-time constants:

```facto
Signal base = ("signal-B", 10);

for x in 0..4 {
    Entity lamp = place("small-lamp", x, 2);
    lamp.enable = base > x * 2;
}
```

### Practical Example: Binary Display

```facto
Memory bits: "signal-A";
bits.write((bits.read() + 1) % 16);

# 4-bit binary display
for bit in 0..4 {
    Entity lamp = place("small-lamp", bit, 0);
    lamp.enable = ((bits.read() >> bit) AND 1) > 0;
}
```

---

## Operator Precedence

From highest to lowest:

| Precedence | Operators |
|------------|-----------|
| 1 | `()` parentheses |
| 2 | `+`, `-`, `!` unary |
| 3 | `**` power |
| 4 | `*`, `/`, `%` |
| 5 | `+`, `-` |
| 6 | `<<`, `>>` |
| 7 | `AND` |
| 8 | `XOR` |
| 9 | `OR` |
| 10 | `\|` projection |
| 11 | `==`, `!=`, `<`, `<=`, `>`, `>=` |
| 12 | `:` output specifier |
| 13 | `&&`, `and` |
| 14 | `\|\|`, `or` |

---

## Practical Examples

### Temperature Monitor

```facto
Signal temp = ("signal-T", 0);  # Input from sensor

int too_cold = 100;
int too_hot = 500;
int critical = 700;

Signal normal = (temp >= too_cold) && (temp <= too_hot);
Signal warning = (temp > too_hot) && (temp < critical);
Signal danger = temp >= critical;

# Labeled outputs using conditional values
Signal status_normal = (normal != 0) : 1 | "signal-G";
Signal status_warning = (warning != 0) : 1 | "signal-Y";
Signal status_danger = (danger != 0) : 1 | "signal-R";
```

### Resource Ratio Calculator

```facto
Signal iron = ("iron-plate", 0);
Signal copper = ("copper-plate", 0);

# Electronic circuits need 1 iron + 3 copper each
Signal iron_limited = iron;
Signal copper_limited = copper / 3;

# Minimum determines capacity — using conditional values
Signal can_make = ((iron_limited < copper_limited) : iron_limited)
               + ((copper_limited <= iron_limited) : copper_limited);
```

### RGB Color Packer

```facto
Signal red = ("signal-R", 128);
Signal green = ("signal-G", 64);
Signal blue = ("signal-B", 255);

# Pack into 0xRRGGBB format
Signal packed = (red << 16) + (green << 8) + blue;
```

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Signals** | Type + Value, creates constant combinator |
| **`int`** | Compile-time constant, no combinator |
| **Conditional Values** | `condition : value` — most efficient conditional logic |
| **Projections** | `\|` changes signal type |
| **`.type`** | Access signal's type at compile time |
| **Bundles** | Collections for parallel operations via `signal-each` |
| **`any()`/`all()`** | Bundle-wide comparisons |
| **For loops** | Compile-time unrolling |

---

**[← Quick Start](02_quick_start.md)** | **[Memory →](04_memory.md)**
