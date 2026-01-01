# Factorio Circuit DSL Language Specification

**Version 2.5**  
**Date: December 2025**

This document provides a complete specification of the Factorio Circuit DSL (Domain Specific Language) as currently implemented. The DSL compiles to Factorio 2.0 blueprint strings that can be imported into the game.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Lexical Structure](#lexical-structure)
4. [Type System](#type-system)
5. [Expressions](#expressions)
6. [Statements](#statements)
7. [Memory System](#memory-system)
8. [Entity System](#entity-system)
9. [Functions and Modules](#functions-and-modules)
10. [Compilation Model](#compilation-model)
11. [Circuit Network Integration](#circuit-network-integration)
12. [Optimization Passes](#optimization-passes)
13. [Best Practices](#best-practices)
14. [Examples](#examples)
15. [Error Handling](#error-handling)

---

## Overview

The Factorio Circuit DSL is a **statically-typed**, **imperative language** designed to generate Factorio combinator circuits and entity blueprints. It provides high-level abstractions over Factorio's circuit network while maintaining direct control over signal routing and entity behavior.

### Design Philosophy

- **Signal-Centric**: Everything is a signal flowing through circuit networks
- **Type Safety**: Explicit signal types prevent channel conflicts
- **Implicit Optimization**: Compiler optimizes common patterns automatically
- **Direct Mapping**: Clear correspondence between DSL and Factorio entities

### Compilation Pipeline

```
Source Code → Parser → Semantic Analysis → IR Generation → 
Layout Planning → Wire Routing → Blueprint Emission
```

Each stage validates and optimizes the program before generating the final Factorio blueprint JSON.

---

## Quick Start

### Hello, Lamp!

```fcdsl
# Create a blinking lamp controlled by a counter
Memory counter: "signal-A";
counter.write(counter.read() + 1);

Signal blink = (counter.read() % 10) < 5;

Entity lamp = place("small-lamp", 0, 0);
lamp.enable = blink;
```

Save this program to `hello_lamp.fcdsl` and compile it with:
```bash
python compile.py hello_lamp.fcdsl
```

Import the resulting blueprint string into Factorio and watch your lamp blink!

---

## Lexical Structure

### Comments

```fcdsl
# Hash-style line comments
// C-style line comments also work
Signal x = 5;  // End-of-line comments
```

### Literals

**Integers:**

The DSL supports integer literals in multiple bases:

```fcdsl
# Decimal
42        # Positive integer
-17       # Negative integer
255       # Standard decimal

# Binary (0b prefix)
0b1010    # = 10 in decimal
0b11111111  # = 255 in decimal

# Octal (0o prefix)
0o17      # = 15 in decimal
0o377     # = 255 in decimal

# Hexadecimal (0x prefix)
0xA       # = 10 in decimal
0xFF      # = 255 in decimal
0xff      # = 255 in decimal (same as 0xFF)
```

All number bases represent 32-bit signed integers and compile to the same values.

**Strings:**
```fcdsl
"iron-plate"    # Item signal
"signal-A"      # Virtual signal
"water"         # Fluid signal
```

Strings are used exclusively for signal type names and entity prototypes.

### Operators

**Arithmetic:** `+` `-` `*` `/` `%` `**`  
**Bitwise:** `AND` `OR` `XOR` `<<` `>>`  
**Comparison:** `==` `!=` `<` `<=` `>` `>=`  
**Logical:** `&&` `||` `and` `or` `!`  
**Projection:** `|`  
**Output Specifier:** `:`  
**Assignment:** `=`

### Operator Precedence

From highest (tightest binding) to lowest (loosest binding):

1. **Parentheses** `()`
2. **Unary** `+` `-` `!`
3. **Power** `**` (right-associative)
4. **Multiplicative** `*` `/` `%`
5. **Additive** `+` `-`
6. **Shift** `<<` `>>`
7. **Bitwise AND** `AND`
8. **Bitwise XOR** `XOR`
9. **Bitwise OR** `OR`
10. **Projection** `|`
11. **Comparison** `==` `!=` `<` `<=` `>` `>=`
12. **Output Specifier** `:`
13. **Logical AND** `&&` `and`
14. **Logical OR** `||` `or`

**Important Notes:**
- The power operator (`**`) is **right-associative**: `2 ** 3 ** 2` evaluates as `2 ** (3 ** 2)` = 512
- All other binary operators are left-associative
- Bitwise operators (`AND`, `OR`, `XOR`) use uppercase keywords to distinguish them from logical operators

#### Type Precedence in Operations

When combining operands of different types, the compiler follows these precedence rules:

**Integer + Integer = Integer**
```fcdsl
int result = 5 + 3;  # Pure integer arithmetic
```

**Signal + Integer = Signal** (signal type takes precedence)
```fcdsl
Signal iron = ("iron-plate", 100);
Signal more = iron + 50;  # Result: iron-plate signal with value 150
# The integer constant is coerced to match the signal's type
```

**Signal + Signal = Signal** (left operand's type takes precedence)
```fcdsl
Signal iron = ("iron-plate", 100);
Signal copper = ("copper-plate", 50);
Signal mixed = iron + copper;  # Result: iron-plate signal with value 150
# Warning: Mixed signal types in binary operation
```

**Type Precedence Rule:** When one operand is a signal with an explicit type and the other is an integer constant or a signal with a different type, the **existing signal type always takes precedence**. The compiler automatically coerces the other operand to match.

To avoid type mismatch warnings, use explicit projection:
```fcdsl
Signal aligned = (copper | "iron-plate") + iron;  # Both iron-plate, no warning
```

---

## Type System

The DSL has five fundamental value types that map directly to Factorio circuit concepts.

### Integer (`int`)

Plain integer values used for constants and direct computations.

```fcdsl
int count = 42;
int threshold = count + 10;
```

Integers are **not** signals and cannot flow through circuit networks. Use them for compile-time constants or intermediate calculations that will be folded into signal literals.

### Signal (`Signal`)

Single-channel Factorio signals carrying both a **type** (iron-plate, signal-A) and a **value** (integer count).

#### Explicit Signal Types

```fcdsl
Signal iron = ("iron-plate", 100);      # Item signal
Signal virtual = ("signal-A", 42);      # Virtual signal
Signal water = ("water", 1000);         # Fluid signal
```

#### Implicit Signal Types

When you don't specify a type, the compiler allocates a virtual signal:

```fcdsl
Signal implicit = 5;  # Compiler assigns __v1 → signal-A
Signal another = 10;  # Compiler assigns __v2 → signal-B
```

The compiler maps `__v1`, `__v2`, etc. to Factorio's virtual signals `signal-A` through `signal-Z`.

#### Constant Folding in Literals

Signal literals can contain arithmetic expressions that are evaluated at compile time:

```fcdsl
Signal step = ("signal-A", 5 * 2 - 9);  # Compiles to ("signal-A", 1)
Signal threshold = ("iron-plate", 100 / 2 + 50);  # Compiles to ("iron-plate", 100)
```

### Memory (`Memory`)

Stateful storage cells that persist values across game ticks using write-gated latch circuits.

```fcdsl
Memory counter: "signal-A";             # Explicit type
Memory state;                           # Implicit type (inferred from first write)
```

#### Type Inference

If you don't specify a type, the compiler infers it from the first `.write()`:

```fcdsl
Memory accumulator;  # Type unknown

Signal iron = ("iron-plate", 50);
accumulator.write(iron);  # Now accumulator stores iron-plate signals
```

**Warning:** All writes to a memory cell must use the same signal type. Mixed types will generate warnings (or errors in `--strict` mode).

### Entity (`Entity`)

References to placed Factorio entities that can be controlled via circuit networks.

```fcdsl
Entity lamp = place("small-lamp", 5, 0);
Entity train_stop = place("train-stop", 10, 5);
```

Entities expose **properties** that can be read from or written to using circuit signals.

### Bundle (`Bundle`)

Bundles are **unordered collections of signals** that can be operated on as a unit. When used in arithmetic operations, bundles compile to Factorio's "each" signal combinators. When used in comparisons, bundles use "anything" or "everything" virtual signals.

#### Bundle Creation

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

**Important:** Each signal type within a bundle must be unique. Duplicate signal types are a compile-time error:

```fcdsl
Bundle bad = { ("iron-plate", 10), ("iron-plate", 20) };  # ERROR: Duplicate signal type
```

#### Bundle Element Selection

Extract a specific signal from a bundle using bracket notation:

```fcdsl
Bundle resources = { ("iron-plate", 100), ("copper-plate", 80) };

Signal iron = resources["iron-plate"];   # Value: 100
Signal copper = resources["copper-plate"];  # Value: 80

# Use directly in expressions
Signal doubled = resources["iron-plate"] * 2;
```

#### Bundle Arithmetic

Apply arithmetic operations to **all signals** in a bundle simultaneously:

```fcdsl
Bundle resources = { ("iron-plate", 100), ("copper-plate", 80), ("coal", 50) };

Bundle doubled = resources * 2;      # All values doubled
Bundle incremented = resources + 10; # 10 added to all values
Bundle shifted = resources >> 1;     # All values right-shifted

# Supported operators: +, -, *, /, %, **, <<, >>, AND, OR, XOR
```

**Factorio Output:** Each bundle arithmetic operation compiles to exactly one arithmetic combinator using the `signal-each` virtual signal:

```
Input: signal-each
Operation: * (or +, -, etc.)
Output: signal-each
```

**Important:** Bundle arithmetic requires a **scalar operand** (Signal or int). Bundle + Bundle is not supported:

```fcdsl
Bundle a = { ("iron-plate", 100) };
Bundle b = { ("copper-plate", 80) };
Bundle c = a + b;  # ERROR: Bundle operations require Signal or int operand
```

#### Bundle Comparisons with `any()` and `all()`

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
- `any()` compiles to a decider combinator outputting `signal-anything`
- `all()` compiles to a decider combinator outputting `signal-everything`

#### Bundle Use Cases

Bundles are ideal for:
- **Parallel processing**: Apply the same operation to multiple signals
- **Signal aggregation**: Combine multiple signals for bulk operations
- **Conditional logic**: Check conditions across multiple signals at once

```fcdsl
# 7-segment display example: store digit patterns as a bundle
Bundle digit_0 = {
    ("signal-A", 0b1111110),  # Segments a-g encoded as bits
    ("signal-B", 0b0110000),
    ("signal-C", 0b1101101)
};

# Shift to extract bit for each segment
Bundle segment_bits = digit_0 >> position;
Bundle active = segment_bits AND 1;
```

---

## Expressions

### Signal Literals

#### Typed Literals

```fcdsl
Signal iron = ("iron-plate", 100);
Signal copper = ("copper-plate", 50);
Signal flag = ("signal-A", 1);
```

**Type Literal Syntax:**

The type name in signal literals is a string:

```fcdsl
# Both forms are equivalent:
Signal a = ("signal-A", 100);
```

#### Syntactic Sugar for Type Annotations

The DSL provides a convenient shorthand using the projection operator for type annotations:

```fcdsl
# Long form (explicit tuple)
Signal iron = ("iron-plate", 100);

# Short form (projection syntax sugar)
Signal iron = 100 | "iron-plate";

# Both are equivalent and compile to the same IR
```

This syntactic sugar is particularly useful when you need to add a type annotation to an existing value:

```fcdsl
# Adding type to a function parameter
func process(Signal value) {
    Signal typed_value = value | "signal-S";  # Annotate with signal-S
    Memory storage: "signal-S";
    storage.write(typed_value);
}
```

**Important:** The projection syntax `value | "type"` is semantically equivalent to `("type", value)` when used in signal declarations. Both create a signal with the specified type and value.

#### Untyped Literals (Implicit Allocation)

```fcdsl
Signal x = 5;   # Compiler allocates __v1 → signal-A
Signal y = 10;  # Compiler allocates __v2 → signal-B
```

### Arithmetic Operations

```fcdsl
Signal sum = a + b;
Signal diff = a - b;
Signal product = a * 2;
Signal quotient = a / 3;
Signal remainder = a % 10;
```

#### Type Inference Rules

**Int + Int = Int**
```fcdsl
int x = 5 + 3;  # Result: 8 (integer)
```

**Signal + Int = Signal** (integer coerced to signal type)
```fcdsl
Signal iron = ("iron-plate", 100);
Signal more = iron + 50;  # Result: iron-plate signal with value 150
# Warning: Mixed types in binary operation
```

**Signal + Signal = Signal** (left operand type wins)
```fcdsl
Signal iron = ("iron-plate", 100);
Signal copper = ("copper-plate", 50);
Signal mixed = iron + copper;  # Result: iron-plate signal with value 150
# Warning: Mixed signal types in binary operation
```

To avoid warnings, use explicit projections:

```fcdsl
Signal aligned = (copper | "iron-plate") + iron;  # Both iron-plate
```

### Comparison Operations

All comparison operators return signals:

```fcdsl
Signal is_greater = iron > 100;     # Returns signal (1 if true, 0 if false)
Signal is_equal = count == 50;      # Returns signal
Signal in_range = (x >= 10) && (x <= 100);
```

The result inherits the signal type from the left operand, or uses a virtual signal if comparing integers.

### Logical Operations

Logical operators work with boolean values (treating non-zero as true, zero as false):

```fcdsl
# Logical AND - both operands must be non-zero
Signal both1 = (a > 0) && (b > 0);  # Symbolic form
Signal both2 = (a > 0) and (b > 0); # Word form (equivalent)

# Logical OR - at least one operand must be non-zero
Signal either1 = (a > 0) || (b > 0);  # Symbolic form
Signal either2 = (a > 0) or (b > 0);  # Word form (equivalent)

# Logical NOT - inverts boolean value
Signal not_zero = !(x == 0);
```

**Operator Aliases:** The word forms `and` and `or` are semantically identical to `&&` and `||`. Use whichever form improves readability.

**Compilation:** Logical operations are compiled to arithmetic combinators:
- `&&` compiles to multiplication (both must be non-zero)
- `||` compiles to addition followed by comparison (at least one non-zero)
- `!` compiles to a zero-equality check

### Output Specifier

The output specifier (`:`) is used with decider combinators to copy signal values from input:

```fcdsl
# Syntax: condition : output_value
Signal filtered = (count > 10) : input_signal;

# The condition determines WHEN to output
# The output_value determines WHICH signal to copy
```

**Use Cases:**
- Filtering signals based on conditions
- Gating signal flow
- Conditional pass-through

**Precedence:** The output specifier has lower precedence than comparison operators, so:
```fcdsl
Signal result = x > 5 : y;    # Equivalent to: (x > 5) : y
```

**Type Inference:** The result type comes from the output_value, not the condition.

### Projection Operator (`|`)

The projection operator **casts** a signal to a different channel:

```fcdsl
Signal iron = ("iron-plate", 100);
Signal as_copper = iron | "copper-plate";  # Now a copper-plate signal
Signal as_virtual = iron | "signal-A";     # Now a virtual signal
```

#### Same-Type Projections (No-Op)

```fcdsl
Signal iron = ("iron-plate", 100);
Signal same = iron | "iron-plate";  # No-op; compiler optimizes away
```

#### Channel Aggregation

To sum signals across different channels onto one channel:

```fcdsl
Signal iron = ("iron-plate", 100);
Signal copper = ("copper-plate", 80);
Signal coal = ("coal", 50);

# Aggregate onto signal-total channel
Signal total = (iron | "signal-total")
             + (copper | "signal-total")
             + (coal | "signal-total");
```

This pattern creates a single combinator that outputs `signal-total` with value 230.

### Signal Type Access (`.type`)

Access a signal's type at compile time using the `.type` property. This allows you to create signals that inherit their type from another signal.

#### Basic Usage

```fcdsl
Signal a = ("iron-plate", 60);
Signal b = 50 | a.type;   # b is projected to iron-plate (same type as a)
```

#### In Signal Literals

Use `.type` in signal literal syntax:

```fcdsl
Signal iron = ("iron-plate", 100);
Signal derived = (iron.type, 42);  # Creates iron-plate signal with value 42
```

#### Practical Use Cases

**Type Propagation:** Keep signal types consistent without repeating type names:

```fcdsl
Signal input = ("signal-A", 0);       # Input from circuit
Signal doubled = input * 2;            # Inherits signal-A type
Signal offset = 10 | input.type;       # Explicit same type as input
Signal result = doubled + offset;      # No type mismatch warning
```

**Dynamic Type Matching:** Match the type of a parameter:

```fcdsl
func add_offset(Signal value, int offset) {
    Signal typed_offset = offset | value.type;
    return value + typed_offset;
}
```


---

## Statements

### Variable Declarations

```fcdsl
int count = 42;
Signal iron = ("iron-plate", 100);
Entity lamp = place("small-lamp", 0, 0);
Memory counter: "signal-A";
```

All variables are **immutable** except memories (which are stateful by nature).

### Assignments

```fcdsl
x = y + 5;              # Variable reassignment
lamp.enable = count > 0; # Property assignment
```

**Note:** Variable reassignment creates a new signal in the IR; it doesn't mutate the original. Think of it as creating an alias.

### Expression Statements

Any expression can be a statement for side effects:

```fcdsl
memory.write(value);   # Memory write side effect
place("lamp", 5, 0);   # Entity placement side effect
```

### Return Statements

Used in functions to specify return values:

```fcdsl
func double(Signal x) {
    return x * 2;
}
```

### Import Statements

```fcdsl
import "stdlib/memory_utils.fcdsl";
```

Imports use **C-style preprocessing**: the imported file's content is inlined before parsing. Circular imports are detected and skipped.

**Note:** The `import "file" as alias` syntax is recognized by the grammar but aliases are not currently utilized since files are textually inlined. All functions from imported files become available in the global namespace.

### For Loop Statements

For loops allow you to repeat code with an iterator variable. They are **unrolled at compile time**, meaning each iteration generates separate entities and IR operations.

#### Range Iteration

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
    lamp.enable = count > 0;
}

# Counting down: 10, 8, 6, 4, 2 (excludes end value)
for k in 10..0 step -2 {
    Entity lamp = place("small-lamp", k, 0);
    lamp.enable = count > 0;
}
```

**Range Syntax:** `start..end [step value]`
- **start**: Starting value (inclusive)
- **end**: Ending value (exclusive)
- **step**: Optional increment/decrement (default: 1)

#### List Iteration

Iterate over an explicit list of values:

```fcdsl
for value in [1, 3, 5, 7, 9] {
    Entity lamp = place("small-lamp", value, 0);
    lamp.enable = count >= value;
}
```

**List Syntax:** `[value1, value2, ...]`
- Values must be integer literals
- Empty lists are allowed: `for x in [] { }` (no iterations)

#### Using Iterator Variables

The iterator variable can be used in expressions:

```fcdsl
Signal base = ("signal-B", 10);

for x in 0..4 {
    Entity lamp = place("small-lamp", x, 2);
    # Use iterator in comparisons
    lamp.enable = base > x * 2;
}

for y in 0..3 {
    # Multiple entities per iteration
    Entity lampA = place("small-lamp", y * 2, 4);
    Entity lampB = place("small-lamp", y * 2 + 1, 4);
    lampA.enable = count > y;
    lampB.enable = count > y;
}
```

**Important:** Iterator variables are **compile-time constants** - they are substituted directly into expressions during loop unrolling. The iterator variable is scoped to the loop body.

#### Compile-Time Unrolling

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

**Use Cases:**
- Creating arrays of entities (lamps, inserters, etc.)
- Generating repetitive signal processing logic
- Building multi-channel displays

---

## Memory System

Memory in the DSL models **persistent state** using write-gated latch circuits or optimized arithmetic feedback loops. The compiler automatically chooses the most efficient implementation based on usage patterns.

### Memory Declaration

```fcdsl
Memory counter: "signal-A";              # Explicit type
Memory state;                            # Implicit type (inferred from first write)
```

### Memory Operations

#### Reading Memory

```fcdsl
Signal current = counter.read();
```

Returns the current stored value as a Signal with the memory's declared type.

#### Writing Memory

```fcdsl
memory.write(value_expr);                  # Unconditional
memory.write(value_expr, when=condition);  # Conditional
```

**Parameters:**
- `value_expr`: Expression to store (must match memory's signal type)
- `when` (optional): Condition controlling the write (default: always write)

**Examples:**

```fcdsl
# Unconditional write (every tick)
counter.write(counter.read() + 1);

# Conditional write
Signal should_update = input > threshold;
buffer.write(new_value, when=should_update);
```

#### Write Enable Semantics

The `when` parameter controls the `signal-W` write enable:

- **When > 0**: Write occurs, value is latched
- **When == 0**: Write blocked, previous value held

If omitted, `when=1` (always write) is used as the default.

### SR/RS Latches (Set/Reset Memory)

For binary state control with explicit set and reset triggers, use the latch write syntax:

```fcdsl
memory.write(value, set=set_condition, reset=reset_condition);
```

**Parameters:**
- `value`: The value to output when latched ON (typically 1, or any constant/signal)
- `set=`: Condition that turns the latch ON
- `reset=`: Condition that turns the latch OFF

**Priority is determined by argument order:**
- `set=..., reset=...` → **SR latch** (set priority: set wins ties)
- `reset=..., set=...` → **RS latch** (reset priority: reset wins ties)

#### RS Latch (Reset Priority)

When both set AND reset are active simultaneously, the latch **resets** (outputs 0):

```fcdsl
Memory pump_on: "signal-P";
Signal low_tank = tank_level < 20;
Signal high_tank = tank_level >= 80;

# RS latch: reset= comes first → reset priority
pump_on.write(1, reset=high_tank, set=low_tank);

Entity pump = place("pump", 0, 0);
pump.enable = pump_on.read() > 0;
```

Uses single-condition decider: `Set > Reset`. Compiles to 1 combinator.

#### SR Latch (Set Priority)

When both set AND reset are active simultaneously, the latch **stays ON**:

```fcdsl
Memory pump_on: "signal-P";
Signal low_tank = tank_level < 20;
Signal high_tank = tank_level >= 80;

# SR latch: set= comes first → set priority
pump_on.write(1, set=low_tank, reset=high_tank);
```

Uses multi-condition decider with wire filtering. Compiles to 1 combinator.

#### Hysteresis Control

Latches are ideal for **hysteresis** – preventing rapid on/off cycling around a threshold.

**Example: Backup Steam Power**

Turn on steam generators when accumulators drop below 20%, keep running until they reach 80%:

```fcdsl
Signal accum = ("signal-A", 0);  # From accumulator
Memory steam_on: "signal-S";

steam_on.write(1, set=accum < 20, reset=accum >= 80);

Entity steam_switch = place("power-switch", 0, 0);
steam_switch.enable = steam_on.read() > 0;
```

The latch prevents flickering: once ON, it stays ON until the high threshold is reached.

#### Output Values

Latches output the specified value when ON, 0 when OFF:

```fcdsl
# Output 1 when latched (binary)
pump.write(1, set=low, reset=high);

# Output 100 when latched (adds a multiplier combinator)
pump.write(100, set=low, reset=high);

# Output a signal value when latched
pump.write(speed_setting, set=low, reset=high);
```

**Implementation:**
- `value=1` → 1 combinator (latch only)
- `value≠1` → 2 combinators (latch + multiplier)

### Memory Implementation Details

#### Write-Gated Latch Architecture (Standard)

For conditional writes, the compiler generates a **write-gated latch** (also known as a sample-and-hold latch) using two decider combinators:

**Combinator Configuration:**
```
Write Gate (Decider): if (signal-W > 0) then output = data_signal
Hold Gate (Decider):  if (signal-W == 0) then output = data_signal
```

**Wire Color Strategy:**
- **RED wires**: Data and feedback signals (e.g., signal-A, signal-B, iron-plate)
- **GREEN wires**: Control signal (signal-W write enable)

This wire color separation is **critical** for correctness. If control and data signals used the same wire color, they would sum at combinator inputs, causing incorrect behavior.

**Feedback Topology:**

The write-gated latch uses a **unidirectional forward feedback + self-loop** topology:

1. **Data Input** → Write Gate (RED wire)
2. **Write Gate Output** → Hold Gate Input (RED wire, forward feedback)
3. **Hold Gate Output** → Hold Gate Input (RED wire, self-loop)
4. **Control Signal (signal-W)** → Both Gates (GREEN wire)

**Why This Works:**

- **Write Phase** (signal-W > 0): Write gate outputs the new data value, which feeds into hold gate
- **Hold Phase** (signal-W == 0): Hold gate outputs its stored value, which feeds back to itself (self-loop), maintaining the value
- **Key Design**: Write gate never receives feedback, preventing value accumulation

#### Arithmetic Feedback Optimization

For **unconditional writes** that read the same memory:

```fcdsl
Memory counter: "signal-A";
counter.write(counter.read() + 1);  # Always-on counter
```

The compiler optimizes this to **arithmetic combinators with feedback loops** instead of the two-decider latch. This saves space and reduces tick delay.

**When This Applies:**
- `when=1` or omitted (unconditional write)
- Value expression (directly or indirectly) depends on reading from the same memory
- The dependency chain can be arbitrarily long (up to 50 operations deep)

**Results:**

**Single-Operation Pattern:**
```fcdsl
Memory counter: "signal-A";
counter.write(counter.read() + 1);
```
Result: One arithmetic combinator with red self-feedback wire instead of two deciders.

**Multi-Operation Chain Pattern:**
```fcdsl
Memory pattern_gen: "signal-D";
Signal step1 = pattern_gen.read() + 1;
Signal step2 = step1 * 3;
Signal step3 = step2 % 17;
Signal final = step3 % 100;
pattern_gen.write(final);
```
Result: Chain of arithmetic combinators with a feedback wire from the last combinator's output back to the first combinator's input (e.g., final → step1).

### Reserved Signals

**CRITICAL:** The `signal-W` virtual signal is **reserved** for memory write enables. Never use `signal-W` for user data:

```fcdsl
Signal bad = ("signal-W", 5);  # COMPILE ERROR
Signal ok = ("signal-A", 5);   # OK
```

The compiler enforces this at compile time.

**Wire Color Assignment:** The `signal-W` control signal is automatically routed on **GREEN wires** to all memory gates, while data and feedback signals use **RED wires**. This color separation prevents signal summation at combinator inputs, which would cause incorrect memory behavior.

---

## Entity System

Entities represent physical Factorio structures placed in the blueprint. The DSL lets you place entities and control them via circuit networks.

### Entity Placement

```fcdsl
Entity entity_name = place(prototype, x, y, [properties]);
```

**Parameters:**
- `prototype`: String literal entity name (e.g., `"small-lamp"`)
- `x`, `y`: Integer coordinates or signal expressions
- `properties` (optional): Dictionary of static prototype properties

**Examples:**

```fcdsl
Entity lamp = place("small-lamp", 5, 0);
Entity train_stop = place("train-stop", 10, 5, {station: "Iron Pickup"});
Entity assembler = place("assembling-machine-1", 0, 10);
```

#### Position Parameters

The position parameters (`x`, `y`) determine how the entity is placed:

**Fixed Positions:** When both coordinates are constant integers (literals or constant integer variables), the entity is placed at the exact specified position:

```fcdsl
int lamp_y = 20;
Entity lamp1 = place("small-lamp", 0, lamp_y);  # Fixed at (0, 20)
Entity lamp2 = place("small-lamp", 5, 10);      # Fixed at (5, 10)
```

**Optimized Positions:** When either coordinate is a dynamic signal expression (not a compile-time constant), or when coordinates are omitted, the layout engine automatically optimizes the entity's position based on signal flow and wire connections:

```fcdsl
Signal x_pos = counter * 2;
Entity lamp3 = place("small-lamp", x_pos, 0);  # Position optimized by layout engine
```

**Best Practice:** Use fixed positions when you need entities at specific coordinates (e.g., for visual alignment or testing). Let the layout engine optimize positions for functional circuits.

#### Static vs Dynamic Properties

**Static properties** (in the dictionary) are applied at entity creation:

```fcdsl
Entity station = place("train-stop", 10, 5, {
    station: "Central Depot",
    color: {r: 1, g: 0, b: 0}
});
```

**Dynamic properties** (via circuit signals) are controlled after placement:

```fcdsl
Entity lamp = place("small-lamp", 5, 0);
lamp.enable = signal > 0;  # Circuit control
```

### Property Access

#### Writing Properties

```fcdsl
lamp.enable = signal > 0;         # Circuit-controlled enable
train_stop.manual_mode = 1;       # Constant value
assembler.enable = production_flag;
```

The compiler translates property assignments into Factorio circuit conditions.

#### Reading Properties

```fcdsl
Signal lamp_status = lamp.enable;  # Read entity state
```

Property reads create signals that track the entity's current state.

#### Reading Entity Circuit Output

Entities like chests, tanks, and storage units output their contents as circuit signals. Access this using the `.output` property:

```fcdsl
Entity chest = place("steel-chest", 0, 0);
Bundle contents = chest.output;  # All item signals from chest

# Use the bundle for operations
Signal iron = contents["iron-plate"];
Signal total = any(contents);
Bundle doubled = contents * 2;

# Control other entities based on chest contents
Entity lamp = place("small-lamp", 2, 0);
lamp.enable = all(contents) > 100;  # Light when all items > 100
```

**Supported Entities:**
- `steel-chest`, `iron-chest`, `wooden-chest` - Item counts
- `storage-tank` - Fluid levels
- `roboport` - Logistics network contents
- Any entity with circuit network output

The `.output` property returns a **dynamic Bundle** whose signals are determined at runtime by the entity's contents.

### Inline Comparison Optimization

For simple comparisons, the compiler **inlines** them into the entity's circuit condition:

```fcdsl
Entity lamp = place("small-lamp", 5, 0);
lamp.enable = count > 10;
```

Instead of creating a separate decider combinator, the compiler configures the lamp's built-in circuit condition to `count > 10`. This saves combinators and reduces tick delay.

**When This Applies:**
- Property is `enable`
- Value is a simple comparison (`signal OP constant`)
- Comparison isn't used elsewhere

#### Bundle Condition Inlining

When using `all()` or `any()` in an enable condition, the compiler inlines the check directly into the entity's circuit condition:

```fcdsl
Entity chest = place("steel-chest", 0, 0);
Entity lamp = place("small-lamp", 2, 0);

Bundle items = chest.output;
lamp.enable = all(items) > 100;  # Inlined: signal-everything > 100
```

Instead of creating a decider combinator with `signal-everything`, the lamp's circuit condition is set directly to use `signal-everything > 100`. This saves one combinator.

**When This Applies:**
- Property is `enable`
- Value is `all(bundle) OP constant` or `any(bundle) OP constant`
- Comparison operator is: `<`, `<=`, `>`, `>=`, `==`, `!=`

### Common Entity Properties

#### Lamps
```fcdsl
lamp.enable = condition;  # Turn on/off
```

**Color Control:**

Lamps can display colored light using RGB signals. First, enable color mode in the placement properties:

```fcdsl
Entity lamp = place("small-lamp", 0, 0, {use_colors: 1, always_on: 1, color_mode: 1});
lamp.r = red_signal;    # Red component (0-255)
lamp.g = green_signal;  # Green component (0-255)
lamp.b = blue_signal;   # Blue component (0-255)
```

**Lamp Properties:**
- `use_colors`: Enable RGB color mode (1 = on, 0 = off)
- `always_on`: Lamp stays on regardless of circuit condition (1 = on, 0 = off)
- `color_mode`: Use custom color mode (1 = on, 0 = off)
- `r`, `g`, `b`: Circuit-controlled color components (0-255 each)

#### Train Stops
```fcdsl
train_stop.manual_mode = 1;        # Enable/disable trains
train_stop.enable = condition;      # Enable/disable station
train_stop.read_from_train = 1;     # Read train contents
train_stop.send_to_train = 1;       # Send signals to train
```

#### Assembling Machines
```fcdsl
assembler.enable = condition;       # Enable/disable production
```

#### Inserters
```fcdsl
inserter.enable = condition;        # Enable/disable
```

#### Belts
```fcdsl
belt.enable = condition;            # Enable/disable belt movement
```

---

## Functions and Modules

### Function Declarations

```fcdsl
func function_name(type1 param1, type2 param2, ...) {
    # Function body
    return expression;
}
```

**Parameter Types:**
- `int` - Integer value
- `Signal` - Circuit signal value
- `Entity` - Reference to a placed entity

**Example:**

```fcdsl
func clamp(Signal value, int min_val, int max_val) {
    Signal too_low = value < min_val;
    Signal too_high = value > max_val;
    Signal result = too_low * min_val 
                  + too_high * max_val 
                  + (!too_low && !too_high) * value;
    return result;
}
```

**Restrictions:**
- `Memory` cannot be used as a parameter type (stateful, incompatible with inlining)
- Recursive functions are not supported (functions are inlined at call sites)

### Function Inlining

**IMPORTANT:** Functions are **always inlined** at call sites. They don't create reusable circuit modules; they're templates for code generation.

```fcdsl
Signal x = clamp(input, 0, 100);
Signal y = clamp(other, 10, 50);
```

This generates **two separate** copies of the clamping logic in the IR.

### Type Coercion for Parameters

Signal and int types can be implicitly converted:

```fcdsl
func process(Signal s, int n) {
    # s can receive both Signal and int
    # n can receive both int and Signal
    return s + n;
}

Signal a = ("iron-plate", 50);
int b = 10;

Signal result1 = process(a, b);     # OK: Signal, int
Signal result2 = process(b, a);     # OK: int coerced to Signal, Signal coerced to int
Signal result3 = process(100, 200); # OK: int coerced to Signal, int stays int
```

### Entity Parameters

Entity parameters allow functions to operate on existing entities:

```fcdsl
func configure_lamp(Entity lamp, Signal red, Signal green, Signal blue) {
    lamp.r = red;
    lamp.g = green;
    lamp.b = blue;
}

Entity my_lamp = place("small-lamp", 0, 0, {use_colors: 1});
configure_lamp(my_lamp, 255, 0, 0);  # Configure it to be red
```

### Local Variables

Variables declared in functions are local to that scope:

```fcdsl
func compute(Signal input) {
    Signal temp = input * 2;  # Local to function
    Signal result = temp + 10;
    return result;
}
```

### Returning Entities from Functions

Functions can return entities placed within them. This is useful for creating entity factory functions:

```fcdsl
func place_colored_lamp(int x, int y, Signal red, Signal green, Signal blue) {
    Entity lamp = place("small-lamp", x, y, {use_colors: 1, always_on: 1, color_mode: 1});
    lamp.r = red;
    lamp.g = green;
    lamp.b = blue;
    return lamp;
}

# Use the factory function
Entity my_lamp = place_colored_lamp(0, 0, 255, 0, 0);  # Red lamp
```

When a function returns an entity, the caller can assign it to an `Entity` variable and continue to access its properties.

### Memory in Functions

Functions can declare local memory, but remember each call site gets its own copy:

```fcdsl
func counter() {
    Memory count: "signal-C";
    count.write(count.read() + 1);
    return count.read();
}

Signal count1 = counter();  # Separate memory instance
Signal count2 = counter();  # Another separate instance
```

## Compilation Model

### Compilation Stages

```
1. Preprocessing → Inline imports (C-style)
2. Parsing → Generate AST from source
3. Semantic Analysis → Type inference, validation
4. IR Generation → Lower AST to intermediate representation
5. Optimization → CSE, constant folding, memory optimization
6. Layout Planning → Assign entity positions, plan wire routing
7. Blueprint Emission → Generate Factorio JSON
```

### Intermediate Representation (IR)

The compiler generates IR nodes representing Factorio combinators:

**Value-Producing Operations:**
- `IR_Const`: Constant combinator
- `IR_Arith`: Arithmetic combinator (+, -, *, /, %)
- `IR_Decider`: Decider combinator (comparisons)
- `IR_WireMerge`: Virtual node for wire-only merging
- `IR_MemRead`: Memory read operation
- `IR_EntityPropRead`: Entity property read

**Effect Operations:**
- `IR_MemCreate`: Memory cell creation
- `IR_MemWrite`: Memory write operation
- `IR_PlaceEntity`: Entity placement
- `IR_EntityPropWrite`: Entity property assignment

### Signal Type Mapping

The compiler maintains a **signal type registry** that maps DSL signal identifiers to Factorio signal names:

```
__v1 → signal-A (virtual)
__v2 → signal-B (virtual)
iron-plate → iron-plate (item)
water → water (fluid)
```

This mapping is exported in debug metadata for troubleshooting.

---

## Circuit Network Integration

### How DSL Maps to Factorio

#### Signal Types

Factorio recognizes several signal categories:

**Virtual Signals:** `signal-A`, `signal-B`, ..., `signal-Z`, `signal-0`, ..., `signal-9`, `signal-everything`, `signal-anything`, `signal-each`

**Color Signals:** `signal-red`, `signal-green`, `signal-blue`, `signal-yellow`, `signal-magenta`, `signal-cyan`, `signal-white`, `signal-grey`, `signal-black`, `signal-pink`

**Item Signals:** `iron-plate`, `copper-plate`, `electronic-circuit`, etc.

**Fluid Signals:** `water`, `crude-oil`, `petroleum-gas`, `steam`, etc.

**Recipe Signals:** Correspond to crafting recipes

**Entity Signals:** Correspond to entity prototypes

The DSL respects these categories and validates signal names against the Factorio database (via Draftsman).

#### Wire Colors

Factorio has two circuit wire colors: **red** and **green**.

The compiler's **wire router** automatically assigns colors to avoid conflicts when multiple sources produce the same signal type to the same destination. Additionally, the compiler uses wire colors strategically for memory systems:

**Memory Wire Color Strategy:**
- **RED wires**: Data signals and feedback loops (e.g., signal-A, signal-B, iron-plate)
- **GREEN wires**: Control signals (signal-W for memory write enable)

This separation prevents signal summation at combinator inputs. For example, if a memory's data signal (signal-B) and control signal (signal-W) both arrived on RED wires at the write gate, they would sum together, causing incorrect behavior.

**Example of Automatic Color Assignment:**

```fcdsl
Signal a = ("signal-A", 10);
Signal b = ("signal-A", 20);
Signal c = a * b;
```

If both `a` and `b` feed into the same combinator on the same wire color, they'd merge and just add up. The compiler detects this and assigns different colors (red for `a`, green for `b`) to keep them separate so that the multiplication can be executed correctly.

**Merge Conflict Detection (Transitive Conflicts):**

The compiler detects complex conflicts involving bundle merges. When the same source participates in multiple merges with a **transitive relationship** (one merge's output feeds into another merge), the paths need different wire colors:

```fcdsl
Bundle sum = {c1.output, c2.output, c3.output};  # Merge M1: all chests
Bundle neg_avg = sum / -3;                        # Computes negative average
Bundle input1 = {neg_avg, c1.output};             # Merge M2: avg + chest1

# c1.output reaches input1 via TWO paths:
# 1. Direct: c1 → input1 (individual content)
# 2. Indirect: c1 → sum → neg_avg → input1 (contributes to average)
```

Without color separation, `c1.output` would be **double-counted**—once directly, once through the average. The compiler tracks:

1. **Merge membership**: Which sources participate in which merges
2. **Transitive paths**: Whether a merge's sink is a source in another merge

When a transitive conflict is detected, the compiler assigns:
- `c1.output → combinator` (merge M1): **RED wire**
- `c1.output → inserter1` (merge M2): **GREEN wire**

This pattern is commonly used in **balanced loaders** (MadZuri pattern) where each inserter needs to compare its individual chest against the average of all chests.

### Edge Layout Conventions

The compiler uses spatial conventions for blueprint organization:

**North Edge (Y < 0):** Constant combinators for literals  
**South Edge (Y > 0):** Export anchors for dangling outputs (`Signal` that is explicitly defined but not consumed, used for e.g. outputs)<br>
**Center (Y ≈ 0):** Logic combinators (arithmetic, deciders, memory)

This creates visually organized blueprints where inputs are at the top and outputs are at the bottom.

### Wire Routing and Relays

Factorio circuit wires have a **maximum span of 9 tiles**. For connections exceeding this distance, the compiler automatically inserts **medium electric poles** as relay points.

**Relay Insertion Algorithm:**
1. Calculate distance between source and sink
2. If distance > 9 tiles, compute intermediate relay positions
3. Reserve positions in layout grid
4. Insert poles and wire them in sequence

This happens transparently; you never need to manually place relay poles.

---

## Optimization Passes

The compiler applies several optimization passes to reduce entity count and improve performance.

### Common Subexpression Elimination (CSE)

Identical expressions are computed once and reused:

```fcdsl
Signal x = a + b;
Signal y = a + b;  # Reuses x's combinator
```

The IR optimizer detects that both expressions are identical and makes `y` reference `x`'s output instead of creating a duplicate combinator.

### Condition Folding

Logical AND/OR chains of comparisons are folded into a single multi-condition decider combinator:

```fcdsl
Signal a = ("signal-A", 0);
Signal b = ("signal-B", 0);
Signal c = ("signal-C", 0);

# Without optimization: 2 deciders + 1 arithmetic = 3 combinators
# With optimization: 1 multi-condition decider
Signal result = (a > 5) && (b < 10) && (c == 3);
```

**Foldable patterns:**
- Two or more comparisons chained with `&&` (AND)
- Two or more comparisons chained with `||` (OR)
- Signal vs constant comparisons (`a > 5`)
- Signal vs signal comparisons (`a > b`)

**Non-foldable patterns (use traditional lowering):**
- Mixed `&&` and `||` operators: `(a > 0) && (b > 0) || (c > 0)`
- Complex operands requiring computation: `((a + b) > 0) && (c > 0)`
- Non-comparison operands: `a && b` (where `a` and `b` are signals)

This optimization takes advantage of Factorio 2.0's multi-condition decider combinators, which can evaluate up to 8 conditions in a single entity.

### Wire Merge Optimization

When adding **simple sources** (constants, entity outputs, other wire merges) on the same channel, the compiler skips creating an arithmetic combinator and wires them directly:

```fcdsl
Signal iron_a = ("iron-plate", 100);  # Constant
Signal iron_b = ("iron-plate", 200);  # Constant
Signal iron_c = ("iron-plate", 50);   # Constant

Signal total = iron_a + iron_b + iron_c;  # No combinator! Just wires.
```

**Requirements for Wire Merge:**
- All operands are simple sources (constants, entity reads, or wire merges)
- All operands have the same signal type
- Operator is `+`
- No operand is used twice (e.g., `a + a` still needs a combinator)

### Constant Folding

Arithmetic in signal literals is evaluated at compile time:

```fcdsl
Signal step = ("signal-A", 5 * 2 - 9);  # Becomes ("signal-A", 1)
```

### Projection Elimination

Same-type projections are no-ops and are optimized away:

```fcdsl
Signal iron = ("iron-plate", 100);
Signal same = iron | "iron-plate";  # No combinator generated
```

### Memory Optimization

As described earlier, unconditional memory writes with feedback are optimized to arithmetic combinators with feedback loops:

```fcdsl
# Single-operation: self-feedback on one combinator
Memory counter: "signal-A";
counter.write(counter.read() + 1);

# Multi-operation: feedback wire from last to first combinator
Memory pattern: "signal-B";
Signal s1 = pattern.read() + 1;
Signal s2 = s1 * 3;
Signal s3 = s2 % 17;
pattern.write(s3);  # Creates feedback loop: s3 → s1
```

The compiler detects dependency chains up to 50 operations deep and optimizes them automatically.

### Entity Property Inlining

Simple comparisons in entity property assignments are compiled to circuit conditions instead of separate combinators:

```fcdsl
lamp.enable = count > 10;  # No decider; uses lamp's circuit condition
```

---

## Best Practices

### Signal Type Management

**DO:**
```fcdsl
Signal iron = ("iron-plate", 100);        # Explicit types for clarity
Signal copper = ("copper-plate", 50);
Signal total = (iron | "signal-total") 
             + (copper | "signal-total");  # Aggregate with projections
```

**DON'T:**
```fcdsl
Signal iron = ("iron-plate", 100);
Signal copper = ("copper-plate", 50);
Signal mixed = iron + copper;  # Warning: Mixed types, left wins
```

### Memory Usage

**DO:**
```fcdsl
Memory counter: "signal-A";  # Explicit type is best
counter.write(counter.read() + 1);
```

**DON'T:**
```fcdsl
Memory counter;  # Implicit type is confusing
counter.write(value);  # Hard to track what type counter uses
```

### Function Design

**DO:**
```fcdsl
func clamp(Signal value, int min_val, int max_val) {
    Signal result = (value < min_val) * min_val
                  + (value > max_val) * max_val
                  + ((value >= min_val) && (value <= max_val)) * value;
    return result;
}
```

**DON'T:**
```fcdsl
func stateful_function(Signal input) {
    Memory state;  # Each call creates separate memory!
    # ...
}
```

### Entity Control

**DO:**
```fcdsl
Entity lamp = place("small-lamp", 5, 0);
lamp.enable = count > threshold;  # Compiler inlines comparison
```

**DON'T:**
```fcdsl
Signal enable = count > threshold;  # Creates extra decider
Entity lamp = place("small-lamp", 5, 0);
lamp.enable = enable;
```

### Projection Patterns

**DO:**
```fcdsl
# Aggregate onto one channel
Signal total = (iron | "signal-T") + (copper | "signal-T");

# Extract specific channel (explicit)
Signal iron_only = total | "iron-plate";
```

**DON'T:**
```fcdsl
Signal mixed = iron + copper;  # Loses copper's type
Signal extracted = mixed | "copper-plate";  # Can't recover lost info
```

---

## Examples

### Basic Counter

```fcdsl
Memory counter: "signal-A";
counter.write(counter.read() + 1);

Signal output = counter.read() | "signal-output";
```

### Blinking Lamps

```fcdsl
Memory tick: "signal-T";
tick.write(tick.read() + 1);

Signal pattern = tick.read() % 20;

Entity lamp1 = place("small-lamp", 0, 0);
Entity lamp2 = place("small-lamp", 2, 0);

lamp1.enable = pattern < 10;
lamp2.enable = pattern >= 10;
```

### Production Controller

```fcdsl
# Inputs (wire these signals from your factory)
Signal demand = ("signal-demand", 0);
Signal supply = ("signal-supply", 0);

# Calculate shortage
Signal shortage = demand - supply;
Signal should_produce = shortage > 0;

# Control assemblers
Entity assembler1 = place("assembling-machine-1", 0, 0);
Entity assembler2 = place("assembling-machine-1", 3, 0);

assembler1.enable = should_produce;
assembler2.enable = shortage > 100;  # Second machine for high demand
```

### Bitwise Operations and other Tricks

```fcdsl
# Using different number bases
Signal binary = 0b11111111;      # 255 in binary
Signal octal = 0o377;            # 255 in octal
Signal hex = 0xFF;               # 255 in hexadecimal
Signal dec = 255;                # 255 in decimal
# All four represent the same value

# Bit masking for RGB color channels
Signal color_value = 0xFF8040;   # RGB color as hex
Signal red = (color_value >> 16) AND 0xFF;    # Extract red channel
Signal green = (color_value >> 8) AND 0xFF;   # Extract green channel
Signal blue = color_value AND 0xFF;           # Extract blue channel

# M ultiplication using shifts
Signal base = 10;
Signal doubled = base << 1;      # 20 (multiply by 2)
Signal quadrupled = base << 2;   # 40 (multiply by 4)

# Power calculations
Signal area = side ** 2;         # Square
Signal volume = side ** 3;       # Cube
Signal compound = 2 ** 10;       # 1024

# Bitwise flags for state combinations
Signal flag_a = 0b0001;
Signal flag_b = 0b0010;
Signal flag_c = 0b0100;
Signal combined_state = flag_a OR flag_b;     # 0b0011
Signal has_both = combined_state AND flag_a;  # Check if flag_a is set

# Logical operators with word forms
Signal valid = (value > 0) and (value < 100);    # Range check
Signal active = enabled or override;             # Either condition
```

### State Machine

```fcdsl
Memory state: "signal-S";

Signal current_state = state.read();

# Inputs (wire from factory)
Signal start_signal = ("signal-start", 0);
Signal stop_signal = ("signal-stop", 1);
Signal error_signal = ("signal-error", 2);

# State transitions
Signal next_state = 
    (current_state == 0 && start_signal > 0) * 1 +      # Idle → Running
    (current_state == 1 && stop_signal > 0) * 2 +       # Running → Stopped
    (current_state == 1 && error_signal > 0) * 3 +      # Running → Error
    (current_state == 2 && start_signal > 0) * 1 +      # Stopped → Running
    (current_state == 3 && start_signal > 0) * 0 +      # Error → Idle
    # Stay in current state if no transition matches
    ((current_state == 0 && start_signal == 0) ||
     (current_state == 1 && stop_signal == 0 && error_signal == 0) ||
     (current_state == 2 && start_signal == 0) ||
     (current_state == 3 && start_signal == 0)) * current_state;

state.write(next_state);

# Outputs based on state
Signal running = (current_state == 1) | "signal-running";
Signal stopped = (current_state == 2) | "signal-stopped";
Signal error = (current_state == 3) | "signal-error";
```

### Balanced Train Loader (MadZuri Pattern)

A classic Factorio circuit pattern that balances item distribution across multiple chests. Each inserter is enabled only when its chest contains fewer items than the average:

```fcdsl
# Place chests along the train cargo wagons
Entity c1 = place("steel-chest", 0, 0);
Entity c2 = place("steel-chest", 1, 0);
Entity c3 = place("steel-chest", 2, 0);

# Place inserters to load/unload the chests
Entity i1 = place("fast-inserter", 0, 1);
Entity i2 = place("fast-inserter", 1, 1);
Entity i3 = place("fast-inserter", 2, 1);

# Sum all chest contents and compute negative average
Bundle total = {c1.output, c2.output, c3.output};
Bundle neg_avg = total / -3;  # Negative of the average

# Each inserter receives: its own chest + negative average
# Result is positive only if chest is below average
Bundle in1 = {neg_avg, c1.output};  # c1.output - avg
Bundle in2 = {neg_avg, c2.output};  # c2.output - avg
Bundle in3 = {neg_avg, c3.output};  # c3.output - avg

# Enable inserter only when chest is below average (all item types)
i1.enable = in1 < 0;  # Multi-signal comparison
i2.enable = in2 < 0;
i3.enable = in3 < 0;
```

The compiler automatically detects the **transitive conflict** where each chest's signal appears in both:
1. The `total` merge (contributing to the average)
2. Its individual `in1/in2/in3` merge (direct comparison)

Different wire colors are assigned to prevent double-counting:
- Chest → combinator (total): RED wire
- Chest → own inserter: GREEN wire

### Filter and Accumulator

```fcdsl
func smooth_filter(Signal input, int window_size) {
    Memory sum: "signal-sum";
    Memory count: "signal-count";
    
    Signal current_sum = sum.read();
    Signal current_count = count.read();
    
    Signal should_reset = current_count >= window_size;
    Signal new_sum = should_reset * input + (!should_reset) * (current_sum + input);
    Signal new_count = should_reset * 1 + (!should_reset) * (current_count + 1);
    
    sum.write(new_sum);
    count.write(new_count);
    
    return new_sum / new_count;
}

Signal raw_input = ("iron-plate", 0);
Signal smoothed = smooth_filter(raw_input, 10);
```

### For Loop: Lamp Array

```fcdsl
# Create a row of 10 lamps controlled by a counter
Memory tick: "signal-T";
tick.write(tick.read() + 1);
Signal position = tick.read() % 10;

for i in 0..10 {
    Entity lamp = place("small-lamp", i, 0);
    lamp.enable = position == i;  # Only one lamp on at a time
}
```

### For Loop: Knight Rider Effect

```fcdsl
# Create a bouncing light pattern
Memory counter: "signal-C";
counter.write(counter.read() + 1);

# Create 8 lamps
for i in 0..8 {
    Entity lamp = place("small-lamp", i, 0);
    # Calculate distance from current position (modulo ping-pong)
    Signal pos = counter.read() % 14;  # 0-13 for 8 positions
    Signal actual_pos = (pos < 8) * pos + (pos >= 8) * (14 - pos);
    lamp.enable = actual_pos == i;
}
```

### Signal Type Access: Type Propagation

```fcdsl
# Keep consistent types without repeating type strings
Signal source = ("iron-plate", 100);
Signal offset = 50 | source.type;        # iron-plate type from source
Signal doubled = source * 2;              # inherits iron-plate
Signal result = doubled + offset;         # no type mismatch

# Type-aware factory function
func scaled_signal(Signal input, int factor) {
    Signal scale = factor | input.type;   # Match input's type
    return input * scale;
}

Signal output = scaled_signal(source, 10);
```

### Bundle Processing: Resource Monitor

```fcdsl
# Monitor multiple resource levels
Bundle resources = { 
    ("iron-plate", 0),      # Wired from storage 
    ("copper-plate", 0), 
    ("coal", 0) 
};

# Double all resource readings for display
Bundle display_values = resources * 2;

# Check if any resource is low (< 100)
Signal alert = any(resources) < 100;

# Check if all resources are present (> 0)
Signal all_present = all(resources) > 0;

# Control warning lamp
Entity warning = place("small-lamp", 0, 0);
warning.enable = alert;
```

---

## Error Handling

### Warning Types

#### Mixed Type Warnings

```fcdsl
Signal iron = ("iron-plate", 100);
Signal copper = ("copper-plate", 50);
Signal mixed = iron + copper;
# Warning: Mixed signal types in binary operation: 'iron-plate' + 'copper-plate'
# Result will be 'iron-plate'. Use | "type" to explicitly set output channel.
```

**Fix:**
```fcdsl
Signal aligned = (copper | "iron-plate") + iron;  # Both iron-plate
```

#### Unknown Signal Warnings

```fcdsl
Signal unknown = ("nonexistent-signal", 0);
# Warning: Unknown signal type: nonexistent-signal, using signal-0
```

The compiler validates signals against the Factorio database. If a signal doesn't exist, it falls back to `signal-0`.

### Error Types

#### Undefined Variables

```fcdsl
Signal result = undefined_var + 5;
# Error: Undefined variable 'undefined_var'
```

#### Type Mismatches

```fcdsl
Entity lamp = 42;
# Error: Cannot assign int to Entity type
```

#### Memory Type Conflicts

```fcdsl
Memory buffer: "iron-plate";
Signal copper = ("copper-plate", 50);
buffer.write(copper);
# Error: Type mismatch: Memory 'buffer' expects 'iron-plate' but write provides 'copper-plate'
```

#### Reserved Signal Violations

```fcdsl
Signal bad = ("signal-W", 5);
# Error: Signal 'signal-W' is reserved for memory write-enable and cannot be used in signal literals
```

### Strict Mode

Compile with `--strict` to promote all warnings to errors:

```bash
python compile.py program.fcdsl --strict
```

This enforces stricter type checking and catches potential bugs earlier.

### Diagnostic Output

```
Compiling example.fcdsl...
Diagnostics:
  WARNING [semantic:example.fcdsl:12]: Mixed signal types in binary operation
  ERROR [lowering:example.fcdsl:25]: Undefined memory 'invalid_mem' in read operation

Compilation failed: Semantic analysis failed
```

Diagnostics include:
- Severity (INFO, WARNING, ERROR)
- Stage (parsing, semantic, lowering, layout, emission)
- Location (file:line:column)
- Message

---

## Compiler Usage

### Basic Compilation

```bash
python compile.py input.fcdsl
```

Prints blueprint string to stdout.

### Save to File

```bash
python compile.py input.fcdsl -o output.blueprint
```

### Strict Type Checking

```bash
python compile.py input.fcdsl --strict
```

### Debug Logging

```bash
python compile.py input.fcdsl --log-level debug
```

### Extended Diagnostics

```bash
python compile.py input.fcdsl --explain
```

### Add Power Poles

```bash
python compile.py input.fcdsl --power-poles medium
```

Options: `small`, `medium`, `big`, `substation`

### Custom Blueprint Name

```bash
python compile.py input.fcdsl --name "My Factory Controller"
```

### Disable Optimizations

```bash
python compile.py input.fcdsl --no-optimize
```

### Output JSON Format

```bash
python compile.py input.fcdsl --json
```

Outputs raw blueprint JSON instead of base64-encoded blueprint string.


**Happy building!**