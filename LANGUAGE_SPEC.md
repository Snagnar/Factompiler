# Factorio Circuit DSL Language Specification

**Version 2.2**  
**Date: December 8, 2025**

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
write(read(counter) + 1, counter);

Signal blink = (read(counter) % 10) < 5;

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
```fcdsl
42        # Positive integer
-17       # Negative integer
```

**Strings:**
```fcdsl
"iron-plate"    # Item signal
"signal-A"      # Virtual signal
"water"         # Fluid signal
```

Strings are used exclusively for signal type names and entity prototypes.

### Operators

**Arithmetic:** `+` `-` `*` `/` `%`  
**Comparison:** `==` `!=` `<` `<=` `>` `>=`  
**Logical:** `&&` `||` `!`  
**Projection:** `|`  
**Assignment:** `=`

### Operator Precedence

From highest to lowest:

1. **Parentheses** `()`
2. **Unary** `+` `-` `!`
3. **Multiplicative** `*` `/` `%`
4. **Additive** `+` `-`
5. **Projection** `|`
6. **Comparison** `==` `!=` `<` `<=` `>` `>=`
7. **Logical AND** `&&`
8. **Logical OR** `||`

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

The DSL has four fundamental value types that map directly to Factorio circuit concepts.

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

Stateful storage cells that persist values across game ticks using SR latch circuits.

```fcdsl
Memory counter: "signal-A";             # Explicit type
Memory state;                           # Implicit type (inferred from first write)
```

#### Type Inference

If you don't specify a type, the compiler infers it from the first `write()`:

```fcdsl
Memory accumulator;  # Type unknown

Signal iron = ("iron-plate", 50);
write(iron, accumulator);  # Now accumulator stores iron-plate signals
```

**Warning:** All writes to a memory cell must use the same signal type. Mixed types will generate warnings (or errors in `--strict` mode).

### Entity (`Entity`)

References to placed Factorio entities that can be controlled via circuit networks.

```fcdsl
Entity lamp = place("small-lamp", 5, 0);
Entity train_stop = place("train-stop", 10, 5);
```

Entities expose **properties** that can be read from or written to using circuit signals.

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
func process(value) {
    Signal typed_value = value | "signal-S";  # Annotate with signal-S
    Memory storage: "signal-S";
    write(typed_value, storage);
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

```fcdsl
Signal both = (a > 0) && (b > 0);   # AND: a * b
Signal either = (a > 0) || (b > 0); # OR: (a + b) > 0
Signal not_zero = !(x == 0);        # NOT: x == 0 ? 0 : 1
```

Logical operations are compiled to arithmetic combinators for efficiency.

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
write(value, memory);  # Memory write side effect
place("lamp", 5, 0);   # Entity placement side effect
```

### Return Statements

Used in functions to specify return values:

```fcdsl
func double(x) {
    return x * 2;
}
```

### Import Statements

```fcdsl
import "stdlib/memory_utils.fcdsl";
```

Imports use **C-style preprocessing**: the imported file's content is inlined before parsing. Circular imports are detected and skipped.

**Note:** The `import "file" as alias` syntax is recognized by the grammar but aliases are not currently utilized since files are textually inlined. All functions from imported files become available in the global namespace.

---

## Memory System

Memory in the DSL models **persistent state** using SR latch circuits or optimized arithmetic feedback loops. The compiler automatically chooses the most efficient implementation based on usage patterns.

### Memory Declaration

```fcdsl
Memory counter: "signal-A";              # Explicit type
Memory state;                            # Implicit type (inferred from first write)
```

### Memory Operations

#### Reading Memory

```fcdsl
Signal current = read(counter);
```

This connects to the memory's output (the hold gate in the SR latch).

#### Writing Memory

```fcdsl
write(value_expr, memory_name, when=enable_signal);
```

**Parameters:**
- `value_expr`: Expression to store (must match memory's signal type)
- `memory_name`: Target memory cell
- `when` (optional): Signal controlling the write (default: `1` = always write)

**Examples:**

```fcdsl
# Unconditional write (every tick)
write(read(counter) + 1, counter);

# Conditional write
Signal should_update = input > threshold;
write(new_value, buffer, when=should_update);
```

#### Write Enable Semantics

The `when` parameter controls the `signal-W` write enable:

- **When > 0**: Write occurs, value is latched
- **When == 0**: Write blocked, previous value held

If omitted, `when=1` (always write) is used as the default.

### Memory Implementation Details

#### SR Latch Architecture (Standard)

For conditional writes, the compiler generates an SR latch with the following architecture:

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

The SR latch uses a **unidirectional forward feedback + self-loop** topology:

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
write(read(counter) + 1, counter);  # Always-on counter
```

The compiler optimizes this to **arithmetic combinators with feedback loops** instead of the two-decider SR latch. This saves space and reduces tick delay.

**When This Applies:**
- `when=1` or omitted (unconditional write)
- Value expression (directly or indirectly) depends on reading from the same memory
- The dependency chain can be arbitrarily long (up to 50 operations deep)

**Results:**

**Single-Operation Pattern:**
```fcdsl
Memory counter: "signal-A";
write(read(counter) + 1, counter);
```
Result: One arithmetic combinator with red self-feedback wire instead of two deciders.

**Multi-Operation Chain Pattern:**
```fcdsl
Memory pattern_gen: "signal-D";
Signal step1 = read(pattern_gen) + 1;
Signal step2 = step1 * 3;
Signal step3 = step2 % 17;
Signal final = step3 % 100;
write(final, pattern_gen);
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
func function_name(param1, param2, ...) {
    # Function body
    return expression;
}
```

**Example:**

```fcdsl
func clamp(value, min_val, max_val) {
    Signal too_low = value < min_val;
    Signal too_high = value > max_val;
    Signal result = too_low * min_val 
                  + too_high * max_val 
                  + (!too_low && !too_high) * value;
    return result;
}
```

### Function Inlining

**IMPORTANT:** Functions are **always inlined** at call sites. They don't create reusable circuit modules; they're templates for code generation.

```fcdsl
Signal x = clamp(input, 0, 100);
Signal y = clamp(other, 10, 50);
```

This generates **two separate** copies of the clamping logic in the IR.

### Parameter Types

Parameters are **untyped** and take the type of the passed argument:

```fcdsl
func double(x) {
    return x * 2;
}

Signal a = ("iron-plate", 50);
Signal b = ("signal-A", 10);

Signal doubled_iron = double(a);    # x is iron-plate
Signal doubled_virtual = double(b); # x is signal-A
```

### Local Variables

Variables declared in functions are local to that scope:

```fcdsl
func compute(input) {
    Signal temp = input * 2;  # Local to function
    Signal result = temp + 10;
    return result;
}
```

### Returning Entities from Functions

Functions can return entities placed within them. This is useful for creating entity factory functions:

```fcdsl
func place_colored_lamp(x, y, red, green, blue) {
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
    write(read(count) + 1, count);
    return read(count);
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
write(read(counter) + 1, counter);

# Multi-operation: feedback wire from last to first combinator
Memory pattern: "signal-B";
Signal s1 = read(pattern) + 1;
Signal s2 = s1 * 3;
Signal s3 = s2 % 17;
write(s3, pattern);  # Creates feedback loop: s3 → s1
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
write(read(counter) + 1, counter);
```

**DON'T:**
```fcdsl
Memory counter;  # Implicit type is confusing
write(value, counter);  # Hard to track what type counter uses
```

### Function Design

**DO:**
```fcdsl
func clamp(value, min_val, max_val) {
    Signal result = (value < min_val) * min_val
                  + (value > max_val) * max_val
                  + ((value >= min_val) && (value <= max_val)) * value;
    return result;
}
```

**DON'T:**
```fcdsl
func stateful_function(input) {
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
write(read(counter) + 1, counter);

Signal output = read(counter) | "signal-output";
```

### Blinking Lamps

```fcdsl
Memory tick: "signal-T";
write(read(tick) + 1, tick);

Signal pattern = read(tick) % 20;

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

### State Machine

```fcdsl
Memory state: "signal-S";

Signal current_state = read(state);

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

write(next_state, state);

# Outputs based on state
Signal running = (current_state == 1) | "signal-running";
Signal stopped = (current_state == 2) | "signal-stopped";
Signal error = (current_state == 3) | "signal-error";
```

### Filter and Accumulator

```fcdsl
func smooth_filter(input, window_size) {
    Memory sum: "signal-sum";
    Memory count: "signal-count";
    
    Signal current_sum = read(sum);
    Signal current_count = read(count);
    
    Signal should_reset = current_count >= window_size;
    Signal new_sum = should_reset * input + (!should_reset) * (current_sum + input);
    Signal new_count = should_reset * 1 + (!should_reset) * (current_count + 1);
    
    write(new_sum, sum);
    write(new_count, count);
    
    return new_sum / new_count;
}

Signal raw_input = ("iron-plate", 0);
Signal smoothed = smooth_filter(raw_input, 10);
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
write(copper, buffer);
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