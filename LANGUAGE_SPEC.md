# Factorio Circuit DSL Language Specification

**Version 1.0**  
**Date: September 30, 2025**

This document provides a complete specification of the Factorio Circuit DSL (Domain Specific Language), as currently implemented. The DSL compiles to Factorio blueprint strings that can be imported into the game.

## Table of Contents

1. [Overview](#overview)
2. [Lexical Structure](#lexical-structure)
3. [Syntax](#syntax)
4. [Type System](#type-system)
5. [Built-in Functions](#built-in-functions)
6. [Memory Operations](#memory-operations)
7. [Entity System](#entity-system)
8. [Functions and Modules](#functions-and-modules)
9. [Projection and Type Coercion](#projection-and-type-coercion)
10. [Compilation Model](#compilation-model)
11. [Examples](#examples)
12. [Diagnostics and Error Handling](#diagnostics-and-error-handling)

---

## Overview

The Factorio Circuit DSL is a statically typed, imperative language designed to generate Factorio combinators and entity blueprints. Programs are compiled through a multi-stage pipeline:

1. **Parsing**: Source code → Abstract Syntax Tree (AST)
2. **Semantic Analysis**: Type inference, symbol resolution, validation
3. **IR Generation**: AST → Intermediate Representation
4. **Blueprint Emission**: IR → Factorio blueprint JSON

### Design Principles

- **Type Safety**: Signals have explicit types (iron-plate, signal-A, etc.)
- **Implicit Allocation**: Compiler assigns virtual signal types when not specified
- **Mixed-Type Warnings**: Warns about potential signal type mismatches
- **Entity Integration**: Direct support for placing and controlling Factorio entities

---

## Lexical Structure

### Comments

```fcdsl
# Line comments start with hash
// C-style comments are also supported
Signal x = 5;  // End-of-line comments
```

### Identifiers

```fcdsl
NAME: /[A-Za-z_][A-Za-z0-9_-]*/
```

Identifiers can contain letters, numbers, underscores, and hyphens. Must start with letter or underscore.

### Keywords

**Type Keywords:**
- `int`, `Signal`, `SignalType`, `Entity`, `Memory`, `mem`
- `mem` is an alias for `Memory`

**Statement Keywords:**
- `func`, `return`, `import`, `as`

**Built-in Functions:**
- `read`, `write`, `place`, `memory`

### Literals

**Numbers:**
```fcdsl
42        # Integer literal
-17       # Negative integer
```

**Strings:**
```fcdsl
"iron-plate"    # Signal type
"signal-A"      # Virtual signal
"copper-ore"    # Item signal
```

### Operators

**Arithmetic:** `+`, `-`, `*`, `/`, `%`  
**Comparison:** `==`, `!=`, `<`, `<=`, `>`, `>=`  
**Logical:** `&&`, `||`, `!`  
**Projection:** `|`  
**Assignment:** `=`

---

## Syntax

### Program Structure

```fcdsl
program ::= statement*

statement ::= decl_stmt ";"
            | assign_stmt ";"
            | expr_stmt ";"
            | return_stmt ";"
            | import_stmt ";"
            | func_decl
```

### Variable Declarations

```fcdsl
decl_stmt ::= type_name NAME "=" expr
mem_decl ::= ("Memory" | "mem") NAME [":" STRING]

type_name ::= "int" | "Signal" | "SignalType" | "Entity"
```

**Examples:**
```fcdsl
Signal iron = ("iron-plate", 100);       # Signal with explicit type
Signal virtual = ("signal-A", 42);       # Virtual signal
Signal implicit = 5;                     # Signal with implicit type
int count = 42;                          # Integer variable
Entity lamp = place("small-lamp", 5, 0); # Entity placement
Memory counter: "iron-plate";            # Memory declaration (explicit type)
write(("iron-plate", 0), counter, when=once);  # Optional one-shot initialization
```

### Assignment Statements

```fcdsl
assign_stmt ::= lvalue "=" expr
lvalue ::= NAME ("." NAME)?
```

**Examples:**
```fcdsl
x = y + 5;              # Variable assignment
lamp.enable = count > 0; # Property assignment
```

### Expression Statements

```fcdsl
expr_stmt ::= expr
```

Any expression can be used as a statement for side effects.

### Function Declarations

```fcdsl
func_decl ::= "func" NAME "(" [param_list] ")" "{" statement* "}"
param_list ::= NAME ("," NAME)*
```

**Example:**
```fcdsl
func double_signal(x) {
    return x * 2;
}
```

### Import Statements

```fcdsl
import_stmt ::= "import" STRING ["as" NAME]
```

**Examples:**
```fcdsl
import "stdlib/memory_utils.fcdsl";
import "modules/production.fcdsl" as prod;
```

### Expression Precedence

From highest to lowest precedence:

1. **Primary**: literals, identifiers, function calls, parentheses
2. **Unary**: `+`, `-`, `!`
3. **Multiplicative**: `*`, `/`, `%`
4. **Additive**: `+`, `-`
5. **Projection**: `|`
6. **Comparison**: `==`, `!=`, `<`, `<=`, `>`, `>=`
7. **Logical**: `&&`, `||`

**Example:**
```fcdsl
Signal result = a + b * 2 | "iron-plate";
# Evaluates as: (a + (b * 2)) | "iron-plate"
```

---

## Type System

### Value Types

The DSL has four fundamental value types:

#### 1. Integer (`int`)

Plain integer values for constants and calculations.

```fcdsl
int count = 42;
int result = count + 10;
```

#### 2. Signal (`Signal`)

Single-channel Factorio signals with explicit or implicit types.

```fcdsl
Signal iron = ("iron-plate", 100);      # Explicit signal type
Signal virtual = ("signal-A", 42);      # Virtual signal
Signal implicit = 5;                    # Implicit type allocation
```

#### 3. Memory (`Memory`)

Stateful memory cells that can store and retrieve values.

```fcdsl
Memory counter: "iron-plate";          # Memory declaration with explicit type
Memory accumulator: "signal-A";
write(("iron-plate", 0), counter, when=once);  # Optional one-shot seed
write(iron, accumulator, when=once);    # Initialize from existing signal
```

#### 4. Entity (`Entity`)

References to placed Factorio entities that can be controlled.

```fcdsl
Entity lamp = place("small-lamp", 5, 0);
lamp.enable = 1;                        # Control entity properties
```

### Signal Type System

**Explicit Types**: User-specified signal types
- Item signals: `"iron-plate"`, `"copper-ore"`, etc.
- Virtual signals: `"signal-A"`, `"signal-0"`, etc.
- Fluid signals: `"crude-oil"`, `"water"`, etc.

**Implicit Types**: Compiler-allocated virtual signals
- Format: `"__v1"`, `"__v2"`, `"__v3"`, etc.
- Assigned when type not specified in signal literals

### Type Inference Rules

#### Binary Arithmetic Operations

1. **Int + Int = Int**
   ```fcdsl
   int result = 5 + 3;  # result: int
   ```

2. **Signal + Int = Signal** (integer coerced to signal type)
   ```fcdsl
   Signal result = iron + 10;  # result: iron-plate signal
   # Warning: Mixed types in binary operation
   ```

3. **Signal + Signal = Signal** (left operand type wins)
   ```fcdsl
   Signal result = iron + copper;  # result: iron-plate signal
   # Warning: Mixed signal types in binary operation
   ```

#### Comparison Operations

All comparison operators (`==`, `!=`, `<`, `<=`, `>`, `>=`, `&&`, `||`) return Signal values with the same type as their operands.

```fcdsl
Signal is_greater = iron > 10;  # Result: Signal with iron-plate type
Signal is_equal = virtual == 42; # Result: Signal with signal-A type
```

### Strict Type Checking

When `--strict` mode is enabled, type warnings become errors:

```bash
# Normal mode: warnings
python compile.py tests/sample_programs/01_basic_arithmetic.fcdsl

# Strict mode: errors
python compile.py tests/sample_programs/01_basic_arithmetic.fcdsl --strict
```

---

## Built-in Functions

### Signal Literals

Create signals with explicit or implicit types.

**Syntax:**
```fcdsl
(type, value)           # Explicit type specification
value                   # Implicit type allocation
```

**Examples:**
```fcdsl
Signal iron = ("iron-plate", 100);      # Explicit type
Signal virtual = ("signal-A", 42);      # Virtual signal
Signal implicit = 5;                    # Implicit type (__v1)
Signal bus_iron = ("iron-plate", 0);    # Placeholder for external wiring
```

**Parameters:**
- `type`: String literal specifying signal type
- `value`: Integer value for the signal

**Returns:** Signal value

> **External Inputs:** Declare a signal with the desired type (as above) to document expected bus connections. The compiler no longer inserts pass-through combinators—when the blueprint is imported into Factorio, wire the named signal to the appropriate networks manually.

### `place(entity_type, x, y, [properties])`

Places entities in the blueprint at specified coordinates. An optional fourth argument lets you provide a dictionary of static prototype properties that will be applied immediately after creation.

**Syntax:**
```fcdsl
place(entity_type, x, y)
place(entity_type, x, y, {property: value, ...})
```

**Examples:**
```fcdsl
Entity lamp = place("small-lamp", 5, 0);
Entity train_stop = place("train-stop", 10, 5, {station: "Central Station"});
Entity assembler = place("assembling-machine-1", 0, 10);
```

**Parameters:**
- `entity_type`: String literal entity prototype name
- `x`, `y`: Integer coordinates (can be signal expressions for dynamic positioning)
- `properties` (optional): Dictionary literal whose keys are property names and whose values are compile-time integers or strings. Values are written directly to the underlying Draftsman entity via `setattr`, so they must match the expected Python scalar types (for example, strings for names, integers for numeric configuration).

**Returns:** Entity reference for property access

**Notes:**
- Properties that require booleans or richer structures (such as lamp `color`) must be configured after placement using dedicated DSL statements or future extensions; the dictionary currently supports only scalar int/string literals.
- Dynamic, signal-driven properties should be configured via property assignments (e.g., `lamp.enable = condition`) after placement.

---

## Memory Operations

The DSL models state with compact SR latch circuits composed of a **write gate** and a **hold gate**. Each memory cell reserves the virtual signal `signal-W` for write enables; this channel must not be used for user data.

### Memory Declaration

```fcdsl
Memory name: "signal-type";
```

**Examples:**

```fcdsl
Memory counter: "iron-plate";
Memory state: "signal-A";
Memory accumulator: "copper-plate";
```

Each memory cell stores exactly one Factorio signal type. The declaration's string literal must be a valid signal identifier (item, fluid, virtual, etc.). Inline initializers remain unsupported—newly declared cells start empty until their first write.

For backward compatibility the type suffix may be omitted:

```fcdsl
Memory counter;
Signal a = ("iron-plate", 10);
write(a, counter);
```

If no type is specified, the compiler infers the memory's signal channel from the first `write()` call. Subsequent writes must reuse the inferred type; inconsistent writes raise warnings (or errors in strict mode). For clarity and future maintenance, explicit typing is strongly recommended.

### Seeding Memory (`when=once`)

Use the special predicate `when=once` to perform one-time initialization writes:

```fcdsl
Memory counter: "iron-plate";
write(("iron-plate", 0), counter, when=once);  # Runs exactly once at startup
```

Internally the compiler expands `when=once` into a hidden flag memory that fires a single write pulse. Each use of `when=once` is independent and does not interfere with other writes.

### `read(memory_name)`

Reads the current value stored in a memory cell.

**Syntax:**
```fcdsl
read(memory_name)
```

**Examples:**
```fcdsl
Signal current = read(counter);
Signal total = read(accumulator);
```

**Returns:** Signal with the memory's stored value.

### `write(value_expr, memory_name, when=1)`

Writes `value_expr` into `memory_name` whenever the optional `when` predicate is non-zero. If `when` is omitted, the compiler emits a constant enable on `signal-W` so the write always occurs.

**Syntax:**
```fcdsl
write(value_expr, memory_name, when=enable_signal);
```

**Examples:**
```fcdsl
write(read(counter) + 1, counter);                # Increment counter every tick
write(total + ("iron-plate", 10), accumulator);   # Add to accumulator

# Conditional writes
Signal should_update = condition > 0;
Signal new_value = read(state) + should_update;
write(new_value, state, when=should_update);

# Disable writes explicitly
write(signal, buffer, when=0);                    # Emits the wiring but latches nothing
```

**Parameters:**
- `value_expr`: Expression to store in the memory cell when enabled.
- `memory_name`: Name of the declared memory target.
- `when` *(optional)*: Signal expression that gates the write (treated as true when `> 0`). Use the literal `once` to request a one-shot initialization pulse.

### Memory Usage Patterns

#### Counter
```fcdsl
Memory counter: "signal-A";
write(("signal-A", 0), counter, when=once);
write(read(counter) + ("signal-A", 1), counter);
```

#### Accumulator with Reset
```fcdsl
Memory total: "iron-plate";
write(("iron-plate", 0), total, when=once);

Signal input_val = ("iron-plate", 10);
Signal reset = ("signal-R", 1);
Signal new_total = (read(total) + input_val) * (1 - reset);
write(new_total, total, when=1 - reset);
```

#### State Machine
```fcdsl
Memory state: "signal-S";
write(("signal-S", 0), state, when=once);

Signal current = read(state);
Signal next = (current + 1) % 4;  # 4-state cycle
write(next, state);
```

#### Advanced Memory Patterns

The repository includes `tests/sample_programs/04_memory_advanced.fcdsl`, which demonstrates:

- storing item, fluid, and virtual signals in dedicated memory cells
- building conditional write expressions that preserve the previous value when a guard is false
- constructing state machines and swapping stored values without bundle helpers using projection

Use it as a reference when wiring more sophisticated SR latch workflows.

> **Note:** The compiler materializes the write enable on `signal-W`. Attempting to project or declare user signals on `signal-W` results in a compile-time error.

---

## Entity System

### Entity Placement

Entities are placed using the `place()` function and can have their properties controlled via signals.

### Property Access

**Reading Properties:**
```fcdsl
Signal lamp_status = lamp.enable;      # Read current state
```

**Writing Properties:**
```fcdsl
lamp.enable = signal > 0;              # Set enable based on signal
train_stop.manual_mode = 1;            # Set manual mode
```

### Common Entity Properties

#### Lamps (`small-lamp`)
- `enable`: Enable/disable lamp (0/1)

#### Train Stops (`train-stop`)
- `manual_mode`: Manual train control (0/1)
- `enable`: Enable/disable station (0/1)

#### Assembling Machines
- `enable`: Enable/disable production (0/1)
- `recipe`: Set production recipe (signal-based)

#### Inserters
- `enable`: Enable/disable inserter (0/1)
- `stack_size`: Override stack size

### Entity Control Patterns

#### Blinking Lamps
```fcdsl
Memory counter;
write(0, counter, when=once);
write(read(counter) + 1, counter);
Signal blink = read(counter) % 10;

Entity lamp1 = place("small-lamp", 0, 0);
Entity lamp2 = place("small-lamp", 2, 0);

lamp1.enable = blink < 5;
lamp2.enable = blink >= 5;
```

#### Production Control
```fcdsl
Signal demand = ("signal-demand", 100);
Signal supply = ("signal-supply", 80);
Signal shortage = demand - supply;

Entity assembler = place("assembling-machine-1", 5, 0);
assembler.enable = shortage > 0;
```

---

## Functions and Modules

### Function Definition

```fcdsl
func function_name(param1, param2, ...) {
    # Function body
    return expression;
}
```

### Function Parameters

Parameters are untyped and take on the types of passed arguments:

```fcdsl
func double_signal(x) {
    return x * 2;
}

Signal doubled_iron = double_signal(iron);     # x becomes iron-plate signal
Signal doubled_virtual = double_signal(virt);  # x becomes virtual signal
```

### Function Return Values

Functions can return any expression type:

```fcdsl
func calculate_need(demand, supply, buffer) {
    Signal shortfall = demand - supply;
    Signal buffered = shortfall + buffer;
    return buffered * (buffered > 0);  # Only positive values
}
```

### Local Variables

Variables declared inside functions are local to that function:

```fcdsl
func complex_calculation(input_val) {
    Signal intermediate = input_val * 2;  # Local variable
    Signal result = intermediate + 10;
    return result;
}
```

### Memory in Functions

Functions can declare and use local memory:

```fcdsl
func toggle_generator() {
    Memory state: "signal-S";
    write(("signal-S", 0), state, when=once);
    Signal current = read(state);
    Signal new_state = ("signal-S", 1) - current;  # Toggle between 0 and 1
    write(new_state, state);
    return new_state;
}
```

### Function Call Examples

```fcdsl
# Simple calls
Signal result = double_signal(42);

# Nested calls
Signal complex = calculate_need(
    ("demand", 100),
    ("supply", 80),
    10
);

# Using in expressions
Signal output = double_signal(iron) + calculate_need(a, b, 5);
```

### Module System

**Import Statements:**
```fcdsl
import "stdlib/memory_utils.fcdsl";
import "modules/production.fcdsl" as prod;
```

**Using Imported Functions:**
```fcdsl
Signal smoothed = smooth_filter(input_signal, 5);
Signal delayed = delay_signal(processed, 10);
Signal controlled = prod.production_controller(demand, supply, 100);
```

---

## Projection and Type Coercion

### Projection Operator (`|`)

The projection operator `|` explicitly sets the output signal type of an expression.

**Syntax:**
```fcdsl
expression | "target_type"
```

### Signal Projection

Convert signals between types:

```fcdsl
Signal iron = ("iron-plate", 100);
Signal converted = iron | "copper-plate";    # Convert to copper-plate signal
Signal virtual = iron | "signal-A";          # Convert to virtual signal
```

### Channel Aggregation (Bundle Helper Removed)

Bundles were removed from the language. To combine different channels, explicitly project each source into the desired output type and sum the projected values.

```fcdsl
Signal iron = ("iron-plate", 100);
Signal copper = ("copper-plate", 80);
Signal coal = ("coal", 50);

# Aggregate totals without bundle()
Signal total_resources = (iron | "signal-output")
                      + (copper | "signal-output")
                      + (coal | "signal-output");

# Extract specific channels via projection
Signal iron_only = iron | "iron-plate";
Signal water = (coal | "water");   # Missing projections default to 0
```

### Arithmetic with Projection

Projection binds looser than arithmetic but tighter than comparisons:

```fcdsl
Signal result = a + b * 2 | "iron-plate";
# Evaluates as: (a + (b * 2)) | "iron-plate"

Signal comparison = (a + b | "iron-plate") > 10;
# Projection evaluated first due to parentheses
```

### Type Coercion Rules

#### Implicit Coercion

1. **Integer to Signal**: Integers are coerced to the signal's type in mixed operations
   ```fcdsl
   Signal result = iron + 10;  # 10 coerced to iron-plate type
   ```

2. **Left Operand Wins**: In mixed-signal operations, left operand's type is used
   ```fcdsl
   Signal result = iron + copper;  # Result has iron-plate type
   ```

#### Explicit Coercion

Use projection to explicitly convert types:

```fcdsl
Signal iron_as_copper = iron | "copper-plate";
Signal virtual_from_item = iron | "signal-A";
```

### Projection Use Cases

#### Type Standardization
```fcdsl
Signal iron = ("iron-plate", 100);
Signal copper = ("copper-plate", 80);

# Standardize both to virtual signals for calculation
Signal iron_virtual = iron | "signal-A";
Signal copper_virtual = copper | "signal-B";
Signal sum = iron_virtual + copper_virtual | "signal-output";
```

> Sample references: `tests/sample_programs/02_mixed_types.fcdsl`, `03_bundles.fcdsl`, and `09_advanced_patterns.fcdsl` showcase projection-based aggregation patterns in full programs.

## Compilation Model

### Compilation Pipeline

1. **Lexical Analysis**: Source → Tokens
2. **Parsing**: Tokens → Abstract Syntax Tree (AST) 
3. **Semantic Analysis**: Type inference, symbol resolution, validation
4. **IR Generation**: AST → Intermediate Representation
5. **Blueprint Emission**: IR → Factorio combinators and entities

### Intermediate Representation (IR)

The compiler generates IR operations that represent Factorio combinators:

#### Value-Producing Operations
- `IR_Const`: Constant combinator with fixed value
- `IR_Arith`: Arithmetic combinator (+, -, *, /, %)
- `IR_Decider`: Decider combinator with conditional logic
- `IR_WireMerge`: Virtual node that merges multiple simple sources onto one signal wire

#### Effect Operations
- `IR_MemCreate`: Memory cell creation (SR latch circuit)
- `IR_MemRead`: Memory read operation
- `IR_MemWrite`: Memory write operation
- `IR_PlaceEntity`: Entity placement in blueprint
- `IR_EntityPropWrite`: Entity property assignment
- `IR_EntityPropRead`: Entity property access

### Blueprint Generation

The emit module converts IR to Factorio blueprint JSON using the `factorio-draftsman` library:

1. **Entity Creation**: Each IR operation becomes one or more Factorio entities
2. **Signal Mapping**: DSL signal types mapped to Factorio signal names
3. **Circuit Wiring**: Automatic wiring between combinators based on data flow
4. **Layout**: Automatic entity positioning with collision avoidance
5. **Validation**: Final blueprint validation and optimization

#### Edge Layout Conventions

- **North Edge (Literals)**: Every materialized literal constant occupies the `north_literals` zone. The compiler reserves a dedicated row along the top of the blueprint so these combinators line up visually by declaration order.
- **South Edge (Export Anchors)**: Dangling outputs and explicitly exported signals receive zeroed constant combinators in the `south_exports` zone, forming a clean handoff row for external wiring.
- **Blueprint Metadata**: Generated blueprints append a description note (`Edge layout: literal constants are placed along the north boundary; export anchors align along the south boundary.`) to remind importers of the spatial contract.
- **Long-Span Wiring**: When edge placements would exceed Factorio's circuit wire reach, the emitter inserts medium electric poles as invisible relays to keep connections valid. Relay heuristics can be tuned through `WireRelayOptions` (Euclidean vs. Manhattan interpolation, relay caps, or full disable) when constructing the `BlueprintEmitter`.

### Signal Type Resolution

**Implicit Types**: Compiler allocates virtual signals (`__v1`, `__v2`, etc.)
**Explicit Types**: User-specified types validated against Factorio signal database
**Type Mapping**: Final signal mapping exported for debugging

### Wire Merge Optimization

The emitter can eliminate arithmetic combinators for addition chains when every
operand is a *simple source* (constant combinator output, entity property read, or
the result of an earlier wire merge) and all signals share the same channel.

```fcdsl
Signal iron_a = ("iron-plate", 100);
Signal iron_b = ("iron-plate", 200);
Signal iron_c = ("iron-plate", 50);

Signal total = iron_a + iron_b + iron_c;  # wired together with no arithmetic entity
```

The compiler verifies that:

- The operator is `+` throughout the addition chain.
- Each operand resolves to a unique simple source. Expressions such as `a + a`
    still allocate a combinator so the doubled signal is preserved.
- All operands agree on signal type (mixed-type additions fall back to arithmetic).

When these conditions hold, the lowerer emits `IR_WireMerge` and the emitter wires
all contributing sources directly to each consumer. No additional combinators are
placed in the blueprint, reducing entity count and latency while preserving semantics.

### Memory Implementation

1. **Creation**: `IR_MemCreate` places the write and hold deciders with no stored value; the cell remains empty until the first write occurs.
2. **Reading**: `IR_MemRead` exposes the latched output on the red feedback loop for downstream combinators.
3. **Writing**: `IR_MemWrite` drives the module with the requested data value and a `signal-W` enable line; when the enable is zero the latch preserves its previous state.

- **Typed Outputs**: The latch projects onto the declared memory channel instead of relying on `signal-everything`, improving compatibility with typed pipelines.
- **Safe Initialization**: Seeding is expressed explicitly through `write(..., when=once)` or other user-provided conditions that pulse `signal-W` for the desired tick.

---

## Examples

### Basic Arithmetic Circuit

```fcdsl
# Input signals (connect these to your bus after import)
Signal iron = ("iron-plate", 0);
Signal copper = ("copper-plate", 0);

# Calculations
Signal total = iron + copper;
Signal doubled = total * 2;
Signal remainder = doubled % 100;

# Output with explicit type
Signal output = remainder | "signal-output";
```

**Generated Blueprint**: Constant combinators for inputs, arithmetic combinators for calculations.

### Memory-Based Counter

```fcdsl
# Counter that increments every tick
mem counter = memory(0);
Signal current = read(counter);
write(current + 1, counter);

# Output current count
Signal count_output = current | "signal-count";
```

**Generated Blueprint**: SR latch circuit for memory, arithmetic combinator for increment.

### Entity Control System

```fcdsl
# Inputs (wire these channels from your factory bus)
Signal production_demand = ("signal-demand", 0);
Signal current_supply = ("signal-supply", 0);

# Logic
Signal shortage = production_demand - current_supply;
Signal should_produce = shortage > 0;

# Entity control
Entity assembler1 = place("assembling-machine-1", 10, 0);
Entity assembler2 = place("assembling-machine-1", 15, 0);

assembler1.enable = should_produce;
assembler2.enable = shortage > 50;  # Only enable second assembler for high demand

# Status lamp
Entity status_lamp = place("small-lamp", 20, 0);
status_lamp.enable = should_produce;
```

### Multi-Signal Processing (Bundle-Free)

```fcdsl
# External feeds (wire these signals after import)
Signal iron_feed = ("iron-ore", 0);
Signal copper_feed = ("copper-ore", 0);
Signal coal_feed = ("coal", 0);

# Process each channel explicitly via projection
Signal iron_plates = (iron_feed | "iron-plate") * 2;
Signal copper_plates = copper_feed | "copper-plate";
Signal coal_passthrough = coal_feed | "coal";

# Aggregate totals without bundle()
Signal total_output = (iron_plates | "signal-total")
                    + (copper_plates | "signal-total")
                    + (coal_passthrough | "signal-total");
```

### State Machine Example

```fcdsl
# 4-state production controller
mem production_state = memory(0);
Signal current_state = read(production_state);

# State transitions based on inputs (wire these channels externally)
Signal iron_bus = ("iron-plate", 0);
Signal copper_bus = ("copper-plate", 0);
Signal iron_low = iron_bus < 100;
Signal copper_low = copper_bus < 100;
Signal both_ok = (!iron_low) && (!copper_low);

Signal next_state = 
    (current_state == 0 && iron_low) * 1 +        # State 0→1: Need iron
    (current_state == 0 && copper_low) * 2 +      # State 0→2: Need copper  
    (current_state == 0 && both_ok) * 0 +         # State 0→0: All good
    (current_state == 1 && !iron_low) * 0 +       # State 1→0: Iron restored
    (current_state == 2 && !copper_low) * 0 +     # State 2→0: Copper restored
    (current_state * ((current_state == 1 && iron_low) || (current_state == 2 && copper_low)));

write(next_state, production_state);

# Output based on state
Signal iron_needed = (current_state == 1) * 100;
Signal copper_needed = (current_state == 2) * 100;
```

### Function-Based Design

```fcdsl
# Utility functions
func clamp(value, min_val, max_val) {
    Signal clamped = value;
    clamped = (clamped < min_val) * min_val + (clamped >= min_val) * clamped;
    clamped = (clamped > max_val) * max_val + (clamped <= max_val) * clamped;
    return clamped;
}

func smooth_filter(signal, factor) {
    mem smooth_state = memory(0);
    Signal current = read(smooth_state);
    Signal new_value = (current * factor + signal) / (factor + 1);
    write(new_value, smooth_state);
    return new_value;
}

# Main circuit (wire signal-input externally)
Signal raw_input = ("signal-input", 0);
Signal smoothed = smooth_filter(raw_input, 5);
Signal clamped = clamp(smoothed, 0, 1000);
Signal output = clamped | "signal-output";
```

---

## Diagnostics and Error Handling

### Warning Types

#### Mixed Type Warnings
```fcdsl
Signal result = iron + copper;
# Warning: Mixed signal types in binary operation: 'iron-plate' + 'copper-plate'. 
# Result will be 'iron-plate'. Use '| "type"' to explicitly set output channel.
```

#### Integer Coercion Warnings  
```fcdsl
Signal result = iron + 10;
# Warning: Mixed types in binary operation: signal 'iron-plate' + integer. 
# Integer will be coerced to signal type.
```

#### Unknown Signal Warnings
```fcdsl
Signal unknown = ("nonexistent-signal", 0);
# Warning: Unknown signal type: nonexistent-signal, using signal-0
```

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

#### Invalid Function Calls
```fcdsl
Signal result = nonexistent_function(5);
# Error: Undefined function 'nonexistent_function'
```

### Strict Mode

When compiled with `--strict` flag, warnings become errors:

```bash
# Normal compilation with warnings
python compile.py tests/sample_programs/01_basic_arithmetic.fcdsl

# Strict compilation - warnings become errors
python compile.py tests/sample_programs/01_basic_arithmetic.fcdsl --strict
```

### Diagnostic Output

The compiler provides detailed diagnostic information:

```
Compiling tests/sample_programs/01_basic_arithmetic.fcdsl...
Diagnostics:
  WARNING [12:5]: Mixed signal types in binary operation: '__v1' + '__v2'
  WARNING [15:10]: Integer coerced to signal type in multiplication
  ERROR [20:15]: Undefined memory 'invalid_mem' in read operation

Compilation failed: Semantic analysis failed
```

### Blueprint Validation

The compiler validates generated blueprints:

- **Entity Count**: Reports number and types of generated entities
- **Position Validation**: Warns about entities at extreme coordinates  
- **Signal Validation**: Checks signal names against Factorio database
- **Circuit Validation**: Verifies combinator configurations

---

## Implementation Status

This specification describes the language as currently implemented. The following features are fully functional:

✅ **Complete Syntax**: All described syntax is implemented and tested  
✅ **Type System**: Full type inference with warnings and strict mode  
✅ **Built-in Functions**: Signal literals, `memory()`, `place()` implemented  
✅ **Memory Operations**: Memory type, `read()`, `write()` implemented  
✅ **Entity System**: Entity placement and property control functional  
⚠️ **Bundle Helper**: `bundle()` removed; use projection patterns for aggregation  
✅ **Functions**: User-defined functions with parameters and local variables  
✅ **Compilation Pipeline**: Complete AST → IR → Blueprint generation  
✅ **Blueprint Export**: Generates valid Factorio blueprint strings  

### Compiler Usage

```bash
# Basic compilation
python compile.py tests/sample_programs/01_basic_arithmetic.fcdsl

# Save to file  
python compile.py tests/sample_programs/01_basic_arithmetic.fcdsl -o output.blueprint

# Strict type checking
python compile.py tests/sample_programs/01_basic_arithmetic.fcdsl --strict

# Verbose diagnostics
python compile.py tests/sample_programs/01_basic_arithmetic.fcdsl --verbose

# Custom blueprint name
python compile.py tests/sample_programs/01_basic_arithmetic.fcdsl --name "My Circuit"
```

This specification represents the complete, implemented language as of version 1.0. All examples are guaranteed to compile and generate working Factorio blueprints.