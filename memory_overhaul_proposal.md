# Memory System Specification (Revised)

## Overview

Memory in the DSL is treated as a **first-class entity type** with methods and properties specific to each memory implementation. Different memory types provide different control semantics and access patterns optimized for specific use cases.

---

## Memory Declaration Syntax

```fcdsl
Memory name: (signal_type, memory_type) [= initial_value] [properties];
```

**Parameters:**
- `name`: Identifier for the memory cell
- `signal_type`: String literal signal type (e.g., `"signal-A"`, `"iron-plate"`)
- `memory_type`: Memory implementation type (see below)
- `initial_value` (optional): Starting value (default: 0)
- `properties` (optional): Type-specific configuration

**Examples:**

```fcdsl
Memory counter: ("signal-A", always_write) = 0;
Memory state: ("signal-S", conditional);
Memory power_ctrl: ("signal-P", sr_latch) {priority: "set"};
Memory history: ("iron-plate", delay) {depth: 10};
```

---

## Memory Types

### 1. `always_write` - Unconditional Feedback

**Description:** Writes every tick. Optimized to arithmetic combinator with self-feedback loop.

**Use Cases:** Counters, accumulators, pattern generators

**Declaration:**
```fcdsl
Memory name: (signal_type, always_write) = initial_value;
```

**Methods:**

#### `.write(signal)`
Unconditionally writes the signal value every tick.

```fcdsl
Memory counter: ("signal-A", always_write) = 0;
counter.write(counter.read() + 1);  # Increment every tick
```

#### `.read()`
Returns the current stored value as a Signal.

```fcdsl
Signal current = counter.read();
```

**Implementation:**
- Single arithmetic combinator with RED wire self-feedback
- Initial value from constant combinator (first tick only)

**Example:**
```fcdsl
Memory tick: ("signal-T", always_write) = 0;
tick.write(tick.read() + 1);

Signal blink = (tick.read() % 60) < 30;
```

---

### 2. `conditional` - Conditional Write (D-Latch)

**Description:** Writes only when condition is true, otherwise holds previous value. Implements the D flip-flop/latch semantics.

**Use Cases:** State capture, conditional updates, gated storage

**Declaration:**
```fcdsl
Memory name: (signal_type, conditional) = initial_value;
```

**Methods:**

#### `.write(signal, when)`
Writes signal when condition is true (> 0), holds previous value when false (== 0).

```fcdsl
Memory buffer: ("iron-plate", conditional) = 0;
Signal should_update = demand > supply;
buffer.write(new_value, when=should_update);
```

#### `.read()`
Returns the current stored value.

```fcdsl
Signal stored = buffer.read();
```

**Implementation:**
- Write gate: `if (signal-W > 0) then output everything`
- Hold gate: `if (signal-W == 0) then output everything` (self-loop)
- Control signal routed on GREEN wire, data on RED wire

**Example:**
```fcdsl
Memory snapshot: ("signal-data", conditional);
Signal capture_trigger = (tick.read() % 100) == 0;
snapshot.write(sensor_reading, when=capture_trigger);
```

---

### 3. `sr_latch` - Set-Reset Latch

**Description:** Binary memory with separate set and reset controls. Can output both Q and !Q.

**Use Cases:** Power control, state machines, hysteresis, binary flags

**Declaration:**
```fcdsl
Memory name: (signal_type, sr_latch) = initial_value {
    priority: "set" | "reset" | "none"  # default: "none"
};
```

**Properties:**

#### `priority`
Defines behavior when both set and reset are active simultaneously:
- `"set"`: Set takes priority, output goes to 1
- `"reset"`: Reset takes priority, output goes to 0
- `"none"`: Undefined behavior (warning issued)

**Methods:**

#### `.set(when)`
Sets output to 1 when condition is true.

```fcdsl
Memory power: ("signal-P", sr_latch) {priority: "set"};
power.set(when=accumulator_low);
```

#### `.reset(when)`
Resets output to 0 when condition is true.

```fcdsl
power.reset(when=accumulator_charged);
```

#### `.read()`
Returns Q output (1 or 0).

```fcdsl
Signal is_on = power.read();
```

#### `.read_inv()`
Returns !Q output (inverted, 0 or 1).

```fcdsl
Signal is_off = power.read_inv();
```

**Implementation:**
- Set gate: `if (signal-S > 0) then output signal-Q = 1`
- Reset gate: `if (signal-R > 0) then output signal-Q = 0`
- Hold gate: `if (signal-S == 0 && signal-R == 0) then output signal-Q`
- Priority handling via combinator ordering

**Example:**
```fcdsl
Memory backup_power: ("signal-B", sr_latch) = 0 {priority: "reset"};

Signal low_charge = accumulator < 20;
Signal high_charge = accumulator > 90;

backup_power.set(when=low_charge);
backup_power.reset(when=high_charge);

Entity steam_engine = place("steam-engine", 0, 0);
steam_engine.enable = backup_power.read();
```

---

### 4. `toggle` - Toggle Flip-Flop (T-FF)

**Description:** Toggles between 0 and 1 on each trigger pulse. Requires edge detection.

**Use Cases:** Binary state toggling, alternating patterns, clock division

**Declaration:**
```fcdsl
Memory name: (signal_type, toggle) = initial_value;
```

**Methods:**

#### `.toggle(when)`
Toggles state (0→1 or 1→0) on rising edge of condition.

```fcdsl
Memory flipper: ("signal-F", toggle) = 0;
Signal button_pressed = button > 0;
flipper.toggle(when=button_pressed);
```

#### `.read()`
Returns current state (0 or 1).

```fcdsl
Signal state = flipper.read();
```

**Implementation:**
- Edge detector: Rising edge of condition signal
- XOR logic: `new_state = current_state XOR pulse`
- Conditional memory to hold state

**Example:**
```fcdsl
Memory alternate: ("signal-A", toggle) = 0;
Signal pulse = (tick.read() % 10) == 0;
alternate.toggle(when=pulse);

Entity lamp1 = place("small-lamp", 0, 0);
Entity lamp2 = place("small-lamp", 2, 0);

lamp1.enable = alternate.read();
lamp2.enable = alternate.read_inv();  # Compile error: toggle doesn't have read_inv()

# Fix:
lamp2.enable = (alternate.read() == 0);
```

---

### 5. `counter` - Specialized Counter

**Description:** Optimized for increment/decrement operations with optional wraparound.

**Use Cases:** Counting events, timers, modulo counters, index tracking

**Declaration:**
```fcdsl
Memory name: (signal_type, counter) = initial_value {
    wrap_at: integer | null  # default: null (no wrap)
};
```

**Properties:**

#### `wrap_at`
If set, counter wraps using modulo: `count % wrap_at`

**Methods:**

#### `.increment(when)`
Adds 1 when condition is true.

```fcdsl
Memory event_count: ("signal-E", counter) = 0;
event_count.increment(when=event_detected);
```

#### `.decrement(when)`
Subtracts 1 when condition is true.

```fcdsl
event_count.decrement(when=event_cleared);
```

#### `.reset(when)`
Resets to initial value when condition is true.

```fcdsl
event_count.reset(when=system_restart);
```

#### `.set(value, when)`
Sets to specific value when condition is true.

```fcdsl
event_count.set(100, when=initialize);
```

#### `.read()`
Returns current count.

```fcdsl
Signal count = event_count.read();
```

**Implementation:**
- Conditional memory with compound expressions
- Modulo operation applied if `wrap_at` is set

**Example:**
```fcdsl
Memory timer: ("signal-T", counter) = 0 {wrap_at: 3600};
timer.increment(when=1);  # Always increment (every tick)

Signal seconds = timer.read() / 60;
Signal minutes = (timer.read() / 3600) % 60;
```

**Advanced Example:**
```fcdsl
Memory ring_buffer_idx: ("signal-I", counter) = 0 {wrap_at: 16};

Signal write_pulse = new_data_available;
Signal read_pulse = consumer_ready;

ring_buffer_idx.increment(when=write_pulse);
ring_buffer_idx.decrement(when=read_pulse);

Signal index = ring_buffer_idx.read();
```

---

### 6. `delay` - Delay Line / Shift Register

**Description:** Stores signal history, allowing reads from N ticks in the past.

**Use Cases:** Temporal buffering, moving averages, synchronization, pipeline delays

**Declaration:**
```fcdsl
Memory name: (signal_type, delay) {
    depth: integer  # Required, number of ticks to store
};
```

**Properties:**

#### `depth`
Number of historical values to retain (1 to 100).

**Methods:**

#### `.push(signal)`
Stores a new value and shifts history back by one tick.

```fcdsl
Memory history: ("signal-H", delay) {depth: 10};
history.push(sensor_reading);
```

#### `.read(offset=0)`
Reads value from `offset` ticks ago:
- `offset=0`: Current tick's pushed value
- `offset=1`: Previous tick's value
- `offset=N`: N ticks ago

```fcdsl
Signal current = history.read(0);
Signal prev = history.read(1);
Signal old = history.read(5);
```

#### `.read()` (shorthand)
Equivalent to `.read(offset=1)` - previous tick's value.

```fcdsl
Signal delayed = history.read();  # 1 tick ago
```

**Implementation:**
- Chain of `depth` conditional memory cells
- Each cell: `cell[i].write(cell[i-1].read(), when=1)`
- First cell writes the pushed signal

**Example:**
```fcdsl
Memory signal_delay: ("iron-plate", delay) {depth: 5};
signal_delay.push(current_production);

Signal delayed_production = signal_delay.read();  # 1 tick ago
Signal old_production = signal_delay.read(5);      # 5 ticks ago

Signal delta = current_production - delayed_production;
```

**Moving Average Example:**
```fcdsl
Memory samples: ("signal-S", delay) {depth: 10};
samples.push(sensor_value);

Signal sum = samples.read(0) + samples.read(1) + samples.read(2)
           + samples.read(3) + samples.read(4) + samples.read(5)
           + samples.read(6) + samples.read(7) + samples.read(8)
           + samples.read(9);

Signal average = sum / 10;
```

---

## Common Patterns

### Pattern 1: Conditional Counter

```fcdsl
Memory count: ("signal-C", conditional) = 0;
Signal should_count = sensor > threshold;
count.write(count.read() + 1, when=should_count);
```

vs specialized counter:

```fcdsl
Memory count: ("signal-C", counter) = 0;
Signal should_count = sensor > threshold;
count.increment(when=should_count);
```

### Pattern 2: State Machine with SR Latch

```fcdsl
Memory running: ("signal-R", sr_latch) = 0 {priority: "set"};
Memory error: ("signal-E", sr_latch) = 0 {priority: "reset"};

Signal start_button = button == 1;
Signal stop_button = button == 2;
Signal fault_detected = temperature > 100;

running.set(when=start_button);
running.reset(when=stop_button || fault_detected);

error.set(when=fault_detected);
error.reset(when=start_button);

Entity machine = place("assembling-machine-1", 0, 0);
machine.enable = running.read() && !error.read();
```

### Pattern 3: Debouncing with Delay

```fcdsl
Memory button_history: ("signal-B", delay) {depth: 5};
button_history.push(button_raw);

# Button is stable if all 5 samples are the same
Signal stable = (button_history.read(0) == button_history.read(1))
             && (button_history.read(1) == button_history.read(2))
             && (button_history.read(2) == button_history.read(3))
             && (button_history.read(3) == button_history.read(4));

Signal debounced_button = stable * button_history.read(0);
```

### Pattern 4: Toggle for Alternating Sequences

```fcdsl
Memory phase: ("signal-P", toggle) = 0;

Signal cycle_complete = (tick.read() % 100) == 0;
phase.toggle(when=cycle_complete);

Entity pump1 = place("pump", 0, 0);
Entity pump2 = place("pump", 3, 0);

pump1.enable = phase.read();
pump2.enable = (phase.read() == 0);
```

---

## Type Selection Guide

| Use Case | Recommended Type | Why |
|----------|------------------|-----|
| Simple counter | `always_write` or `counter` | Unconditional, every tick |
| Conditional update | `conditional` | Explicit control over when to write |
| Power management | `sr_latch` | Set on low power, reset on full charge |
| Binary toggle | `toggle` | Flip state on button press |
| Event counting | `counter` | Dedicated increment/decrement/reset |
| Signal history | `delay` | Temporal buffering, averaging |
| Circular index | `counter` with `wrap_at` | Automatic modulo |
| State machine | `sr_latch` or `conditional` | Depends on control logic |

---

## Memory Properties (Read-Only)

All memory types support introspection:

```fcdsl
Signal type = counter.signal_type;      # Returns "signal-A" (as signal)
Signal initial = counter.initial_value; # Returns initial value
Signal mem_type = counter.memory_type;  # Returns type enum (for debugging)
```

---

## Compilation Behavior

### Entity Count Estimates

| Memory Type | Typical Combinator Count |
|-------------|-------------------------|
| `always_write` | 1 arithmetic |
| `conditional` | 2 deciders (write + hold gate) |
| `sr_latch` | 3-4 deciders (set, reset, hold, priority) |
| `toggle` | 3-4 combinators (edge detect + XOR) |
| `counter` | 2-4 combinators (depends on operations) |
| `delay` with depth N | N conditional memories |

### Optimization Notes

1. **always_write**: Optimized to single arithmetic combinator with self-feedback
2. **conditional**: Uses standard SR latch topology (write gate → hold gate with self-loop)
3. **sr_latch**: May inline priority logic if only one control is used
4. **counter** with `wrap_at`: Applies modulo only if property is set
5. **delay**: Chain is optimized if only reading offset=1 (last value)

---

## Error Handling

### Type Mismatches

```fcdsl
Memory state: ("signal-A", sr_latch);
state.write(value, when=condition);  # ERROR: sr_latch doesn't have .write()
# Use: state.set() or state.reset()
```

### Invalid Properties

```fcdsl
Memory count: ("signal-C", always_write) {wrap_at: 100};
# ERROR: always_write doesn't support 'wrap_at' property
# Use: counter type instead
```

### Signal Type Consistency

```fcdsl
Memory buffer: ("iron-plate", conditional);
Signal copper = ("copper-plate", 50);
buffer.write(copper, when=1);
# WARNING: Type mismatch: buffer expects 'iron-plate', got 'copper-plate'
```

### Method Availability

```fcdsl
Memory toggle: ("signal-T", toggle);
Signal inverted = toggle.read_inv();
# ERROR: toggle type doesn't support read_inv() method
# Use: (toggle.read() == 0) instead
```

---

## Reserved Signals

Memory types reserve specific virtual signals:

| Signal | Reserved For | Used By |
|--------|-------------|---------|
| `signal-W` | Write enable | `conditional` |
| `signal-S` | Set control | `sr_latch` |
| `signal-R` | Reset control | `sr_latch` |
| `signal-T` | Toggle pulse | `toggle` |
| `signal-I[0-N]` | Delay chain indices | `delay` |

**CRITICAL:** Never use these signals for user data when using corresponding memory types.

---

## Migration from Old Syntax

### Old `write()` / `read()` Functions

**Old:**
```fcdsl
Memory counter: "signal-A" = 0;
write(read(counter) + 1, counter);
Signal value = read(counter);
```

**New:**
```fcdsl
Memory counter: ("signal-A", always_write) = 0;
counter.write(counter.read() + 1);
Signal value = counter.read();
```

### Old `when=once` Initialization

**Old:**
```fcdsl
Memory counter: "signal-A";
write(("signal-A", 0), counter, when=once);
```

**New:**
```fcdsl
Memory counter: ("signal-A", conditional) = 0;
# Initialization is automatic
```

### Old Conditional Write

**Old:**
```fcdsl
Memory buffer: "iron-plate" = 0;
Signal condition = sensor > 100;
write(new_value, buffer, when=condition);
```

**New:**
```fcdsl
Memory buffer: ("iron-plate", conditional) = 0;
Signal condition = sensor > 100;
buffer.write(new_value, when=condition);
```

---

## Grammar Extensions

### Memory Declaration

```lark
mem_decl: "Memory" NAME ":" "(" STRING "," mem_type ")" ["=" expr] [mem_properties] ";"

mem_type: "always_write"
        | "conditional"
        | "sr_latch"
        | "toggle"
        | "counter"
        | "delay"

mem_properties: "{" mem_prop ("," mem_prop)* "}"

mem_prop: "priority" ":" STRING          # For sr_latch
        | "wrap_at" ":" INT              # For counter
        | "depth" ":" INT                # For delay
```

### Memory Method Calls

```lark
expr: expr "." method_call

method_call: "read" "(" [method_args] ")"
           | "read_inv" "(" ")"
           | "write" "(" expr ["," "when" "=" expr] ")"
           | "set" "(" "when" "=" expr ")"
           | "reset" "(" "when" "=" expr ")"
           | "toggle" "(" "when" "=" expr ")"
           | "increment" "(" "when" "=" expr ")"
           | "decrement" "(" "when" "=" expr ")"
           | "push" "(" expr ")"

method_args: "offset" "=" INT
           | expr "," "when" "=" expr
```

---

## Complete Example: Production Line Controller

```fcdsl
# Clock for synchronization
Memory tick: ("signal-T", always_write) = 0;
tick.write(tick.read() + 1);

# Production state machine
Memory producing: ("signal-P", sr_latch) = 0 {priority: "set"};
Memory error_state: ("signal-E", sr_latch) = 0 {priority: "reset"};

# Input monitoring with delay for stability
Memory demand_history: ("signal-D", delay) {depth: 5};
demand_history.push(demand_signal);

Signal stable_demand = demand_history.read(0);
Signal demand_rising = stable_demand > demand_history.read(5);

# Counter for production cycles
Memory cycles: ("signal-C", counter) = 0 {wrap_at: 1000};

# Control logic
Signal start_condition = stable_demand > 100 && !error_state.read();
Signal stop_condition = stable_demand < 50;
Signal error_detected = temperature > 150;

producing.set(when=start_condition);
producing.reset(when=stop_condition || error_detected);

error_state.set(when=error_detected);
error_state.reset(when=reset_button);

# Increment cycle counter when producing
cycles.increment(when=producing.read());

# Output to machines
Entity assembler1 = place("assembling-machine-3", 0, 0);
Entity assembler2 = place("assembling-machine-3", 5, 0);
Entity warning_lamp = place("small-lamp", -3, 0);

assembler1.enable = producing.read();
assembler2.enable = producing.read() && (cycles.read() > 10);
warning_lamp.enable = error_state.read();
```

---

## Future Extensions

### Potential Additional Memory Types

1. **`addressable`** - Array-like memory with dynamic indexing
2. **`fifo`** / **`lifo`** - Queue/stack semantics
3. **`multi_signal`** - Store multiple signal types in one cell
4. **`clocked`** - Synchronous memory with explicit clock input

### Potential Additional Methods

1. `.snapshot()` - Capture current value atomically
2. `.compare_and_swap(expected, new)` - Atomic update
3. `.clear()` - Reset to zero/default
4. `.freeze()` - Make read-only for debugging

---

## Summary

This revised memory system provides:

✅ **Type safety** - Each memory type has well-defined methods  
✅ **Clarity** - Method names clearly indicate operation semantics  
✅ **Flexibility** - Six specialized types cover all common patterns  
✅ **Consistency** - Follows entity-like property access pattern  
✅ **Optimization** - Compiler can optimize based on declared type  
✅ **Extensibility** - Easy to add new memory types without breaking existing code

The entity-style syntax `memory.method()` aligns with how entities work (`lamp.enable = ...`) and makes memory a first-class component in circuit design.