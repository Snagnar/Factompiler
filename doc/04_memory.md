# Working with Memory

Most Factorio circuits need to remember something: a counter, a previous state, a buffered value. Memory stores data that persists across game ticks.

---

## Understanding Memory in Factorio

In Factorio's circuit network, signals exist only during the tick they're produced. If a combinator outputs "signal-A: 50" on tick 100, that signal is gone by tick 101 unless something preserves it.

Memory creates a **feedback loop** — a circuit that feeds its output back to its input, remembering values indefinitely.

---

## Declaring Memory

```facto
Memory counter: "signal-A";    # Explicit type (recommended)
Memory buffer;                 # Type inferred from first write
```

### Explicit vs. Implicit Types

**Explicit types** make your code clear:

```facto
Memory counter: "signal-A";
Memory item_count: "iron-plate";
Memory state: "signal-S";
```

**Implicit types** are inferred from the first write:

```facto
Memory buffer;  # Type unknown until first write

Signal iron = ("iron-plate", 50);
buffer.write(iron);  # Now buffer uses iron-plate
```

We recommend always specifying the type.

---

## Reading and Writing Memory

### Reading: `.read()`

```facto
Memory counter: "signal-A";
Signal current_value = counter.read();
```

Returns a signal with the memory's type and stored value.

### Writing: `.write()`

```facto
memory.write(value);                  # Unconditional (every tick)
memory.write(value, when=condition);  # Conditional
```

**Unconditional** — writes every tick:

```facto
Memory counter: "signal-A";
counter.write(counter.read() + 1);  # Increment every tick
```

**Conditional** — writes only when condition is met:

```facto
Memory buffer: "signal-A";
Signal trigger = ("signal-T", 0);

buffer.write(new_value, when=trigger > 0);
```

When trigger is 0, the previous value is held.

---

## Common Memory Patterns

### Counter

The classic increment-every-tick:

```facto
Memory counter: "signal-A";
counter.write(counter.read() + 1);
```

**Behavior:** 0 → 1 → 2 → 3 → ...

### Wrapping Counter

Count up to a limit, then reset:

```facto
Memory counter: "signal-A";
int max_value = 60;

counter.write((counter.read() + 1) % max_value);
# Counts: 0, 1, 2, ..., 59, 0, 1, 2, ...
```

Use for:
- Blinking lights (cycle every N ticks)
- Timed sequences
- Animation frames

### Sample and Hold

Capture a value when triggered:

```facto
Memory captured: "signal-A";
Signal input = ("signal-input", 0);
Signal trigger = ("signal-trigger", 0);

captured.write(input, when=trigger > 0);
Signal output = captured.read();
```

When trigger is 0, memory holds previous value. When trigger becomes positive, it captures current input.

### Toggle (Flip-Flop)

Switch between 0 and 1:

```facto
Memory toggle: "signal-A";
Signal button = ("signal-button", 0);

Signal new_state = 1 - toggle.read();  # 0→1 or 1→0
toggle.write(new_state, when=button > 0);

Signal output = toggle.read();
```

Press once to turn on, again to turn off.

**Note:** If button stays high for multiple ticks, it flips every tick. For real buttons, you need edge detection (see Advanced Concepts).

### Accumulator

Sum values over time:

```facto
Memory total: "signal-A";
Signal incoming = ("iron-plate", 0);

total.write(total.read() + incoming);
Signal running_total = total.read();
```

### Conditional Accumulator

Only accumulate when enabled:

```facto
Memory total: "iron-plate";
Signal incoming = ("iron-plate", 0);
Signal should_count = ("signal-enable", 0);

total.write(total.read() + incoming, when=should_count > 0);
```

### Maximum Tracker

Remember the highest value seen (using conditional values):

```facto
Memory maximum: "signal-A";
Signal input = ("signal-input", 0);

Signal current_max = maximum.read();
Signal new_max = ((input > current_max) : input) 
              + ((input <= current_max) : current_max);

maximum.write(new_max);
```

### Minimum Tracker

```facto
Memory minimum: "signal-A";
Memory initialized: "signal-I";
Signal input = ("signal-input", 0);

Signal current_min = minimum.read();
Signal is_first = initialized.read() == 0;

Signal new_min = ((is_first != 0) : input)
              + ((is_first == 0 && input < current_min) : input)
              + ((is_first == 0 && input >= current_min) : current_min);

minimum.write(new_min);
initialized.write(1, when=is_first);
```

---

## Latches (Set/Reset Memory)

Latches are binary state memory — either "on" or "off" — controlled by separate **set** and **reset** triggers.

### Use Cases

- **Hysteresis control** — prevent rapid on/off cycling
- **Alarm systems** — stay on until manually reset
- **State machines** — track which mode you're in

### The Latch Syntax

```facto
memory.write(value, set=set_condition, reset=reset_condition);
```

| Parameter | Description |
|-----------|-------------|
| `value` | Output when latch is ON (usually 1) |
| `set=` | Condition that turns ON |
| `reset=` | Condition that turns OFF |

**The order of `set=` and `reset=` matters!** It determines priority when both are true.

### SR Latch (Set Priority)

Put `set=` **first** — set wins ties:

```facto
Memory state: "signal-S";
state.write(1, set=turn_on, reset=turn_off);
```

| turn_on | turn_off | Result |
|---------|----------|--------|
| true | false | ON |
| false | true | OFF |
| true | true | **ON (set wins)** |

### RS Latch (Reset Priority)

Put `reset=` **first** — reset wins ties:

```facto
Memory state: "signal-S";
state.write(1, reset=turn_off, set=turn_on);
```

| turn_on | turn_off | Result |
|---------|----------|--------|
| true | false | ON |
| false | true | OFF |
| true | true | **OFF (reset wins)** |

---

## Practical Example: Hysteresis Control

**The Problem:** Turn on backup steam power at 20% battery, off at 80%. A simple threshold causes flickering near 20%.

**The Solution:** Use a latch to create a **hysteresis gap**.

```facto
Signal battery = ("signal-A", 0);  # Wire from accumulator

Memory steam_enabled: "signal-S";
steam_enabled.write(1, 
    set=battery < 20,     # ON when battery drops below 20%
    reset=battery >= 80   # OFF when battery reaches 80%
);

Entity steam_switch = place("power-switch", 0, 0);
steam_switch.enable = steam_enabled.read() > 0;
```

**Behavior:**

1. Battery at 100% → latch OFF
2. Battery drains to 19% → set triggers → latch ON, steam starts
3. Steam charges battery to 40% → latch **stays ON** (no trigger)
4. Battery reaches 80% → reset triggers → latch OFF, steam stops

The latch "remembers" the state between the thresholds.

### Choosing SR vs RS

**SR (set priority) when "on" is safe:**
- Emergency systems
- Alarms
- Production that shouldn't stop

**RS (reset priority) when "off" is safe:**
- Power systems (conserve fuel)
- Potentially dangerous machines
- Cost-critical operations

### Output Values

Latches can output any value:

```facto
# Binary (most common)
state.write(1, set=on_trigger, reset=off_trigger);

# Custom value (adds a multiplier combinator)
speed.write(100, set=fast_mode, reset=slow_mode);

# Dynamic value
output.write(speed_setting, set=enabled, reset=disabled);
```

---

## Multiple Memories

Declare as many as needed:

```facto
Memory counter: "signal-A";
Memory previous: "signal-B";
Memory accumulator: "signal-C";

counter.write(counter.read() + 1);
previous.write(counter.read());
accumulator.write(accumulator.read() + counter.read());
```

Each memory is independent with its own signal type.

---

## Memory Type Rules

### All Writes Must Match

Every write must use the memory's signal type:

```facto
Memory buffer: "iron-plate";

Signal iron = ("iron-plate", 50);
Signal copper = ("copper-plate", 30);

buffer.write(iron);    # OK
buffer.write(copper);  # ERROR - type mismatch!
```

Fix with projection:

```facto
buffer.write(copper | "iron-plate");  # OK
```

### Reserved Signal: signal-W

`signal-W` is used internally for memory write-enable logic:

```facto
Signal bad = ("signal-W", 5);  # COMPILE ERROR!
```

All other signals are available.

---

## How Memory Works Under the Hood

### Write-Gated Latch

For conditional writes, the compiler generates two decider combinators:

```
Data ──► [Write Gate: if W > 0] ──┬──► Output
                                  │
         ┌────────────────────────┘
         │
         └► [Hold Gate: if W == 0] ─── (feedback)
```

- **Write Gate**: Passes data when write-enable (signal-W) is positive
- **Hold Gate**: Recirculates output when write-enable is zero

### Arithmetic Feedback Optimization

Unconditional writes optimize to a single combinator with feedback:

```facto
Memory counter: "signal-A";
counter.write(counter.read() + 1);
```

Becomes one arithmetic combinator whose output feeds back to its input.

---

## Common Mistakes

### Forgetting to Read

```facto
# WRONG - just stores 1 forever
Memory counter: "signal-A";
counter.write(1);
```

```facto
# CORRECT - increment based on current value
Memory counter: "signal-A";
counter.write(counter.read() + 1);
```

### Type Mismatch

```facto
Memory buffer: "signal-A";
Signal value = ("signal-B", 50);

buffer.write(value);  # Warning - type mismatch
buffer.write(value | "signal-A");  # Fixed with projection
```

### Using signal-W

```facto
Memory write_count: "signal-W";  # ERROR - reserved!
Memory write_count: "signal-V";  # OK
```

---

## Practical Example: Binary Clock

4-bit binary counter with lamp display:

<table>
<tr>
<td>

```facto
Memory counter: "signal-A";
counter.write((counter.read() + 1) % 16);

Signal bits = counter.read();

# Extract individual bits using conditional values
Signal bit0 = ((bits >> 0) AND 1) > 0 : 1;
Signal bit1 = ((bits >> 1) AND 1) > 0 : 1;
Signal bit2 = ((bits >> 2) AND 1) > 0 : 1;
Signal bit3 = ((bits >> 3) AND 1) > 0 : 1;

# 4 lamps showing binary count
for i in 0..4 {
    Entity lamp = place("small-lamp", i * 2, 0);
    lamp.enable = ((bits >> i) AND 1) > 0;
}
```

</td>
<td>
<img src="img/placeholder_binary_clock.gif" width="300" alt="4-bit binary counter with lamps in Factorio"/>
</td>
</tr>
</table>

## Practical Example: Moving Average

Smooth a noisy signal with 4-sample averaging:

```facto
Memory sample1: "signal-A";
Memory sample2: "signal-A";
Memory sample3: "signal-A";
Memory sample4: "signal-A";

Signal input = ("signal-input", 0);

# Shift samples (oldest falls off)
sample4.write(sample3.read());
sample3.write(sample2.read());
sample2.write(sample1.read());
sample1.write(input);

# Calculate average
Signal sum = sample1.read() + sample2.read() + sample3.read() + sample4.read();
Signal average = sum / 4;
```

---

## Summary

| Concept | Syntax |
|---------|--------|
| Declare memory | `Memory name: "type";` |
| Read value | `memory.read()` |
| Write always | `memory.write(value);` |
| Write conditionally | `memory.write(value, when=condition);` |
| SR latch (set priority) | `memory.write(1, set=on, reset=off);` |
| RS latch (reset priority) | `memory.write(1, reset=off, set=on);` |

**Key points:**
- Memory persists across ticks via feedback loops
- Use latches for hysteresis control (prevents flickering)
- All writes must match the memory's signal type
- `signal-W` is reserved for internal use

---

**[← Signals and Types](03_signals_and_types.md)** | **[Entities →](05_entities.md)**
