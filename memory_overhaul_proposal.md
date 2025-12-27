# Memory System Overhaul: Method-Based Syntax

**Date: December 2025**

This document specifies the migration from function-based memory syntax to method-based syntax for improved consistency and readability.

---

## Table of Contents

1. [Syntax Change Overview](#syntax-change-overview)
2. [Implementation Changes](#implementation-changes)
3. [Documentation Updates](#documentation-updates)
4. [Standard Library](#standard-library)
5. [Test Cases](#test-cases)
6. [Implementation Checklist](#implementation-checklist)

---

## Syntax Change Overview

### Current Syntax (Being Replaced)

```fcdsl
Memory counter: "signal-A";
Signal current = read(counter);
write(current + 1, counter);
write(value, buffer, when=trigger);
```

### New Syntax (Method-Based)

```fcdsl
Memory counter: "signal-A";
Signal current = counter.read();
counter.write(current + 1);
counter.write(value, when=trigger);
```

### Why This Change?

1. **Consistency**: Entity properties already use dot notation (`lamp.enable = x`)
2. **Readability**: `counter.write(x)` reads more naturally than `write(x, counter)`
3. **Discoverability**: Method syntax makes it clear what operations are available on memory

### Behavior Summary

| Usage Pattern | Behavior | Combinators |
|---------------|----------|-------------|
| `mem.write(mem.read() + x)` | Arithmetic feedback | 1 (optimized) |
| `mem.write(value)` | Unconditional write | 1-2 |
| `mem.write(value, when=cond)` | Conditional write (SR latch) | 2 |

### Complete Examples

```fcdsl
# Counter - increments every tick
Memory tick: "signal-T";
tick.write(tick.read() + 1);

# Modulo counter (clock) - wraps every 600 ticks
Memory clock: "signal-C";
clock.write((clock.read() + 1) % 600);

# Conditional capture - stores value only when triggered
Memory snapshot: "signal-S";
Signal trigger = (clock.read() % 100) == 0;
snapshot.write(sensor_value, when=trigger);

# State machine - update state only on valid transitions
Memory state: "signal-S";
Signal transition_valid = (state.read() == 0) && (start_signal > 0);
state.write(1, when=transition_valid);
```

---

## Implementation Changes

### 1. Grammar Changes

**File:** [dsl_compiler/grammar/fcdsl.lark](dsl_compiler/grammar/fcdsl.lark)

**Remove** from `primary`:
```lark
     | READ_KW "(" NAME ")"
     | WRITE_KW "(" expr "," NAME ["," WHEN_KW "=" write_when] ")"
```

**Remove** these tokens:
```lark
READ_KW: "read"
WRITE_KW: "write"
WHEN_KW: "when"
```

**Add** `WHEN_KW` back only for method syntax:
```lark
WHEN_KW: "when"
```

**Modify** `method_call` rule (currently handles entity methods):
```lark
method_call: NAME "." NAME "(" [arglist] ")"
           | NAME "." "read" "(" ")"                           -> memory_read
           | NAME "." "write" "(" expr ")"                     -> memory_write
           | NAME "." "write" "(" expr "," WHEN_KW "=" expr ")" -> memory_write_when
```

### 2. Parser/Transformer Changes

**File:** [dsl_compiler/src/parsing/transformer.py](dsl_compiler/src/parsing/transformer.py)

Replace handlers for the old `read()`/`write()` syntax:

```python
def memory_read(self, items):
    """Handle mem.read() method call."""
    memory_name = str(items[0])
    return ReadExpr(memory_name, line=..., column=...)

def memory_write(self, items):
    """Handle mem.write(value) method call."""
    memory_name = str(items[0])
    value = items[1]
    return WriteExpr(value, memory_name, when=None, line=..., column=...)

def memory_write_when(self, items):
    """Handle mem.write(value, when=cond) method call."""
    memory_name = str(items[0])
    value = items[1]
    condition = items[2]
    return WriteExpr(value, memory_name, when=condition, line=..., column=...)
```

### 3. AST Nodes (No Change)

The existing `ReadExpr` and `WriteExpr` nodes in [expressions.py](dsl_compiler/src/ast/expressions.py) remain unchanged:

```python
class ReadExpr(Expr):
    """memory.read() expression"""
    memory_name: str

class WriteExpr(Expr):
    """memory.write(value, when=condition) statement"""
    value: Expr
    memory_name: str
    when: Optional[Expr]  # None for unconditional
```

### 4. Semantic Analysis (Minor Changes)

**File:** [dsl_compiler/src/semantic/analyzer.py](dsl_compiler/src/semantic/analyzer.py)

The semantic analysis for `ReadExpr` and `WriteExpr` remains the same since the AST nodes haven't changed. However, update error messages to reflect new syntax:

```python
# Old:
f"Undefined memory '{expr.memory_name}' in read()"
# New:
f"Undefined memory '{expr.memory_name}' in .read()"
```

### 5. Lowering (No Change)

**File:** [dsl_compiler/src/lowering/memory_lowerer.py](dsl_compiler/src/lowering/memory_lowerer.py)

No changes needed. `lower_read_expr()` and `lower_write_expr()` work the same since the AST nodes are unchanged.

### 6. Layout/Memory Builder (No Change)

**File:** [dsl_compiler/src/layout/memory_builder.py](dsl_compiler/src/layout/memory_builder.py)

No changes needed. The IR nodes (`IR_MemCreate`, `IR_MemRead`, `IR_MemWrite`) are unchanged.

---

## Documentation Updates

### LANGUAGE_SPEC.md Updates

**File:** [LANGUAGE_SPEC.md](LANGUAGE_SPEC.md)

**Section: Memory System** (around line 779)

Replace:

```markdown
#### Reading Memory

\`\`\`fcdsl
Signal current = read(counter);
\`\`\`

This connects to the memory's output (the hold gate in the SR latch).

#### Writing Memory

\`\`\`fcdsl
write(value_expr, memory_name, when=enable_signal);
\`\`\`

**Parameters:**
- `value_expr`: Expression to store (must match memory's signal type)
- `memory_name`: Target memory cell
- `when` (optional): Signal controlling the write (default: `1` = always write)

**Examples:**

\`\`\`fcdsl
# Unconditional write (every tick)
write(read(counter) + 1, counter);

# Conditional write
Signal should_update = input > threshold;
write(new_value, buffer, when=should_update);
\`\`\`
```

With:

```markdown
#### Reading Memory

\`\`\`fcdsl
Signal current = counter.read();
\`\`\`

Returns the current stored value as a Signal with the memory's declared type.

#### Writing Memory

\`\`\`fcdsl
memory.write(value_expr);                  # Unconditional
memory.write(value_expr, when=condition);  # Conditional
\`\`\`

**Parameters:**
- `value_expr`: Expression to store (must match memory's signal type)
- `when` (optional): Condition controlling the write (default: always write)

**Examples:**

\`\`\`fcdsl
# Unconditional write (every tick)
counter.write(counter.read() + 1);

# Conditional write
Signal should_update = input > threshold;
buffer.write(new_value, when=should_update);
\`\`\`
```

**Section: Quick Start** (around line 54)

Replace:

```markdown
Memory counter: "signal-A";
write(read(counter) + 1, counter);

Signal blink = (read(counter) % 10) < 5;
```

With:

```markdown
Memory counter: "signal-A";
counter.write(counter.read() + 1);

Signal blink = (counter.read() % 10) < 5;
```

### doc/04_memory.md Updates

**File:** [doc/04_memory.md](doc/04_memory.md)

**Section: Reading and Writing Memory**

Replace all function-based syntax with method syntax throughout the file:

```markdown
### Reading: The `.read()` Method

Read the current value stored in memory:

\`\`\`fcdsl
Memory counter: "signal-A";

Signal current_value = counter.read();
\`\`\`

`.read()` returns a signal with the memory's type and its stored value.

### Writing: The `.write()` Method

Store a new value in memory:

\`\`\`fcdsl
memory.write(value);                  # Write every tick
memory.write(value, when=condition);  # Write only when condition > 0
\`\`\`

Parameters:
- **value** – The value to store (must match memory's signal type)
- **when** (optional) – Condition for when to write (default: always)
```

**Section: Common Memory Patterns**

Update all examples. For instance, Counter:

```markdown
### Counter

The classic increment-every-tick counter:

\`\`\`fcdsl
Memory counter: "signal-A";
counter.write(counter.read() + 1);
\`\`\`
```

Wrapping Counter:

```markdown
### Wrapping Counter

Count up to a limit, then reset:

\`\`\`fcdsl
Memory counter: "signal-A";
int max_value = 60;

counter.write((counter.read() + 1) % max_value);
# Counts 0, 1, 2, ..., 59, 0, 1, 2, ...
\`\`\`
```

Sample and Hold:

```markdown
### Sample and Hold

Capture a value when triggered:

\`\`\`fcdsl
Memory captured: "signal-A";
Signal input = ("signal-input", 0);
Signal trigger = ("signal-trigger", 0);

captured.write(input, when=trigger > 0);

Signal output = captured.read();
\`\`\`
```

Toggle:

```markdown
### Toggle (Flip-Flop)

Switch between 0 and 1:

\`\`\`fcdsl
Memory toggle: "signal-A";
Signal button = ("signal-button", 0);

Signal new_state = 1 - toggle.read();
toggle.write(new_state, when=button > 0);

Signal output = toggle.read();
\`\`\`
```

### Sample Programs Updates

**Directory:** [tests/sample_programs/](tests/sample_programs/)

Update all sample programs using memory. Examples:

**04_basic_memory.fcdsl:**
```fcdsl
Memory counter: "signal-A";
Signal step_size = 1;

counter.write((counter.read() + step_size) % 10);

Signal pulse = counter.read() == 0;

Entity pulse_lamp = place("small-lamp", 0, 0);
pulse_lamp.enable = pulse;
```

**04_memory.fcdsl**, **21_memory_feedback_loop.fcdsl**, etc. - all need updating.

---

## Standard Library

**File:** `lib/memory_patterns.fcdsl` (new file)

```fcdsl
# =============================================================================
# Memory Patterns Standard Library
# =============================================================================
# Common memory patterns using the method-based syntax.
# Import with: import "lib/memory_patterns.fcdsl";
# =============================================================================

# -----------------------------------------------------------------------------
# Hysteresis (Schmitt Trigger)
# -----------------------------------------------------------------------------
# Returns 1 when input drops below low_threshold.
# Stays 1 until input rises above high_threshold.
#
# Example:
#   Signal backup_on = hysteresis(accumulator_charge, 20, 70);
#   steam_engine.enable = backup_on;
#
func hysteresis(Signal input, int low_threshold, int high_threshold) {
    Memory state: "signal-H";
    
    Signal should_set = input < low_threshold;
    Signal should_reset = input > high_threshold;
    
    # Compute new state: set wins over hold, reset wins over all
    Signal new_state = (should_reset == 0) * (
        should_set + (should_set == 0) * state.read()
    );
    
    state.write(new_state);
    return state.read();
}


# -----------------------------------------------------------------------------
# Rising Edge Detector
# -----------------------------------------------------------------------------
# Outputs 1 for one tick when input transitions from 0 to non-zero.
#
func edge_rising(Signal input) {
    Memory prev: "signal-P";
    Signal is_rising = (input > 0) * (prev.read() == 0);
    prev.write(input > 0);
    return is_rising;
}


# -----------------------------------------------------------------------------
# Falling Edge Detector
# -----------------------------------------------------------------------------
# Outputs 1 for one tick when input transitions from non-zero to 0.
#
func edge_falling(Signal input) {
    Memory prev: "signal-P";
    Signal is_falling = (input == 0) * (prev.read() > 0);
    prev.write(input > 0);
    return is_falling;
}


# -----------------------------------------------------------------------------
# Toggle (T Flip-Flop)
# -----------------------------------------------------------------------------
# Alternates between 0 and 1 each time trigger goes high.
#
func toggle(Signal trigger) {
    Memory state: "signal-T";
    Signal edge = edge_rising(trigger);
    
    Signal new_state = (edge > 0) * (1 - state.read()) 
                     + (edge == 0) * state.read();
    
    state.write(new_state);
    return state.read();
}


# -----------------------------------------------------------------------------
# Sample and Hold
# -----------------------------------------------------------------------------
# Captures input value when trigger goes high, holds until next trigger.
#
func sample_hold(Signal input, Signal trigger) {
    Memory held: "signal-S";
    Signal edge = edge_rising(trigger);
    held.write(input, when=edge);
    return held.read();
}


# -----------------------------------------------------------------------------
# Pulse Stretcher
# -----------------------------------------------------------------------------
# Extends a 1-tick pulse to last for 'duration' ticks.
#
func pulse_stretch(Signal input, int duration) {
    Memory countdown: "signal-C";
    
    Signal triggered = input > 0;
    Signal next_count = triggered * duration 
                      + (triggered == 0) * (countdown.read() - 1);
    Signal clamped = (next_count > 0) * next_count;
    
    countdown.write(clamped);
    return (countdown.read() > 0) + triggered;
}


# -----------------------------------------------------------------------------
# Modulo Counter (Clock)
# -----------------------------------------------------------------------------
# Counter that wraps at specified value.
#
func clock(int period) {
    Memory tick: "signal-T";
    tick.write((tick.read() + 1) % period);
    return tick.read();
}


# -----------------------------------------------------------------------------
# Debounce
# -----------------------------------------------------------------------------
# Requires input to be stable for 'ticks' before changing output.
#
func debounce(Signal input, int ticks) {
    Memory stable_value: "signal-S";
    Memory counter: "signal-C";
    
    Signal matches_stored = input == stable_value.read();
    
    counter.write(0, when=matches_stored == 0);
    counter.write(counter.read() + 1, when=matches_stored);
    
    Signal reached_threshold = counter.read() >= ticks;
    stable_value.write(input, when=reached_threshold);
    
    return stable_value.read();
}


# -----------------------------------------------------------------------------
# Rate Limiter
# -----------------------------------------------------------------------------
# Outputs input value, but limits how fast it can change per tick.
#
func rate_limit(Signal input, int max_change) {
    Memory current: "signal-R";
    
    Signal delta = input - current.read();
    Signal clamped_delta = (delta > max_change) * max_change
                         + (delta < (0 - max_change)) * (0 - max_change)
                         + (delta >= (0 - max_change)) * (delta <= max_change) * delta;
    
    current.write(current.read() + clamped_delta);
    return current.read();
}


# -----------------------------------------------------------------------------
# Exponential Moving Average
# -----------------------------------------------------------------------------
# Using integer math: alpha = alpha_num / alpha_denom
#
func ema(Signal input, int alpha_num, int alpha_denom) {
    Memory avg: "signal-E";
    
    Signal delta = input - avg.read();
    Signal adjustment = (delta * alpha_num) / alpha_denom;
    
    avg.write(avg.read() + adjustment);
    return avg.read();
}
```

---

## Test Cases

### Test 1: Basic Counter

```fcdsl
Memory counter: "signal-A";
counter.write(counter.read() + 1);
Signal output = counter.read();
```

**Expected:** Optimizes to 1 arithmetic combinator with self-feedback.

### Test 2: Conditional Write

```fcdsl
Memory buffer: "signal-B";
Signal trigger = ("signal-T", 0);
Signal data = ("signal-D", 0);

buffer.write(data, when=trigger);
Signal output = buffer.read();
```

**Expected:** 2 decider combinators (SR latch), GREEN wire for signal-W, RED for data.

### Test 3: Multiple Conditional Writes

```fcdsl
Memory state: "signal-S";
Signal set_trigger = ("signal-1", 0);
Signal reset_trigger = ("signal-2", 0);

state.write(1, when=set_trigger);
state.write(0, when=reset_trigger);

Signal output = state.read();
```

**Expected:** Both writes compile, last true condition wins each tick.

### Test 4: Complex Feedback Chain

```fcdsl
Memory pattern: "signal-D";
Signal step1 = pattern.read() + 1;
Signal step2 = step1 * 3;
Signal step3 = step2 % 17;
pattern.write(step3);
Signal output = pattern.read();
```

**Expected:** Multi-combinator feedback chain with optimization.

---

## Implementation Checklist

### Grammar ([fcdsl.lark](dsl_compiler/grammar/fcdsl.lark))
- [ ] Remove `READ_KW "(" NAME ")"` from primary
- [ ] Remove `WRITE_KW "(" expr "," NAME ... ")"` from primary
- [ ] Add `memory_read`, `memory_write`, `memory_write_when` rules to method_call

### Parser ([transformer.py](dsl_compiler/src/parsing/transformer.py))
- [ ] Add `memory_read` handler
- [ ] Add `memory_write` handler  
- [ ] Add `memory_write_when` handler
- [ ] Remove old `read_expr` / `write_expr` handlers

### Documentation
- [ ] Update [LANGUAGE_SPEC.md](LANGUAGE_SPEC.md) Memory System section
- [ ] Update [LANGUAGE_SPEC.md](LANGUAGE_SPEC.md) Quick Start example
- [ ] Rewrite [doc/04_memory.md](doc/04_memory.md) with new syntax
- [ ] Update all sample programs in [tests/sample_programs/](tests/sample_programs/)

### Standard Library
- [ ] Create `lib/memory_patterns.fcdsl`

### Tests
- [ ] Update existing memory tests for new syntax
- [ ] Verify optimization still triggers
- [ ] Add parser tests for new grammar rules
