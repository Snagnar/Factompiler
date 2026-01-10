# SR/RS Latch Implementation Plan (Revised)

## Overview

This document describes the implementation of SR (Set-Reset) and RS (Reset-Set) latches in the compiler. SR/RS latches are **binary memory cells** that output 0 or 1, with an optional multiplier combinator to scale the output to arbitrary values.

---

## Key Insight: Latches are Binary

The underlying decider combinator in an SR/RS latch can only output:
- **1** when latched (SET state)
- **0** when not latched (RESET state)

To output arbitrary values like 100, we need a **multiplier combinator** after the latch.

---

## Syntax

```fcdsl
memory.write(value, set=condition1, reset=condition2)
```

**Parameters:**
- `value`: The value to output when latch is ON (can be 1, constant N, or an expression)
- `set=condition1`: Condition signal - when > 0, latch turns ON
- `reset=condition2`: Condition signal - when > 0, latch turns OFF

**Argument order determines priority:**
- `set=..., reset=...` → SR latch (Set Priority) - if both true, stays ON
- `reset=..., set=...` → RS latch (Reset Priority) - if both true, turns OFF

---

## Compilation Patterns

### Pattern 1: `write(1, set=..., reset=...)` → 1 combinator

The simplest case - latch outputs 1 directly.

```
Decider Combinator (latch):
  Condition: S > R (RS) or multi-condition (SR)
  Output: memory_signal = 1
  Feedback: GREEN wire from output to input

Wire Configuration:
  RED wire (input): Set (S) and Reset (R) conditions
  GREEN wire: Feedback loop + output to consumers
```

### Pattern 2: `write(N, set=..., reset=...)` where N ≠ 1 → 2 combinators

Latch outputs 1, multiplier scales to N.

```
Decider Combinator (latch):
  Condition: S > R (RS) or multi-condition (SR)
  Output: __latch_{memory_id} = 1  (internal signal)
  Feedback: GREEN wire from output to input

Arithmetic Combinator (multiplier):
  Operation: __latch_{memory_id} × N → memory_signal
  
Wire Configuration:
  Latch output connects to multiplier input
  Multiplier output is the memory's read source
```

### Pattern 3: `write(expr, set=..., reset=...)` → 2 combinators

Latch outputs 1, multiplier scales by expression value.

```
Decider Combinator (latch):
  Condition: S > R (RS) or multi-condition (SR)
  Output: __latch_{memory_id} = 1  (internal signal)
  Feedback: GREEN wire from output to input

Arithmetic Combinator (multiplier):
  Operation: __latch_{memory_id} × expr_signal → memory_signal
  Inputs: latch output + expression value
```

---

## RS Latch Circuit (Reset Priority)

Single condition: `S > R`

```
┌─────────────────────────────────────┐
│         DECIDER COMBINATOR          │
│                                     │
│  Condition: S > R                   │
│  Output: S = 1                      │
│                                     │
│  INPUT ◄──── RED wire ──── Set(S) + Reset(R) signals
│        ◄──── GREEN wire ─┐ (feedback)
│                          │
│  OUTPUT ────────────────►┘ (GREEN)
└─────────────────────────────────────┘
```

**Truth Table:**
| S (external) | R (external) | S (feedback) | Total S | S > R | Output |
|--------------|--------------|--------------|---------|-------|--------|
| 0 | 0 | 0 | 0 | FALSE | 0 |
| 1 | 0 | 0 | 1 | TRUE  | 1 (SET) |
| 0 | 0 | 1 | 1 | TRUE  | 1 (HOLD) |
| 0 | 1 | 1 | 1 | FALSE | 0 (RESET) |
| 1 | 1 | 0 | 1 | FALSE | 0 (R wins) |

---

## SR Latch Circuit (Set Priority) - Factorio 2.0

Multi-condition with wire filtering:

```
Row 1: S > R (read from RED wire only) [OR]
Row 2: S > 0 (read from GREEN wire only - feedback)
Output: S = 1
```

This ensures that when R is active but feedback S=1 is present, the latch stays ON (set priority).

```
┌─────────────────────────────────────┐
│         DECIDER COMBINATOR          │
│                                     │
│  Condition: (S > R [red]) OR (S > 0 [green])  │
│  Output: S = 1                      │
│                                     │
│  INPUT ◄──── RED wire ──── Set(S) + Reset(R) signals
│        ◄──── GREEN wire ─┐ (feedback)
│                          │
│  OUTPUT ────────────────►┘ (GREEN)
└─────────────────────────────────────┘
```

---

## Signal Type Handling

### Critical Constraint
The **output signal must be the same type as the set signal** for feedback to work.

When set/reset signals come from user conditions:
1. The **set condition's signal type** becomes the latch's output signal type
2. If memory declared type differs from set signal type, issue a **warning**
3. Update `memory_types` dict so subsequent reads return the correct type

### Wire Color Separation
- **RED wire**: External set and reset condition signals
- **GREEN wire**: Feedback from latch output to input + output to consumers

---

## Implementation Checklist

### Already Implemented ✅
- [x] Grammar: `write(value, set=..., reset=...)` syntax
- [x] AST: `WriteExpr` with `set_signal`, `reset_signal`, `set_priority`
- [x] IR: `IR_LatchWrite` node with latch type
- [x] Memory builder: Basic latch placement creation
- [x] Entity emitter: Single-condition decider configuration

### Needs Implementation/Fix 🔧

#### 1. Multiplier Pattern (memory_builder.py + memory_lowerer.py)
- [ ] Detect if value is literal 1 vs other constant vs expression
- [ ] For value=1: single combinator (current behavior, mostly correct)
- [ ] For value≠1: create latch + arithmetic multiplier combinator
- [ ] Use internal signal for latch output when multiplier needed
- [ ] Wire latch output to multiplier input
- [ ] Update signal graph so reads point to multiplier output

#### 2. Signal Type Coercion (memory_lowerer.py)
- [x] Update memory_types when latch write detected (already done)
- [ ] Ensure the latch outputs on set signal type, not memory declared type

#### 3. SR Latch Multi-Condition (memory_builder.py + entity_emitter.py)
- [x] `_create_sr_latch_placement()` creates conditions list
- [ ] **Verify** conditions flow through to entity emitter
- [ ] **Verify** `_configure_decider_multi_condition()` sets conditions correctly
- [ ] **Test** that wire filtering (red/green) is applied

#### 4. Wire Connections (memory_builder.py)
- [x] Green wire feedback from output to input
- [ ] Ensure set/reset conditions connect via RED wire
- [ ] For multiplier pattern: wire latch → multiplier → consumers

#### 5. Cleanup Standard Memory Gates
- [x] Mark write_gate and hold_gate as unused when latch created
- [x] cleanup_unused_gates() removes them

---

## Test Cases

### Test 1: Simple RS Latch (value=1)
```fcdsl
Memory toggle: "signal-S";
Signal button = ("signal-S", 0);
Signal clear = ("signal-R", 0);
toggle.write(1, reset=clear, set=button);
```
Expected: 1 decider combinator with S > R condition.

### Test 2: RS Latch with Multiplier (value=100)
```fcdsl
Memory pump_speed: "signal-P";
Signal low = ("signal-L", 0);   # tank < 20
Signal high = ("signal-H", 0);  # tank >= 80
pump_speed.write(100, set=low, reset=high);
```
Expected: 1 decider (latch) + 1 arithmetic (multiplier).

### Test 3: SR Latch (set priority)
```fcdsl
Memory state: "signal-S";
Signal set_sig = ("signal-S", 0);
Signal reset_sig = ("signal-R", 0);
state.write(1, set=set_sig, reset=reset_sig);  # set= first = set priority
```
Expected: 1 decider with multi-condition: (S > R [red]) OR (S > 0 [green]).

---

## Files to Modify

1. **memory_lowerer.py** - Detect value type, decide 1 vs 2 combinator pattern
2. **memory_builder.py** - Create multiplier combinator when needed, wire connections
3. **entity_emitter.py** - Ensure multi-condition configuration works (may be fine)
4. **layout_plan.py** - No changes expected
5. **ir/nodes.py** - May need to update IR_LatchWrite to carry "needs_multiplier" flag

---

## Debugging Checklist

If SR latch multi-condition doesn't appear in blueprint:
1. Check `_create_sr_latch_placement()` creates conditions list correctly
2. Check `create_and_add_placement()` passes conditions to properties
3. Check `PlanEntityEmitter._configure_decider_multi_condition()` is called
4. Check Draftsman's DeciderCombinator.Condition format is correct
5. Check entity.conditions is being set (not just returned)
