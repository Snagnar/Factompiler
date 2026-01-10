# SR/RS Latch Clarification for Factorio 2.0 DSL Implementation

This document clarifies the implementation of SR and RS latches in Factorio 2.0, including the binary nature of latches and how to output arbitrary values.

---

## Key Terminology

- **RS Latch (Reset Priority)**: When both Set AND Reset are active simultaneously, the latch **resets** (outputs 0). Reset has priority.
- **SR Latch (Set Priority)**: When both Set AND Reset are active simultaneously, the latch **sets** (outputs 1). Set has priority.

The first letter indicates which has priority when both inputs are true.

---

## CRITICAL: Latches are Binary Memory Cells

**SR/RS latches are inherently BINARY**. The underlying decider combinator outputs either:
- **1** when the latch condition is true (SET state)
- **0** when the latch condition is false (RESET state)

To output arbitrary values (e.g., 100 when latched), we need a **multiplier combinator**.

---

## DSL Syntax and Compilation Pattern

### User-Facing Syntax
```fcdsl
Memory pump_speed: "signal-P";
pump_speed.write(100, set=tank < 20, reset=tank >= 80);

Signal output = pump_speed.read();  # Returns 0 or 100
```

### Compilation Strategy

**Case 1: `write(1, set=..., reset=...)` → 1 combinator**
```
Decider Combinator:
  Condition: Set > Reset (RS) or multi-condition (SR)
  Output: memory_signal_type (e.g., signal-P) = 1
  Feedback: GREEN wire output → input
```
The latch outputs 0 or 1 directly on the memory's declared signal type.

**Case 2: `write(N, set=..., reset=...)` where N ≠ 1 → 2 combinators**
```
Decider Combinator (latch):
  Condition: Set > Reset (RS) or multi-condition (SR)
  Output: memory_signal_type = 1
  Feedback: GREEN wire output → input

Arithmetic Combinator (multiplier):
  Input: memory_signal_type from GREEN wire (0 or 1)
  Operation: memory_signal_type × N → memory_signal_type
```
The latch outputs 0 or 1, then the multiplier scales it to 0 or N.

**Case 3: `write(signal_expr, set=..., reset=...)` → 2 combinators**
```
Decider Combinator (latch):
  Condition: Set > Reset (RS) or multi-condition (SR)
  Output: memory_signal_type = 1
  Feedback: GREEN wire output → input

Arithmetic Combinator (multiplier):
  Left input: memory_signal_type from GREEN wire (0 or 1)
  Right input: signal_expr from RED wire
  Output: memory_signal_type
```

### Combinator Cost Summary
| Pattern | Combinators |
|---------|-------------|
| `write(1, set=..., reset=...)` | 1 |
| `write(N, set=..., reset=...)` where N ≠ 1 | 2 |
| `write(expr, set=..., reset=...)` | 2 |

---

## The Classic Single-Combinator RS Latch

### The Canonical Design: `S > R`

This is the simplest and most common single-combinator latch in Factorio:

**Decider Combinator Configuration:**
- **Condition:** `S > R`
- **Output:** `S = 1`
- **Feedback:** Output wire connected back to input (on a DIFFERENT wire color than inputs)

### Critical Insight: The Output Signal IS the Memory Signal Type

The design works because:

1. **The output signal type is the memory's declared signal type (M)**
2. **The feedback carries M back to the input on GREEN wire**
3. **When the latch is ON, the feedback adds M=1 to the input**
4. **The `S > R` comparison uses external S and R signals from RED wire**

So when latched ON with no external inputs:
- External S and R = 0 (from RED wire)
- Feedback M = 1 (from GREEN wire, but not used in S > R comparison)
- The latch output signal participates in feedback but NOT in the set/reset comparison

Wait - this reveals an important design consideration. For RS latch with `S > R`:
- External S goes to RED wire
- Feedback M goes to GREEN wire
- If M and S are the same signal type, feedback adds to S total

### Wire Color Configuration

```
┌─────────────────────────────────────┐
│         DECIDER COMBINATOR          │
│                                     │
│  Condition: S > R                   │
│  Output: M = 1 (memory signal type) │
│                                     │
│  INPUT ◄──── RED wire ──── External S and R signals
│        ◄──── GREEN wire ─┐ (feedback loop)
│                          │
│  OUTPUT ────────────────►┘
│         └──── GREEN wire ───► To controlled device
└─────────────────────────────────────┘
```

**The two wire colors serve distinct purposes:**
- **RED wire (input only):** Carries external Set (S) and Reset (R) signals
- **GREEN wire (feedback + output):** Carries the latch output; loops back to input AND goes to controlled devices

This separation is essential. If feedback were on the same wire as inputs, the signals would mix in unintended ways.

### Behavior Truth Table (RS Latch)

| External S | External R | Feedback S | Total S | Condition S > R | New Output |
|------------|------------|------------|---------|-----------------|------------|
| 0          | 0          | 0          | 0       | 0 > 0 = FALSE   | 0 (OFF)    |
| 1          | 0          | 0          | 1       | 1 > 0 = TRUE    | 1 (SET!)   |
| 0          | 0          | 1          | 1       | 1 > 0 = TRUE    | 1 (HOLD)   |
| 0          | 1          | 1          | 1       | 1 > 1 = FALSE   | 0 (RESET!) |
| 1          | 1          | 0          | 1       | 1 > 1 = FALSE   | 0 (R wins) |
| 1          | 1          | 1          | 2       | 2 > 1 = TRUE    | 1 (edge case) |

Note: When S=1 and R=1 are both provided externally and the latch is currently OFF, `1 > 1` is FALSE, so it stays OFF (Reset priority).

---

## SR Latch (Set Priority) in Factorio 2.0

### Using Multi-Condition Decider Combinators

Factorio 2.0 added the ability to have multiple conditions with AND/OR in a single decider. This enables a true SR latch (set priority) in one combinator.

### Hysteresis Example (the common use case)

For a backup steam generator that turns ON at 20% accumulator and OFF at 90%:

**Decider Configuration:**
```
Row 1: A < 20              [OR]
Row 2: S > 0  AND  A < 90
Output: S = 1
```

**Wire Configuration:**
- RED wire: Accumulator signal (A) to input
- GREEN wire: Feedback from output to input, and output to power switch

**Logic Explanation:**
- `A < 20`: SET condition - turn on when accumulator drops below 20%
- `S > 0 AND A < 90`: HOLD condition - stay on while output is active AND accumulator below 90%

Once A < 20 triggers, S becomes 1. Then even if A rises above 20, the second condition (`S > 0 AND A < 90`) keeps it latched until A reaches 90.

### Wire Filtering in 2.0

Factorio 2.0 decider combinators can filter which wire each condition reads from:
- You can specify "red only" or "green only" for individual conditions
- This prevents signal mixing and enables cleaner designs

For the hysteresis latch:
- Row 1 (`A < 20`): reads from RED wire only (external input)
- Row 2 (`S > 0`): reads from GREEN wire only (feedback)
- Row 2 (`A < 90`): reads from RED wire only (external input)

---

## Answering Your Specific Questions

### Q1: Should the SET signal be the same type as the OUTPUT signal?

**YES, for the classic RS latch design.**

The `S > R` latch requires that the output signal type be `S` (the same as the set signal) because:
1. The feedback must contribute to the `S` value in the comparison
2. When latched ON, feedback S=1 maintains `S > R` even with no external S input

If you used a different output signal (e.g., output `A = 1` while comparing `S > R`), the feedback wouldn't affect the condition and the latch wouldn't hold.

### Q2: What about separate Set and Reset signal types?

You CAN use different signal types for Set and Reset inputs. The key constraint is:
- **Output signal = Set signal** (so feedback works)

Example:
- Set signal: `signal-S` 
- Reset signal: `signal-R` (can be any different signal)
- Output signal: `signal-S` (must match Set signal)
- Condition: `S > R`

### Q3: DSL Syntax Implications

For your DSL, consider two approaches:

**Approach A: Explicit signal-based (matches Factorio's model)**
```fcdsl
# User specifies the latch signal type; Set signal must match
Memory latch_state: "signal-S";

# Set condition provides signal-S, reset provides signal-R
latch_state.rs_latch(set=external_S, reset=external_R);
```

**Approach B: Threshold-based for hysteresis (common use case)**
```fcdsl
Memory pump_control: "signal-S";

# Internally generates the multi-condition decider
pump_control.hysteresis(
    input=accumulator_signal,    # What signal to monitor
    low_threshold=20,            # SET when input < low
    high_threshold=90            # RESET when input >= high  
);
```

### Q4: Is the current RS latch implementation correct?

Based on your description, **mostly yes**, but verify:

1. ✅ Single decider with `S > R` condition
2. ✅ Output value = 1
3. ⚠️ **Output signal type must be `S`** (not a separate memory signal type)
4. ✅ Green feedback wire from output to input
5. ✅ Red wire for external S and R inputs

**The issue:** If your DSL allows `Memory latch_state: "signal-A"` and then uses external `signal-S` and `signal-R` for set/reset with condition `S > R`, but outputs `signal-A = 1`, **the latch won't hold**. The output must be `S = 1`.

---

## Summary: Correct Single-Combinator Latch Designs

### RS Latch (Reset Priority)
```
Condition: S > R
Output: S = 1
Feedback: Green wire, output → input
External inputs: Red wire (S and R signals)
```

### SR Latch (Set Priority) - Factorio 2.0

For SR latch (set priority), we use multi-condition with separate feedback signal.

**IMPORTANT:** Factorio 2.0 evaluates multi-conditions LEFT-TO-RIGHT without operator precedence.
We want the logic: `(L > 0 AND R = 0) OR S > 0`
So we must order conditions as: L > 0 [first], R = 0 [AND], S > 0 [OR]

```
┌─────────────────────────────────────────────────────┐
│  CONDITIONS (evaluated left-to-right):              │
│                                                     │
│  Row 1: L > 0        (green wire only)   [first]   │
│  Row 2: R = 0        (red wire only)     [AND]     │
│  Row 3: S > 0        (red wire only)     [OR]      │
│                                                     │
│  Evaluation: ((L > 0) AND (R = 0)) OR (S > 0)      │
│                                                     │
│  OUTPUT: L = 1                                      │
└─────────────────────────────────────────────────────┘

Wiring:
  - RED wire (input):   External S and R signals
  - GREEN wire (input): Feedback from own output (L)
  - GREEN wire (output): To controlled device + feedback to input
```

**Logic breakdown:**
- **Row 1 (HOLD check):** `L > 0` - Feedback signal (latch is currently ON)
- **Row 2 (AND NOT RESET):** `R = 0` - Combined with Row 1: stay on if latched AND no reset
- **Row 3 (OR SET):** `S > 0` - Override: turn on if set signal active (set priority)
- Combined: Output 1 if `(latched AND no reset) OR set_active`

**Truth Table:**
| S | R | L (feedback) | Row 1 | R1&R2 | R1&R2|R3 | Result |
|---|---|--------------|-------|-------|---------|--------|
| 0 | 0 | 0 | F | F | F | Hold OFF |
| 0 | 0 | 1 | T | T | T | Hold ON |
| 1 | 0 | 0 | F | F | T | SET |
| 1 | 0 | 1 | T | T | T | Stay ON |
| 0 | 1 | 0 | F | F | F | Stay OFF |
| 0 | 1 | 1 | T | F | F | RESET |
| 1 | 1 | 0 | F | F | T | **Set wins** |
| 1 | 1 | 1 | T | F | T | **Set wins** |

Note: The output signal L can be any signal (the memory's declared type).
This is different from RS latch where output MUST be S for feedback to work.

### Hysteresis Latch (Most Practical)
```
Row 1: A < low_threshold (red only)    [OR]
Row 2: S > 0 (green only) AND A < high_threshold (red only)
Output: S = 1
```

---

## Key Takeaways for DSL Implementation

1. **For RS latch (`S > R`):** The output signal MUST be the same type as the Set signal
2. **Wire separation is mandatory:** External inputs on one color, feedback on another
3. **Factorio 2.0 multi-conditions:** Enable SR latch and hysteresis in a single combinator
4. **Wire filtering:** 2.0 allows per-condition wire source specification (red_only/green_only)
5. **The "memory signal" concept in your DSL may need revision:** The signal isn't arbitrary; it's tied to the set signal for the `S > R` design to work