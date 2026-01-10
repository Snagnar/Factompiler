# SR/RS Latch Implementation Status

## Current State Summary

The SR/RS latch implementation is partially complete but has several issues that need to be resolved.

---

## What Works

1. **Grammar and Parsing**: The `write(value, set=condition, reset=condition)` syntax parses correctly.

2. **IR Generation**: `IR_LatchWrite` nodes are created with the correct latch type (SR vs RS) based on argument order.

3. **Basic Wire Connections**: 
   - External signals connect to latch input via RED wire
   - Feedback connects via GREEN wire (output to input self-loop)
   - Latch output connects to consumers (e.g., lamps)

4. **RS Latch Decider**: Creates a single-condition decider with `S > R`.

5. **SR Latch Decider**: Creates a multi-condition decider with wire filtering.

---

## What Needs Fixing

### Issue 1: SR Latch Multi-Condition May Not Be Working

**Symptom**: User reports the multi-condition for SR latch "does not exist" in the generated blueprint.

**Current Implementation** (in `memory_builder.py`):
```python
conditions = [
    {
        "comparator": ">",
        "compare_type": "or",
        "first_signal": set_signal_name,
        "first_signal_wires": {"red"},
        "second_signal": reset_signal_name,
        "second_signal_wires": {"red"},
    },
    {
        "comparator": ">",
        "compare_type": "or",
        "first_signal": set_signal_name,
        "first_signal_wires": {"green"},
        "second_constant": 0,
    },
]
```

**Potential Issues**:
- The `conditions` property may not be passed correctly to `create_and_add_placement()`
- The entity emitter may not be using the `conditions` list for multi-condition mode
- Wire filtering (`first_signal_wires`, `second_signal_wires`) may not translate correctly to Factorio format

**Files to Check**:
- `dsl_compiler/src/layout/memory_builder.py` - `_create_sr_latch_placement()`
- `dsl_compiler/src/layout/layout_plan.py` - `create_and_add_placement()` 
- `dsl_compiler/src/emission/entity_emitter.py` - `_configure_decider()` and `_configure_decider_multi_condition()`

---

### Issue 2: Signal Type Constraints Not Enforced

**Symptom**: Memory signal type same as set signal type may cause issues with reset.

**Required Behavior from Fundamentals**:
- For RS latch: Output signal MUST be same type as set signal (for feedback to work)
- Set and Reset signals can be any different types
- External inputs come via RED, feedback via GREEN

**Current State**:
- We warn if memory type differs from set signal type
- We auto-change memory type to match set signal

**What Might Be Missing**:
- Validation that set signal ≠ reset signal types (they should be different for proper comparison)
- Proper wire color assignment for external inputs vs feedback

**Files to Check**:
- `dsl_compiler/src/layout/memory_builder.py` - `handle_latch_write()`
- Semantic analyzer may need validation

---

### Issue 3: Hysteresis SR Latch Design Not Implemented

**Symptom**: Current SR latch design differs from hysteresis example in fundamentals.

**Fundamentals Example**:
```
Row 1: A < 20              [OR]
Row 2: S > 0  AND  A < 90
Output: S = 1
```

**Current Implementation**:
```
Row 1: S > R (RED only) [OR]
Row 2: S > 0 (GREEN only)
```

**Issue**: The hysteresis example uses AND between `S > 0` and `A < 90`, but our SR latch just uses OR between conditions.

**Question**: Is the current design correct for a simple SR latch, or do we need the AND-based hysteresis design?

---

## Files Modified During This Session

1. **`dsl_compiler/src/lowering/memory_lowerer.py`**:
   - `_lower_latch_write()` now returns `SignalRef(expected_signal_type, memory_id)` instead of calling `_coerce_to_signal_type()` which was creating extra constant combinators

2. **`dsl_compiler/src/layout/memory_builder.py`**:
   - `handle_latch_write()` refactored with separate methods for RS and SR latches
   - `_create_rs_latch_placement()` for single-condition RS latch
   - `_create_sr_latch_placement()` for multi-condition SR latch with wire filtering
   - `handle_read()` now checks for `latch_combinator` to connect reads properly

---

## Test Files

- `tests/sample_programs/30_rs_latch.fcdsl` - RS latch test with lamp
- `tests/sample_programs/31_sr_latch.fcdsl` - SR latch test

---

## Recommended Next Steps

1. **Verify entity emitter handles multi-condition mode correctly**:
   - Check if `conditions` key is being processed in `_configure_decider()`
   - Trace the path from placement properties to draftsman API calls

2. **Test the generated blueprint in Factorio**:
   - Import the blueprint
   - Check decider combinator settings
   - Verify wire connections and colors

3. **Add validation for signal types**:
   - Ensure set signal and reset signal are different types
   - Ensure memory output type equals set signal type

4. **Consider if hysteresis design is needed**:
   - Current simple SR latch may not be true "set priority"
   - May need AND condition like the hysteresis example
