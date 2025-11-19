# Wire Color Filtering Refactoring Guide

## Problem Statement

**Bug**: Arithmetic and Decider combinators are reading from BOTH red and green wires simultaneously, adding the values together before performing operations. This is incorrect when we've deliberately assigned different input signals to different wire colors for signal isolation.

**Example**: In `00_wire_separation.fcdsl`:
- `input1 = 50` on RED wire
- `input2 = 30` on GREEN wire  
- Multiplication combinator should see `50 * 30 = 1500`
- **Currently sees**: `(50+30) * (50+30) = 80 * 80 = 6400` ❌

**Root Cause**: We assign wire colors correctly for wiring topology, but don't configure the combinator's `first_operand_wires` and `second_operand_wires` fields to filter which wire colors each operand reads from.

---

## Technical Background

### Factorio 2.0 Wire Color Filtering

In Factorio 2.0, combinators have fields that control which wire colors contribute to each operand:

```json
{
  "control_behavior": {
    "arithmetic_conditions": {
      "first_signal": {"name": "signal-A", "type": "virtual"},
      "first_signal_networks": {"red": false},  // ← Only read from GREEN
      "operation": "*",
      "second_signal": {"name": "signal-A", "type": "virtual"},
      "second_signal_networks": {"green": false},  // ← Only read from RED
      "output_signal": {"name": "signal-A", "type": "virtual"}
    }
  }
}
```

**Draftsman API**:
- `ArithmeticCombinator.first_operand_wires` → serializes to `first_signal_networks`
- `ArithmeticCombinator.second_operand_wires` → serializes to `second_signal_networks`
- Type: `CircuitNetworkSelection` (a set like `{"red"}`, `{"green"}`, or `{"red", "green"}`)
- Default: `{"red", "green"}` (reads from both, adds values)

**DeciderCombinator** has similar fields:
- `DeciderCombinator.Condition.first_signal_networks` (for each condition)
- `DeciderCombinator.Condition.second_signal_networks` (for each condition)
- Note: DeciderCombinator uses a `Condition` object, not direct attributes

---

## Current Architecture

### 1. Wire Color Assignment (Working Correctly ✅)

**File**: `dsl_compiler/src/layout/connection_planner.py`

The `ConnectionPlanner` correctly assigns red/green colors to wires:
1. Builds conflict graph of sources feeding same sinks
2. Performs bipartite coloring to isolate signals
3. Stores in `_edge_color_map: Dict[(source_id, sink_id, signal), wire_color]`
4. Creates `WireConnection` objects with `wire_color` field

**This part works!** Wires ARE colored correctly.

### 2. Entity Creation (Missing Wire Filters ❌)

**File**: `dsl_compiler/src/emission/entity_emitter.py`

The `_configure_arithmetic()` and `_configure_decider()` methods create combinators but don't set wire filters:

```python
def _configure_arithmetic(self, entity: ArithmeticCombinator, props: Dict[str, Any]) -> None:
    entity.first_operand = left_operand
    entity.second_operand = right_operand
    entity.operation = operation
    entity.output_signal = output_signal
    # ❌ Missing: entity.first_operand_wires = ???
    # ❌ Missing: entity.second_operand_wires = ???
```

**Problem**: We need to know which wire colors are connected to this combinator for each operand's signal.

---

## Solution Architecture

### Overview

We need to propagate wire color information from `ConnectionPlanner` to `entity_emitter.py` so that when we configure combinators, we can set the correct wire filters.

### Data Flow

```
IR_Arith/IR_Decider
    ↓
EntityPlacer (creates EntityPlacement with properties)
    ↓
LayoutPlanner → ConnectionPlanner (assigns wire colors)
    ↓
[NEW] Store wire color info in EntityPlacement.properties
    ↓
entity_emitter.py (reads wire colors, sets operand_wires)
```

### Key Insight

When `ConnectionPlanner.plan_connections()` runs, it:
1. Knows which entities are connected
2. Knows which signals they're consuming
3. Knows which wire colors are used for each connection

We need to aggregate this information PER ENTITY PER OPERAND and store it so the emitter can use it.

---

## Detailed Refactoring Plan

### Phase 1: Extend ConnectionPlanner to Track Operand Wire Colors

**File**: `dsl_compiler/src/layout/connection_planner.py`

**Changes**:

1. **Add new data structure** to track which wire colors feed each entity's operand signals:

```python
class ConnectionPlanner:
    def __init__(self, ...):
        # ... existing fields ...
        # NEW: Maps (entity_id, signal_name) → set of wire colors
        self._entity_signal_wire_colors: Dict[Tuple[str, str], Set[str]] = {}
```

2. **Populate during `_populate_wire_connections()`**:

After determining the wire color for an edge, record which colors deliver which signals to which entities:

```python
def _populate_wire_connections(self) -> None:
    # ... existing code ...
    
    for edge in self._circuit_edges:
        # ... determine color ...
        
        # NEW: Track which wire colors deliver this signal to the sink
        sink_key = (edge.sink_entity_id, edge.resolved_signal_name)
        if sink_key not in self._entity_signal_wire_colors:
            self._entity_signal_wire_colors[sink_key] = set()
        self._entity_signal_wire_colors[sink_key].add(color)
        
        # ... create WireConnection ...
```

3. **Add accessor method**:

```python
def get_signal_wire_colors(self, entity_id: str, signal_name: str) -> Set[str]:
    """Get which wire colors (red/green) deliver a signal to an entity."""
    return self._entity_signal_wire_colors.get((entity_id, signal_name), {"red", "green"})
```

**Rationale**: This centralizes wire color knowledge in the ConnectionPlanner where it's computed.

---

### Phase 2: Propagate Wire Colors to EntityPlacement

**File**: `dsl_compiler/src/layout/planner.py`

**Changes**:

After `connection_planner.plan_connections()` completes, inject wire color info into entity placements:

```python
def _create_connectivity(self, signal_graph, wire_merge_junctions) -> None:
    # ... existing code ...
    
    connection_planner.plan_connections(signal_graph, entities, wire_merge_junctions, locked_colors)
    
    # NEW: Inject wire colors into combinator placements
    self._inject_wire_colors_into_placements(connection_planner)
    
def _inject_wire_colors_into_placements(self, connection_planner: ConnectionPlanner) -> None:
    """Store wire color information in combinator placement properties."""
    for placement in self.layout_plan.placements.values():
        if placement.entity_type not in ("arithmetic-combinator", "decider-combinator"):
            continue
        
        # Get the operand signals from placement properties
        left_signal = placement.properties.get("left_operand")
        right_signal = placement.properties.get("right_operand")
        
        # Skip if operands are constants (wire colors don't apply)
        if isinstance(left_signal, int) or isinstance(right_signal, int):
            continue
        
        # Get wire colors for each operand
        if left_signal and not isinstance(left_signal, int):
            left_colors = connection_planner.get_signal_wire_colors(
                placement.ir_node_id, left_signal
            )
            placement.properties["left_operand_wires"] = left_colors
        
        if right_signal and not isinstance(right_signal, int):
            right_colors = connection_planner.get_signal_wire_colors(
                placement.ir_node_id, right_signal
            )
            placement.properties["right_operand_wires"] = right_colors
```

**Important Note**: We need to handle signal name resolution. The `left_operand` in properties might be a virtual signal name like `"signal-A"`. We need to ensure this matches what's tracked in `_entity_signal_wire_colors`.

---

### Phase 3: Update Entity Emitter to Use Wire Colors

**File**: `dsl_compiler/src/emission/entity_emitter.py`

**Changes**:

1. **Update `_configure_arithmetic()`**:

```python
def _configure_arithmetic(self, entity: ArithmeticCombinator, props: Dict[str, Any]) -> None:
    operation = props.get("operation", "+")
    left_operand = props.get("left_operand")
    right_operand = props.get("right_operand")
    output_signal = props.get("output_signal")
    
    # NEW: Get wire color filters
    left_operand_wires = props.get("left_operand_wires", {"red", "green"})
    right_operand_wires = props.get("right_operand_wires", {"red", "green"})
    
    # Validate signal-each usage per Draftsman requirements
    if output_signal == "signal-each":
        if left_operand != "signal-each" and right_operand != "signal-each":
            output_signal = "signal-0"
    
    entity.first_operand = left_operand
    entity.second_operand = right_operand
    entity.operation = operation
    entity.output_signal = output_signal
    
    # NEW: Set wire color filters
    entity.first_operand_wires = left_operand_wires
    entity.second_operand_wires = right_operand_wires
```

2. **Update `_configure_decider()`**:

```python
def _configure_decider(self, entity: DeciderCombinator, props: Dict[str, Any]) -> None:
    operation = props.get("operation", "=")
    left_operand = props.get("left_operand")
    right_operand = props.get("right_operand")
    output_signal = props.get("output_signal")
    output_value = props.get("output_value", 1)
    copy_count = props.get("copy_count_from_input", False)
    
    # NEW: Get wire color filters
    left_operand_wires = props.get("left_operand_wires", {"red", "green"})
    right_operand_wires = props.get("right_operand_wires", {"red", "green"})
    
    # Build condition
    condition_kwargs = {"comparator": operation}
    if isinstance(left_operand, int):
        condition_kwargs["first_signal"] = "signal-0"
        condition_kwargs["constant"] = left_operand
    else:
        condition_kwargs["first_signal"] = left_operand
        condition_kwargs["first_signal_networks"] = left_operand_wires  # NEW
    
    if isinstance(right_operand, int):
        condition_kwargs["constant"] = right_operand
    else:
        condition_kwargs["second_signal"] = right_operand
        condition_kwargs["second_signal_networks"] = right_operand_wires  # NEW
    
    entity.conditions = [DeciderCombinator.Condition(**condition_kwargs)]
    
    # Configure output
    output_kwargs = {
        "signal": output_signal,
        "copy_count_from_input": copy_count,
    }
    if not copy_count and isinstance(output_value, int):
        output_kwargs["constant"] = output_value
    
    entity.outputs = [DeciderCombinator.Output(**output_kwargs)]
```

---

### Phase 4: Handle Edge Cases

**Important Considerations**:

1. **Constant Operands**: If `left_operand` or `right_operand` is an integer, wire colors don't matter. Skip wire color assignment.

2. **Single Wire Connections**: If only ONE wire color feeds a signal, set that explicitly. Example:
   - Signal comes from RED only → `{"red"}`
   - Signal comes from GREEN only → `{"green"}`
   - Signal comes from BOTH → `{"red", "green"}` (default behavior)

3. **Signal Name Resolution**: Ensure the signal names used in `_entity_signal_wire_colors` match those in `placement.properties["left_operand"]`.

4. **Memory Feedback Edges**: These use GREEN wires explicitly. Already handled by existing code, but verify compatibility.

5. **Wire Merge Junctions**: These may combine signals from multiple sources. Need to aggregate all wire colors.

6. **Relay Connections**: When signals route through relays, track color changes correctly.

---

## Implementation Steps

### Step 1: Add Wire Color Tracking to ConnectionPlanner
- [ ] Add `_entity_signal_wire_colors` dictionary
- [ ] Populate it in `_populate_wire_connections()`
- [ ] Add `get_signal_wire_colors()` accessor
- [ ] Add unit test for tracking

### Step 2: Inject into EntityPlacements
- [ ] Add `_inject_wire_colors_into_placements()` to LayoutPlanner
- [ ] Call after `plan_connections()`
- [ ] Handle signal name resolution
- [ ] Add debug logging to verify injection

### Step 3: Update Entity Emitter
- [ ] Modify `_configure_arithmetic()`
- [ ] Modify `_configure_decider()`
- [ ] Add fallback to `{"red", "green"}` for safety

### Step 4: Testing
- [ ] Test `00_wire_separation.fcdsl` - should show correct multiplication
- [ ] Test simple 2-input multiplication
- [ ] Test complex memory programs (shouldn't break)
- [ ] Verify single-wire scenarios
- [ ] Check that constants don't get wire filters

### Step 5: Validation
- [ ] Inspect generated blueprints in game
- [ ] Verify combinator behavior matches expected
- [ ] Run full test suite
- [ ] Check performance impact (should be minimal)

---

## Potential Issues & Mitigations

### Issue 1: Signal Name Mismatch
**Problem**: `_entity_signal_wire_colors` uses resolved signal names, but `left_operand` might be different.

**Mitigation**: Ensure signal resolution happens consistently. May need to use the same resolution logic from `SignalAnalyzer`.

### Issue 2: Relay Color Changes
**Problem**: When signal routes through relay, color might change.

**Mitigation**: Track wire color at the FINAL hop to the combinator, not the source. The `_populate_wire_connections()` loop already does this.

### Issue 3: Wire Merge Aggregation
**Problem**: Wire merges combine multiple sources, potentially on different colors.

**Mitigation**: Aggregate all colors for a merged signal. This is already what we want - read from all contributing wires.

### Issue 4: Default Behavior Change
**Problem**: Existing programs might rely on implicit both-wire reading.

**Mitigation**: Default to `{"red", "green"}` if no explicit wire colors found. This preserves existing behavior for simple cases.

---

## Testing Strategy

### Unit Tests
1. Test `_entity_signal_wire_colors` correctly tracks single-color connections
2. Test aggregation when signal comes from multiple colors
3. Test constant operands don't get wire filters

### Integration Tests
1. Compile `00_wire_separation.fcdsl` and verify blueprint JSON has correct `first_signal_networks` / `second_signal_networks`
2. Import to game and verify combinator output values
3. Test memory programs still work correctly

### Regression Tests
1. Run all existing sample programs
2. Verify output blueprints are valid
3. Check that simple programs without conflicts still work

---

## Success Criteria

✅ `00_wire_separation.fcdsl` multiplication shows `50 * 30 = 1500` (not `80 * 80`)
✅ Blueprint JSON contains correct `first_signal_networks` / `second_signal_networks` fields
✅ All existing sample programs still compile correctly
✅ No performance degradation
✅ Code is well-documented with comments explaining wire color filtering

---

## Alternative Approaches Considered

### Alternative 1: Set Wire Filters During Wiring
**Idea**: When creating `WireConnection`, also update the combinator's wire filters.

**Rejected**: Too tightly coupled. Wiring logic shouldn't modify entity properties. Separation of concerns is cleaner.

### Alternative 2: Post-Process After Emission
**Idea**: After all entities are created, scan connections and update wire filters.

**Rejected**: Requires second pass, more complex logic. Better to set correctly during entity creation.

### Alternative 3: Store in IR Properties
**Idea**: Add wire color info to `IR_Arith` / `IR_Decider` nodes during IR construction.

**Rejected**: IR shouldn't know about physical wire colors. That's a layout concern, not a semantic one.

---

## Files to Modify

1. `dsl_compiler/src/layout/connection_planner.py` - Track wire colors per entity/signal
2. `dsl_compiler/src/layout/planner.py` - Inject wire colors into placements
3. `dsl_compiler/src/emission/entity_emitter.py` - Use wire colors when configuring combinators

**Estimated LOC**: ~100 lines added, ~10 lines modified

**Risk Level**: Medium - touches critical path, but changes are localized and testable

---

## Rollback Plan

If this introduces issues:
1. Comment out `_inject_wire_colors_into_placements()` call
2. Remove wire filter assignments in entity_emitter
3. System reverts to current (broken but stable) behavior

---

## Future Enhancements

1. **Smart Optimization**: If both operands read from same wire color, could merge them on single wire
2. **Wire Color Hints**: Allow user to specify wire colors in DSL for manual control
3. **Visual Debug**: Add wire color info to debug output for easier troubleshooting

---

## Conclusion

This refactoring correctly implements wire color filtering for combinator operands, fixing the bug where inputs from different wires are incorrectly summed. The approach is:

- **Minimal invasive**: Only 3 files modified
- **Architecturally sound**: Data flows naturally through existing pipeline  
- **Testable**: Clear success criteria and test cases
- **Safe**: Defaults preserve existing behavior for simple cases
- **Maintainable**: Well-documented with clear responsibilities

The key insight is that wire color information is already computed correctly - we just need to propagate it to the final entity configuration step.
