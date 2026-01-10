# Condition Folding Optimization Plan

## Overview

In Factorio 2.0, decider combinators support multiple conditions that can be combined with AND/OR logic. This optimization takes advantage of that feature to fold multiple separate comparison deciders into a single multi-condition decider combinator, reducing entity count.

## Current State

### How Logical AND/OR Currently Works

When the DSL compiles `(a > 0) && (b < 10)`:

1. **Current Implementation (without optimization):**
   - Creates decider 1: `a > 0 → 1`
   - Creates decider 2: `b < 10 → 1`
   - Creates arithmetic: `result1 * result2` (for AND)
   - Result: **3 combinators**

2. **With Condition Folding Optimization:**
   - Creates single decider with two conditions: `(a > 0) AND (b < 10) → 1`
   - Result: **1 combinator**

### Factorio 2.0 Multi-Condition Deciders

A single decider combinator can have multiple condition rows, each combined with AND/OR:

```
Condition 1: signal-A > 0
AND Condition 2: signal-B < 10
AND Condition 3: signal-C = 5
→ Output: 1
```

The conditions are evaluated as: `(cond1) AND (cond2) AND (cond3)`.

Mixed AND/OR is also supported:
```
Condition 1: signal-A > 0
OR Condition 2: signal-B > 0
→ Output: 1
```

## Scope

### What This Optimization Will Handle

1. **Simple Logical AND chains:**
   - `(a > 0) && (b < 10)` → single decider with 2 AND conditions
   - `(a > 0) && (b < 10) && (c == 5)` → single decider with 3 AND conditions

2. **Simple Logical OR chains:**
   - `(a > 0) || (b > 0)` → single decider with 2 OR conditions
   - `(a > 0) || (b > 0) || (c > 0)` → single decider with 3 OR conditions

3. **Comparisons involving:**
   - Signal vs constant: `signal > 5`
   - Signal vs signal: `signal_a > signal_b`
   - Bundle with any()/all() vs constant/signal

### What This Optimization Will NOT Handle (out of scope for now)

1. **Mixed AND/OR expressions:**
   - `(a > 0) && (b > 0) || (c > 0)` - requires parenthesis analysis
   - These continue to use the current multi-combinator approach

2. **Non-comparison operands:**
   - `x && y` where x and y are arbitrary integer signals (not comparisons)
   - These must use the `!= 0` booleanization approach

3. **Nested complex expressions:**
   - `((a > 0) && (b > 0)) && ((c > 0) && (d > 0))`
   - Would require flattening nested structures

## Implementation Plan

### Phase 1: IR Layer - New IR Node Support

The IR already has `DeciderCondition` dataclass but it stores signal names as strings (for SR latches built at layout time). For the optimization, we need to work with `ValueRef` objects during IR construction.

**Key Insight:** We have two use cases:
1. **Layout-time conditions** (SR latches): Uses signal name strings, constructed in `memory_builder.py`
2. **IR-time conditions** (condition folding): Uses `ValueRef` objects, constructed during lowering

**Solution:** Extend `DeciderCondition` to optionally hold ValueRefs, and extend IR_Decider:

**File: `dsl_compiler/src/ir/nodes.py`**

```python
@dataclass
class DeciderCondition:
    """Single condition row in a decider combinator (Factorio 2.0 multi-condition support)."""

    comparator: str = ">"
    
    # String-based operands (for layout-time construction like SR latches)
    first_signal: str = ""
    first_constant: Optional[int] = None
    first_signal_wires: Optional[Set[str]] = None
    
    second_signal: str = ""
    second_constant: Optional[int] = None
    second_signal_wires: Optional[Set[str]] = None
    
    compare_type: str = "or"
    
    # NEW: ValueRef-based operands (for IR-time construction)
    # These take precedence when set
    first_operand: Optional[ValueRef] = None  # NEW
    second_operand: Optional[ValueRef] = None  # NEW
```

**File: `dsl_compiler/src/ir/builder.py`**

```python
def decider_multi(
    self,
    conditions: List[Tuple[str, ValueRef, ValueRef]],  # [(op, left, right), ...]
    combine_type: str,  # "and" or "or"
    output_value: Union[ValueRef, int],
    output_type: str,
    source_ast: Optional[ASTNode] = None,
    copy_count_from_input: bool = False,
) -> SignalRef:
    """Create a multi-condition decider combinator.
    
    Args:
        conditions: List of (comparator, left_operand, right_operand) tuples
        combine_type: How conditions are combined ("and" or "or")
        output_value: Value to output when all conditions pass
        output_type: Signal type for output
        source_ast: Source AST for debugging
        copy_count_from_input: If True, copy signal value instead of constant
    """
    node_id = self.next_id("decider")
    decider_op = IR_Decider(node_id, output_type, source_ast)
    decider_op.output_value = output_value
    decider_op.copy_count_from_input = copy_count_from_input
    
    for i, (op, left, right) in enumerate(conditions):
        cond = DeciderCondition(
            comparator=op,
            compare_type=combine_type if i > 0 else "or",  # First condition ignored
            first_operand=left,
            second_operand=right,
        )
        decider_op.conditions.append(cond)
    
    self.add_operation(decider_op)
    return SignalRef(output_type, node_id, source_ast=source_ast)
```

### Phase 2: Lowering Layer - Detect Foldable Patterns

**CRITICAL INSIGHT:** The detection must happen BEFORE lowering sub-expressions in `lower_binary_op()`. 
Currently, when we process `(a > 0) && (b > 0)`:
1. `lower_binary_op` is called for `&&`
2. It first lowers `expr.left` (the `a > 0` comparison) → creates a decider IR node
3. It then lowers `expr.right` (the `b > 0` comparison) → creates another decider IR node  
4. Finally it calls `_lower_logical_and` with the already-created refs

By this point it's too late - we've already created the individual decider nodes!

**SOLUTION:** Add early detection in `lower_binary_op()` BEFORE lowering sub-expressions:

**File: `dsl_compiler/src/lowering/expression_lowerer.py`**

Modify `lower_binary_op` to detect foldable logical chains early:

```python
def lower_binary_op(self, expr: BinaryOp) -> ValueRef:
    """Lower binary operation to IR."""
    
    # NEW: Early detection of foldable logical chains
    # This must happen BEFORE we lower sub-expressions
    if expr.op in LOGICAL_OPS:
        folded = self._try_fold_logical_chain(expr)
        if folded is not None:
            return folded
    
    # ... rest of existing code (constant folding, then lower sub-expressions) ...
```

**New helper methods needed:**

```python
def _try_fold_logical_chain(self, expr: BinaryOp) -> Optional[SignalRef]:
    """Try to fold a logical AND/OR chain into a single multi-condition decider.
    
    Returns a SignalRef if folding succeeded, None otherwise.
    Folding succeeds when all operands in the chain are simple comparisons.
    """
    comparisons = self._collect_comparison_chain(expr, expr.op)
    if not comparisons:
        return None  # Can't fold, fall back to standard lowering
    
    # All comparisons in the chain are foldable - proceed with optimization
    combine_type = "and" if expr.op == "&&" else "or"
    return self._create_folded_decider(comparisons, combine_type, expr)

def _collect_comparison_chain(self, expr: Expr, logical_op: str) -> Optional[List[BinaryOp]]:
    """Collect all simple comparisons in a logical chain.
    
    For (a > 0) && (b > 0) && (c > 0), returns [a > 0, b > 0, c > 0].
    Returns None if ANY operand is not a simple comparison (mixed chain).
    """
    if isinstance(expr, BinaryOp) and expr.op == logical_op:
        # Recursive case: nested logical op of same type
        left_chain = self._collect_comparison_chain(expr.left, logical_op)
        right_chain = self._collect_comparison_chain(expr.right, logical_op)
        
        if left_chain is None or right_chain is None:
            return None  # Mixed chain, can't fold
        return left_chain + right_chain
    
    elif isinstance(expr, BinaryOp) and expr.op in COMPARISON_OPS:
        # Base case: simple comparison - check operands are simple
        if self._is_simple_operand(expr.left) and self._is_simple_operand(expr.right):
            return [expr]
    
    return None  # Not foldable

def _is_simple_operand(self, expr: Expr) -> bool:
    """Check if an operand is simple enough to be in a multi-condition decider.
    
    Simple operands:
    - Number literals (5, 10, -3)
    - Signal literals (("signal-A", 0))
    - Identifiers referring to signals/ints (variable references)
    - any(bundle) or all(bundle) function calls
    """
    if isinstance(expr, NumberLiteral):
        return True
    if isinstance(expr, SignalLiteral):
        return True
    if isinstance(expr, IdentifierExpr):
        # Check it refers to a Signal or int, not a complex expression
        return True  # Variables are simple
    if isinstance(expr, BundleAnyExpr) or isinstance(expr, BundleAllExpr):
        return True
    return False

def _create_folded_decider(
    self, 
    comparisons: List[BinaryOp], 
    combine_type: str,
    expr: BinaryOp
) -> SignalRef:
    """Create a multi-condition decider from a list of comparison expressions.
    
    Each comparison is a BinaryOp with op in COMPARISON_OPS.
    We lower their operands individually and build a multi-condition IR node.
    """
    result_type = self.semantic.get_expr_type(expr)
    output_type = (
        get_signal_type_name(result_type) or self.ir_builder.allocate_implicit_type()
    )
    
    conditions = []
    for comp in comparisons:
        # Lower each comparison's operands
        left_ref = self.lower_expr(comp.left)
        right_ref = self.lower_expr(comp.right)
        conditions.append((comp.op, left_ref, right_ref))
    
    return self.ir_builder.decider_multi(
        conditions,
        combine_type=combine_type,
        output_value=1,
        output_type=output_type,
        source_ast=expr,
    )
```

### Phase 3: Layout Layer - Handle Multi-Condition IR

**File: `dsl_compiler/src/layout/entity_placer.py`**

Modify `_place_decider` to handle both single-condition and multi-condition IR_Decider nodes.
The key challenge is that conditions can have either:
- `first_operand`/`second_operand` (ValueRef) from IR-time construction
- `first_signal`/`second_signal` (string) from layout-time construction

```python
def _place_decider(self, op: IR_Decider) -> None:
    """Place decider combinator."""
    
    if op.conditions:
        # Multi-condition mode - either from condition folding or SR latches
        self._place_multi_condition_decider(op)
    else:
        # Legacy single-condition mode
        self._place_single_condition_decider(op)

def _place_single_condition_decider(self, op: IR_Decider) -> None:
    """Place a single-condition decider combinator (existing logic)."""
    # ... existing _place_decider code ...

def _place_multi_condition_decider(self, op: IR_Decider) -> None:
    """Place a multi-condition decider combinator."""
    
    # Track all input operands for signal graph
    all_operands = []
    
    # Build conditions list for the placement properties
    conditions_list = []
    for cond in op.conditions:
        cond_dict = {
            'comparator': cond.comparator,
            'compare_type': cond.compare_type,
        }
        
        # Handle first operand - check ValueRef first, then string fallback
        if cond.first_operand is not None:
            # IR-time: ValueRef needs resolution
            first_op = self.signal_analyzer.get_operand_for_combinator(cond.first_operand)
            if isinstance(cond.first_operand, int):
                cond_dict['first_constant'] = cond.first_operand
            else:
                cond_dict['first_signal'] = first_op
                all_operands.append(cond.first_operand)
        elif cond.first_signal:
            # Layout-time: string already resolved
            cond_dict['first_signal'] = cond.first_signal
            if cond.first_signal_wires:
                cond_dict['first_signal_wires'] = cond.first_signal_wires
        elif cond.first_constant is not None:
            cond_dict['first_constant'] = cond.first_constant
        
        # Same for second operand...
        if cond.second_operand is not None:
            second_op = self.signal_analyzer.get_operand_for_combinator(cond.second_operand)
            if isinstance(cond.second_operand, int):
                cond_dict['second_constant'] = cond.second_operand
            else:
                cond_dict['second_signal'] = second_op
                all_operands.append(cond.second_operand)
        elif cond.second_signal:
            cond_dict['second_signal'] = cond.second_signal
            if cond.second_signal_wires:
                cond_dict['second_signal_wires'] = cond.second_signal_wires
        elif cond.second_constant is not None:
            cond_dict['second_constant'] = cond.second_constant
        
        conditions_list.append(cond_dict)
    
    # Resolve output
    usage = self.signal_usage.get(op.node_id)
    output_signal = self.signal_analyzer.resolve_signal_name(op.output_type, usage)
    output_value = self.signal_analyzer.get_operand_for_combinator(op.output_value)
    
    self.plan.create_and_add_placement(
        ir_node_id=op.node_id,
        entity_type="decider-combinator",
        position=None,
        footprint=(1, 2),
        role="decider",
        conditions=conditions_list,
        output_signal=output_signal,
        output_value=output_value,
        copy_count_from_input=op.copy_count_from_input,
    )
    
    # Signal graph: this node is source of its output
    self.signal_graph.set_source(op.node_id, op.node_id)
    
    # All operands from conditions are sinks
    for operand in all_operands:
        self._add_signal_sink(operand, op.node_id)
    
    # Output value if it's a signal
    if not isinstance(op.output_value, int):
        self._add_signal_sink(op.output_value, op.node_id)
```

### Phase 4: Emission Layer - Already Supports Multi-Condition

The `entity_emitter.py` already has `_configure_decider_multi_condition()` that handles the `conditions` property key. No changes needed here!

### Phase 5: Optimizer Updates

**File: `dsl_compiler/src/ir/optimizer.py`**

Update `CSEOptimizer._make_key()` to properly hash multi-condition deciders:

```python
def _make_key(self, op: IRNode) -> str:
    if isinstance(op, IR_Decider):
        if op.conditions:
            # Multi-condition mode
            cond_keys = []
            for c in op.conditions:
                cond_keys.append(f"{c.comparator}:{c.first_signal or c.first_constant}:"
                               f"{c.second_signal or c.second_constant}:{c.compare_type}")
            return f"decider_multi:{':'.join(cond_keys)}:{op.output_type}"
        else:
            # Legacy single-condition
            ...
```

## Detailed Implementation Steps

### Step 1: Create helper to extract comparison info

In `expression_lowerer.py`, add a method that can look at a sub-expression and determine if it's a "foldable comparison":

```python
def _is_foldable_comparison(self, expr: Expr) -> bool:
    """Check if an expression is a simple comparison that can be folded."""
    if not isinstance(expr, BinaryOp):
        return False
    if expr.op not in COMPARISON_OPS:
        return False
    # Check operands are signals or constants (not complex expressions)
    return self._is_simple_operand(expr.left) and self._is_simple_operand(expr.right)

def _is_simple_operand(self, expr: Expr) -> bool:
    """Check if an operand is a simple signal or literal."""
    from dsl_compiler.src.ast.literals import NumberLiteral, SignalLiteral
    from dsl_compiler.src.ast.expressions import Identifier, FunctionCall
    
    if isinstance(expr, (NumberLiteral, SignalLiteral, Identifier)):
        return True
    # any(bundle) and all(bundle) are also simple
    if isinstance(expr, FunctionCall) and expr.name in ('any', 'all'):
        return True
    return False
```

### Step 2: Modify logical AND lowering

```python
def _lower_logical_and(self, expr: BinaryOp, left_ref: ValueRef, right_ref: ValueRef, output_type: str) -> SignalRef:
    """Lower logical AND with multi-condition folding optimization."""
    
    # Try to fold if both operands are simple comparisons
    if self._is_foldable_comparison(expr.left) and self._is_foldable_comparison(expr.right):
        return self._fold_and_comparisons(expr, output_type)
    
    # Existing boolean optimization path
    left_is_bool = self._is_boolean_producer(left_ref)
    right_is_bool = self._is_boolean_producer(right_ref)
    
    if left_is_bool and right_is_bool:
        # Optimization: multiply for 0/1 values (1 combinator)
        result = self.ir_builder.arithmetic("*", left_ref, right_ref, output_type, expr)
        self._attach_expr_context(result.source_id, expr)
        return result
    
    # Full 3-combinator implementation
    ...
```

### Step 3: Implement the folding logic

```python
def _fold_and_comparisons(self, expr: BinaryOp, output_type: str) -> SignalRef:
    """Fold two comparisons into a single multi-condition AND decider."""
    left_comp = expr.left  # BinaryOp comparison
    right_comp = expr.right  # BinaryOp comparison
    
    # Lower the individual operands
    left_left = self.lower_expr(left_comp.left)
    left_right = self.lower_expr(left_comp.right) if not isinstance(left_comp.right, int) else left_comp.right
    right_left = self.lower_expr(right_comp.left)
    right_right = self.lower_expr(right_comp.right) if not isinstance(right_comp.right, int) else right_comp.right
    
    # Handle constant folding for number literals
    left_right_val = self._get_constant_value(left_comp.right)
    right_right_val = self._get_constant_value(right_comp.right)
    
    conditions = [
        (left_comp.op, left_left, left_right_val if left_right_val is not None else left_right, "and"),
        (right_comp.op, right_left, right_right_val if right_right_val is not None else right_right, "and"),
    ]
    
    return self.ir_builder.decider_multi(
        conditions, 
        output_value=1, 
        output_type=output_type, 
        source_ast=expr
    )
```

### Step 4: Handle chains of AND/OR

For expressions like `(a > 0) && (b > 0) && (c > 0)`, we need to recognize the pattern recursively:

```python
def _collect_and_chain(self, expr: Expr) -> List[BinaryOp]:
    """Collect all comparisons in an AND chain."""
    if not isinstance(expr, BinaryOp):
        return []
    
    if expr.op == "&&":
        left_chain = self._collect_and_chain(expr.left)
        right_chain = self._collect_and_chain(expr.right)
        
        if left_chain and right_chain:
            return left_chain + right_chain
        elif self._is_foldable_comparison(expr.left) and self._is_foldable_comparison(expr.right):
            return [expr.left, expr.right]
    
    return []
```

## Testing Strategy

### Unit Tests

1. **Test simple AND folding:**
   ```fcdsl
   Signal a = ("signal-A", 0);
   Signal b = ("signal-B", 0);
   Signal result = (a > 0) && (b > 0);
   ```
   - Verify only 1 decider is created (not 3)
   - Verify the decider has 2 conditions

2. **Test simple OR folding:**
   ```fcdsl
   Signal a = ("signal-A", 0);
   Signal b = ("signal-B", 0);
   Signal result = (a > 0) || (b > 0);
   ```
   - Verify only 1 decider is created (not 4)

3. **Test chained AND:**
   ```fcdsl
   Signal a = ("signal-A", 0);
   Signal b = ("signal-B", 0);
   Signal c = ("signal-C", 0);
   Signal result = (a > 0) && (b > 0) && (c > 0);
   ```
   - Verify only 1 decider with 3 conditions

4. **Test mixed operands (signal vs constant):**
   ```fcdsl
   Signal a = ("signal-A", 0);
   Signal result = (a > 5) && (a < 10);
   ```

5. **Test signal vs signal comparison:**
   ```fcdsl
   Signal a = ("signal-A", 0);
   Signal b = ("signal-B", 0);
   Signal result = (a > b) && (a < 100);
   ```

6. **Test non-foldable cases still work:**
   ```fcdsl
   Signal a = ("signal-A", 0);
   Signal result = a && 5;  // Not comparisons, should use != 0 approach
   ```

7. **Test mixed AND/OR (should NOT fold):**
   ```fcdsl
   Signal a = ("signal-A", 0);
   Signal b = ("signal-B", 0);
   Signal c = ("signal-C", 0);
   Signal result = (a > 0) && (b > 0) || (c > 0);
   ```
   - Should fall back to current implementation

### Integration Tests

1. Compile `07_control_flow.fcdsl` and verify combinator count reduction
2. Compile `01_comprehensive_operators.fcdsl` and verify logical operators still work
3. Create a new test file specifically for condition folding edge cases

### Blueprint Verification

Import the generated blueprints into Factorio and verify:
- Deciders have multiple condition rows visible
- Logic behaves correctly (test with known inputs)
- Output values are correct

## Edge Cases to Handle

1. **Constant comparisons that fold at compile time:**
   ```fcdsl
   Signal result = (5 > 3) && (10 < 20);  // Both true at compile time
   ```
   - Should fold to constant 1, not create decider

2. **Mixed foldable and non-foldable in chain:**
   ```fcdsl
   Signal result = (a > 0) && x && (b > 0);  // x is not a comparison
   ```
   - Should NOT fold (middle operand isn't a comparison)

3. **Comparisons with complex sub-expressions:**
   ```fcdsl
   Signal result = ((a + b) > 0) && (c > 0);
   ```
   - The first comparison's left operand is `a + b`, which needs an arithmetic combinator
   - This is still foldable, but the arithmetic needs to be separate

4. **Bundle comparisons:**
   ```fcdsl
   Bundle items = { ... };
   Signal result = any(items) > 0 && all(items) < 100;
   ```
   - Should handle any/all signals correctly

## Implementation Order

1. **Phase 1:** Add `decider_multi()` to IRBuilder and update IR_Decider to properly store ValueRefs for conditions
2. **Phase 2:** Add early detection in `lower_binary_op()` for foldable logical chains
3. **Phase 3:** Implement `_collect_comparison_chain()` helper to recursively collect comparisons
4. **Phase 4:** Implement `_is_simple_operand()` helper to check operand foldability  
5. **Phase 5:** Implement `_create_folded_decider()` to build multi-condition IR
6. **Phase 6:** Update `_place_decider()` in entity_placer to handle multi-condition IR with ValueRefs
7. **Phase 7:** Update CSE optimizer for multi-condition deciders
8. **Phase 8:** Add unit tests
9. **Phase 9:** Test with existing sample programs (especially 07_control_flow.fcdsl)
10. **Phase 10:** Create new test file for edge cases (32_condition_folding.fcdsl)

## Success Criteria

1. `(a > 0) && (b > 0)` produces 1 decider instead of 3
2. `(a > 0) || (b > 0)` produces 1 decider instead of 4
3. All existing tests pass
4. Generated blueprints work correctly in Factorio
5. Non-foldable cases fall back gracefully to existing implementation

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing behavior | Comprehensive test suite, fallback to old implementation |
| Wire color conflicts | Reuse existing wire color assignment logic |
| Signal name resolution | Reuse existing SignalAnalyzer infrastructure |
| Complex expression handling | Only fold simple comparisons, fallback otherwise |

## Files to Modify

1. `dsl_compiler/src/ir/builder.py` - Add `decider_multi()` method
2. `dsl_compiler/src/lowering/expression_lowerer.py` - Main folding logic
3. `dsl_compiler/src/layout/entity_placer.py` - Handle multi-condition IR
4. `dsl_compiler/src/ir/optimizer.py` - Update CSE for multi-condition
5. `tests/test_condition_folding.py` - New test file
6. `tests/sample_programs/32_condition_folding.fcdsl` - New sample program
