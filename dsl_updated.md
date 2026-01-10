# DSL Feature Additions: Entity Circuit Integration

## Executive Summary

This document proposes minimal extensions to support circuits like the MadZuri balanced loader. After careful analysis:

1. **Entity output access** (`.output` property) - New feature needed
2. **Bundle combination** (`{bundle1, bundle2}`) - Already in syntax, needs proper IR implementation  
3. **Wire color assignment** - Can be automatically inferred, NO user annotation needed

**NO new keywords needed** - The DSL already has `all()` and `any()` functions that compile to `signal-everything` and `signal-anything`. These work correctly with the existing `entity.enable = expression` pattern.

**NO `.input` property needed** - The DSL models signal flow through expressions. When you write `entity.enable = all(bundle_expression) < 1`, the signals in that expression automatically connect to the entity. This is consistent with the design philosophy: *"Signal-Centric: Everything is a signal flowing through circuit networks"*.

---

## Analysis: How Existing DSL Features Apply

### Entity Conditions with `all()`

The balanced loader needs each inserter to check: "are ALL signals (items) below average?"

This is exactly what `all()` does:

```fcdsl
# From doc/03_signals_and_types.md:
# all() compiles to a decider combinator using signal-everything

Bundle combined = {neg_avg, chest.output};
inserter.enable = all(combined) < 1;  # All signals must be < 1
```

But wait - this creates a decider combinator, not an inline entity condition. Let me check if that's a problem...

Actually, examining the inline comparison optimization (LANGUAGE_SPEC.md lines 1079-1093):

> For simple comparisons, the compiler **inlines** them into the entity's circuit condition:
> `lamp.enable = count > 10;`
> Instead of creating a separate decider combinator, the compiler configures the lamp's built-in circuit condition.

The key insight: Factorio entities CAN have `signal-everything` or `signal-anything` as their first_signal in circuit conditions directly! So we can extend the inline optimization to recognize `all(bundle) OP const` and set:
- first_signal = "signal-everything"
- comparator = OP  
- constant = const

This requires NO new syntax - just improved lowering of the existing `all()` pattern.

---

## Wire Color Analysis

### The Balanced Loader Pattern

```
Chests:    c1  c2  c3  c4  c5  c6
            \  |   |   |   |   /
             [RED: wire merge sum]
                    |
            [Arithmetic: Each / -6]
                    |
             [RED: to all inserters]
                    
Individual:  c1→i1  c2→i2  c3→i3  (GREEN: isolated)
```

Each inserter needs:
- The **negative average** (shared, from combinator output)  
- Its **individual chest** contents (isolated)

Both use the same signal types (iron-plate, etc.), so they MUST be on different wire colors.

### Why the Compiler Can Infer This

The key insight: **a signal source that participates in multiple distinct wire merges creates a conflict when those merges both connect to the same sink.**

Detection algorithm:
1. `c1.output` is used in merge M1: `{c1.output, c2.output, ...}` (chest_sum)
2. `c1.output` is ALSO used in merge M2: `{neg_avg, c1.output}` (inserter1's input)
3. M1 flows through combinator to become `neg_avg`, which is part of M2
4. At the inserter: signals from M2 arrive
5. BUT M2 contains both `neg_avg` (derived from M1) and `c1.output` (also in M1)
6. If same color: `c1.output` would connect to chest network AND individually - wrong!
7. **Conflict detected** → assign different colors

### What Enables Automatic Inference

The DSL syntax already expresses:
1. These sources are MERGED (bundle combination `{a, b}`)
2. Where signals flow (through expressions to entity properties)

With proper `.output` semantics and correct bundle combination lowering, the compiler has enough information.

---

## Feature 1: Entity Output Access (`.output` property)

### Syntax

```fcdsl
Entity chest = place("steel-chest", 0, 0, {read_contents: 1});
Bundle contents = chest.output;  # Read entity's circuit network output
```

### Semantics

- Available on entities with circuit output (when `read_contents`, `read_hand_contents`, etc. is enabled)
- Returns a `Bundle` representing what the entity outputs on the circuit network
- Signal types are dynamic (unknown at compile time, depends on chest contents at runtime)

### Type System

```python
class DynamicBundleValue(BundleValue):
    """Bundle with runtime-determined signal types."""
    
    def __init__(self, source_entity_id: str):
        super().__init__(signal_types=set())  # Unknown at compile time
        self.source_entity_id = source_entity_id
        self.is_dynamic = True
```

### IR Representation

```python
class IR_EntityOutput(IRValue):
    """Read entity's circuit output as bundle."""
    
    def __init__(self, node_id: str, entity_id: str, source_ast=None):
        super().__init__(node_id, "bundle", source_ast)
        self.entity_id = entity_id
```

---

## Feature 2: Bundle Combination Fix

### Current State

The syntax `{bundle1, bundle2}` already exists and is documented in LANGUAGE_SPEC.md:

> **Bundle Flattening:** Bundles can contain other bundles, which are automatically flattened

However, the current implementation in `lower_bundle_literal()` does NOT properly create `IR_WireMerge` for computed bundles. It just returns the first source:

```python
# Current code (expression_lowerer.py lines 1019-1027):
# Multiple computed signals - they'll be on the same wire
# Create a wire merge of all computed signals
source_ids = set()
for ref in computed_refs:
    if isinstance(ref, (SignalRef, BundleRef)):
        source_ids.add(ref.source_id)
# Return a BundleRef pointing to the merged sources
# For simplicity, use the first source - proper wiring handles the rest  # <-- BUG!
first_source = computed_refs[0]
if isinstance(first_source, (SignalRef, BundleRef)):
    return BundleRef(all_signal_types, first_source.source_id, source_ast=expr)
```

### Required Fix

Create proper `IR_WireMerge` when combining computed bundles:

```python
def lower_bundle_literal(self, expr: BundleLiteral) -> BundleRef:
    constant_signals: Dict[str, int] = {}
    computed_refs: List[ValueRef] = []
    all_signal_types: set[str] = set()

    for element in expr.elements:
        element_type = self.semantic.get_expr_type(element)

        if isinstance(element_type, SignalValue):
            # ... existing constant handling ...
            ref = self.lower_expr(element)
            computed_refs.append(ref)

        elif isinstance(element_type, (BundleValue, DynamicBundleValue)):
            nested_ref = self.lower_expr(element)
            if isinstance(nested_ref, BundleRef):
                all_signal_types.update(nested_ref.signal_types)
            computed_refs.append(nested_ref)

    # Case 1: All constants
    if constant_signals and not computed_refs:
        return self.ir_builder.bundle_const(constant_signals, expr)

    # Case 2: All computed - proper wire merge
    if computed_refs and not constant_signals:
        if len(computed_refs) == 1:
            ref = computed_refs[0]
            if isinstance(ref, BundleRef):
                return ref
            if isinstance(ref, SignalRef):
                return BundleRef({ref.signal_type}, ref.source_id, source_ast=expr)
        
        # FIX: Create proper IR_WireMerge for multiple computed sources
        merge_ref = self.ir_builder.wire_merge(computed_refs, "bundle", expr)
        return BundleRef(all_signal_types, merge_ref.source_id, source_ast=expr)

    # Case 3: Mixed
    const_ref = self.ir_builder.bundle_const(constant_signals, expr)
    all_sources = [const_ref] + computed_refs
    merge_ref = self.ir_builder.wire_merge(all_sources, "bundle", expr)
    return BundleRef(all_signal_types, merge_ref.source_id, source_ast=expr)
```

### Semantics

When you write:
```fcdsl
Bundle merged = {chest1.output, chest2.output, chest3.output};
```

This means:
- All three entity outputs are connected to the same wire
- Signals with the same type automatically sum (Factorio wire behavior)
- Creates `IR_WireMerge` in IR (wire junction, no combinator created)

---

## Feature 3: Automatic Wire Color Inference

### No New Syntax Needed

The existing wire color algorithm in `wire_router.py` already handles 2-coloring. We enhance the signal graph to track **merge membership**.

### Enhanced Signal Graph

Track which sources belong to which wire merges:

```python
@dataclass
class SignalPath:
    """Represents a signal's path to a sink."""
    source_entity_id: str
    sink_entity_id: str
    via_merge: Optional[str] = None  # merge_node_id if through merge, None if direct
    signal_types: Set[str] = field(default_factory=set)
```

### Conflict Detection

Two signal paths conflict if:
1. They share a common upstream source entity
2. They go to the same sink entity  
3. They are in DIFFERENT merges (or one is in a merge, one isn't)
4. They share signal types (or either has unknown/dynamic types)

```python
def detect_wire_conflicts(self, paths: List[SignalPath]) -> List[Tuple[SignalPath, SignalPath]]:
    """Find pairs of paths that need different wire colors."""
    conflicts = []
    
    # Group paths by sink
    by_sink = defaultdict(list)
    for path in paths:
        by_sink[path.sink_entity_id].append(path)
    
    for sink_id, sink_paths in by_sink.items():
        for i, path_a in enumerate(sink_paths):
            for path_b in sink_paths[i+1:]:
                # Different merges (or merge vs direct) = potential conflict
                if path_a.via_merge != path_b.via_merge:
                    # Check for shared upstream source
                    if self._share_upstream_source(path_a, path_b):
                        conflicts.append((path_a, path_b))
    
    return conflicts
```

### Color Assignment

When conflicts are detected, the existing `plan_wire_colors()` graph coloring handles it:
- One merge's connections get one color (e.g., RED)
- Other merge's connections get the other color (e.g., GREEN)

---

## Feature 4: Enhanced `all()` Inlining for Entity Conditions

### Current Behavior

Currently, `all(bundle) < 1` creates a decider combinator with `signal-everything`.

### Enhanced Behavior

When used directly in an entity enable condition:
```fcdsl
inserter.enable = all(combined_bundle) < 1;
```

The compiler can inline this directly to the entity's circuit condition:
- first_signal = "signal-everything"
- comparator = "<"
- constant = 1

This avoids creating a decider combinator for simple cases.

### Implementation

In statement lowering, detect the pattern:

```python
def lower_property_write(self, stmt: PropertyWrite) -> None:
    entity_name = stmt.object_name
    prop_name = stmt.property_name
    
    if prop_name == "enable":
        # Check for all(bundle) OP const or any(bundle) OP const
        if self._is_inlinable_bundle_condition(stmt.value):
            self._lower_inlined_bundle_condition(stmt)
            return
    
    # ... existing logic

def _is_inlinable_bundle_condition(self, expr: Expr) -> bool:
    """Check if expression is all(bundle) OP const or any(bundle) OP const."""
    if isinstance(expr, BinaryOp) and expr.op in ('<', '<=', '>', '>=', '==', '!='):
        if isinstance(expr.left, FunctionCall):
            if expr.left.name in ('all', 'any'):
                if self._is_constant(expr.right):
                    return True
    return False

def _lower_inlined_bundle_condition(self, stmt: PropertyWrite) -> None:
    """Inline all(bundle) < N directly to entity condition."""
    entity_id = self.entity_refs[stmt.object_name]
    expr = stmt.value
    
    func_name = expr.left.name  # 'all' or 'any'
    special_signal = "signal-everything" if func_name == "all" else "signal-anything"
    operator = expr.op
    constant = self._extract_constant(expr.right)
    
    # Lower the bundle argument to connect it to the entity
    bundle_arg = expr.left.args[0]
    bundle_ref = self.lower_expr(bundle_arg)
    
    # Create entity property write with inline condition
    op = IR_EntityPropWrite(
        self.ir_builder.next_id("prop_write"),
        entity_id, "enable", stmt
    )
    op.inline_condition = InlineCondition(
        signal=special_signal,
        operator=operator, 
        constant=constant,
        input_source=bundle_ref
    )
    self.ir_builder.add_operation(op)
```

---

## Implementation Guide

### Phase 1: Grammar Updates

Add to `fcdsl.lark`:

```lark
// Property access can include .output
property_access: IDENTIFIER "." IDENTIFIER
```

### Phase 2: AST Nodes

Add to `ast/expressions.py`:

```python
@dataclass
class EntityOutputExpr(Expr):
    """Access to entity's circuit output (entity.output)."""
    entity_name: str
    source_location: Optional[SourceLocation] = None
```

### Phase 3: Semantic Analysis

```python
def visit_EntityOutputExpr(self, node: EntityOutputExpr) -> DynamicBundleValue:
    """Type check entity.output expression."""
    entity_name = node.entity_name
    
    if entity_name not in self.entity_symbols:
        self.diagnostics.error(f"Unknown entity: {entity_name}", ...)
        return DynamicBundleValue(entity_name)
    
    entity_info = self.entity_symbols[entity_name]
    if not entity_info.has_circuit_output:
        self.diagnostics.warning(
            f"Entity '{entity_name}' may not have circuit output enabled", ...)
    
    return DynamicBundleValue(entity_name)
```

### Phase 4: Expression Lowering

Add `lower_entity_output`:

```python
def lower_entity_output(self, expr: EntityOutputExpr) -> BundleRef:
    """Lower entity.output to IR_EntityOutput."""
    entity_id = self.parent.entity_refs.get(expr.entity_name)
    if not entity_id:
        self._error(f"Unknown entity: {expr.entity_name}", expr)
        return BundleRef(set(), "error", source_ast=expr)
    
    op_id = self.ir_builder.next_id("entity_out")
    op = IR_EntityOutput(op_id, entity_id, expr)
    self.ir_builder.add_operation(op)
    
    return BundleRef(set(), op_id, source_ast=expr)
```

Fix `lower_bundle_literal()` (as shown in Feature 2 above).

### Phase 5: Layout - Wire Merge Tracking

In `entity_placer.py`, track merge membership:

```python
def _place_wire_merge(self, op: IR_WireMerge) -> None:
    """Handle wire merge operations with membership tracking."""
    self._wire_merge_junctions[op.node_id] = {
        "inputs": list(op.sources),
        "output_id": op.node_id,
    }
    
    # Track which sources belong to this merge
    for source in op.sources:
        if isinstance(source, (SignalRef, BundleRef)):
            source_id = source.source_id
            if source_id not in self._merge_membership:
                self._merge_membership[source_id] = set()
            self._merge_membership[source_id].add(op.node_id)
    
    self.signal_graph.set_source(op.node_id, op.node_id)
    for input_sig in op.sources:
        self._add_signal_sink(input_sig, op.node_id)
```

### Phase 6: Connection Planning - Conflict Detection

Enhance `connection_planner.py` to detect merge conflicts:

```python
def _detect_color_conflicts(self) -> Dict[Tuple[str, str], str]:
    """Detect paths that need locked colors due to merge conflicts."""
    locked_colors = {}
    
    # For each source that's in multiple merges
    for source_id, merge_ids in self._merge_membership.items():
        if len(merge_ids) <= 1:
            continue
            
        # Find sinks that receive from multiple merges containing this source
        for merge_a, merge_b in combinations(merge_ids, 2):
            sinks_a = self._find_merge_sinks(merge_a)
            sinks_b = self._find_merge_sinks(merge_b)
            
            # If both merges connect to same sink, need different colors
            common_sinks = sinks_a & sinks_b
            for sink in common_sinks:
                locked_colors[(merge_a, sink)] = "red"
                locked_colors[(merge_b, sink)] = "green"
    
    return locked_colors
```

### Phase 7: Blueprint Emission

Handle inline conditions:

```python
def emit_circuit_condition(self, entity: BlueprintEntity, 
                           prop_write: IR_EntityPropWrite) -> None:
    if prop_write.inline_condition:
        cond = prop_write.inline_condition
        entity.set_circuit_condition(
            first_signal=cond.signal,
            comparator=cond.operator,
            constant=cond.constant
        )
    else:
        # Normal condition handling
        pass
```

---

## Complete Balanced Loader Example

```fcdsl
# Balanced Loader - MadZuri Pattern
# Distributes items evenly across 6 chests
# Uses ONLY existing DSL features + .output

int NUM_CHESTS = 6;

# Place chests with circuit output enabled
Entity c1 = place("steel-chest", 0, 0, {read_contents: 1});
Entity c2 = place("steel-chest", 1, 0, {read_contents: 1});
Entity c3 = place("steel-chest", 2, 0, {read_contents: 1});
Entity c4 = place("steel-chest", 3, 0, {read_contents: 1});
Entity c5 = place("steel-chest", 4, 0, {read_contents: 1});
Entity c6 = place("steel-chest", 5, 0, {read_contents: 1});

# Combine all chest outputs (creates wire merge - signals sum)
Bundle chest_sum = {c1.output, c2.output, c3.output, c4.output, c5.output, c6.output};

# Calculate negative average: Each / -NUM_CHESTS
# Creates ONE arithmetic combinator
Bundle neg_avg = chest_sum / -NUM_CHESTS;

# Place inserters
Entity i1 = place("fast-inserter", 0, 1, {circuit_enabled: 1, direction: 0});
Entity i2 = place("fast-inserter", 1, 1, {circuit_enabled: 1, direction: 0});
Entity i3 = place("fast-inserter", 2, 1, {circuit_enabled: 1, direction: 0});
Entity i4 = place("fast-inserter", 3, 1, {circuit_enabled: 1, direction: 0});
Entity i5 = place("fast-inserter", 4, 1, {circuit_enabled: 1, direction: 0});
Entity i6 = place("fast-inserter", 5, 1, {circuit_enabled: 1, direction: 0});

# Each inserter gets negative average + its individual chest contents
# The all() check enables when ALL item types are below average
Bundle input1 = {neg_avg, c1.output};
Bundle input2 = {neg_avg, c2.output};
Bundle input3 = {neg_avg, c3.output};
Bundle input4 = {neg_avg, c4.output};
Bundle input5 = {neg_avg, c5.output};
Bundle input6 = {neg_avg, c6.output};

# Enable inserters when ALL signals in combined bundle are < 1
# (meaning: individual chest - average < 1, i.e., below average)
i1.enable = all(input1) < 1;
i2.enable = all(input2) < 1;
i3.enable = all(input3) < 1;
i4.enable = all(input4) < 1;
i5.enable = all(input5) < 1;
i6.enable = all(input6) < 1;

# Place supply belts
for x in 0..6 {
    Entity belt = place("express-transport-belt", x, 2, {direction: 0});
}
```

### How the Compiler Processes This

1. **Entity output**: `c1.output` creates `IR_EntityOutput` referencing chest's circuit output

2. **Bundle combination**: `{c1.output, c2.output, ...}` creates `IR_WireMerge`
   - Marks all chest outputs as "in merge M1"

3. **Arithmetic**: `chest_sum / -6` creates arithmetic combinator
   - Input: wire merge M1
   - Output: neg_avg bundle

4. **Second bundle combinations**: `{neg_avg, c1.output}` creates another `IR_WireMerge`
   - Marks neg_avg and c1.output as "in merge M2"
   - Note: c1.output is now in BOTH M1 and M2

5. **Enable condition**: `all(input1) < 1`
   - input1 bundle connects to inserter1
   - Condition is inlined: first_signal="signal-everything", operator="<", constant=1

6. **Wire color conflict detection**:
   - c1.output is in merge M1 (all chests) AND merge M2 (inserter1 input)
   - Both merges ultimately connect to inserter1
   - **Conflict detected** → M1 connections use RED, M2 uses GREEN

### Generated Blueprint

**Entities:**
- 6 steel chests with `read_contents: true`
- 1 arithmetic combinator: `Each / -6 → Each`  
- 6 fast inserters with condition `Everything < 1`
- 6 transport belts

**Wire Connections (automatically determined):**
- RED: All chests → arithmetic combinator input (merged as chest_sum)
- RED: Arithmetic combinator output → all inserters (via neg_avg in input bundles)
- GREEN: c1→i1, c2→i2, c3→i3, c4→i4, c5→i5, c6→i6 (individual connections in input bundles)

---

## Summary of Changes

| Feature | Type | Description |
|---------|------|-------------|
| `.output` property | New | Read entity circuit output as Bundle |
| Bundle combination | Fix | Properly create `IR_WireMerge` for `{a, b, ...}` |
| Wire color inference | Enhancement | Auto-detect conflicts from merge membership |
| `all()`/`any()` inlining | Enhancement | Inline bundle conditions to entity circuit condition |

### What Users Write

```fcdsl
Bundle merged = {chest1.output, chest2.output};  # Merge outputs
Bundle processed = merged / -2;                   # Combinator
Bundle combined = {processed, chest1.output};     # Another merge
inserter.enable = all(combined) < 1;              # Uses existing all() function
```

### What Compiler Does

1. Creates wire merge M1 for `{chest1.output, chest2.output}`
2. Creates arithmetic combinator for `merged / -2`
3. Creates wire merge M2 for `{processed, chest1.output}`
4. Detects: chest1.output is in M1 AND M2, both reach inserter
5. **Auto-assigns**: M1 connections → RED, M2 connections → GREEN
6. Inlines `all(...) < 1` to inserter's circuit condition

---

## Testing Strategy

### Unit Tests

```python
def test_entity_output_creates_ir():
    """entity.output creates IR_EntityOutput."""
    code = 'Entity c = place("steel-chest", 0, 0, {read_contents: 1}); Bundle b = c.output;'
    ops = compile_to_ir(code)
    assert any(isinstance(op, IR_EntityOutput) for op in ops)

def test_bundle_combination_creates_wire_merge():
    """Bundle combination creates IR_WireMerge."""
    code = '''
    Entity c1 = place("steel-chest", 0, 0, {read_contents: 1});
    Entity c2 = place("steel-chest", 1, 0, {read_contents: 1});
    Bundle sum = {c1.output, c2.output};
    '''
    ops = compile_to_ir(code)
    merges = [op for op in ops if isinstance(op, IR_WireMerge)]
    assert len(merges) == 1
    assert len(merges[0].sources) == 2

def test_wire_color_conflict_detection():
    """Detect merge conflict and assign different colors."""
    code = '''
    Entity c1 = place("steel-chest", 0, 0, {read_contents: 1});
    Entity c2 = place("steel-chest", 1, 0, {read_contents: 1});
    Bundle sum = {c1.output, c2.output};
    Bundle neg = sum / -2;
    Entity i1 = place("inserter", 0, 1, {circuit_enabled: 1});
    Bundle input = {neg, c1.output};
    i1.enable = all(input) > 0;
    '''
    blueprint = compile_to_blueprint(code)
    # Verify c1 connections use two different colors
    connections = get_connections(blueprint, "i1")
    colors = {c.color for c in connections}
    assert colors == {"red", "green"}

def test_all_condition_inlined():
    """all(bundle) < N is inlined to entity condition."""
    code = '''
    Bundle b = { ("signal-A", 10), ("signal-B", 20) };
    Entity i = place("inserter", 0, 0, {circuit_enabled: 1});
    i.enable = all(b) < 50;
    '''
    blueprint = compile_to_blueprint(code)
    inserter = get_entity(blueprint, "i")
    assert inserter.circuit_condition.first_signal == "signal-everything"
    assert inserter.circuit_condition.comparator == "<"
    assert inserter.circuit_condition.constant == 50
```

### Integration Test

```python
def test_balanced_loader():
    """Full balanced loader compiles with correct wire colors."""
    source = Path("tests/sample_programs/30_balanced_loader.fcdsl").read_text()
    blueprint = compile_source(source)
    
    # Should have exactly 1 arithmetic combinator
    assert count_entities(blueprint, "arithmetic-combinator") == 1
    
    # All inserters should have Everything < 1 condition
    inserters = get_entities(blueprint, "fast-inserter")
    for ins in inserters:
        assert ins.circuit_condition.first_signal == "signal-everything"
    
    # Each inserter should have both RED and GREEN connections
    for ins in inserters:
        colors = {c.color for c in ins.connections}
        assert "red" in colors and "green" in colors
```

---

## Backward Compatibility

All changes are backward compatible:
- `.output` is a new property (doesn't conflict with any existing property)
- Bundle combination fix improves existing syntax behavior
- Wire color inference is automatic (no syntax change needed)
- `all()`/`any()` inlining is an optimization (same semantics)
