# Entity Circuit Integration - Refactoring Plan

This document provides a detailed refactoring plan for adding entity circuit integration features to support patterns like the MadZuri balanced loader.

## Executive Summary

After detailed analysis of the balanced loader requirements and the current DSL implementation, the following features are needed:

1. **Entity output access** (`.output` property) - **New feature required**
2. **Bundle combination fix** - **Bug fix required** - Current implementation doesn't create proper `IR_WireMerge`
3. **Wire color inference enhancement** - **Enhancement** - Detect and handle merge conflicts automatically
4. **`all()`/`any()` inlining for entity conditions** - **Enhancement** - Inline directly to entity circuit conditions

**What the balanced_loader_proposed.fcdsl got WRONG:**

1. **"We can't specify signal-each as the first operand"** - INCORRECT. The DSL already has full support for bundle arithmetic which uses `signal-each`. Writing `Bundle neg_avg = chest_sum / -NUM_CHESTS;` works perfectly and compiles to an arithmetic combinator with `signal-each` input/output.

2. **"The DSL's property system doesn't expose the full combinator configuration"** - PARTIALLY INCORRECT. The DSL intentionally abstracts this away. You don't need to manually configure combinators - the DSL generates them from high-level operations. Bundle arithmetic already creates the correct combinator configuration.

3. **"We need wire color control"** - INCORRECT. Wire colors can be automatically inferred by the compiler based on signal flow analysis. No user annotation is needed.

**What the balanced_loader_proposed.fcdsl got RIGHT:**

1. The DSL currently cannot read entity circuit outputs (chests, tanks, etc.). The `.output` property is genuinely needed.
2. The existing `all()` and `any()` functions DO work for bundle-wide comparisons.
3. For loops for entity placement work exactly as shown.

---

## Detailed Analysis

### Current State vs Required

| Feature | Current State | Required |
|---------|---------------|----------|
| Entity output access | Not implemented | `.output` property on entities |
| Bundle combination `{a, b}` | **BUGGY**: Returns first source only | Create proper `IR_WireMerge` |
| Wire color inference | Basic 2-coloring | Enhanced merge conflict detection |
| `all()`/`any()` inlining | Creates decider combinator | Direct inline to entity condition |

### Bundle Combination Bug (Critical)

**Location:** [expression_lowerer.py#L1007-L1027](dsl_compiler/src/lowering/expression_lowerer.py#L1007-L1027)

**Current code (lines 1019-1027):**
```python
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

**Problem:** The comment "proper wiring handles the rest" is a lie. Only the first source is tracked, so subsequent sources are never wired to anything. When you write:

```fcdsl
Bundle merged = {c1.output, c2.output, c3.output};
```

Only `c1.output` is actually connected; `c2.output` and `c3.output` are silently dropped!

**Fix:** Create a proper `IR_WireMerge` node:

```python
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
```

---

## Feature 1: Entity Output Access (`.output` property)

### Semantics

When an entity can output signals to circuit networks, the `.output` property returns a `Bundle` representing those signals.

```fcdsl
Entity chest = place("steel-chest", 0, 0, {read_contents: 1});
Bundle contents = chest.output;  # All items in chest as a bundle
```

### Implementation Steps

#### 1.1 Grammar Update

**File:** [grammar/fcdsl.lark](dsl_compiler/grammar/fcdsl.lark)

The grammar already supports property access (`entity.property`). The `.output` case will be handled specially in semantic analysis.

No grammar changes needed - `output` is treated as a property name.

#### 1.2 AST Node

**File:** [ast/expressions.py](dsl_compiler/src/ast/expressions.py)

Add a new expression type:

```python
@dataclass
class EntityOutputExpr(Expr):
    """Access to entity's circuit output (entity.output).
    
    Represents reading the circuit network signals that an entity outputs.
    For chests: all items in the chest as a bundle
    For tanks: fluid amount as a signal
    For train stops with read_from_train: train contents as a bundle
    """
    entity_name: str
    source_location: Optional[SourceLocation] = None
```

#### 1.3 Transformer Update

**File:** [parsing/transformer.py](dsl_compiler/src/parsing/transformer.py)

When transforming property access, detect `.output` and create `EntityOutputExpr`:

```python
def property_access(self, items):
    obj_name = str(items[0])
    prop_name = str(items[1])
    
    if prop_name == "output":
        return EntityOutputExpr(entity_name=obj_name, source_location=self._loc())
    
    # Existing property access handling...
```

#### 1.4 Type System Addition

**File:** [semantic/type_system.py](dsl_compiler/src/semantic/type_system.py)

Add a new type for dynamic bundles with unknown signal types:

```python
@dataclass
class DynamicBundleValue(BundleValue):
    """Bundle with runtime-determined signal types.
    
    Used for entity outputs where the actual signals depend on
    runtime state (e.g., chest contents, train cargo).
    """
    source_entity_id: str = ""
    is_dynamic: bool = True
    
    def __init__(self, source_entity_id: str = ""):
        super().__init__(signal_types=set())
        self.source_entity_id = source_entity_id
        self.is_dynamic = True
```

#### 1.5 Semantic Analysis

**File:** [semantic/analyzer.py](dsl_compiler/src/semantic/analyzer.py)

Add visitor for `EntityOutputExpr`:

```python
def visit_EntityOutputExpr(self, node: EntityOutputExpr) -> DynamicBundleValue:
    """Type check entity.output expression."""
    entity_name = node.entity_name
    
    if entity_name not in self.entity_symbols:
        self.diagnostics.error(f"Unknown entity: {entity_name}", node)
        return DynamicBundleValue(entity_name)
    
    entity_info = self.entity_symbols[entity_name]
    
    # Optional: Warn if entity might not have circuit output enabled
    # This requires tracking entity properties from placement
    
    return DynamicBundleValue(entity_name)
```

#### 1.6 IR Node

**File:** [ir/nodes.py](dsl_compiler/src/ir/nodes.py)

Add IR representation:

```python
class IR_EntityOutput(IRValue):
    """Read entity's circuit output as bundle.
    
    This represents reading all signals an entity outputs to the circuit network.
    In layout planning, this creates a virtual signal source at the entity.
    """
    
    def __init__(self, node_id: str, entity_id: str, source_ast: Optional[ASTNode] = None):
        super().__init__(node_id, "bundle", source_ast)
        self.entity_id = entity_id

    def __str__(self) -> str:
        return f"IR_EntityOutput({self.node_id}: bundle = {self.entity_id}.output)"
```

#### 1.7 Expression Lowering

**File:** [lowering/expression_lowerer.py](dsl_compiler/src/lowering/expression_lowerer.py)

Add handler and register in handlers dict:

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
    
    # Return empty signal_types since they're dynamic
    return BundleRef(set(), op_id, source_ast=expr)
```

#### 1.8 Layout Planning

**File:** [layout/entity_placer.py](dsl_compiler/src/layout/entity_placer.py)

Handle `IR_EntityOutput` - it doesn't create a new entity, just references an existing one:

```python
def _place_entity_output(self, op: IR_EntityOutput) -> None:
    """Handle entity output operations.
    
    Entity output doesn't create a new entity - it references the circuit
    output of an existing entity. We track this as a signal source.
    """
    # The entity's output is a signal source
    self.signal_graph.set_source(op.node_id, op.entity_id)
    
    # Track that this entity needs circuit connections
    placement = self.plan.get_placement(op.entity_id)
    if placement:
        placement.properties.setdefault("has_circuit_output", True)
```

---

## Feature 2: Bundle Combination Fix

### Problem

Creating bundles from computed values (not constants) doesn't properly wire them together.

### Fix Location

**File:** [lowering/expression_lowerer.py](dsl_compiler/src/lowering/expression_lowerer.py), lines 1007-1033

### Fixed Implementation

```python
def lower_bundle_literal(self, expr: BundleLiteral) -> BundleRef:
    """Lower a bundle literal { signal1, signal2, ... } to IR.

    For all-constant bundles: Creates a single IR_Const with multiple signals.
    For computed bundles: Creates IR_WireMerge to combine all sources.
    For mixed bundles: Creates IR_Const + IR_WireMerge.
    """
    constant_signals: Dict[str, int] = {}
    computed_refs: List[ValueRef] = []
    all_signal_types: set[str] = set()

    for element in expr.elements:
        element_type = self.semantic.get_expr_type(element)

        if isinstance(element_type, SignalValue):
            signal_name = (
                element_type.signal_type.name if element_type.signal_type else None
            )
            if signal_name:
                all_signal_types.add(signal_name)

                # Check if it's a constant signal literal
                if isinstance(element, SignalLiteral) and element.signal_type:
                    const_value = ConstantFolder.extract_constant_int(
                        element.value, self.diagnostics
                    )
                    if const_value is not None:
                        constant_signals[signal_name] = const_value
                        continue

                # Not a constant - lower it as a computed signal
                ref = self.lower_expr(element)
                computed_refs.append(ref)

        elif isinstance(element_type, (BundleValue, DynamicBundleValue)):
            # Nested bundle or entity output - recursively lower and merge
            nested_ref = self.lower_expr(element)
            if isinstance(nested_ref, BundleRef):
                all_signal_types.update(nested_ref.signal_types)
            computed_refs.append(nested_ref)

    # Case 1: All constants - single constant combinator
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
        
        # Create proper IR_WireMerge for multiple computed sources
        merge_ref = self.ir_builder.wire_merge(computed_refs, "bundle", expr)
        return BundleRef(all_signal_types, merge_ref.source_id, source_ast=expr)

    # Case 3: Mixed - constant combinator + wire merge of all sources
    const_ref = self.ir_builder.bundle_const(constant_signals, expr)
    all_signal_types.update(constant_signals.keys())
    
    # Merge constant combinator output with computed refs
    all_sources = [const_ref] + computed_refs
    merge_ref = self.ir_builder.wire_merge(all_sources, "bundle", expr)
    return BundleRef(all_signal_types, merge_ref.source_id, source_ast=expr)
```

### Semantic Clarification

**Bundle combination `{a, b, c}` is ALWAYS a wire merge.** This is the key insight:

- In Factorio, wiring multiple signal sources together causes their values to sum
- `{c1.output, c2.output, c3.output}` means "connect all three chest outputs to the same wire"
- This is NOT creating new signals; it's defining which signals flow together
- The resulting bundle is a virtual concept - there's no entity, just wire topology

---

## Feature 3: Wire Color Inference Enhancement

### Problem

When a signal source participates in multiple independent wire merges that both connect to the same sink, a wire color conflict occurs. Example:

```fcdsl
Bundle sum = {c1.output, c2.output, c3.output};  # Merge M1
Bundle neg_avg = sum / -3;
Bundle input1 = {neg_avg, c1.output};  # Merge M2

# At inserter: needs both neg_avg AND c1.output
# But c1.output is in BOTH M1 (via neg_avg) and M2
# If same wire color: c1 would be double-counted!
```

### Solution

Track merge membership in the signal graph and detect conflicts during wire planning.

### Implementation Location

**Files:**
- [layout/signal_graph.py](dsl_compiler/src/layout/signal_graph.py)
- [layout/wire_router.py](dsl_compiler/src/layout/wire_router.py)
- [layout/entity_placer.py](dsl_compiler/src/layout/entity_placer.py)

### Implementation Steps

#### 3.1 Track Merge Membership

In `entity_placer.py`, track which sources belong to which merges:

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

#### 3.2 Detect Conflicts During Wire Planning

In `wire_router.py`, detect when different merges containing the same source connect to the same sink:

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

---

## Feature 4: `all()`/`any()` Inlining for Entity Conditions

### Problem

Currently, `inserter.enable = all(bundle) < 1` creates a decider combinator. But Factorio entities can have `signal-everything`/`signal-anything` as their circuit condition directly.

### Solution

Detect the pattern and inline to entity condition instead of creating a decider.

### Implementation Location

**File:** [lowering/statement_lowerer.py](dsl_compiler/src/lowering/statement_lowerer.py)

### Implementation

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
    if not isinstance(expr, BinaryOp):
        return False
    if expr.op not in ('<', '<=', '>', '>=', '==', '!='):
        return False
    if not isinstance(expr.left, (BundleAnyExpr, BundleAllExpr)):
        return False
    # Right side must be a constant
    return self._is_constant(expr.right)

def _lower_inlined_bundle_condition(self, stmt: PropertyWrite) -> None:
    """Inline all(bundle) < N directly to entity condition."""
    entity_id = self.entity_refs[stmt.object_name]
    expr = stmt.value
    
    func_name = "all" if isinstance(expr.left, BundleAllExpr) else "any"
    special_signal = "signal-everything" if func_name == "all" else "signal-anything"
    
    # Lower the bundle argument to get the source connection
    bundle_arg = expr.left.bundle
    bundle_ref = self.expr_lowerer.lower_expr(bundle_arg)
    
    # Get the constant value
    constant = self._extract_constant(expr.right)
    
    # Create property write with inline bundle condition
    op = IR_EntityPropWrite(
        self.ir_builder.next_id("prop_write"),
        entity_id, "enable", stmt
    )
    op.inline_bundle_condition = {
        "signal": special_signal,
        "operator": expr.op,
        "constant": constant,
        "input_source": bundle_ref,
    }
    self.ir_builder.add_operation(op)
```

### Entity Emitter Update

**File:** [emission/entity_emitter.py](dsl_compiler/src/emission/entity_emitter.py)

Handle inline bundle conditions:

```python
if prop_type == "inline_bundle_condition":
    cond = prop_data
    signal_dict = {"name": cond["signal"], "type": "virtual"}
    
    if hasattr(entity, "circuit_enabled"):
        entity.circuit_enabled = True
    if hasattr(entity, "set_circuit_condition"):
        entity.set_circuit_condition(signal_dict, cond["operator"], cond["constant"])
    else:
        entity.control_behavior = entity.control_behavior or {}
        entity.control_behavior["circuit_condition"] = {
            "first_signal": signal_dict,
            "comparator": cond["operator"],
            "constant": cond["constant"],
        }
```

---

## Language Spec Updates Required

### Section: Type System - Bundle

Add documentation for `DynamicBundleValue`:

```markdown
#### Dynamic Bundles

When reading from entity outputs, the bundle has **dynamic signal types** that are determined at runtime:

```fcdsl
Entity chest = place("steel-chest", 0, 0, {read_contents: 1});
Bundle contents = chest.output;  # Dynamic - depends on chest contents
```

Dynamic bundles have these properties:
- Signal types are unknown at compile time
- Can be combined with other bundles via `{a, b}`
- Can have arithmetic applied via `bundle * n`
- Can be checked via `all(bundle) < n` or `any(bundle) > n`
```

### Section: Entity System

Add documentation for `.output` property:

```markdown
### Entity Circuit Outputs

Entities that can output to circuit networks have an `.output` property:

```fcdsl
Entity chest = place("steel-chest", 0, 0, {read_contents: 1});
Bundle contents = chest.output;
```

**Entities with circuit outputs:**

| Entity Type | Output Type | Enable Property | Description |
|-------------|-------------|-----------------|-------------|
| Container (chests) | Bundle | `read_contents: 1` | All items in chest |
| Tank | Signal | `read_contents: 1` | Fluid amount |
| Train Stop | Bundle | `read_from_train: 1` | Train contents |
| Accumulator | Signal | (always) | Charge percentage |
| Roboport | Bundle | (various) | Robot/logistics stats |

**Example: Balanced Loader**

```fcdsl
# Read chest contents
Entity c1 = place("steel-chest", 0, 0, {read_contents: 1});
Entity c2 = place("steel-chest", 1, 0, {read_contents: 1});

# Combine outputs (wire merge - signals sum)
Bundle total = {c1.output, c2.output};

# Calculate negative average
Bundle neg_avg = total / -2;

# Enable inserter when below average
Entity ins = place("inserter", 0, 1, {circuit_enabled: 1});
Bundle input = {neg_avg, c1.output};
ins.enable = all(input) < 1;
```
```

### Section: Bundle Operations

Clarify bundle combination semantics:

```markdown
#### Bundle Combination (Wire Merge)

Combining bundles with `{a, b, c}` creates a **wire merge**:

```fcdsl
Bundle combined = {bundle1, bundle2, bundle3};
```

This means:
- All sources are connected to the **same wire**
- Signals with the **same type automatically sum** (Factorio wire behavior)
- No combinator is created - just wire topology

**Wire Merge Example:**

```fcdsl
Signal iron1 = ("iron-plate", 100);
Signal iron2 = ("iron-plate", 50);
Bundle merged = {iron1, iron2};
# On the wire: iron-plate = 150 (summed automatically)
```

**Bundle Flattening:**

Bundles can contain other bundles:

```fcdsl
Bundle a = { ("signal-A", 10), ("signal-B", 20) };
Bundle b = { ("signal-C", 30) };
Bundle all = { a, b };  # Contains signal-A, signal-B, signal-C
```
```

---

## Documentation Issues Causing Confusion

The balanced loader proposal revealed some documentation gaps:

### 1. Bundle Arithmetic Is Already Supported

The docs explain bundle arithmetic but don't emphasize that it works with `signal-each` automatically:

**Add to LANGUAGE_SPEC.md Section 4 (Type System - Bundle):**

```markdown
**Important:** Bundle arithmetic uses Factorio's `signal-each` automatically. You don't need to configure combinator internals - the DSL handles this:

```fcdsl
Bundle resources = { ("iron-plate", 100), ("copper-plate", 80) };
Bundle doubled = resources * 2;  
# Compiles to: Each * 2 → Each
```
```

### 2. Wire Color Is Automatic

Users don't need to specify wire colors. The compiler infers them.

**Add to LANGUAGE_SPEC.md Section 11 (Circuit Network Integration):**

```markdown
**Wire Color Inference:**

The compiler automatically assigns wire colors to avoid signal conflicts. You never need to specify colors manually.

When the same signal source is used in multiple ways (e.g., summed with others AND used individually), the compiler detects this and assigns different colors.
```

### 3. Entity Circuit Properties

The ENTITY_REFERENCE_DSL.md documents `read_contents` but doesn't explain how to read the resulting signals. The `.output` property fills this gap.

---

## Test Plan

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
    Signal a = ("signal-A", 10);
    Signal b = ("signal-B", 20);
    Bundle sum = {a, b};
    '''
    ops = compile_to_ir(code)
    merges = [op for op in ops if isinstance(op, IR_WireMerge)]
    assert len(merges) == 1
    assert len(merges[0].sources) == 2

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

Add new sample program:

**File:** `tests/sample_programs/31_balanced_loader.fcdsl`

```fcdsl
# Balanced Loader - MadZuri Pattern
# Tests entity output access, bundle combination, and all() inlining

int NUM_CHESTS = 6;

# Place chests with circuit output enabled
Entity c1 = place("steel-chest", 0, 0, {read_contents: 1});
Entity c2 = place("steel-chest", 1, 0, {read_contents: 1});
Entity c3 = place("steel-chest", 2, 0, {read_contents: 1});
Entity c4 = place("steel-chest", 3, 0, {read_contents: 1});
Entity c5 = place("steel-chest", 4, 0, {read_contents: 1});
Entity c6 = place("steel-chest", 5, 0, {read_contents: 1});

# Sum all chest outputs (wire merge)
Bundle chest_sum = {c1.output, c2.output, c3.output, c4.output, c5.output, c6.output};

# Calculate negative average
Bundle neg_avg = chest_sum / -NUM_CHESTS;

# Place inserters
Entity i1 = place("fast-inserter", 0, 1, {circuit_enabled: 1, direction: 0});
Entity i2 = place("fast-inserter", 1, 1, {circuit_enabled: 1, direction: 0});
Entity i3 = place("fast-inserter", 2, 1, {circuit_enabled: 1, direction: 0});
Entity i4 = place("fast-inserter", 3, 1, {circuit_enabled: 1, direction: 0});
Entity i5 = place("fast-inserter", 4, 1, {circuit_enabled: 1, direction: 0});
Entity i6 = place("fast-inserter", 5, 1, {circuit_enabled: 1, direction: 0});

# Each inserter: negative average + individual chest
Bundle input1 = {neg_avg, c1.output};
Bundle input2 = {neg_avg, c2.output};
Bundle input3 = {neg_avg, c3.output};
Bundle input4 = {neg_avg, c4.output};
Bundle input5 = {neg_avg, c5.output};
Bundle input6 = {neg_avg, c6.output};

# Enable when ALL signals < 1 (below average)
i1.enable = all(input1) < 1;
i2.enable = all(input2) < 1;
i3.enable = all(input3) < 1;
i4.enable = all(input4) < 1;
i5.enable = all(input5) < 1;
i6.enable = all(input6) < 1;

# Supply belts
for x in 0..6 {
    Entity belt = place("express-transport-belt", x, 2, {direction: 0});
}
```

---

## Implementation Order

1. **Bundle Combination Fix** (Critical bug fix)
   - Fix `lower_bundle_literal` to create proper `IR_WireMerge`
   - Add tests for multi-source bundle combination
   
2. **Entity Output Access**
   - Add `EntityOutputExpr` AST node
   - Add `DynamicBundleValue` type
   - Add `IR_EntityOutput` IR node
   - Implement lowering
   - Implement layout planning
   - Add tests

3. **Wire Color Inference**
   - Add merge membership tracking
   - Implement conflict detection
   - Add tests for color assignment

4. **`all()`/`any()` Inlining**
   - Detect pattern in statement lowering
   - Implement inline to entity condition
   - Add tests

5. **Documentation Updates**
   - Update LANGUAGE_SPEC.md
   - Update ENTITY_REFERENCE_DSL.md
   - Add balanced loader to sample programs

---

## Summary of Changes

| Feature | Type | Files Changed |
|---------|------|---------------|
| Bundle combination fix | Bug fix | expression_lowerer.py |
| `.output` property | New feature | expressions.py, type_system.py, analyzer.py, nodes.py, expression_lowerer.py, entity_placer.py |
| Wire color inference | Enhancement | entity_placer.py, wire_router.py |
| `all()`/`any()` inlining | Enhancement | statement_lowerer.py, entity_emitter.py |
| Documentation | Update | LANGUAGE_SPEC.md, ENTITY_REFERENCE_DSL.md |
