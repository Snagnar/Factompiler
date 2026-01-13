# IR (Intermediate Representation) Module

## Purpose
Defines the intermediate representation that sits between the AST and the final blueprint. IR is a low-level, combinator-centric representation that's easier to optimize and lay out than the AST.

## Files

### nodes.py
**IR Node Definitions**
Core IR data structures representing Factorio combinators and operations:

**Value References**:
- `SignalRef`: Reference to a signal value produced by an operation
  - `signal_type`: The Factorio signal type ("iron-plate", "signal-A", etc.)
  - `source_id`: ID of the operation that produces this signal
- `BundleRef`: Reference to multiple signals (bundle of signals on one wire)
  - `signal_types`: Set of signal types in the bundle
  - `source_id`: ID of the operation producing the bundle
- `ValueRef` = `SignalRef | BundleRef | int` (union type)

**Base Classes**:
- `IRNode`: Base for all IR nodes, tracks `node_id` and source AST
- `IRValue`: Nodes that produce values (have `output_type`)
- `IREffect`: Nodes that have side effects (no output value)

**Value-Producing Operations** (extend `IRValue`):
- `IRConst`: Constant combinator (produces fixed signal values)
- `IRArith`: Arithmetic combinator (`+`, `-`, `*`, `/`, `%`, `^`, etc.)
- `IRDecider`: Decider combinator (conditionals: `==`, `<`, `>`, etc.)
- `IRWireMerge`: Merge multiple signals onto one wire (no physical combinator)
- `IRMemRead`: Read from memory cell
- `IREntityPropRead`: Read entity property (e.g., accumulator charge)
- `IREntityOutput`: Read entity's circuit output (e.g., chest contents)

**Effect Operations** (extend `IREffect`):
- `IRMemCreate`: Create memory cell
- `IRMemWrite`: Write to memory cell
- `IRLatchWrite`: Write to memory with SR-latch behavior
- `IRPlaceEntity`: Place a Factorio entity
- `IREntityPropWrite`: Write to entity property (e.g., lamp color)

### builder.py
**IR Builder - Construction Interface**
- `IRBuilder`: Fluent API for creating IR operations
  - Tracks all operations in order
  - Assigns unique node IDs
  - Manages signal type map (signal name → type metadata)
  - Provides helper methods for common patterns

**Key Methods**:
- `const(output_type, value, ast)`: Create constant
- `arithmetic(op, left, right, output_type, ast)`: Create arithmetic operation
- `decider(op, left, right, output, output_type, ast)`: Create decider
- `wire_merge(sources, output_type, ast)`: Merge signals
- `get_operation(node_id)`: Retrieve operation by ID
- `allocate_implicit_type()`: Generate unique signal type for intermediate values

**Signal Type Management**:
- Tracks which signal types are used
- Validates signal types against Factorio's signal registry
- Handles implicit types for unnamed intermediate values

### optimizer.py
**IR Optimizations**
Post-construction optimizations on IR:

- `IROptimizer`: Main optimizer class
  - `optimize(operations)`: Apply all optimization passes

**Optimization Passes**:
1. **Common Subexpression Elimination (CSE)**:
   - Find duplicate computations
   - Reuse existing results instead of recomputing
   - Example: `a + b` used twice → compute once, reference twice

2. **Constant Propagation**:
   - Evaluate constant expressions at compile time
   - Fold arithmetic on constants
   - Remove dead constant operations

3. **Dead Code Elimination**:
   - Remove operations whose results are never used
   - Track usage through references

4. **Wire Merge Optimization**:
   - Combine multiple wire merges
   - Eliminate redundant merges

## Information Flow

```
AST (from semantic analysis)
    ↓
[lowering/lowerer.py] - Walk AST, build IR
    ↓
├─ [builder.py] - Construct IR operations
│      ↓
│   IR operations list
│      ↓
└─ [optimizer.py] - Optimize IR
       ↓
   Optimized IR operations
    ↓
[nodes.py] - IR data structures
    ↓
→ layout/planner.py (for physical layout)
```

## Key Features

### Signal Flow Tracking
Every `SignalRef` points to its producing operation:
```python
# a + b creates IRArith node
arith_op = IRArith(...)  # node_id = "arith_1"
result = SignalRef("signal-A", "arith_1")  # Points to arith_1
```

This enables:
- Dependency tracking
- Optimization (CSE, dead code elimination)
- Layout planning (where to place combinators)

### Type Preservation
Each operation has an `output_type` specifying the Factorio signal type:
```python
IRArith(op="+", left=a, right=b, output_type="iron-plate")
# This operation outputs on the "iron-plate" signal
```

### Debug Metadata
All operations carry metadata for debugging:
- Source AST node reference
- User-declared names
- Expression context (what code generated this)
- Location info (file, line, column)

### Constant Handling
Constants can be:
- **Single signal**: `IRConst` with value and output_type
- **Multiple signals** (bundle): `IRConst` with signals dict

```python
# Single signal constant
const = IRConst(node_id="const_1", output_type="iron-plate")
const.value = 100

# Bundle constant
const = IRConst(node_id="const_2", output_type="bundle")
const.signals = {"iron-plate": 50, "copper-plate": 30}
```

### Memory Representation
Memory operations are tracked with special IR nodes:
```python
# Create memory cell
mem_create = IRMemCreate(mem_id="counter", signal_type="signal-A")

# Write to memory
mem_write = IRMemWrite(mem_id="counter", value_ref=..., condition_ref=...)

# Read from memory
mem_read = IRMemRead(node_id="read_1", mem_id="counter")
```

## Usage Example

```python
from ir.builder import IRBuilder
from ir.optimizer import IROptimizer

# Build IR
builder = IRBuilder()

# Create constant
const_ref = builder.const("iron-plate", 100, None)

# Create arithmetic operation
result_ref = builder.arithmetic(
    op="+",
    left=const_ref,
    right=50,
    output_type="iron-plate",
    source_ast=None
)

# Get all operations
operations = builder.operations

# Optimize
optimizer = IROptimizer()
optimized_ops = optimizer.optimize(operations)

# optimized_ops is a list of IR nodes
for op in optimized_ops:
    print(f"{op.node_id}: {type(op).__name__}")
```

## IR vs AST

| Aspect | AST | IR |
|--------|-----|-----|
| Level | High (language constructs) | Low (combinator operations) |
| Structure | Tree with statements/expressions | Flat list of operations |
| Types | Language types (Signal, Int, Bundle) | Signal types ("iron-plate") |
| Optimization | Minimal (semantic simplification) | Extensive (CSE, const prop, DCE) |
| Platform | Language-independent | Factorio-specific |

## Optimization Details

### Common Subexpression Elimination
Finds operations that compute the same result:
```python
# Before CSE:
arith_1: iron-plate = signal-A + 10
arith_2: iron-plate = signal-A + 10  # Duplicate!

# After CSE:
arith_1: iron-plate = signal-A + 10
# arith_2 removed, all references point to arith_1
```

### Constant Folding
Evaluates constant expressions:
```python
# Before:
const_1: __implicit_1 = 5
const_2: __implicit_2 = 10
arith_1: iron-plate = __implicit_1 + __implicit_2

# After:
const_1: iron-plate = 15
# const_2 and arith_1 removed
```

### Dead Code Elimination
Removes unused operations:
```python
# Before:
arith_1: signal-A = x + y  # Never used
arith_2: signal-B = a * b  # Used in later operation

# After:
arith_2: signal-B = a * b  # Only this remains
```

## Testing

Tests verify:
- IR node creation and properties
- Builder API correctness
- Signal reference tracking
- Optimization correctness (CSE, constant folding)
- Metadata preservation
- Type handling

## Performance

- Operations stored in flat list for cache efficiency
- Node IDs are strings for debugging, integers internally
- Minimal object allocation
- Optimizations are opt-in (can be disabled)

## Future Extensions

- More optimization passes (algebraic simplification)
- SSA (Static Single Assignment) form
- Control flow graph for loops
- Dataflow analysis
- Register allocation equivalent for signal reuse
