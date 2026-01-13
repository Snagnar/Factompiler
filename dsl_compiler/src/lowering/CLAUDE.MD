# Lowering Module

## Purpose
Translates validated AST into IR (Intermediate Representation). Bridges the gap between high-level language constructs and low-level combinator operations.

## Files

### lowerer.py
**Main Lowering Orchestrator**
- `ASTLowerer`: Coordinates AST → IR translation
  - Entry point: `lower_program(ast)` → list of IR operations
  - Manages symbol tracking (variables, entities, memories)
  - Delegates to specialized lowerers
  - Tracks signal references and types
  - Handles function inlining
  - Collects diagnostics (errors/warnings)

### expression_lowerer.py
**Expression Translation**
- `ExpressionLowerer`: Converts expressions to IR operations
  - `lower_expr(expr)` → `ValueRef` (SignalRef, BundleRef, or int)
  - Handles all expression types (binary ops, calls, projections, etc.)
  - Performs constant folding during lowering
  - Implements **projection folding optimization** (new!)
    - Folds `expr | "type"` into parent operation's output type
    - Reduces combinator count
  - Creates arithmetic/decider combinators
  - Manages signal type resolution

**Key Lowering Rules**:
- Constants → `IRConst`
- Binary ops → `IRArith` or `IRDecider`
- Projections → Signal type change (or folded into source)
- Function calls → Inline function body with parameter substitution
- Bundle operations → Wire merges or each-mode operations

### statement_lowerer.py
**Statement Translation**
- `StatementLowerer`: Converts statements to IR effects
  - Declarations → Register in symbol table, lower RHS expression
  - Assignments → Lower expression, track variable
  - For loops → Unroll loop iterations
  - Function declarations → Store for later inlining

### memory_lowerer.py
**Memory Operation Translation**
- `MemoryLowerer`: Handles memory cells and latch logic
  - Memory declaration → `IRMemCreate`
  - `mem.read()` → `IRMemRead`
  - `mem.write(value)` → `IRMemWrite`
  - `mem.write(value, set=..., reset=...)` → `IRLatchWrite` (SR/RS latch)

### constant_folder.py
**Compile-Time Evaluation**
- `ConstantFolder`: Evaluates constant expressions
  - Arithmetic on integer literals
  - Boolean operations
  - Bitwise operations
  - Reduces IR size, improves performance

## Information Flow

```
AST (from semantic analysis)
    ↓
[lowerer.py] - Main coordinator
    ↓
├─ [statement_lowerer.py] - Lower statements
│   ├─ For declarations/assignments
│   └─ Delegates expr lowering to expression_lowerer
│
├─ [expression_lowerer.py] - Lower expressions
│   ├─ Binary ops → IRArith/IRDecider
│   ├─ Constants → IRConst
│   ├─ Projections → Type changes (with folding optimization!)
│   └─ Function calls → Inline body
│
├─ [memory_lowerer.py] - Lower memory operations
│   ├─ Memory declarations → IRMemCreate
│   └─ Read/write → IRMemRead/IRMemWrite/IRLatchWrite
│
└─ [constant_folder.py] - Fold constants
    ├─ Evaluate at compile time
    └─ Reduce IR operation count

All components use:
[ir/builder.py] - Construct IR operations
    ↓
List of IR operations
    ↓
→ ir/optimizer.py (optional optimization)
→ layout/planner.py (physical layout)
```

## Key Features

### Projection Folding Optimization (NEW!)
When lowering `expr | "signal-type"`:
1. Check if `expr` just created an IR operation
2. If operation is "fresh" (not user-declared, not multiply referenced)
3. Change operation's output type directly instead of creating projection combinator
4. Reduces combinator count significantly

Example:
```facto
Signal b = a * 255 | "iron-plate";

# Without folding:
# arith_1: signal-A = signal-A * 255
# arith_2: iron-plate = signal-A + 0  (projection)

# With folding (new!):
# arith_1: iron-plate = signal-A * 255  (folded!)
```

### Function Inlining
Functions are inlined at call sites:
```facto
func double(Signal x) { return x * 2; }
Signal y = double(5);

# Lowered to:
# const_1: __implicit_1 = 5
# arith_1: signal-A = __implicit_1 * 2
```

Parameter substitution happens during lowering.

### Constant Folding
Arithmetic on constants evaluated at compile time:
```facto
Signal x = 10 + 20 * 3;

# Lowered to:
# const_1: signal-A = 70  (computed at compile time!)
```

### Loop Unrolling
For loops are unrolled:
```facto
for i in 0..2 {
    Signal s = i * 10;
}

# Lowered to:
# const_1: signal-A = 0
# const_2: signal-A = 10
# const_3: signal-A = 20
```

### Signal Type Tracking
Lowerer maintains `signal_refs` dict mapping variable names to IR references:
```python
signal_refs = {
    "a": SignalRef("iron-plate", "const_1"),
    "b": SignalRef("copper-plate", "arith_2"),
}
```

Enables looking up variables during expression lowering.

## Usage Example

```python
from lowering.lowerer import ASTLowerer
from semantic.analyzer import SemanticAnalyzer
from parsing.parser import DSLParser
from common.diagnostics import ProgramDiagnostics

# Parse and analyze
parser = DSLParser()
ast = parser.parse("Signal x = 10 + 20;", "test.facto")

diagnostics = ProgramDiagnostics()
analyzer = SemanticAnalyzer(diagnostics)
analyzer.visit(ast)

# Lower to IR
lowerer = ASTLowerer(analyzer, diagnostics)
ir_operations = lowerer.lower_program(ast)

# ir_operations is a list of IR nodes
for op in ir_operations:
    print(op)
# Output: IRConst(const_1: signal-A = 30)  # Constant folded!
```

## Error Handling

Lowering can fail with errors:
- Undefined variables
- Type mismatches (should be caught by semantic analysis)
- Unsupported operations
- Invalid function calls

Errors collected in `diagnostics` object.

## Testing

Tests verify:
- Expression lowering correctness
- Statement lowering
- Function inlining
- Loop unrolling
- Constant folding
- Memory operation lowering
- **Projection folding optimization** (new!)
- Error detection

## Performance

- Single-pass lowering
- Minimal temporary allocations
- Constant folding during lowering reduces IR size
- **Projection folding reduces combinator count** (new!)

## Future Extensions

- Tail call optimization
- More aggressive constant propagation
- Function specialization
- Loop optimization (strength reduction)
