# Semantic Analysis Module

## Purpose
Performs type checking, symbol resolution, and semantic validation on the AST. Ensures program correctness before lowering to IR.

## Files

### analyzer.py
**Main Semantic Analyzer**
- `SemanticAnalyzer`: Visitor that traverses AST to validate semantics
  - Type inference for all expressions
  - Symbol table management (variables, functions, entities)
  - Type checking for assignments and operations
  - Function call validation (argument count, types)
  - Memory operation validation
  - Entity property validation
  - Scope management for functions and loops

**Key Methods**:
- `visit(node)`: Main visitor dispatcher
- `infer_expr_type(expr)`: Type inference for expressions
- `visit_decl_stmt`, `visit_assign_stmt`: Statement validation
- `check_type_compatibility`: Type checking between values
- `_infer_*_type` methods: Specialized type inference

### type_system.py
**Type Definitions**
- `ValueInfo`: Base class for all types
- `IntValue`: Integer type (untyped numeric values)
- `SignalValue`: Signal type with specific signal type (e.g., "iron-plate", "signal-A")
  - `SignalTypeInfo`: Metadata about signal type (name, category, icon)
- `EntityValue`: Entity type (placed Factorio entities)
- `BundleValue`: Bundle type (collection of signals)
  - `DynamicBundleValue`: Bundle with runtime-determined signal types (e.g., chest contents)
- `MemoryValue`: Memory cell type
- `FunctionValue`: Function type with parameter and return types

**Type Categories**:
- **Scalar types**: `IntValue`, `SignalValue`
- **Composite types**: `BundleValue`, `EntityValue`
- **Special types**: `MemoryValue`, `FunctionValue`

### symbol_table.py
**Symbol Management**
- `Symbol`: Represents a named entity with type and metadata
  - `symbol_type`: Enum (VARIABLE, FUNCTION, ENTITY, MEMORY, PARAMETER)
  - `value_type`: The ValueInfo type of the symbol
  - `metadata`: Additional info (entity prototype, function params, etc.)

- `SymbolTable`: Scoped symbol storage
  - Supports nested scopes (functions, loops)
  - `define(name, symbol)`: Add symbol to current scope
  - `lookup(name)`: Find symbol in current or parent scopes
  - `push_scope`, `pop_scope`: Scope management

## Information Flow

```
AST (from parsing)
    ↓
[analyzer.py] - Main semantic analysis
    ↓
├─ [symbol_table.py] - Track definitions and scopes
│      ↓
│   Symbol resolution (names → types)
│
├─ [type_system.py] - Type checking and inference
│      ↓
│   Type validation (operations, assignments)
│
└─ Error/Warning collection via diagnostics
    ↓
Validated AST + Symbol information
    ↓
→ lowering/lowerer.py (for IR generation)
```

## Key Features

### Type Inference
The analyzer infers types for expressions:
```facto
Signal a = 10;           # IntValue → converted to SignalValue
Signal b = a * 2;        # SignalValue (inferred from a)
Signal c = b | "copper"; # SignalValue with type "copper-plate"
```

Type inference rules:
- Literals: `10` → `IntValue`, `("iron", 5)` → `SignalValue`
- Operations: Result type from operands (signals dominate)
- Projections: Target type from `|` operator
- Function calls: Return type from function signature
- Entity properties: Type from entity definition

### Type Checking
Validates type compatibility:
- Assignment: RHS type must be compatible with LHS declared type
- Operations: Operands must have compatible types
- Function arguments: Must match parameter types
- Memory operations: Value must match memory cell type

**Compatibility Rules**:
- `IntValue` can be assigned to `SignalValue` (auto-converted)
- `SignalValue` types must match or be projectable
- `BundleValue` operations require matching bundle structures

### Symbol Resolution
Tracks all named entities:
- Variables (Signal, Int, Bundle)
- Functions (with parameter types)
- Entities (with placement info)
- Memory cells (with signal types)
- Loop iterators (function parameters)

**Scoping**:
- Global scope: Top-level declarations
- Function scope: Parameters and local variables
- Loop scope: Iterator variables

### Special Validations

**Entity Property Access**:
- Only valid properties allowed (e.g., `entity.output`, `lamp.enable`)
- Type checking based on entity prototype
- Read vs write properties enforced

**Memory Operations**:
- Memory cells must be declared before use
- Read/write type consistency
- Latch operations (set/reset) validated

**Function Calls**:
- Argument count matches parameters
- Argument types compatible with parameters
- Recursive calls detected (future: allow with tail call optimization)

**Bundle Operations**:
- Bundle literals validate signal types
- Bundle selection checks signal existence (for static bundles)
- Dynamic bundles (entity.output) allow any signal type

### Projection Simplification
Analyzer optimizes projections at semantic stage:
```facto
Signal a = 10 | "iron-plate";        # Simplified to signal literal
Signal b = (10 | "iron") | "copper"; # Nested projection simplified
```

## Usage Example

```python
from semantic.analyzer import SemanticAnalyzer
from parsing.parser import DSLParser
from common.diagnostics import ProgramDiagnostics

# Parse code
parser = DSLParser()
ast = parser.parse("Signal x = 10;", "test.facto")

# Analyze
diagnostics = ProgramDiagnostics()
analyzer = SemanticAnalyzer(diagnostics)
analyzer.visit(ast)

# Check for errors
if diagnostics.has_errors():
    for msg in diagnostics.get_messages():
        print(msg)
else:
    # Access symbol information
    symbol = analyzer.symbol_table.lookup("x")
    print(f"Variable x has type: {symbol.value_type}")
```

## Error Detection

The analyzer catches:
- **Type errors**: Incompatible types in operations, assignments
- **Name errors**: Undefined variables, functions, entities
- **Scope errors**: Variable redefinition, shadowing
- **Operation errors**: Invalid operations on types (e.g., bundle + bundle)
- **Function errors**: Wrong argument count/types, undefined functions
- **Entity errors**: Invalid entity types, unknown properties
- **Memory errors**: Undefined memory cells, type mismatches

## Type System Details

### Signal Types
Signals have explicit types matching Factorio signal types:
- **Items**: "iron-plate", "copper-cable", "electronic-circuit"
- **Fluids**: "water", "crude-oil", "petroleum-gas"
- **Virtual signals**: "signal-A", "signal-B", ..., "signal-0", "signal-1", ...
- **Special**: "signal-each", "signal-anything", "signal-everything"

### Dynamic Types
Some expressions have runtime-determined types:
- `entity.output`: `DynamicBundleValue` (contents unknown until runtime)
- Function parameters: Type from call site

### Type Widening
`IntValue` automatically widens to `SignalValue` when needed:
```facto
Signal x = 10;  # 10 (IntValue) → SignalValue with implicit type
```

## Testing

Tests verify:
- Type inference correctness
- Type error detection
- Symbol resolution
- Scope management
- Entity validation
- Memory operation validation
- Function analysis
- Edge cases and error messages

## Performance

- Single-pass analysis
- Lazy type inference (cached)
- Efficient symbol table lookups
- Minimal memory allocation

## Future Extensions

- Flow-sensitive typing (track value changes)
- More sophisticated type inference
- Generic functions
- Type aliases
- Const/readonly tracking
