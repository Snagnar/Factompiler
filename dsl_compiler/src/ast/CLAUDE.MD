# AST Module

## Purpose
Defines the Abstract Syntax Tree (AST) node classes for the Facto language. These represent the parsed structure of source code before semantic analysis and lowering to IR.

## Files

### base.py
**Base AST Node**
- `ASTNode`: Base class for all AST nodes
- Tracks source location (file, line, column) for error reporting
- Provides visitor pattern interface for tree traversal

### expressions.py
**Expression AST Nodes**
Expression nodes represent computations that produce values:

- `Expr`: Base class for all expressions
- `BinaryOp`: Binary operations (`+`, `-`, `*`, `/`, `==`, `<`, etc.)
- `UnaryOp`: Unary operations (`-`, `!`, `+`)
- `IdentifierExpr`: Variable references in expression context
- `CallExpr`: Function calls (`func(arg1, arg2)`)
- `PropertyAccessExpr`: Property access in expressions (`entity.property`)
- `ProjectionExpr`: Signal type projection (`expr | "type"`)
- `OutputSpecExpr`: Decider output specification (`condition : value`)
- `ReadExpr`: Memory read operations (`mem.read()`)
- `WriteExpr`: Memory write operations (`mem.write(value)` and variants)
- `SignalLiteral`: Signal literals (`("iron-plate", 100)`)
- `SignalTypeAccess`: Signal type access (`signal.type`)
- `EntityOutputExpr`: Entity circuit output (`entity.output`)
- `BundleLiteral`: Bundle literals (`{ signal1, signal2 }`)
- `BundleSelectExpr`: Bundle element selection (`bundle["signal-type"]`)
- `BundleAnyExpr`: Bundle any operation (`any(bundle)`)
- `BundleAllExpr`: Bundle all operation (`all(bundle)`)

### literals.py
**Literal and LValue Nodes**
Basic value nodes and assignment targets:

- `NumberLiteral`: Integer literals (`42`, `0xFF`)
- `StringLiteral`: String literals (`"text"`)
- `DictLiteral`: Dictionary literals (`{key: value}`)
- `LValue`: Base class for assignable expressions
- `Identifier`: Simple variable names (can be assigned to)
- `PropertyAccess`: Property access as lvalue (`entity.property = value`)

### statements.py
**Statement AST Nodes**
Statements represent program actions:

- `Stmt`: Base class for all statements
- `Program`: Root node containing list of statements
- `DeclStmt`: Variable declarations (`Signal x = 10;`)
- `AssignStmt`: Assignments (`x = value;`)
- `ExprStmt`: Expression statements (expressions followed by semicolon)
- `ReturnStmt`: Function returns (`return value;`)
- `ImportStmt`: Module imports (`import "file.facto";`)
- `FuncDecl`: Function declarations with typed parameters
- `MemDecl`: Memory cell declarations (`Memory counter: "signal-A";`)
- `ForStmt`: For loops with range or list iterators
- `TypedParam`: Function parameter with type annotation

## Information Flow

1. **Parser creates AST**: The parser (`parsing/parser.py`) generates a Lark tree, which is transformed into AST nodes by `parsing/transformer.py`

2. **Semantic Analysis**: The `semantic/analyzer.py` traverses the AST using the visitor pattern to:
   - Build symbol tables
   - Infer and check types
   - Validate expressions and statements
   - Resolve references

3. **Lowering to IR**: The `lowering/lowerer.py` walks the AST to generate IR operations:
   - Statements are lowered by `statement_lowerer.py`
   - Expressions are lowered by `expression_lowerer.py`
   - Memory operations by `memory_lowerer.py`

4. **Source Location Tracking**: Every AST node maintains source location info from `base.py`, enabling precise error messages throughout compilation

## Key Design Patterns

- **Visitor Pattern**: AST nodes accept visitors (analyzers, lowerers) for tree traversal
- **Composition**: Complex expressions built from simpler expression nodes
- **Separation of Concerns**: Expressions vs statements vs literals vs lvalues
- **Position Tracking**: All nodes track source file, line, and column for debugging

## Usage Example

```python
# Creating an AST node manually
binary_op = BinaryOp(
    op="+",
    left=IdentifierExpr("a"),
    right=NumberLiteral(10),
    line=5,
    column=10
)

# Typically created by transformer from parse tree
from parsing.parser import DSLParser
parser = DSLParser()
ast = parser.parse("Signal x = 10;", "test.facto")
# ast is a Program node containing DeclStmt nodes
```

## Testing

Test files in `tests/` directory verify:
- AST node creation and properties
- Node inheritance relationships
- Source location tracking
- Tree structure validation
