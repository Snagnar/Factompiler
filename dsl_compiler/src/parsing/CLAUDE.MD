# Parsing Module

## Purpose
Converts Facto source code into an Abstract Syntax Tree (AST). Handles lexing, parsing with Lark grammar, import resolution, and transformation to typed AST nodes.

## Files

### parser.py
**Main Parser Interface**
- `DSLParser`: Entry point for parsing Facto code
  - Uses Lark parser with `grammar/facto.lark` grammar file
  - Handles parse errors with helpful messages
  - Coordinates preprocessing and transformation
  - `parse(source, filename)`: Main parsing method

**Flow**: `source text` → Lark parse tree → AST (via transformer)

### preprocessor.py
**Import Resolution and Preprocessing**
- `Preprocessor`: Handles import statements before main parsing
  - Resolves relative import paths
  - Reads imported files
  - Inlines imported code into main source
  - Prevents circular imports
  - Tracks file dependencies

**Import handling**:
- `import "path/to/file.facto"` → file contents inserted inline
- `import "lib/module.facto" as prefix` → future feature for namespacing

### transformer.py
**Lark Tree to AST Transformation**
- `DSLTransformer`: Converts Lark parse tree nodes to AST nodes
  - Implements Lark's `Transformer` interface
  - One method per grammar rule (e.g., `decl_stmt`, `binary_op`)
  - Handles operator precedence and associativity
  - Constructs typed AST nodes from tokens
  - Preserves source location information

**Key transformations**:
- `literal` → `NumberLiteral`, `StringLiteral`
- `decl_stmt` → `DeclStmt`
- `binary_op` → `BinaryOp`
- `bundle_select_chain` → `BundleSelectExpr` (supports chained indexing like `entity.output["signal-A"]`)
- `lvalue` → `Identifier`, `PropertyAccess`
- `primary` → Various expression nodes with special handling for `entity.output`

## Information Flow

```
Source Code
    ↓
[preprocessor.py] - Resolve imports, inline code
    ↓
Preprocessed Source
    ↓
[Lark Parser] - Parse with facto.lark grammar
    ↓
Lark Parse Tree (Token/Tree nodes)
    ↓
[transformer.py] - Transform to typed AST
    ↓
AST (Program node with statement list)
    ↓
→ semantic/analyzer.py (for type checking)
```

## Key Features

### Grammar-Driven Parsing
The `facto.lark` grammar defines:
- Operator precedence (from lowest to highest):
  - Logic OR (`||`, `or`)
  - Logic AND (`&&`, `and`)
  - Output specification (`:`)
  - Comparison (`==`, `<`, `>`, etc.)
  - Projection (`|`)
  - Bitwise OR, XOR, AND
  - Shift (`<<`, `>>`)
  - Addition/Subtraction
  - Multiplication/Division/Modulo
  - Power (`**`) - right associative
  - Unary (`-`, `!`, `+`)

- Statement types: declarations, assignments, loops, functions
- Expression forms: literals, identifiers, calls, projections
- Special constructs: bundles, memory operations, entity placement

### Import System
Preprocessor resolves imports before parsing:
- Relative to current file location
- Recursive import resolution
- Circular dependency detection
- Error reporting with proper file context

### Postfix Expression Handling
The `bundle_select_chain` rule enables method-chaining style:
```facto
entity.output["signal-A"]["signal-B"]  # Chained subscripts
```

This is transformed into nested `BundleSelectExpr` nodes.

### Special Expression Detection
Transformer recognizes special patterns:
- `entity.output` → `EntityOutputExpr` (not generic property access)
- `mem.read()` → `ReadExpr`
- `mem.write(...)` → `WriteExpr` variants

## Usage Example

```python
from parsing.parser import DSLParser

parser = DSLParser()

# Parse a simple program
code = '''
Signal a = 10;
Signal b = a * 2;
'''
ast = parser.parse(code, "example.facto")

# ast is a Program node
print(f"Parsed {len(ast.statements)} statements")

# Parse with imports
code_with_imports = '''
import "lib/utilities.facto";
Signal result = helper_function(5);
'''
ast = parser.parse(code_with_imports, "main.facto")
# Preprocessor will inline utilities.facto before parsing
```

## Error Handling

### Parse Errors
- Lark provides detailed syntax error messages
- DSLParser wraps errors with file context
- Shows line/column and expected tokens

### Import Errors
- Missing files reported with search paths
- Circular imports detected and reported
- Import path resolution errors with context

## Testing

Test files in `tests/` verify:
- Individual transformer methods
- End-to-end parsing of various constructs
- Import resolution and preprocessing
- Error message quality
- Operator precedence correctness
- Source location preservation

## Performance Considerations

- Lark parser is cached and reused
- Preprocessor minimizes file I/O
- Transformer operates in single pass
- No backtracking during transformation

## Future Extensions

- Namespace support for imports (`import "x" as y`)
- More sophisticated import aliasing
- Incremental parsing for IDE integration
- Better error recovery for partial parses
