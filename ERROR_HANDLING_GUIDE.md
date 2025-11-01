# Error Handling Guide

## Overview

The Factom compiler uses a multi-level diagnostic system with consistent error handling patterns across all stages.

## Diagnostic Severity Levels

### DEBUG
- Internal compiler state information
- Signal allocation details  
- Wire color assignments
- Layout planning decisions

**User Impact**: None (only shown with `--debug` flag)

### INFO
- Successful compilation stage completion
- File processing notifications
- Optimization summaries

**User Impact**: Helpful context (shown with `--verbose` flag)

### WARNING
- Type coercion (mixing signal types)
- Potential wiring issues (missing placements)
- Non-critical configuration issues
- Memory type mismatches that can be handled

**User Impact**: Program compiles but may not be optimal

### ERROR
- Syntax errors (parsing failures)
- Undefined variables/memories
- Type mismatches in strict contexts
- Invalid IR operations
- Blueprint generation failures

**User Impact**: Compilation stops

## Error Message Patterns

### Good Error Messages

All error messages follow these patterns:

1. **Context-Specific**: Include the entity/variable/operation causing the error
   ```python
   self.diagnostics.error(f"Undefined variable '{expr.name}'", expr)
   ```

2. **Source Location**: Pass AST node when available for line/column tracking
   ```python
   self.diagnostics.error("Type mismatch", node=ast_node)
   ```

3. **Actionable**: Provide suggestions when possible
   ```
   Mixed signal types in binary operation:
   To align types, consider:
     - (left | "type") op right
     - left op (right | "type")
   ```

### Error Handling Flow

1. **Parsing Stage**: Lark parser handles syntax errors automatically
   - Parser catches exceptions and converts to diagnostics
   - Source locations tracked via Lark tokens

2. **Semantic Analysis**: Validates program semantics
   - Undefined variable/memory detection
   - Type checking and coercion warnings
   - Property access validation

3. **Lowering Stage**: Transforms AST to IR
   - Memory operation validation
   - Statement type checking
   - IR construction errors

4. **Layout Planning**: Physical layout generation
   - Wiring conflicts
   - Power network issues
   - Placement failures

5. **Blueprint Emission**: Final output generation
   - Entity creation errors
   - Blueprint serialization failures
   - Draftsman library warnings (GridAlignment, OverlappingObjects)

## Exception Hierarchy

### RuntimeError
Used for internal compiler errors that should not occur in normal operation:
- Uninitialized state (e.g., calling methods before prepare())
- Unexpected AST node types
- Grammar loading failures

### ValueError  
Used for input validation errors:
- Malformed AST structures
- Invalid configuration values
- Missing required data

## Diagnostic Collection

All stages use the shared `ProgramDiagnostics` system:

```python
from dsl_compiler.src.common import ProgramDiagnostics

diagnostics = ProgramDiagnostics()
diagnostics.error("Error message", node=ast_node)
diagnostics.warning("Warning message", location="semantic")
diagnostics.debug("Debug info")
```

Diagnostics are accumulated through the pipeline and displayed at the end.

## Best Practices

1. **Always pass source location when available**
   - Use AST nodes for semantic/lowering errors
   - Use line numbers for parsing errors
   - Use operation names for layout/emission errors

2. **Provide context in error messages**
   - Include variable/memory/entity names
   - Show expected vs actual types
   - Reference the specific operation

3. **Use appropriate severity levels**
   - Errors: Compilation cannot continue
   - Warnings: Potential issues but compilation succeeds
   - Info: Helpful context for users
   - Debug: Internal details for compiler developers

4. **Don't use defensive programming**
   - Let exceptions propagate when appropriate
   - Don't catch and silence errors
   - Use assertions for invariants in development

## Common Error Scenarios

### Undefined Variables
```python
self.diagnostics.error(f"Undefined variable '{expr.name}'", expr)
```

### Type Mismatches
```python
self.diagnostics.warning(
    f"Type mismatch: Memory expects {expected} but got {actual}",
    node
)
```

### Missing Placements
```python
self.diagnostics.warning(
    f"Skipped wiring for '{signal}' due to missing placement ({source} -> {dest})"
)
```

### Blueprint Failures
```python
self.diagnostics.error(f"Blueprint emission failed: {e}")
```

## Testing Error Handling

All error handling is tested through:
- Unit tests for individual stages
- Integration tests for end-to-end compilation
- Sample programs covering edge cases

See `tests/test_stage_interfaces.py` for error handling validation tests.
