# Factompiler Architecture

## Overview

Factompiler is a multi-stage compiler that transforms Factorio Circuit DSL (FCDSL) source code into Factorio blueprint strings. The compilation pipeline consists of five distinct stages, each with well-defined responsibilities and interfaces.

## Compilation Pipeline

```
Source Code (.fcdsl)
    ↓
┌─────────────────────┐
│  1. Parser          │  Input: str  →  Output: AST
└─────────────────────┘
    ↓
┌─────────────────────┐
│  2. Semantic        │  Input: AST  →  Output: Annotated AST + Diagnostics
│     Analyzer        │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  3. AST Lowerer     │  Input: Annotated AST  →  Output: IR
└─────────────────────┘
    ↓
┌─────────────────────┐
│  4. Layout Planner  │  Input: IR  →  Output: LayoutPlan
└─────────────────────┘
    ↓
┌─────────────────────┐
│  5. Blueprint       │  Input: LayoutPlan  →  Output: Blueprint String
│     Emitter         │
└─────────────────────┘
    ↓
Blueprint String (base64)
```

## Stage Responsibilities

### 1. Parser (`dsl_compiler.grammar.parser`)

**Responsibility:** Transform source text into Abstract Syntax Tree (AST)

**Entry Point:** `parse(source_text: str) -> ast.Program`

**Inputs:**
- Source code as string

**Outputs:**
- AST representing the program structure
- Raises `SyntaxError` on parse failures

**Key Operations:**
- Lexical analysis (tokenization)
- Syntactic analysis (parsing)
- AST construction

**Data Structures:**
- Uses Lark grammar definition
- Produces `ast.Program` node with source locations

**Dependencies:**
- Lark parser library
- `dsl_compiler.grammar.ast` module

---

### 2. Semantic Analyzer (`dsl_compiler.src.semantic.analyzer`)

**Responsibility:** Type checking, symbol resolution, and semantic validation

**Entry Point:** `analyze_program(ast_program: ast.Program) -> tuple[ast.Program, ProgramDiagnostics]`

**Inputs:**
- Raw AST from parser
- Optional initial `SignalTypeRegistry`

**Outputs:**
- Annotated AST with type information
- `ProgramDiagnostics` containing warnings and errors
- Populated `SignalTypeRegistry` with all signal types

**Key Operations:**
- Type inference and checking
- Symbol table construction
- Signal type allocation (implicit virtual signals)
- Memory declaration validation
- Function signature validation
- Control flow validation

**Data Structures:**
- `symbol_table: dict[str, SymbolInfo]` - Maps identifiers to their types and metadata
- `signal_registry: SignalTypeRegistry` - Central registry of all signal types
- `expr_types: dict[ast.Expression, Type]` - Type annotations for expressions
- `simple_sources: dict[str, ast.Expression]` - Tracks simple computed values
- `computed_values: dict[str, any]` - Compile-time constant values

**Shared State:**
- `SignalTypeRegistry` is shared with `SignalAllocator` and passed to IR builder
- Signal allocator shares the registry to coordinate implicit type allocation

**Dependencies:**
- `dsl_compiler.src.common.signal_registry.SignalTypeRegistry`
- `dsl_compiler.src.common.diagnostics.ProgramDiagnostics`
- `dsl_compiler.src.semantic.signal_allocator.SignalAllocator`

---

### 3. AST Lowerer (`dsl_compiler.src.lowering.lowerer`)

**Responsibility:** Transform high-level AST into low-level Intermediate Representation (IR)

**Entry Point:** `lower_program(ast_program: ast.Program, semantic: SemanticAnalyzer) -> IRProgram`

**Inputs:**
- Annotated AST from semantic analyzer
- `SemanticAnalyzer` instance with type information and signal registry

**Outputs:**
- `IRProgram` containing low-level operations

**Key Operations:**
- Lower high-level constructs to combinator-level operations
- Generate IR for arithmetic, logic, and memory operations
- Transform control flow into combinator networks
- Expand function calls into inline operations

**Data Structures:**
- `IRProgram` - Container for all IR operations
- `IROperation` - Individual low-level operation (arithmetic, comparison, memory access, etc.)
- Shares `SignalTypeRegistry` from semantic analyzer

**Shared State:**
- Receives `signal_registry` from semantic analyzer via IR builder
- IR builder stores registry to maintain signal type information

**Dependencies:**
- `dsl_compiler.src.ir.builder.IRBuilder`
- `dsl_compiler.src.semantic.analyzer.SemanticAnalyzer`
- `dsl_compiler.src.common.signal_registry.SignalTypeRegistry`

---

### 4. Layout Planner (`dsl_compiler.src.layout.planner`)

**Responsibility:** Determine physical placement of entities in 2D space

**Entry Point:** `plan_layout(ir_program: IRProgram) -> LayoutPlan`

**Inputs:**
- IR program with operations and memory declarations

**Outputs:**
- `LayoutPlan` containing entity placements and connections

**Key Operations:**
- Allocate entity instances for each IR operation
- Calculate entity positions (row-based layout)
- Handle entity alignment and grid constraints
- Plan wire connections between entities
- Place memory cells with appropriate spacing

**Data Structures:**
- `LayoutPlan` - Contains all entity placements and wire connections
- `EntityPlacement` - Position, rotation, and metadata for each entity
- `EntityDataHelper` - Dynamic entity footprint and alignment information

**Dependencies:**
- `dsl_compiler.src.layout.entity_placer.EntityPlacer`
- `dsl_compiler.src.layout.memory.MemoryLayoutPlanner`
- `dsl_compiler.src.common.entity_data` - For dynamic entity information
- Draftsman library for entity prototype data

---

### 5. Blueprint Emitter (`dsl_compiler.src.emission.emitter`)

**Responsibility:** Generate Factorio blueprint string from layout plan

**Entry Point:** `emit_from_plan(layout_plan: LayoutPlan, ir_program: IRProgram) -> str`

**Inputs:**
- `LayoutPlan` with entity placements
- `IRProgram` for signal and operation metadata

**Outputs:**
- Base64-encoded blueprint string

**Key Operations:**
- Instantiate Draftsman entities at planned positions
- Configure combinator parameters (operations, signals, constants)
- Establish wire connections (red/green circuit network)
- Generate blueprint JSON and encode to string

**Data Structures:**
- Uses Draftsman `Blueprint` class
- `SignalResolver` for mapping signal names to Factorio signal IDs

**Dependencies:**
- Draftsman library for blueprint generation
- `dsl_compiler.src.emission.signal_resolver.SignalResolver`

---

## Shared Infrastructure (`dsl_compiler.src.common`)

### ProgramDiagnostics

**Purpose:** Centralized diagnostic collection across all stages

**Key Methods:**
- `add(severity, message, location)` - Add a diagnostic
- `merge(other_diagnostics)` - Combine diagnostics from multiple stages
- `has_errors()` - Check if any errors were recorded
- `print()` - Display diagnostics to user

**Severity Levels:** DEBUG, INFO, WARNING, ERROR

---

### SignalTypeRegistry

**Purpose:** Centralized registry for all signal types in the program

**Storage Format:** `dict[str, dict[str, str]]`
- Key: Signal name
- Value: `{"name": str, "type": str}` - Type is "item", "fluid", "virtual", etc.

**Key Methods:**
- `register(name, signal_type)` - Register a signal with its type
- `get_type(name)` - Retrieve signal type
- `contains(name)` - Check if signal is registered
- `to_dict()` - Export as dictionary

**Usage Pattern:**
1. Semantic analyzer creates registry
2. Signal allocator shares registry to allocate implicit virtual signals
3. IR builder receives registry from analyzer
4. Registry is shared (not copied) across stages

**Critical Note:** Registry has `__len__` method, so empty registries are falsy. Always use explicit `if registry is None` checks instead of `if registry`.

---

### SourceLocation

**Purpose:** Track source code positions for error reporting

**Fields:**
- `line: int` - Line number (1-indexed)
- `column: int` - Column number (1-indexed)
- `filename: str` - Source file path

---

### EntityDataHelper

**Purpose:** Dynamic extraction of entity information from Draftsman

**Key Functions:**
- `get_entity_footprint(prototype: str) -> tuple[int, int]` - Get entity dimensions
- `get_entity_alignment(prototype: str) -> int` - Get grid alignment requirement
- `supports_circuit_network(prototype: str) -> bool` - Check circuit connection support

**Rationale:** Eliminates hardcoded entity data, making compiler adaptable to any Factorio entity.

---

## Stage Isolation Principles

1. **Single Responsibility:** Each stage has one clear responsibility
2. **Minimal Coupling:** Stages communicate through well-defined interfaces
3. **Shared State Discipline:** Only `SignalTypeRegistry` is shared (semantic → lowerer)
4. **Diagnostic Aggregation:** All stages contribute diagnostics, merged in `compile.py`
5. **No Backflow:** Later stages never modify earlier stage outputs

---

## Data Flow

### Signal Type Information Flow

```
SemanticAnalyzer (creates SignalTypeRegistry)
    ↓ (shares reference)
SignalAllocator (allocates implicit types)
    ↓ (shares reference)
IRBuilder (receives via lowerer)
    ↓ (used for IR generation)
LayoutPlanner (reads signal types from IR)
    ↓ (resolves signals)
BlueprintEmitter (uses SignalResolver)
```

### Diagnostic Information Flow

```
Parser → SyntaxError exceptions
SemanticAnalyzer → ProgramDiagnostics (warnings, errors)
IRBuilder → Errors raised as exceptions
LayoutPlanner → Warnings (e.g., entity alignment)
BlueprintEmitter → Draftsman warnings

All merged in compile.py → Final diagnostic output
```

---

## Extension Points

### Adding New Language Features

1. **Extend Grammar:** Update `grammar/dsl.lark` and `grammar/ast.py`
2. **Add Semantic Rules:** Update `semantic/analyzer.py` type checking
3. **Lower to IR:** Add IR operations in `lowering/lowerer.py`
4. **Plan Layout:** Update `layout/planner.py` if new entity types needed
5. **Emit Blueprint:** Update `emission/emitter.py` for new combinator configurations

### Adding New Entity Types

1. Draftsman automatically provides entity data via `EntityDataHelper`
2. If entity has special layout requirements, update `EntityPlacer`
3. If entity has unique combinator parameters, update `emitter.py`

### Adding New Signal Types

1. Register in `SignalTypeRegistry` during semantic analysis
2. Signal allocator automatically handles implicit virtual signals
3. Update `SignalResolver` if new signal type requires special mapping

---

## Testing Strategy

### Unit Tests
- **Grammar:** `tests/test_parser.py` - AST construction
- **Semantics:** `tests/test_semantic.py` - Type checking
- **Lowering:** `tests/test_lowering.py` - IR generation
- **Layout:** `tests/test_layout.py` - Entity placement
- **Common:** `tests/test_common_infrastructure.py` - Shared utilities

### Integration Tests
- **Sample Programs:** `tests/sample_programs/*.fcdsl` - End-to-end compilation
- Each sample tests specific language features
- Validates complete pipeline from source to blueprint

### Test Execution
```bash
pytest tests/                    # All tests
pytest tests/test_semantic.py   # Specific stage
pytest -v                        # Verbose output
```

---

## Future Improvements

### Phase 6: Remove Legacy Code
- Identify and remove any remaining dead code
- Clean up unused imports

### Phase 7: Error Handling Consistency
- Standardize exception types across stages
- Ensure all errors include source locations
- Add recovery strategies for common errors

### Phase 8: Final Cleanup
- Update all documentation
- Add comprehensive docstrings
- Performance profiling and optimization

---

## References

- **Language Specification:** `Language_SPEC.md`
- **Compiler Specification:** `COMPILER_SPEC.md`
- **Refactoring Plan:** `REFACTORING_PLAN.MD`
- **Agent Guidelines:** `AGENTS.MD`
