# Factompiler Compilation Flow Guide

This document explains how a Factorio Circuit DSL source file is transformed into a Blueprint string, and describes the role of every module in `dsl_compiler/src`. It is intended as a quick orientation for contributors and coding agents.

## End-to-End Control Flow
1. **CLI orchestration (`compile.py`)** – The command-line entry point reads the DSL file, configures diagnostic modes, and invokes the compiler pipeline.
2. **Preprocessing (`parsing/preprocessor.py`)** – Resolves `import` statements by inlining referenced DSL files before parsing, ensuring a single compilation unit.
3. **Parsing (`parsing/parser.py`, `parsing/transformer.py`)** – Lark-based parsing builds a typed AST defined in `ast/`. `parser.py` loads the grammar, `transformer.py` converts the parse tree into AST nodes, and `parsing/__init__.py` exposes the public parser API.
4. **AST definitions (`ast/`)** – The AST packages (`base.py`, `expressions.py`, `statements.py`, `literals.py`, `types.py`, aggregated by `ast/__init__.py`) provide immutable node classes and visitors used throughout the pipeline.
5. **Semantic analysis (`semantic/analyzer.py`)** – Performs symbol resolution, type inference, constant signal allocation, and diagnostic emission using helpers from `semantic/diagnostics.py`, `symbol_table.py`, `type_system.py`, `signal_allocator.py`, and `validators.py`. The package façade (`semantic/__init__.py`) exports the analyzer and support types.
6. **Lowering to IR (`lowering/lowerer.py` and helpers)** – Converts typed AST into IR operations via the orchestrator in `lowering/lowerer.py`, delegating expression handling to `expression_lowerer.py`, statement control to `statement_lowerer.py`, memory-specific logic to `memory_lowerer.py`, and compile-time evaluation to `constant_folder.py`. `lowering/__init__.py` exposes `ASTLowerer` and `lower_program`.
7. **Intermediate representation (`ir/`)** – `ir/nodes.py` defines canonical IR nodes and signal references, `ir/builder.py` constructs them, and `ir/optimizer.py` performs common subexpression elimination. The package initializer publishes the IR API.
8. **Emission preparation (`emission/emitter.py`)** – The `BlueprintEmitter` prepares signal usage, coordinates helper components, and ultimately emits Draftsman entities. It is backed by `signal_resolver.py` for signal naming, `entity_emitter.py` for combinator/entity instantiation, and `connection_builder.py` for wire planning. Supporting utilities (`layout.py`, `memory.py`, `signals.py`, `wiring.py`, `debug_format.py`) handle placement heuristics, memory macro expansion, signal usage graphs, wiring constraints, and debug formatting.
9. **Blueprint generation (`emission/__init__.py`)** – Exposes `emit_blueprint_string` which serializes the Draftsman blueprint assembled by the emitter into the encoded string returned to callers.
10. **Shared limits (`signal_limits.py`)** – Centralizes physical constraints (e.g., maximum implicit signals) consumed by semantic analysis and lowering.

## Module Reference (per file)

### Root Package (`dsl_compiler/src`)
- `__init__.py` – Provides package-level exports so external callers can access parsing, semantic analysis, lowering, IR, and emission from a single import.
- `signal_limits.py` – Defines constants describing Factorio signal limitations used across semantic analysis and lowering.

### AST Package (`dsl_compiler/src/ast`)
- `__init__.py` – Re-exports all AST classes and visitors for convenient imports.
- `base.py` – Declares the `ASTNode` base class, visitor utilities, and serialization helpers.
- `expressions.py` – Contains expression node definitions (`BinaryOp`, `CallExpr`, `ProjectionExpr`, etc.).
- `statements.py` – Houses statement node classes (`DeclStmt`, `FuncDecl`, `ReturnStmt`, etc.).
- `literals.py` – Defines literal and lvalue nodes (`NumberLiteral`, `Identifier`, `PropertyAccess`, etc.).
- `types.py` – Provides lightweight type markers referenced during semantic analysis.

### Parsing Package (`dsl_compiler/src/parsing`)
- `__init__.py` – Bundles the public parsing API.
- `preprocessor.py` – Implements recursive import expansion prior to parsing.
- `parser.py` – Loads the grammar, configures the Lark parser, and attaches source metadata.
- `transformer.py` – Transforms parse trees into concrete AST objects.

### Semantic Package (`dsl_compiler/src/semantic`)
- `__init__.py` – Aggregates analyzer, diagnostic, and type definitions.
- `analyzer.py` – Executes symbol resolution, type inference, and warning generation.
- `diagnostics.py` – Defines structured diagnostics and their collectors.
- `symbol_table.py` – Implements hierarchical symbol tables and symbol records.
- `type_system.py` – Describes compiler-facing value and signal type metadata.
- `signal_allocator.py` – Manages implicit signal naming and mappings to Factorio channels.
- `validators.py` – Supplies helpers for source location rendering and optional explain-mode.

### IR Package (`dsl_compiler/src/ir`)
- `__init__.py` – Exposes IR nodes, builder, and optimizers.
- `nodes.py` – Defines IR node classes (`IR_Arith`, `IR_Decider`, `IR_MemWrite`, etc.) and signal references.
- `builder.py` – Provides utilities to create and annotate IR operations during lowering.
- `optimizer.py` – Runs structural optimizations such as common subexpression elimination and reference rewrites.

### Lowering Package (`dsl_compiler/src/lowering`)
- `__init__.py` – Re-exports the lowering façade.
- `lowerer.py` – Coordinates the lowering workflow, owns the IR builder, and delegates tasks.
- `expression_lowerer.py` – Lowers each expression form to IR, handles wire merges, and enforces projection rules.
- `statement_lowerer.py` – Processes statements, manages scope-mapped signal references, and tracks entity bindings.
- `memory_lowerer.py` – Generates IR for memory declarations, reads, writes, and once-only enable nets.
- `constant_folder.py` – Evaluates constant expressions and supplies compile-time folding services.

### Emission Package (`dsl_compiler/src/emission`)
- `__init__.py` – Provides high-level emission helpers (`emit_blueprint`, `emit_blueprint_string`) and re-exports supporting classes.
- `emitter.py` – Hosts `BlueprintEmitter`, the orchestrator for converting IR to a Draftsman blueprint.
- `entity_emitter.py` – Places specific Factorio entities (combinators, memory cells, user entities) based on IR operations.
- `signal_resolver.py` – Maps logical signal references to concrete Factorio signal identifiers and ensures registration.
- `connection_builder.py` – Plans and instantiates circuit wire connections, including relay insertion when necessary.
- `layout.py` – Computes spatial placement for entities and groups to keep wiring tractable.
- `memory.py` – Builds predesigned memory structures and helpers used by memory_lowerer output.
- `signals.py` – Maintains signal usage graphs and metadata for emission-time decisions.
- `wiring.py` – Contains wiring heuristics and utility routines shared by the connection builder.
- `debug_format.py` – Produces human-friendly descriptions of emitted entities for diagnostics.

### Miscellaneous
- `dsl_compiler/src/emission/__init__.py` plus exported helpers are leveraged by `compile.py` to gather the final blueprint string.

## Working With the Pipeline
- **Adjusting language syntax or AST** – Modify the grammar in `parsing/parser.py` and accompanying transformer, then update corresponding nodes in `ast/`.
- **Extending type rules** – Implement changes inside `semantic/analyzer.py`, adding any new metadata to `type_system.py` or `signal_allocator.py` as needed.
- **Introducing new IR operations** – Define them in `ir/nodes.py`, wire creation helpers in `ir/builder.py`, and make sure optimizations and emission understand the new constructs.
- **Tweaking lowering behavior** – Focus on the appropriate lowering helper (`expression_lowerer.py`, `statement_lowerer.py`, or `memory_lowerer.py`).
- **Changing physical blueprint output** – Update logic in the emission package; placement decisions largely live in `layout.py` and wiring constraints in `connection_builder.py` / `wiring.py`.

Use `compile.py --help` for available CLI flags, and refer to `tests/` for practical examples that cover parsing, lowering, and end-to-end emission.
