# Contributing to Factompiler

Welcome! This guide will help you understand how Factompiler works and how to contribute effectively.

## Development Setup

Clone the repository and set up your environment:

```bash
git clone https://github.com/your-username/Factompiler.git
cd Factompiler
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

Run the tests to make sure everything works:

```bash
pytest -n auto
```

The `-n auto` flag runs tests in parallel using all available CPU cores, which significantly speeds up the test suite.

## Understanding the Compiler

Factompiler transforms Facto source code into Factorio blueprint strings through a pipeline of distinct stages. Understanding this pipeline is key to working with the codebase.

### The Pipeline at a Glance

```
Source Code (.facto)
    ↓
[ Parsing ] → AST (Abstract Syntax Tree)
    ↓
[ Semantic Analysis ] → Validated AST with type information
    ↓
[ Lowering ] → IR (Intermediate Representation)
    ↓
[ Layout Planning ] → Physical entity placements and wire routes
    ↓
[ Blueprint Emission ] → Factorio blueprint string
```

Each stage has a clear responsibility and communicates with the next through well-defined data structures. Let's walk through them.

### Stage 1: Parsing

The parser lives in `dsl_compiler/src/parsing/` and has three parts:

**The Grammar** (`grammar/facto.lark`) defines the syntax using Lark's EBNF notation. When you add new syntax, this is where you start.

**The Preprocessor** (`preprocessor.py`) handles `import` statements before parsing, inlining the contents of imported files.

**The Transformer** (`transformer.py`) converts Lark's parse tree into our AST nodes. Each grammar rule has a corresponding method that creates the appropriate AST node.

The AST nodes themselves are defined in `src/ast/`. They're straightforward dataclasses representing expressions (`expressions.py`), statements (`statements.py`), and literals (`literals.py`). Every node tracks its source location for error messages.

### Stage 2: Semantic Analysis

The semantic analyzer (`dsl_compiler/src/semantic/analyzer.py`) walks the AST and performs validation:

- **Type checking**: Ensures operations make sense (you can't add a Signal to an Entity)
- **Symbol resolution**: Tracks variable definitions and scopes
- **Signal type inference**: Determines which Factorio signal type each value will use

The analyzer uses the visitor pattern—each AST node type has a `visit_NodeName` method. When you add a new AST node, you'll need to add corresponding visit methods here.

Type information is tracked in `type_system.py`. Key types include `SignalValue` (a circuit signal), `IntValue` (a constant), `EntityValue` (a placed entity), and `BundleValue` (a group of signals).

### Stage 3: Lowering

Lowering translates the high-level AST into a flat sequence of IR operations representing individual combinators. The code is in `dsl_compiler/src/lowering/`.

The main orchestrator is `lowerer.py`, but the heavy lifting happens in specialized modules:
- `expression_lowerer.py` handles expressions (arithmetic, comparisons, function calls)
- `statement_lowerer.py` handles statements (declarations, assignments)
- `memory_lowerer.py` handles memory operations (the tricky part involving feedback loops)
- `constant_folder.py` evaluates compile-time constants

The IR nodes (`src/ir/nodes.py`) represent individual combinator operations: `IRArithmetic` for arithmetic combinators, `IRDecider` for decider combinators, etc.

### Stage 4: Layout Planning

Given a list of IR operations, the layout planner figures out where to physically place entities and how to wire them together. This is in `dsl_compiler/src/layout/`.

`planner.py` coordinates the process:
1. `entity_placer.py` assigns grid positions to each combinator
2. `connection_planner.py` determines which entities need wire connections
3. `wire_router.py` optimizes wire paths using minimum spanning trees
4. `power_planner.py` adds power poles if requested

The result is a `LayoutPlan` containing placed entities and planned connections.

### Stage 5: Blueprint Emission

The emitter (`dsl_compiler/src/emission/emitter.py`) takes the layout plan and produces a Factorio blueprint. It uses the Draftsman library to create actual Factorio entities and serialize them to the blueprint string format.

`entity_emitter.py` handles the details of creating each entity type with the correct settings.

## Adding a New Feature

Here's the typical workflow when adding something new:

### Adding New Syntax

1. **Update the grammar** in `grammar/facto.lark`
2. **Add AST node(s)** in `src/ast/expressions.py` or `statements.py`
3. **Update the transformer** in `parsing/transformer.py` to create your new nodes
4. **Add semantic analysis** in `semantic/analyzer.py` for type checking
5. **Add lowering** in the appropriate lowerer module
6. **Write tests** - both unit tests and integration tests

### Adding a New Entity Property

1. Check if Draftsman already supports it (see the `factorio-draftsman` directory)
2. Update `emission/entity_emitter.py` to handle the property
3. Run `doc/generate_entity_docs.py` to regenerate ENTITY_REFERENCE.md
4. Add examples to the documentation

### Adding a New Library Function

1. Create or edit a `.facto` file in `lib/`
2. Document it in `doc/LIBRARY_REFERENCE.md`
3. Add an example program showing its use

## Code Style

We don't have formal style enforcement, but please follow the patterns you see in the existing code:

- Use type hints everywhere
- Keep functions focused and reasonably short
- Write docstrings for public methods
- Prefer clarity over cleverness

When writing tests:
- Unit tests go alongside the code they test (e.g., `src/semantic/tests/`)
- Integration tests go in `dsl_compiler/integration_tests/`
- Use descriptive test names that explain what's being tested

## Running Tests

```bash
# Run all tests in parallel
pytest -n auto

# Run tests for a specific module
pytest dsl_compiler/src/semantic/tests/ -n auto

# Run with coverage
pytest dsl_compiler/ --cov=dsl_compiler -n auto

# Run a specific test
pytest -k "test_conditional_value" -v
```

## Debugging Tips

**Print the AST**: The parser has a `--debug` mode that prints the parse tree.

**Check IR output**: Add print statements in `lowerer.py` to see the generated IR.

**Test in isolation**: Write a minimal `.facto` file that reproduces your issue, then step through the pipeline.

**Use the REPL pattern**: Create a small script in `temp/` that imports the compiler modules and lets you experiment interactively.

**Check Draftsman**: Many blueprint issues come from incorrect entity parameters. Test with Draftsman directly to isolate the problem.

## Where to Find Things

- **Grammar definition**: `dsl_compiler/grammar/facto.lark`
- **AST nodes**: `dsl_compiler/src/ast/`
- **Type system**: `dsl_compiler/src/semantic/type_system.py`
- **Entity definitions**: `dsl_compiler/src/common/entity_data.py`
- **Signal registry**: `dsl_compiler/src/common/signal_registry.py`
- **Compile entry point**: `compile.py`
- **Example programs**: `example_programs/`
- **Standard library**: `lib/`

## Questions?

If you're stuck or have questions:
1. Check the existing tests for examples of similar functionality
2. Look at how similar features are implemented
3. Open an issue describing what you're trying to do

Happy hacking!
