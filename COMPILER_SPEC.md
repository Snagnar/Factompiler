# Complete language design + parser & compiler plan — Factorio circuit DSL (definitive spec)

Below is a single, consistent, implementable specification you can hand to an engineer or another code-generating model to produce a parser and compiler. I thought carefully about every corner we discussed (signals as `(type,count)`, `|` projection, left-operand default for mixed types, bundle removal, `place`/entities, `mem` atomic, imports, layout, diagnostics, and a minimal set of primitives). This is meant to be complete and actionable.

---

# 1 — Design goals (short)

* Minimal atomic primitives (arithmetic, decider, constant, wire, `mem`, `place`) — everything else is library-level.
* Signals are first-class as `(type, count)` but DSL acts like integer arithmetic on counts by default.
* Focus on single-channel signals; multi-channel aggregation uses explicit projections (no bundle helper).
* `|` operator projects/forces result channel/type.

## Grammar Overview

```
program         ::= statement*

statement        ::= decl_stmt ";"
                 |  assign_stmt ";"
                 |  expr_stmt ";"
                 |  return_stmt ";"
                 |  import_stmt ";"
                 |  func_decl

decl_stmt        ::= type_name NAME "=" expr

type_name        ::= "int" | "Signal" | "SignalType" | "Entity" | "Memory" | "mem"

assign_stmt      ::= lvalue "=" expr
lvalue           ::= NAME ("." NAME)?

expr_stmt        ::= expr

return_stmt      ::= "return" expr

import_stmt      ::= "import" STRING ["as" NAME]

func_decl        ::= "func" NAME "(" [param_list] ")" "{" statement* "}"
param_list       ::= NAME ("," NAME)*

expr             ::= logic
logic            ::= comparison ( ("&&" | "||") comparison )*
comparison       ::= projection ( ("==" | "!=" | "<" | "<=" | ">" | ">=") projection )*
projection       ::= add ( "|" type_literal )*
add              ::= mul ( ("+" | "-") mul )*
mul              ::= unary ( ("*" | "/" | "%") unary )*
unary            ::= ("+" | "-" | "!") unary
                 |  primary

primary          ::= literal
                 |  signal_literal
                 |  "read" "(" NAME ")"
                 |  "write" "(" NAME "," expr ")"
                 |  dict_literal
                 |  NAME "(" [arglist] ")"
                 |  NAME "." NAME "(" [arglist] ")"
                 |  lvalue
                 |  "(" expr ")"

arglist          ::= expr ("," expr)*

dict_literal     ::= "{" [dict_item ("," dict_item)*] "}"
dict_item        ::= (STRING | NAME) ":" expr

signal_literal   ::= "(" type_literal "," expr ")"
                   |  NUMBER

type_literal     ::= STRING | NAME

literal          ::= STRING
```

### Memory


### Entities / place

* `place(proto, x, y, props)` creates an entity instance in the blueprint. Entities have *ports* and *properties*. `entity.property` read/write maps to wires or control_behavior fields as defined by entity descriptors.

---

**Note on `|` precedence:** it binds *looser* than `+/-` but *tighter* than comparisons. So `a + b | "iron"` parses as `(a + b) | "iron"`.

---

# 4 — Lexical tokens & suggestions

* IDENT: `[A-Za-z_][A-Za-z0-9_-]*` (allow hyphens for `iron-plate`)
* NUMBER: `-?[0-9]+`
* STRING: double quoted `"..."` (for types with characters)
* Symbols: `+ - * / % == != < <= >= && || = ( ) , . ; | { }`
* Comments: `#` single-line
* Keywords: `int Signal SignalType Entity Memory mem read write func return import`

---

# 5 — Operator precedence table (highest → lowest)

1. parentheses `( ... )`
2. property access `obj.prop`
3. function calls `f(...)`
4. unary `+ - !`
5. multiplicative `* / %`
6. additive `+ -`
7. projection `|`  (binds looser than `+`)
8. comparisons `== != < <= > >=`
9. logical `&& ||`
10. assignment `=`

---

# 6 — Static & dynamic typing (rules & inference)

### Value forms

* `Int` — integer literal or expression
* `Signal` — single channel `{ type: SignalType, count: Int }`

### Inference rules (summary)

* `read(mem)` → returns the mem’s `Signal` type
* Binary arithmetic `e1 OP e2`:

  * If both `Signal` and same `type` → result `Signal` with that `type` (wire-merge if both are simple sources).
  * If both `Signal` and different types → **default** result `Signal(type = e1.type)` (left-operand's type). Compiler emits **warning** unless user used projection or in strict mode.
  * If one is `Int` and other is `Signal` → coerce `Int` to `Signal` with the `Signal`'s type; operate on counts.
* `expr | TYPE` (projection):

  * If `expr` is `Signal` → create `Signal(TYPE, count(expr))` (count preserved).

---

# 7 — Mixed-type binary default (left-dominant) — rationale & UX

* **Rule**: `a + b` with `a.type != b.type` → result `Signal(type = a.type)` and combinator added to add counts with outputs into `a.type`.
* **UX safeguards**:

  * Emit a **warning** on mixed-type arithmetic: "mixed signal types in `a + b`: result will be `a.type`. Use `|` to explicitly set the output channel."
  * `--strict-types` flag turns warning into compile-time error.
  * Compiler `--map-signals` output lists chosen implicit types for debugging.

This gives a practical balance: ergonomic default, explicit escape hatches for correctness.

---

# 8 — Entities / place & properties

### `place(proto, x, y, props)`

* Creates an entity instance at coordinates `(x,y)` in generated blueprint.
* `proto` is a TYPE\_LITERAL (string like `"small-lamp"`, or a device alias).
* `props` is optional compile-time configuration dict (control\_behavior values).

### Entity descriptors

Each entity type has a descriptor (JSON-like) that defines:

* `proto_name` (Factorio prototype)
* `ports` (e.g., `circuit`: port coordinates relative to entity)
* `properties` map:

  * each property: `{ name, direction: in|out|inout, type: signal|int|bool, mapping }`
  * `mapping` explains how reads/writes map to wires or control\_behavior.

**Example** (lamp):

```json
{
  "proto_name": "small-lamp",
  "ports": { "circuit": { "pos":[0,0], "type":"circuit" } },
  "properties": {
    "enable": { "direction":"in", "type":"signal", "mapping": { "port":"circuit", "op":"write_signal" } }
  }
}
```

* `lamp.enable = expr` lowers to: compute `expr` (signal) → connect it to lamp's `circuit` port.

### Reading `entity.prop`

* `let x = lamp.arriving_train` returns a `Signal` or `Int` depending on `property` type. The compiler will insert combinators or read nodes to capture the port output and present it as a normal DSL `Signal`.

---

# 9 — Imports & modules

* `import "file"` loads another DSL module or a blueprint library file.
* Import semantics:

  * namespace import allowed: `import "lib.fcdsl" as lib`
  * imported file can define functions, entity descriptor registrations, or templates (e.g., `MemoryCell(width)`).
  * imports can also parse `.blueprint` or `.blueprint-book` (JSON) and expose functions `from_blueprint_page(name)` that instantiate those pages as templates.

---

# 10 — Functions / modules / scoping

* `func name(params) { ... }` defines reusable module.
* Functions are templates that can be `inline` or `instantiate`:

  * `inline` expands body into caller (no new entity instance id collisions).
  * `instantiate` creates a new module instance with fresh internal signal names and bounding box.
* Lexical scoping. Top-level declarations are module-global.

---

# 11 — AST node overview (suggested classes)

Implement a typed AST; these are the important node kinds reflected in `dsl_ast.py`:

* `Program` (root node of statement list)
* Statements: `DeclStmt`, `AssignStmt`, `MemDecl`, `ExprStmt`, `ReturnStmt`, `ImportStmt`, `FuncDecl`
* LValues: `Identifier`, `PropertyAccess`
* Expression nodes:

  * `NumberLiteral`, `StringLiteral`, `DictLiteral`, `SignalLiteral`
  * `IdentifierExpr`, `PropertyAccessExpr`
  * `CallExpr`, `PlaceExpr`, `ReadExpr`, `WriteExpr`
  * `BinaryOp`, `UnaryOp`, `ProjectionExpr`

---

# 12 — Intermediate Representation (IR) (must be explicit)

IR is what lowers to combinators & blueprint entities. Keep it small and canonical.

Key IR ops:

* `IR_Const(id, type?, value)` → constant combinator producing `(type, value)`
* `IR_Arith(id, op, left_signal, right_signal, out_type)` → instantiate arithmetic combinator if needed
* `IR_Decider(id, test, left_signal, right_value, out_signal_type, out_value_expr)`
* `IR_MemCreate(mem_id, initial_signal)` → memory module instance
* `IR_MemWrite(mem_id, data_signal, write_enable_signal)` → memory write port wiring
* `IR_MemRead(mem_id) -> signal_id`
* `IR_PlaceEntity(entity_id, proto, x, y, props)`
* `IR_EntityPropWrite(entity_id, propName, signal_id)`
* `IR_EntityPropRead(entity_id, propName) -> signal_id`
* `IR_ConnectToWire(signal_id, channel_type)` → connect source signal to channel or entity port
* `IR_Group(module_id, [child_ops], bounding_box)` → for placement

IR values: `SignalRef` (logical handle to `(type, count_expr)`).

---

# 13 — Lowering rules (concrete algorithms)

Below are the deterministic lowering rules the compiler must implement.

### 13.1 Numeric literal `42`

* If used in context requiring a `Signal` (coerce): create `IR_Const(constId, type=contextType, value=42)` (pool constants if same type/value).
* If used as raw Int: keep as Int.

### 13.3 `BinaryOp(op, left, right)`

* Lower left & right to `SignalRef` or Int.
* Case analysis:

  * Both `SignalRef` with same type T:

    * If both are *simple sources* (inputs/constants/entity outputs, no computation) → **wire-merge**: create no arithmetic combinator; record that both must feed channel T. Return `SignalRef(T, src = virtual_wire_T)`.
    * Otherwise (one or both need computation) → create `IR_Arith(..., out_type=T)` with op operating on counts; inputs connected; return `SignalRef(T, out_of_arith)`.
  * Both `SignalRef` with different types `T1, T2`:

    * Default result type `T1` (left). Emit **warning**.
    * Create `IR_Arith(..., left_signal, right_signal, out_type=T1)` — an arithmetic combinator set to `T1 = T1 + T2` semantics. Return `SignalRef(T1, id)`.
  * One is `Int`:

    * Coerce to `SignalRef` of the other operand’s type.
  * Both `Int`:

    * Standard integer arithmetic; result `Int`.

### 13.4 `Projection expr | TYPE`

* Lower `expr`.
* If `SignalRef(type == TYPE)` → no-op return same `SignalRef`.
* If `SignalRef` with different type:

  * Create `IR_Arith(op="+", left=expr, right=IR_Const(0,type=expr.type), out_type=TYPE)` (or simpler approach: arithmetic passthrough with output signal type set to TYPE). Return `SignalRef(TYPE, id)`.
### 13.5 `mem name = memory(init)`

* Lower `init` to SignalRef or Int; if Int convert to SignalRef with allocated implicit or user-specified type per rules.
* Create `IR_MemCreate(name, initial_signal)` and return mem reference.

### 13.6 `write(mem, expr)`

* Lower expr to `SignalRef` with a `type`.
* If `signal.type != mem.type`: coerce/`projection` automatically if possible (compiler warns); else error in strict mode.
* Emit `IR_MemWrite(mem, data_signal, write_enable=const_1)` or conditional if write gated.

### 13.7 Entity property writes/reads

* `place(...)` → `IR_PlaceEntity(...)`.
* `entity.prop = expr` → Lower `expr` to `SignalRef` or Int, then `IR_EntityPropWrite(entity, prop, signal)`.
* `x = entity.prop` → `IR_EntityPropRead(entity, prop)` which returns a `SignalRef` or Int. Lowering may require placing forwarding combinator if the entity port is not a direct combinator output.

### 13.9 Function instantiation

* On `func` calls that produce sub-circuit:

  * Inline if flagged inline (expand AST into caller).
  * Instantiate module: create fresh namespace, lower body into a `IR_Group(moduleId, ...)` and return handle(s) for outputs.

---

# 14 — Placement & layout (practical algorithm)

A shelving/greedy layout sufficient for first implementation:

1. **Modules and grouping**

   * Each function/module is a rectangular group. Internal combinators placed relative to group origin.
2. **Grid**

   * Use a grid (e.g., combinator cell size 1x1). Entities placed at integer coordinates.
3. **Port placement**

   * Each group exposes left-side input ports and right-side output ports (for clean wiring).
4. **Placement order**

   * Constants & inputs at leftmost column of module, computation nodes to the right in topological order.
5. **Wire routing**

   * Connect ports with straight green wires where possible. If collisions occur, shift group vertically.
6. **Poles**

   * Optionally place small power poles to satisfy entity power if required.

Heuristics for improvements:

* Merge identical constants into a constant comb pool at top-left.
* Localize frequently connected nodes (minimize long wires).
* Allow manual annotations from user for placements (advanced).

---

# 15 — Optimization & canonicalization passes

1. **Constant folding**: fold arithmetic on pure numeric constants.
2. **Common subexpression elimination**: within module, unify equal `IR_Arith` nodes.
3. **Constant pooling**: reuse `IR_Const(type,value)` to avoid duplicate constant combinators.
4. **Wire-merge elision**: remove unnecessary arithmetic combinators when the sum can be realized by simply wiring multiple sources to the same channel (only safe when types equal).
5. **Dead code elimination**: remove unreferenced entities or temporaries.

---

# 16 — Diagnostics & developer UX

* Map files: `source_location -> [IR_entity_ids]` so errors refer back to code lines.
* Warning examples:

  * Mixed-type arithmetic: show operand types and suggest applying `|` to target the intended output channel.
  * Implicit type allocation: list assigned implicit types.
  * Projections to unused channels: warn when `expr | "type"` produces a zeroed result because no source contributed to that channel.
* Flags:

  * `--strict-types`: mixed-type arithmetic is error.
  * `--no-warn-mixed` to silence particular warnings.
* Emit a compile-time `signal_map.json` that maps implicit names like `__v1` → `signal-A` (Factorio name) to help in-game debugging.

---

# 17 — Parser implementation plan (step-by-step)

**Recommended technology**: Lark (Python) for quick dev; ANTLR for multi-language or production. I’ll give the Lark-oriented plan but it’s translatable.

1. **Lexer + Grammar**: implement grammar above in Lark with operator precedence (LALR). Ensure `|` precedence set as described.
2. **AST transformer**: generate typed AST node objects from parse tree.
3. **Name-resolution & symbol table**:

  * Pass 1: record `DeclStmt`, `MemDecl`, `FuncDecl`, `ImportStmt`, `place` calls. Create scope stack for functions.
4. **Type-inference pass** (bottom-up):

  * Evaluate `infer_type(node)` returning `Signal`, `Int`, `Memory`, `Entity`, or `SignalType` depending on the expression or declaration.
   * Allocate implicit types when required (record mapping).
   * Emit warnings on mixed types.
5. **Lowering to IR**:

   * Walk AST and emit IR ops per rules in section 13.
   * Keep mapping from AST node → IR node list for diagnostics.
6. **Optimizations**: run canonicalization & optimizations.
7. **Placement**: group IR into modules and place entities.
8. **Template expansion**: instantiate combinator templates (arithmetic, decider, constants, memory subgraph if expanded).
9. **Serialize**: produce Factorio blueprint JSON and encode to string. Use an existing library to compress/encode.
10. **Output**: blueprint string + `signal_map.json` + diagnostics/warnings.

---

# 18 — Implementation skeleton (files & major classes)

Project layout (Python example):

```
/dsl_compiler
  /grammar
    fcdsl.lark
  /src
    parser.py         # uses Lark, outputs AST
    ast.py            # AST node classes
    semantic.py       # symbol table, type inference
    ir.py             # IR classes and builder
    lowerer.py        # AST -> IR lowering
    optim.py          # optimization passes
    layout.py         # placement engine
    templates.py      # combinator templates & entity descriptors
    emit.py           # blueprint JSON builder + encode (using Draftsman or own serialization)
    cli.py            # CLI, flags (strict, mapping), entrypoint
  /lib
    stdlib/           # standard library templates (memory, mux, pulse)
    devices/          # entity descriptors (lamp, train station)
  tests/
    ...               # unit & E2E tests
```

Key classes:

* `SymbolTable`, `TypeInfo`, `DiagnosticCollector`
* `IRBuilder`, `IRNode` subclasses
* `LayoutEngine`, `EntityPlacer`
* `BlueprintEmitter` (wraps Draftsman or builds JSON)

---

# 19 — Example: the 100-tick sampler (end-to-end)

Below is the DSL program (minimal, consistent with spec), and a concise lowering summary.

### DSL

```text
# every 100 ticks sample "signal-input" and if even output value*10
mem tick = memory(0);

let ONE = 1;

# increment tick every game tick (unconditional write)
write(tick, read(tick) + ONE);

let tick_mod = read(tick) % 100;
let sample_now = (tick_mod == 0);

Signal v = ("signal-input", 0);  # external bus placeholder
let rem = v % 2;
let is_even = (rem == 0);

let do_output = sample_now * is_even;    # 0/1

let scaled = v * 10;
let out_value = scaled * do_output;

output(out_value | "signal-output");  # explicit output channel
```

### Lowering highlights

1. `mem tick` → `IR_MemCreate(tick, init=IR_Const(type=__v_tick, value=0))`. `__v_tick` allocated implicitly.
2. `write(tick, read(tick) + ONE)`:

   * `read(tick)` → `SignalRef(type=__v_tick)`
   * `ONE` numeric coerced to same type -> `IR_Const(type=__v_tick, value=1)`
   * `IR_Arith(op="+", left=read(tick), right=ONE, out_type=__v_tick)` → arithmetic comb or optimized to feedback memory update per memory def.
   * `IR_MemWrite(tick, data_signal=arith_output, write_enable=const1)`
3. `tick_mod = read(tick) % 100`:

   * Lower `read(tick)` to `SignalRef(__v_tick)`
   * If `%` is not directly expressible on `Signal` counts? It's arithmetic: `IR_Arith(op="%", left=read(tick), right=IR_Const(type=__v_tick, value=100), out_type=__v_tick)` or better: do `%` on count, output can be same type or ephemeral type. (Compiler might prefer Int intermediate then project back.)
4. `sample_now = tick_mod == 0` → `IR_Decider(test: "==", left=tick_mod, right=0)` producing `SignalRef(type=virtual_bool_signal)` or `Int 0/1`.
5. Compose `is_even`, `do_output`, `scaled`, and gating to produce `out_value`.
6. `output(out_value | "signal-output")` → projection to explicit output type and `IR_ConnectToWire(out_valueSignal, "signal-output")`.
7. Optimize: fold constants, detect where simple wire-merge can be used, pool constants.
8. Placement: place memory module, combinators for mod/decider/arithmetic, connect wires, and produce blueprint.

(If memory is atomic, the compiler generates the canonical 4-combinator memory subgraph when `mem` is requested, or uses a black-box memory module that expects `DATA`/`WRITE`/`OUT` ports.)

---

# 20 — Test cases & unit tests to include (essential)

* Parse & AST shape:

  * `Signal a = ("signal-A", 0); Signal b = ("iron-plate", 0); Signal c = a + b;`
  * `a + b | "iron"`
  * `(a | "iron") + (b | "iron")` to confirm aggregation via projection
* Typing cases:

  * mixed-type `a + b` warns and chooses left type
  * `--strict-types` enforces error
* Lowering cases:

  * `same type` wire-merge (two simple sources on same channel) → no arithmetic combinator
  * `different type` arithmetic combinator produced
  * projection to an unused channel produces zero and emits warning in non-strict mode
* Integration:

  * Produce blueprint string, paste into Factorio (manual) — test sample program.

---

# 21 — Final recommendations & tradeoffs (blunt)

* Keep `mem` atomic initially (makes implementation simpler). Expand to combinator memory later.
* Stick with projection-based aggregation; removing the bundle helper keeps the type system predictable and simpler to reason about.
* Left-operand default is ergonomic — keep it, but *warn* and provide a strict mode.
* The hardest engineering portion: **placement/layout**. Solve the language and lowering first, then iterate on placement heuristics.
* Provide a good `signal_map` and compiler warnings — these will pay off hugely for debugging.

---

# 22 — Deliverables for a code-generating model

If you want this handed off to another model to implement parser + compiler, give them:

1. The grammar file `fcdsl.lark` following the grammar above (I can generate it).
2. AST class definitions (I can generate full class scaffolding).
3. Type inference algorithm pseudocode (I can generate runnable Python).
4. IR class skeleton + lowering rules (I can generate code).
5. Template library (JSON for arithmetic, decider, canonical memory pattern) and a mapping into Draftsman calls (I can generate).
6. A test-suite file list.

I can produce any of these artifacts next (e.g., the Lark grammar file + AST transformer + type inference pass in Python). Tell me which piece you want first and I’ll generate it.

---

Would you like me to:

* A) generate the Lark grammar + AST transformer code (Python), or
* B) generate the IR + lowering code skeleton (Python) that implements type inference & the mixed-type lowering rules, or
* C) create the JSON entity descriptors + canonical combinator templates and an example Draftsman emitter for the 100-tick sampler?

Pick one and I’ll produce runnable code next.
