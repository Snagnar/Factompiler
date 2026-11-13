# Changelog

All notable changes to Factompile / the Facto language are listed here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] – unreleased

The first release since 0.1.6 (2026-01-27). Most of these features were
needed to express a real-world project (a multi-module logistic train
request system).

### Added – language

- **Bundle filtering.** `(bundle > 0) : bundle` keeps only the signals that
  pass the comparison, with their values. `(bundle > 0) : 1` outputs 1 for
  every signal that passes. All comparison operators work.
- **Bundle–bundle arithmetic.** `requests - delivered` works element-wise
  (each signal against the signal of the same name) for `+ - * / % ** << >>
  AND OR XOR`.
- **Wire colour pinning.** `sig.wire = red;` / `bundle.wire = green;` forces
  a signal or bundle onto a specific wire colour. It works for computed
  values too. This is how a module connects to an existing red/green bus.
- **Wiring bundles into entities.** Assigning a bundle to an entity
  property, e.g. `selector.input = items;`, wires the bundle into the
  entity. For combinators the wire goes to the input side. In 0.1.6 the
  same code compiled, but the wire was silently left out.
- **Resettable accumulator.** `mem.write(mem.read() + x, reset=cond);`
  is a fourth memory write mode: the counter adds `x` every tick and is
  cleared while `cond` is non-zero. It cannot be combined with `when=`.
- **Trailing commas** are allowed in bundle/dict literals, argument and
  parameter lists, `for` lists, signal tuples and `set=/reset=` arguments.

### Changed

- **A bare bundle comparison is now an error.** `Bundle x = b > 0;` was
  ambiguous. Use `any(b) > 0` / `all(b) > 0` for a yes/no answer, or the
  filter syntax `(b > 0) : b` to keep matching signals.
- **Error messages now say where the problem is**:
  `ERROR: [semantic] main.facto:12:5: Undefined variable 'x'`. Syntax
  errors show the line with a caret and a hint, e.g. a missing `;`.
  The CLI no longer prints Python tracebacks for user errors. Pass
  `--log-level debug` to get one for internal errors.
- Line numbers stay correct in programs that use `import` (imports used to
  shift every later line), and errors inside an imported file name that file.
- `compile.py` is now a thin wrapper around the installed `factompile`
  command, so both behave identically (including `--input "<source>"`).
- Depends on `factorio-draftsman>=4.0.1,<5` from PyPI instead of a git fork.

### Fixed

- Layout no longer fails with "Failed to find feasible layout" for larger
  programs or programs with entities placed far from the origin. Six of the
  bundled example programs were affected.
- The standard library (`lib/math.facto`, `lib/memory_patterns.facto`,
  `lib/signal_processing.facto`) now ships inside the package, so
  `import "lib/math.facto";` works after `pip install factompile`.
- A missing import is reported as a normal error with the importing file
  and line, not as an internal error.
- Test packages are no longer installed with the wheel.
- **Memories inside functions are now separate per call.** Calling a
  function that declares a `Memory` twice used to produce two cells that
  were wired into each other. Two functions with a same-named memory also
  mixed up their signal types (this affected `lib/memory_patterns.facto`).
- **Constant arithmetic matches the game**: 32-bit wrap-around, `/` rounds
  toward zero, `%` keeps the sign of the left operand, `x / 0 == 0`.
  Integer literals outside the 32-bit range are an error. Hex, octal and
  binary literals up to 32 bits are bit patterns (`0xFFFFFFFF == -1`).
- `lamp.r = signal;` (also `.g`, `.b`) now sets the lamp's colour signal.
  Before, it only wired the signal in, and the lamp kept reading
  `signal-red`.
- Unknown entity names, `place()` options and entity properties
  (`lamp.enabel = ...`) now produce a warning with a "did you mean"
  suggestion. Before, they were silently dropped.
- `entity.enable = bundle < 0;` is a clear error (use `any()`/`all()`)
  instead of a crash.
- Library functions called with constant arguments (e.g.
  `priority_encode(1, 0, 1, 0)`) no longer crash.
- All documentation examples were checked against the compiler and fixed.
  Undocumented features are now documented: the resettable accumulator,
  the one-write-per-memory rule, trailing commas, `.input` wiring and the
  import search path.

## [0.1.6] – 2026-01-27

Last release before this changelog was started.
