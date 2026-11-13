# Missing Features and Roadmap

This document outlines features that are planned but not yet implemented in Facto, and the current limitations of the language.

---

## Selector Combinator Syntax

The **selector combinator** was introduced in Factorio 2.0. It can:

- **Select input by value**: Sort signals ascending/descending and output the nth signal (e.g., highest, lowest, or by index)
- **Count inputs**: Output the count of unique input signals
- **Random input**: Pass through a random input signal at configurable intervals
- **Stack size**: Output the stack sizes of input items
- **Rocket capacity** (Space Age): Output rocket capacity of input signals
- **Quality filter** (Space Age): Filter signals by quality grade
- **Quality transfer** (Space Age): Apply quality grades to signals

### What Works Today

You can already use a selector combinator through the generic entity API: place it with `place()`, configure it with placement properties, wire signals into it with `.input`, and read the result with `.output`:

```facto
Bundle resources = { ("iron-plate", 100), ("copper-plate", 80) };

Entity selector = place("selector-combinator", 0, 0, {
    operation: "count",
    count_signal: "signal-C"
});
selector.input = resources;

Signal count = selector.output["signal-C"];
```

See [Entities](05_entities.md#wiring-signals-to-entities-input) and the [Entity Reference](ENTITY_REFERENCE.md#selectorcombinator) for all selector properties.

### What's Missing

There is no dedicated syntax yet. Future versions may offer functions like:

```facto
# Select the maximum value signal
Signal highest = select_max(bundle);

# Select by index (0 = first after sorting)
Signal third_highest = select_index(bundle, 2, descending=true);

# Count unique signals
Signal count = count_signals(bundle);

# Get stack sizes
Bundle stacks = stack_size(items);
```

---

## Entity Arrays

For loops already place many entities at once:

```facto
Memory counter: "signal-A";
counter.write((counter.read() + 1) % 10);

for i in 0..10 {
    Entity lamp = place("small-lamp", i, 0);
    lamp.enable = counter.read() > i;
}
```

What's missing is a way to keep references to those entities after the loop. A future feature would allow **entity arrays**:

```facto
# Proposed syntax - NOT YET IMPLEMENTED
Entity[10] lamps;
for i in 0..10 {
    lamps[i] = place("small-lamp", i, 0);
    lamps[i].enable = counter.read() > i;
}
```

This would enable:
- Configuring entities from a loop later in the program
- Dynamic indexing and configuration
- Passing groups of entities to functions

---

## Current Limitations

These restrictions apply to the current compiler:

- **No `if`/`else` statements.** Use conditional values (`(cond) : value`) to select between values.
- **Loop range bounds must be a number literal or a variable name.** `for i in 0..n` works, `for i in 0..n*2` is a syntax error — compute `int limit = n * 2;` first.
- **List iterators only accept number literals.** `for x in [1, 3, 5]` works; variables or expressions inside the list do not.
- **Function parameters are `int`, `Signal`, or `Entity` only.** There are no `Bundle` (or `Memory`) parameters, although functions can *return* a Bundle.
- **One write per memory.** Each memory cell can have exactly one `write()` call (see [Memory](04_memory.md#memory-type-rules)).
- **No quality support.** Signals and entities always use normal quality (see [GitHub issue #3](https://github.com/Snagnar/Factompiler/issues/3)).
- **`import "file" as name` is not supported.** Imports are always inlined into the global namespace.

---

## Other Planned Features

Additional features under consideration:

- **Display panel messages**: Display panels can be placed (`place("display-panel", 0, 0, {text: "hello"})`), but there is no syntax yet for circuit-driven messages
- **Interrupt signals**: Support for interrupt-based control flow
- **Blueprint books**: Generate blueprint books containing multiple related circuits
- **Import from blueprint**: Reverse-compile existing blueprints to Facto code for modification

The web-based compiler is already available: <https://snagnar.github.io/factompile/>.
