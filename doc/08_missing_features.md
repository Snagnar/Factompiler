# Missing Features and Roadmap

This document outlines features that are planned but not yet implemented in Facto.

---

## Selector Combinator Support

The **selector combinator** was introduced in Factorio 2.0 and provides advanced signal filtering capabilities that Facto does not currently support. This combinator can:

- **Select input by value**: Sort signals ascending/descending and output the nth signal (e.g., highest, lowest, or by index)
- **Count inputs**: Output the count of unique input signals
- **Random input**: Pass through a random input signal at configurable intervals
- **Stack size**: Output the stack sizes of input items
- **Rocket capacity** (Space Age): Output rocket capacity of input signals
- **Quality filter** (Space Age): Filter signals by quality grade
- **Quality transfer** (Space Age): Apply quality grades to signals

### Potential Facto Syntax

Future versions may support selector combinators with syntax like:

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

Currently, entities must be placed individually:

```facto
Entity lamp1 = place("small-lamp", 0, 0);
Entity lamp2 = place("small-lamp", 1, 0);
Entity lamp3 = place("small-lamp", 2, 0);
```

A future feature would allow **entity arrays** for dynamic placement using for-loops:

```facto
# Proposed syntax - NOT YET IMPLEMENTED
Entity[10] lamps;
for i in 0..10 {
    lamps[i] = place("small-lamp", i, 0);
    lamps[i].enable = counter.read() > i;
}
```

This would enable:
- Creating rows/grids of entities with a single loop
- Dynamic indexing and configuration
- More concise blueprints for repetitive structures

---

## Web-Based Compiler Platform

A web-based version of the Facto compiler is planned, which would allow:

- **In-browser compilation**: Write and compile Facto code without installing anything
- **Blueprint preview**: Visual preview of the generated circuit layout
- **Interactive examples**: Try sample programs directly in the browser
- **Sharing**: Generate shareable links to Facto programs

This will lower the barrier to entry for new users and enable quick experimentation without local setup.

---

## Other Planned Features

Additional features under consideration:

- **Display panel support**: Control display panels for in-game text output
- **Interrupt signals**: Support for interrupt-based control flow
- **Blueprint books**: Generate blueprint books containing multiple related circuits
- **Import from blueprint**: Reverse-compile existing blueprints to Facto code for modification
