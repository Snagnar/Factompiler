# Introduction to Facto

Welcome to **Facto**, a high-level programming language that compiles to Factorio circuit network blueprints!

## What is Facto?

Facto is a **domain-specific language** designed to make building complex circuit networks in Factorio accessible, maintainable, and fun. Instead of manually wiring hundreds of combinators together, you write code that describes *what* you want your circuit to do, and the **Factompiler** compiler figures out *how* to build it.

**Write this:**
```facto
Memory counter: "signal-A";
counter.write(counter.read() + 1);

Signal blink = (counter.read() % 10) < 5;

Entity lamp = place("small-lamp", 0, 0);
lamp.enable = blink;
```

**Get a working Factorio blueprint** that you can paste directly into your game!

> **[IMAGE PLACEHOLDER]**: Screenshot showing a blinking lamp circuit in Factorio, with the combinator layout visible.

## Why Use Facto?

### 🎯 **Focus on Logic, Not Wiring**
Traditional circuit building in Factorio requires you to:
- Manually place each combinator
- Remember which signal types go where
- Carefully wire inputs and outputs
- Debug visual spaghetti when things go wrong

With Facto, you describe the *behavior* you want, and the compiler handles placement, wiring, and signal routing automatically.

### 🔒 **Type Safety**
The compiler catches common mistakes before you paste anything into your game:
- Mixing incompatible signal types
- Using undefined variables
- Writing to memory with the wrong signal type

### 🔄 **Reusable Code**
Define functions once, use them everywhere. No more copy-pasting combinator setups:
```facto
func clamp(Signal value, int min_val, int max_val) {
    return (value < min_val) * min_val
         + (value > max_val) * max_val
         + ((value >= min_val) && (value <= max_val)) * value;
}

Signal safe_input = clamp(raw_input, 0, 100);
```

### 🚀 **Automatic Optimization**
The compiler applies optimizations that would be tedious to do by hand:
- Combines identical expressions (no duplicate combinators)
- Eliminates unnecessary projections
- Optimizes memory patterns to use fewer entities
- Automatically handles wire distance limits with relay poles

## Key Concepts at a Glance

### Signals
Signals are the lifeblood of Factorio circuits. In Facto, they're first-class values:

```facto
Signal iron = ("iron-plate", 100);  # Explicit type
Signal count = 42;                   # Compiler picks a type
Signal doubled = count * 2;          # Arithmetic just works
```

### Bundles
Group multiple signals together and operate on them as a unit:

```facto
Bundle resources = { ("iron-plate", 100), ("copper-plate", 80) };
Bundle doubled = resources * 2;           # Multiply all at once
Signal iron = resources["iron-plate"];    # Extract specific signal
Signal anyLow = any(resources) < 50;      # Check across all signals
```

### Memory
Need to store state across ticks? Declare memory:

```facto
Memory counter: "signal-A";
counter.write(counter.read() + 1);  # Increment every tick
```

### Entities
Place and control Factorio entities with circuit signals:

```facto
Entity lamp = place("small-lamp", 5, 0);
lamp.enable = count > 50;  # Turn on when count exceeds 50
```

### For Loops
Create multiple entities or repeat patterns efficiently:

```facto
for i in 0..8 {
    Entity lamp = place("small-lamp", i * 2, 0);
    lamp.enable = counter == i;  # Chaser effect
}
```

### Functions
Group reusable logic into functions:

```facto
func make_lamp(int x, int y, Signal trigger) {
    Entity lamp = place("small-lamp", x, y);
    lamp.enable = trigger > 0;
    return lamp;
}
```

## The Compilation Pipeline

When you compile a Facto program, it goes through several stages:

```
┌─────────────────┐
│  Your Code      │  (.facto file)
└────────┬────────┘
         ▼
┌─────────────────┐
│  Parser         │  Turns text into a syntax tree
└────────┬────────┘
         ▼
┌─────────────────┐
│  Type Checker   │  Validates types and signal usage
└────────┬────────┘
         ▼
┌─────────────────┐
│  IR Generator   │  Creates intermediate representation
└────────┬────────┘
         ▼
┌─────────────────┐
│  Optimizer      │  Eliminates redundancy
└────────┬────────┘
         ▼
┌─────────────────┐
│  Layout Engine  │  Places entities on the grid
└────────┬────────┘
         ▼
┌─────────────────┐
│  Wire Router    │  Connects everything with wires
└────────┬────────┘
         ▼
┌─────────────────┐
│  Blueprint      │  Base64-encoded Factorio blueprint
└─────────────────┘
```

The result is a blueprint string you can import directly into Factorio!

## Getting Started

Ready to build your first circuit? Head to the **[Quick Start Guide](02_quick_start.md)** to create your first blinking lamp!

## Documentation Map

| Chapter | What You'll Learn |
|---------|-------------------|
| [Quick Start](02_quick_start.md) | Installation, first program, basic workflow |
| [Signals and Types](03_signals_and_types.md) | Signal types, arithmetic, bundles, for loops, type inference |
| [Memory](04_memory.md) | Storing state, counters, latches |
| [Entities](05_entities.md) | Placing entities, controlling them with circuits |
| [Functions](06_functions.md) | Reusable code, imports, modules |
| [Advanced Concepts](07_advanced_concepts.md) | Optimizations, patterns, debugging |

## Reference Documentation

- **[Language Specification](../LANGUAGE_SPEC.md)** - Complete language reference
- **[Entity Reference](ENTITY_REFERENCE.md)** - All entities and their properties

## Requirements

- **Python 3.8+**
- **Factorio 2.0** (for importing blueprints)
- The `factorio-draftsman` library (installed automatically with requirements)

## Community & Support

Found a bug? Have a feature request? Check out the [project repository](https://github.com/Snagnar/Factompiler) on GitHub!

---

**Next:** [Quick Start Guide →](02_quick_start.md)
