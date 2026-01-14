<p align="center">
  <img src="doc/img/facto-logo-placeholder.png" alt="Facto Logo" width="200"/>
</p>

<h1 align="center">Facto</h1>

<p align="center">
  <strong>A programming language that compiles to Factorio circuit network blueprints</strong>
</p>

<p align="center">
  <a href="https://github.com/Snagnar/Factompiler/actions/workflows/ci.yml"><img src="https://github.com/Snagnar/Factompiler/actions/workflows/ci.yml/badge.svg" alt="CI Pipeline"/></a>
  <a href="https://codecov.io/gh/Snagnar/Factompiler"><img src="https://codecov.io/gh/Snagnar/Factompiler/branch/main/graph/badge.svg" alt="codecov"/></a>
  <a href="https://pypi.org/project/factompile/"><img src="https://img.shields.io/pypi/v/factompile.svg" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/factompile/"><img src="https://img.shields.io/pypi/pyversions/factompile.svg" alt="Python versions"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
</p>

<p align="center">
  <a href="#-quick-example">Quick Example</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-documentation">Documentation</a> •
  <a href="#-features">Features</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 🎮 What is Facto?

**Facto** is a high-level programming language designed specifically for building complex Factorio circuit networks. Instead of manually wiring hundreds of combinators together, you write clean, readable code — and the Facto compiler generates an optimized blueprint you can paste directly into your game.

<p align="center">
  <img src="doc/img/header_bp.png" alt="Facto Intro Screenshot" width="1200"/>
</p>

**Write this:**

```facto
# A blinking lamp that cycles every 20 ticks
Memory counter: "signal-A";
counter.write((counter.read() + 1) % 20);

Signal blink = counter.read() < 10;

Entity lamp = place("small-lamp", 0, 0);
lamp.enable = blink;
```

**Get a working Factorio blueprint** → Import it and watch your lamp blink! ✨

<!-- [IMAGE PLACEHOLDER: Side-by-side showing the code above and the resulting blueprint placed in Factorio] -->

---

## 🚀 Quick Example

Here's a fun example — an **LED chaser display** (Knight Rider style) where a light bounces back and forth:

```facto
# LED Chaser - a light that bounces back and forth across 8 lamps

int NUM_LAMPS = 8;

# Timer counts 0-13 for a complete back-and-forth cycle
Memory tick: "signal-T";
tick.write((tick.read() + 1) % 14);

# Convert to bounce position: 0→7→0 pattern using arithmetic
Signal t = tick.read();
Signal position = (t < 8) * t + (t >= 8) * (14 - t);

# Create the lamp row - each lamp lights when position matches
for i in 0..NUM_LAMPS {
    Entity lamp = place("small-lamp", i, 0);
    lamp.enable = position == i;
}
```

This example demonstrates:
- **For loops** — create 8 lamps with a single loop
- **Memory** — persistent counter that survives across game ticks
- **Arithmetic conditions** — `(t < 8) * t` acts as an inline conditional

**Compile it:**

```bash
factompile led_chaser.facto -o chaser.blueprint
```

Import the blueprint, add power, and watch the light chase! ✨

<!-- [IMAGE PLACEHOLDER: The LED chaser in Factorio showing the bouncing light effect] -->

---

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install factompile
```

### From Source

```bash
git clone https://github.com/Snagnar/Factompiler.git
cd Factompiler
pip install -e .
```

### Verify Installation

```bash
factompile --help
```

---

## 📖 Documentation

| Guide | Description |
|-------|-------------|
| **[🚀 Quick Start](doc/02_quick_start.md)** | Get your first circuit running in 5 minutes |
| **[📘 Introduction](doc/01_introduction.md)** | Understand what Facto is and why it exists |
| **[📊 Signals & Types](doc/03_signals_and_types.md)** | Learn the type system, bundles, and operations |
| **[💾 Memory](doc/04_memory.md)** | Counters, latches, and state management |
| **[🏗️ Entities](doc/05_entities.md)** | Placing and controlling Factorio entities |
| **[🔧 Functions](doc/06_functions.md)** | Reusable code and module imports |
| **[⚡ Advanced Concepts](doc/07_advanced_concepts.md)** | Optimizations, patterns, and debugging |
| **[📋 Entity Reference](doc/ENTITY_REFERENCE.md)** | Complete list of all entities and properties |
| **[📜 Language Specification](LANGUAGE_SPEC.md)** | Formal language reference |

---

## ✨ Features

### 🎯 Focus on Logic, Not Wiring
Write what your circuit should *do*, not how to wire it. The compiler handles entity placement, signal routing, and wire connections automatically.

### 🔒 Type Safety
Catch signal type mismatches, undefined variables, and other errors at compile time — before you paste anything into your game.

### ⚡ Automatic Optimizations
- **Common Subexpression Elimination** — Identical expressions share combinators
- **Condition Folding** — Chains of comparisons become single multi-condition deciders
- **Wire Merge** — Same-type additions skip arithmetic combinators entirely
- **Memory Optimization** — Simple counters use efficient feedback loops

### 🧩 Reusable Functions
Define logic once, use it everywhere:

```facto
func clamp(Signal value, int min, int max) {
    return (value < min) * min 
         + (value > max) * max 
         + ((value >= min) && (value <= max)) * value;
}

Signal safe_speed = clamp(raw_speed, 0, 100);
```

### 📦 Bundles for Parallel Operations
Operate on multiple signals at once using Factorio's "each" signal:

```facto
Bundle resources = { ("iron-plate", 0), ("copper-plate", 0), ("coal", 0) };
Signal anyLow = any(resources) < 100;  # True if any resource is low
Bundle scaled = resources * 2;          # Double all values at once
```

### 🔄 Compile-Time Loops
Generate multiple entities or repeated logic with for loops:

```facto
for i in 0..8 {
    Entity lamp = place("small-lamp", i * 2, 0);
    lamp.enable = counter == i;  # Chaser effect
}
```

---

## 🛠️ CLI Usage

```
Usage: factompile [OPTIONS] INPUT_FILE

  Compile Facto source files to Factorio blueprints.

Options:
  -o, --output PATH               Save blueprint to file (default: stdout)
  --name TEXT                     Blueprint name (default: from filename)
  --log-level [debug|info|warning|error]
                                  Set logging verbosity
  --no-optimize                   Disable optimizations
  --power-poles [small|medium|big|substation]
                                  Add power poles to blueprint
  --json                          Output raw JSON instead of encoded string
  --help                          Show this message and exit.
```

### Examples

```bash
# Compile and print to terminal
factompile my_circuit.facto

# Save to file with custom name
factompile my_circuit.facto -o circuit.blueprint --name "My Awesome Circuit"

# Add power poles and see debug output
factompile my_circuit.facto --power-poles medium --log-level debug

# Export as JSON for inspection
factompile my_circuit.facto --json | jq '.blueprint.entities | length'
```

---

## 🎨 Example Programs

The [`example_programs/`](example_programs/) directory contains many working examples:

| Example | Description |
|---------|-------------|
| `01_basic_arithmetic.facto` | Simple signal arithmetic and type projection |
| `03_blinker.facto` | Classic blinking lamp with memory |
| `04_memory.facto` | Counters, latches, and state patterns |
| `05_entities.facto` | Placing and controlling various entities |
| `06_lamp_array.facto` | LED bar graphs and displays |
| `07_colored_lamp.facto` | RGB color cycling and effects |
| `08_functions.facto` | Reusable functions and modules |

<!-- [IMAGE PLACEHOLDER: A gallery of 4 screenshots showing different example circuits in Factorio] -->

---

## 🧪 Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests (parallel execution)
pytest -n auto

# Run only unit tests (faster)
pytest -m "not end2end" -n auto

# Run with coverage
pytest -m "not end2end" --cov=dsl_compiler --cov-report=html -n auto
```

### Code Quality

```bash
# Lint and format
ruff check .
ruff format .

# Type checking
mypy dsl_compiler/ compile.py --ignore-missing-imports
```

---

## 🤝 Contributing

Contributions are welcome! Whether it's:
- 🐛 Bug reports and fixes
- ✨ New features and improvements
- 📖 Documentation updates
- 🧪 Additional test cases

Please check out the [GitHub Issues](https://github.com/Snagnar/Factompiler/issues) for open tasks or create a new issue to discuss your ideas.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[factorio-draftsman](https://github.com/redruin1/factorio-draftsman)** — The excellent Python library that makes blueprint generation possible
- **The Factorio community** — For endless inspiration and incredible circuit creations
- **Wube Software** — For creating the best factory game ever made

---

<p align="center">
  <strong>Happy automating! 🏭</strong>
</p>

<p align="center">
  <sub>The factory must grow.</sub>
</p>
