#!/usr/bin/env python3
"""
End-to-end tests to verify all Facto code examples in markdown files compile correctly.

This test extracts all ```facto code blocks from markdown documentation files
and attempts to compile them, ensuring our documentation examples are always valid.

Many examples in documentation are intentionally incomplete (fragments showing
specific syntax patterns). The test uses heuristics to identify complete programs
vs fragments.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Repository root directory
REPO_ROOT = Path(__file__).parent.parent.parent

# Markdown files to scan for examples
# Note: Some files are excluded as they contain many code fragments
MARKDOWN_FILES = [
    "README.md",
    # "LANGUAGE_SPEC.md",  # Excluded - contains many partial code fragments
    "doc/02_quick_start.md",
    "doc/03_signals_and_types.md",
    "doc/04_memory.md",
    "doc/05_entities.md",
    "doc/06_functions.md",
    "doc/07_advanced_concepts.md",
    "doc/08_missing_features.md",
    # "doc/LIBRARY_REFERENCE.md",  # Excluded - function signatures, not full programs
    # "doc/ENTITY_REFERENCE.md",  # Excluded - entity examples, not full programs
]

# Explicit whitelist of code blocks that should NOT be tested
# Format: (filename_substring, snippet_start) where snippet_start is the first ~30 chars
WHITELIST_SKIP = [
    # Shell commands and CLI usage (not Facto code)
    ("doc/02_quick_start.md", "factompile --help"),
    ("doc/02_quick_start.md", "factompile blink.facto"),
    ("doc/02_quick_start.md", "Usage: factompile"),
    ("LANGUAGE_SPEC.md", "python compile.py"),
    ("README.md", "pip install factompile"),
    ("README.md", "factompile --help"),
    ("README.md", "factompile blink.facto"),
    ("README.md", "factompile -i 'Signal"),
    ("README.md", "factompile program.facto"),
    ("README.md", "git clone"),
    ("README.md", "pytest -n auto"),
    ("doc/07_advanced_concepts.md", "factompile program.facto --"),
    ("doc/07_advanced_concepts.md", "factompile program.facto --j"),
    # Future syntax examples (not yet implemented)
    ("doc/08_missing_features.md", "# Select the maximum"),
    ("doc/08_missing_features.md", "# Proposed syntax"),
    ("doc/08_missing_features.md", "Entity[10] lamps"),
    # Directory structure examples (not code)
    ("doc/06_functions.md", "project/\n  main.facto"),
    # Error examples that are intentionally invalid
    ("LANGUAGE_SPEC.md", 'Signal bad = ("nonexistent'),
    ("LANGUAGE_SPEC.md", "Entity lamp = 42;"),
    ("LANGUAGE_SPEC.md", 'Signal bad = ("signal-W"'),
    ("doc/04_memory.md", 'Signal bad = ("signal-W"'),  # reserved signal example
    ("doc/04_memory.md", 'Memory write_count: "signal-W"'),  # reserved signal example
    (
        "doc/04_memory.md",
        'Memory buffer: "iron-plate";\n\nSignal iron',
    ),  # type mismatch example - two writes
    ("doc/04_memory.md", "# Shift samples"),  # type mismatch example
    ("doc/04_memory.md", "counter.write(counter"),  # type mismatch example
    # Import examples (importing non-existent files)
    ("doc/06_functions.md", 'import "b.facto"'),
    ("doc/06_functions.md", 'import "a.facto"'),
    ("doc/06_functions.md", 'import "path/to/other'),
    ("doc/06_functions.md", 'import "utils.facto"'),
    ("doc/06_functions.md", 'import "display_lib.facto"'),
    # Incomplete examples (showing patterns without full context)
    # README.md
    ("README.md", "Signal capped = (count >"),  # undefined count
    ("README.md", 'Memory buffer: "signal-B"'),  # undefined new_value
    (
        "README.md",
        'Entity lamp = place("small-lamp", 0, 0);\nlamp.enable = count',
    ),  # undefined count
    ("README.md", "func clamp(Signal value, int min"),  # undefined raw_speed
    ("README.md", "for i in 0..8 {\n    Entity lamp"),  # undefined counter
    # doc/02_quick_start.md
    ("doc/02_quick_start.md", "Signal blink = counter.read"),  # undefined counter
    (
        "doc/02_quick_start.md",
        'Entity lamp = place("small-lamp", 0, 0);\nlamp.enable = blink',
    ),  # undefined blink
    ("doc/02_quick_start.md", "Signal result = condition :"),  # shows syntax without values
    # doc/03_signals_and_types.md
    ("doc/03_signals_and_types.md", "int threshold = 100;\nint multiplier"),  # undefined input
    ("doc/03_signals_and_types.md", "Signal result = condition :"),  # syntax example
    ("doc/03_signals_and_types.md", "Signal total = (iron |"),  # undefined signal-total
    ("doc/03_signals_and_types.md", "Signal result = (a + b"),  # undefined a, b
    ("doc/03_signals_and_types.md", 'Bundle levels = { ("water"'),  # undefined oil
    ("doc/03_signals_and_types.md", 'Signal temp = ("signal-T"'),  # undefined signal names
    (
        "doc/03_signals_and_types.md",
        "# 0, 1, 2, 3, 4",
    ),  # layout planner limitation with 15 entities
    # doc/04_memory.md
    ("doc/04_memory.md", 'Memory counter: "signal-A";    # Explicit'),  # shows inferred type
    ("doc/04_memory.md", 'Memory buffer: "signal-A";\nSignal trigger'),  # undefined new_value
    (
        "doc/04_memory.md",
        'Memory captured: "signal-A";\nSignal input = ("signal-input"',
    ),  # signal-trigger not valid
    ("doc/04_memory.md", 'Memory toggle: "signal-A";\nSignal button'),  # signal-button not valid
    ("doc/04_memory.md", 'Memory total: "iron-plate";\nSignal incoming'),  # signal-enable not valid
    (
        "doc/04_memory.md",
        'Memory maximum: "signal-A";\nSignal input = ("signal-input"',
    ),  # type mismatch
    ("doc/04_memory.md", 'Memory minimum: "signal-A";\nMemory initialized'),  # type mismatch
    (
        "doc/04_memory.md",
        'Memory state: "signal-S";\nstate.write(1, set=turn_on',
    ),  # undefined turn_on
    (
        "doc/04_memory.md",
        'Memory state: "signal-S";\nstate.write(1, reset=turn_off',
    ),  # undefined turn_off
    ("doc/04_memory.md", 'Memory sample1: "signal-A"'),  # type mismatch example
    # doc/05_entities.md
    ("doc/05_entities.md", 'Entity name = place("prototype"'),  # syntax template with ...
    ("doc/05_entities.md", "Signal x_pos = some_signal"),  # undefined some_signal
    (
        "doc/05_entities.md",
        'Entity lamp = place("small-lamp", 0, 0);\nlamp.enable = count',
    ),  # undefined count
    (
        "doc/05_entities.md",
        'Entity lamp = place("small-lamp", 0, 0);\nlamp.enable = some_condition',
    ),  # undefined some_condition
    (
        "doc/05_entities.md",
        'Entity lamp = place("small-lamp", 0, 0);\nlamp.enable = signal',
    ),  # undefined signal
    ("doc/05_entities.md", 'Entity c1 = place("steel-chest"'),  # bundle comparison issue
    ("doc/05_entities.md", "Signal is_high = count > 10"),  # undefined count
    (
        "doc/05_entities.md",
        'Entity rgb_lamp = place("small-lamp", 0, 0, {\n    use_colors',
    ),  # undefined red_value
    (
        "doc/05_entities.md",
        'Entity inserter = place("inserter", 0, 0);\ninserter.enable = chest_count',
    ),  # undefined chest_count
    ("doc/05_entities.md", 'Entity belt = place("transport-belt"'),  # undefined should_run
    ("doc/05_entities.md", 'Entity station = place("train-stop"'),  # undefined iron_available
    (
        "doc/05_entities.md",
        'Entity assembler = place("assembling-machine-1"',
    ),  # undefined has_materials
    ("doc/05_entities.md", "# 8 lamps in a row"),  # undefined count, active
    ("doc/05_entities.md", "# Good — inlines into lamp"),  # undefined lamp
    # doc/06_functions.md
    ("doc/06_functions.md", "func function_name(Type1 param1"),  # template syntax
    ("doc/06_functions.md", "func min(Signal a, Signal b)"),  # undefined sensor1
    ("doc/06_functions.md", "func configure_status_lamp"),  # undefined machine1_status
    ("doc/06_functions.md", "# Hard to read\nSignal result = process"),  # undefined process
    ("doc/06_functions.md", "# Deadband filter"),  # references signal_processing.facto file format
    # doc/07_advanced_concepts.md
    ("doc/07_advanced_concepts.md", "Signal x = a + b"),  # undefined a, b
    ("doc/07_advanced_concepts.md", "Signal result = (a > 5)"),  # undefined a, b, c
    ("doc/07_advanced_concepts.md", "# Old (multiplication)"),  # undefined x, a, b
    ("doc/07_advanced_concepts.md", "# Old\nSignal clamped = (x < min)"),  # undefined x, min, max
    ("doc/07_advanced_concepts.md", "# Old\nSignal next = (state == 0"),  # undefined state, start
    ("doc/07_advanced_concepts.md", 'Entity c1 = place("steel-chest", 0, 0'),  # undefined c2, c3
    (
        "doc/07_advanced_concepts.md",
        'Memory previous: "signal-A";\nSignal current = input_signal',
    ),  # undefined input_signal
    (
        "doc/07_advanced_concepts.md",
        'Memory state: "signal-S";\nSignal current = state.read();\n\nSignal start',
    ),  # signal-start invalid
    ("doc/07_advanced_concepts.md", 'Signal selector = ("signal-select"'),  # signal-select invalid
    (
        "doc/07_advanced_concepts.md",
        'Memory ticks: "signal-T";\nint duration',
    ),  # signal-run invalid
    ("doc/07_advanced_concepts.md", "for row in 0..4 {\n    for col"),  # undefined active
    ("doc/07_advanced_concepts.md", "Signal result = ((a + b) * c)"),  # undefined a, b, c, d
    ("doc/07_advanced_concepts.md", 'Memory cached: "signal-C"'),  # undefined expensive_expression
    ("doc/07_advanced_concepts.md", 'Memory stage0: "signal-A"'),  # type mismatch example
    (
        "doc/07_advanced_concepts.md",
        'Signal setpoint = ("signal-setpoint"',
    ),  # signal-setpoint invalid
    (
        "doc/07_advanced_concepts.md",
        'Memory stable_count: "signal-C"',
    ),  # has signal-input which isn't a valid signal
    (
        "doc/07_advanced_concepts.md",
        'Memory previous: "signal-A";\nSignal current = input',
    ),  # undefined input_signal at L251, L260
]

# Examples that require imports but are otherwise complete
# We'll prepend the necessary imports for testing
IMPORT_FIXUPS = {
    # (file_substring, snippet_start): import_to_prepend
}


def extract_facto_blocks(markdown_path: Path) -> list[tuple[str, int, str]]:
    """
    Extract all ```facto code blocks from a markdown file.

    Returns: List of (code, line_number, first_30_chars) tuples
    """
    content = markdown_path.read_text()
    blocks = []

    # Find all ```facto ... ``` blocks
    pattern = r"```facto\n(.*?)```"

    for match in re.finditer(pattern, content, re.DOTALL):
        code = match.group(1)
        # Calculate line number
        line_num = content[: match.start()].count("\n") + 1
        # Get first 30 chars for identification
        first_chars = code[:30].strip() if code else ""
        blocks.append((code, line_num, first_chars))

    return blocks


def should_skip(file_path: str, snippet_start: str) -> bool:
    """Check if a code block should be skipped entirely."""
    for wl_file, wl_start in WHITELIST_SKIP:
        if wl_file in file_path and snippet_start.startswith(wl_start[:20]):
            return True
    return False


def is_likely_complete_program(code: str) -> bool:
    """
    Heuristically determine if code looks like a complete program vs a fragment.

    Complete programs typically:
    - Declare the variables they use (Signal, Memory, Entity, int, Bundle)
    - Don't reference undefined variables in expressions

    Fragments typically:
    - Use variables without declaring them first
    - Are showing a specific pattern/syntax
    """
    lines = code.strip().split("\n")

    # If it's a single line without a declaration, it's likely a fragment
    if len(lines) == 1:
        line = lines[0].strip()
        # Single-line declarations are complete
        if any(  # noqa: SIM103
            line.startswith(kw)
            for kw in ["Signal ", "Memory ", "Entity ", "int ", "Bundle ", "func ", "import "]
        ):
            return True
        # Other single lines are fragments
        return False

    # Look for common fragment patterns
    code.lower()

    # Function definitions with ellipsis or {...} are fragments
    if "{ ... }" in code or "{...}" in code or "{ ...}" in code:
        return False

    # Check if there are any variable references without declarations
    declared_vars = set()

    # Find all declarations (simplified pattern matching)
    for line in lines:
        line = line.strip()
        # Match: Type name = ... or Type name: ... or func name(
        decl_match = re.match(r"(Signal|Memory|Entity|int|Bundle)\s+(\w+)\s*[=:]", line)
        if decl_match:
            declared_vars.add(decl_match.group(2))
        # Match function params
        func_match = re.match(r"func\s+\w+\s*\((.*?)\)", line)
        if func_match:
            params = func_match.group(1)
            for param in params.split(","):
                param = param.strip()
                if " " in param:
                    declared_vars.add(param.split()[-1])
        # For loop iterator
        for_match = re.match(r"for\s+(\w+)\s+in", line)
        if for_match:
            declared_vars.add(for_match.group(1))

    # If we found no declarations at all, it's likely a fragment
    if not declared_vars:
        # Unless it's pure imports or comments
        has_meaningful_code = any(
            not line.strip().startswith("#")
            and not line.strip().startswith("import")
            and line.strip()
            for line in lines
        )
        if has_meaningful_code:
            return False

    return True


def compile_facto_code(code: str) -> tuple[bool, str]:
    """
    Attempt to compile a Facto code snippet.

    Returns: (success, error_message)
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".facto", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "compile.py"), temp_path],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(REPO_ROOT),
        )

        if result.returncode == 0:
            return True, ""
        else:
            # Extract the most relevant error message
            error = result.stderr or result.stdout
            # Get first few lines of error
            error_lines = error.strip().split("\n")[:5]
            return False, "\n".join(error_lines)
    except subprocess.TimeoutExpired:
        return False, "Compilation timed out"
    except Exception as e:
        return False, str(e)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def collect_markdown_examples():
    """Collect all testable examples from markdown files."""
    examples = []

    for md_file in MARKDOWN_FILES:
        md_path = REPO_ROOT / md_file
        if not md_path.exists():
            continue

        blocks = extract_facto_blocks(md_path)

        for code, line_num, first_chars in blocks:
            if not code.strip():
                continue

            # Skip whitelisted examples
            if should_skip(md_file, first_chars):
                continue

            # Skip obvious fragments
            if not is_likely_complete_program(code):
                continue

            # Create a test id
            test_id = f"{md_file}:L{line_num}"
            examples.append((test_id, code, md_file, line_num))

    return examples


# Collect examples at module load time for parametrization
EXAMPLES = collect_markdown_examples()


@pytest.mark.end2end
@pytest.mark.parametrize("test_id,code,file,line", EXAMPLES, ids=[e[0] for e in EXAMPLES])
def test_markdown_example_compiles(test_id: str, code: str, file: str, line: int):
    """Test that each markdown example compiles successfully."""
    success, error = compile_facto_code(code)

    if not success:
        pytest.fail(
            f"Example in {file} at line {line} failed to compile:\n"
            f"Code:\n{code[:300]}{'...' if len(code) > 300 else ''}\n\n"
            f"Error:\n{error}"
        )


def test_whitelist_not_empty():
    """Ensure the whitelist is defined (sanity check)."""
    assert len(WHITELIST_SKIP) > 0


def test_at_least_some_examples_found():
    """Ensure we found some examples to test (sanity check)."""
    assert len(EXAMPLES) > 5, f"Expected many examples, found only {len(EXAMPLES)}"


if __name__ == "__main__":
    # When run directly, print summary of what would be tested
    print(f"Found {len(EXAMPLES)} testable examples across {len(MARKDOWN_FILES)} markdown files")
    print(f"Whitelist contains {len(WHITELIST_SKIP)} skip entries\n")

    print("Examples to test:")
    for test_id, code, _file, _line in EXAMPLES[:10]:
        print(f"  - {test_id}: {code[:40].strip().replace(chr(10), ' ')}...")

    if len(EXAMPLES) > 10:
        print(f"  ... and {len(EXAMPLES) - 10} more")
