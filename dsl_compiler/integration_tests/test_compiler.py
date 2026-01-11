"""
Integration tests for compile.py - the main compiler entry point.

These tests verify that the compiler correctly orchestrates all pipeline stages.
"""

import os

# Import the main compiler function
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from compile import compile_dsl_file

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestCompileDslFile:
    """Tests for compile_dsl_file function."""

    def test_compile_basic_arithmetic(self):
        """Test compiling a basic arithmetic program."""
        success, result, messages = compile_dsl_file(FIXTURES_DIR / "basic_arithmetic.facto")
        assert success, f"Compilation failed: {messages}"
        assert result.startswith("0")  # Factorio blueprint starts with 0

    def test_compile_with_name(self):
        """Test compiling with a custom blueprint name."""
        success, result, messages = compile_dsl_file(
            FIXTURES_DIR / "basic_arithmetic.facto", program_name="Custom Name"
        )
        assert success

    def test_compile_nonexistent_file(self):
        """Test compiling a file that doesn't exist."""
        success, result, messages = compile_dsl_file(Path("/nonexistent/path/file.facto"))
        assert not success
        assert "does not exist" in result

    def test_compile_with_optimizations_disabled(self):
        """Test compiling with optimizations disabled."""
        success, result, messages = compile_dsl_file(
            FIXTURES_DIR / "basic_arithmetic.facto", optimize=False
        )
        assert success

    def test_compile_with_json_output(self):
        """Test compiling with JSON output format."""
        success, result, messages = compile_dsl_file(
            FIXTURES_DIR / "basic_arithmetic.facto", use_json=True
        )
        assert success
        # JSON output should be a valid JSON string
        import json

        parsed = json.loads(result)
        assert "blueprint" in parsed

    def test_compile_with_power_poles(self):
        """Test compiling with power poles enabled."""
        success, result, messages = compile_dsl_file(
            FIXTURES_DIR / "basic_arithmetic.facto", power_pole_type="medium"
        )
        assert success

    def test_compile_bundle_program(self):
        """Test compiling a bundle-based program."""
        success, result, messages = compile_dsl_file(FIXTURES_DIR / "bundle.facto")
        assert success

    def test_compile_memory_program(self):
        """Test compiling a memory-based program."""
        success, result, messages = compile_dsl_file(FIXTURES_DIR / "memory_conditional.facto")
        assert success

    def test_compile_function_program(self):
        """Test compiling a program with functions."""
        success, result, messages = compile_dsl_file(FIXTURES_DIR / "function.facto")
        assert success

    def test_compile_for_loop_program(self):
        """Test compiling a program with for loops."""
        success, result, messages = compile_dsl_file(FIXTURES_DIR / "for_loop.facto")
        assert success

    def test_compile_entity_program(self):
        """Test compiling a program with entities."""
        success, result, messages = compile_dsl_file(FIXTURES_DIR / "entity.facto")
        assert success


class TestCompileErrors:
    """Tests for compile error handling."""

    def test_syntax_error_raises_exception(self):
        """Test that syntax errors raise an exception."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".facto", delete=False) as f:
            f.write("Signal x = ;")  # Invalid syntax
            temp_path = Path(f.name)

        try:
            # Compile should raise SyntaxError for parse errors
            with pytest.raises(SyntaxError):
                compile_dsl_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_semantic_error_raises_exception(self):
        """Test that semantic errors raise an exception."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".facto", delete=False) as f:
            f.write("Signal x = undefined_var;")  # Undefined variable
            temp_path = Path(f.name)

        try:
            # Compile should raise Exception for semantic errors, ruff ignore assert blank exception
            with pytest.raises(Exception):  # noqa: B017
                compile_dsl_file(temp_path)
        finally:
            os.unlink(temp_path)


class TestCompilePowerPoles:
    """Tests for power pole options."""

    def test_compile_with_small_poles(self):
        """Test compiling with small power poles."""
        success, result, messages = compile_dsl_file(
            FIXTURES_DIR / "basic_arithmetic.facto", power_pole_type="small"
        )
        assert success

    def test_compile_with_big_poles(self):
        """Test compiling with big power poles."""
        success, result, messages = compile_dsl_file(
            FIXTURES_DIR / "basic_arithmetic.facto", power_pole_type="big"
        )
        assert success

    def test_compile_with_substation(self):
        """Test compiling with substations."""
        success, result, messages = compile_dsl_file(
            FIXTURES_DIR / "basic_arithmetic.facto", power_pole_type="substation"
        )
        assert success
