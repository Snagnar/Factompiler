"""
Tests for compile.py - the main compile script.

Tests the compile_dsl_file function and CLI for file-based compilation.
"""

import json

import click
import pytest
from click.testing import CliRunner

from compile import compile_dsl_file, main, setup_logging, validate_power_poles


class TestCompileDslFile:
    """Tests for the compile_dsl_file function."""

    @pytest.fixture
    def simple_facto_file(self, tmp_path):
        """Create a simple .facto file."""
        file = tmp_path / "simple.facto"
        file.write_text("Signal x = 42;", encoding="utf-8")
        return file

    @pytest.fixture
    def complex_facto_file(self, tmp_path):
        """Create a complex .facto file with memory and entities."""
        file = tmp_path / "complex.facto"
        file.write_text(
            """
Memory counter: "signal-A";
counter.write((counter.read() + 1) % 60);
Entity lamp = place("small-lamp", 0, 0);
lamp.enable = counter.read() < 30;
        """,
            encoding="utf-8",
        )
        return file

    def test_simple_compile(self, simple_facto_file):
        """Basic file compiles successfully."""
        success, result, messages = compile_dsl_file(simple_facto_file)
        assert success is True
        assert result.startswith("0")  # Blueprint string

    def test_complex_compile(self, complex_facto_file):
        """Complex file with memory and entities compiles."""
        success, result, messages = compile_dsl_file(complex_facto_file)
        assert success is True
        assert result.startswith("0")

    def test_nonexistent_file(self, tmp_path):
        """Non-existent file returns error."""
        fake_file = tmp_path / "nonexistent.facto"
        success, result, messages = compile_dsl_file(fake_file)
        assert success is False
        assert "does not exist" in result

    def test_custom_program_name(self, simple_facto_file):
        """Custom program name is used."""
        success, result, messages = compile_dsl_file(simple_facto_file, program_name="Custom Name")
        assert success is True

    def test_json_output(self, simple_facto_file):
        """JSON output is valid."""
        success, result, messages = compile_dsl_file(simple_facto_file, use_json=True)
        assert success is True
        data = json.loads(result)
        assert "blueprint" in data

    def test_optimization_disabled(self, simple_facto_file):
        """Compiles with optimization disabled."""
        success, result, messages = compile_dsl_file(simple_facto_file, optimize=False)
        assert success is True

    def test_power_poles(self, complex_facto_file):
        """Compiles with power poles."""
        success, result, messages = compile_dsl_file(complex_facto_file, power_pole_type="medium")
        assert success is True

    def test_different_log_levels(self, simple_facto_file):
        """Different log levels work."""
        for level in ["debug", "info", "warning", "error"]:
            success, result, messages = compile_dsl_file(simple_facto_file, log_level=level)
            assert success is True


class TestValidatePowerPoles:
    """Tests for power pole validation."""

    def test_valid_types(self):
        """All valid types work."""
        for pole_type in ["small", "medium", "big", "substation"]:
            result = validate_power_poles(None, None, pole_type)
            assert result == pole_type

    def test_case_insensitive(self):
        """Validation is case insensitive."""
        result = validate_power_poles(None, None, "BIG")
        assert result == "big"

    def test_none_returns_none(self):
        """None value returns None."""
        result = validate_power_poles(None, None, None)
        assert result is None

    def test_invalid_raises(self):
        """Invalid type raises BadParameter."""
        with pytest.raises(click.BadParameter):
            validate_power_poles(None, None, "invalid")


class TestSetupLogging:
    """Tests for logging setup."""

    def test_valid_levels(self):
        """Valid log levels work."""
        for level in ["debug", "info", "warning", "error"]:
            setup_logging(level)

    def test_invalid_level_raises(self):
        """Invalid level raises ValueError."""
        with pytest.raises(ValueError):
            setup_logging("invalid")


class TestCompileMain:
    """Tests for the main CLI command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def simple_facto_file(self, tmp_path):
        """Create a simple .facto file."""
        file = tmp_path / "simple.facto"
        file.write_text("Signal x = 42;", encoding="utf-8")
        return file

    @pytest.fixture
    def complex_facto_file(self, tmp_path):
        """Create a complex .facto file."""
        file = tmp_path / "complex.facto"
        file.write_text(
            """
Memory counter: "signal-A";
counter.write((counter.read() + 1) % 60);
Entity lamp = place("small-lamp", 0, 0);
lamp.enable = counter.read() < 30;
        """,
            encoding="utf-8",
        )
        return file

    def test_compile_to_stdout(self, runner, simple_facto_file):
        """Compile outputs to stdout."""
        result = runner.invoke(main, [str(simple_facto_file)])
        assert result.exit_code == 0
        assert result.output.startswith("0")

    def test_compile_to_file(self, runner, simple_facto_file, tmp_path):
        """Compile outputs to file."""
        output = tmp_path / "out.blueprint"
        result = runner.invoke(main, [str(simple_facto_file), "-o", str(output)])
        assert result.exit_code == 0
        assert output.exists()
        assert output.read_text().startswith("0")

    def test_custom_name(self, runner, simple_facto_file):
        """Custom blueprint name."""
        result = runner.invoke(main, [str(simple_facto_file), "--name", "My Circuit"])
        assert result.exit_code == 0

    def test_no_optimize(self, runner, simple_facto_file):
        """--no-optimize flag works."""
        result = runner.invoke(main, [str(simple_facto_file), "--no-optimize"])
        assert result.exit_code == 0

    def test_power_poles_value(self, runner, complex_facto_file):
        """--power-poles with value."""
        result = runner.invoke(main, [str(complex_facto_file), "--power-poles", "big"])
        assert result.exit_code == 0

    def test_json_flag(self, runner, simple_facto_file):
        """--json flag outputs valid JSON."""
        result = runner.invoke(main, [str(simple_facto_file), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "blueprint" in data

    def test_log_level_options(self, runner, simple_facto_file):
        """Different log levels work."""
        for level in ["debug", "info", "warning", "error"]:
            result = runner.invoke(main, [str(simple_facto_file), "--log-level", level])
            assert result.exit_code == 0

    def test_layout_retries(self, runner, simple_facto_file):
        """--layout-retries option works."""
        result = runner.invoke(main, [str(simple_facto_file), "--layout-retries", "5"])
        assert result.exit_code == 0

    def test_output_creates_directories(self, runner, simple_facto_file, tmp_path):
        """Output creates parent directories."""
        output = tmp_path / "nested" / "dir" / "out.blueprint"
        result = runner.invoke(main, [str(simple_facto_file), "-o", str(output)])
        assert result.exit_code == 0
        assert output.exists()

    def test_verbose_output_debug(self, runner, simple_facto_file):
        """Debug level shows verbose messages."""
        result = runner.invoke(main, [str(simple_facto_file), "--log-level", "debug"])
        assert result.exit_code == 0

    def test_verbose_output_info(self, runner, simple_facto_file):
        """Info level shows verbose messages."""
        result = runner.invoke(main, [str(simple_facto_file), "--log-level", "info"])
        assert result.exit_code == 0
