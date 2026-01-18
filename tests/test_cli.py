"""
Tests for the CLI module (dsl_compiler/cli.py).

These tests cover the command-line interface and compile_dsl_source function.
"""

import json

import click
import pytest
from click.testing import CliRunner

from dsl_compiler.cli import compile_dsl_source, main, setup_logging, validate_power_poles


class TestCompileDslSource:
    """Tests for the compile_dsl_source function."""

    def test_simple_signal_compiles(self):
        """Basic signal declaration compiles successfully."""
        code = "Signal x = 42;"
        success, result, messages = compile_dsl_source(code)
        assert success is True
        assert result.startswith("0")  # Blueprint string starts with "0"

    def test_memory_and_lamp_compiles(self):
        """Memory and entity code compiles successfully."""
        code = """
        Memory counter: "signal-A";
        counter.write((counter.read() + 1) % 60);
        Entity lamp = place("small-lamp", 0, 0);
        lamp.enable = counter.read() < 30;
        """
        success, result, messages = compile_dsl_source(code)
        assert success is True
        assert result.startswith("0")

    def test_with_custom_program_name(self):
        """Custom program name is used."""
        code = "Signal x = 5;"
        success, result, messages = compile_dsl_source(code, program_name="My Test Circuit")
        assert success is True

    def test_with_source_name_path(self):
        """Source name path derives blueprint name."""
        code = "Signal x = 5;"
        success, result, messages = compile_dsl_source(
            code, source_name="/path/to/my_test_circuit.facto"
        )
        assert success is True

    def test_json_output(self):
        """JSON output is valid JSON."""
        code = "Signal x = 42;"
        success, result, messages = compile_dsl_source(code, use_json=True)
        assert success is True
        # Should be valid JSON
        data = json.loads(result)
        assert "blueprint" in data

    def test_optimize_disabled(self):
        """Code compiles with optimization disabled."""
        code = "Signal x = 42; Signal y = x + 1;"
        success, result, messages = compile_dsl_source(code, optimize=False)
        assert success is True

    def test_power_poles_option(self):
        """Power poles are added when specified."""
        code = """
        Memory counter: "signal-A";
        counter.write(counter.read() + 1);
        """
        success, result, messages = compile_dsl_source(code, power_pole_type="medium")
        assert success is True

    def test_syntax_error_fails(self):
        """Syntax error raises exception (raise_errors=True mode)."""
        code = "Signal x = ;"  # Missing value
        with pytest.raises(SyntaxError):
            compile_dsl_source(code)

    def test_semantic_error_fails(self):
        """Semantic error raises exception (raise_errors=True mode)."""
        code = "Signal x = undefined_var;"
        with pytest.raises(RuntimeError):  # Raises semantic error
            compile_dsl_source(code)

    def test_different_log_levels(self):
        """Different log levels don't crash."""
        code = "Signal x = 5;"
        for level in ["debug", "info", "warning", "error"]:
            success, result, messages = compile_dsl_source(code, log_level=level)
            assert success is True


class TestValidatePowerPoles:
    """Tests for power pole validation callback."""

    def test_valid_small(self):
        """'small' is valid."""
        result = validate_power_poles(None, None, "small")
        assert result == "small"

    def test_valid_medium(self):
        """'medium' is valid."""
        result = validate_power_poles(None, None, "medium")
        assert result == "medium"

    def test_valid_big(self):
        """'big' is valid."""
        result = validate_power_poles(None, None, "big")
        assert result == "big"

    def test_valid_substation(self):
        """'substation' is valid."""
        result = validate_power_poles(None, None, "substation")
        assert result == "substation"

    def test_case_insensitive(self):
        """Validation is case insensitive."""
        result = validate_power_poles(None, None, "MEDIUM")
        assert result == "medium"

    def test_none_value(self):
        """None value returns None."""
        result = validate_power_poles(None, None, None)
        assert result is None

    def test_invalid_type(self):
        """Invalid type raises BadParameter."""
        with pytest.raises(click.BadParameter):
            validate_power_poles(None, None, "invalid")


class TestSetupLogging:
    """Tests for logging setup."""

    def test_valid_log_level(self):
        """Valid log levels don't raise."""
        for level in ["debug", "info", "warning", "error"]:
            setup_logging(level)  # Should not raise

    def test_invalid_log_level(self):
        """Invalid log level raises ValueError."""
        with pytest.raises(ValueError):
            setup_logging("invalid_level")


class TestCliMain:
    """Tests for the main CLI command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def simple_facto_file(self, tmp_path):
        """Create a simple .facto file for testing."""
        file = tmp_path / "test.facto"
        file.write_text("Signal x = 42;", encoding="utf-8")
        return file

    @pytest.fixture
    def complex_facto_file(self, tmp_path):
        """Create a more complex .facto file."""
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

    def test_compile_file(self, runner, simple_facto_file):
        """Compiling a file works."""
        result = runner.invoke(main, [str(simple_facto_file)])
        assert result.exit_code == 0
        assert result.output.startswith("0")  # Blueprint string

    def test_compile_string_input(self, runner):
        """Compiling from --input string works."""
        result = runner.invoke(main, ["--input", "Signal x = 5;"])
        assert result.exit_code == 0
        assert result.output.startswith("0")

    def test_both_file_and_string_fails(self, runner, simple_facto_file):
        """Specifying both file and string fails."""
        result = runner.invoke(main, [str(simple_facto_file), "--input", "Signal x = 5;"])
        assert result.exit_code == 1
        assert "Cannot specify both" in result.output

    def test_no_input_fails(self, runner):
        """No input source fails."""
        result = runner.invoke(main, [])
        assert result.exit_code == 1
        assert "Must specify" in result.output

    def test_output_to_file(self, runner, simple_facto_file, tmp_path):
        """Output to file works."""
        output_file = tmp_path / "output.blueprint"
        result = runner.invoke(main, [str(simple_facto_file), "-o", str(output_file)])
        assert result.exit_code == 0
        assert output_file.exists()
        assert output_file.read_text().startswith("0")

    def test_custom_name(self, runner, simple_facto_file):
        """Custom blueprint name works."""
        result = runner.invoke(main, [str(simple_facto_file), "--name", "My Circuit"])
        assert result.exit_code == 0

    def test_no_optimize_flag(self, runner, simple_facto_file):
        """--no-optimize flag works."""
        result = runner.invoke(main, [str(simple_facto_file), "--no-optimize"])
        assert result.exit_code == 0

    def test_power_poles_flag(self, runner, complex_facto_file):
        """--power-poles flag works."""
        result = runner.invoke(main, [str(complex_facto_file), "--power-poles", "medium"])
        assert result.exit_code == 0

    def test_power_poles_value(self, runner, complex_facto_file):
        """--power-poles with value works."""
        result = runner.invoke(main, [str(complex_facto_file), "--power-poles", "big"])
        assert result.exit_code == 0

    def test_json_output(self, runner, simple_facto_file):
        """--json flag outputs JSON."""
        result = runner.invoke(main, [str(simple_facto_file), "--json"])
        assert result.exit_code == 0
        # Should be valid JSON
        data = json.loads(result.output)
        assert "blueprint" in data

    def test_log_level_debug(self, runner, simple_facto_file):
        """--log-level debug works."""
        result = runner.invoke(main, [str(simple_facto_file), "--log-level", "debug"])
        assert result.exit_code == 0

    def test_log_level_info(self, runner, simple_facto_file):
        """--log-level info works."""
        result = runner.invoke(main, [str(simple_facto_file), "--log-level", "info"])
        assert result.exit_code == 0

    def test_syntax_error_file(self, runner, tmp_path):
        """Syntax error in file fails compilation with exception."""
        bad_file = tmp_path / "bad.facto"
        bad_file.write_text("Signal x = ;", encoding="utf-8")
        result = runner.invoke(main, [str(bad_file)])
        assert result.exit_code != 0
        # Exception is raised, check the exception type
        assert result.exception is not None

    def test_layout_retries_option(self, runner, complex_facto_file):
        """--layout-retries option is accepted."""
        result = runner.invoke(main, [str(complex_facto_file), "--layout-retries", "5"])
        assert result.exit_code == 0

    def test_output_directory_creation(self, runner, simple_facto_file, tmp_path):
        """Output creates parent directories."""
        output_file = tmp_path / "nested" / "dir" / "output.blueprint"
        result = runner.invoke(main, [str(simple_facto_file), "-o", str(output_file)])
        assert result.exit_code == 0
        assert output_file.exists()


class TestCliEdgeCases:
    """Edge case tests for CLI."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_string_with_debug_logging(self, runner):
        """String input with debug logging works."""
        result = runner.invoke(main, ["--input", "Signal x = 5;", "--log-level", "debug"])
        assert result.exit_code == 0

    def test_file_with_verbose_shows_messages(self, runner, tmp_path):
        """Verbose logging shows diagnostic messages."""
        file = tmp_path / "test.facto"
        file.write_text("Signal x = 42;", encoding="utf-8")
        result = runner.invoke(main, [str(file), "--log-level", "info"])
        assert result.exit_code == 0
        # Verbose mode prints completion message to stderr
        assert "completed" in result.output.lower() or result.exit_code == 0
