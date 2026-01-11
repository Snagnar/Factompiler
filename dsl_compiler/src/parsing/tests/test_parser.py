"""
Tests for parsing/parser.py - DSL parser functionality.
"""

from pathlib import Path

import pytest

from dsl_compiler.src.ast.statements import Program
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer


class TestDSLParser:
    """Tests for DSLParser class."""

    @pytest.fixture
    def parser(self):
        """Create a new parser instance."""
        return DSLParser()

    def test_parser_initialization(self):
        """Test DSLParser can be initialized."""
        parser = DSLParser()
        assert parser is not None

    def test_parse_returns_program(self, parser):
        """Test parse() returns a Program AST node."""
        result = parser.parse("Signal x = 42;")
        assert isinstance(result, Program)

    def test_parse_basic_statement(self, parser):
        """Test parsing a basic signal declaration."""
        program = parser.parse("Signal x = 42;")
        assert isinstance(program, Program)
        assert len(program.statements) == 1

    def test_parse_multiple_statements(self, parser):
        """Test parsing multiple statements."""
        code = """
        Signal x = 10;
        Signal y = 20;
        Signal z = x + y;
        """
        program = parser.parse(code)
        assert isinstance(program, Program)
        assert len(program.statements) == 3

    def test_parse_raises_on_syntax_error(self, parser):
        """Test parser raises SyntaxError on invalid syntax."""
        with pytest.raises(SyntaxError):
            parser.parse("Signal x = ;")

    def test_parse_independent_calls(self):
        """Test multiple parse calls are independent (no shared state)."""
        parser = DSLParser()
        ast1 = parser.parse("Signal x = 1;")
        ast2 = parser.parse("Signal y = 2;")

        assert ast1 is not ast2
        assert len(ast1.statements) == 1
        assert len(ast2.statements) == 1


class TestDSLParserImports:
    """Tests for import resolution in DSLParser."""

    def test_relative_imports_resolve_from_file_directory(self, tmp_path):
        """Ensure preprocess_imports resolves paths relative to the source file."""
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()

        helper_path = lib_dir / "helper.facto"
        helper_path.write_text(
            "Signal helper_signal = 5;\n",
            encoding="utf-8",
        )

        main_path = tmp_path / "main.facto"
        main_path.write_text(
            'import "lib/helper.facto";\nSignal result = helper_signal + 1;\n',
            encoding="utf-8",
        )

        parser = DSLParser()
        program = parser.parse_file(main_path)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)

        assert not diagnostics.has_errors(), diagnostics.get_messages()


class TestDSLParserSampleFiles:
    """Tests for parsing sample program files."""

    @pytest.fixture
    def parser(self):
        """Create a new parser instance."""
        return DSLParser()

    def test_parse_sample_files(self, parser):
        """Test parsing of sample files if they exist."""
        # Note: Path updated for new example_programs location
        sample_files = [
            "example_programs/01_basic_arithmetic.facto",
            "example_programs/04_memory.facto",
        ]

        for file_path in sample_files:
            path = Path(file_path)
            if path.exists():
                with open(path) as f:
                    code = f.read()
                program = parser.parse(code)
                assert isinstance(program, Program)
                assert len(program.statements) > 0


class TestDSLParserErrorHandling:
    """Tests for parser error handling."""

    def test_parse_file_not_found(self):
        """Test parse_file raises FileNotFoundError for missing file."""
        parser = DSLParser()
        with pytest.raises(FileNotFoundError) as exc_info:
            parser.parse_file(Path("/nonexistent/path/to/file.facto"))
        assert "Source file not found" in str(exc_info.value)

    def test_parse_with_string_filename_uses_default_base_path(self):
        """Test parsing with <string> filename uses default base path."""
        parser = DSLParser()
        # Just verify it doesn't crash when using the default base path
        program = parser.parse("Signal x = 1;", "<string>")
        assert isinstance(program, Program)
