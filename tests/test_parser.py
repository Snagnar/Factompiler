"""
Tests for parser.py - Core parsing functionality.
"""

import os

import pytest

from dsl_compiler.src.ast.statements import Program
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer


class TestParser:
    """Test core parser functionality."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_parser_basic(self, parser):
        """Test basic parsing works."""
        program = parser.parse("Signal x = 42;")
        assert isinstance(program, Program)
        assert len(program.statements) == 1

    def test_parse_sample_files(self, parser):
        """Test parsing of sample files."""

        sample_files = [
            "tests/sample_programs/01_basic_arithmetic.facto",
            "tests/sample_programs/04_memory.facto",
        ]

        for file_path in sample_files:
            if os.path.exists(file_path):
                with open(file_path) as f:
                    code = f.read()
                program = parser.parse(code)
                assert isinstance(program, Program)
                assert len(program.statements) > 0

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
