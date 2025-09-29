"""
Tests for parser.py - Core parsing functionality.
"""

import pytest
from dsl_compiler.src.parser import DSLParser
from dsl_compiler.src.dsl_ast import Program, DeclStmt, NumberLiteral


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
        import os
        sample_files = [
            "tests/sample_programs/01_basic_arithmetic.fcdsl",
            "tests/sample_programs/04_memory.fcdsl",
        ]
        
        for file_path in sample_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    code = f.read()
                program = parser.parse(code)
                assert isinstance(program, Program)
                assert len(program.statements) > 0