"""
Tests for lowerer.py - IR lowering functionality.
"""

import pytest
from dsl_compiler.src.parser import DSLParser
from dsl_compiler.src.semantic import SemanticAnalyzer, analyze_program
from dsl_compiler.src.lowerer import lower_program


class TestLowerer:
    """Test IR lowering functionality."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    @pytest.fixture
    def analyzer(self):
        return SemanticAnalyzer()

    def test_basic_lowering(self, parser, analyzer):
        """Test basic lowering to IR."""
        program = parser.parse("Signal x = 42;")
        analyze_program(program, strict_types=False, analyzer=analyzer)
        ir_operations, diagnostics, signal_map = lower_program(program, analyzer)

        assert isinstance(ir_operations, list)
        # lower_program returns DiagnosticCollector, not list
        from dsl_compiler.src.semantic import DiagnosticCollector

        assert isinstance(diagnostics, DiagnosticCollector)
        assert isinstance(signal_map, dict)

    def test_lowering_sample_files(self, parser, analyzer):
        """Test lowering on sample files."""
        import os

        sample_files = [
            "tests/sample_programs/01_basic_arithmetic.fcdsl",
            "tests/sample_programs/04_memory.fcdsl",
        ]

        for file_path in sample_files:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    code = f.read()
                program = parser.parse(code)
                analyze_program(program, strict_types=False, analyzer=analyzer)
                ir_operations, diagnostics, signal_map = lower_program(
                    program, analyzer
                )

                assert isinstance(ir_operations, list)
                assert len(ir_operations) > 0
