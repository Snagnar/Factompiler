"""
Tests for semantic.py - Semantic analysis functionality.
"""

import pytest
from dsl_compiler.src.parser import DSLParser
from dsl_compiler.src.semantic import SemanticAnalyzer, analyze_program


class TestSemanticAnalyzer:
    """Test semantic analysis functionality."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    @pytest.fixture
    def analyzer(self):
        return SemanticAnalyzer()

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer can be initialized."""
        assert analyzer is not None

    def test_basic_semantic_analysis(self, parser, analyzer):
        """Test basic semantic analysis."""
        program = parser.parse("Signal x = 42;")
        diagnostics = analyze_program(program, strict_types=False, analyzer=analyzer)
        # analyze_program returns DiagnosticCollector, not list
        from dsl_compiler.src.semantic import DiagnosticCollector

        assert isinstance(diagnostics, DiagnosticCollector)

    def test_semantic_analysis_sample_files(self, parser, analyzer):
        """Test semantic analysis on sample files."""
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
                diagnostics = analyze_program(
                    program, strict_types=False, analyzer=analyzer
                )
                # analyze_program returns DiagnosticCollector, not list
                from dsl_compiler.src.semantic import DiagnosticCollector

                assert isinstance(diagnostics, DiagnosticCollector)

    def test_write_legacy_syntax_rejected(self, parser, analyzer):
        """Ensure legacy write(memory, value) form produces a migration error."""

        code = """
    Memory counter: "signal-A";
    write(("signal-A", 0), counter, when=once);
        Signal value = 42;
        write(counter, value);
        """

        program = parser.parse(code)
        diagnostics = analyze_program(program, strict_types=False, analyzer=analyzer)

        assert diagnostics.has_errors(), "Legacy write syntax should raise an error"
        messages = diagnostics.get_messages()
        assert any("not a memory symbol" in msg for msg in messages), (
            "Expected a diagnostic explaining that the target must be a memory"
        )

    def test_write_with_enable_signal_passes(self, parser, analyzer):
        """Verify semantic analysis accepts write(value, memory, when=signal)."""

        code = """
    Memory counter: "signal-A";
    write(("signal-A", 0), counter, when=once);
        Signal enable = 1;
        write(read(counter) + 1, counter, when=enable);
        """

        program = parser.parse(code)
        diagnostics = analyze_program(program, strict_types=False, analyzer=analyzer)

        assert not diagnostics.has_errors(), (
            diagnostics.get_messages() if diagnostics.has_errors() else ""
        )
