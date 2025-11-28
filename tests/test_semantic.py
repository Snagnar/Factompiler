"""
Tests for semantic.py - Semantic analysis functionality.
"""
import os

import pytest
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics


class TestSemanticAnalyzer:
    """Test semantic analysis functionality."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    @pytest.fixture
    def diagnostics(self):
        return ProgramDiagnostics()

    @pytest.fixture
    def analyzer(self, diagnostics):
        return SemanticAnalyzer(diagnostics, strict_types=False)

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer can be initialized."""
        assert analyzer is not None

    def test_basic_semantic_analysis(self, parser, analyzer, diagnostics):
        """Test basic semantic analysis."""
        program = parser.parse("Signal x = 42;")
        analyzer.visit(program)
        # Check diagnostics
        assert isinstance(diagnostics, ProgramDiagnostics)

    def test_semantic_analysis_sample_files(self, parser):
        """Test semantic analysis on sample files."""

        sample_files = [
            "tests/sample_programs/01_basic_arithmetic.fcdsl",
            "tests/sample_programs/04_memory.fcdsl",
        ]

        for file_path in sample_files:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    code = f.read()
                program = parser.parse(code)
                diagnostics = ProgramDiagnostics()
                analyzer = SemanticAnalyzer(diagnostics, strict_types=False)
                analyzer.visit(program)
                assert not diagnostics.has_errors()

    def test_write_legacy_syntax_rejected(self, parser):
        """Ensure legacy write(memory, value) form produces a migration error."""

        code = """
    Memory counter: "signal-A";
        Signal value = 42;
        write(counter, value);
        """

        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics, strict_types=False)
        analyzer.visit(program)

        assert diagnostics.has_errors(), "Legacy write syntax should raise an error"
        messages = diagnostics.get_messages()
        assert any("not a memory symbol" in msg for msg in messages), (
            "Expected a diagnostic explaining that the target must be a memory"
        )

    def test_write_with_enable_signal_passes(self, parser):
        """Verify semantic analysis accepts write(value, memory, when=signal)."""

        code = """
    Memory counter: "signal-A";
        Signal enable = 1;
        write(read(counter) + 1, counter, when=enable);
        """

        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics, strict_types=False)
        analyzer.visit(program)

        assert not diagnostics.has_errors(), (
            diagnostics.get_messages() if diagnostics.has_errors() else ""
        )

    def test_signal_w_literal_is_rejected(self, parser):
        """User-declared signal literals must not target the reserved signal-W channel."""

        code = 'Signal bad = ("signal-W", 10);'

        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics, strict_types=False)
        analyzer.visit(program)

        assert diagnostics.has_errors(), (
            "Expected an error when using reserved signal-W"
        )
        messages = diagnostics.get_messages()
        assert any(
            "signal-W" in message and "reserved" in message for message in messages
        ), "Diagnostic should explain that signal-W is reserved"

    def test_signal_w_projection_is_rejected(self, parser):
        """Projecting onto signal-W must surface a reservation error."""

        code = 'Signal x = 10 | "signal-W";'

        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics, strict_types=False)
        analyzer.visit(program)

        assert diagnostics.has_errors(), (
            "Expected an error when projecting onto reserved signal-W"
        )
        messages = diagnostics.get_messages()
        assert any(
            "signal-W" in message and "reserved" in message for message in messages
        ), "Diagnostic should explain that signal-W is reserved"

    def test_signal_w_memory_declaration_is_rejected(self, parser):
        """Memory declarations must not claim the reserved signal-W channel."""

        code = 'Memory w: "signal-W";'

        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics, strict_types=False)
        analyzer.visit(program)

        assert diagnostics.has_errors(), (
            "Expected an error when reserving signal-W for memory storage"
        )
        messages = diagnostics.get_messages()
        assert any(
            "signal-W" in message and "reserved" in message for message in messages
        ), "Diagnostic should explain that signal-W is reserved"

    def test_virtual_signal_allocation_unbounded(self):
        """Allocator must keep producing unique implicit virtual signals."""

        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics, strict_types=False)

        allocations = [analyzer.allocate_implicit_type() for _ in range(60)]
        names = [info.name for info in allocations]

        # All allocated names should be unique
        assert len(names) == len(set(names))
        # Implicit signals use __v prefix
        assert names[0] == "__v1"
        assert names[1] == "__v2"
        assert names[59] == "__v60"
