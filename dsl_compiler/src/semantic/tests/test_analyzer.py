"""
Tests for semantic/analyzer.py - Semantic analysis functionality.
"""

from pathlib import Path

import pytest

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer


class TestSemanticAnalyzer:
    """Tests for SemanticAnalyzer class."""

    @pytest.fixture
    def parser(self):
        """Create a new parser instance."""
        return DSLParser()

    @pytest.fixture
    def diagnostics(self):
        """Create a new diagnostics instance."""
        return ProgramDiagnostics()

    @pytest.fixture
    def analyzer(self, diagnostics):
        """Create a new analyzer instance."""
        return SemanticAnalyzer(diagnostics)

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer can be initialized."""
        assert analyzer is not None

    def test_analyzer_has_symbol_table(self, analyzer):
        """Test analyzer has a symbol table."""
        assert analyzer.symbol_table is not None

    def test_analyzer_has_signal_registry(self, analyzer):
        """Test analyzer has a signal registry."""
        assert analyzer.signal_registry is not None


class TestSemanticAnalyzerVisit:
    """Tests for SemanticAnalyzer.visit() method."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_visit_basic_declaration(self, parser):
        """Test visiting a basic signal declaration."""
        program = parser.parse("Signal x = 42;")
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()

    def test_visit_detects_undefined_variable(self, parser):
        """Test analyzer detects undefined variable usage."""
        program = parser.parse("Signal x = 5; Signal y = x + z;")
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert diagnostics.has_errors()


class TestSemanticAnalyzerLegacySyntax:
    """Tests for legacy syntax rejection."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_write_legacy_syntax_rejected(self, parser):
        """Ensure legacy write(memory, value) form produces a migration error."""
        code = """
        Memory counter: "signal-A";
        Signal value = 42;
        write(counter, value);
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)

        assert diagnostics.has_errors(), "Legacy write syntax should raise an error"
        messages = diagnostics.get_messages()
        assert any("not a memory symbol" in msg for msg in messages)


class TestSemanticAnalyzerMemoryWrite:
    """Tests for memory.write() semantic analysis."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_write_with_enable_signal_passes(self, parser):
        """Verify semantic analysis accepts memory.write(value, when=signal)."""
        code = """
        Memory counter: "signal-A";
        Signal enable = 1;
        counter.write(counter.read() + 1, when=enable);
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)

        assert not diagnostics.has_errors(), diagnostics.get_messages()


class TestSemanticAnalyzerSignalWReservation:
    """Tests for signal-W reservation enforcement."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_signal_w_literal_is_rejected(self, parser):
        """User-declared signal literals must not target the reserved signal-W channel."""
        code = 'Signal bad = ("signal-W", 10);'
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)

        assert diagnostics.has_errors(), "Expected an error when using reserved signal-W"
        messages = diagnostics.get_messages()
        assert any("signal-W" in msg and "reserved" in msg for msg in messages)

    def test_signal_w_projection_is_rejected(self, parser):
        """Projecting onto signal-W must surface a reservation error."""
        code = 'Signal x = 10 | "signal-W";'
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)

        assert diagnostics.has_errors(), "Expected an error when projecting onto reserved signal-W"
        messages = diagnostics.get_messages()
        assert any("signal-W" in msg and "reserved" in msg for msg in messages)

    def test_signal_w_memory_declaration_is_rejected(self, parser):
        """Memory declarations must not claim the reserved signal-W channel."""
        code = 'Memory w: "signal-W";'
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)

        assert diagnostics.has_errors()
        messages = diagnostics.get_messages()
        assert any("signal-W" in msg and "reserved" in msg for msg in messages)


class TestSemanticAnalyzerImplicitSignalAllocation:
    """Tests for implicit signal allocation."""

    def test_allocate_implicit_type(self):
        """Test allocate_implicit_type produces unique virtual signals."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)

        allocations = [analyzer.allocate_implicit_type() for _ in range(60)]
        names = [info.name for info in allocations]

        # All allocated names should be unique
        assert len(names) == len(set(names))
        # Implicit signals use __v prefix
        assert names[0] == "__v1"
        assert names[1] == "__v2"
        assert names[59] == "__v60"


class TestSemanticAnalyzerSampleFiles:
    """Tests for semantic analysis on sample files."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_semantic_analysis_sample_files(self, parser):
        """Test semantic analysis on sample files if they exist."""
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
                diagnostics = ProgramDiagnostics()
                analyzer = SemanticAnalyzer(diagnostics)
                analyzer.visit(program)
                assert not diagnostics.has_errors()
