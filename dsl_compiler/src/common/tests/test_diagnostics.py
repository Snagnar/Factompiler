"""
Tests for common/diagnostics.py - Diagnostic collection and reporting.
"""

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics


class TestProgramDiagnostics:
    """Tests for ProgramDiagnostics class."""

    def test_diagnostics_initialization(self):
        """Test ProgramDiagnostics initializes with empty diagnostics."""
        diag = ProgramDiagnostics()
        assert not diag.has_errors()
        assert diag.error_count() == 0
        assert diag.warning_count() == 0

    def test_error_collection(self):
        """Test adding and counting errors."""
        diag = ProgramDiagnostics()
        diag.error("Test error", stage="test")
        assert diag.has_errors()
        assert diag.error_count() == 1

    def test_warning_collection(self):
        """Test adding and counting warnings."""
        diag = ProgramDiagnostics()
        diag.warning("Test warning", stage="test")
        assert not diag.has_errors()  # Warnings don't count as errors
        assert diag.warning_count() == 1

    def test_info_collection(self):
        """Test adding info messages."""
        diag = ProgramDiagnostics()
        diag.info("Test info", stage="test")
        assert not diag.has_errors()

    def test_multiple_diagnostics(self):
        """Test collecting multiple diagnostics of different types."""
        diag = ProgramDiagnostics()
        diag.error("Test error", stage="test")
        diag.warning("Test warning", stage="test")
        diag.info("Test info", stage="test")
        assert diag.has_errors()
        assert diag.error_count() == 1
        assert diag.warning_count() == 1

    def test_get_messages(self):
        """Test retrieving diagnostic messages."""
        diag = ProgramDiagnostics()
        diag.error("Test error message", stage="test")
        messages = diag.get_messages()
        assert isinstance(messages, list)
        assert len(messages) >= 1

    def test_raise_errors_mode(self):
        """Test that raise_errors=True causes errors to raise exceptions."""
        import pytest

        diag = ProgramDiagnostics(raise_errors=True)
        with pytest.raises(Exception) as exc_info:
            diag.error("This should raise", stage="test")
        assert "This should raise" in str(exc_info.value)

    def test_get_messages_filters_by_severity(self):
        """Test that get_messages filters out lower severity messages."""
        from dsl_compiler.src.common.diagnostics import DiagnosticSeverity

        diag = ProgramDiagnostics()
        diag.info("Info message", stage="test")
        diag.warning("Warning message", stage="test")
        diag.error("Error message", stage="test")

        # Default min_severity is WARNING, so INFO should be filtered out
        messages = diag.get_messages(min_severity=DiagnosticSeverity.WARNING)
        assert len(messages) == 2  # warning + error only
        assert not any("Info message" in m for m in messages)
        assert any("Warning message" in m for m in messages)
        assert any("Error message" in m for m in messages)

    def test_get_messages_filters_out_debug(self):
        """Test severity filtering by explicitly using DEBUG severity."""
        from dsl_compiler.src.common.diagnostics import Diagnostic, DiagnosticSeverity

        diag = ProgramDiagnostics()
        # Manually add a DEBUG diagnostic (no public method for it)
        diag.diagnostics.append(
            Diagnostic(
                severity=DiagnosticSeverity.DEBUG,
                message="Debug message",
                stage="test",
            )
        )
        diag.warning("Warning message", stage="test")

        # With WARNING min_severity, DEBUG should be filtered out
        messages = diag.get_messages(min_severity=DiagnosticSeverity.WARNING)
        assert not any("Debug message" in m for m in messages)
        assert any("Warning message" in m for m in messages)

    def test_error_with_node_extracts_location(self):
        """Test that error extracts line/column/source_file from node when line=0."""

        class MockNode:
            line = 42
            column = 7
            source_file = "/path/to/test.facto"

        diag = ProgramDiagnostics()
        diag.error("Test error", stage="test", node=MockNode())

        # Check that the diagnostic captured node's location
        assert len(diag.diagnostics) == 1
        d = diag.diagnostics[0]
        assert d.line == 42
        assert d.column == 7
        assert d.source_file == "/path/to/test.facto"

    def test_format_diagnostic_with_full_location(self):
        """Test diagnostic formatting with source file, line, and column."""
        diag = ProgramDiagnostics()
        diag.error(
            "Test error",
            stage="test",
            source_file="/path/to/test.facto",
            line=42,
            column=7,
        )

        messages = diag.get_messages()
        assert len(messages) == 1
        # Should include filename, line, and column
        assert "test.facto" in messages[0]
        assert "42" in messages[0]
        assert "7" in messages[0]
