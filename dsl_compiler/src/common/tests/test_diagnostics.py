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
