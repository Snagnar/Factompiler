"""Test that common infrastructure works correctly."""

from dsl_compiler.src.common import (
    ProgramDiagnostics,
    SourceLocation,
    SignalTypeRegistry,
)


def test_program_diagnostics_basic():
    """Test basic diagnostic collection."""
    diag = ProgramDiagnostics()

    diag.error("Test error", stage="test")
    diag.warning("Test warning", stage="test")
    diag.info("Test info", stage="test")

    assert diag.has_errors()
    assert diag.error_count() == 1
    assert diag.warning_count() == 1


def test_signal_registry():
    """Test signal type registry."""
    registry = SignalTypeRegistry()

    # Register a signal
    registry.register("iron-plate", "iron-plate", "item")

    # Resolve it
    assert registry.resolve_name("iron-plate") == "iron-plate"
    assert registry.resolve_type("iron-plate") == "item"

    # Allocate implicit
    implicit = registry.allocate_implicit()
    assert implicit == "__v1"
    assert registry.resolve_name(implicit) == "signal-A"


def test_source_location():
    """Test source location formatting."""
    loc = SourceLocation(file="test.fcdsl", line=10, column=5)
    assert "test.fcdsl" in str(loc)
    assert "10" in str(loc)
