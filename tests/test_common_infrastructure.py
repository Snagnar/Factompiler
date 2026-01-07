"""Test that common infrastructure works correctly."""

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.common.signal_registry import SignalTypeRegistry
from dsl_compiler.src.common.source_location import SourceLocation


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

    # Resolve it (returns full mapping dict)
    mapping = registry.resolve("iron-plate")
    assert mapping is not None
    assert mapping["name"] == "iron-plate"
    assert mapping["type"] == "item"

    # Allocate implicit - returns key but doesn't immediately register
    implicit = registry.allocate_implicit()
    assert implicit == "__v1"
    # Implicit signals are NOT immediately mapped - deferred to layout phase
    assert registry.resolve(implicit) is None


def test_source_location():
    """Test source location formatting."""
    loc = SourceLocation(file="test.facto", line=10, column=5)
    assert "test.facto" in str(loc)
    assert "10" in str(loc)
