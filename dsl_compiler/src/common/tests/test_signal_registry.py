"""
Tests for common/signal_registry.py - Signal type registration and resolution.
"""

from dsl_compiler.src.common.signal_registry import SignalTypeRegistry


class TestSignalTypeRegistry:
    """Tests for SignalTypeRegistry class."""

    def test_registry_initialization(self):
        """Test SignalTypeRegistry initializes correctly."""
        registry = SignalTypeRegistry()
        assert registry is not None

    def test_register_signal(self):
        """Test registering a new signal type."""
        registry = SignalTypeRegistry()
        registry.register("iron-plate", "iron-plate", "item")
        mapping = registry.resolve("iron-plate")
        assert mapping is not None
        assert mapping["name"] == "iron-plate"
        assert mapping["type"] == "item"

    def test_resolve_unregistered_signal(self):
        """Test resolving an unregistered signal returns None."""
        registry = SignalTypeRegistry()
        mapping = registry.resolve("nonexistent-signal")
        assert mapping is None

    def test_allocate_implicit(self):
        """Test allocating implicit signal names."""
        registry = SignalTypeRegistry()
        implicit = registry.allocate_implicit()
        assert implicit == "__v1"
        # Implicit signals are NOT immediately mapped - deferred to layout phase
        assert registry.resolve(implicit) is None

    def test_allocate_multiple_implicit(self):
        """Test allocating multiple implicit signals produces unique names."""
        registry = SignalTypeRegistry()
        names = [registry.allocate_implicit() for _ in range(5)]
        assert len(names) == len(set(names))
        assert names == ["__v1", "__v2", "__v3", "__v4", "__v5"]
