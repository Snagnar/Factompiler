"""
Tests for common/signal_registry.py - Signal type registration and resolution.
"""

from dsl_compiler.src.common.signal_registry import (
    SignalTypeInfo,
    SignalTypeRegistry,
    is_valid_factorio_signal,
)


class TestIsValidFactorioSignal:
    """Tests for is_valid_factorio_signal function."""

    def test_empty_signal_name(self):
        """Test empty signal name is invalid."""
        valid, error = is_valid_factorio_signal("")
        assert valid is False
        assert error is not None
        assert "empty" in error.lower()

    def test_implicit_signal_is_valid(self):
        """Test implicit compiler signal __v* is valid."""
        valid, error = is_valid_factorio_signal("__v1")
        assert valid is True
        assert error is None

    def test_known_factorio_signal_is_valid(self):
        """Test known Factorio signal is valid."""
        valid, error = is_valid_factorio_signal("signal-A")
        assert valid is True
        assert error is None

    def test_unknown_signal_is_invalid(self):
        """Test unknown signal is invalid."""
        valid, error = is_valid_factorio_signal("unknown-signal-xyz")
        assert valid is False
        assert error is not None
        assert "unknown" in error.lower()


class TestSignalTypeInfo:
    """Tests for SignalTypeInfo dataclass."""

    def test_signal_type_info_creation(self):
        """Test SignalTypeInfo creation with defaults."""
        info = SignalTypeInfo(name="signal-A")
        assert info.name == "signal-A"
        assert info.is_implicit is False
        assert info.is_virtual is False

    def test_signal_type_info_implicit(self):
        """Test SignalTypeInfo for implicit signal."""
        info = SignalTypeInfo(name="__v1", is_implicit=True, is_virtual=True)
        assert info.name == "__v1"
        assert info.is_implicit is True
        assert info.is_virtual is True


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

    def test_allocate_implicit_type(self):
        """Test allocate_implicit_type returns SignalTypeInfo."""
        registry = SignalTypeRegistry()
        info = registry.allocate_implicit_type()
        assert isinstance(info, SignalTypeInfo)
        assert info.name == "__v1"
        assert info.is_implicit is True
        assert info.is_virtual is True

    def test_get_all_mappings(self):
        """Test get_all_mappings returns the internal map."""
        registry = SignalTypeRegistry()
        registry.register("signal-A", "signal-A", "virtual")
        registry.register("iron-plate", "iron-plate", "item")

        mappings = registry.get_all_mappings()
        assert "signal-A" in mappings
        assert "iron-plate" in mappings
        assert mappings["signal-A"]["type"] == "virtual"

    def test_register_unknown_signal_registers_with_draftsman(self):
        """Test registering unknown signal adds it to draftsman."""
        registry = SignalTypeRegistry()
        # This should not raise even for a made-up signal name
        registry.register("custom-signal-xyz", "custom-signal-xyz", "virtual")
        mapping = registry.resolve("custom-signal-xyz")
        assert mapping["name"] == "custom-signal-xyz"
