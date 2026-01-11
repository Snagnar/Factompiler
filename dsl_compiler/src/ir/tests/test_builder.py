"""
Tests for ir/builder.py - IR builder functionality.
"""

from dsl_compiler.src.ir.builder import IRBuilder


class TestIRBuilder:
    """Tests for IRBuilder class."""

    def test_ir_builder_initialization(self):
        """Test IRBuilder can be initialized."""
        builder = IRBuilder()
        assert builder is not None

    def test_ir_builder_has_signal_type_map(self):
        """Test IRBuilder has a signal_type_map."""
        builder = IRBuilder()
        assert hasattr(builder, "signal_type_map")
        assert isinstance(builder.signal_type_map, dict)


class TestIRBuilderImplicitSignalAllocation:
    """Tests for implicit signal allocation in IRBuilder."""

    def test_allocate_implicit_type(self):
        """Test allocate_implicit_type produces unique names."""
        builder = IRBuilder()
        names = [builder.allocate_implicit_type() for _ in range(60)]

        # All allocated names should be unique
        assert len(names) == len(set(names))

    def test_allocate_implicit_type_prefix(self):
        """Test implicit signals use __v prefix with sequential numbering."""
        builder = IRBuilder()
        names = [builder.allocate_implicit_type() for _ in range(5)]

        assert names[0] == "__v1"
        assert names[1] == "__v2"
        assert names[2] == "__v3"
        assert names[3] == "__v4"
        assert names[4] == "__v5"

    def test_allocate_more_than_26_virtual_signals(self):
        """Ensure implicit signal allocation scales beyond 26 signals."""
        builder = IRBuilder()
        names = [builder.allocate_implicit_type() for _ in range(60)]

        # All allocated names should be unique
        assert len(names) == len(set(names))
        # Implicit signals use __v prefix with sequential numbering
        assert names[0] == "__v1"
        assert names[59] == "__v60"
