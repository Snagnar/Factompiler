"""
Tests for common/source_location.py - Source location tracking.
"""

from dsl_compiler.src.common.source_location import SourceLocation


class TestSourceLocation:
    """Tests for SourceLocation class."""

    def test_source_location_creation(self):
        """Test SourceLocation stores file, line, and column."""
        loc = SourceLocation(file="test.facto", line=10, column=5)
        assert loc.file == "test.facto"
        assert loc.line == 10
        assert loc.column == 5

    def test_source_location_string_representation(self):
        """Test SourceLocation string formatting includes file and line."""
        loc = SourceLocation(file="test.facto", line=10, column=5)
        str_repr = str(loc)
        assert "test.facto" in str_repr
        assert "10" in str_repr
