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

    def test_source_location_str_without_file(self):
        """Test SourceLocation str without file shows line:column."""
        loc = SourceLocation(line=5, column=3)
        assert str(loc) == "5:3"

    def test_source_location_str_without_column(self):
        """Test SourceLocation str without column shows file:line."""
        loc = SourceLocation(file="test.facto", line=5, column=0)
        assert str(loc) == "test.facto:5"

    def test_source_location_str_empty(self):
        """Test SourceLocation str with no data shows 'unknown'."""
        loc = SourceLocation()
        assert str(loc) == "unknown"


class TestSourceLocationRender:
    """Tests for SourceLocation.render() static method."""

    def test_render_with_none_node(self):
        """Test render returns None for None node."""
        result = SourceLocation.render(None)
        assert result is None

    def test_render_with_node_having_source_file_and_line(self):
        """Test render with node having both source_file and line."""

        class MockNode:
            source_file = "/path/to/test.facto"
            line = 42

        result = SourceLocation.render(MockNode())
        assert result == "test.facto:42"

    def test_render_with_default_file(self):
        """Test render uses default_file when node has no source_file."""

        class MockNode:
            source_file = None
            line = 10

        result = SourceLocation.render(MockNode(), default_file="default.facto")
        assert result == "default.facto:10"

    def test_render_with_only_file(self):
        """Test render with only file, no line."""

        class MockNode:
            source_file = "/path/to/test.facto"
            line = 0

        result = SourceLocation.render(MockNode())
        assert result == "test.facto"

    def test_render_with_only_line(self):
        """Test render with only line, no file."""

        class MockNode:
            source_file = None
            line = 25

        result = SourceLocation.render(MockNode())
        assert result == "?:25"

    def test_render_with_nothing(self):
        """Test render returns None when node has no file or line."""

        class MockNode:
            source_file = None
            line = 0

        result = SourceLocation.render(MockNode())
        assert result is None
