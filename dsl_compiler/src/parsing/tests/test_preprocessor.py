"""
Tests for parsing/preprocessor.py - Import preprocessing functionality.
"""

import pytest

from dsl_compiler.src.parsing.preprocessor import (
    preprocess_imports,
    resolve_import_path,
)


class TestResolveImportPath:
    """Tests for resolve_import_path function."""

    def test_resolve_import_from_base_path(self, tmp_path):
        """Test resolving import from provided base_path."""
        lib_file = tmp_path / "lib.facto"
        lib_file.write_text("Signal x = 1;", encoding="utf-8")

        result = resolve_import_path("lib.facto", tmp_path)
        assert result == lib_file.resolve()

    def test_resolve_import_not_found_raises_error(self, tmp_path):
        """Test that missing import raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_import_path("nonexistent.facto", tmp_path)
        assert "Import file not found" in str(exc_info.value)

    def test_resolve_import_from_env_path(self, tmp_path, monkeypatch):
        """Test resolving import from FACTORIO_IMPORT_PATH environment variable."""
        lib_dir = tmp_path / "libs"
        lib_dir.mkdir()
        lib_file = lib_dir / "envlib.facto"
        lib_file.write_text("Signal env = 1;", encoding="utf-8")

        # Set environment variable
        monkeypatch.setenv("FACTORIO_IMPORT_PATH", str(lib_dir))

        # Resolve without base_path - should still find via env
        result = resolve_import_path("envlib.facto", None)
        assert result == lib_file.resolve()


class TestPreprocessImports:
    """Tests for preprocess_imports function."""

    def test_preprocess_no_imports(self):
        """Test preprocessing code without imports."""
        source = "Signal x = 1;\nSignal y = 2;"
        result = preprocess_imports(source)
        assert result == source

    def test_preprocess_simple_import(self, tmp_path):
        """Test preprocessing a simple import statement."""
        lib_file = tmp_path / "lib.facto"
        lib_file.write_text("Signal lib_val = 100;", encoding="utf-8")

        source = 'import "lib.facto";\nSignal x = lib_val + 1;'
        result = preprocess_imports(source, tmp_path)

        assert "lib_val = 100" in result
        assert "Signal x = lib_val + 1" in result
        assert "Imported from" in result

    def test_preprocess_circular_import_skipped(self, tmp_path):
        """Test that circular imports are detected and skipped."""
        # Create two files that import each other
        file_a = tmp_path / "a.facto"
        file_b = tmp_path / "b.facto"

        file_a.write_text('import "b.facto";\nSignal a = 1;', encoding="utf-8")
        file_b.write_text('import "a.facto";\nSignal b = 2;', encoding="utf-8")

        source = 'import "a.facto";'
        result = preprocess_imports(source, tmp_path)

        # Should contain 'a' content but skip circular import of a from b
        assert "Signal a = 1" in result
        assert "Skipped circular import" in result

    def test_preprocess_adds_facto_extension(self, tmp_path):
        """Test that .facto extension is added automatically."""
        lib_file = tmp_path / "mylib.facto"
        lib_file.write_text("Signal lib_signal = 42;", encoding="utf-8")

        # Import without extension
        source = 'import "mylib";'
        result = preprocess_imports(source, tmp_path)

        assert "lib_signal = 42" in result

    def test_preprocess_nested_imports(self, tmp_path):
        """Test preprocessing nested imports."""
        # Create a chain: main -> helper -> utils
        utils_file = tmp_path / "utils.facto"
        utils_file.write_text("Signal util_val = 1;", encoding="utf-8")

        helper_file = tmp_path / "helper.facto"
        helper_file.write_text(
            'import "utils.facto";\nSignal helper_val = util_val + 1;',
            encoding="utf-8",
        )

        source = 'import "helper.facto";\nSignal main_val = helper_val + 1;'
        result = preprocess_imports(source, tmp_path)

        assert "util_val = 1" in result
        assert "helper_val = util_val + 1" in result
        assert "main_val = helper_val + 1" in result

    def test_preprocess_nonexistent_import_preserved(self, tmp_path):
        """Test that import statement for nonexistent file is preserved (edge case)."""
        # This tests the else branch where file doesn't exist but import path resolves
        # In practice this shouldn't happen often since resolve_import_path raises FileNotFoundError
        source = 'import "nonexistent.facto";\nSignal x = 1;'

        # This will raise FileNotFoundError from resolve_import_path
        with pytest.raises(FileNotFoundError):
            preprocess_imports(source, tmp_path)
