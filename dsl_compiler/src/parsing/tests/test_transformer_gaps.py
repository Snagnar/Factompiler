"""Tests for transformer.py coverage gaps."""

import pytest

from dsl_compiler.src.parsing.parser import DSLParser


class TestTransformerOutputSpecMultiple:
    """Cover lines 210-216: multiple output specs."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_multiple_output_specs_parsed(self, parser):
        """Test parsing expression with multiple output specs."""
        code = """
        Signal x = 5 | "signal-A";
        Signal y = 10 | "signal-B";
        Signal result = (x > 0) : 1;
        """
        program = parser.parse(code)
        assert program is not None


class TestTransformerBundleLiteral:
    """Cover lines 246-251: bundle literal parsing."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_bundle_literal_empty(self, parser):
        """Test empty bundle literal."""
        code = """
        Bundle empty = {};
        """
        program = parser.parse(code)
        assert program is not None

    def test_bundle_literal_single_item(self, parser):
        """Test bundle with single item."""
        code = """
        Bundle single = {("iron-plate", 10)};
        """
        program = parser.parse(code)
        assert program is not None

    def test_bundle_literal_multiple_items(self, parser):
        """Test bundle with multiple items."""
        code = """
        Bundle multi = {("iron-plate", 10), ("copper-plate", 20), ("steel-plate", 5)};
        """
        program = parser.parse(code)
        assert program is not None


class TestTransformerSignalLiteral:
    """Cover lines 297-303: signal literal parsing."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_signal_literal_tuple_form(self, parser):
        """Test signal literal in tuple form."""
        code = """
        Signal s = ("signal-A", 5);
        """
        program = parser.parse(code)
        assert program is not None

    def test_signal_literal_projection_form(self, parser):
        """Test signal literal with projection operator."""
        code = """
        Signal s = 5 | "signal-A";
        """
        program = parser.parse(code)
        assert program is not None

    def test_signal_literal_expression_value(self, parser):
        """Test signal literal with expression as value."""
        code = """
        Signal a = 5 | "signal-A";
        Signal b = (a + 10) | "signal-B";
        """
        program = parser.parse(code)
        assert program is not None


class TestTransformerPropertyAccess:
    """Cover lines 364-368: property access parsing."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_property_access_signal_type(self, parser):
        """Test .type property access on signal."""
        code = """
        Signal a = 5 | "signal-A";
        Signal b = 10 | a.type;
        """
        program = parser.parse(code)
        assert program is not None

    def test_property_access_entity(self, parser):
        """Test property access on entity."""
        code = """
        Entity lamp = place("small-lamp", 0, 0);
        lamp.enable = 1;
        """
        program = parser.parse(code)
        assert program is not None

    def test_property_access_memory_read(self, parser):
        """Test .read() method access on memory."""
        code = """
        Memory mem: "signal-A";
        mem.write(1);
        Signal val = mem.read();
        """
        program = parser.parse(code)
        assert program is not None


class TestTransformerFunctionDecl:
    """Cover lines 911-913: function declaration parsing."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_function_decl_no_params(self, parser):
        """Test function with no parameters."""
        code = """
        func simple() {
            return 5;
        }
        """
        program = parser.parse(code)
        assert program is not None

    def test_function_decl_multiple_params(self, parser):
        """Test function with multiple parameters."""
        code = """
        func add(Signal x, Signal y) {
            return x + y;
        }
        Signal result = add(1, 2);
        """
        program = parser.parse(code)
        assert program is not None

    def test_function_decl_entity_param(self, parser):
        """Test function with Entity parameter."""
        code = """
        func configure_lamp(Entity lamp, Signal brightness) {
            lamp.enable = brightness > 0;
        }
        Entity my_lamp = place("small-lamp", 0, 0);
        Signal b = 100 | "signal-B";
        configure_lamp(my_lamp, b);
        """
        program = parser.parse(code)
        assert program is not None

    def test_function_decl_int_param(self, parser):
        """Test function with int parameter."""
        code = """
        func scale(Signal x, int factor) {
            return x * factor;
        }
        Signal result = scale(5 | "signal-A", 10);
        """
        program = parser.parse(code)
        assert program is not None
