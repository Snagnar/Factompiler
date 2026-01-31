"""
Tests for bundle expression lowering.

Tests bundle parsing, semantic analysis, and IR lowering for the Bundle type.
"""

import contextlib

import pytest

from dsl_compiler.src.ast.expressions import (
    BinaryOp,
    BundleAllExpr,
    BundleAnyExpr,
    BundleLiteral,
    BundleSelectExpr,
)
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.ir.builder import SignalRef
from dsl_compiler.src.ir.nodes import IRArith, IRConst, IRDecider
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer
from dsl_compiler.src.semantic.type_system import BundleValue, SignalValue

from .conftest import lower_program


class TestBundleParsing:
    """Test bundle parsing and AST generation."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_parse_simple_bundle_literal(self, parser):
        """Parse a simple bundle literal with signal literals."""
        code = 'Bundle b = { ("iron-plate", 100), ("copper-plate", 80) };'
        program = parser.parse(code)
        assert len(program.statements) == 1
        assert isinstance(program.statements[0].value, BundleLiteral)
        assert len(program.statements[0].value.elements) == 2

    def test_parse_empty_bundle(self, parser):
        """Parse empty bundle literal."""
        code = "Bundle empty = {};"
        program = parser.parse(code)
        assert len(program.statements) == 1
        assert isinstance(program.statements[0].value, BundleLiteral)
        assert len(program.statements[0].value.elements) == 0

    def test_parse_bundle_select(self, parser):
        """Parse bundle selection syntax."""
        code = """
        Bundle b = { ("iron-plate", 100) };
        Signal x = b["iron-plate"];
        """
        program = parser.parse(code)
        assert len(program.statements) == 2
        assert isinstance(program.statements[1].value, BundleSelectExpr)
        assert program.statements[1].value.signal_type == "iron-plate"

    def test_parse_bundle_any(self, parser):
        """Parse any(bundle) expression."""
        code = """
        Bundle b = { ("iron-plate", 100) };
        Signal x = any(b) > 0;
        """
        program = parser.parse(code)
        binary_op = program.statements[1].value
        assert isinstance(binary_op, BinaryOp)
        assert isinstance(binary_op.left, BundleAnyExpr)

    def test_parse_bundle_all(self, parser):
        """Parse all(bundle) expression."""
        code = """
        Bundle b = { ("iron-plate", 100) };
        Signal x = all(b) > 0;
        """
        program = parser.parse(code)
        binary_op = program.statements[1].value
        assert isinstance(binary_op, BinaryOp)
        assert isinstance(binary_op.left, BundleAllExpr)

    def test_parse_bundle_arithmetic(self, parser):
        """Parse bundle arithmetic operations."""
        code = """
        Bundle b = { ("iron-plate", 100) };
        Bundle doubled = b * 2;
        """
        program = parser.parse(code)
        assert len(program.statements) == 2


class TestBundleSemantics:
    """Test bundle semantic analysis and type checking."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_bundle_literal_type_inference(self, parser):
        """Bundle literal should infer BundleValue type."""
        code = 'Bundle b = { ("iron-plate", 100), ("copper-plate", 80) };'
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)

        assert not diagnostics.has_errors(), diagnostics.get_messages()
        symbol = analyzer.symbol_table.lookup("b")
        assert symbol is not None
        assert isinstance(symbol.value_type, BundleValue)
        assert "iron-plate" in symbol.value_type.signal_types
        assert "copper-plate" in symbol.value_type.signal_types

    def test_bundle_select_type_inference(self, parser):
        """Bundle selection should return Signal type."""
        code = """
        Bundle b = { ("iron-plate", 100), ("copper-plate", 80) };
        Signal x = b["iron-plate"];
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)

        assert not diagnostics.has_errors(), diagnostics.get_messages()
        symbol = analyzer.symbol_table.lookup("x")
        assert symbol is not None
        assert isinstance(symbol.value_type, SignalValue)

    def test_bundle_arithmetic_type_inference(self, parser):
        """Bundle arithmetic should return Bundle type."""
        code = """
        Bundle b = { ("iron-plate", 100), ("copper-plate", 80) };
        Bundle doubled = b * 2;
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)

        assert not diagnostics.has_errors(), diagnostics.get_messages()
        symbol = analyzer.symbol_table.lookup("doubled")
        assert symbol is not None
        assert isinstance(symbol.value_type, BundleValue)

    def test_bundle_any_type_inference(self, parser):
        """any(bundle) should return Signal type."""
        code = """
        Bundle b = { ("iron-plate", 100) };
        Signal x = any(b) > 0;
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)

        assert not diagnostics.has_errors(), diagnostics.get_messages()
        symbol = analyzer.symbol_table.lookup("x")
        assert symbol is not None
        assert isinstance(symbol.value_type, SignalValue)

    def test_bundle_all_type_inference(self, parser):
        """all(bundle) should return Signal type."""
        code = """
        Bundle b = { ("iron-plate", 100) };
        Signal x = all(b) > 0;
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)

        assert not diagnostics.has_errors(), diagnostics.get_messages()
        symbol = analyzer.symbol_table.lookup("x")
        assert symbol is not None
        assert isinstance(symbol.value_type, SignalValue)

    def test_bundle_bundle_arithmetic_rejected(self, parser):
        """Bundle + Bundle should be rejected (not supported per spec)."""
        code = """
        Bundle a = { ("iron-plate", 100) };
        Bundle b = { ("copper-plate", 80) };
        Bundle c = a + b;
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)

        with contextlib.suppress(Exception):
            analyzer.visit(program)

        assert diagnostics.has_errors()


class TestBundleLowering:
    """Test bundle lowering to IR."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    @pytest.fixture
    def diagnostics(self):
        return ProgramDiagnostics()

    @pytest.fixture
    def analyzer(self, diagnostics):
        return SemanticAnalyzer(diagnostics)

    def test_bundle_literal_creates_multi_signal_const(self, parser, analyzer, diagnostics):
        """Bundle literal with constants should create multi-signal IRConst."""
        code = 'Bundle b = { ("iron-plate", 100), ("copper-plate", 80), ("coal", 50) };'
        program = parser.parse(code)
        analyzer.visit(program)
        ir_operations, lower_diags, _ = lower_program(program, analyzer)

        assert not lower_diags.has_errors(), lower_diags.get_messages()

        # Should have constants for the bundle
        consts = [op for op in ir_operations if isinstance(op, IRConst)]
        assert len(consts) >= 1

        # Find the bundle constant (should have multiple signals)
        bundle_const = next((c for c in consts if hasattr(c, "signals") and c.signals), None)
        if bundle_const:
            assert len(bundle_const.signals) == 3
            assert bundle_const.signals.get("iron-plate") == 100
            assert bundle_const.signals.get("copper-plate") == 80
            assert bundle_const.signals.get("coal") == 50

    def test_bundle_arithmetic_creates_each_combinator(self, parser, analyzer, diagnostics):
        """Bundle arithmetic should create arithmetic combinator with signal-each."""
        code = """
        Bundle b = { ("iron-plate", 100), ("copper-plate", 80) };
        Bundle doubled = b * 2;
        """
        program = parser.parse(code)
        analyzer.visit(program)
        ir_operations, lower_diags, _ = lower_program(program, analyzer)

        assert not lower_diags.has_errors(), lower_diags.get_messages()

        # Should have arithmetic combinator for bundle * 2
        ariths = [op for op in ir_operations if isinstance(op, IRArith)]
        assert len(ariths) >= 1

        # Find the bundle arithmetic (uses signal-each output)
        bundle_arith = next((a for a in ariths if a.output_type == "signal-each"), None)
        assert bundle_arith is not None
        assert bundle_arith.op == "*"

    def test_bundle_any_comparison_creates_decider(self, parser, analyzer, diagnostics):
        """any(bundle) comparison should create decider with signal-anything input."""
        code = """
        Bundle b = { ("iron-plate", 100) };
        Signal x = any(b) > 0;
        """
        program = parser.parse(code)
        analyzer.visit(program)
        ir_operations, lower_diags, _ = lower_program(program, analyzer)

        assert not lower_diags.has_errors(), lower_diags.get_messages()

        # Should have decider combinator for comparison
        deciders = [op for op in ir_operations if isinstance(op, IRDecider)]
        assert len(deciders) >= 1

        # Find the any() decider (uses signal-anything as left input)
        any_decider = next(
            (
                d
                for d in deciders
                if isinstance(d.left, SignalRef) and d.left.signal_type == "signal-anything"
            ),
            None,
        )
        assert any_decider is not None

    def test_bundle_all_comparison_creates_decider(self, parser, analyzer, diagnostics):
        """all(bundle) comparison should create decider with signal-everything input."""
        code = """
        Bundle b = { ("iron-plate", 100) };
        Signal x = all(b) > 0;
        """
        program = parser.parse(code)
        analyzer.visit(program)
        ir_operations, lower_diags, _ = lower_program(program, analyzer)

        assert not lower_diags.has_errors(), lower_diags.get_messages()

        # Should have decider combinator for comparison
        deciders = [op for op in ir_operations if isinstance(op, IRDecider)]
        assert len(deciders) >= 1

        # Find the all() decider (uses signal-everything as left input)
        all_decider = next(
            (
                d
                for d in deciders
                if isinstance(d.left, SignalRef) and d.left.signal_type == "signal-everything"
            ),
            None,
        )
        assert all_decider is not None


class TestBundleFilter:
    """Test bundle filter pattern: (bundle > 0) : bundle.

    This pattern filters a bundle to include only signals that pass
    the comparison, using a decider combinator with signal-each.
    """

    @pytest.fixture
    def parser(self):
        return DSLParser()

    @pytest.fixture
    def diagnostics(self):
        return ProgramDiagnostics()

    @pytest.fixture
    def analyzer(self, diagnostics):
        return SemanticAnalyzer(diagnostics)

    def test_bundle_filter_parses(self, parser):
        """Bundle filter syntax should parse correctly."""
        code = """
        Bundle b = { ("signal-A", 10), ("signal-B", -5) };
        Bundle filtered = (b > 0) : b;
        """
        program = parser.parse(code)
        assert len(program.statements) == 2

    def test_bundle_filter_type_inference(self, parser, diagnostics, analyzer):
        """Bundle filter should return BundleValue type."""
        code = """
        Bundle b = { ("signal-A", 10), ("signal-B", -5) };
        Bundle filtered = (b > 0) : b;
        """
        program = parser.parse(code)
        analyzer.visit(program)

        assert not diagnostics.has_errors(), diagnostics.get_messages()
        symbol = analyzer.symbol_table.lookup("filtered")
        assert symbol is not None
        assert isinstance(symbol.value_type, BundleValue)

    def test_bundle_filter_with_different_operators(self, parser, diagnostics, analyzer):
        """Bundle filter should work with various comparison operators."""
        operators = [">", "<", ">=", "<=", "==", "!="]
        for op in operators:
            code = f"""
            Bundle b = {{ ("signal-A", 10) }};
            Bundle result = (b {op} 5) : b;
            """
            program = parser.parse(code)
            diagnostics_local = ProgramDiagnostics()
            analyzer_local = SemanticAnalyzer(diagnostics_local)
            analyzer_local.visit(program)

            assert not diagnostics_local.has_errors(), (
                f"Failed for operator {op}: {diagnostics_local.get_messages()}"
            )

    def test_bundle_filter_creates_each_decider(self, parser, analyzer, diagnostics):
        """Bundle filter should create decider with signal-each input and output."""
        code = """
        Bundle b = { ("signal-A", 10), ("signal-B", -5) };
        Bundle filtered = (b > 0) : b;
        """
        program = parser.parse(code)
        analyzer.visit(program)
        ir_operations, lower_diags, _ = lower_program(program, analyzer)

        assert not lower_diags.has_errors(), lower_diags.get_messages()

        # Should have decider combinator for filtering
        deciders = [op for op in ir_operations if isinstance(op, IRDecider)]
        assert len(deciders) >= 1

        # Find the bundle filter decider (uses signal-each as both input and output)
        filter_decider = next(
            (
                d
                for d in deciders
                if (
                    isinstance(d.left, SignalRef)
                    and d.left.signal_type == "signal-each"
                    and d.output_type == "signal-each"
                )
            ),
            None,
        )
        assert filter_decider is not None, "No decider with signal-each input and output found"
        assert filter_decider.copy_count_from_input is True, (
            "Bundle filter should preserve input values"
        )

    def test_bundle_filter_with_constant_output(self, parser, analyzer, diagnostics):
        """Bundle filter with constant output should set copy_count_from_input=False."""
        code = """
        Bundle b = { ("signal-A", 10), ("signal-B", -5) };
        Bundle counts = (b > 0) : 1;
        """
        program = parser.parse(code)
        analyzer.visit(program)
        ir_operations, lower_diags, _ = lower_program(program, analyzer)

        assert not lower_diags.has_errors(), lower_diags.get_messages()

        # Find the decider
        deciders = [op for op in ir_operations if isinstance(op, IRDecider)]
        filter_decider = next(
            (d for d in deciders if d.output_type == "signal-each"),
            None,
        )
        assert filter_decider is not None
        assert filter_decider.copy_count_from_input is False, (
            "Constant output should not copy from input"
        )

    def test_bundle_filter_comparison_operator_preserved(self, parser, analyzer, diagnostics):
        """Bundle filter should preserve the comparison operator."""
        code = """
        Bundle b = { ("signal-A", 10), ("signal-B", -5) };
        Bundle filtered = (b >= 0) : b;
        """
        program = parser.parse(code)
        analyzer.visit(program)
        ir_operations, lower_diags, _ = lower_program(program, analyzer)

        assert not lower_diags.has_errors(), lower_diags.get_messages()

        deciders = [op for op in ir_operations if isinstance(op, IRDecider)]
        filter_decider = next(
            (d for d in deciders if d.output_type == "signal-each"),
            None,
        )
        assert filter_decider is not None
        assert filter_decider.test_op == ">=", f"Expected >= but got {filter_decider.test_op}"

    def test_bundle_filter_with_comparison_value(self, parser, analyzer, diagnostics):
        """Bundle filter should preserve comparison value."""
        code = """
        Bundle b = { ("signal-A", 10), ("signal-B", -5) };
        Bundle filtered = (b > 5) : b;
        """
        program = parser.parse(code)
        analyzer.visit(program)
        ir_operations, lower_diags, _ = lower_program(program, analyzer)

        assert not lower_diags.has_errors(), lower_diags.get_messages()

        deciders = [op for op in ir_operations if isinstance(op, IRDecider)]
        filter_decider = next(
            (d for d in deciders if d.output_type == "signal-each"),
            None,
        )
        assert filter_decider is not None
        assert filter_decider.right == 5, (
            f"Expected comparison value 5 but got {filter_decider.right}"
        )
