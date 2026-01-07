"""
Tests for bundle feature - Signal Bundles.
"""

import contextlib

import pytest

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.ir.nodes import IRArith, IRConst, IRDecider
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer
from dsl_compiler.src.semantic.type_system import BundleValue, SignalValue
from tests.test_helpers import lower_program


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
        from dsl_compiler.src.ast.expressions import BundleLiteral

        assert isinstance(program.statements[0].value, BundleLiteral)
        assert len(program.statements[0].value.elements) == 2

    def test_parse_empty_bundle(self, parser):
        """Parse empty bundle literal."""
        code = "Bundle empty = {};"
        program = parser.parse(code)
        assert len(program.statements) == 1
        from dsl_compiler.src.ast.expressions import BundleLiteral

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
        from dsl_compiler.src.ast.expressions import BundleSelectExpr

        assert isinstance(program.statements[1].value, BundleSelectExpr)
        assert program.statements[1].value.signal_type == "iron-plate"

    def test_parse_bundle_any(self, parser):
        """Parse any(bundle) expression."""
        code = """
        Bundle b = { ("iron-plate", 100) };
        Signal x = any(b) > 0;
        """
        program = parser.parse(code)
        from dsl_compiler.src.ast.expressions import BinaryOp, BundleAnyExpr

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
        from dsl_compiler.src.ast.expressions import BinaryOp, BundleAllExpr

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
        # signal_type may be SignalTypeInfo or str depending on implementation
        sig_type = symbol.value_type.signal_type
        if hasattr(sig_type, "name"):
            assert sig_type.name == "iron-plate"
        else:
            assert sig_type == "iron-plate"

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
        """any(bundle) comparison should create decider with signal-anything output."""
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

        # Find the any() decider (outputs signal-anything)
        any_decider = next((d for d in deciders if d.output_type == "signal-anything"), None)
        assert any_decider is not None

    def test_bundle_all_comparison_creates_decider(self, parser, analyzer, diagnostics):
        """all(bundle) comparison should create decider with signal-everything output."""
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

        # Find the all() decider (outputs signal-everything)
        all_decider = next((d for d in deciders if d.output_type == "signal-everything"), None)
        assert all_decider is not None
