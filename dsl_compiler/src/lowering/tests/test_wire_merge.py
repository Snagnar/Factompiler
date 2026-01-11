"""
Tests for wire-merge optimization in the lowerer.
"""

import pytest

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.ir.builder import IRArith, IRWireMerge
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer


def lower_program(program, semantic_analyzer):
    """Helper to lower a program to IR."""
    diagnostics = ProgramDiagnostics()
    lowerer = ASTLowerer(semantic_analyzer, diagnostics)
    ir_operations = lowerer.lower_program(program)
    return ir_operations, lowerer.diagnostics, lowerer.ir_builder.signal_type_map


@pytest.fixture
def parser():
    return DSLParser()


def _lower_ir(parser: DSLParser, code: str):
    diagnostics = ProgramDiagnostics()
    analyzer = SemanticAnalyzer(diagnostics)
    program = parser.parse(code)
    analyzer.visit(program)
    assert not diagnostics.has_errors(), diagnostics.get_messages()
    ir_operations, lower_diags, signal_map = lower_program(program, analyzer)
    assert not lower_diags.has_errors(), lower_diags.get_messages()
    return ir_operations, signal_map


class TestWireMergeDetection:
    """Tests for wire merge detection during lowering."""

    def test_simple_addition_creates_wire_merge(self, parser):
        """Simple same-signal addition should create wire merge."""
        code = """
        Signal iron_a = ("iron-plate", 100);
        Signal iron_b = ("iron-plate", 200);
        Signal total = iron_a + iron_b;
        """
        ir_operations, _ = _lower_ir(parser, code)

        merges = [op for op in ir_operations if isinstance(op, IRWireMerge)]
        ariths = [op for op in ir_operations if isinstance(op, IRArith)]

        assert len(merges) == 1, "Expected one wire-merge operation"
        assert len(merges[0].sources) == 2
        # No arithmetic combinators should be necessary for the addition itself.
        assert all(op.op != "+" for op in ariths)

    def test_addition_chain_creates_single_wire_merge(self, parser):
        """Chained same-signal additions should create single wire merge."""
        code = """
        Signal a = ("iron-plate", 10);
        Signal b = ("iron-plate", 20);
        Signal c = ("iron-plate", 30);
        Signal d = ("iron-plate", 40);
        Signal total = a + b + c + d;
        """
        ir_operations, _ = _lower_ir(parser, code)
        merges = [op for op in ir_operations if isinstance(op, IRWireMerge)]

        assert len(merges) == 1
        assert len(merges[0].sources) == 4


class TestWireMergeExclusions:
    """Tests for cases where wire merge should NOT be applied."""

    def test_computed_values_do_not_wire_merge(self, parser):
        """Computed values (not just constants) should not wire merge."""
        code = """
        Signal a = ("iron-plate", 10);
        Signal computed = a - ("iron-plate", 5);
        Signal total = a + computed;
        """
        ir_operations, _ = _lower_ir(parser, code)

        assert not any(isinstance(op, IRWireMerge) for op in ir_operations)
        assert any(isinstance(op, IRArith) and op.op == "+" for op in ir_operations)

    def test_mixed_types_do_not_wire_merge(self, parser):
        """Different signal types should not wire merge."""
        code = """
        Signal iron = ("iron-plate", 100);
        Signal copper = ("copper-plate", 200);
        Signal total = iron + copper;
        """
        ir_operations, _ = _lower_ir(parser, code)

        assert not any(isinstance(op, IRWireMerge) for op in ir_operations)

    def test_non_addition_ops_do_not_wire_merge(self, parser):
        """Non-addition operations should not wire merge."""
        code = """
        Signal a = ("iron-plate", 100);
        Signal b = ("iron-plate", 50);
        Signal diff = a - b;
        Signal product = a * b;
        """
        ir_operations, _ = _lower_ir(parser, code)

        assert not any(isinstance(op, IRWireMerge) for op in ir_operations)
