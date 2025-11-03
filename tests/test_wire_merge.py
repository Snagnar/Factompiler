"""Tests for wire-merge optimization in the lowerer and emitter."""

import pytest

from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer, analyze_program
from dsl_compiler.src.lowering.lowerer import lower_program
from dsl_compiler.src.emission.emitter import emit_blueprint
from dsl_compiler.src.ir.builder import IR_Arith, IR_WireMerge


@pytest.fixture
def parser():
    return DSLParser()


def _lower_ir(parser: DSLParser, code: str):
    analyzer = SemanticAnalyzer()
    program = parser.parse(code)
    diagnostics = analyze_program(program, strict_types=False, analyzer=analyzer)
    assert not diagnostics.has_errors(), diagnostics.get_messages()
    ir_operations, lower_diags, signal_map = lower_program(program, analyzer)
    assert not lower_diags.has_errors(), lower_diags.get_messages()
    return ir_operations, signal_map


class TestWireMerge:
    def test_simple_addition_creates_wire_merge(self, parser):
        code = """
        Signal iron_a = ("iron-plate", 100);
        Signal iron_b = ("iron-plate", 200);
        Signal total = iron_a + iron_b;
        """
        ir_operations, _ = _lower_ir(parser, code)

        merges = [op for op in ir_operations if isinstance(op, IR_WireMerge)]
        ariths = [op for op in ir_operations if isinstance(op, IR_Arith)]

        assert len(merges) == 1, "Expected one wire-merge operation"
        assert len(merges[0].sources) == 2
        # No arithmetic combinators should be necessary for the addition itself.
        assert all(op.op != "+" for op in ariths)

    def test_addition_chain_creates_single_wire_merge(self, parser):
        code = """
        Signal a = ("iron-plate", 10);
        Signal b = ("iron-plate", 20);
        Signal c = ("iron-plate", 30);
        Signal d = ("iron-plate", 40);
        Signal total = a + b + c + d;
        """

        ir_operations, _ = _lower_ir(parser, code)
        merges = [op for op in ir_operations if isinstance(op, IR_WireMerge)]

        assert len(merges) == 1
        assert len(merges[0].sources) == 4

    def test_computed_values_do_not_wire_merge(self, parser):
        code = """
        Signal a = ("iron-plate", 10);
        Signal computed = a - ("iron-plate", 5);
        Signal total = a + computed;
        """

        ir_operations, _ = _lower_ir(parser, code)

        assert not any(isinstance(op, IR_WireMerge) for op in ir_operations)
        assert any(isinstance(op, IR_Arith) and op.op == "+" for op in ir_operations)

    def test_mixed_types_do_not_wire_merge(self, parser):
        code = """
        Signal iron = ("iron-plate", 100);
        Signal copper = ("copper-plate", 200);
        Signal total = iron + copper;
        """

        ir_operations, _ = _lower_ir(parser, code)

        assert not any(isinstance(op, IR_WireMerge) for op in ir_operations)

    def test_blueprint_entity_count_reduced(self, parser):
        code_with_merge = """
        Signal a = ("iron-plate", 10);
        Signal b = ("iron-plate", 20);
        Signal c = ("iron-plate", 30);
        Signal total = a + b + c;
        """

        ir_merge, signal_map_merge = _lower_ir(parser, code_with_merge)
        blueprint_merge, diagnostics_merge = emit_blueprint(
            ir_merge,
            label="Merge",
            signal_type_map=signal_map_merge,
        )
        assert not diagnostics_merge.has_errors(), diagnostics_merge.get_messages()
        entities_with_merge = len(blueprint_merge.entities)

        code_without_merge = """
        Signal a = ("iron-plate", 10);
        Signal b = ("iron-plate", 20);
        Signal c = ("iron-plate", 30);
        Signal computed_b = b - ("iron-plate", 0);
        Signal computed_c = c - ("iron-plate", 0);
        Signal total = a + computed_b + computed_c;
        """

        ir_no_merge, signal_map_no_merge = _lower_ir(parser, code_without_merge)
        blueprint_no_merge, diagnostics_no_merge = emit_blueprint(
            ir_no_merge,
            label="No Merge",
            signal_type_map=signal_map_no_merge,
        )
        assert not diagnostics_no_merge.has_errors(), (
            diagnostics_no_merge.get_messages()
        )
        entities_without_merge = len(blueprint_no_merge.entities)

        assert entities_with_merge < entities_without_merge

    def test_non_addition_ops_do_not_wire_merge(self, parser):
        code = """
        Signal a = ("iron-plate", 100);
        Signal b = ("iron-plate", 50);
        Signal diff = a - b;
        Signal product = a * b;
        """

        ir_operations, _ = _lower_ir(parser, code)

        assert not any(isinstance(op, IR_WireMerge) for op in ir_operations)
