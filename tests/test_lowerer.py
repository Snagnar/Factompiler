"""
Tests for lowerer.py - IR lowering functionality.
"""

import os

import pytest

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.ir.nodes import (
    IRArith,
    IRConst,
    IRMemWrite,
    SignalRef,
)
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer
from tests.test_helpers import lower_program


class TestLowerer:
    """Test IR lowering functionality."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    @pytest.fixture
    def diagnostics(self):
        return ProgramDiagnostics()

    @pytest.fixture
    def analyzer(self, diagnostics):
        return SemanticAnalyzer(diagnostics)

    def test_basic_lowering(self, parser, analyzer, diagnostics):
        """Test basic lowering to IR."""
        program = parser.parse("Signal x = 42;")
        analyzer.visit(program)
        ir_operations, lower_diags, signal_map = lower_program(program, analyzer)

        assert isinstance(ir_operations, list)
        assert isinstance(lower_diags, ProgramDiagnostics)
        assert isinstance(signal_map, dict)

    def test_lowering_sample_files(self, parser):
        """Test lowering on sample files."""

        sample_files = [
            "tests/sample_programs/01_basic_arithmetic.facto",
            "tests/sample_programs/04_memory.facto",
        ]

        for file_path in sample_files:
            if os.path.exists(file_path):
                with open(file_path) as f:
                    code = f.read()
                program = parser.parse(code)
                diagnostics = ProgramDiagnostics()
                analyzer = SemanticAnalyzer(diagnostics)
                analyzer.visit(program)
                ir_operations, lower_diags, signal_map = lower_program(program, analyzer)

                assert isinstance(ir_operations, list)
                assert len(ir_operations) > 0

    def test_write_without_when_uses_signal_w_enable(self, parser, analyzer, diagnostics):
        """Ensure lowering injects a signal-W enable when none is provided."""

        code = """
        Memory counter: "iron-plate";
        counter.write(counter.read() + 1);
        """

        program = parser.parse(code)
        analyzer.visit(program)
        ir_operations, lower_diags, _ = lower_program(program, analyzer)

        assert not lower_diags.has_errors(), lower_diags.get_messages()

        mem_writes = [op for op in ir_operations if isinstance(op, IRMemWrite)]
        assert mem_writes, "Expected at least one memory write operation"

        write_op = mem_writes[0]
        assert isinstance(write_op.write_enable, SignalRef), (
            "Write enable should be materialized as a SignalRef"
        )
        assert write_op.write_enable.signal_type == "signal-W", (
            "Write enable should default to signal-W"
        )

    def test_constant_folding_eliminates_arith(self, parser, analyzer, diagnostics):
        """Constant arithmetic should fold to IR constants with no combinators."""

        code = """
        Signal a = 10 + 20;
        Signal b = 100 * 2;
        Signal c = 50 / 5;
        """

        program = parser.parse(code)
        analyzer.visit(program)
        ir_operations, lower_diags, _ = lower_program(program, analyzer)

        assert not lower_diags.has_errors(), lower_diags.get_messages()

        consts = [op for op in ir_operations if isinstance(op, IRConst)]
        ariths = [op for op in ir_operations if isinstance(op, IRArith)]

        assert len(consts) >= 3
        assert len(ariths) == 0

    def test_redundant_projection_eliminated(self, parser, analyzer, diagnostics):
        """Projecting to the same signal should not create arithmetic nodes."""

        code = """
        Signal iron = ("iron-plate", 100);
        Signal same = iron | "iron-plate";
        """

        program = parser.parse(code)
        analyzer.visit(program)
        ir_operations, lower_diags, _ = lower_program(program, analyzer)

        assert not lower_diags.has_errors(), lower_diags.get_messages()

        ariths = [op for op in ir_operations if isinstance(op, IRArith)]
        assert len(ariths) == 0
