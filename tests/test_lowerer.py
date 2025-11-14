"""
Tests for lowerer.py - IR lowering functionality.
"""
import os

import pytest
from dsl_compiler.src.ir.nodes import (
    IR_Arith,
    IR_Const,
    IR_MemCreate,
    IR_MemWrite,
    SignalRef,
)
from dsl_compiler.src.lowering.lowerer import lower_program
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics


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
        return SemanticAnalyzer(diagnostics, strict_types=False)

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
            "tests/sample_programs/01_basic_arithmetic.fcdsl",
            "tests/sample_programs/04_memory.fcdsl",
        ]

        for file_path in sample_files:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    code = f.read()
                program = parser.parse(code)
                diagnostics = ProgramDiagnostics()
                analyzer = SemanticAnalyzer(diagnostics, strict_types=False)
                analyzer.visit(program)
                ir_operations, lower_diags, signal_map = lower_program(
                    program, analyzer
                )

                assert isinstance(ir_operations, list)
                assert len(ir_operations) > 0

    def test_write_without_when_uses_signal_w_enable(
        self, parser, analyzer, diagnostics
    ):
        """Ensure lowering injects a signal-W enable when none is provided."""

        code = """
        Memory counter: "iron-plate";
        write(read(counter) + 1, counter);
        """

        program = parser.parse(code)
        analyzer.visit(program)
        ir_operations, lower_diags, _ = lower_program(program, analyzer)

        assert not lower_diags.has_errors(), lower_diags.get_messages()

        mem_writes = [op for op in ir_operations if isinstance(op, IR_MemWrite)]
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

        consts = [op for op in ir_operations if isinstance(op, IR_Const)]
        ariths = [op for op in ir_operations if isinstance(op, IR_Arith)]

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

        ariths = [op for op in ir_operations if isinstance(op, IR_Arith)]
        assert len(ariths) == 0

    def test_memory_initialization_sugar(self, parser):
        """Memory declarations with initializers should emit one-shot writes."""

        code = 'Memory counter: "signal-A" = ("signal-A", 5);'

        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics, strict_types=False)
        analyzer.visit(program)
        assert not diagnostics.has_errors(), diagnostics.get_messages()

        ir_operations, lower_diag, _ = lower_program(program, analyzer)
        assert not lower_diag.has_errors(), lower_diag.get_messages()

        mem_creates = [
            op
            for op in ir_operations
            if isinstance(op, IR_MemCreate) and op.memory_id == "mem_counter"
        ]
        mem_writes = [
            op
            for op in ir_operations
            if isinstance(op, IR_MemWrite) and op.memory_id == "mem_counter"
        ]

        assert len(mem_creates) == 1
        assert len(mem_writes) == 1
        assert mem_writes[0].is_one_shot is True
