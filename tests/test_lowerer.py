"""
Tests for lowerer.py - IR lowering functionality.
"""

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
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer, analyze_program


class TestLowerer:
    """Test IR lowering functionality."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    @pytest.fixture
    def analyzer(self):
        return SemanticAnalyzer()

    def test_basic_lowering(self, parser, analyzer):
        """Test basic lowering to IR."""
        program = parser.parse("Signal x = 42;")
        analyze_program(program, strict_types=False, analyzer=analyzer)
        ir_operations, diagnostics, signal_map = lower_program(program, analyzer)

        assert isinstance(ir_operations, list)
        # lower_program returns ProgramDiagnostics, not list
        from dsl_compiler.src.common.diagnostics import ProgramDiagnostics

        assert isinstance(diagnostics, ProgramDiagnostics)
        assert isinstance(signal_map, dict)

    def test_lowering_sample_files(self, parser, analyzer):
        """Test lowering on sample files."""
        import os

        sample_files = [
            "tests/sample_programs/01_basic_arithmetic.fcdsl",
            "tests/sample_programs/04_memory.fcdsl",
        ]

        for file_path in sample_files:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    code = f.read()
                program = parser.parse(code)
                analyze_program(program, strict_types=False, analyzer=analyzer)
                ir_operations, diagnostics, signal_map = lower_program(
                    program, analyzer
                )

                assert isinstance(ir_operations, list)
                assert len(ir_operations) > 0

    def test_write_without_when_uses_signal_w_enable(self, parser, analyzer):
        """Ensure lowering injects a signal-W enable when none is provided."""

        code = """
        Memory counter: "iron-plate";
        write(read(counter) + 1, counter);
        """

        program = parser.parse(code)
        analyze_program(program, strict_types=False, analyzer=analyzer)
        ir_operations, diagnostics, _ = lower_program(program, analyzer)

        assert not diagnostics.has_errors(), diagnostics.get_messages()

        mem_writes = [op for op in ir_operations if isinstance(op, IR_MemWrite)]
        assert mem_writes, "Expected at least one memory write operation"

        write_op = mem_writes[0]
        assert isinstance(write_op.write_enable, SignalRef), (
            "Write enable should be materialized as a SignalRef"
        )
        assert write_op.write_enable.signal_type == "signal-W", (
            "Write enable should default to signal-W"
        )

    def test_constant_folding_eliminates_arith(self, parser, analyzer):
        """Constant arithmetic should fold to IR constants with no combinators."""

        code = """
        Signal a = 10 + 20;
        Signal b = 100 * 2;
        Signal c = 50 / 5;
        """

        program = parser.parse(code)
        analyze_program(program, strict_types=False, analyzer=analyzer)
        ir_operations, diagnostics, _ = lower_program(program, analyzer)

        assert not diagnostics.has_errors(), diagnostics.get_messages()

        consts = [op for op in ir_operations if isinstance(op, IR_Const)]
        ariths = [op for op in ir_operations if isinstance(op, IR_Arith)]

        assert len(consts) >= 3
        assert len(ariths) == 0

    def test_redundant_projection_eliminated(self, parser, analyzer):
        """Projecting to the same signal should not create arithmetic nodes."""

        code = """
        Signal iron = ("iron-plate", 100);
        Signal same = iron | "iron-plate";
        """

        program = parser.parse(code)
        analyze_program(program, strict_types=False, analyzer=analyzer)
        ir_operations, diagnostics, _ = lower_program(program, analyzer)

        assert not diagnostics.has_errors(), diagnostics.get_messages()

        ariths = [op for op in ir_operations if isinstance(op, IR_Arith)]
        assert len(ariths) == 0

    def test_memory_initialization_sugar(self, parser, analyzer):
        """Memory declarations with initializers should emit one-shot writes."""

        code = 'Memory counter: "signal-A" = ("signal-A", 5);'

        program = parser.parse(code)
        diagnostics = analyze_program(program, strict_types=False, analyzer=analyzer)
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
