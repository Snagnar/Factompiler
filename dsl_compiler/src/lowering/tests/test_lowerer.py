"""
Tests for lowering/lowerer.py - AST lowering to IR.
"""

from pathlib import Path

import pytest

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.ir.nodes import IRArith, IRConst, IRMemWrite, SignalRef
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer


def lower_program(program, semantic_analyzer):
    """Helper to lower a program to IR."""
    diagnostics = ProgramDiagnostics()
    lowerer = ASTLowerer(semantic_analyzer, diagnostics)
    ir_operations = lowerer.lower_program(program)
    return ir_operations, lowerer.diagnostics, lowerer.ir_builder.signal_type_map


class TestASTLowerer:
    """Tests for ASTLowerer class."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    @pytest.fixture
    def diagnostics(self):
        return ProgramDiagnostics()

    @pytest.fixture
    def analyzer(self, diagnostics):
        return SemanticAnalyzer(diagnostics)

    def test_lowerer_initialization(self, analyzer, diagnostics):
        """Test ASTLowerer can be initialized."""
        lowerer = ASTLowerer(analyzer, diagnostics)
        assert lowerer is not None

    def test_basic_lowering(self, parser, analyzer, diagnostics):
        """Test basic lowering to IR."""
        program = parser.parse("Signal x = 42;")
        analyzer.visit(program)
        ir_operations, lower_diags, signal_map = lower_program(program, analyzer)

        assert isinstance(ir_operations, list)
        assert isinstance(lower_diags, ProgramDiagnostics)
        assert isinstance(signal_map, dict)


class TestASTLowererMemoryWrite:
    """Tests for memory write lowering."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    @pytest.fixture
    def diagnostics(self):
        return ProgramDiagnostics()

    @pytest.fixture
    def analyzer(self, diagnostics):
        return SemanticAnalyzer(diagnostics)

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
        assert isinstance(write_op.write_enable, SignalRef)
        assert write_op.write_enable.signal_type == "signal-W"


class TestASTLowererConstantFolding:
    """Tests for constant folding during lowering."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    @pytest.fixture
    def diagnostics(self):
        return ProgramDiagnostics()

    @pytest.fixture
    def analyzer(self, diagnostics):
        return SemanticAnalyzer(diagnostics)

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


class TestASTLowererProjection:
    """Tests for signal projection lowering."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    @pytest.fixture
    def diagnostics(self):
        return ProgramDiagnostics()

    @pytest.fixture
    def analyzer(self, diagnostics):
        return SemanticAnalyzer(diagnostics)

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


class TestASTLowererSampleFiles:
    """Tests for lowering sample files."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_lowering_sample_files(self, parser):
        """Test lowering on sample files if they exist."""
        # Note: Path updated for new example_programs location
        sample_files = [
            "example_programs/01_basic_arithmetic.facto",
            "example_programs/04_memory.facto",
        ]

        for file_path in sample_files:
            path = Path(file_path)
            if path.exists():
                with open(path) as f:
                    code = f.read()
                program = parser.parse(code)
                diagnostics = ProgramDiagnostics()
                analyzer = SemanticAnalyzer(diagnostics)
                analyzer.visit(program)
                ir_operations, lower_diags, signal_map = lower_program(program, analyzer)

                assert isinstance(ir_operations, list)
                assert len(ir_operations) > 0
