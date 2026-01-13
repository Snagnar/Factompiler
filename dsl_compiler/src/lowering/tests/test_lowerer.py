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

from .conftest import lower_program


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

    def test_projection_folding_simple(self, parser, analyzer, diagnostics):
        """Test that simple projections are folded into operations."""
        code = """
        Signal a = ("signal-A", 100);
        Signal b = a * 255 | "iron-plate";
        """
        program = parser.parse(code)
        analyzer.visit(program)
        ir_operations, lower_diags, _ = lower_program(program, analyzer)

        assert not lower_diags.has_errors(), lower_diags.get_messages()

        # Should have 1 IRArith (multiply with direct output to iron-plate), not 2
        ariths = [op for op in ir_operations if isinstance(op, IRArith)]
        assert len(ariths) == 1, f"Expected 1 IRArith, got {len(ariths)}"
        assert ariths[0].output_type == "iron-plate", (
            f"Expected iron-plate output, got {ariths[0].output_type}"
        )

    def test_projection_folding_chained_operations(self, parser, analyzer, diagnostics):
        """Test projection folding with chained arithmetic operations."""
        code = """
        Signal a = ("signal-A", 10);
        Signal b = ("signal-B", 20);
        Signal result = (a + b) * 2 | "copper-plate";
        """
        program = parser.parse(code)
        analyzer.visit(program)
        ir_operations, lower_diags, _ = lower_program(program, analyzer)

        assert not lower_diags.has_errors(), lower_diags.get_messages()

        # Should have 2 IRArith: add and multiply (with copper-plate output)
        # Without folding: would have 3 (add, multiply, projection)
        ariths = [op for op in ir_operations if isinstance(op, IRArith)]
        assert len(ariths) == 2, f"Expected 2 IRArith, got {len(ariths)}"

        # Find the multiply operation and verify it outputs to copper-plate
        multiply_ops = [op for op in ariths if op.op == "*"]
        assert len(multiply_ops) == 1
        assert multiply_ops[0].output_type == "copper-plate"

    def test_projection_no_fold_user_declared(self, parser, analyzer, diagnostics):
        """Test that user-declared signals are NOT folded."""
        code = """
        Signal a = ("signal-A", 100);
        Signal intermediate = a * 255;
        Signal b = intermediate | "iron-plate";
        """
        program = parser.parse(code)
        analyzer.visit(program)
        ir_operations, lower_diags, _ = lower_program(program, analyzer)

        assert not lower_diags.has_errors(), lower_diags.get_messages()

        # 'intermediate' is user-declared, so projection should create a separate operation
        # Should have 2 IRArith: one for multiply, one for projection
        ariths = [op for op in ir_operations if isinstance(op, IRArith)]
        assert len(ariths) == 2, (
            f"Expected 2 IRArith (no folding for user-declared), got {len(ariths)}"
        )

    def test_projection_folding_with_decider(self, parser, analyzer, diagnostics):
        """Test projection folding works with decider operations."""
        code = """
        Signal a = ("signal-A", 100);
        Signal result = (a > 50 : 1) | "signal-X";
        """
        program = parser.parse(code)
        analyzer.visit(program)
        ir_operations, lower_diags, _ = lower_program(program, analyzer)

        assert not lower_diags.has_errors(), lower_diags.get_messages()

        # Should have 1 IRDecider with direct output to signal-X
        from dsl_compiler.src.ir.nodes import IRDecider

        deciders = [op for op in ir_operations if isinstance(op, IRDecider)]
        assert len(deciders) == 1
        assert deciders[0].output_type == "signal-X"


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


def lower_program_full(program, semantic_analyzer):
    """Helper to lower a program and return lowerer instance."""
    diagnostics = ProgramDiagnostics()
    lowerer = ASTLowerer(semantic_analyzer, diagnostics)
    ir_operations = lowerer.lower_program(program)
    return lowerer, ir_operations


class TestASTLowererSignalRegistry:
    """Tests for signal registration during lowering."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    @pytest.fixture
    def diagnostics(self):
        return ProgramDiagnostics()

    @pytest.fixture
    def analyzer(self, diagnostics):
        return SemanticAnalyzer(diagnostics)

    def test_item_signal_registered(self, parser, analyzer, diagnostics):
        """Test that item signals are properly registered in signal_refs."""
        code = 'Signal iron = ("iron-plate", 100);'
        program = parser.parse(code)
        analyzer.visit(program)
        lowerer, ir_operations = lower_program_full(program, analyzer)

        assert not lowerer.diagnostics.has_errors()
        # Signal variable should be registered in signal_refs
        assert "iron" in lowerer.signal_refs

    def test_virtual_signal_registered(self, parser, analyzer, diagnostics):
        """Test that virtual signals are properly registered in signal_refs."""
        code = 'Signal sig = ("signal-A", 50);'
        program = parser.parse(code)
        analyzer.visit(program)
        lowerer, ir_operations = lower_program_full(program, analyzer)

        assert not lowerer.diagnostics.has_errors()
        # Signal variable should be registered in signal_refs
        assert "sig" in lowerer.signal_refs

    def test_implicit_signal_allocated(self, parser, analyzer, diagnostics):
        """Test that implicit signals get allocated during lowering."""
        code = "Signal x = 42;"
        program = parser.parse(code)
        analyzer.visit(program)
        lowerer, ir_operations = lower_program_full(program, analyzer)

        assert not lowerer.diagnostics.has_errors()
        # Signal variable should be registered in signal_refs
        assert "x" in lowerer.signal_refs


class TestASTLowererExpressionContext:
    """Tests for expression context during lowering."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    @pytest.fixture
    def diagnostics(self):
        return ProgramDiagnostics()

    @pytest.fixture
    def analyzer(self, diagnostics):
        return SemanticAnalyzer(diagnostics)

    def test_nested_expressions(self, parser, analyzer, diagnostics):
        """Test lowering nested expressions."""
        code = """
        Signal a = 10;
        Signal b = 20;
        Signal c = (a + b) * 2;
        """
        program = parser.parse(code)
        analyzer.visit(program)
        ir_operations, lower_diags, _ = lower_program(program, analyzer)

        assert not lower_diags.has_errors()
        assert len(ir_operations) > 0

    def test_comparison_expressions(self, parser, analyzer, diagnostics):
        """Test lowering comparison expressions."""
        code = """
        Signal val = 50;
        Signal result = (val > 25): 1;
        """
        program = parser.parse(code)
        analyzer.visit(program)
        ir_operations, lower_diags, _ = lower_program(program, analyzer)

        assert not lower_diags.has_errors()

    def test_function_call_lowering(self, parser, analyzer, diagnostics):
        """Test lowering function calls."""
        code = """
        func triple(int x) { return x * 3; }
        Signal result = triple(10);
        """
        program = parser.parse(code)
        analyzer.visit(program)
        ir_operations, lower_diags, _ = lower_program(program, analyzer)

        assert not lower_diags.has_errors()


class TestSignalCategoryInference:
    """Tests for _infer_signal_category (lines 136-173)."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    @pytest.fixture
    def diagnostics(self):
        return ProgramDiagnostics()

    @pytest.fixture
    def analyzer(self, diagnostics):
        return SemanticAnalyzer(diagnostics)

    def test_infer_item_signal_category(self, parser, analyzer, diagnostics):
        """Test inferring item signal category."""
        code = 'Signal iron = ("iron-plate", 100);'
        program = parser.parse(code)
        analyzer.visit(program)
        lowerer, ir_operations = lower_program_full(program, analyzer)

        # iron-plate is an item type
        category = lowerer._infer_signal_category("iron-plate")
        assert category == "item"

    def test_infer_fluid_signal_category(self, parser, analyzer, diagnostics):
        """Test inferring fluid signal category."""
        code = 'Signal water = ("water", 100);'
        program = parser.parse(code)
        analyzer.visit(program)
        lowerer, ir_operations = lower_program_full(program, analyzer)

        category = lowerer._infer_signal_category("water")
        assert category == "fluid"

    def test_infer_virtual_signal_category(self, parser, analyzer, diagnostics):
        """Test inferring virtual signal category."""
        code = 'Signal sig = ("signal-A", 50);'
        program = parser.parse(code)
        analyzer.visit(program)
        lowerer, ir_operations = lower_program_full(program, analyzer)

        category = lowerer._infer_signal_category("signal-A")
        assert category == "virtual"

    def test_infer_implicit_signal_category(self, parser, analyzer, diagnostics):
        """Test inferring category for implicit signal (__v*)."""
        code = "Signal x = 42;"
        program = parser.parse(code)
        analyzer.visit(program)
        lowerer, ir_operations = lower_program_full(program, analyzer)

        # __v* signals should be virtual
        category = lowerer._infer_signal_category("__v1")
        assert category == "virtual"

    def test_infer_unknown_signal_defaults_to_virtual(self, parser, analyzer, diagnostics):
        """Test that unknown signals default to virtual category."""
        code = "Signal x = 42;"
        program = parser.parse(code)
        analyzer.visit(program)
        lowerer, ir_operations = lower_program_full(program, analyzer)

        category = lowerer._infer_signal_category("unknown-signal-xyz")
        assert category == "virtual"


class TestEnsureSignalRegistered:
    """Tests for ensure_signal_registered (lines 176-198)."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    @pytest.fixture
    def diagnostics(self):
        return ProgramDiagnostics()

    @pytest.fixture
    def analyzer(self, diagnostics):
        return SemanticAnalyzer(diagnostics)

    def test_register_known_item_signal(self, parser, analyzer, diagnostics):
        """Test registering a known item signal."""
        code = 'Signal iron = ("iron-plate", 100);'
        program = parser.parse(code)
        analyzer.visit(program)
        lowerer, _ = lower_program_full(program, analyzer)

        # Should not raise for known signal
        lowerer.ensure_signal_registered("iron-plate")

    def test_register_custom_signal(self, parser, analyzer, diagnostics):
        """Test registering a custom signal name."""
        code = "Signal x = 42;"
        program = parser.parse(code)
        analyzer.visit(program)
        lowerer, _ = lower_program_full(program, analyzer)

        # Register a custom signal
        lowerer.ensure_signal_registered("custom-signal", "virtual")
        # Should be registered in signal registry
        assert lowerer.ir_builder.signal_registry.resolve("custom-signal") is not None

    def test_skip_implicit_signal_registration(self, parser, analyzer, diagnostics):
        """Test that __v* implicit signals are not registered."""
        code = "Signal x = 42;"
        program = parser.parse(code)
        analyzer.visit(program)
        lowerer, _ = lower_program_full(program, analyzer)

        # This should be a no-op
        lowerer.ensure_signal_registered("__v99")
        # __v signals should not be in the registry
        assert lowerer.ir_builder.signal_registry.resolve("__v99") is None

    def test_signal_already_in_data(self, parser, analyzer, diagnostics):
        """Test that signals already in data are not re-registered."""
        code = 'Signal iron = ("iron-plate", 100);'
        program = parser.parse(code)
        analyzer.visit(program)
        lowerer, _ = lower_program_full(program, analyzer)

        # iron-plate is in signal_data, should not raise
        lowerer.ensure_signal_registered("iron-plate")
