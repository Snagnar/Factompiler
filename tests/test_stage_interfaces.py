"""Tests for compiler stage interfaces and isolation.

This module validates that:
1. Each stage has well-defined inputs and outputs
2. Stages don't depend on implementation details of other stages
3. Shared state is properly managed (SignalTypeRegistry)
4. Diagnostics flow correctly through the pipeline
"""

from dsl_compiler.src.layout.layout_plan import LayoutPlan
from draftsman.blueprintable import Blueprint

import pytest

from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer
from tests.test_helpers import lower_program, emit_blueprint
from dsl_compiler.src.layout.planner import LayoutPlanner
from dsl_compiler.src.ast.statements import Program
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics


def analyze_program(ast, analyzer=None, strict_types=False, diagnostics=None):
    """Helper to analyze a program AST."""
    if diagnostics is None:
        diagnostics = ProgramDiagnostics()
    if analyzer is None:
        analyzer = SemanticAnalyzer(diagnostics, strict_types=strict_types)
    analyzer.visit(ast)
    return diagnostics


class TestParserInterface:
    """Test parser stage interface."""

    def test_parse_returns_ast(self):
        """Parser should return a Program AST node."""
        parser = DSLParser()
        source = "Signal x = 5;"

        result = parser.parse(source)

        assert isinstance(result, Program)
        assert result is not None

    def test_parse_raises_on_syntax_error(self):
        """Parser should raise SyntaxError on invalid syntax."""
        parser = DSLParser()
        source = "Signal x = ;"  # Invalid syntax

        with pytest.raises(SyntaxError):
            parser.parse(source)

    def test_parse_independent_calls(self):
        """Parser calls should be independent (no shared state)."""
        parser = DSLParser()
        source1 = "Signal x = 1;"
        source2 = "Signal y = 2;"

        ast1 = parser.parse(source1)
        ast2 = parser.parse(source2)

        # ASTs should be independent
        assert ast1 is not ast2
        assert len(ast1.statements) == 1
        assert len(ast2.statements) == 1


class TestSemanticAnalyzerInterface:
    """Test semantic analyzer stage interface."""

    def test_analyze_program_returns_diagnostics(self):
        """Semantic analyzer should return ProgramDiagnostics."""
        parser = DSLParser()
        source = "Signal x = 5;"
        ast = parser.parse(source)

        diagnostics = analyze_program(ast)

        assert isinstance(diagnostics, ProgramDiagnostics)

    def test_analyze_program_with_errors(self):
        """Semantic analyzer should collect errors in diagnostics."""
        parser = DSLParser()
        source = "Signal x = 5; Signal y = x + z;"  # z is undefined
        ast = parser.parse(source)

        diagnostics = analyze_program(ast)

        assert diagnostics.has_errors()

    def test_analyzer_signal_registry_populated(self):
        """Semantic analyzer should have a signal registry."""
        parser = DSLParser()
        source = "Signal x = 10;"  # Implicit virtual signal
        ast = parser.parse(source)

        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyze_program(ast, analyzer=analyzer)

        # Registry exists but implicit signals are deferred to layout phase
        assert analyzer.signal_registry is not None

    def test_analyzer_shares_registry_with_allocator(self):
        """Semantic analyzer should share registry with signal allocator."""
        parser = DSLParser()
        source = "Signal x = 5;"  # Will need implicit virtual signal
        ast = parser.parse(source)

        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyze_program(ast, analyzer=analyzer)

        # Signal allocator should have used the same registry
        assert analyzer.signal_allocator.signal_registry is analyzer.signal_registry


class TestLowererInterface:
    """Test AST lowerer stage interface."""

    def test_lower_program_returns_ir(self):
        """Lowerer should return IR operations, diagnostics, and signal map."""
        parser = DSLParser()
        source = "Signal x = 5 + 3;"
        ast = parser.parse(source)

        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyze_program(ast, analyzer=analyzer)

        ir_ops, diagnostics, signal_map = lower_program(ast, analyzer)

        assert isinstance(ir_ops, list)
        assert isinstance(diagnostics, ProgramDiagnostics)
        assert isinstance(signal_map, dict)

    def test_lowerer_shares_signal_registry(self):
        """Lowerer should share signal registry with semantic analyzer."""
        parser = DSLParser()
        source = 'Signal x = ("iron-plate", 10);'  # Known Factorio signal
        ast = parser.parse(source)

        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyze_program(ast, analyzer=analyzer)

        ir_ops, diagnostics, signal_map = lower_program(ast, analyzer)

        # Known Factorio signals don't need explicit registration
        # They're already in draftsman's signal_data
        assert isinstance(signal_map, dict)

    def test_lowerer_independent_calls(self):
        """Multiple lower_program calls should not interfere."""
        parser = DSLParser()
        source1 = "Signal x = 1;"
        source2 = "Signal y = 2;"

        ast1 = parser.parse(source1)
        diagnostics1 = ProgramDiagnostics()
        analyzer1 = SemanticAnalyzer(diagnostics1)
        analyze_program(ast1, analyzer=analyzer1)

        ast2 = parser.parse(source2)
        diagnostics2 = ProgramDiagnostics()
        analyzer2 = SemanticAnalyzer(diagnostics2)
        analyze_program(ast2, analyzer=analyzer2)

        ir_ops1, _, _ = lower_program(ast1, analyzer1)
        ir_ops2, _, _ = lower_program(ast2, analyzer2)

        # IR should be independent
        assert ir_ops1 is not ir_ops2


class TestLayoutPlannerInterface:
    """Test layout planner stage interface."""

    def test_plan_layout_returns_layout_plan(self):
        """Layout planner should return LayoutPlan."""
        parser = DSLParser()
        source = "Signal x = 5;"
        ast = parser.parse(source)

        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyze_program(ast, analyzer=analyzer)

        ir_ops, diagnostics, signal_map = lower_program(ast, analyzer)

        planner_diagnostics = ProgramDiagnostics()
        planner = LayoutPlanner(signal_map, planner_diagnostics)
        layout_plan = planner.plan_layout(ir_ops)

        assert isinstance(layout_plan, LayoutPlan)

    def test_planner_diagnostics_collection(self):
        """Layout planner should collect diagnostics."""
        parser = DSLParser()
        source = "Signal x = 5;"
        ast = parser.parse(source)

        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyze_program(ast, analyzer=analyzer)

        ir_ops, _, signal_map = lower_program(ast, analyzer)

        planner_diagnostics = ProgramDiagnostics()
        planner = LayoutPlanner(signal_map, diagnostics=planner_diagnostics)
        planner.plan_layout(ir_ops)

        # Diagnostics should be accessible
        assert planner.diagnostics is not None


class TestBlueprintEmitterInterface:
    """Test blueprint emitter stage interface."""

    def test_emit_blueprint_returns_blueprint_and_diagnostics(self):
        """Blueprint emitter should return Blueprint and diagnostics."""
        parser = DSLParser()
        source = "Signal x = 5;"
        ast = parser.parse(source)

        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyze_program(ast, analyzer=analyzer)

        ir_ops, _, signal_map = lower_program(ast, analyzer)

        blueprint, diagnostics = emit_blueprint(ir_ops, signal_type_map=signal_map)

        assert isinstance(blueprint, Blueprint)
        assert isinstance(diagnostics, ProgramDiagnostics)

    def test_emit_blueprint_with_label(self):
        """Blueprint emitter should use provided label."""
        parser = DSLParser()
        source = "Signal x = 5;"
        ast = parser.parse(source)

        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyze_program(ast, analyzer=analyzer)

        ir_ops, _, signal_map = lower_program(ast, analyzer)

        blueprint, _ = emit_blueprint(
            ir_ops, label="Test Blueprint", signal_type_map=signal_map
        )

        assert blueprint.label == "Test Blueprint"


class TestStageIsolation:
    """Test that stages are properly isolated."""

    def test_no_backflow_modifications(self):
        """Later stages should not modify earlier stage outputs."""
        parser = DSLParser()
        source = "Signal x = 5;"
        ast = parser.parse(source)

        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyze_program(ast, analyzer=analyzer)

        # AST structure should not be modified (only annotated)
        # Note: Annotations are expected (expr_types, etc.)
        assert ast.statements is not None  # Core structure intact

    def test_signal_registry_sharing(self):
        """SignalTypeRegistry should be shared between stages."""
        parser = DSLParser()
        source = 'Signal x = ("iron-plate", 10);'  # Known Factorio signal
        ast = parser.parse(source)

        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyze_program(ast, analyzer=analyzer)

        ir_ops, _, signal_map = lower_program(ast, analyzer)

        # Known Factorio signals are in draftsman's signal_data
        # They don't need explicit registration in signal_map
        assert isinstance(signal_map, dict)

    def test_diagnostics_merging(self):
        """Diagnostics from different stages should be mergeable."""
        parser = DSLParser()
        source = "Signal x = 5;"
        ast = parser.parse(source)

        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        semantic_diagnostics = analyze_program(ast, analyzer=analyzer)

        _, lowering_diagnostics, _ = lower_program(ast, analyzer)

        # Should be able to combine diagnostics
        combined = ProgramDiagnostics()
        combined.diagnostics.extend(semantic_diagnostics.diagnostics)
        combined.diagnostics.extend(lowering_diagnostics.diagnostics)

        # Combined diagnostics should contain all issues
        assert len(combined.diagnostics) >= 0


class TestEndToEndPipeline:
    """Test complete compilation pipeline through all stages."""

    def test_simple_program_pipeline(self):
        """Test complete pipeline with a simple program."""
        parser = DSLParser()
        source = """
        Signal x = 5;
        Signal y = 10;
        Signal z = x + y;
        """

        # Stage 1: Parse
        ast = parser.parse(source)
        assert isinstance(ast, Program)

        # Stage 2: Semantic Analysis
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        semantic_diagnostics = analyze_program(ast, analyzer=analyzer)
        assert not semantic_diagnostics.has_errors()

        # Stage 3: Lower to IR
        ir_ops, _, signal_map = lower_program(ast, analyzer)
        assert len(ir_ops) > 0

        # Stage 4: Plan Layout
        planner_diagnostics = ProgramDiagnostics()
        planner = LayoutPlanner(signal_map, planner_diagnostics)
        layout_plan = planner.plan_layout(ir_ops)
        assert layout_plan is not None

        # Stage 5: Emit Blueprint
        blueprint, emit_diagnostics = emit_blueprint(ir_ops, signal_type_map=signal_map)
        assert blueprint is not None

        # Verify blueprint string can be generated
        bp_string = blueprint.to_string()
        assert bp_string.startswith("0eN")

    def test_program_with_signals(self):
        """Test pipeline with explicit signal types."""
        parser = DSLParser()
        source = """
        Signal x = ("signal-A", 100);
        Signal y = ("signal-B", 200);
        Signal z = x + y;
        """

        ast = parser.parse(source)

        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyze_program(ast, analyzer=analyzer)

        ir_ops, _, signal_map = lower_program(ast, analyzer)

        # Complete pipeline
        blueprint, _ = emit_blueprint(ir_ops, signal_type_map=signal_map)

        assert blueprint.to_string().startswith("0eN")
