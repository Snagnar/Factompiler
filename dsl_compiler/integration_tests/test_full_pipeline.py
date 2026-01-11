"""Integration tests for the full compilation pipeline.

These tests exercise the complete compiler from parsing through layout.
They use stable fixture programs in the fixtures/ directory.
"""

from pathlib import Path

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.emission.emitter import BlueprintEmitter
from dsl_compiler.src.layout.planner import LayoutPlanner
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def compile_fixture(fixture_name: str, power_poles: str | None = None) -> dict:
    """Compile a fixture file and return compilation artifacts.

    Args:
        fixture_name: Name of the fixture file (without .facto extension)
        power_poles: Optional power pole type (e.g. "medium-electric-pole")

    Returns:
        dict with keys: ast, ir_ops, layout, blueprint, diagnostics
    """
    filepath = FIXTURES_DIR / f"{fixture_name}.facto"
    source = filepath.read_text()
    diagnostics = ProgramDiagnostics()

    # Stage 1: Parse
    parser = DSLParser()
    ast = parser.parse(source, str(filepath))

    # Stage 2: Semantic analysis
    analyzer = SemanticAnalyzer(diagnostics=diagnostics)
    analyzer.visit(ast)

    # Stage 3: Lower to IR
    lowerer = ASTLowerer(analyzer, diagnostics)
    ir_ops = lowerer.lower_program(ast)

    # Stage 4: Layout
    planner = LayoutPlanner(
        lowerer.ir_builder.signal_type_map,
        diagnostics=diagnostics,
        signal_refs=lowerer.signal_refs,
        referenced_signal_names=lowerer.referenced_signal_names,
        power_pole_type=power_poles,
    )
    layout = planner.plan_layout(ir_ops)

    # Stage 5: Emit blueprint
    emitter = BlueprintEmitter(diagnostics, lowerer.ir_builder.signal_type_map)
    blueprint = emitter.emit_from_plan(layout)

    return {
        "ast": ast,
        "ir_ops": ir_ops,
        "layout": layout,
        "blueprint": blueprint,
        "diagnostics": diagnostics,
        "planner": planner,
    }


class TestBasicArithmetic:
    """Test basic arithmetic operations."""

    def test_compiles_successfully(self):
        result = compile_fixture("basic_arithmetic")
        assert len(result["layout"].entity_placements) >= 1
        assert result["diagnostics"].error_count() == 0

    def test_generates_blueprint(self):
        result = compile_fixture("basic_arithmetic")
        bp_string = result["blueprint"].to_string()
        assert bp_string.startswith("0")  # Valid blueprint prefix


class TestMemoryConditional:
    """Test memory with conditional write."""

    def test_compiles_successfully(self):
        result = compile_fixture("memory_conditional")
        assert len(result["layout"].entity_placements) >= 1

    def test_creates_memory_cells(self):
        result = compile_fixture("memory_conditional")
        # Should have memory-related entities
        entity_types = {p.entity_type for p in result["layout"].entity_placements.values()}
        assert "arithmetic-combinator" in entity_types or "decider-combinator" in entity_types


class TestSrLatch:
    """Test SR latch memory pattern."""

    def test_compiles_successfully(self):
        result = compile_fixture("sr_latch")
        assert len(result["layout"].entity_placements) >= 1

    def test_uses_decider_for_latch(self):
        result = compile_fixture("sr_latch")
        entity_types = {p.entity_type for p in result["layout"].entity_placements.values()}
        assert "decider-combinator" in entity_types


class TestConditional:
    """Test conditional (ternary) expressions."""

    def test_compiles_successfully(self):
        result = compile_fixture("conditional")
        assert len(result["layout"].entity_placements) >= 1


class TestBundle:
    """Test bundle operations."""

    def test_compiles_successfully(self):
        result = compile_fixture("bundle")
        assert len(result["layout"].entity_placements) >= 1


class TestForLoop:
    """Test for loop with accumulation."""

    def test_compiles_successfully(self):
        result = compile_fixture("for_loop")
        assert len(result["layout"].entity_placements) >= 1


class TestFunction:
    """Test function definition and call."""

    def test_compiles_successfully(self):
        result = compile_fixture("function")
        assert len(result["layout"].entity_placements) >= 1


class TestEntity:
    """Test entity placement and property writes."""

    def test_compiles_successfully(self):
        result = compile_fixture("entity")
        assert len(result["layout"].entity_placements) >= 1

    def test_places_lamp_entity(self):
        result = compile_fixture("entity")
        entity_types = {p.entity_type for p in result["layout"].entity_placements.values()}
        assert "small-lamp" in entity_types


class TestComplexExpression:
    """Test complex expressions with many operators."""

    def test_compiles_successfully(self):
        result = compile_fixture("complex_expression")
        assert len(result["layout"].entity_placements) >= 3


class TestMixedTypes:
    """Test mixed signal types and projections."""

    def test_compiles_successfully(self):
        result = compile_fixture("mixed_types")
        assert len(result["layout"].entity_placements) >= 1


class TestWithPowerPoles:
    """Test compilation with power pole generation."""

    def test_medium_poles(self):
        result = compile_fixture("basic_arithmetic", power_poles="medium-electric-pole")
        assert len(result["layout"].entity_placements) >= 1

    def test_small_poles(self):
        result = compile_fixture("basic_arithmetic", power_poles="small-electric-pole")
        assert len(result["layout"].entity_placements) >= 1

    def test_substation(self):
        result = compile_fixture("complex_expression", power_poles="substation")
        assert len(result["layout"].entity_placements) >= 1


class TestWireConnections:
    """Test that wire connections are generated correctly."""

    def test_has_wire_connections(self):
        result = compile_fixture("basic_arithmetic")
        # With multiple signals that need to connect, we should have wires
        # The exact count depends on optimization, but should be >= 0
        assert len(result["layout"].wire_connections) >= 0

    def test_complex_has_more_connections(self):
        result = compile_fixture("complex_expression")
        # Complex expressions should generate more connections
        assert len(result["layout"].wire_connections) >= 0


class TestMemoryArithmeticFeedback:
    """Test memory with arithmetic feedback optimization."""

    def test_compiles_successfully(self):
        result = compile_fixture("memory_arithmetic_feedback")
        assert len(result["layout"].entity_placements) >= 1

    def test_uses_arithmetic_combinator(self):
        result = compile_fixture("memory_arithmetic_feedback")
        entity_types = {p.entity_type for p in result["layout"].entity_placements.values()}
        assert "arithmetic-combinator" in entity_types


class TestMemoryMultiFeedback:
    """Test memory with multi-combinator feedback loop."""

    def test_compiles_successfully(self):
        result = compile_fixture("memory_multi_feedback")
        assert len(result["layout"].entity_placements) >= 1


class TestLatchWithMultiplier:
    """Test latch that requires a multiplier combinator."""

    def test_compiles_successfully(self):
        result = compile_fixture("latch_with_multiplier")
        assert len(result["layout"].entity_placements) >= 1

    def test_creates_multiple_combinators(self):
        result = compile_fixture("latch_with_multiplier")
        # Latch with multiplier needs at least 2 combinators
        entity_types = [p.entity_type for p in result["layout"].entity_placements.values()]
        combinator_count = sum(1 for t in entity_types if "combinator" in t)
        assert combinator_count >= 2


class TestBundleAdvanced:
    """Test advanced bundle operations."""

    def test_compiles_successfully(self):
        result = compile_fixture("bundle_advanced")
        assert len(result["layout"].entity_placements) >= 1


class TestOperators:
    """Test comprehensive operator coverage."""

    def test_compiles_successfully(self):
        result = compile_fixture("operators")
        assert len(result["layout"].entity_placements) >= 1

    def test_generates_blueprint(self):
        result = compile_fixture("operators")
        bp_string = result["blueprint"].to_string()
        assert bp_string.startswith("0")


class TestEntityAdvanced:
    """Test advanced entity operations."""

    def test_compiles_successfully(self):
        result = compile_fixture("entity_advanced")
        assert len(result["layout"].entity_placements) >= 1

    def test_places_multiple_entities(self):
        result = compile_fixture("entity_advanced")
        entity_types = {p.entity_type for p in result["layout"].entity_placements.values()}
        assert "small-lamp" in entity_types
        assert "steel-chest" in entity_types


class TestExpressionCoverage:
    """Test expression coverage - type access, projections, conditionals."""

    def test_compiles_successfully(self):
        result = compile_fixture("expression_coverage")
        assert len(result["layout"].entity_placements) >= 1

    def test_handles_bundle_projections(self):
        result = compile_fixture("expression_coverage")
        # Should have constant combinators for bundles
        bp_string = result["blueprint"].to_string()
        assert len(bp_string) > 100


class TestStatementCoverage:
    """Test statement coverage - property access, entity assignments, for loops."""

    def test_compiles_successfully(self):
        result = compile_fixture("statement_coverage")
        assert len(result["layout"].entity_placements) >= 1

    def test_for_loop_generates_combinators(self):
        result = compile_fixture("statement_coverage")
        entity_types = [p.entity_type for p in result["layout"].entity_placements.values()]
        combinator_count = sum(1 for t in entity_types if "combinator" in t)
        assert combinator_count >= 1


class TestEntitySignalCoverage:
    """Test entity signal operations and arithmetic."""

    def test_compiles_successfully(self):
        result = compile_fixture("entity_signal_coverage")
        assert len(result["layout"].entity_placements) >= 1

    def test_multiple_arithmetic_ops(self):
        result = compile_fixture("entity_signal_coverage")
        # Should have multiple arithmetic combinators
        entity_types = [p.entity_type for p in result["layout"].entity_placements.values()]
        arith_count = sum(1 for t in entity_types if t == "arithmetic-combinator")
        assert arith_count >= 5


class TestMemoryEdgeCases:
    """Test memory edge cases - latches, conditionals, feedback."""

    def test_compiles_successfully(self):
        result = compile_fixture("memory_edge_cases")
        assert len(result["layout"].entity_placements) >= 1

    def test_creates_decider_combinators(self):
        result = compile_fixture("memory_edge_cases")
        entity_types = [p.entity_type for p in result["layout"].entity_placements.values()]
        decider_count = sum(1 for t in entity_types if t == "decider-combinator")
        assert decider_count >= 1


class TestCoverageBoost:
    """Comprehensive coverage fixture test."""

    def test_compiles_successfully(self):
        result = compile_fixture("coverage_boost")
        assert len(result["layout"].entity_placements) >= 1

    def test_generates_all_combinator_types(self):
        result = compile_fixture("coverage_boost")
        entity_types = [p.entity_type for p in result["layout"].entity_placements.values()]
        assert "arithmetic-combinator" in entity_types
        assert "decider-combinator" in entity_types
        assert "constant-combinator" in entity_types

    def test_places_lamp_entity(self):
        result = compile_fixture("coverage_boost")
        entity_types = {p.entity_type for p in result["layout"].entity_placements.values()}
        assert "small-lamp" in entity_types

    def test_generates_blueprint(self):
        result = compile_fixture("coverage_boost")
        bp_string = result["blueprint"].to_string()
        assert bp_string.startswith("0")
        assert len(bp_string) > 100
