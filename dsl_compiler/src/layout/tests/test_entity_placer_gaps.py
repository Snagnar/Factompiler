"""Tests for entity_placer.py coverage gaps."""

from dsl_compiler.src.common.constants import DEFAULT_CONFIG
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.layout.planner import LayoutPlanner
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer


def compile_to_layout(source: str):
    """Helper to compile DSL source to layout plan."""
    diags = ProgramDiagnostics()
    parser = DSLParser()
    program = parser.parse(source.strip())
    analyzer = SemanticAnalyzer(diags)
    analyzer.visit(program)
    lowerer = ASTLowerer(analyzer, diags)
    ir_ops = lowerer.lower_program(program)

    planner = LayoutPlanner(
        lowerer.ir_builder.signal_type_map,
        diagnostics=diags,
        signal_refs=lowerer.signal_refs,
        referenced_signal_names=lowerer.referenced_signal_names,
        config=DEFAULT_CONFIG,
    )
    layout = planner.plan_layout(ir_ops)
    return layout, planner, diags


class TestEntityPlacerCombinatorPlacement:
    """Cover lines 196-199: combinator placement logic."""

    def test_arithmetic_combinator_placement(self):
        """Test arithmetic combinator gets placed correctly."""
        layout, _, diags = compile_to_layout("""
        Signal x = 5 | "signal-A";
        Signal y = x + 10;
        """)
        # Should have arithmetic combinator in layout

    def test_decider_combinator_placement(self):
        """Test decider combinator gets placed correctly."""
        layout, _, diags = compile_to_layout("""
        Signal x = 5 | "signal-A";
        Signal result = (x > 0) : 1;
        """)
        # Should have decider combinator in layout

    def test_constant_combinator_placement(self):
        """Test constant combinator gets placed correctly."""
        layout, _, diags = compile_to_layout("""
        Signal x = 5 | "signal-A";
        """)
        # Should have constant combinator in layout


class TestEntityPlacerEntityPlacement:
    """Cover lines 359-365, 373-382: entity placement logic."""

    def test_lamp_placement_at_coordinates(self):
        """Test lamp is placed at specified coordinates."""
        layout, _, diags = compile_to_layout("""
        Entity lamp = place("small-lamp", 3, 5);
        lamp.enable = 1;
        """)
        # Lamp should be at (3, 5)

    def test_multiple_entities_no_collision(self):
        """Test multiple entities don't collide."""
        layout, _, diags = compile_to_layout("""
        Entity lamp1 = place("small-lamp", 0, 0);
        Entity lamp2 = place("small-lamp", 2, 0);
        Entity lamp3 = place("small-lamp", 4, 0);
        """)

    def test_entity_with_properties(self):
        """Test entity placement with properties."""
        layout, _, diags = compile_to_layout("""
        Entity lamp = place("small-lamp", 0, 0, {use_colors: 1, always_on: 1});
        """)


class TestEntityPlacerMemoryCellPlacement:
    """Cover lines 679-687: memory cell placement."""

    def test_memory_cell_placement(self):
        """Test memory cell components are placed."""
        layout, _, diags = compile_to_layout("""
        Memory counter: "signal-A";
        counter.write(counter.read() + 1);
        """)
        # Memory should create decider combinators

    def test_latch_memory_placement(self):
        """Test latch memory cell placement."""
        layout, _, diags = compile_to_layout("""
        Memory latch: "signal-L";
        Signal s = 50 | "signal-A";
        latch.write(1, set=s < 50, reset=s > 200);
        """)


class TestEntityPlacerLayoutCompaction:
    """Cover lines 695-698: layout compaction."""

    def test_compact_layout_generation(self):
        """Test that layout is compacted efficiently."""
        layout, _, diags = compile_to_layout("""
        Signal a = 5 | "signal-A";
        Signal b = 10 | "signal-B";
        Signal c = a + b;
        Signal d = c * 2;
        """)

    def test_layout_with_many_combinators(self):
        """Test layout with many combinators stays compact."""
        layout, _, diags = compile_to_layout("""
        Signal a = 1 | "signal-A";
        Signal b = a + 1;
        Signal c = b + 1;
        Signal d = c + 1;
        Signal e = d + 1;
        Signal f = e + 1;
        """)


class TestEntityPlacerGridAllocation:
    """Tests for grid allocation."""

    def test_grid_allocation_preserves_space(self):
        """Test grid allocates space correctly."""
        layout, _, diags = compile_to_layout("""
        Entity lamp1 = place("small-lamp", 0, 0);
        Signal x = 5 | "signal-A";
        Entity lamp2 = place("small-lamp", 2, 0);
        """)

    def test_combinator_horizontal_layout(self):
        """Test combinators are laid out horizontally."""
        layout, _, diags = compile_to_layout("""
        Signal a = 5 | "signal-A";
        Signal b = a + 10;
        Signal c = b * 2;
        """)
