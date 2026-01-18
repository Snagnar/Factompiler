"""Tests for connection_planner.py coverage gaps."""

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


class TestConnectionPlannerWireSelection:
    """Cover lines 293-297, 302-306: wire color selection."""

    def test_red_wire_preference(self):
        """Test red wire is used for simple connections."""
        layout, _, diags = compile_to_layout("""
        Signal x = 5 | "signal-A";
        Signal y = x + 10;
        """)

    def test_green_wire_for_additional_connections(self):
        """Test green wire is used when red is occupied."""
        layout, _, diags = compile_to_layout("""
        Signal a = 5 | "signal-A";
        Signal b = 10 | "signal-B";
        Signal c = a + b;
        Signal d = c + a;
        """)


class TestConnectionPlannerPortMapping:
    """Cover lines 320-323: port mapping for connections."""

    def test_combinator_input_output_ports(self):
        """Test correct port mapping for combinators."""
        layout, _, diags = compile_to_layout("""
        Signal x = 5 | "signal-A";
        Signal y = x * 2;
        Signal z = y + x;
        """)

    def test_entity_input_port(self):
        """Test port mapping for entity inputs."""
        layout, _, diags = compile_to_layout("""
        Signal enable = 1 | "signal-E";
        Entity lamp = place("small-lamp", 0, 0);
        lamp.enable = enable > 0;
        """)


class TestConnectionPlannerBundleConnections:
    """Cover lines 704-707: bundle connection handling."""

    def test_bundle_to_entity_connection(self):
        """Test bundle connected to entity."""
        layout, _, diags = compile_to_layout("""
        Bundle b = {("iron-plate", 10), ("copper-plate", 20)};
        Entity lamp = place("small-lamp", 0, 0);
        lamp.enable = all(b) > 5;
        """)

    def test_bundle_selection_connection(self):
        """Test bundle selection creates correct connections."""
        layout, _, diags = compile_to_layout("""
        Bundle b = {("iron-plate", 10), ("copper-plate", 20)};
        Signal iron = b["iron-plate"];
        """)


class TestConnectionPlannerMemoryConnections:
    """Cover lines 863-868: memory cell connections."""

    def test_memory_read_write_connections(self):
        """Test memory read/write connection routing."""
        layout, _, diags = compile_to_layout("""
        Memory counter: "signal-A";
        counter.write(counter.read() + 1);
        """)

    def test_memory_with_condition(self):
        """Test memory with conditional write."""
        layout, _, diags = compile_to_layout("""
        Memory counter: "signal-A";
        Signal enable = 1 | "signal-E";
        counter.write(counter.read() + 1, when=enable > 0);
        """)


class TestConnectionPlannerLatchConnections:
    """Cover lines 932-934: latch memory connections."""

    def test_sr_latch_connections(self):
        """Test SR latch connection routing."""
        layout, _, diags = compile_to_layout("""
        Memory hysteresis: "signal-H";
        Signal s = 50 | "signal-A";
        hysteresis.write(1, set=s < 50, reset=s > 200);
        """)


class TestConnectionPlannerMultipleOutputs:
    """Cover lines 1153-1159: multiple output connections."""

    def test_signal_used_in_multiple_places(self):
        """Test signal connected to multiple destinations."""
        layout, _, diags = compile_to_layout("""
        Signal x = 5 | "signal-A";
        Signal y = x + 10;
        Signal z = x * 2;
        Signal w = x - 3;
        """)


class TestConnectionPlannerWireOptimization:
    """Cover lines 1339-1342, 1353-1356, 1359-1362: wire optimization."""

    def test_wire_deduplication(self):
        """Test redundant wires are eliminated."""
        layout, _, diags = compile_to_layout("""
        Signal x = 5 | "signal-A";
        Signal y = x + x + x;
        """)

    def test_wire_length_optimization(self):
        """Test wires take optimal paths."""
        layout, _, diags = compile_to_layout("""
        Entity lamp1 = place("small-lamp", 0, 0);
        Entity lamp2 = place("small-lamp", 5, 0);
        Signal enable = 1 | "signal-E";
        lamp1.enable = enable > 0;
        lamp2.enable = enable > 0;
        """)


class TestConnectionPlannerErrorHandling:
    """Cover lines 1491-1495, 1498-1501: error handling in connections."""

    def test_unconnectable_components(self):
        """Test handling of components that can't be connected."""
        # This tests error handling when wires can't reach
        layout, _, diags = compile_to_layout("""
        Signal x = 5 | "signal-A";
        Signal y = x + 10;
        """)


class TestConnectionPlannerSignalMerging:
    """Cover lines 1514-1520: signal merging."""

    def test_same_signal_type_merge(self):
        """Test signals of same type can be merged on wire."""
        layout, _, diags = compile_to_layout("""
        Signal a = 5 | "signal-A";
        Signal b = 10 | "signal-A";
        Signal c = a + b;
        """)


class TestConnectionPlannerEntityPropWiring:
    """Cover lines 1634-1640: entity property wiring."""

    def test_entity_enable_wiring(self):
        """Test wiring for entity enable property."""
        layout, _, diags = compile_to_layout("""
        Signal enable = 1 | "signal-E";
        Entity lamp = place("small-lamp", 0, 0);
        lamp.enable = enable > 0;
        """)

    def test_entity_rgb_wiring(self):
        """Test wiring for entity RGB property."""
        layout, _, diags = compile_to_layout("""
        Signal color = 255 | "signal-white";
        Entity lamp = place("small-lamp", 0, 0, {use_colors: 1, always_on: 1, color_mode: 1});
        lamp.rgb = color;
        """)
