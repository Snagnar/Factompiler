"""Tests for entity_emitter.py coverage gaps."""

from dsl_compiler.src.common.constants import DEFAULT_CONFIG
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.emission.emitter import BlueprintEmitter
from dsl_compiler.src.layout.planner import LayoutPlanner
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer


def compile_to_blueprint(source: str):
    """Helper to compile DSL source to blueprint."""
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

    emitter = BlueprintEmitter(diags, lowerer.ir_builder.signal_type_map)
    blueprint = emitter.emit_from_plan(layout)
    return blueprint, diags


class TestEntityEmitterArithmeticCombinator:
    """Cover lines 402-413: arithmetic combinator emission."""

    def test_emit_arithmetic_combinator_add(self):
        """Test emitting addition combinator."""
        blueprint, diags = compile_to_blueprint("""
        Signal x = 5 | "signal-A";
        Signal y = x + 10;
        """)
        bp_dict = blueprint.to_dict()
        assert "blueprint" in bp_dict

    def test_emit_arithmetic_combinator_multiply(self):
        """Test emitting multiplication combinator."""
        blueprint, diags = compile_to_blueprint("""
        Signal x = 5 | "signal-A";
        Signal y = x * 2;
        """)

    def test_emit_arithmetic_combinator_modulo(self):
        """Test emitting modulo combinator."""
        blueprint, diags = compile_to_blueprint("""
        Signal x = 100 | "signal-A";
        Signal y = x % 7;
        """)


class TestEntityEmitterDeciderCombinator:
    """Cover lines 433-444: decider combinator emission."""

    def test_emit_decider_combinator_greater_than(self):
        """Test emitting greater than comparison."""
        blueprint, diags = compile_to_blueprint("""
        Signal x = 5 | "signal-A";
        Signal result = (x > 0) : 1;
        """)

    def test_emit_decider_combinator_less_than(self):
        """Test emitting less than comparison."""
        blueprint, diags = compile_to_blueprint("""
        Signal x = 5 | "signal-A";
        Signal result = (x < 10) : 1;
        """)

    def test_emit_decider_combinator_equal(self):
        """Test emitting equality comparison."""
        blueprint, diags = compile_to_blueprint("""
        Signal x = 5 | "signal-A";
        Signal result = (x == 5) : 1;
        """)


class TestEntityEmitterConstantCombinator:
    """Cover lines 471-474: constant combinator emission."""

    def test_emit_constant_combinator_single_signal(self):
        """Test emitting constant with single signal."""
        blueprint, diags = compile_to_blueprint("""
        Signal x = 42 | "signal-A";
        """)

    def test_emit_constant_combinator_bundle(self):
        """Test emitting constant bundle."""
        blueprint, diags = compile_to_blueprint("""
        Bundle b = {("iron-plate", 10), ("copper-plate", 20)};
        Signal x = b["iron-plate"];
        """)


class TestEntityEmitterMemoryCell:
    """Cover lines 483-485: memory cell emission."""

    def test_emit_memory_cell(self):
        """Test emitting memory cell."""
        blueprint, diags = compile_to_blueprint("""
        Memory counter: "signal-A";
        counter.write(counter.read() + 1);
        """)

    def test_emit_latch_memory(self):
        """Test emitting latch memory cell."""
        blueprint, diags = compile_to_blueprint("""
        Memory latch: "signal-L";
        Signal s = 50 | "signal-A";
        latch.write(1, set=s < 50, reset=s > 200);
        """)


class TestEntityEmitterLampEntity:
    """Tests for lamp entity emission."""

    def test_emit_lamp_with_enable(self):
        """Test emitting lamp with enable condition."""
        blueprint, diags = compile_to_blueprint("""
        Entity lamp = place("small-lamp", 0, 0);
        Signal enable = 1 | "signal-E";
        lamp.enable = enable > 0;
        """)

    def test_emit_lamp_with_color(self):
        """Test emitting lamp with color."""
        blueprint, diags = compile_to_blueprint("""
        Entity lamp = place("small-lamp", 0, 0, {use_colors: 1, always_on: 1, color_mode: 1});
        Signal color = 255 | "signal-white";
        lamp.rgb = color;
        """)

    def test_emit_lamp_with_properties(self):
        """Test emitting lamp with multiple properties."""
        blueprint, diags = compile_to_blueprint("""
        Entity lamp = place("small-lamp", 0, 0, {use_colors: 1, always_on: 1, color_mode: 0});
        Signal r = 200 | "signal-red";
        lamp.rgb = r;
        """)


class TestEntityEmitterWireConnections:
    """Tests for wire connection emission."""

    def test_emit_red_wire_connections(self):
        """Test emitting red wire connections."""
        blueprint, diags = compile_to_blueprint("""
        Signal x = 5 | "signal-A";
        Signal y = x + 10;
        """)

    def test_emit_green_wire_connections(self):
        """Test emitting green wire connections."""
        blueprint, diags = compile_to_blueprint("""
        Signal a = 5 | "signal-A";
        Signal b = 10 | "signal-B";
        Signal c = a + b;
        Signal d = c + a;
        Signal e = d + b;
        """)

    def test_emit_mixed_wire_connections(self):
        """Test emitting mixed red and green wire connections."""
        blueprint, diags = compile_to_blueprint("""
        Signal a = 5 | "signal-A";
        Signal b = 10 | "signal-B";
        Signal c = 15 | "signal-C";
        Signal sum = a + b + c;
        """)


class TestEntityEmitterBlueprintMetadata:
    """Tests for blueprint metadata emission."""

    def test_blueprint_has_entities(self):
        """Test blueprint contains entities."""
        blueprint, diags = compile_to_blueprint("""
        Signal x = 5 | "signal-A";
        """)
        bp_dict = blueprint.to_dict()
        assert "blueprint" in bp_dict
        bp = bp_dict["blueprint"]
        assert "entities" in bp

    def test_blueprint_version(self):
        """Test blueprint has correct version."""
        blueprint, diags = compile_to_blueprint("""
        Signal x = 5 | "signal-A";
        """)
        bp_dict = blueprint.to_dict()
        assert "blueprint" in bp_dict
