"""Tests for wire color pinning via .wire attribute.

Tests the full pipeline: grammar → AST → lowering → layout → connection planner.
"""

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.ir.builder import BundleRef, SignalRef
from dsl_compiler.src.ir.nodes import IRArith, IRConst, IRDecider
from dsl_compiler.src.layout.planner import LayoutPlanner
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer


def compile_to_ir(source: str):
    """Compile source to IR, returning (ir_ops, lowerer, diagnostics)."""
    parser = DSLParser()
    diagnostics = ProgramDiagnostics()
    analyzer = SemanticAnalyzer(diagnostics)
    program = parser.parse(source)
    analyzer.visit(program)
    lowerer = ASTLowerer(analyzer, diagnostics)
    ir_ops = lowerer.lower_program(program)
    return ir_ops, lowerer, diagnostics


def compile_to_layout(source: str):
    """Compile source through layout phase, returning (layout_plan, lowerer, diagnostics)."""
    ir_ops, lowerer, diagnostics = compile_to_ir(source)
    planner = LayoutPlanner(
        lowerer.ir_builder.signal_type_map,
        diagnostics=diagnostics,
        signal_refs=lowerer.signal_refs,
        referenced_signal_names=lowerer.referenced_signal_names,
    )
    layout = planner.plan_layout(ir_ops)
    return layout, lowerer, diagnostics


# ── Parsing tests ─────────────────────────────────────────────────────────


class TestWirePinningParsing:
    """Test that .wire = red/green parses correctly."""

    def test_wire_red_parses(self):
        """a.wire = red; should parse without errors."""
        source = 'Signal a = ("signal-A", 10);\na.wire = red;'
        _, _, diag = compile_to_ir(source)
        assert not diag.has_errors(), diag.get_messages()

    def test_wire_green_parses(self):
        """a.wire = green; should parse without errors."""
        source = 'Signal a = ("signal-A", 10);\na.wire = green;'
        _, _, diag = compile_to_ir(source)
        assert not diag.has_errors(), diag.get_messages()

    def test_bundle_wire_parses(self):
        """bundle.wire = red; should parse without errors."""
        source = 'Bundle b = { ("signal-A", 1), ("signal-B", 2) };\nb.wire = red;'
        _, _, diag = compile_to_ir(source)
        assert not diag.has_errors(), diag.get_messages()


# ── Lowering tests ────────────────────────────────────────────────────────


class TestWirePinningLowering:
    """Test that .wire assignments set wire_color in IR debug_metadata."""

    def test_signal_wire_red(self):
        """signal.wire = red should set wire_color='red' on the IRConst."""
        source = 'Signal a = ("signal-A", 10);\na.wire = red;'
        ir_ops, lowerer, diag = compile_to_ir(source)
        assert not diag.has_errors(), diag.get_messages()

        ref = lowerer.signal_refs["a"]
        assert isinstance(ref, SignalRef)
        ir_op = lowerer.ir_builder.get_operation(ref.source_id)
        assert isinstance(ir_op, IRConst)
        assert ir_op.debug_metadata.get("wire_color") == "red"

    def test_signal_wire_green(self):
        """signal.wire = green should set wire_color='green' on the IRConst."""
        source = 'Signal a = ("signal-A", 10);\na.wire = green;'
        ir_ops, lowerer, diag = compile_to_ir(source)
        assert not diag.has_errors(), diag.get_messages()

        ref = lowerer.signal_refs["a"]
        ir_op = lowerer.ir_builder.get_operation(ref.source_id)
        assert ir_op.debug_metadata.get("wire_color") == "green"

    def test_computed_signal_wire_color(self):
        """Wire color on arithmetic result should set wire_color on IRArith."""
        source = """
Signal a = ("signal-A", 10);
Signal b = ("signal-B", 20);
Signal c = a + b;
c.wire = red;
"""
        ir_ops, lowerer, diag = compile_to_ir(source)
        assert not diag.has_errors(), diag.get_messages()

        ref = lowerer.signal_refs["c"]
        assert isinstance(ref, SignalRef)
        ir_op = lowerer.ir_builder.get_operation(ref.source_id)
        assert isinstance(ir_op, IRArith)
        assert ir_op.debug_metadata.get("wire_color") == "red"

    def test_decider_result_wire_color(self):
        """Wire color on decider result should set wire_color on IRDecider."""
        source = """
Signal a = ("signal-A", 10);
Signal b = a > 5;
b.wire = green;
"""
        ir_ops, lowerer, diag = compile_to_ir(source)
        assert not diag.has_errors(), diag.get_messages()

        ref = lowerer.signal_refs["b"]
        assert isinstance(ref, SignalRef)
        ir_op = lowerer.ir_builder.get_operation(ref.source_id)
        assert isinstance(ir_op, IRDecider)
        assert ir_op.debug_metadata.get("wire_color") == "green"

    def test_bundle_wire_color(self):
        """Wire color on bundle should set wire_color on the producing IRConst."""
        source = """
Bundle b = { ("signal-A", 1), ("signal-B", 2) };
b.wire = green;
"""
        ir_ops, lowerer, diag = compile_to_ir(source)
        assert not diag.has_errors(), diag.get_messages()

        ref = lowerer.signal_refs["b"]
        assert isinstance(ref, BundleRef)
        ir_op = lowerer.ir_builder.get_operation(ref.source_id)
        assert ir_op.debug_metadata.get("wire_color") == "green"

    def test_invalid_wire_color_reports_error(self):
        """a.wire = blue should produce a diagnostic error."""
        source = 'Signal a = ("signal-A", 10);\na.wire = blue;'
        _, _, diag = compile_to_ir(source)
        assert diag.has_errors()
        assert any("blue" in d.message for d in diag.diagnostics)

    def test_wire_color_on_undefined_variable(self):
        """x.wire = red on undefined variable should error."""
        source = "x.wire = red;"
        _, _, diag = compile_to_ir(source)
        assert diag.has_errors()

    def test_wire_color_on_int_constant(self):
        """Wire color on a compile-time int (not a signal) should error."""
        source = "int x = 5;\nx.wire = red;"
        _, _, diag = compile_to_ir(source)
        assert diag.has_errors()

    def test_wire_color_overwrite(self):
        """Second .wire assignment should overwrite the first."""
        source = """
Signal a = ("signal-A", 10);
a.wire = red;
a.wire = green;
"""
        ir_ops, lowerer, diag = compile_to_ir(source)
        assert not diag.has_errors(), diag.get_messages()

        ref = lowerer.signal_refs["a"]
        ir_op = lowerer.ir_builder.get_operation(ref.source_id)
        assert ir_op.debug_metadata.get("wire_color") == "green"


# ── Layout integration tests ─────────────────────────────────────────────


class TestWirePinningLayout:
    """Test that wire_color flows through to EntityPlacement.properties."""

    def test_constant_placement_has_wire_color(self):
        """Wire color should appear in EntityPlacement.properties for constants."""
        source = 'Signal a = ("signal-A", 10);\na.wire = red;'
        layout, _, diag = compile_to_layout(source)
        assert not diag.has_errors(), diag.get_messages()

        # Find the placement for signal a
        placements = list(layout.entity_placements.values())
        const_placements = [p for p in placements if p.entity_type == "constant-combinator"]
        assert len(const_placements) >= 1

        wire_colored = [p for p in const_placements if p.properties.get("wire_color") == "red"]
        assert len(wire_colored) >= 1, (
            f"Expected at least one constant combinator with wire_color='red', "
            f"got properties: {[p.properties for p in const_placements]}"
        )

    def test_arithmetic_placement_has_wire_color(self):
        """Wire color should appear in EntityPlacement.properties for arithmetic combinators."""
        source = """
Signal a = ("signal-A", 10);
Signal b = ("signal-B", 20);
Signal c = a + b;
c.wire = green;
"""
        layout, _, diag = compile_to_layout(source)
        assert not diag.has_errors(), diag.get_messages()

        arith_placements = [
            p for p in layout.entity_placements.values() if p.entity_type == "arithmetic-combinator"
        ]
        assert len(arith_placements) >= 1

        wire_colored = [p for p in arith_placements if p.properties.get("wire_color") == "green"]
        assert len(wire_colored) >= 1, (
            f"Expected at least one arithmetic combinator with wire_color='green', "
            f"got properties: {[p.properties for p in arith_placements]}"
        )

    def test_decider_placement_has_wire_color(self):
        """Wire color should appear in EntityPlacement.properties for decider combinators."""
        source = """
Signal a = ("signal-A", 10);
Signal b = a > 5;
b.wire = red;
"""
        layout, _, diag = compile_to_layout(source)
        assert not diag.has_errors(), diag.get_messages()

        decider_placements = [
            p for p in layout.entity_placements.values() if p.entity_type == "decider-combinator"
        ]
        assert len(decider_placements) >= 1

        wire_colored = [p for p in decider_placements if p.properties.get("wire_color") == "red"]
        assert len(wire_colored) >= 1, (
            f"Expected at least one decider combinator with wire_color='red', "
            f"got properties: {[p.properties for p in decider_placements]}"
        )

    def test_unpinned_signals_still_work(self):
        """Signals without .wire should still get automatic color assignment."""
        source = """
Signal a = ("signal-A", 10);
Signal b = ("signal-B", 20);
Signal c = a + b;
"""
        layout, _, diag = compile_to_layout(source)
        assert not diag.has_errors(), diag.get_messages()

    def test_mixed_pinned_and_unpinned(self):
        """Mix of pinned and unpinned signals should compile without errors."""
        source = """
Signal a = ("signal-A", 10);
Signal b = ("signal-B", 20);
a.wire = red;
Signal c = a + b;
"""
        layout, _, diag = compile_to_layout(source)
        assert not diag.has_errors(), diag.get_messages()

    def test_two_different_colors(self):
        """Two signals pinned to different colors feeding same combinator."""
        source = """
Signal a = ("signal-A", 10);
Signal b = ("signal-B", 20);
a.wire = red;
b.wire = green;
Signal c = a + b;
"""
        layout, _, diag = compile_to_layout(source)
        assert not diag.has_errors(), diag.get_messages()

    def test_wire_color_constraint_applied_in_connections(self):
        """Pinned wire color should be reflected in wire connections."""
        source = """
Signal a = ("signal-A", 10);
Signal b = ("signal-B", 20);
a.wire = red;
b.wire = green;
Signal c = a + b;
"""
        layout, _, diag = compile_to_layout(source)
        assert not diag.has_errors(), diag.get_messages()

        # Find connections from the pinned signal sources
        a_placements = [
            p for p in layout.entity_placements.values() if p.properties.get("wire_color") == "red"
        ]
        b_placements = [
            p
            for p in layout.entity_placements.values()
            if p.properties.get("wire_color") == "green"
        ]
        assert len(a_placements) >= 1
        assert len(b_placements) >= 1

        # Verify connections from a use red wire
        a_entity_ids = {p.ir_node_id for p in a_placements}
        for conn in layout.wire_connections:
            if conn.source_entity_id in a_entity_ids:
                assert conn.wire_color == "red", (
                    f"Expected red wire from pinned entity, got {conn.wire_color}"
                )

        # Verify connections from b use green wire
        b_entity_ids = {p.ir_node_id for p in b_placements}
        for conn in layout.wire_connections:
            if conn.source_entity_id in b_entity_ids:
                assert conn.wire_color == "green", (
                    f"Expected green wire from pinned entity, got {conn.wire_color}"
                )

    def test_bundle_wire_color_in_layout(self):
        """Bundle with pinned wire color should appear in layout."""
        source = """
Bundle b = { ("signal-A", 1), ("signal-B", 2) };
b.wire = green;
Signal out = b["signal-A"];
"""
        layout, _, diag = compile_to_layout(source)
        assert not diag.has_errors(), diag.get_messages()

        const_placements = [
            p for p in layout.entity_placements.values() if p.entity_type == "constant-combinator"
        ]
        wire_colored = [p for p in const_placements if p.properties.get("wire_color") == "green"]
        assert len(wire_colored) >= 1
