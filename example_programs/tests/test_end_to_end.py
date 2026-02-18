#!/usr/bin/env python3
"""
End-to-end tests for the Facto compiler.

Tests the complete pipeline using example programs:
Source -> Parser -> Semantic Analysis -> IR -> Layout -> Blueprint
"""

import warnings
from pathlib import Path

import pytest
from draftsman.blueprintable import Blueprint

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.emission.emitter import BlueprintEmitter
from dsl_compiler.src.layout.planner import LayoutPlanner
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer


def lower_program(program, semantic_analyzer):
    """Lower a semantic-analyzed program to IR."""
    diagnostics = ProgramDiagnostics()
    lowerer = ASTLowerer(semantic_analyzer, diagnostics)
    ir_operations = lowerer.lower_program(program)
    return (
        ir_operations,
        lowerer.diagnostics,
        lowerer.ir_builder.signal_type_map,
        lowerer.signal_refs,
        lowerer.referenced_signal_names,
    )


def emit_blueprint_string(
    ir_operations, label, signal_type_map, signal_refs=None, referenced_signal_names=None
):
    """Convert IR operations to Factorio blueprint string."""
    signal_type_map = signal_type_map or {}

    emitter_diagnostics = ProgramDiagnostics()
    emitter = BlueprintEmitter(emitter_diagnostics, signal_type_map)

    planner_diagnostics = ProgramDiagnostics()
    planner = LayoutPlanner(
        signal_type_map,
        diagnostics=planner_diagnostics,
        max_wire_span=9.0,
        signal_refs=signal_refs,
        referenced_signal_names=referenced_signal_names,
    )

    layout_plan = planner.plan_layout(
        ir_operations,
        blueprint_label=label,
        blueprint_description="",
    )

    combined_diagnostics = ProgramDiagnostics()
    combined_diagnostics.diagnostics.extend(planner.diagnostics.diagnostics)

    if planner.diagnostics.has_errors():
        return "", combined_diagnostics

    blueprint = emitter.emit_from_plan(layout_plan)
    combined_diagnostics.diagnostics.extend(emitter.diagnostics.diagnostics)

    try:
        blueprint_string = blueprint.to_string()
        return blueprint_string, combined_diagnostics
    except Exception as e:
        combined_diagnostics.error(f"Blueprint string generation failed: {e}")
        return "", combined_diagnostics


# Get list of sample files
example_programs_dir = Path(__file__).parent.parent
sample_files = list(example_programs_dir.glob("*.facto"))


@pytest.mark.end2end
class TestEndToEndCompilation:
    """End-to-end compilation tests using example programs."""

    def setup_method(self):
        """Set up for each test."""
        self.parser = DSLParser()
        self.example_dir = example_programs_dir

    def _run_full_pipeline(self, dsl_code: str, program_name: str = "Test") -> tuple:
        """Run the complete compilation pipeline."""
        # Parse
        program = self.parser.parse(dsl_code.strip())

        # Semantic analysis
        semantic_diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(semantic_diagnostics)
        analyzer.visit(program)

        if semantic_diagnostics.has_errors():
            return False, f"Semantic errors: {semantic_diagnostics.get_messages()}"

        # IR generation
        (
            ir_operations,
            lowering_diagnostics,
            signal_type_map,
            signal_refs,
            referenced_signal_names,
        ) = lower_program(program, analyzer)

        if lowering_diagnostics.has_errors():
            return False, f"IR lowering errors: {lowering_diagnostics.get_messages()}"

        # Blueprint generation
        blueprint_string, emit_diagnostics = emit_blueprint_string(
            ir_operations,
            f"{program_name} Blueprint",
            signal_type_map,
            signal_refs,
            referenced_signal_names,
        )

        if emit_diagnostics.has_errors():
            return (
                False,
                f"Blueprint emission errors: {emit_diagnostics.get_messages()}",
            )

        # Validate blueprint string
        if (
            not blueprint_string
            or len(blueprint_string) < 50
            or not blueprint_string.startswith("0eN")
        ):
            return False, "Invalid blueprint string generated"

        return True, blueprint_string

    def _blueprint_from_string(self, blueprint_string: str) -> Blueprint:
        """Decode a blueprint string while ignoring Draftsman alignment warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return Blueprint.from_string(blueprint_string)

    def _save_successful_blueprint(self, blueprint_string: str, program_name: str):
        """Save successful blueprint for inspection."""
        if isinstance(blueprint_string, str) and blueprint_string.startswith("0eN"):
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            safe_name = program_name.replace(" ", "_").lower()
            with open(output_dir / f"{safe_name}.blueprint", "w") as f:
                f.write(blueprint_string)

    @pytest.mark.parametrize(
        "sample_file",
        [f.name for f in sample_files],
    )
    def test_example_program_end_to_end(self, sample_file):
        """Test end-to-end compilation of each example program."""
        sample_path = self.example_dir / sample_file

        if not sample_path.exists():
            pytest.skip(f"Example file {sample_file} not found")

        with open(sample_path) as f:
            dsl_code = f.read()

        program_name = sample_file.replace(".facto", "").replace("_", " ").title()
        success, result = self._run_full_pipeline(dsl_code, program_name)

        # ALL example programs should work - they define the target specification
        assert success, f"End-to-end compilation failed for {sample_file}: {result}"

        # Save successful blueprints for inspection
        self._save_successful_blueprint(result, program_name)

    def test_blueprint_format_validation(self):
        """Test that generated blueprints have correct format."""
        simple_dsl = """
            Signal a = 100;
            Signal b = 200;
            Signal sum = a + b;
        """

        success, result = self._run_full_pipeline(simple_dsl, "Format Test")

        assert success, f"Blueprint generation failed: {result}"
        assert isinstance(result, str), "Blueprint should be a string"
        assert len(result) > 50, "Blueprint string should be substantial"
        assert result.startswith("0eN"), "Blueprint should start with base64 header"

    def test_same_signal_operand_wire_separation(self):
        """Test that two operands of the same signal type from different sources
        get different wire colors on the combinator input.

        Regression test: without wire color separation, both operands see the
        same summed value on the same wire, making subtraction always produce 0.
        """
        dsl_code = """
        # Two sources of iron-plate: one from a bundle, one from a train stop
        Entity stop = place("train-stop", 0, 0, {
            station: "Test",
            read_from_train: 1,
            read_stopped_train: 1
        });
        Signal cargo = stop.output["iron-plate"];

        Bundle bus = { ("iron-plate", 0) };
        Signal qty = bus["iron-plate"];

        # Same signal type on both sides of the subtraction
        Signal remaining = qty - cargo;
        """
        success, result = self._run_full_pipeline(dsl_code, "Wire Separation Test")
        assert success, f"Compilation failed: {result}"

        bp = self._blueprint_from_string(result)
        bp_dict = bp.to_dict()
        entities = bp_dict["blueprint"]["entities"]

        # Find the arithmetic combinator performing the subtraction
        arith = None
        for e in entities:
            if e["name"] == "arithmetic-combinator":
                ac = e.get("control_behavior", {}).get("arithmetic_conditions", {})
                if ac.get("operation") == "-":
                    arith = ac
                    break

        assert arith is not None, "No subtraction arithmetic combinator found"

        # Verify the two operands read from DIFFERENT networks
        first_nets = arith.get("first_signal_networks", {})
        second_nets = arith.get("second_signal_networks", {})

        first_red_only = first_nets.get("green") is False and first_nets.get("red") is not False
        first_green_only = first_nets.get("red") is False and first_nets.get("green") is not False
        second_red_only = second_nets.get("green") is False and second_nets.get("red") is not False
        second_green_only = (
            second_nets.get("red") is False and second_nets.get("green") is not False
        )

        # They must NOT both read from the same single network
        assert not (first_red_only and second_red_only), (
            "Both operands read from RED only — same-signal values will sum and "
            f"subtraction always produces 0. first_nets={first_nets}, second_nets={second_nets}"
        )
        assert not (first_green_only and second_green_only), (
            "Both operands read from GREEN only — same-signal values will sum and "
            f"subtraction always produces 0. first_nets={first_nets}, second_nets={second_nets}"
        )


@pytest.mark.end2end
class TestCompilerPipelineStages:
    """Test individual stages of the compiler pipeline on all example programs."""

    def setup_method(self):
        """Set up for each test."""
        self.parser = DSLParser()
        self.example_dir = example_programs_dir

    def test_parser_stage_all_examples(self):
        """Test that parser can handle all example programs."""
        sample_files = list(self.example_dir.glob("*.facto"))
        assert len(sample_files) > 0, "No example files found"

        for sample_path in sample_files:
            with open(sample_path) as f:
                dsl_code = f.read()

            program = self.parser.parse(dsl_code, str(sample_path.resolve()))
            assert program is not None, f"Parser failed on {sample_path.name}"
            assert len(program.statements) > 0, f"No statements parsed from {sample_path.name}"

    def test_semantic_stage_all_examples(self):
        """Test that semantic analysis can handle all example programs."""
        sample_files = list(self.example_dir.glob("*.facto"))

        for sample_path in sample_files:
            with open(sample_path) as f:
                dsl_code = f.read()

            program = self.parser.parse(dsl_code, str(sample_path.resolve()))
            diagnostics = ProgramDiagnostics()
            analyzer = SemanticAnalyzer(diagnostics)
            analyzer.visit(program)

            assert not diagnostics.has_errors(), (
                f"Semantic analysis failed on {sample_path.name}: {diagnostics.get_messages()}"
            )

    def test_ir_generation_all_examples(self):
        """Test that IR generation can handle all example programs."""
        sample_files = list(self.example_dir.glob("*.facto"))

        for sample_path in sample_files:
            with open(sample_path) as f:
                dsl_code = f.read()

            program = self.parser.parse(dsl_code, str(sample_path.resolve()))
            diagnostics = ProgramDiagnostics()
            analyzer = SemanticAnalyzer(diagnostics)
            analyzer.visit(program)

            ir_operations, lowering_diags, signal_type_map, signal_refs, referenced_signal_names = (
                lower_program(program, analyzer)
            )

            assert not lowering_diags.has_errors(), (
                f"IR generation failed on {sample_path.name}: {lowering_diags.get_messages()}"
            )
            assert len(ir_operations) > 0, f"No IR operations generated from {sample_path.name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
