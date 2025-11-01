#!/usr/bin/env python3
"""
End-to-end tests for the Factorio Circuit DSL compiler.
Tests the complete pipeline using sample programs: DSL Source -> Parser -> Semantic Analysis -> IR -> Blueprint
"""

import glob
import warnings
from pathlib import Path

import pytest
from draftsman.blueprintable import Blueprint

from dsl_compiler.src.parsing import DSLParser
from dsl_compiler.src.semantic import analyze_program, SemanticAnalyzer
from dsl_compiler.src.lowering import lower_program
from dsl_compiler.src.emission import emit_blueprint_string

sample_files = glob.glob("tests/sample_programs/*.fcdsl")


class TestEndToEndCompilation:
    """End-to-end compilation tests using sample programs."""

    def setup_method(self):
        """Set up for each test."""
        self.parser = DSLParser()
        self.sample_dir = Path(__file__).parent / "sample_programs"

    def _run_full_pipeline(self, dsl_code: str, program_name: str = "Test") -> tuple:
        """Run the complete compilation pipeline."""
        # Parse
        program = self.parser.parse(dsl_code.strip())

        # Semantic analysis
        analyzer = SemanticAnalyzer()
        semantic_diagnostics = analyze_program(
            program, strict_types=False, analyzer=analyzer
        )

        if semantic_diagnostics.has_errors():
            return False, f"Semantic errors: {semantic_diagnostics.get_messages()}"

        # IR generation
        ir_operations, lowering_diagnostics, signal_type_map = lower_program(
            program, analyzer
        )

        if lowering_diagnostics.has_errors():
            return False, f"IR lowering errors: {lowering_diagnostics.get_messages()}"

        # Blueprint generation
        blueprint_string, emit_diagnostics = emit_blueprint_string(
            ir_operations, f"{program_name} Blueprint", signal_type_map
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
        [Path(f).name for f in sample_files],
    )
    def test_sample_program_end_to_end(self, sample_file):
        """Test end-to-end compilation of each sample program."""
        sample_path = self.sample_dir / sample_file

        if not sample_path.exists():
            pytest.skip(f"Sample file {sample_file} not found")

        with open(sample_path, "r") as f:
            dsl_code = f.read()

        program_name = sample_file.replace(".fcdsl", "").replace("_", " ").title()
        success, result = self._run_full_pipeline(dsl_code, program_name)

        # ALL sample programs should work - they define the target specification
        # No skips - if something fails, we need to fix it
        assert success, f"End-to-end compilation failed for {sample_file}: {result}"

        # Save successful blueprints for inspection
        self._save_successful_blueprint(result, program_name)

    def test_blueprint_format_validation(self):
        """Test that generated blueprints have correct format using a simple sample."""
        simple_dsl = """
            Signal a = 100;
            Signal b = 200;
            Signal sum = a + b;
        """

        success, result = self._run_full_pipeline(simple_dsl, "Format Test")

        assert success, f"Blueprint generation failed: {result}"

        # Check blueprint string format
        assert isinstance(result, str), "Blueprint should be a string"
        assert len(result) > 50, "Blueprint string should be substantial"
        assert result.startswith("0eN"), "Blueprint should start with base64 header"

    def test_memory_advanced_blueprint_structure(self):
        """Validate that advanced memory sample emits rich combinator networks."""
        sample_path = self.sample_dir / "04_memory_advanced.fcdsl"
        with open(sample_path, "r", encoding="utf-8") as f:
            dsl_code = f.read()

        success, result = self._run_full_pipeline(
            dsl_code, "Memory Advanced Integration"
        )
        assert success, f"Memory advanced sample failed: {result}"

        blueprint = self._blueprint_from_string(result)
        entity_dicts = [entity.to_dict() for entity in blueprint.entities]

        deciders = [ent for ent in entity_dicts if ent["name"] == "decider-combinator"]
        assert len(deciders) >= 6, "Expected SR latch and state machine deciders"

        decider_conditions = [
            condition
            for dec in deciders
            for condition in dec.get("control_behavior", {})
            .get("decider_conditions", {})
            .get("conditions", [])
        ]
        assert any(
            cond.get("first_signal", {}).get("name") == "signal-W"
            for cond in decider_conditions
        ), "Memory latch should react to explicit signal-W write enables"

        arithmetic_outputs = [
            (
                cond.get("first_signal", {}).get("name"),
                cond.get("output_signal", {}).get("name"),
            )
            for ent in entity_dicts
            if ent["name"] == "arithmetic-combinator"
            for cond in [
                ent.get("control_behavior", {}).get("arithmetic_conditions", {})
            ]
            if cond
        ]
        assert any(
            first == "signal-everything" or first == output
            for first, output in arithmetic_outputs
            if output is not None
        ), "Projected outputs should normalize or coerce bundles via declared signals"

        projected_outputs = {output for _, output in arithmetic_outputs if output}
        assert {
            "signal-R",
            "signal-S",
        }.issubset(projected_outputs), (
            "Expected projected control and state channels in memory pipeline"
        )
        assert len(projected_outputs) >= 8, (
            "Memory pipeline should expose a rich set of virtual channels"
        )

    def test_entity_property_blueprint_behavior(self):
        """Ensure entity property wiring and projections appear in blueprints."""
        sample_path = self.sample_dir / "19_advanced_entity_properties_fixed.fcdsl"
        with open(sample_path, "r", encoding="utf-8") as f:
            dsl_code = f.read()

        success, result = self._run_full_pipeline(
            dsl_code, "Entity Property Integration"
        )
        assert success, f"Entity property sample failed: {result}"

        blueprint = self._blueprint_from_string(result)
        entity_dicts = [entity.to_dict() for entity in blueprint.entities]

        lamp = next(ent for ent in entity_dicts if ent["name"] == "small-lamp")
        assert lamp.get("control_behavior", {}).get("circuit_enabled") is True

        train_stop = next(ent for ent in entity_dicts if ent["name"] == "train-stop")
        first_signal = (
            train_stop.get("control_behavior", {})
            .get("circuit_condition", {})
            .get("first_signal", {})
        )
        assert first_signal.get("type") == "virtual"

        inserter = next(ent for ent in entity_dicts if ent["name"] == "inserter")
        assert "first_signal" in inserter.get("control_behavior", {}).get(
            "circuit_condition", {}
        )

        arithmetic_outputs = {
            (
                cond.get("output_signal", {}).get("type"),
                cond.get("output_signal", {}).get("name"),
            )
            for ent in entity_dicts
            if ent["name"] == "arithmetic-combinator"
            for cond in [
                ent.get("control_behavior", {}).get("arithmetic_conditions", {})
            ]
            if cond.get("output_signal")
        }
        output_types = {output[0] for output in arithmetic_outputs}
        assert {"virtual", "item"}.issubset(output_types), (
            "Projection logic should mix virtual and item outputs"
        )

        # For now, just check that we got a reasonable blueprint string
        # The actual base64 validation can be tricky due to draftsman's encoding quirks
        # but we know the blueprints work since all sample programs pass


class TestCompilerPipeline:
    """Test individual stages of the compiler pipeline."""

    def setup_method(self):
        """Set up for each test."""
        self.parser = DSLParser()
        self.sample_dir = Path(__file__).parent / "sample_programs"

    def test_parser_stage_all_samples(self):
        """Test that parser can handle all sample programs."""
        sample_files = list(self.sample_dir.glob("*.fcdsl"))
        assert len(sample_files) > 0, "No sample files found"

        for sample_path in sample_files:
            with open(sample_path, "r") as f:
                dsl_code = f.read()

            program = self.parser.parse(dsl_code)
            assert program is not None, f"Parser failed on {sample_path.name}"
            assert len(program.statements) > 0, (
                f"No statements parsed from {sample_path.name}"
            )

    def test_semantic_stage_all_samples(self):
        """Test that semantic analysis can handle all sample programs."""
        sample_files = list(self.sample_dir.glob("*.fcdsl"))

        for sample_path in sample_files:
            with open(sample_path, "r") as f:
                dsl_code = f.read()

            program = self.parser.parse(dsl_code)
            analyzer = SemanticAnalyzer()
            diagnostics = analyze_program(
                program, strict_types=False, analyzer=analyzer
            )

            assert not diagnostics.has_errors(), (
                f"Semantic analysis failed on {sample_path.name}: {diagnostics.get_messages()}"
            )

    def test_ir_generation_all_samples(self):
        """Test that IR generation can handle all sample programs."""
        sample_files = list(self.sample_dir.glob("*.fcdsl"))

        for sample_path in sample_files:
            with open(sample_path, "r") as f:
                dsl_code = f.read()

            program = self.parser.parse(dsl_code)
            analyzer = SemanticAnalyzer()
            analyze_program(program, strict_types=False, analyzer=analyzer)

            ir_operations, diagnostics, signal_type_map = lower_program(
                program, analyzer
            )

            assert not diagnostics.has_errors(), (
                f"IR generation failed on {sample_path.name}: {diagnostics.get_messages()}"
            )
            assert len(ir_operations) > 0, (
                f"No IR operations generated from {sample_path.name}"
            )


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
