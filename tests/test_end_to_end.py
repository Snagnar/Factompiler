#!/usr/bin/env python3
"""
End-to-end tests for the Factorio Circuit DSL compiler.
Tests the complete pipeline using sample programs: DSL Source -> Parser -> Semantic Analysis -> IR -> Blueprint
"""

import pytest
from pathlib import Path
import glob

from dsl_compiler.src.parser import DSLParser
from dsl_compiler.src.semantic import analyze_program, SemanticAnalyzer
from dsl_compiler.src.lowerer import lower_program
from dsl_compiler.src.emit import emit_blueprint_string, DRAFTSMAN_AVAILABLE


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
        semantic_diagnostics = analyze_program(program, strict_types=False, analyzer=analyzer)
        
        if semantic_diagnostics.has_errors():
            return False, f"Semantic errors: {semantic_diagnostics.get_messages()}"
        
        # IR generation
        ir_operations, lowering_diagnostics, signal_type_map = lower_program(program, analyzer)
        
        if lowering_diagnostics.has_errors():
            return False, f"IR lowering errors: {lowering_diagnostics.get_messages()}"
        
        # Blueprint generation
        if not DRAFTSMAN_AVAILABLE:
            return True, "Blueprint generation skipped - Draftsman not available"
        
        blueprint_string, emit_diagnostics = emit_blueprint_string(ir_operations, f"{program_name} Blueprint", signal_type_map)
        
        if emit_diagnostics.has_errors():
            return False, f"Blueprint emission errors: {emit_diagnostics.get_messages()}"
        
        # Validate blueprint string
        if not blueprint_string or len(blueprint_string) < 50 or not blueprint_string.startswith("0eN"):
            return False, "Invalid blueprint string generated"
        
        return True, blueprint_string
    
    def _save_successful_blueprint(self, blueprint_string: str, program_name: str):
        """Save successful blueprint for inspection."""
        if isinstance(blueprint_string, str) and blueprint_string.startswith("0eN"):
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            safe_name = program_name.replace(" ", "_").lower()
            with open(output_dir / f"{safe_name}.blueprint", "w") as f:
                f.write(blueprint_string)
    
    @pytest.mark.parametrize("sample_file", [
        # Working samples - only include features that are implemented
        "working_01_basic.fcdsl",
        "working_02_bundles.fcdsl", 
        "working_03_memory.fcdsl",
        "working_04_entities.fcdsl",
        "working_05_logic.fcdsl",
        
        # Original samples - these test the parser/semantic phases even if they don't fully compile
        "01_basic_arithmetic.fcdsl",
        "02_mixed_types.fcdsl", 
        "03_bundles.fcdsl",
        "04_memory.fcdsl",
        "05_entities.fcdsl",
        "06_functions.fcdsl",
        "07_control_flow.fcdsl",
        "08_sampler.fcdsl",
        "09_advanced_patterns.fcdsl",
        "10_imports_modules.fcdsl",
        "11_edge_cases.fcdsl",
        "12_comments_formatting.fcdsl",
        "13_complex_expressions.fcdsl"
    ])
    def test_sample_program_end_to_end(self, sample_file):
        """Test end-to-end compilation of each sample program."""
        sample_path = self.sample_dir / sample_file
        
        if not sample_path.exists():
            pytest.skip(f"Sample file {sample_file} not found")
        
        with open(sample_path, 'r') as f:
            dsl_code = f.read()
        
        program_name = sample_file.replace('.fcdsl', '').replace('_', ' ').title()
        success, result = self._run_full_pipeline(dsl_code, program_name)
        
        # ALL sample programs should work - they define the target specification
        # No skips - if something fails, we need to fix it
        assert success, f"End-to-end compilation failed for {sample_file}: {result}"
        
        # Save successful blueprints for inspection
        self._save_successful_blueprint(result, program_name)
    
    def test_blueprint_format_validation(self):
        """Test that generated blueprints have correct format using a simple sample."""
        simple_dsl = """
            let a = input(0);
            let b = input(1);
            let sum = a + b;
        """
        
        success, result = self._run_full_pipeline(simple_dsl, "Format Test")
        
        assert success, f"Blueprint generation failed: {result}"
        
        # Check blueprint string format
        assert isinstance(result, str), "Blueprint should be a string"
        assert len(result) > 50, "Blueprint string should be substantial"
        assert result.startswith("0eN"), "Blueprint should start with base64 header"
        
        # Test that it's valid base64
        import base64
        try:
            decoded = base64.b64decode(result[3:])  # Skip "0eN" prefix
            assert len(decoded) > 0, "Blueprint should decode to non-empty data"
        except Exception as e:
            pytest.fail(f"Blueprint string is not valid base64: {e}")


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
            with open(sample_path, 'r') as f:
                dsl_code = f.read()
            
            program = self.parser.parse(dsl_code)
            assert program is not None, f"Parser failed on {sample_path.name}"
            assert len(program.statements) > 0, f"No statements parsed from {sample_path.name}"
    
    def test_semantic_stage_all_samples(self):
        """Test that semantic analysis can handle all sample programs."""
        sample_files = list(self.sample_dir.glob("*.fcdsl"))
        
        for sample_path in sample_files:
            with open(sample_path, 'r') as f:
                dsl_code = f.read()
            
            program = self.parser.parse(dsl_code)
            analyzer = SemanticAnalyzer()
            diagnostics = analyze_program(program, strict_types=False, analyzer=analyzer)
            
            assert not diagnostics.has_errors(), f"Semantic analysis failed on {sample_path.name}: {diagnostics.get_messages()}"
    
    def test_ir_generation_all_samples(self):
        """Test that IR generation can handle all sample programs."""
        sample_files = list(self.sample_dir.glob("*.fcdsl"))
        
        for sample_path in sample_files:
            with open(sample_path, 'r') as f:
                dsl_code = f.read()
            
            program = self.parser.parse(dsl_code)
            analyzer = SemanticAnalyzer()
            analyze_program(program, strict_types=False, analyzer=analyzer)
            
            ir_operations, diagnostics, signal_type_map = lower_program(program, analyzer)
            
            assert not diagnostics.has_errors(), f"IR generation failed on {sample_path.name}: {diagnostics.get_messages()}"
            assert len(ir_operations) > 0, f"No IR operations generated from {sample_path.name}"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
