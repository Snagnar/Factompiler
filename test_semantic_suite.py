"""
Comprehensive test suite for the DSL semantic analyzer.

Tests semantic analysis features including:
- Symbol table construction and name resolution
- Type inference with implicit signal type allocation  
- Mixed-type arithmetic validation and warnings
- Diagnostic collection and error reporting
- Function analysis and scoping
- Memory operation validation
- Entity and property analysis
"""

import pytest
from pathlib import Path
import sys

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "dsl_compiler" / "src"))

from semantic import (
    SemanticAnalyzer, analyze_program, analyze_file,
    DiagnosticLevel, ValueType, SignalTypeInfo, 
    SignalValue, BundleValue, IntValue, Symbol
)
from parser import DSLParser
from dsl_ast import *


class TestSemanticAnalyzer:
    """Test suite for semantic analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = DSLParser()
        self.analyzer = SemanticAnalyzer()
        
    def parse_and_analyze(self, code: str, strict_types: bool = False):
        """Helper: parse code and run semantic analysis."""
        program = self.parser.parse(code)
        analyzer = SemanticAnalyzer(strict_types=strict_types)
        analyzer.visit(program)
        return analyzer
    
    def get_diagnostics_by_level(self, analyzer, level: DiagnosticLevel):
        """Helper: filter diagnostics by level."""
        return [d for d in analyzer.diagnostics.diagnostics if d.level == level]
    
    # =========================================================================
    # Symbol Table Tests
    # =========================================================================
    
    def test_variable_definition_and_lookup(self):
        """Test basic variable definition and lookup."""
        code = """
        let x = 42;
        let y = x + 1;
        """
        analyzer = self.parse_and_analyze(code)
        
        # Check symbol table
        x_symbol = analyzer.symbol_table.lookup("x")
        y_symbol = analyzer.symbol_table.lookup("y")
        
        assert x_symbol is not None
        assert x_symbol.name == "x"
        assert x_symbol.symbol_type == "variable"
        
        assert y_symbol is not None
        assert y_symbol.name == "y"
        
        # Should have no errors
        errors = self.get_diagnostics_by_level(analyzer, DiagnosticLevel.ERROR)
        assert len(errors) == 0
        
    def test_undefined_variable_error(self):
        """Test error on undefined variables."""
        code = """
        let x = undefined_var + 1;
        """
        analyzer = self.parse_and_analyze(code)
        
        errors = self.get_diagnostics_by_level(analyzer, DiagnosticLevel.ERROR)
        assert len(errors) > 0
        assert any("undefined" in err.message.lower() for err in errors)
        
    def test_duplicate_variable_definition(self):
        """Test error on duplicate variable definition."""
        code = """
        let x = 1;
        let x = 2;  # Should be an error
        """
        analyzer = self.parse_and_analyze(code)
        
        errors = self.get_diagnostics_by_level(analyzer, DiagnosticLevel.ERROR)
        assert len(errors) > 0
        assert any("already defined" in err.message.lower() for err in errors)
    
    # =========================================================================
    # Type Inference Tests
    # =========================================================================
    
    def test_integer_literal_type(self):
        """Test type inference for integer literals."""
        code = """
        let num = 42;
        """
        analyzer = self.parse_and_analyze(code)
        
        # Check that integer type is inferred
        num_symbol = analyzer.symbol_table.lookup("num")
        assert num_symbol is not None
        assert isinstance(num_symbol.value_type, IntValue)
        
    def test_input_signal_type(self):
        """Test type inference for input expressions."""
        code = """
        let signal_A = input(0);
        let iron_plates = input("iron-plate", 1);
        """
        analyzer = self.parse_and_analyze(code)
        
        signal_a_symbol = analyzer.symbol_table.lookup("signal_A")
        iron_symbol = analyzer.symbol_table.lookup("iron_plates")
        
        assert signal_a_symbol is not None
        assert isinstance(signal_a_symbol.value_type, SignalValue)
        
        assert iron_symbol is not None
        assert isinstance(iron_symbol.value_type, SignalValue)
        
    def test_mixed_type_arithmetic(self):
        """Test mixed-type arithmetic with implicit type allocation."""
        code = """
        let a = input(0);      # Signal 
        let b = 10;            # Integer
        let mixed = a + b;     # Should allocate implicit type
        """
        analyzer = self.parse_and_analyze(code)
        
        mixed_symbol = analyzer.symbol_table.lookup("mixed")
        assert mixed_symbol is not None
        assert isinstance(mixed_symbol.value_type, SignalValue)
        
        # Should generate warning about mixed types
        warnings = self.get_diagnostics_by_level(analyzer, DiagnosticLevel.WARNING)
        assert len(warnings) > 0
        
    def test_projection_type_conversion(self):
        """Test type projection operations."""
        code = """
        let a = input(0);
        let projected = a | "iron-plate";
        """
        analyzer = self.parse_and_analyze(code)
        
        projected_symbol = analyzer.symbol_table.lookup("projected")
        assert projected_symbol is not None
        assert isinstance(projected_symbol.value_type, SignalValue)
        
        # Check that signal type is correctly set
        signal_info = projected_symbol.value_type.signal_type
        assert signal_info.name == "iron-plate"
        
    def test_bundle_type_inference(self):
        """Test bundle expression type inference."""
        code = """
        let a = input("iron-plate", 0);
        let b = input("copper-plate", 1);
        let combined = bundle(a, b);
        """
        analyzer = self.parse_and_analyze(code)
        
        combined_symbol = analyzer.symbol_table.lookup("combined")
        assert combined_symbol is not None
        assert isinstance(combined_symbol.value_type, BundleValue)
        
        # Check bundle has correct channels
        bundle_val = combined_symbol.value_type
        assert len(bundle_val.channels) == 2
        assert "iron-plate" in bundle_val.channels
        assert "copper-plate" in bundle_val.channels
    
    # =========================================================================
    # Memory Analysis Tests
    # =========================================================================
    
    def test_memory_declaration(self):
        """Test memory declaration analysis."""
        code = """
        mem counter = memory(0);
        mem accumulator = memory();
        """
        analyzer = self.parse_and_analyze(code)
        
        counter_symbol = analyzer.symbol_table.lookup("counter")
        accumulator_symbol = analyzer.symbol_table.lookup("accumulator")
        
        assert counter_symbol is not None
        assert counter_symbol.symbol_type == "memory"
        assert counter_symbol.is_mutable == True
        
        assert accumulator_symbol is not None
        assert accumulator_symbol.symbol_type == "memory"
        
    def test_memory_operations(self):
        """Test memory read/write operations."""
        code = """
        mem counter = memory(0);
        let current = read(counter);
        write(counter, current + 1);
        """
        analyzer = self.parse_and_analyze(code)
        
        # Should have no errors
        errors = self.get_diagnostics_by_level(analyzer, DiagnosticLevel.ERROR)
        assert len(errors) == 0
        
    def test_undefined_memory_error(self):
        """Test error on undefined memory access."""
        code = """
        let value = read(undefined_memory);
        """
        analyzer = self.parse_and_analyze(code)
        
        errors = self.get_diagnostics_by_level(analyzer, DiagnosticLevel.ERROR)
        assert len(errors) > 0
    
    # =========================================================================
    # Function Analysis Tests
    # =========================================================================
    
    def test_function_definition(self):
        """Test function definition analysis."""
        code = """
        func add(a, b) {
            return a + b;
        }
        """
        analyzer = self.parse_and_analyze(code)
        
        add_symbol = analyzer.symbol_table.lookup("add")
        assert add_symbol is not None
        assert add_symbol.symbol_type == "function"
        
    def test_function_call_analysis(self):
        """Test function call analysis."""
        code = """
        func multiply(x, y) {
            return x * y;
        }
        
        let result = multiply(3, 4);
        """
        analyzer = self.parse_and_analyze(code)
        
        # Should resolve function call
        errors = self.get_diagnostics_by_level(analyzer, DiagnosticLevel.ERROR)
        assert len(errors) == 0
        
    def test_function_parameter_scoping(self):
        """Test function parameter scoping."""
        code = """
        func test_scope(param) {
            let local_var = param + 1;
            return local_var;
        }
        
        let global_var = 10;
        """
        analyzer = self.parse_and_analyze(code)
        
        # Function should be defined
        func_symbol = analyzer.symbol_table.lookup("test_scope")
        assert func_symbol is not None
        
        # Global variable should be accessible
        global_symbol = analyzer.symbol_table.lookup("global_var")
        assert global_symbol is not None
    
    # =========================================================================
    # Entity and Property Tests  
    # =========================================================================
    
    def test_entity_placement(self):
        """Test entity placement analysis."""
        code = """
        let lamp = Place("small-lamp", 0, 0);
        """
        analyzer = self.parse_and_analyze(code)
        
        lamp_symbol = analyzer.symbol_table.lookup("lamp")
        assert lamp_symbol is not None
        assert lamp_symbol.symbol_type == "entity"
        
    def test_entity_property_assignment(self):
        """Test entity property assignment."""
        code = """
        let lamp = Place("small-lamp", 0, 0);
        lamp.enable = input(0) > 0;
        """
        analyzer = self.parse_and_analyze(code)
        
        # Should have no errors
        errors = self.get_diagnostics_by_level(analyzer, DiagnosticLevel.ERROR)
        assert len(errors) == 0
        
    def test_undefined_entity_property_error(self):
        """Test error on undefined entity property access."""
        code = """
        undefined_entity.property = 1;
        """
        analyzer = self.parse_and_analyze(code)
        
        errors = self.get_diagnostics_by_level(analyzer, DiagnosticLevel.ERROR)
        assert len(errors) > 0
    
    # =========================================================================
    # Module and Import Tests
    # =========================================================================
    
    def test_import_statement(self):
        """Test import statement analysis."""
        code = """
        import "stdlib/math.fcdsl" as math;
        """
        analyzer = self.parse_and_analyze(code)
        
        # Import should create a module symbol
        math_symbol = analyzer.symbol_table.lookup("math")
        assert math_symbol is not None
        assert math_symbol.symbol_type == "module"
        
    def test_module_method_call(self):
        """Test module method call analysis."""
        code = """
        import "stdlib/math.fcdsl" as math;
        let result = math.sqrt(16);
        """
        analyzer = self.parse_and_analyze(code)
        
        # Should resolve module method call
        errors = self.get_diagnostics_by_level(analyzer, DiagnosticLevel.ERROR)
        # Note: May have warnings about unknown module, but no critical errors
        critical_errors = [e for e in errors if "undefined" in e.message.lower()]
        assert len(critical_errors) == 0
    
    # =========================================================================
    # Error Recovery Tests
    # =========================================================================
    
    def test_multiple_errors_collection(self):
        """Test that multiple errors are collected."""
        code = """
        let x = undefined1 + undefined2;
        let y = another_undefined;
        mem bad_mem = memory(invalid_expr);
        """
        analyzer = self.parse_and_analyze(code)
        
        errors = self.get_diagnostics_by_level(analyzer, DiagnosticLevel.ERROR)
        # Should have multiple errors but continue analysis
        assert len(errors) >= 2
        
    def test_strict_type_checking(self):
        """Test strict type checking mode."""
        code = """
        let a = input(0);
        let b = 10;
        let mixed = a + b;  # Should be error in strict mode
        """
        
        # Non-strict mode: should be warning
        analyzer_normal = self.parse_and_analyze(code, strict_types=False)
        warnings_normal = self.get_diagnostics_by_level(analyzer_normal, DiagnosticLevel.WARNING)
        errors_normal = self.get_diagnostics_by_level(analyzer_normal, DiagnosticLevel.ERROR)
        
        assert len(warnings_normal) > 0
        assert len(errors_normal) == 0
        
        # Strict mode: should be error
        analyzer_strict = self.parse_and_analyze(code, strict_types=True)
        errors_strict = self.get_diagnostics_by_level(analyzer_strict, DiagnosticLevel.ERROR)
        
        assert len(errors_strict) > 0


class TestSemanticAnalysisIntegration:
    """Test semantic analysis on actual test files."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent / "dsl_compiler" / "tests"
        
    @pytest.mark.parametrize("test_file", [
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
        "11_edge_cases.fcdsl"
    ])
    def test_semantic_analysis_on_test_files(self, test_file):
        """Test semantic analysis on all test files."""
        file_path = self.test_dir / test_file
        
        if not file_path.exists():
            pytest.skip(f"Test file {test_file} not found")
            
        try:
            # Parse and analyze
            diagnostics = analyze_file(str(file_path))
            
            # Count diagnostics
            errors = [d for d in diagnostics.diagnostics if d.level == DiagnosticLevel.ERROR]
            warnings = [d for d in diagnostics.diagnostics if d.level == DiagnosticLevel.WARNING]
            
            print(f"✓ {test_file}: {len(errors)} errors, {len(warnings)} warnings")
            
            # Report any errors found
            if errors:
                for error in errors:
                    print(f"  ERROR: {error.message}")
            
            # Analysis should complete without crashing
            assert diagnostics is not None
            
        except Exception as e:
            pytest.fail(f"Semantic analysis failed on {test_file}: {e}")


if __name__ == "__main__":
    # Run specific tests during development
    pytest.main([__file__, "-v"])
