"""
Tests for emit.py - Blueprint emission functionality.
"""

import pytest
from dsl_compiler.src.parser import DSLParser
from dsl_compiler.src.semantic import SemanticAnalyzer, analyze_program
from dsl_compiler.src.lowerer import lower_program
from dsl_compiler.src.emit import BlueprintEmitter


class TestBlueprintEmitter:
    """Test blueprint emission functionality."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    @pytest.fixture 
    def analyzer(self):
        return SemanticAnalyzer()

    def test_emitter_initialization(self):
        """Test emitter can be initialized."""
        emitter = BlueprintEmitter()
        assert emitter is not None

    def test_basic_emission(self, parser, analyzer):
        """Test basic blueprint emission."""
        program = parser.parse("Signal x = 42;")
        analyze_program(program, strict_types=False, analyzer=analyzer)
        ir_operations, _, signal_map = lower_program(program, analyzer)
        
        emitter = BlueprintEmitter(signal_type_map=signal_map)
        blueprint = emitter.emit_blueprint(ir_operations)
        
        assert blueprint is not None
        assert len(emitter.entities) > 0

    def test_emission_sample_files(self, parser, analyzer):
        """Test emission on sample files."""
        import os
        sample_files = [
            "tests/sample_programs/01_basic_arithmetic.fcdsl",
            "tests/sample_programs/04_memory.fcdsl",
        ]
        
        for file_path in sample_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    code = f.read()
                program = parser.parse(code)
                analyze_program(program, strict_types=False, analyzer=analyzer)
                ir_operations, _, signal_map = lower_program(program, analyzer)
                
                emitter = BlueprintEmitter(signal_type_map=signal_map)
                blueprint = emitter.emit_blueprint(ir_operations)
                
                assert blueprint is not None
                assert len(emitter.entities) > 0

    def test_signal_name_resolution(self):
        """Test signal name resolution with mapping."""
        signal_map = {"__v1": "signal-A", "__v2": "signal-B"}
        emitter = BlueprintEmitter(signal_type_map=signal_map)
        
        # Test that implicit signals are resolved
        resolved = emitter._get_signal_name("__v1")
        assert resolved == "signal-A"
        
        # Test that explicit signals pass through
        resolved = emitter._get_signal_name("iron-plate")
        assert resolved == "iron-plate"


class TestWarningAnalysis:
    """Test for analyzing and addressing warnings."""

    def test_no_memory_signal_warnings(self):
        """Test that memory signal warnings are eliminated."""
        from dsl_compiler.src.parser import DSLParser
        from dsl_compiler.src.semantic import SemanticAnalyzer, analyze_program
        from dsl_compiler.src.lowerer import lower_program
        from dsl_compiler.src.emit import BlueprintEmitter
        
        code = '''
        Memory counter = 0;
        Signal current = read(counter);
        write(counter, current + 1);
        '''
        
        parser = DSLParser()
        analyzer = SemanticAnalyzer()
        
        program = parser.parse(code)
        analyze_program(program, strict_types=False, analyzer=analyzer)
        ir_operations, _, signal_map = lower_program(program, analyzer)
        
        emitter = BlueprintEmitter(signal_type_map=signal_map)
        blueprint = emitter.emit_blueprint(ir_operations)
        
        # Check for warnings
        warnings = []
        if hasattr(emitter.diagnostics, 'diagnostics'):
            warnings = [d for d in emitter.diagnostics.diagnostics if d.level.name == 'WARNING']
        
        # Should not have memory signal warnings
        memory_warnings = [w for w in warnings if "Unknown signal type for memory" in w.message]
        assert len(memory_warnings) == 0, f"Found memory signal warnings: {[w.message for w in memory_warnings]}"