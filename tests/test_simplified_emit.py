#!/usr/bin/env python3
"""
Test suite for the simplified emit system.
"""

import pytest
from pathlib import Path

from dsl_compiler.src.emit import (
    DraftsmanEntityFactory, MemoryCircuitBuilder, BlueprintEmitter,
    LayoutEngine, DRAFTSMAN_AVAILABLE
)

from dsl_compiler.src.ir import *
from dsl_compiler.src.semantic import DiagnosticCollector


@pytest.mark.skipif(not DRAFTSMAN_AVAILABLE, reason="Draftsman not available")
class TestSimplifiedDraftsmanEntityFactory:
    """Test the simplified entity factory."""

    def setup_method(self):
        """Set up for each test."""
        self.factory = DraftsmanEntityFactory()

    def test_signal_loading(self):
        """Test that signals are loaded properly."""
        # Should have loaded many signals
        assert len(self.factory.all_valid_signals) > 1000
        assert len(self.factory.virtual_signals) > 30
        
        # Should have specific signals
        assert 'signal-0' in self.factory.virtual_signals
        assert 'signal-A' in self.factory.virtual_signals
        assert 'iron-plate' in self.factory.all_valid_signals
        
        print(f"Total signals: {len(self.factory.all_valid_signals)}")
        print(f"Virtual signals: {len(self.factory.virtual_signals)}")

    def test_entity_creation(self):
        """Test creating entities with the factory."""
        # Test basic entities
        combinator = self.factory.create_entity('constant-combinator', tile_position=(0, 0))
        assert combinator is not None
        
        lamp = self.factory.create_entity('small-lamp', tile_position=(1, 1))
        assert lamp is not None
        
        # Test entity with prototype name
        chest = self.factory.create_entity('iron-chest', tile_position=(2, 2))
        assert chest is not None

    def test_signal_validation(self):
        """Test signal validation."""
        assert self.factory.is_valid_signal('signal-0')
        assert self.factory.is_valid_signal('signal-A')
        assert self.factory.is_valid_signal('iron-plate')
        assert not self.factory.is_valid_signal('invalid-signal-name')

    def test_virtual_signal_detection(self):
        """Test virtual signal detection."""
        assert self.factory.is_virtual_signal('signal-0')
        assert self.factory.is_virtual_signal('signal-A')
        assert not self.factory.is_virtual_signal('iron-plate')
        assert not self.factory.is_virtual_signal('invalid-signal')


@pytest.mark.skipif(not DRAFTSMAN_AVAILABLE, reason="Draftsman not available")
class TestBlueprintEmitter:
    """Test blueprint generation with simplified emitter."""

    def setup_method(self):
        """Set up for each test."""
        self.signal_type_map = {
            '__v1': 'signal-A',
            '__v2': 'signal-B',
            '__v3': 'signal-C',
        }
        self.emitter = BlueprintEmitter(self.signal_type_map)

    def test_constant_emission(self):
        """Test emitting constant combinators."""
        const_op = IR_Const('const1', '__v1', 42, None)
        self.emitter.emit_ir_operation(const_op)
        
        assert len(self.emitter.blueprint.entities) == 1
        assert 'const1' in self.emitter.entities

    def test_arithmetic_emission(self):
        """Test emitting arithmetic combinators."""
        arith_op = IR_Arith('arith1', '__v2', '+', 'const1', 'const2', None)
        self.emitter.emit_ir_operation(arith_op)
        
        assert 'arith1' in self.emitter.entities
        
    def test_memory_emission(self):
        """Test emitting memory circuits."""
        mem_create = IR_MemCreate('mem1', 'signal-A', 0, None)
        self.emitter.emit_ir_operation(mem_create)
        
        # Should have created memory components
        assert len(self.emitter.memory_builder.memory_modules) > 0

    def test_entity_placement(self):
        """Test placing entities."""
        place_op = IR_PlaceEntity('entity1', 'small-lamp', 5, 5, {}, None)
        self.emitter.emit_ir_operation(place_op)
        
        assert 'entity1' in self.emitter.entities
        assert len(self.emitter.blueprint.entities) > 0

    def test_signal_name_resolution(self):
        """Test signal name resolution."""
        # Test implicit signal mapping
        signal_name = self.emitter._get_signal_name('__v1')
        assert signal_name == 'signal-A'
        
        # Test valid signal passthrough
        signal_name = self.emitter._get_signal_name('iron-plate')
        assert signal_name == 'iron-plate'
        
        # Test fallback
        signal_name = self.emitter._get_signal_name('__v999')
        assert signal_name.startswith('signal-')

    def test_complete_ir_emission(self):
        """Test emitting a complete IR sequence."""
        ir_ops = [
            IR_Const('const1', '__v1', 10, None),
            IR_Const('const2', '__v2', 20, None),
            IR_Arith('sum', '__v3', '+', 'const1', 'const2', None),
            IR_PlaceEntity('lamp', 'small-lamp', 0, 0, {}, None),
        ]
        
        for op in ir_ops:
            self.emitter.emit_ir_operation(op)
        
        # Should have created multiple entities
        assert len(self.emitter.blueprint.entities) >= 4
        assert len(self.emitter.entities) >= 4


@pytest.mark.skipif(not DRAFTSMAN_AVAILABLE, reason="Draftsman not available")
class TestIntegration:
    """Integration tests for the emit system."""
    
    def test_sample_compilation(self):
        """Test that we can compile a sample program end-to-end."""
        from dsl_compiler.src.parser import DSLParser
        from dsl_compiler.src.semantic import analyze_program, SemanticAnalyzer
        from dsl_compiler.src.lowerer import lower_program
        from dsl_compiler.src.emit import emit_blueprint_string
        
        # Simple test program
        code = '''
        Signal a = input("iron-plate", 0);
        Signal b = input("copper-plate", 1);
        Signal sum = a + b;
        Entity lamp = Place("small-lamp", 0, 0);
        lamp.enable = sum > 10;
        '''
        
        # Parse
        parser = DSLParser()
        program = parser.parse(code)
        
        # Semantic analysis
        analyzer = SemanticAnalyzer()
        semantic_diagnostics = analyze_program(program, strict_types=False, analyzer=analyzer)
        assert not semantic_diagnostics.has_errors()
        
        # IR generation
        ir_operations, lowering_diagnostics, signal_type_map = lower_program(program, analyzer)
        assert not lowering_diagnostics.has_errors()
        
        # Blueprint generation
        blueprint_string, emit_diagnostics = emit_blueprint_string(ir_operations, "Test", signal_type_map)
        assert not emit_diagnostics.has_errors()
        assert len(blueprint_string) > 0
        
        print(f"Generated blueprint with {len(blueprint_string)} characters")


if __name__ == "__main__":
    pytest.main([__file__])