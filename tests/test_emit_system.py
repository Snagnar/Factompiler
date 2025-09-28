#!/usr/bin/env python3
"""
Comprehensive tests for the emit system and draftsman integration.
"""

import pytest
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dsl_compiler.src.emit import (
    DraftsmanEntityFactory, 
    MemoryCircuitBuilder, 
    BlueprintEmitter,
    LayoutEngine,
    DRAFTSMAN_AVAILABLE
)
from dsl_compiler.src.ir import *


class TestDraftsmanEntityFactory:
    """Test the entity factory that dynamically discovers entities."""

    def setup_method(self):
        """Set up for each test."""
        if not DRAFTSMAN_AVAILABLE:
            pytest.skip("Draftsman not available")
        self.factory = DraftsmanEntityFactory()

    def test_entity_discovery(self):
        """Test that the factory discovers entities properly."""
        # Should have discovered many entity classes
        assert len(self.factory.entity_classes) > 10
        
        # Should have basic combinators
        assert 'constant-combinator' in self.factory.entity_classes
        assert 'arithmetic-combinator' in self.factory.entity_classes
        assert 'decider-combinator' in self.factory.entity_classes
        
        # Should have common entities
        assert 'small-lamp' in self.factory.entity_classes
        assert 'train-stop' in self.factory.entity_classes

    def test_signal_loading(self):
        """Test that signals are loaded properly."""
        # Should have loaded virtual signals
        assert 'virtual' in self.factory.valid_signals
        assert len(self.factory.valid_signals['virtual']) > 10
        
        # Should have basic virtual signals
        assert 'signal-0' in self.factory.valid_signals['virtual']
        assert 'signal-A' in self.factory.valid_signals['virtual']
        assert 'signal-each' in self.factory.valid_signals['virtual']

    def test_entity_creation(self):
        """Test creating entities with the factory."""
        # Test combinator creation
        combinator = self.factory.create_entity('constant-combinator', tile_position=(0, 0))
        assert combinator is not None
        
        # Test lamp creation
        lamp = self.factory.create_entity('small-lamp', tile_position=(1, 1))
        assert lamp is not None
        
        # Test unknown entity
        with pytest.raises(ValueError, match="Unknown entity prototype"):
            self.factory.create_entity('nonexistent-entity')

    def test_signal_validation(self):
        """Test signal validation."""
        # Valid signals
        assert self.factory.is_valid_signal('signal-0')
        assert self.factory.is_valid_signal('signal-A')
        assert self.factory.is_valid_signal('iron-plate')  # Should be an item signal
        
        # Invalid signals
        assert not self.factory.is_valid_signal('invalid-signal-name')

    def test_signal_type_detection(self):
        """Test signal type detection."""
        assert self.factory.get_signal_type('signal-0') == 'virtual'
        assert self.factory.get_signal_type('iron-plate') == 'item'
        assert self.factory.get_signal_type('water') == 'fluid'


class TestMemoryCircuitBuilder:
    """Test memory circuit generation."""

    def setup_method(self):
        """Set up for each test."""
        if not DRAFTSMAN_AVAILABLE:
            pytest.skip("Draftsman not available")
        
        from draftsman.blueprintable import Blueprint
        self.blueprint = Blueprint()
        self.layout = LayoutEngine()
        self.factory = DraftsmanEntityFactory()
        self.memory_builder = MemoryCircuitBuilder(self.layout, self.blueprint, self.factory)

    def test_memory_circuit_creation(self):
        """Test creating a memory circuit."""
        components = self.memory_builder.build_sr_latch("test_memory", "signal-A", initial_value=5)
        
        # Should have at least the memory combinator
        assert 'memory_combinator' in components
        assert components['memory_combinator'].entity_id == "test_memory_memory"
        
        # Should have added entities to the blueprint
        assert len(self.blueprint.entities) >= 1

    def test_memory_wiring(self):
        """Test memory circuit wiring."""
        self.memory_builder.build_sr_latch("test_memory", "signal-A")
        
        # Wiring should not fail (even if simplified)
        self.memory_builder.wire_sr_latch("test_memory")


class TestBlueprintEmitter:
    """Test the complete blueprint emission system."""

    def setup_method(self):
        """Set up for each test."""
        if not DRAFTSMAN_AVAILABLE:
            pytest.skip("Draftsman not available")
        self.emitter = BlueprintEmitter()

    def test_constant_emission(self):
        """Test emitting constant combinators."""
        op = IR_Const("const_1", "signal-A")
        op.value = 42
        
        self.emitter.emit_constant(op)
        
        # Should have created an entity
        assert len(self.emitter.blueprint.entities) == 1
        assert "const_1" in self.emitter.entities

    def test_arithmetic_emission(self):
        """Test emitting arithmetic combinators."""
        op = IR_Arith("arith_1", "signal-B")
        op.op = "+"
        op.left = SignalRef("signal-A", "const_1")
        op.right = 5
        
        self.emitter.emit_arithmetic(op)
        
        # Should have created an entity
        assert len(self.emitter.blueprint.entities) == 1
        assert "arith_1" in self.emitter.entities

    def test_memory_emission(self):
        """Test emitting memory operations."""
        # Create memory
        create_op = IR_MemCreate("mem_test", "signal-C", 0)
        self.emitter.emit_memory_create(create_op)
        
        # Should have created memory components
        assert len(self.emitter.blueprint.entities) >= 1
        assert "mem_test" in self.emitter.memory_builder.memory_modules

    def test_entity_placement(self):
        """Test placing entities."""
        op = IR_PlaceEntity("entity_1", "small-lamp", 5, 3)
        
        self.emitter.emit_place_entity(op)
        
        # Should have created the entity
        assert len(self.emitter.blueprint.entities) == 1
        assert "entity_1" in self.emitter.entities

    def test_signal_name_resolution(self):
        """Test signal name resolution and mapping."""
        # Test implicit signal mapping
        assert self.emitter._get_signal_name("__v1") == "signal-A"
        assert self.emitter._get_signal_name("__v2") == "signal-B"
        
        # Test explicit signals
        assert self.emitter._get_signal_name("iron-plate") == "iron-plate"
        assert self.emitter._get_signal_name("signal-0") == "signal-0"

    def test_complete_ir_emission(self):
        """Test emitting a complete IR program."""
        # Create a simple IR program: constant -> arithmetic -> output
        ir_ops = [
            IR_Const("const_1", "signal-A"),
            IR_Arith("arith_1", "signal-B"),
        ]
        
        # Set up the operations
        ir_ops[0].value = 10
        ir_ops[1].op = "*"
        ir_ops[1].left = SignalRef("signal-A", "const_1")
        ir_ops[1].right = 2
        
        # Emit the blueprint
        blueprint = self.emitter.emit_blueprint(ir_ops)
        
        # Should have created entities
        assert len(blueprint.entities) >= 2
        assert not self.emitter.diagnostics.has_errors()


class TestEntityDiscovery:
    """Test discovery of all available entities from draftsman."""

    def setup_method(self):
        """Set up for each test."""
        if not DRAFTSMAN_AVAILABLE:
            pytest.skip("Draftsman not available")

    def test_discover_all_entities(self):
        """Discover and list all available entities."""
        factory = DraftsmanEntityFactory()
        
        print(f"\nDiscovered {len(factory.entity_classes)} entity types:")
        for prototype, entity_class in sorted(factory.entity_classes.items()):
            print(f"  {prototype} -> {entity_class.__name__}")
        
        # Should have discovered many entities
        assert len(factory.entity_classes) > 20

    def test_discover_all_signals(self):
        """Discover and list all available signals."""
        factory = DraftsmanEntityFactory()
        
        total_signals = 0
        for signal_type, signals in factory.valid_signals.items():
            print(f"\n{signal_type.upper()} signals ({len(signals)}):")
            print(f"  {signals[:10]}...")  # First 10 signals
            total_signals += len(signals)
        
        print(f"\nTotal signals available: {total_signals}")
        
        # Should have discovered many signals
        assert total_signals > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])