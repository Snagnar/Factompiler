# emit.py
"""
Blueprint emission module for the Factorio Circuit DSL.

This module converts IR operations into actual Factorio combinators and entities
using the factorio-draftsman library to generate blueprint JSON.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Add draftsman to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "factorio-draftsman"))

try:
    from draftsman.blueprintable import Blueprint
    from draftsman.entity import (
        ConstantCombinator, ArithmeticCombinator, DeciderCombinator,
        Lamp, TrainStop, Container
    )
    from draftsman.constants import Direction
    DRAFTSMAN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: factorio-draftsman not available: {e}")
    DRAFTSMAN_AVAILABLE = False
    # Mock classes for development
    class Blueprint: pass
    class ConstantCombinator: pass
    class ArithmeticCombinator: pass
    class DeciderCombinator: pass

# Try relative imports when run as module
try:
    from .ir import *
    from .semantic import DiagnosticCollector
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.ir import *
    from src.semantic import DiagnosticCollector


@dataclass
class EntityPlacement:
    """Information about placed entity for wiring."""
    entity: Any  # Draftsman entity
    entity_id: str
    position: Tuple[int, int]
    output_signals: Dict[str, str]  # signal_type -> wire_color
    input_signals: Dict[str, str]   # signal_type -> wire_color


class LayoutEngine:
    """Simple layout engine for entity placement."""
    
    def __init__(self):
        self.next_x = 0
        self.next_y = 0
        self.row_height = 3
        self.entities_per_row = 8
        self.current_row_count = 0
        self.used_positions = set()
    
    def get_next_position(self) -> Tuple[int, int]:
        """Get next available position for entity placement."""
        while True:
            pos = (self.next_x, self.next_y)
            
            if pos not in self.used_positions:
                self.used_positions.add(pos)
                break
            
            # Move to next position
            self.current_row_count += 1
            if self.current_row_count >= self.entities_per_row:
                # Move to next row
                self.next_x = 0
                self.next_y += self.row_height
                self.current_row_count = 0
            else:
                self.next_x += 3  # Space entities further apart
        
        # Advance for next call
        self.current_row_count += 1
        if self.current_row_count >= self.entities_per_row:
            self.next_x = 0
            self.next_y += self.row_height
            self.current_row_count = 0
        else:
            self.next_x += 3
        
        return pos


class BlueprintEmitter:
    """Converts IR operations to Factorio blueprint using Draftsman."""
    
    def __init__(self, signal_type_map: Dict[str, str] = None):
        if not DRAFTSMAN_AVAILABLE:
            raise RuntimeError("factorio-draftsman library not available")
        
        self.blueprint = Blueprint()
        self.blueprint.label = "DSL Generated Blueprint"
        self.blueprint.version = (2, 0)
        
        self.layout = LayoutEngine()
        self.diagnostics = DiagnosticCollector()
        
        # Signal type mapping from IR builder
        self.signal_type_map = signal_type_map or {}
        
        # Entity tracking
        self.entities: Dict[str, EntityPlacement] = {}
        self.next_entity_number = 1
        
        # Signal tracking for wiring
        self.signal_sources: Dict[str, str] = {}  # signal_id -> entity_id
        self.signal_sinks: Dict[str, List[str]] = {}  # signal_id -> [entity_ids]
    
    def emit_blueprint(self, ir_operations: List[IRNode]) -> Blueprint:
        """Convert IR operations to blueprint."""
        try:
            # Process all IR operations
            for op in ir_operations:
                self.emit_ir_operation(op)
            
            # Add wiring connections
            self.create_circuit_connections()
            
            return self.blueprint
            
        except Exception as e:
            self.diagnostics.error(f"Blueprint emission failed: {e}")
            raise
    
    def emit_ir_operation(self, op: IRNode):
        """Emit a single IR operation."""
        if isinstance(op, IR_Const):
            self.emit_constant(op)
        elif isinstance(op, IR_Arith):
            self.emit_arithmetic(op)
        elif isinstance(op, IR_Decider):
            self.emit_decider(op)
        elif isinstance(op, IR_Input):
            self.emit_input(op)
        elif isinstance(op, IR_MemCreate):
            self.emit_memory_create(op)
        elif isinstance(op, IR_MemRead):
            self.emit_memory_read(op)
        elif isinstance(op, IR_MemWrite):
            self.emit_memory_write(op)
        elif isinstance(op, IR_PlaceEntity):
            self.emit_place_entity(op)
        elif isinstance(op, IR_EntityPropWrite):
            self.emit_entity_prop_write(op)
        elif isinstance(op, IR_Bundle):
            self.emit_bundle(op)
        else:
            self.diagnostics.warning(f"Unknown IR operation: {type(op)}")
    
    def emit_constant(self, op: IR_Const):
        """Emit constant combinator for IR_Const."""
        pos = self.layout.get_next_position()
        
        combinator = ConstantCombinator(tile_position=pos)
        
        # Set the constant signal
        section = combinator.add_section()
        signal_name = self._get_signal_name(op.output_type)
        section.set_signal(
            index=0,
            name=signal_name,
            count=op.value
        )
        
        self.blueprint.entities.append(combinator)
        
        # Track entity for wiring
        placement = EntityPlacement(
            entity=combinator,
            entity_id=op.node_id,
            position=pos,
            output_signals={signal_name: "red"},  # Output on red wire
            input_signals={}
        )
        self.entities[op.node_id] = placement
        self.signal_sources[op.node_id] = op.node_id
    
    def emit_arithmetic(self, op: IR_Arith):
        """Emit arithmetic combinator for IR_Arith."""
        pos = self.layout.get_next_position()
        
        combinator = ArithmeticCombinator(tile_position=pos)
        
        # Configure arithmetic operation
        # Note: This is a simplified implementation
        # Real implementation would need proper signal mapping
        combinator.first_operand = self._get_signal_name(op.left)
        combinator.second_operand = self._get_signal_name(op.right)
        combinator.operation = op.op
        combinator.output_signal = self._get_signal_name(op.output_type)
        
        self.blueprint.entities.append(combinator)
        
        # Track entity for wiring
        signal_name = self._get_signal_name(op.output_type)
        placement = EntityPlacement(
            entity=combinator,
            entity_id=op.node_id,
            position=pos,
            output_signals={signal_name: "red"},
            input_signals={}  # Will be populated when wiring
        )
        self.entities[op.node_id] = placement
        self.signal_sources[op.node_id] = op.node_id
        
        # Track signal dependencies
        self._add_signal_sink(op.left, op.node_id)
        self._add_signal_sink(op.right, op.node_id)
    
    def emit_decider(self, op: IR_Decider):
        """Emit decider combinator for IR_Decider."""
        pos = self.layout.get_next_position()
        
        combinator = DeciderCombinator(tile_position=pos)
        
        # Configure decider operation
        combinator.first_operand = self._get_signal_name(op.left)
        combinator.second_operand = self._get_operand_value(op.right)
        combinator.comparator = op.test_op
        combinator.output_signal = self._get_signal_name(op.output_type)
        combinator.copy_count_from_input = op.output_value == "input"
        if not combinator.copy_count_from_input:
            combinator.constant = op.output_value
        
        self.blueprint.entities.append(combinator)
        
        # Track entity for wiring
        signal_name = self._get_signal_name(op.output_type)
        placement = EntityPlacement(
            entity=combinator,
            entity_id=op.node_id,
            position=pos,
            output_signals={signal_name: "red"},
            input_signals={}
        )
        self.entities[op.node_id] = placement
        self.signal_sources[op.node_id] = op.node_id
        
        # Track signal dependencies
        self._add_signal_sink(op.left, op.node_id)
        if isinstance(op.right, str):  # Signal reference
            self._add_signal_sink(op.right, op.node_id)
    
    def emit_input(self, op: IR_Input):
        """Emit input operation (placeholder combinator with label)."""
        pos = self.layout.get_next_position()
        
        # Use constant combinator as input placeholder
        combinator = ConstantCombinator(tile_position=pos)
        
        # Add section with input signal type
        section = combinator.add_section()
        signal_name = self._get_signal_name(op.output_type)
        section.set_signal(
            index=0,
            name=signal_name,
            count=0  # Will be set externally
        )
        
        self.blueprint.entities.append(combinator)
        
        # Track entity
        placement = EntityPlacement(
            entity=combinator,
            entity_id=op.node_id,
            position=pos,
            output_signals={op.output_type: "red"},
            input_signals={}
        )
        self.entities[op.node_id] = placement
        self.signal_sources[op.node_id] = op.node_id
    
    def emit_memory_create(self, op: IR_MemCreate):
        """Emit memory module creation (simplified with constant combinator)."""
        pos = self.layout.get_next_position()
        
        # Memory implementation would be more complex in real scenarios
        # For now, use a constant combinator to represent memory state
        combinator = ConstantCombinator(tile_position=pos)
        
        section = combinator.add_section()
        section.set_signal(
            index=0,
            name=op.signal_type,
            count=0 if isinstance(op.initial_value, int) and op.initial_value == 0 else 1
        )
        
        self.blueprint.entities.append(combinator)
        
        placement = EntityPlacement(
            entity=combinator,
            entity_id=op.memory_id,
            position=pos,
            output_signals={op.signal_type: "red"},
            input_signals={}
        )
        self.entities[op.memory_id] = placement
        self.signal_sources[op.memory_id] = op.memory_id
    
    def emit_memory_read(self, op: IR_MemRead):
        """Emit memory read operation."""
        # Memory read would connect to the memory module
        # For simplicity, this is handled in wiring phase
        self.signal_sources[op.node_id] = op.memory_id
    
    def emit_memory_write(self, op: IR_MemWrite):
        """Emit memory write operation."""
        # Memory write would require complex SR latch circuit
        # For now, just track the dependency
        self._add_signal_sink(op.data_signal, op.memory_id)
        self._add_signal_sink(op.write_enable, op.memory_id)
    
    def emit_place_entity(self, op: IR_PlaceEntity):
        """Emit entity placement."""
        # Use specified position, but offset to avoid combinators
        pos = (op.x + 20, op.y)  # Offset entities to the right
        
        if op.prototype == "small-lamp":
            entity = Lamp(tile_position=pos)
        elif op.prototype == "train-stop":
            entity = TrainStop(tile_position=pos)
        elif op.prototype == "iron-chest":
            entity = Container("iron-chest", tile_position=pos)
        else:
            self.diagnostics.warning(f"Unknown entity prototype: {op.prototype}")
            return
        
        self.blueprint.entities.append(entity)
        
        placement = EntityPlacement(
            entity=entity,
            entity_id=op.entity_id,
            position=pos,
            output_signals={},
            input_signals={}
        )
        self.entities[op.entity_id] = placement
    
    def emit_entity_prop_write(self, op: IR_EntityPropWrite):
        """Emit entity property write (circuit network connection)."""
        # This would connect signals to entity properties
        if op.entity_id in self.entities:
            entity_placement = self.entities[op.entity_id]
            # Track that this entity needs this signal as input
            entity_placement.input_signals[f"{op.property_name}_input"] = "green"
            self._add_signal_sink(op.value, op.entity_id)
    
    def emit_bundle(self, op: IR_Bundle):
        """Emit bundle operation (multiple signals on same wire)."""
        # Bundles are handled in the wiring phase
        # Track all input signals as sources for this bundle
        for signal_type, signal_ref in op.inputs.items():
            if isinstance(signal_ref, str):
                self.signal_sources[f"{op.node_id}_{signal_type}"] = signal_ref
    
    def create_circuit_connections(self):
        """Create circuit wire connections between entities."""
        # This is where we'd implement the complex wiring logic
        # For now, implement basic red-wire connections
        
        for signal_id, sink_entities in self.signal_sinks.items():
            if signal_id in self.signal_sources:
                source_entity_id = self.signal_sources[signal_id]
                
                if source_entity_id in self.entities:
                    source_placement = self.entities[source_entity_id]
                    
                    for sink_entity_id in sink_entities:
                        if sink_entity_id in self.entities:
                            sink_placement = self.entities[sink_entity_id]
                            
                            # Add circuit connection
                            try:
                                self.blueprint.add_circuit_connection(
                                    color="red",
                                    entity_1=source_placement.entity,
                                    entity_2=sink_placement.entity,
                                    side_1="output",
                                    side_2="input"
                                )
                            except Exception as e:
                                self.diagnostics.warning(f"Failed to connect {source_entity_id} -> {sink_entity_id}: {e}")
    
    def _get_signal_name(self, operand) -> str:
        """Get signal name from operand using the signal type mapping."""
        # Convert to string first to handle SignalRef and other objects
        operand_str = str(operand)
        
        # First check if this is a mapped implicit signal type
        if operand_str in self.signal_type_map:
            return self.signal_type_map[operand_str]
        
        # Clean up signal reference strings that may have been concatenated
        clean_name = operand_str.split('@')[0]  # Remove anything after @
        
        # Handle bundle references like "Bundle[__v4, __v5, __v6]"
        if clean_name.startswith('Bundle[') and clean_name.endswith(']'):
            # For bundle references, use a generic signal for bundle operations
            return "signal-each"  # Special signal for bundle operations
        
        # Check if the clean name is in the mapping
        if clean_name in self.signal_type_map:
            return self.signal_type_map[clean_name]
            
        # Handle integer constants
        if isinstance(operand, int):
            return "signal-0"
            
        # Otherwise, assume it's a valid Factorio signal name and let draftsman validate it
        # The DSL should only use actual Factorio signal names, no custom mappings
        return clean_name
    
    def _get_operand_value(self, operand):
        """Get operand value for decider combinator."""
        if isinstance(operand, int):
            return operand
        elif isinstance(operand, str):
            return self._get_signal_name(operand)
        else:
            return str(operand)
    
    def _add_signal_sink(self, signal_ref, entity_id: str):
        """Track that an entity needs this signal as input."""
        if isinstance(signal_ref, str):
            if signal_ref not in self.signal_sinks:
                self.signal_sinks[signal_ref] = []
            if entity_id not in self.signal_sinks[signal_ref]:
                self.signal_sinks[signal_ref].append(entity_id)


# =============================================================================
# Public API
# =============================================================================

def emit_blueprint(ir_operations: List[IRNode], label: str = "DSL Generated", signal_type_map: Dict[str, str] = None) -> Tuple[Blueprint, DiagnosticCollector]:
    """Convert IR operations to Factorio blueprint."""
    emitter = BlueprintEmitter(signal_type_map)
    emitter.blueprint.label = label
    
    try:
        blueprint = emitter.emit_blueprint(ir_operations)
        return blueprint, emitter.diagnostics
    except Exception as e:
        emitter.diagnostics.error(f"Blueprint emission failed: {e}")
        return emitter.blueprint, emitter.diagnostics


def emit_blueprint_string(ir_operations: List[IRNode], label: str = "DSL Generated", signal_type_map: Dict[str, str] = None) -> Tuple[str, DiagnosticCollector]:
    """Convert IR operations to Factorio blueprint string."""
    blueprint, diagnostics = emit_blueprint(ir_operations, label, signal_type_map)
    
    try:
        blueprint_string = blueprint.to_string()
        return blueprint_string, diagnostics
    except Exception as e:
        diagnostics.error(f"Blueprint string generation failed: {e}")
        return "", diagnostics



