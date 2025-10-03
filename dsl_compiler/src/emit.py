# emit.py
"""
Blueprint emission module for the Factorio Circuit DSL.

This module converts IR operations into actual Factorio combinators and entities
using the factorio-draftsman library to generate blueprint JSON.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Type, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Add draftsman to path
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent / "factorio-draftsman")
)

try:
    from draftsman.blueprintable import Blueprint
    from draftsman.entity import *  # Import all entities
    from draftsman.classes.entity import Entity
    from draftsman.constants import Direction
    from draftsman.data import entities as entity_data
    from draftsman.data import signals as signal_data
    
    # Try to get entity map for dynamic lookup
    try:
        from draftsman.data.entities import entity_map
    except ImportError:
        entity_map = None

    DRAFTSMAN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: factorio-draftsman not available: {e}")
    DRAFTSMAN_AVAILABLE = False
    entity_map = None

    # Mock classes for development
    class Blueprint:
        pass

    class Entity:
        pass

    class ConstantCombinator:
        pass

    class ArithmeticCombinator:
        pass

    class DeciderCombinator:
        pass


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


# =============================================================================
# Entity Factory using draftsman's catalog
# =============================================================================


class DraftsmanEntityFactory:
    """Simplified factory that lets draftsman handle entity and signal validation."""

    def __init__(self):
        self.all_valid_signals: Set[str] = set()
        self.virtual_signals: Set[str] = set()
        
        if DRAFTSMAN_AVAILABLE:
            self._load_signals()

    def _load_signals(self):
        """Load signal lists from draftsman for validation and implicit mapping."""
        try:
            # Load all signals into one set for validation
            if hasattr(signal_data, 'virtual'):
                self.virtual_signals.update(signal_data.virtual)
                self.all_valid_signals.update(signal_data.virtual)
            if hasattr(signal_data, 'item'):
                self.all_valid_signals.update(signal_data.item)
            if hasattr(signal_data, 'fluid'):
                self.all_valid_signals.update(signal_data.fluid)
            if hasattr(signal_data, 'recipe'):
                self.all_valid_signals.update(signal_data.recipe)
            if hasattr(signal_data, 'entity'):
                self.all_valid_signals.update(signal_data.entity)
                
        except Exception as e:
            # Minimal fallback - just basic virtual signals
            self.virtual_signals.update([
                'signal-0', 'signal-1', 'signal-2', 'signal-3', 'signal-4', 'signal-5',
                'signal-6', 'signal-7', 'signal-8', 'signal-9', 'signal-A', 'signal-B',
                'signal-C', 'signal-D', 'signal-E', 'signal-F', 'signal-G', 'signal-H',
                'signal-I', 'signal-J', 'signal-K', 'signal-L', 'signal-M', 'signal-N',
                'signal-O', 'signal-P', 'signal-Q', 'signal-R', 'signal-S', 'signal-T',
                'signal-U', 'signal-V', 'signal-W', 'signal-X', 'signal-Y', 'signal-Z',
                'signal-each', 'signal-everything', 'signal-anything'
            ])
            self.all_valid_signals.update(self.virtual_signals)

    def create_entity(self, prototype: str, **kwargs) -> Entity:
        """Create an entity using draftsman - let it handle validation."""
        if not DRAFTSMAN_AVAILABLE:
            raise RuntimeError("Draftsman not available")
            
        # Use draftsman's entity_map if available for fast lookup
        if entity_map and prototype in entity_map:
            entity_class = entity_map[prototype]
            return entity_class(**kwargs)
        
        # Fallback: try to create entity by prototype inspection
        # This is a simple heuristic-based approach
        try:
            # Combinators - these don't take prototype names
            if 'combinator' in prototype:
                if 'constant' in prototype:
                    return ConstantCombinator(**kwargs)
                elif 'arithmetic' in prototype:
                    return ArithmeticCombinator(**kwargs)
                elif 'decider' in prototype:
                    return DeciderCombinator(**kwargs)
                elif 'selector' in prototype:
                    return SelectorCombinator(**kwargs)
            
            # Entities that take prototype names as first argument
            # Order matters - more specific matches should come first
            entity_classes_with_prototypes = [
                (Container, ['chest']),
                (AssemblingMachine, ['assembling-machine', 'cryogenic', 'electromagnetic', 'foundry']),
                (Furnace, ['furnace']),
                (MiningDrill, ['mining-drill']),
                (ElectricPole, ['pole', 'substation']),
                (UndergroundBelt, ['underground']),  # Match underground before belt
                (TransportBelt, ['belt']),
                (Inserter, ['inserter']),
                (Splitter, ['splitter']),
            ]
            
            for entity_class, keywords in entity_classes_with_prototypes:
                if any(keyword in prototype for keyword in keywords):
                    try:
                        return entity_class(prototype, **kwargs)
                    except:
                        continue
            
            # Single-prototype entities
            entity_classes_single = [
                (Lamp, ['lamp']),
                (TrainStop, ['train-stop']),
                (StorageTank, ['storage-tank']),
                (Radar, ['radar']),
                (Roboport, ['roboport']),
                (OffshorePump, ['offshore-pump']),
                (Pump, ['pump']),
            ]
            
            for entity_class, keywords in entity_classes_single:
                if any(keyword in prototype for keyword in keywords):
                    try:
                        return entity_class(**kwargs)
                    except:
                        continue
                        
            # If nothing worked, let draftsman throw a proper error
            # by trying a generic container
            return Container(prototype, **kwargs)
            
        except Exception as e:
            raise ValueError(f"Unknown entity prototype: {prototype}") from e

    def is_valid_signal(self, signal_name: str) -> bool:
        """Check if a signal name is valid."""
        return signal_name in self.all_valid_signals

    def is_virtual_signal(self, signal_name: str) -> bool:
        """Check if a signal is virtual (needed for implicit type mapping)."""
        return signal_name in self.virtual_signals


@dataclass
class EntityPlacement:
    """Information about placed entity for wiring."""

    entity: Entity  # Draftsman entity
    entity_id: str
    position: Tuple[int, int]
    output_signals: Dict[str, str]  # signal_type -> wire_color
    input_signals: Dict[str, str]  # signal_type -> wire_color


# =============================================================================
# Memory Circuit Builder
# =============================================================================


class MemoryCircuitBuilder:
    """Builds proper memory circuits based on real Factorio designs."""
    
    def __init__(self, layout_engine, blueprint: Blueprint, entity_factory: DraftsmanEntityFactory):
        self.layout = layout_engine
        self.blueprint = blueprint
        self.factory = entity_factory
        self.memory_modules: Dict[str, Dict[str, EntityPlacement]] = {}

    def build_sr_latch(self, memory_id: str, signal_type: str, initial_value: int = 0) -> Dict[str, EntityPlacement]:
        """
        Build a 3-combinator memory cell that handles negatives and burst signals.
        
        Based on the design from: https://forums.factorio.com/viewtopic.php?f=193&t=60330
        
        This creates a robust memory cell with:
        - Input combinator (detects input and generates reset signal)
        - Output combinator (handles output during input)  
        - Memory combinator (stores the value when no input)
        """
        from draftsman.entity import DeciderCombinator
        
        placements = {}
        
        # Combinator 1: Input detector and reset generator
        # If I != 0, output R = 1 (reset signal)
        input_pos = self.layout.get_next_position()
        input_combinator = DeciderCombinator(tile_position=input_pos)
        
        # Set condition: I != 0
        input_condition = DeciderCombinator.Condition(
            first_signal=signal_type,
            comparator="!=",
            constant=0
        )
        input_combinator.conditions = [input_condition]
        
        # Output: R = 1 (constant)
        input_output = DeciderCombinator.Output(
            signal="signal-R",
            copy_count_from_input=False,
            constant=1
        )
        input_combinator.outputs.append(input_output)
        
        self.blueprint.entities.append(input_combinator)
        
        # Combinator 2: Output handler during input
        # If R > 0, output M = I (copy input value)
        output_pos = self.layout.get_next_position()
        output_combinator = DeciderCombinator(tile_position=output_pos)
        
        # Set condition: R > 0
        output_condition = DeciderCombinator.Condition(
            first_signal="signal-R",
            comparator=">",
            constant=0
        )
        output_combinator.conditions = [output_condition]
        
        # Output: M = I (copy input signal)
        output_output = DeciderCombinator.Output(
            signal="signal-M",
            copy_count_from_input=True  # Copy the I signal value
        )
        output_combinator.outputs.append(output_output)
        
        self.blueprint.entities.append(output_combinator)
        
        # Combinator 3: Memory storage
        # If R = 0, output M = M (memory feedback)
        memory_pos = self.layout.get_next_position()
        memory_combinator = DeciderCombinator(tile_position=memory_pos)
        
        # Set condition: R = 0
        memory_condition = DeciderCombinator.Condition(
            first_signal="signal-R",
            comparator="=",
            constant=0
        )
        memory_combinator.conditions = [memory_condition]
        
        # Output: M = M (copy memory signal)
        memory_output = DeciderCombinator.Output(
            signal="signal-M",
            copy_count_from_input=True  # Copy the M signal value from input
        )
        memory_combinator.outputs.append(memory_output)
        
        # Set initial value if specified
        if initial_value != 0:
            # Use a constant combinator to initialize the memory
            from draftsman.entity import ConstantCombinator
            init_pos = self.layout.get_next_position()
            init_combinator = ConstantCombinator(tile_position=init_pos)
            section = init_combinator.add_section()
            section.set_signal(index=0, signal="signal-M", count=initial_value)
            self.blueprint.entities.append(init_combinator)
            
            placements['init_combinator'] = EntityPlacement(
                entity=init_combinator,
                entity_id=f"{memory_id}_init",
                position=init_pos,
                output_signals={"signal-M": "red"},
                input_signals={}
            )
        
        self.blueprint.entities.append(memory_combinator)
        
        # Store placements
        placements['input_combinator'] = EntityPlacement(
            entity=input_combinator,
            entity_id=f"{memory_id}_input",
            position=input_pos,
            output_signals={"signal-R": "red"},
            input_signals={signal_type: "green"}  # Input from external source
        )
        
        placements['output_combinator'] = EntityPlacement(
            entity=output_combinator,
            entity_id=f"{memory_id}_output",
            position=output_pos,
            output_signals={"signal-M": "green"},
            input_signals={"signal-R": "red", signal_type: "red"}  # Connected to input combinator output
        )
        
        placements['memory_combinator'] = EntityPlacement(
            entity=memory_combinator,
            entity_id=f"{memory_id}_memory",
            position=memory_pos,
            output_signals={"signal-M": "green"},
            input_signals={"signal-R": "red", "signal-M": "green"}  # Reset from input, feedback from self
        )
        
        # Store the memory module for later wiring
        self.memory_modules[memory_id] = placements
        return placements

    def wire_sr_latch(self, memory_id: str):
        """Wire up the 3-combinator memory cell components."""
        if memory_id not in self.memory_modules:
            return
            
        placements = self.memory_modules[memory_id]
        
        try:
            # Get the entities from placements
            input_comb = placements['input_combinator'].entity
            output_comb = placements['output_combinator'].entity
            memory_comb = placements['memory_combinator'].entity
            
            # Verify all entities exist in the blueprint before wiring
            if input_comb not in self.blueprint.entities:
                print(f"Warning: Input combinator for {memory_id} not in blueprint")
                return
            if output_comb not in self.blueprint.entities:
                print(f"Warning: Output combinator for {memory_id} not in blueprint")
                return
            if memory_comb not in self.blueprint.entities:
                print(f"Warning: Memory combinator for {memory_id} not in blueprint")
                return
            
            # Wire 1: Input combinator output (red) -> Output combinator input (red)
            # This passes the R signal to the output combinator
            self.blueprint.add_circuit_connection("red", input_comb, output_comb, 
                                                 side_1="output", side_2="input")
            
            # Wire 2: Input combinator output (red) -> Memory combinator input (red)  
            # This passes the R (reset) signal to memory combinator
            self.blueprint.add_circuit_connection("red", input_comb, memory_comb, 
                                                 side_1="output", side_2="input")
            
            # Wire 3: Output combinator output (green) -> Memory combinator input (green)
            # This provides the new M value during input phase
            self.blueprint.add_circuit_connection("green", output_comb, memory_comb, 
                                                 side_1="output", side_2="input")
            
            # Wire 4: Memory combinator output (green) -> Memory combinator input (green)
            # This creates the memory feedback loop for storage
            self.blueprint.add_circuit_connection("green", memory_comb, memory_comb, 
                                                 side_1="output", side_2="input")
            
            # If there's an init combinator, wire it to the memory
            if 'init_combinator' in placements:
                init_comb = placements['init_combinator'].entity
                if init_comb in self.blueprint.entities:
                    self.blueprint.add_circuit_connection("red", init_comb, memory_comb, 
                                                         side_1="output", side_2="input")
                else:
                    print(f"Warning: Init combinator for {memory_id} not in blueprint")
            
        except Exception as e:
            # Fallback if wiring fails - the individual combinators will still work partially
            print(f"Warning: Could not wire memory cell {memory_id}: {e}")
            pass


class LayoutEngine:
    """Simple layout engine for entity placement."""

    def __init__(self):
        # Start automatic layout in negative coordinates to avoid manual entities
        self.next_x = -30  # Start well to the left of manual entities
        self.next_y = 0
        self.row_height = 4  # Increased spacing to avoid overlaps
        self.entities_per_row = 8
        self.current_row_count = 0
        self.used_positions = set()
        self.entity_spacing = 4  # Minimum spacing between entities

    def get_next_position(self) -> Tuple[int, int]:
        """Get next available position for entity placement with proper grid alignment."""
        # Ensure current position is grid-aligned (integer coordinates)
        pos = (int(self.next_x), int(self.next_y))
        
        # Check if position is available
        while pos in self.used_positions:
            self._advance_position()
            pos = (int(self.next_x), int(self.next_y))
        
        # Mark position as used
        self.used_positions.add(pos)
        
        # Advance to next position for the next call
        self._advance_position()
        
        return pos

    def _advance_position(self):
        """Advance to the next grid position."""
        self.current_row_count += 1
        if self.current_row_count >= self.entities_per_row:
            # Move to next row
            self.next_x = -30  # Reset to left edge for automatic layout
            self.next_y += self.row_height
            self.current_row_count = 0
        else:
            # Move to next column with proper spacing
            self.next_x += self.entity_spacing


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
        self.entity_factory = DraftsmanEntityFactory()
        self.memory_builder = MemoryCircuitBuilder(self.layout, self.blueprint, self.entity_factory)

        # Signal type mapping from IR builder
        self.signal_type_map = signal_type_map or {}

        # Entity tracking
        self.entities: Dict[str, EntityPlacement] = {}
        self.next_entity_number = 1

        # Signal tracking for wiring
        self.signal_sources: Dict[str, str] = {}  # signal_id -> entity_id
        self.signal_sinks: Dict[str, List[str]] = {}  # signal_id -> [entity_ids]
        
        # Bundle tracking
        self.bundle_components: Dict[str, Dict[str, str]] = {}  # bundle_id -> {signal_type -> entity_id}

    def emit_blueprint(self, ir_operations: List[IRNode]) -> Blueprint:
        """Convert IR operations to blueprint."""
        import warnings
        
        # Collect draftsman warnings during blueprint construction
        captured_warnings = []
        
        def warning_handler(message, category, filename, lineno, file=None, line=None):
            captured_warnings.append((message, category, filename, lineno))
        
        # Set up warning capture
        old_showwarning = warnings.showwarning
        warnings.showwarning = warning_handler
        
        try:
            # Process all IR operations
            for op in ir_operations:
                self.emit_ir_operation(op)

            # Add wiring connections
            self.create_circuit_connections()
            
            # Validate the final blueprint and process captured warnings
            self._validate_blueprint_with_warnings(captured_warnings)

            return self.blueprint

        except Exception as e:
            self.diagnostics.error(f"Blueprint emission failed: {e}")
            raise
        finally:
            # Restore original warning handler
            warnings.showwarning = old_showwarning

    def _validate_blueprint_with_warnings(self, captured_warnings):
        """Perform validation on the generated blueprint and process draftsman warnings."""
        
        # Process captured draftsman warnings
        critical_warning_types = {
            'OverlappingObjectsWarning',
            'UnknownEntityWarning', 
            'InvalidEntityError',
            'InvalidConnectorError'
        }
        
        for message, category, filename, lineno in captured_warnings:
            warning_name = category.__name__
            
            # Convert critical warnings to errors
            if warning_name in critical_warning_types:
                self.diagnostics.error(f"Blueprint validation failed - {warning_name}: {message}")
            else:
                # Keep other warnings as warnings
                self.diagnostics.warning(f"{warning_name}: {message}")
        
        # Basic blueprint validation
        if len(self.blueprint.entities) == 0:
            self.diagnostics.warning("Generated blueprint is empty")
            
        # Check for entities with invalid positions
        for entity in self.blueprint.entities:
            if hasattr(entity, 'tile_position'):
                x, y = entity.tile_position
                if x < -1000 or x > 1000 or y < -1000 or y > 1000:
                    self.diagnostics.warning(f"Entity at extreme position: ({x}, {y})")
        
        # Count entity types for debugging
        entity_counts = {}
        for entity in self.blueprint.entities:
            entity_type = type(entity).__name__
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        self.diagnostics.info(f"Generated blueprint with {len(self.blueprint.entities)} entities: {entity_counts}")

    def emit_ir_operation(self, op: IRNode):
        """Emit a single IR operation."""
        # Dispatch table for IR operation emission
        operation_handlers = {
            IR_Const: self.emit_constant,
            IR_Arith: self.emit_arithmetic,
            IR_Decider: self.emit_decider,
            IR_MemCreate: self.emit_memory_create,
            IR_MemRead: self.emit_memory_read,
            IR_MemWrite: self.emit_memory_write,
            IR_PlaceEntity: self.emit_place_entity,
            IR_EntityPropWrite: self.emit_entity_prop_write,
            IR_EntityPropRead: self.emit_entity_prop_read,
            IR_Bundle: self.emit_bundle,
        }

        handler = operation_handlers.get(type(op))
        if handler:
            handler(op)
        else:
            self.diagnostics.error(f"Unknown IR operation: {type(op)}")

    def emit_constant(self, op: IR_Const):
        """Emit constant combinator for IR_Const."""
        pos = self.layout.get_next_position()

        combinator = self.entity_factory.create_entity('constant-combinator', tile_position=pos)

        # Set the constant signal
        section = combinator.add_section()
        signal_name = self._get_signal_name(op.output_type)
        
        # Validate signal name
        if not self.entity_factory.is_valid_signal(signal_name):
            self.diagnostics.warning(f"Unknown signal type: {signal_name}, using signal-0")
            signal_name = "signal-0"
            
        section.set_signal(index=0, name=signal_name, count=op.value)

        self.blueprint.entities.append(combinator)

        # Track entity for wiring
        placement = EntityPlacement(
            entity=combinator,
            entity_id=op.node_id,
            position=pos,
            output_signals={signal_name: "red"},  # Output on red wire
            input_signals={},
        )
        self.entities[op.node_id] = placement
        self.signal_sources[op.node_id] = op.node_id

    def emit_arithmetic(self, op: IR_Arith):
        """Emit arithmetic combinator for IR_Arith."""
        pos = self.layout.get_next_position()

        combinator = self.entity_factory.create_entity('arithmetic-combinator', tile_position=pos)

        # Configure arithmetic operation with proper signal handling
        left_operand = self._get_operand_for_combinator(op.left)
        right_operand = self._get_operand_for_combinator(op.right)
        output_signal = self._get_signal_name(op.output_type)
        
        # Validate output signal
        if not self.entity_factory.is_valid_signal(output_signal):
            self.diagnostics.warning(f"Unknown output signal: {output_signal}, using signal-0")
            output_signal = "signal-0"

        combinator.first_operand = left_operand
        combinator.second_operand = right_operand
        combinator.operation = op.op
        combinator.output_signal = output_signal

        self.blueprint.entities.append(combinator)

        # Track entity for wiring
        placement = EntityPlacement(
            entity=combinator,
            entity_id=op.node_id,
            position=pos,
            output_signals={output_signal: "red"},
            input_signals={},  # Will be populated when wiring
        )
        self.entities[op.node_id] = placement
        self.signal_sources[op.node_id] = op.node_id

        # Track signal dependencies
        self._add_signal_sink(op.left, op.node_id)
        self._add_signal_sink(op.right, op.node_id)

    def emit_decider(self, op: IR_Decider):
        """Emit decider combinator for IR_Decider."""
        pos = self.layout.get_next_position()

        combinator = self.entity_factory.create_entity('decider-combinator', tile_position=pos)

        # Configure decider operation
        left_operand = self._get_operand_for_combinator(op.left)
        right_operand = self._get_operand_value(op.right)
        output_signal = self._get_signal_name(op.output_type)
        
        # Validate output signal
        if not self.entity_factory.is_valid_signal(output_signal):
            self.diagnostics.warning(f"Unknown output signal: {output_signal}, using signal-0")
            output_signal = "signal-0"

        combinator.first_operand = left_operand
        combinator.second_operand = right_operand
        combinator.comparator = op.test_op
        combinator.output_signal = output_signal
        combinator.copy_count_from_input = op.output_value == "input"
        if not combinator.copy_count_from_input:
            combinator.constant = op.output_value if isinstance(op.output_value, int) else 1

        self.blueprint.entities.append(combinator)

        # Track entity for wiring
        placement = EntityPlacement(
            entity=combinator,
            entity_id=op.node_id,
            position=pos,
            output_signals={output_signal: "red"},
            input_signals={},
        )
        self.entities[op.node_id] = placement
        self.signal_sources[op.node_id] = op.node_id

        # Track signal dependencies
        self._add_signal_sink(op.left, op.node_id)
        if not isinstance(op.right, int):  # Signal reference
            self._add_signal_sink(op.right, op.node_id)

    def emit_memory_create(self, op: IR_MemCreate):
        """Emit memory module creation using SR latch circuit."""
        # Get initial value
        initial_value = 0
        if isinstance(op.initial_value, int):
            initial_value = op.initial_value
        elif hasattr(op.initial_value, 'value'):
            initial_value = getattr(op.initial_value, 'value', 0)
        
        # Validate and resolve signal type
        signal_type = self._get_signal_name(op.signal_type)
        if not self.entity_factory.is_valid_signal(signal_type):
            self.diagnostics.warning(f"Unknown signal type for memory: {op.signal_type} -> {signal_type}, using signal-0")
            signal_type = "signal-0"

        # Build the SR latch circuit
        memory_components = self.memory_builder.build_sr_latch(
            op.memory_id, signal_type, initial_value
        )
        
        # Register all components as entities
        for component_name, placement in memory_components.items():
            self.entities[placement.entity_id] = placement
        
        # The output combinator is the main interface for this memory
        if 'output_combinator' in memory_components:
            output_placement = memory_components['output_combinator']
            self.signal_sources[op.memory_id] = output_placement.entity_id

    def emit_memory_read(self, op: IR_MemRead):
        """Emit memory read operation from the 3-combinator memory cell."""
        # Memory read connects to the output of the memory combinator
        if op.memory_id in self.memory_builder.memory_modules:
            # Get the memory combinator from the memory module (the one that outputs M signal)
            memory_components = self.memory_builder.memory_modules[op.memory_id]
            if 'memory_combinator' in memory_components:
                memory_entity_placement = memory_components['memory_combinator']
                memory_entity_id = memory_entity_placement.entity_id
                
                # The memory cell outputs on the "signal-M" channel
                # Create a signal mapping for the read operation
                self.signal_sources[op.node_id] = memory_entity_id
                
                # Track that this read operation expects the "signal-M" signal
                # This will be used when wiring up connections
                if not hasattr(self, 'memory_read_signals'):
                    self.memory_read_signals = {}
                self.memory_read_signals[op.node_id] = "signal-M"
                
            else:
                self.diagnostics.error(f"Memory combinator not found in {op.memory_id}")
        else:
            self.diagnostics.error(f"Memory {op.memory_id} not found for read operation")

    def emit_memory_write(self, op: IR_MemWrite):
        """Emit memory write operation to the 3-combinator memory cell."""
        memory_id = op.memory_id
        
        if memory_id not in self.memory_builder.memory_modules:
            self.diagnostics.error(f"Memory {memory_id} not found for write operation")
            return
        
        # Get the memory module components
        memory_module = self.memory_builder.memory_modules[memory_id]
        input_combinator_placement = memory_module['input_combinator']
        
        # Create a combinator to convert the write data and enable signal to the memory input format
        from draftsman.entity import ArithmeticCombinator
        write_pos = self.layout.get_next_position()
        write_combinator = ArithmeticCombinator(tile_position=write_pos)
        
        try:
            # Get the signal information from the write operation
            signal_type = self._get_signal_name(op.data_signal)
            if not self.entity_factory.is_valid_signal(signal_type):
                self.diagnostics.warning(f"Unknown signal type for memory write: {op.data_signal} -> {signal_type}, using signal-0")
                signal_type = "signal-0"
            
            # Configure the write combinator to pass through the data when write is enabled
            # This converts the data_signal to the I signal expected by the memory cell
            write_combinator.first_operand = signal_type
            write_combinator.second_operand = 1  # Constant 1
            write_combinator.operation = "*"  # Multiply by 1 to pass through
            write_combinator.output_signal = signal_type  # Output same signal type
            
            self.blueprint.entities.append(write_combinator)
            
            # Wire the write combinator output to the memory cell input
            try:
                input_combinator = input_combinator_placement.entity
                self.blueprint.add_circuit_connection("green", write_combinator, input_combinator,
                                                     side_1="output", side_2="input")
                
                self.diagnostics.info(f"Memory write operation properly connected for {memory_id}")
                
                # If there's a write_enable signal, we could add additional logic here
                # For now, the write happens whenever data_signal is non-zero
                
            except Exception as e:
                self.diagnostics.warning(f"Could not wire memory write combinator for {memory_id}: {e}")
                
        except Exception as e:
            self.diagnostics.warning(f"Could not configure memory write combinator for {memory_id}: {e}")
            # Fallback to logging for compatibility
            self.diagnostics.info(f"Memory write operation for {memory_id} (fallback to simplified logging)")

    def emit_place_entity(self, op: IR_PlaceEntity):
        """Emit entity placement using the entity factory."""
        # Handle both constant and variable coordinates
        if isinstance(op.x, int) and isinstance(op.y, int):
            # Use specified position with offset to avoid combinators
            base_x = op.x + 20  # Offset entities to the right of combinators
            base_y = op.y
            pos = (int(base_x), int(base_y))  # Ensure grid alignment
            
            # Mark this position as used in the layout manager to prevent overlaps
            self.layout.used_positions.add(pos)
        else:
            # Default to layout engine for variable coordinates or fallback
            pos = self.layout.get_next_position()

        try:
            entity = self.entity_factory.create_entity(op.prototype, tile_position=pos)
            
            # Apply any additional properties
            if op.properties:
                for prop_name, prop_value in op.properties.items():
                    if hasattr(entity, prop_name):
                        setattr(entity, prop_name, prop_value)
                    else:
                        self.diagnostics.error(f"Unknown property '{prop_name}' for entity '{op.prototype}'")

            self.blueprint.entities.append(entity)

            placement = EntityPlacement(
                entity=entity,
                entity_id=op.entity_id,
                position=pos,
                output_signals={},
                input_signals={},
            )
            self.entities[op.entity_id] = placement
            
        except ValueError as e:
            self.diagnostics.error(f"Failed to create entity: {e}")
        except Exception as e:
            self.diagnostics.error(f"Unexpected error creating entity '{op.prototype}': {e}")

    def emit_entity_prop_write(self, op: IR_EntityPropWrite):
        """Emit entity property write (circuit network connection)."""
        # This would connect signals to entity properties
        if op.entity_id in self.entities:
            entity_placement = self.entities[op.entity_id]
            # Track that this entity needs this signal as input
            entity_placement.input_signals[f"{op.property_name}_input"] = "green"
            self._add_signal_sink(op.value, op.entity_id)

    def emit_entity_prop_read(self, op: IR_EntityPropRead):
        """Emit entity property read (circuit network connection)."""
        # This would read entity properties and output them as signals
        if op.entity_id in self.entities:
            entity_placement = self.entities[op.entity_id]
            # Track that this entity outputs this signal
            entity_placement.output_signals[f"{op.property_name}_output"] = "red"
            # Mark this operation as a signal source using the existing pattern
            self.signal_sources[op.node_id] = op.entity_id

    def emit_bundle(self, op: IR_Bundle):
        """Emit bundle operation (multiple signals on same wire)."""
        # Bundles combine multiple signals onto the same wire network
        # We track the component signals for proper wiring
        
        bundle_components = {}
        
        for signal_type, signal_ref in op.inputs.items():
            if isinstance(signal_ref, SignalRef):
                # Track signal source for wiring
                bundle_components[signal_type] = signal_ref.source_id
                self.signal_sources[f"{op.node_id}_{signal_type}"] = signal_ref.source_id
            elif isinstance(signal_ref, str):
                # String reference to existing signal
                bundle_components[signal_type] = signal_ref
                self.signal_sources[f"{op.node_id}_{signal_type}"] = signal_ref
        
        # Store bundle composition for later use
        self.bundle_components[op.node_id] = bundle_components
        
        # For now, bundles don't create physical entities - they're just wire groupings
        # The bundle is represented by its component signals

    def create_circuit_connections(self):
        """Create circuit wire connections between entities."""
        # First, wire up all memory circuits
        for memory_id in self.memory_builder.memory_modules:
            try:
                self.memory_builder.wire_sr_latch(memory_id)
            except Exception as e:
                self.diagnostics.error(f"Failed to wire memory {memory_id}: {e}")

        # Create connections between signal sources and sinks
        for signal_id, sink_entities in self.signal_sinks.items():
            if signal_id in self.signal_sources:
                source_entity_id = self.signal_sources[signal_id]

                if source_entity_id in self.entities:
                    source_placement = self.entities[source_entity_id]

                    for sink_entity_id in sink_entities:
                        if sink_entity_id in self.entities:
                            sink_placement = self.entities[sink_entity_id]

                            # Determine wire color based on signal type
                            wire_color = self._get_wire_color(source_placement, sink_placement)

                            # Add circuit connection
                            try:
                                self.blueprint.add_circuit_connection(
                                    color=wire_color,
                                    entity_1=source_placement.entity,
                                    entity_2=sink_placement.entity,
                                    side_1="output",
                                    side_2="input",
                                )
                            except Exception as e:
                                self.diagnostics.error(
                                    f"Failed to connect {source_entity_id} -> {sink_entity_id}: {e}"
                                )
                else:
                    self.diagnostics.error(f"Source entity {source_entity_id} not found for signal {signal_id}")
            else:
                self.diagnostics.error(f"No source found for signal {signal_id}")

    def _get_wire_color(self, source: EntityPlacement, sink: EntityPlacement) -> str:
        """Determine appropriate wire color for connection."""
        # Use red as default, green for memory outputs to avoid conflicts
        if "memory" in source.entity_id and "output" in source.entity_id:
            return "green"
        return "red"

    def _get_signal_name(self, operand) -> str:
        """Get signal name from operand - much simpler approach."""
        # Handle SignalRef objects
        if hasattr(operand, 'signal_type'):
            operand_str = operand.signal_type
        else:
            operand_str = str(operand)

        # Clean up signal reference strings
        clean_name = operand_str.split("@")[0]  # Remove anything after @

        # Handle integer constants
        if isinstance(operand, int):
            return "signal-0"

        # Handle bundle references
        if clean_name.startswith("Bundle["):
            return "signal-each"  # Special signal for bundle operations

        # Check signal mapping first
        if clean_name in self.signal_type_map:
            mapped_signal = self.signal_type_map[clean_name]
            if self.entity_factory.is_valid_signal(mapped_signal):
                return mapped_signal

        # If it's already a valid signal, use it
        if self.entity_factory.is_valid_signal(clean_name):
            return clean_name

        # Handle implicit signals (__v1, __v2, etc.) by mapping to virtual signals
        if clean_name.startswith("__v"):
            try:
                num = int(clean_name[3:])
                # Map to signal-A through signal-Z, cycling if needed
                letter_index = (num - 1) % 26
                virtual_signal = f"signal-{chr(ord('A') + letter_index)}"
                return virtual_signal
            except ValueError:
                pass

        # Fallback to a safe virtual signal
        return "signal-0"

    def _get_operand_for_combinator(self, operand) -> Union[str, int]:
        """Get operand for arithmetic/decider combinator (can be signal name or constant)."""
        if isinstance(operand, int):
            return operand
        elif isinstance(operand, SignalRef):
            return self._get_signal_name(operand.signal_type)
        elif isinstance(operand, str):
            return self._get_signal_name(operand)
        else:
            return self._get_signal_name(str(operand))

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


def emit_blueprint(
    ir_operations: List[IRNode],
    label: str = "DSL Generated",
    signal_type_map: Dict[str, str] = None,
) -> Tuple[Blueprint, DiagnosticCollector]:
    """Convert IR operations to Factorio blueprint."""
    emitter = BlueprintEmitter(signal_type_map)
    emitter.blueprint.label = label

    try:
        blueprint = emitter.emit_blueprint(ir_operations)
        return blueprint, emitter.diagnostics
    except Exception as e:
        emitter.diagnostics.error(f"Blueprint emission failed: {e}")
        return emitter.blueprint, emitter.diagnostics


def emit_blueprint_string(
    ir_operations: List[IRNode],
    label: str = "DSL Generated",
    signal_type_map: Dict[str, str] = None,
) -> Tuple[str, DiagnosticCollector]:
    """Convert IR operations to Factorio blueprint string."""
    blueprint, diagnostics = emit_blueprint(ir_operations, label, signal_type_map)

    try:
        blueprint_string = blueprint.to_string()
        return blueprint_string, diagnostics
    except Exception as e:
        diagnostics.error(f"Blueprint string generation failed: {e}")
        return "", diagnostics
