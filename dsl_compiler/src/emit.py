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

from draftsman.blueprintable import Blueprint
from draftsman.entity import new_entity  # Use draftsman's factory
from draftsman.entity import *  # Import all entities
from draftsman.classes.entity import Entity
from draftsman.constants import Direction
from draftsman.data import entities as entity_data
from draftsman.data import signals as signal_data
from draftsman.signatures import SignalID


# Internal compiler imports
from .ir import *
from .semantic import DiagnosticCollector


# =============================================================================
# Entity Factory using draftsman's catalog
# =============================================================================


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
    
    def __init__(self, layout_engine, blueprint: Blueprint):
        self.layout = layout_engine
        self.blueprint = blueprint
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
        from draftsman.entity import DeciderCombinator, ArithmeticCombinator
        
        placements = {}
        
        # Combinator 1: Input detector and reset generator
        # If I != 0, output R = 1 (reset signal)
        input_pos = self.layout.get_next_position()
        input_combinator = DeciderCombinator(tile_position=input_pos)
        
        # Set condition: write pulse detected
        input_condition = DeciderCombinator.Condition(
            first_signal="signal-R",
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
        
        self.blueprint.entities.append(input_combinator, copy=False)
        
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
            copy_count_from_input=True
        )
        output_combinator.outputs.append(output_output)
        
        self.blueprint.entities.append(output_combinator, copy=False)
        
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
            copy_count_from_input=True
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
            self.blueprint.entities.append(init_combinator, copy=False)
            
            placements['init_combinator'] = EntityPlacement(
                entity=init_combinator,
                entity_id=f"{memory_id}_init",
                position=init_pos,
                output_signals={"signal-M": "red"},
                input_signals={}
            )
        
        self.blueprint.entities.append(memory_combinator, copy=False)
        
        # Converter: translate internal signal-M to the memory's declared signal type
        converter_pos = self.layout.get_next_position()
        converter = ArithmeticCombinator(tile_position=converter_pos)
        converter.first_operand = "signal-M"
        converter.second_operand = 1
        converter.operation = "*"
        converter.output_signal = signal_type
        self.blueprint.entities.append(converter, copy=False)

        # Store placements
        placements['input_combinator'] = EntityPlacement(
            entity=input_combinator,
            entity_id=f"{memory_id}_input",
            position=input_pos,
            output_signals={"signal-R": "red"},
            input_signals={signal_type: "green", "signal-R": "red"}
        )
        
        placements['output_combinator'] = EntityPlacement(
            entity=output_combinator,
            entity_id=f"{memory_id}_output",
            position=output_pos,
            output_signals={"signal-M": "green"},
            input_signals={"signal-R": "red", signal_type: "red"}
        )
        
        placements['memory_combinator'] = EntityPlacement(
            entity=memory_combinator,
            entity_id=f"{memory_id}_memory",
            position=memory_pos,
            output_signals={"signal-M": "green"},
            input_signals={"signal-R": "red", "signal-M": "green"}
        )

        placements['output_converter'] = EntityPlacement(
            entity=converter,
            entity_id=f"{memory_id}_converter",
            position=converter_pos,
            output_signals={signal_type: "red"},
            input_signals={"signal-M": "green"}
        )
        
        # Store the memory module for later wiring
        self.memory_modules[memory_id] = placements
        return placements

    def wire_sr_latch(self, memory_id: str):
        """Wire up the 3-combinator memory cell components."""
        if memory_id not in self.memory_modules:
            return

        module = self.memory_modules[memory_id]
        input_comb = module.get('input_combinator')
        output_comb = module.get('output_combinator')
        memory_comb = module.get('memory_combinator')
        converter = module.get('output_converter')
        init_comb = module.get('init_combinator')

        try:
            if input_comb and output_comb:
                self.blueprint.add_circuit_connection(
                    "red",
                    input_comb.entity,
                    output_comb.entity,
                    side_1="output",
                    side_2="input",
                )

            if input_comb and memory_comb:
                self.blueprint.add_circuit_connection(
                    "red",
                    input_comb.entity,
                    memory_comb.entity,
                    side_1="output",
                    side_2="input",
                )

            if memory_comb:
                # Feedback loop for memory retention
                self.blueprint.add_circuit_connection(
                    "green",
                    memory_comb.entity,
                    memory_comb.entity,
                    side_1="output",
                    side_2="input",
                )

            if output_comb and memory_comb:
                # Allow new writes to reach memory storage
                self.blueprint.add_circuit_connection(
                    "green",
                    output_comb.entity,
                    memory_comb.entity,
                    side_1="output",
                    side_2="input",
                )

            if converter and memory_comb:
                self.blueprint.add_circuit_connection(
                    "green",
                    memory_comb.entity,
                    converter.entity,
                    side_1="output",
                    side_2="input",
                )

            if converter and output_comb:
                self.blueprint.add_circuit_connection(
                    "green",
                    output_comb.entity,
                    converter.entity,
                    side_1="output",
                    side_2="input",
                )

            if init_comb and memory_comb:
                self.blueprint.add_circuit_connection(
                    "green",
                    init_comb.entity,
                    memory_comb.entity,
                    side_1="output",
                    side_2="input",
                )

            if init_comb and converter:
                self.blueprint.add_circuit_connection(
                    "green",
                    init_comb.entity,
                    converter.entity,
                    side_1="output",
                    side_2="input",
                )
        except Exception as exc:
            print(f"Warning: failed to wire memory cell {memory_id}: {exc}")


class LayoutEngine:
    """Simple layout engine for entity placement."""

    def __init__(self):
        # Start automatic layout in negative coordinates to avoid manual entities
        self.next_x = -30  # Start well to the left of manual entities
        self.next_y = 0
        self.row_height = 2  # Keep rows close enough for wiring while avoiding overlap
        self.entities_per_row = 6
        self.current_row_count = 0
        self.used_positions = set()
        self.entity_spacing = 2  # Minimum spacing between entities

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

    def reserve_near(self, target: Tuple[int, int], max_radius: int = 6) -> Tuple[int, int]:
        """Reserve a position near a target coordinate if available."""
        tx, ty = self.snap_to_grid(target)

        if (tx, ty) not in self.used_positions:
            self.used_positions.add((tx, ty))
            return (tx, ty)

        spacing_x = max(1, self.entity_spacing)
        spacing_y = max(1, self.row_height)

        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    pos = (tx + dx * spacing_x, ty + dy * spacing_y)
                    if pos not in self.used_positions:
                        self.used_positions.add(pos)
                        return pos

        # Fall back to default layout if nearby space is exhausted
        return self.get_next_position()

    def snap_to_grid(self, pos: Tuple[Union[int, float], Union[int, float]]) -> Tuple[int, int]:
        """Snap a coordinate to the layout grid respecting spacing and row height."""
        x, y = pos
        spacing_x = max(1, self.entity_spacing)
        spacing_y = max(1, self.row_height)

        snapped_x = int(round(x / spacing_x) * spacing_x)
        snapped_y = int(round(y / spacing_y) * spacing_y)

        return (snapped_x, snapped_y)

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
        self.blueprint = Blueprint()
        self.blueprint.label = "DSL Generated Blueprint"
        self.blueprint.version = (2, 0)

        self.layout = LayoutEngine()
        self.diagnostics = DiagnosticCollector()
        self.memory_builder = MemoryCircuitBuilder(self.layout, self.blueprint)

        # Signal type mapping from IR builder
        self.signal_type_map = signal_type_map or {}

        # Entity tracking
        self.entities: Dict[str, EntityPlacement] = {}
        self.next_entity_number = 1

        # Signal tracking for wiring
        self.signal_sources: Dict[str, str] = {}  # signal_id -> entity_id
        self.signal_sinks: Dict[str, List[str]] = {}  # signal_id -> [entity_ids]
        
        # Entity property signal tracking (for reads)
        self.entity_property_signals: Dict[Tuple[str, str], Any] = {}

    def _extract_dependency_positions(self, value_ref) -> List[Tuple[int, int]]:
        """Find positions of entities that feed a value reference."""
        positions: List[Tuple[int, int]] = []

        if isinstance(value_ref, SignalRef):
            placement = self.entities.get(value_ref.source_id)
            if placement:
                positions.append(placement.position)
        elif isinstance(value_ref, str):
            placement = self.entities.get(value_ref)
            if placement:
                positions.append(placement.position)

        return positions

    def _allocate_position(self, *value_refs) -> Tuple[int, int]:
        """Allocate a placement near the average of dependency positions if possible."""
        positions: List[Tuple[int, int]] = []
        for ref in value_refs:
            positions.extend(self._extract_dependency_positions(ref))

        if positions:
            avg_x = sum(pos[0] for pos in positions) / len(positions)
            avg_y = sum(pos[1] for pos in positions) / len(positions)
            return self.layout.reserve_near((avg_x, avg_y))

        return self.layout.get_next_position()

    def _add_entity(self, entity: Entity) -> Entity:
        """Add an entity to the blueprint and return the stored reference."""
        return self.blueprint.entities.append(entity, copy=False)

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
        }

        handler = operation_handlers.get(type(op))
        if handler:
            handler(op)
        else:
            self.diagnostics.error(f"Unknown IR operation: {type(op)}")

    def emit_constant(self, op: IR_Const):
        """Emit constant combinator for IR_Const."""

        pos = self.layout.get_next_position()

        combinator = new_entity('constant-combinator', tile_position=pos)

        # Set the constant signal
        section = combinator.add_section()
        signal_name = self._get_signal_name(op.output_type)
        
        section.set_signal(index=0, name=signal_name, count=op.value)

        combinator = self._add_entity(combinator)

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

        pos = self._allocate_position(op.left, op.right)

        combinator = new_entity('arithmetic-combinator', tile_position=pos)

        # Configure arithmetic operation with proper signal handling
        left_operand = self._get_operand_for_combinator(op.left)
        right_operand = self._get_operand_for_combinator(op.right)
        output_signal = self._get_signal_name(op.output_type)

        combinator.first_operand = left_operand
        combinator.second_operand = right_operand
        combinator.operation = op.op
        combinator.output_signal = output_signal

        combinator = self._add_entity(combinator)

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
        pos = self._allocate_position(op.left, op.right)

        combinator = new_entity('decider-combinator', tile_position=pos)

        # Configure decider operation
        left_operand = self._get_operand_for_combinator(op.left)
        right_operand = self._get_operand_value(op.right)
        output_signal = self._get_signal_name(op.output_type)
        
        combinator.first_operand = left_operand
        combinator.second_operand = right_operand
        combinator.comparator = op.test_op
        combinator.output_signal = output_signal
        combinator.copy_count_from_input = op.output_value == "input"
        if not combinator.copy_count_from_input:
            combinator.constant = op.output_value if isinstance(op.output_value, int) else 1

        combinator = self._add_entity(combinator)

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
        elif isinstance(op.initial_value, SignalRef):
            source_id = op.initial_value.source_id
            placement = self.entities.get(source_id)
            if placement and hasattr(placement.entity, "sections"):
                try:
                    sections = placement.entity.sections
                    if sections:
                        signals = sections[0].signals
                        if signals:
                            initial_value = signals[0].get("count", 0)
                except Exception:
                    self.diagnostics.warning(
                        f"Unable to extract initial value for memory {op.memory_id} from source {source_id}; defaulting to 0"
                    )
        elif hasattr(op.initial_value, 'value'):
            initial_value = getattr(op.initial_value, 'value', 0)
        
        # Validate and resolve signal type
        signal_type = self._get_signal_name(op.signal_type)

        # Build the SR latch circuit
        memory_components = self.memory_builder.build_sr_latch(
            op.memory_id, signal_type, initial_value
        )
        memory_components['signal_type'] = signal_type
        
        # Register all components as entities
        for component_name, placement in memory_components.items():
            if isinstance(placement, EntityPlacement):
                self.entities[placement.entity_id] = placement
        
        # The converter is the main interface for this memory
        converter_placement = memory_components.get('output_converter')
        if converter_placement:
            self.signal_sources[op.memory_id] = converter_placement.entity_id

    def emit_memory_read(self, op: IR_MemRead):
        """Emit memory read operation from the 3-combinator memory cell."""
        # Memory read connects to the output of the memory combinator
        if op.memory_id in self.memory_builder.memory_modules:
            memory_components = self.memory_builder.memory_modules[op.memory_id]
            converter = memory_components.get('output_converter')
            if converter:
                self.signal_sources[op.node_id] = converter.entity_id

                if not hasattr(self, 'memory_read_signals'):
                    self.memory_read_signals = {}
                declared_type = memory_components.get('signal_type', op.output_type)
                self.memory_read_signals[op.node_id] = declared_type
            else:
                self.diagnostics.error(f"Memory converter not found in {op.memory_id}")
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
        output_combinator_placement = memory_module['output_combinator']

        from draftsman.entity import ArithmeticCombinator, DeciderCombinator

        try:
            # Data writer: normalizes incoming signal to the memory's declared channel
            data_signal_name = memory_module.get('signal_type', self._get_signal_name(op.data_signal))
            write_pos = self.layout.get_next_position()
            data_combinator = ArithmeticCombinator(tile_position=write_pos)
            data_combinator.first_operand = self._get_operand_for_combinator(op.data_signal)
            data_combinator.second_operand = 1
            data_combinator.operation = "*"
            data_combinator.output_signal = data_signal_name
            data_combinator = self._add_entity(data_combinator)

            data_entity_id = f"{memory_id}_write_data_{self.next_entity_number}"
            self.next_entity_number += 1
            data_placement = EntityPlacement(
                entity=data_combinator,
                entity_id=data_entity_id,
                position=write_pos,
                output_signals={data_signal_name: "green"},
                input_signals={},
            )
            self.entities[data_entity_id] = data_placement
            self.signal_sources[data_entity_id] = data_entity_id

            # Ensure upstream signals connect to the writer
            self._add_signal_sink(op.data_signal, data_entity_id)

            # Connect writer output to memory components
            input_combinator = input_combinator_placement.entity
            output_combinator = output_combinator_placement.entity

            self.blueprint.add_circuit_connection(
                "green",
                data_combinator,
                input_combinator,
                side_1="output",
                side_2="input",
            )
            self.blueprint.add_circuit_connection(
                "red",
                data_combinator,
                output_combinator,
                side_1="output",
                side_2="input",
            )

            # Enable writer: produce signal-R pulse based on write enable expression
            enable_pos = self.layout.get_next_position()
            enable_combinator = DeciderCombinator(tile_position=enable_pos)
            enable_signal = self._get_signal_name(op.write_enable)
            condition = DeciderCombinator.Condition(
                first_signal=enable_signal,
                comparator="!=",
                constant=0,
            )
            enable_combinator.conditions = [condition]

            enable_output = DeciderCombinator.Output(
                signal="signal-R",
                copy_count_from_input=False,
                constant=1,
            )
            enable_combinator.outputs.append(enable_output)
            enable_combinator = self._add_entity(enable_combinator)

            enable_entity_id = f"{memory_id}_write_enable_{self.next_entity_number}"
            self.next_entity_number += 1
            enable_placement = EntityPlacement(
                entity=enable_combinator,
                entity_id=enable_entity_id,
                position=enable_pos,
                output_signals={"signal-R": "red"},
                input_signals={},
            )
            self.entities[enable_entity_id] = enable_placement
            self.signal_sources[enable_entity_id] = enable_entity_id

            self._add_signal_sink(op.write_enable, enable_entity_id)

            self.blueprint.add_circuit_connection(
                "red",
                enable_combinator,
                input_combinator,
                side_1="output",
                side_2="input",
            )

        except Exception as e:
            self.diagnostics.warning(
                f"Could not configure memory write combinator for {memory_id}: {e}"
            )

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
            entity = new_entity(op.prototype, tile_position=pos)
            
            # Apply any additional properties
            if op.properties:
                for prop_name, prop_value in op.properties.items():
                    if hasattr(entity, prop_name):
                        setattr(entity, prop_name, prop_value)
                    else:
                        self.diagnostics.error(f"Unknown property '{prop_name}' for entity '{op.prototype}'")

            entity = self._add_entity(entity)

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
        if op.entity_id not in self.entities:
            self.diagnostics.error(f"Entity {op.entity_id} not found for property write")
            return

        entity_placement = self.entities[op.entity_id]
        entity = entity_placement.entity
        property_key = (op.entity_id, op.property_name)
        self.entity_property_signals[property_key] = op.value

        # Static assignments (ints/strings) are applied directly when possible
        if isinstance(op.value, int):
            if hasattr(entity, op.property_name):
                setattr(entity, op.property_name, op.value)
            else:
                self.diagnostics.warning(
                    f"Entity '{op.entity_id}' has no property '{op.property_name}' for static assignment"
                )
            return

        if isinstance(op.value, str):
            if hasattr(entity, op.property_name):
                setattr(entity, op.property_name, op.value)
            else:
                self.diagnostics.warning(
                    f"Entity '{op.entity_id}' has no property '{op.property_name}' for static assignment"
                )
            return

        signal_name = self._get_signal_name(op.value)
        entity_placement.input_signals[signal_name] = "red"
        self._add_signal_sink(op.value, op.entity_id)

        # Only enable/disable is currently supported dynamically
        if op.property_name == "enable":
            try:
                try:
                    signal_id = SignalID(signal_name)
                except Exception:
                    if signal_data is not None and signal_name not in signal_data.raw:
                        signal_data.add_signal(signal_name, "virtual")
                    signal_id = SignalID(signal_name)

                if hasattr(entity, "circuit_enabled"):
                    entity.circuit_enabled = True

                if hasattr(entity, "set_circuit_condition"):
                    entity.set_circuit_condition(first_operand=signal_id, comparator=">", second_operand=0)
                else:
                    behavior = getattr(entity, "control_behavior", {}) or {}
                    behavior["circuit_enable_disable"] = True
                    behavior["circuit_condition"] = {
                        "first_signal": {"name": signal_id.name, "type": signal_id.type},
                        "comparator": ">",
                        "constant": 0,
                    }
                    entity.control_behavior = behavior
            except Exception as exc:
                self.diagnostics.warning(
                    f"Failed to configure circuit enable for entity '{op.entity_id}': {exc}"
                )
        else:
            self.diagnostics.warning(
                f"Dynamic property '{op.property_name}' not yet supported for entity '{op.entity_id}'"
            )

    def emit_entity_prop_read(self, op: IR_EntityPropRead):
        """Emit entity property read (circuit network connection)."""
        property_key = (op.entity_id, op.property_name)
        stored_value = self.entity_property_signals.get(property_key)

        if isinstance(stored_value, SignalRef):
            self.signal_sources[op.node_id] = stored_value.source_id
        elif isinstance(stored_value, int):
            from draftsman.entity import ConstantCombinator

            pos = self.layout.get_next_position()
            combinator = ConstantCombinator(tile_position=pos)
            section = combinator.add_section()
            signal_name = self._get_signal_name(op.output_type)
            section.set_signal(index=0, name=signal_name, count=stored_value)
            combinator = self._add_entity(combinator)

            placement = EntityPlacement(
                entity=combinator,
                entity_id=op.node_id,
                position=pos,
                output_signals={signal_name: "red"},
                input_signals={},
            )
            self.entities[op.node_id] = placement
            self.signal_sources[op.node_id] = op.node_id
        else:
            # Fallback: expose controlling entity output directly
            if op.entity_id in self.entities:
                self.signal_sources[op.node_id] = op.entity_id

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

                            # Determine connection sides based on entity capabilities
                            source_entity = source_placement.entity
                            sink_entity = sink_placement.entity
                            source_dual = getattr(
                                source_entity, "dual_circuit_connectable", False
                            )
                            sink_dual = getattr(
                                sink_entity, "dual_circuit_connectable", False
                            )

                            connection_kwargs: Dict[str, Any] = dict(
                                color=wire_color,
                                entity_1=source_entity,
                                entity_2=sink_entity,
                            )

                            if source_dual:
                                connection_kwargs["side_1"] = "output"
                            if sink_dual:
                                connection_kwargs["side_2"] = "input"

                            # Add circuit connection
                            try:
                                self.blueprint.add_circuit_connection(**connection_kwargs)
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
            if isinstance(mapped_signal, dict):
                signal_name = mapped_signal.get("name", clean_name)
                signal_type = mapped_signal.get("type", "virtual")
                if signal_data is not None and signal_name not in signal_data.raw:
                    try:
                        signal_data.add_signal(signal_name, signal_type)
                    except Exception as e:
                        self.diagnostics.warning(
                            f"Could not register custom signal '{signal_name}': {e}"
                        )
                return signal_name
            if signal_data is not None and mapped_signal not in signal_data.raw:
                try:
                    inferred_type = "virtual"
                    signal_data.add_signal(mapped_signal, inferred_type)
                except Exception as e:
                    self.diagnostics.warning(
                        f"Could not register signal '{mapped_signal}' as virtual: {e}"
                    )
            return mapped_signal

        if signal_data is not None and clean_name in signal_data.raw:
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

        # Default to registering as virtual signal for custom identifiers
        if signal_data is not None and clean_name not in signal_data.raw:
            try:
                signal_data.add_signal(clean_name, "virtual")
            except Exception as e:
                self.diagnostics.warning(
                    f"Could not register signal '{clean_name}' as virtual: {e}"
                )
        return clean_name

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
        signal_ids: List[str] = []

        if isinstance(signal_ref, SignalRef):
            signal_ids.append(signal_ref.source_id)
        elif isinstance(signal_ref, str):
            signal_ids.append(signal_ref)

        for signal_id in signal_ids:
            if signal_id not in self.signal_sinks:
                self.signal_sinks[signal_id] = []
            if entity_id not in self.signal_sinks[signal_id]:
                self.signal_sinks[signal_id].append(entity_id)


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
