# ir.py
"""
Intermediate Representation (IR) for the Factorio Circuit DSL.

The IR represents the program in a form that can be directly lowered to
Factorio combinators and entities. It provides a canonical representation
that separates concerns between semantic analysis and combinator generation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from enum import Enum

from dsl_compiler.src.dsl_ast import ASTNode


# =============================================================================
# Signal and Value References
# =============================================================================

class SignalRef:
    """Reference to a signal value in the IR."""
    
    def __init__(self, signal_type: str, source_id: str):
        self.signal_type = signal_type  # The signal type (e.g., "iron-plate", "__v1")
        self.source_id = source_id      # ID of the IR node that produces this signal
    
    def __str__(self) -> str:
        return f"{self.signal_type}@{self.source_id}"


class BundleRef:
    """Reference to a bundle (multi-channel) value in the IR."""
    
    def __init__(self, channels: Dict[str, SignalRef], bundle_id: str):
        self.channels = channels  # signal_type -> SignalRef
        self.bundle_id = bundle_id
    
    def __str__(self) -> str:
        return f"Bundle[{', '.join(self.channels.keys())}]@{self.bundle_id}"


ValueRef = Union[SignalRef, BundleRef, int]  # IR values can be signals, bundles, or integers


# =============================================================================
# IR Node Base Classes
# =============================================================================

class IRNode(ABC):
    """Base class for all IR nodes."""
    
    def __init__(self, node_id: str, source_ast: Optional[ASTNode] = None):
        self.node_id = node_id
        self.source_ast = source_ast  # For diagnostics
    
    @abstractmethod
    def __str__(self) -> str:
        pass


class IRValue(IRNode):
    """Base class for IR nodes that produce values."""
    
    def __init__(self, node_id: str, output_type: str, source_ast: Optional[ASTNode] = None):
        super().__init__(node_id, source_ast)
        self.output_type = output_type  # The signal type this node outputs


class IREffect(IRNode):
    """Base class for IR nodes that have side effects (memory writes, entity operations)."""
    pass


# =============================================================================
# Value-Producing IR Nodes
# =============================================================================

class IR_Const(IRValue):
    """Constant combinator producing a fixed signal value."""
    
    def __init__(self, node_id: str, output_type: str, source_ast: Optional[ASTNode] = None):
        super().__init__(node_id, output_type, source_ast)
        self.value: int = 0
    
    def __str__(self) -> str:
        return f"IR_Const({self.node_id}: {self.output_type} = {self.value})"


class IR_Input(IRValue):
    """Input from circuit network at specified index."""
    
    def __init__(self, node_id: str, output_type: str, source_ast: Optional[ASTNode] = None):
        super().__init__(node_id, output_type, source_ast)
        self.index: int = 0
    
    def __str__(self) -> str:
        return f"IR_Input({self.node_id}: {self.output_type} = input[{self.index}])"


class IR_Arith(IRValue):
    """Arithmetic combinator operation."""
    
    def __init__(self, node_id: str, output_type: str, source_ast: Optional[ASTNode] = None):
        super().__init__(node_id, output_type, source_ast)
        self.op: str = "+"
        self.left: ValueRef = 0
        self.right: ValueRef = 0
    
    def __str__(self) -> str:
        return f"IR_Arith({self.node_id}: {self.output_type} = {self.left} {self.op} {self.right})"


class IR_Decider(IRValue):
    """Decider combinator operation."""
    
    def __init__(self, node_id: str, output_type: str, source_ast: Optional[ASTNode] = None):
        super().__init__(node_id, output_type, source_ast)
        self.test_op: str = "=="
        self.left: ValueRef = 0
        self.right: ValueRef = 0
        self.output_value: Union[ValueRef, int] = 1
    
    def __str__(self) -> str:
        return f"IR_Decider({self.node_id}: {self.output_type} = if({self.left} {self.test_op} {self.right}) then {self.output_value})"


class IR_MemRead(IRValue):
    """Memory read operation."""
    
    def __init__(self, node_id: str, output_type: str, source_ast: Optional[ASTNode] = None):
        super().__init__(node_id, output_type, source_ast)
        self.memory_id: str = ""
    
    def __str__(self) -> str:
        return f"IR_MemRead({self.node_id}: {self.output_type} = read({self.memory_id}))"


class IR_EntityPropRead(IRValue):
    """Entity property read operation."""
    
    def __init__(self, node_id: str, output_type: str, source_ast: Optional[ASTNode] = None):
        super().__init__(node_id, output_type, source_ast)
        self.entity_id: str = ""
        self.property_name: str = ""
    
    def __str__(self) -> str:
        return f"IR_EntityPropRead({self.node_id}: {self.output_type} = {self.entity_id}.{self.property_name})"


class IR_Bundle(IRValue):
    """Bundle construction from multiple signals."""
    
    def __init__(self, node_id: str, inputs: Dict[str, ValueRef], source_ast: Optional[ASTNode] = None):
        super().__init__(node_id, "bundle", source_ast)
        self.inputs = inputs
    
    def __str__(self) -> str:
        inputs_str = ", ".join(f"{k}: {v}" for k, v in self.inputs.items())
        return f"IR_Bundle({self.node_id}: bundle({inputs_str}))"


# =============================================================================
# Effect IR Nodes
# =============================================================================

class IR_MemCreate(IREffect):
    """Memory cell creation."""
    
    def __init__(self, memory_id: str, signal_type: str, initial_value: ValueRef):
        super().__init__(f"mem_create_{memory_id}")
        self.memory_id = memory_id
        self.signal_type = signal_type
        self.initial_value = initial_value
    
    def __str__(self) -> str:
        return f"IR_MemCreate({self.memory_id}: {self.signal_type} = {self.initial_value})"


class IR_MemWrite(IREffect):
    """Memory write operation."""
    
    def __init__(self, memory_id: str, data_signal: ValueRef, write_enable: ValueRef):
        super().__init__(f"mem_write_{memory_id}")
        self.memory_id = memory_id
        self.data_signal = data_signal
        self.write_enable = write_enable  # Signal that enables the write (usually constant 1)
    
    def __str__(self) -> str:
        return f"IR_MemWrite({self.memory_id} <- {self.data_signal} when {self.write_enable})"


class IR_PlaceEntity(IREffect):
    """Entity placement in blueprint."""
    
    def __init__(self, entity_id: str, prototype: str, x: int, y: int, properties: Dict[str, Any] = None):
        super().__init__(f"place_{entity_id}")
        self.entity_id = entity_id
        self.prototype = prototype
        self.x = x
        self.y = y
        self.properties = properties or {}
    
    def __str__(self) -> str:
        props_str = f", {self.properties}" if self.properties else ""
        return f"IR_PlaceEntity({self.entity_id}: {self.prototype} at ({self.x}, {self.y}){props_str})"


class IR_EntityPropWrite(IREffect):
    """Entity property write operation."""
    
    def __init__(self, entity_id: str, property_name: str, value: ValueRef):
        super().__init__(f"prop_write_{entity_id}_{property_name}")
        self.entity_id = entity_id
        self.property_name = property_name
        self.value = value
    
    def __str__(self) -> str:
        return f"IR_EntityPropWrite({self.entity_id}.{self.property_name} <- {self.value})"


class IR_ConnectToWire(IREffect):
    """Connect a signal to an output wire/channel."""
    
    def __init__(self, signal: ValueRef, channel: str):
        super().__init__(f"connect_{channel}")
        self.signal = signal
        self.channel = channel
    
    def __str__(self) -> str:
        return f"IR_ConnectToWire({self.signal} -> {self.channel})"


# =============================================================================
# IR Module/Grouping
# =============================================================================

class IR_Group(IRNode):
    """Group of IR operations (for functions/modules)."""
    
    def __init__(self, node_id: str, operations: List[IRNode] = None, 
                 inputs: Dict[str, str] = None, outputs: Dict[str, str] = None, 
                 source_ast: Optional[ASTNode] = None):
        super().__init__(node_id, source_ast)
        self.operations = operations or []
        self.inputs = inputs or {}   # param_name -> signal_type
        self.outputs = outputs or {} # output_name -> signal_type
    
    def __str__(self) -> str:
        return f"IR_Group({self.node_id}: {len(self.operations)} operations)"


# =============================================================================
# IR Builder
# =============================================================================

class IRBuilder:
    """Builder for constructing IR from AST."""
    
    def __init__(self):
        self.operations: List[IRNode] = []
        self.node_counter = 0
        self.signal_type_map: Dict[str, str] = {}  # implicit -> factorio signal
        
    def next_id(self, prefix: str = "ir") -> str:
        """Generate next unique IR node ID."""
        self.node_counter += 1
        return f"{prefix}_{self.node_counter}"
    
    def add_operation(self, op: IRNode) -> IRNode:
        """Add an operation to the IR."""
        self.operations.append(op)
        return op
    
    def const(self, signal_type: str, value: int, source_ast: Optional[ASTNode] = None) -> SignalRef:
        """Create a constant signal."""
        node_id = self.next_id("const")
        op = IR_Const(node_id, signal_type, source_ast)
        op.value = value
        self.add_operation(op)
        return SignalRef(signal_type, node_id)
    
    def input_signal(self, index: int, signal_type: Optional[str] = None, source_ast: Optional[ASTNode] = None) -> SignalRef:
        """Create an input signal."""
        if signal_type is None:
            signal_type = self.allocate_implicit_type()
        
        node_id = self.next_id("input")
        op = IR_Input(node_id, signal_type, source_ast)
        op.index = index
        self.add_operation(op)
        return SignalRef(signal_type, node_id)
    
    def arithmetic(self, op: str, left: ValueRef, right: ValueRef, 
                  output_type: str, source_ast: Optional[ASTNode] = None) -> SignalRef:
        """Create an arithmetic operation."""
        node_id = self.next_id("arith")
        arith_op = IR_Arith(node_id, output_type, source_ast)
        arith_op.op = op
        arith_op.left = left
        arith_op.right = right
        self.add_operation(arith_op)
        return SignalRef(output_type, node_id)
    
    def decider(self, test_op: str, left: ValueRef, right: ValueRef,
               output_value: Union[ValueRef, int], output_type: str,
               source_ast: Optional[ASTNode] = None) -> SignalRef:
        """Create a decider combinator operation."""
        node_id = self.next_id("decider")
        decider_op = IR_Decider(node_id, output_type, source_ast)
        decider_op.test_op = test_op
        decider_op.left = left
        decider_op.right = right
        decider_op.output_value = output_value
        self.add_operation(decider_op)
        return SignalRef(output_type, node_id)
    
    def memory_create(self, memory_id: str, signal_type: str, 
                     initial_value: ValueRef, source_ast: Optional[ASTNode] = None):
        """Create a memory cell."""
        op = IR_MemCreate(memory_id, signal_type, initial_value)
        self.add_operation(op)
    
    def memory_read(self, memory_id: str, signal_type: str, source_ast: Optional[ASTNode] = None) -> SignalRef:
        """Read from memory."""
        node_id = self.next_id("mem_read")
        op = IR_MemRead(node_id, signal_type, source_ast)
        op.memory_id = memory_id
        self.add_operation(op)
        return SignalRef(signal_type, node_id)
    
    def memory_write(self, memory_id: str, data_signal: ValueRef, 
                    write_enable: ValueRef, source_ast: Optional[ASTNode] = None):
        """Write to memory."""
        op = IR_MemWrite(memory_id, data_signal, write_enable)
        self.add_operation(op)
    
    def place_entity(self, entity_id: str, prototype: str, x: int, y: int,
                    properties: Optional[Dict[str, Any]] = None, 
                    source_ast: Optional[ASTNode] = None):
        """Place an entity."""
        op = IR_PlaceEntity(entity_id, prototype, x, y, properties)
        self.add_operation(op)
    
    def bundle(self, inputs: Dict[str, ValueRef], source_ast: Optional[ASTNode] = None) -> BundleRef:
        """Create a bundle."""
        node_id = self.next_id("bundle")
        op = IR_Bundle(node_id, inputs, source_ast)
        self.add_operation(op)
        
        # Create SignalRefs for each channel
        channels = {sig_type: SignalRef(sig_type, node_id) for sig_type in inputs.keys()}
        return BundleRef(channels, node_id)
    
    def allocate_implicit_type(self) -> str:
        """Allocate a new implicit signal type."""
        implicit_counter = len([t for t in self.signal_type_map.keys() if t.startswith("__v")])
        implicit_name = f"__v{implicit_counter + 1}"
        
        # Map to Factorio virtual signal
        factorio_signal = f"signal-{chr(ord('A') + implicit_counter % 26)}"
        self.signal_type_map[implicit_name] = factorio_signal
        
        return implicit_name
    
    def get_ir(self) -> List[IRNode]:
        """Get the built IR operations."""
        return self.operations.copy()
    
    def print_ir(self):
        """Print the IR for debugging."""
        print("IR Operations:")
        print("=" * 50)
        for i, op in enumerate(self.operations):
            print(f"{i:2d}: {op}")
        print()
        
        if self.signal_type_map:
            print("Signal Type Mapping:")
            print("=" * 30)
            for implicit, factorio in self.signal_type_map.items():
                print(f"  {implicit} -> {factorio}")
