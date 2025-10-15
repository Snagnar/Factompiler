# ir.py
"""
Intermediate Representation (IR) for the Factorio Circuit DSL.

The IR represents the program in a form that can be directly lowered to
Factorio combinators and entities. It provides a canonical representation
that separates concerns between semantic analysis and combinator generation.
"""

from abc import ABC
from typing import Dict, List, Optional, Union, Any

from dsl_compiler.src.dsl_ast import ASTNode


# =============================================================================
# Signal and Value References
# =============================================================================


class SignalRef:
    """Reference to a signal value in the IR."""

    def __init__(
        self,
        signal_type: str,
        source_id: str,
        *,
        debug_label: Optional[str] = None,
        source_ast: Optional[ASTNode] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.signal_type = signal_type  # The signal type (e.g., "iron-plate", "__v1")
        self.source_id = source_id  # ID of the IR node that produces this signal
        self.debug_label = debug_label
        self.source_ast = source_ast
        self.debug_metadata: Dict[str, Any] = metadata.copy() if metadata else {}

    def __str__(self) -> str:
        return f"{self.signal_type}@{self.source_id}"


ValueRef = Union[SignalRef, int]  # IR values can be signals or integers


# =============================================================================
# IR Node Base Classes
# =============================================================================


class IRNode(ABC):
    """Base class for all IR nodes."""

    def __init__(self, node_id: str, source_ast: Optional[ASTNode] = None):
        self.node_id = node_id
        self.source_ast = source_ast  # For diagnostics
        self.debug_metadata: Dict[str, Any] = {}

    def __str__(self) -> str:
        """String representation of the IR node."""
        return f"{self.__class__.__name__}({self.node_id})"


class IRValue(IRNode):
    """Base class for IR nodes that produce values."""

    def __init__(
        self, node_id: str, output_type: str, source_ast: Optional[ASTNode] = None
    ):
        super().__init__(node_id, source_ast)
        self.output_type = output_type  # The signal type this node outputs
        self.debug_label: Optional[str] = None


class IREffect(IRNode):
    """Base class for IR nodes that have side effects (memory writes, entity operations)."""

    pass


# =============================================================================
# Value-Producing IR Nodes
# =============================================================================


class IR_Const(IRValue):
    """Constant combinator producing a fixed signal value."""

    def __init__(
        self, node_id: str, output_type: str, source_ast: Optional[ASTNode] = None
    ):
        super().__init__(node_id, output_type, source_ast)
        self.value: int = 0

    def __str__(self) -> str:
        return f"IR_Const({self.node_id}: {self.output_type} = {self.value})"


class IR_Arith(IRValue):
    """Arithmetic combinator operation."""

    def __init__(
        self, node_id: str, output_type: str, source_ast: Optional[ASTNode] = None
    ):
        super().__init__(node_id, output_type, source_ast)
        self.op: str = "+"
        self.left: ValueRef = 0
        self.right: ValueRef = 0

    def __str__(self) -> str:
        return f"IR_Arith({self.node_id}: {self.output_type} = {self.left} {self.op} {self.right})"


class IR_Decider(IRValue):
    """Decider combinator operation."""

    def __init__(
        self, node_id: str, output_type: str, source_ast: Optional[ASTNode] = None
    ):
        super().__init__(node_id, output_type, source_ast)
        self.test_op: str = "=="
        self.left: ValueRef = 0
        self.right: ValueRef = 0
        self.output_value: Union[ValueRef, int] = 1

    def __str__(self) -> str:
        return f"IR_Decider({self.node_id}: {self.output_type} = if({self.left} {self.test_op} {self.right}) then {self.output_value})"


class IR_MemRead(IRValue):
    """Memory read operation."""

    def __init__(
        self, node_id: str, output_type: str, source_ast: Optional[ASTNode] = None
    ):
        super().__init__(node_id, output_type, source_ast)
        self.memory_id: str = ""

    def __str__(self) -> str:
        return (
            f"IR_MemRead({self.node_id}: {self.output_type} = read({self.memory_id}))"
        )


class IR_EntityPropRead(IRValue):
    """Entity property read operation."""

    def __init__(
        self, node_id: str, output_type: str, source_ast: Optional[ASTNode] = None
    ):
        super().__init__(node_id, output_type, source_ast)
        self.entity_id: str = ""
        self.property_name: str = ""

    def __str__(self) -> str:
        return f"IR_EntityPropRead({self.node_id}: {self.output_type} = {self.entity_id}.{self.property_name})"


# =============================================================================
# Effect IR Nodes
# =============================================================================


class IR_MemCreate(IREffect):
    """Memory cell creation."""

    def __init__(
        self,
        memory_id: str,
        signal_type: str,
        initial_value: ValueRef,
        source_ast: Optional[ASTNode] = None,
    ):
        super().__init__(f"mem_create_{memory_id}", source_ast)
        self.memory_id = memory_id
        self.signal_type = signal_type
        self.initial_value = initial_value

    def __str__(self) -> str:
        return (
            f"IR_MemCreate({self.memory_id}: {self.signal_type} = {self.initial_value})"
        )


class IR_MemWrite(IREffect):
    """Memory write operation."""

    def __init__(
        self,
        memory_id: str,
        data_signal: ValueRef,
        write_enable: ValueRef,
        source_ast: Optional[ASTNode] = None,
    ):
        super().__init__(f"mem_write_{memory_id}", source_ast)
        self.memory_id = memory_id
        self.data_signal = data_signal
        self.write_enable = (
            write_enable  # Signal that enables the write (usually constant 1)
        )

    def __str__(self) -> str:
        return f"IR_MemWrite({self.memory_id} <- {self.data_signal} when {self.write_enable})"


class IR_PlaceEntity(IREffect):
    """Entity placement in blueprint."""

    def __init__(
        self,
        entity_id: str,
        prototype: str,
        x: Union[int, ValueRef],
        y: Union[int, ValueRef],
        properties: Dict[str, Any] = None,
    ):
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
        return (
            f"IR_EntityPropWrite({self.entity_id}.{self.property_name} <- {self.value})"
        )


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

    def __init__(
        self,
        node_id: str,
        operations: List[IRNode] = None,
        inputs: Dict[str, str] = None,
        outputs: Dict[str, str] = None,
        source_ast: Optional[ASTNode] = None,
    ):
        super().__init__(node_id, source_ast)
        self.operations = operations or []
        self.inputs = inputs or {}  # param_name -> signal_type
        self.outputs = outputs or {}  # output_name -> signal_type

    def __str__(self) -> str:
        return f"IR_Group({self.node_id}: {len(self.operations)} operations)"


class IR_FuncDecl(IRNode):
    """Function declaration IR node."""

    def __init__(
        self,
        node_id: str,
        func_name: str,
        params: List[str],
        body_operations: List[IRNode],
        return_ref: Optional[str] = None,
        source_ast: Optional[ASTNode] = None,
    ):
        super().__init__(node_id, source_ast)
        self.func_name = func_name
        self.params = params
        self.body_operations = body_operations
        self.return_ref = return_ref

    def __str__(self) -> str:
        return f"IR_FuncDecl({self.func_name}, params={self.params})"


class IR_FuncCall(IRNode):
    """Function call IR node - will be inlined during emission."""

    def __init__(
        self,
        node_id: str,
        func_name: str,
        args: List[ValueRef],
        result_ref: str,
        source_ast: Optional[ASTNode] = None,
    ):
        super().__init__(node_id, source_ast)
        self.func_name = func_name
        self.args = args
        self.result_ref = result_ref

    def __str__(self) -> str:
        return f"IR_FuncCall({self.func_name}({', '.join(map(str, self.args))}) -> {self.result_ref})"


# =============================================================================
# IR Builder
# =============================================================================


class IRBuilder:
    """Builder for constructing IR from AST."""

    def __init__(self):
        self.operations: List[IRNode] = []
        self.node_counter = 0
        self.signal_type_map: Dict[str, str] = {}  # implicit -> factorio signal
        self.implicit_type_counter = 0
        self._operation_index: Dict[str, IRNode] = {}

    def next_id(self, prefix: str = "ir") -> str:
        """Generate next unique IR node ID."""
        self.node_counter += 1
        return f"{prefix}_{self.node_counter}"

    def add_operation(self, op: IRNode) -> IRNode:
        """Add an operation to the IR."""
        self.operations.append(op)
        self._operation_index[op.node_id] = op
        return op

    def get_operation(self, node_id: str) -> Optional[IRNode]:
        """Retrieve an operation previously registered with the builder."""

        return self._operation_index.get(node_id)

    def annotate_signal(
        self,
        signal: ValueRef,
        *,
        label: Optional[str] = None,
        source_ast: Optional[ASTNode] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Attach debug metadata to the producing IR node backing a signal reference."""

        if not isinstance(signal, SignalRef):
            return

        op = self.get_operation(signal.source_id)
        if isinstance(op, IRValue):
            if label:
                current = getattr(op, "debug_label", None)
                if current is None or current == label:
                    op.debug_label = label
                else:
                    aliases = op.debug_metadata.setdefault("aliases", [])
                    if label not in aliases:
                        aliases.append(label)
            if source_ast and op.source_ast is None:
                op.source_ast = source_ast
            if metadata:
                for key, value in metadata.items():
                    if value is None:
                        continue

                    if key == "name":
                        existing_name = op.debug_metadata.get("name")
                        if existing_name in (None, value):
                            op.debug_metadata["name"] = value
                        else:
                            aliases = op.debug_metadata.setdefault("aliases", [])
                            if value not in aliases:
                                aliases.append(value)
                        continue

                    if key == "aliases":
                        aliases = op.debug_metadata.setdefault("aliases", [])
                        for alias in value:
                            if (
                                alias != op.debug_metadata.get("name")
                                and alias not in aliases
                            ):
                                aliases.append(alias)
                        continue

                    if key == "location":
                        op.debug_metadata.setdefault(key, value)
                        continue

                    if key not in op.debug_metadata:
                        op.debug_metadata[key] = value

        if label:
            signal.debug_label = label
        if source_ast:
            signal.source_ast = source_ast
        if metadata:
            signal.debug_metadata.update(metadata)

    def const(
        self, signal_type: str, value: int, source_ast: Optional[ASTNode] = None
    ) -> SignalRef:
        """Create a constant signal."""
        node_id = self.next_id("const")
        op = IR_Const(node_id, signal_type, source_ast)
        op.value = value
        self.add_operation(op)
        return SignalRef(signal_type, node_id, source_ast=source_ast)

    def arithmetic(
        self,
        op: str,
        left: ValueRef,
        right: ValueRef,
        output_type: str,
        source_ast: Optional[ASTNode] = None,
    ) -> SignalRef:
        """Create an arithmetic operation."""
        node_id = self.next_id("arith")
        arith_op = IR_Arith(node_id, output_type, source_ast)
        arith_op.op = op
        arith_op.left = left
        arith_op.right = right
        self.add_operation(arith_op)
        return SignalRef(output_type, node_id, source_ast=source_ast)

    def decider(
        self,
        test_op: str,
        left: ValueRef,
        right: ValueRef,
        output_value: Union[ValueRef, int],
        output_type: str,
        source_ast: Optional[ASTNode] = None,
    ) -> SignalRef:
        """Create a decider combinator operation."""
        node_id = self.next_id("decider")
        decider_op = IR_Decider(node_id, output_type, source_ast)
        decider_op.test_op = test_op
        decider_op.left = left
        decider_op.right = right
        decider_op.output_value = output_value
        self.add_operation(decider_op)
        return SignalRef(output_type, node_id, source_ast=source_ast)

    def memory_create(
        self,
        memory_id: str,
        signal_type: str,
        initial_value: ValueRef,
        source_ast: Optional[ASTNode] = None,
    ):
        """Create a memory cell."""
        op = IR_MemCreate(memory_id, signal_type, initial_value, source_ast)
        self.add_operation(op)

    def memory_read(
        self, memory_id: str, signal_type: str, source_ast: Optional[ASTNode] = None
    ) -> SignalRef:
        """Read from memory."""
        node_id = self.next_id("mem_read")
        op = IR_MemRead(node_id, signal_type, source_ast)
        op.memory_id = memory_id
        self.add_operation(op)
        return SignalRef(signal_type, node_id, source_ast=source_ast)

    def memory_write(
        self,
        memory_id: str,
        data_signal: ValueRef,
        write_enable: ValueRef,
        source_ast: Optional[ASTNode] = None,
    ):
        """Write to memory."""
        op = IR_MemWrite(memory_id, data_signal, write_enable, source_ast)
        self.add_operation(op)

    def place_entity(
        self,
        entity_id: str,
        prototype: str,
        x: Union[int, ValueRef],
        y: Union[int, ValueRef],
        properties: Optional[Dict[str, Any]] = None,
        source_ast: Optional[ASTNode] = None,
    ):
        """Place an entity."""
        op = IR_PlaceEntity(entity_id, prototype, x, y, properties)
        self.add_operation(op)

    def func_decl(
        self,
        func_name: str,
        params: List[str],
        body_operations: List[IRNode],
        return_ref: Optional[str] = None,
        source_ast: Optional[ASTNode] = None,
    ) -> IR_FuncDecl:
        """Create a function declaration."""
        node_id = self.next_id("func")
        op = IR_FuncDecl(
            node_id, func_name, params, body_operations, return_ref, source_ast
        )
        self.add_operation(op)
        return op

    def func_call(
        self,
        func_name: str,
        args: List[ValueRef],
        result_ref: str,
        source_ast: Optional[ASTNode] = None,
    ) -> IR_FuncCall:
        """Create a function call."""
        node_id = self.next_id("call")
        op = IR_FuncCall(node_id, func_name, args, result_ref, source_ast)
        self.add_operation(op)
        return op

    def allocate_implicit_type(self) -> str:
        """Allocate a new implicit signal type."""
        self.implicit_type_counter += 1
        implicit_name = f"__v{self.implicit_type_counter}"

        factorio_signal = self._virtual_signal_name(self.implicit_type_counter)
        self.signal_type_map[implicit_name] = factorio_signal

        return implicit_name

    def _virtual_signal_name(self, index: int) -> str:
        """Map implicit index to a unique Factorio virtual signal name."""
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        index -= 1
        name = ""
        while index >= 0:
            name = alphabet[index % 26] + name
            index = index // 26 - 1
        return f"signal-{name}"

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
