from __future__ import annotations
from abc import ABC
from typing import Any, Dict, List, Optional, Union
from dsl_compiler.src.ast.statements import ASTNode

"""IR node and signal representations for the Factorio Circuit DSL."""


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
    ) -> None:
        self.signal_type = signal_type
        self.source_id = source_id
        self.debug_label = debug_label
        self.source_ast = source_ast
        self.debug_metadata: Dict[str, Any] = metadata.copy() if metadata else {}

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.signal_type}@{self.source_id}"


ValueRef = Union[SignalRef, int]


class IRNode(ABC):
    """Base class for all IR nodes."""

    def __init__(self, node_id: str, source_ast: Optional[ASTNode] = None) -> None:
        self.node_id = node_id
        self.source_ast = source_ast
        self.debug_metadata: Dict[str, Any] = {}

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.__class__.__name__}({self.node_id})"


class IRValue(IRNode):
    """Base class for IR nodes that produce values."""

    def __init__(
        self, node_id: str, output_type: str, source_ast: Optional[ASTNode] = None
    ) -> None:
        super().__init__(node_id, source_ast)
        self.output_type = output_type
        self.debug_label: Optional[str] = None


class IREffect(IRNode):
    """Base class for IR nodes that have side effects."""

    pass


class IR_Const(IRValue):
    """Constant combinator producing a fixed signal value."""

    def __init__(
        self, node_id: str, output_type: str, source_ast: Optional[ASTNode] = None
    ) -> None:
        super().__init__(node_id, output_type, source_ast)
        self.value: int = 0

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"IR_Const({self.node_id}: {self.output_type} = {self.value})"


class IR_Arith(IRValue):
    """Arithmetic combinator operation."""

    def __init__(
        self, node_id: str, output_type: str, source_ast: Optional[ASTNode] = None
    ) -> None:
        super().__init__(node_id, output_type, source_ast)
        self.op: str = "+"
        self.left: ValueRef = 0
        self.right: ValueRef = 0

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return (
            "IR_Arith("
            f"{self.node_id}: {self.output_type} = {self.left} {self.op} {self.right}"
            ")"
        )


class IR_Decider(IRValue):
    """Decider combinator operation."""

    def __init__(
        self, node_id: str, output_type: str, source_ast: Optional[ASTNode] = None
    ) -> None:
        super().__init__(node_id, output_type, source_ast)
        self.test_op: str = "=="
        self.left: ValueRef = 0
        self.right: ValueRef = 0
        self.output_value: Union[ValueRef, int] = 1
        self.copy_count_from_input: bool = False

    def __str__(self) -> str:  # pragma: no cover - debug helper
        copy_mode = " COPY" if self.copy_count_from_input else ""
        return (
            "IR_Decider("
            f"{self.node_id}: {self.output_type} = if({self.left} {self.test_op} {self.right}) "
            f"then {self.output_value}{copy_mode})"
        )


class IR_WireMerge(IRValue):
    """Logical wire merge of multiple simple sources on the same signal."""

    def __init__(
        self, node_id: str, output_type: str, source_ast: Optional[ASTNode] = None
    ) -> None:
        super().__init__(node_id, output_type, source_ast)
        self.sources: List[ValueRef] = []

    def add_source(self, source: ValueRef) -> None:
        self.sources.append(source)

    def __str__(self) -> str:  # pragma: no cover - debug helper
        joined = ", ".join(str(src) for src in self.sources)
        return (
            f"IR_WireMerge({self.node_id}: {self.output_type} = wire_merge({joined}))"
        )


class IR_MemRead(IRValue):
    """Memory read operation."""

    def __init__(
        self, node_id: str, output_type: str, source_ast: Optional[ASTNode] = None
    ) -> None:
        super().__init__(node_id, output_type, source_ast)
        self.memory_id: str = ""

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"IR_MemRead({self.node_id}: {self.output_type} = read({self.memory_id}))"
        )


class IR_EntityPropRead(IRValue):
    """Entity property read operation."""

    def __init__(
        self, node_id: str, output_type: str, source_ast: Optional[ASTNode] = None
    ) -> None:
        super().__init__(node_id, output_type, source_ast)
        self.entity_id: str = ""
        self.property_name: str = ""

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return (
            "IR_EntityPropRead("
            f"{self.node_id}: {self.output_type} = {self.entity_id}.{self.property_name})"
        )


class IR_MemCreate(IREffect):
    """Memory cell creation."""

    def __init__(
        self,
        memory_id: str,
        signal_type: str,
        source_ast: Optional[ASTNode] = None,
    ) -> None:
        super().__init__(f"mem_create_{memory_id}", source_ast)
        self.memory_id = memory_id
        self.signal_type = signal_type

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"IR_MemCreate({self.memory_id}: {self.signal_type})"


class IR_MemWrite(IREffect):
    """Memory write operation."""

    def __init__(
        self,
        memory_id: str,
        data_signal: ValueRef,
        write_enable: ValueRef,
        source_ast: Optional[ASTNode] = None,
    ) -> None:
        super().__init__(f"mem_write_{memory_id}", source_ast)
        self.memory_id = memory_id
        self.data_signal = data_signal
        self.write_enable = write_enable

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"IR_MemWrite({self.memory_id} <- {self.data_signal} when {self.write_enable})"


class IR_PlaceEntity(IREffect):
    """Entity placement in a blueprint."""

    def __init__(
        self,
        entity_id: str,
        prototype: str,
        x: Union[int, ValueRef],
        y: Union[int, ValueRef],
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(f"place_{entity_id}")
        self.entity_id = entity_id
        self.prototype = prototype
        self.x = x
        self.y = y
        self.properties = properties or {}

    def __str__(self) -> str:  # pragma: no cover - debug helper
        props_str = f", {self.properties}" if self.properties else ""
        return f"IR_PlaceEntity({self.entity_id}: {self.prototype} at ({self.x}, {self.y}){props_str})"


class IR_EntityPropWrite(IREffect):
    """Entity property write operation."""

    def __init__(self, entity_id: str, property_name: str, value: ValueRef) -> None:
        super().__init__(f"prop_write_{entity_id}_{property_name}")
        self.entity_id = entity_id
        self.property_name = property_name
        self.value = value

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"IR_EntityPropWrite({self.entity_id}.{self.property_name} <- {self.value})"
        )


__all__ = [
    "SignalRef",
    "ValueRef",
    "IRNode",
    "IRValue",
    "IREffect",
    "IR_Const",
    "IR_Arith",
    "IR_Decider",
    "IR_WireMerge",
    "IR_MemRead",
    "IR_EntityPropRead",
    "IR_MemCreate",
    "IR_MemWrite",
    "IR_PlaceEntity",
    "IR_EntityPropWrite",
]
