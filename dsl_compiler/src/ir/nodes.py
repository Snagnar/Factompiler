from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dsl_compiler.src.ast.statements import ASTNode

"""IR node and signal representations for the Facto."""


class SignalRef:
    """Reference to a signal value in the IR."""

    def __init__(
        self,
        signal_type: str,
        source_id: str,
        *,
        debug_label: str | None = None,
        source_ast: ASTNode | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.signal_type = signal_type
        self.source_id = source_id
        self.debug_label = debug_label
        self.source_ast = source_ast
        self.debug_metadata: dict[str, Any] = metadata.copy() if metadata else {}

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.signal_type}@{self.source_id}"


class BundleRef:
    """Reference to a bundle of signals in the IR.

    A bundle contains multiple signals on the same wire. Operations on bundles
    use Factorio's 'each' signal for parallel processing.
    """

    def __init__(
        self,
        signal_types: set[str],
        source_id: str,
        *,
        debug_label: str | None = None,
        source_ast: ASTNode | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.signal_types = signal_types  # Set of signal type names in the bundle
        self.source_id = source_id
        self.debug_label = debug_label
        self.source_ast = source_ast
        self.debug_metadata: dict[str, Any] = metadata.copy() if metadata else {}

    def __str__(self) -> str:  # pragma: no cover - debug helper
        types_str = ", ".join(sorted(self.signal_types))
        return f"Bundle({types_str})@{self.source_id}"


ValueRef = SignalRef | BundleRef | int


class IRNode:
    """Base class for all IR nodes."""

    def __init__(self, node_id: str, source_ast: ASTNode | None = None) -> None:
        self.node_id = node_id
        self.source_ast = source_ast
        self.debug_metadata: dict[str, Any] = {}

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.__class__.__name__}({self.node_id})"


class IRValue(IRNode):
    """Base class for IR nodes that produce values."""

    def __init__(self, node_id: str, output_type: str, source_ast: ASTNode | None = None) -> None:
        super().__init__(node_id, source_ast)
        self.output_type = output_type
        self.debug_label: str | None = None


class IREffect(IRNode):
    """Base class for IR nodes that have side effects."""

    pass


class IRConst(IRValue):
    """Constant combinator producing fixed signal value(s).

    For single signals: uses output_type and value.
    For bundles: uses signals dict mapping signal_type -> value.
    """

    def __init__(self, node_id: str, output_type: str, source_ast: ASTNode | None = None) -> None:
        super().__init__(node_id, output_type, source_ast)
        self.value: int = 0
        # For multi-signal constants (bundles): signal_type -> value
        self.signals: dict[str, int] = {}

    def __str__(self) -> str:  # pragma: no cover - debug helper
        if self.signals:
            signals_str = ", ".join(f"{k}={v}" for k, v in self.signals.items())
            return f"IRConst({self.node_id}: bundle({signals_str}))"
        return f"IRConst({self.node_id}: {self.output_type} = {self.value})"


class IRArith(IRValue):
    """Arithmetic combinator operation."""

    def __init__(self, node_id: str, output_type: str, source_ast: ASTNode | None = None) -> None:
        super().__init__(node_id, output_type, source_ast)
        self.op: str = "+"
        self.left: ValueRef = 0
        self.right: ValueRef = 0
        # For bundle operations with signal operands, we need wire separation:
        # left operand (bundle/each) on one wire color, right operand (scalar) on the other.
        self.needs_wire_separation: bool = False

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"IRArith({self.node_id}: {self.output_type} = {self.left} {self.op} {self.right})"


@dataclass
class DeciderCondition:
    """Single condition row in a decider combinator (Factorio 2.0 multi-condition support).

    Represents one row in a decider combinator's condition block. Multiple conditions
    can be combined with AND/OR logic using compare_type.

    Wire filtering allows specifying which wire colors to read each signal from,
    essential for latches where feedback and external signals must be separated.

    Supports two modes:
    1. String-based (layout-time): Uses first_signal/second_signal strings.
       Used by memory_builder for SR latches where signal names are known.
    2. ValueRef-based (IR-time): Uses first_operand/second_operand ValueRefs.
       Used by condition folding optimization during lowering.
    """

    comparator: str = ">"
    """Comparison operator: =, !=, <, <=, >, >="""

    first_signal: str = ""
    """Left-hand signal name (e.g., 'signal-S') - for layout-time construction"""

    first_constant: int | None = None
    """Left-hand constant value (alternative to first_signal)"""

    first_signal_wires: set[str] | None = None
    """Wire colors to read first_signal from: {'red'}, {'green'}, or None for both"""

    second_signal: str = ""
    """Right-hand signal name (e.g., 'signal-R') - for layout-time construction"""

    second_constant: int | None = None
    """Right-hand constant value (alternative to second_signal)"""

    second_signal_wires: set[str] | None = None
    """Wire colors to read second_signal from: {'red'}, {'green'}, or None for both"""

    compare_type: str = "or"
    """How to combine with previous condition: 'or' or 'and'"""

    # ValueRef-based operands for IR-time construction (condition folding)
    # When set, these take precedence over string-based operands
    first_operand: Any | None = None  # ValueRef - uses Any to avoid circular import
    """Left operand as ValueRef - for IR-time construction. Takes precedence over first_signal."""

    second_operand: Any | None = None  # ValueRef - uses Any to avoid circular import
    """Right operand as ValueRef - for IR-time construction. Takes precedence over second_signal."""

    def __str__(self) -> str:  # pragma: no cover - debug helper
        if self.first_operand is not None:
            left = str(self.first_operand)
        else:
            left = self.first_signal or str(self.first_constant)
        if self.second_operand is not None:
            right = str(self.second_operand)
        else:
            right = self.second_signal or str(self.second_constant)
        return f"({left} {self.comparator} {right})"


class IRDecider(IRValue):
    """Decider combinator operation.

    Supports both single-condition mode (legacy) and multi-condition mode (Factorio 2.0).
    When conditions list is non-empty, it takes precedence over the legacy fields.
    """

    def __init__(self, node_id: str, output_type: str, source_ast: ASTNode | None = None) -> None:
        super().__init__(node_id, output_type, source_ast)
        # Legacy single-condition fields (for backwards compatibility)
        self.test_op: str = "=="
        self.left: ValueRef = 0
        self.right: ValueRef = 0
        self.output_value: ValueRef | int = 1
        self.copy_count_from_input: bool = False

        # Factorio 2.0 multi-condition support
        self.conditions: list[DeciderCondition] = []
        """List of conditions for multi-condition mode. If non-empty, overrides legacy fields."""

    def __str__(self) -> str:  # pragma: no cover - debug helper
        copy_mode = " COPY" if self.copy_count_from_input else ""
        if self.conditions:
            cond_str = " ".join(
                f"{c.compare_type.upper()} {c}" if i > 0 else str(c)
                for i, c in enumerate(self.conditions)
            )
            return (
                f"IRDecider({self.node_id}: {self.output_type} = "
                f"if({cond_str}) then {self.output_value}{copy_mode})"
            )
        return (
            "IRDecider("
            f"{self.node_id}: {self.output_type} = if({self.left} {self.test_op} {self.right}) "
            f"then {self.output_value}{copy_mode})"
        )


class IRWireMerge(IRValue):
    """Logical wire merge of multiple simple sources on the same signal."""

    def __init__(self, node_id: str, output_type: str, source_ast: ASTNode | None = None) -> None:
        super().__init__(node_id, output_type, source_ast)
        self.sources: list[ValueRef] = []

    def add_source(self, source: ValueRef) -> None:
        self.sources.append(source)

    def __str__(self) -> str:  # pragma: no cover - debug helper
        joined = ", ".join(str(src) for src in self.sources)
        return f"IRWireMerge({self.node_id}: {self.output_type} = wire_merge({joined}))"


class IRMemRead(IRValue):
    """Memory read operation."""

    def __init__(self, node_id: str, output_type: str, source_ast: ASTNode | None = None) -> None:
        super().__init__(node_id, output_type, source_ast)
        self.memory_id: str = ""

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"IRMemRead({self.node_id}: {self.output_type} = read({self.memory_id}))"


class IREntityPropRead(IRValue):
    """Entity property read operation."""

    def __init__(self, node_id: str, output_type: str, source_ast: ASTNode | None = None) -> None:
        super().__init__(node_id, output_type, source_ast)
        self.entity_id: str = ""
        self.property_name: str = ""

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return (
            "IREntityPropRead("
            f"{self.node_id}: {self.output_type} = {self.entity_id}.{self.property_name})"
        )


class IREntityOutput(IRValue):
    """Read entity's circuit output as bundle.

    Represents reading all signals an entity outputs to the circuit network.
    For chests: all items stored
    For tanks: fluid amount
    For train stops with read_from_train: train contents

    In layout planning, this creates a virtual signal source at the entity
    rather than creating a new combinator.
    """

    def __init__(self, node_id: str, entity_id: str, source_ast: ASTNode | None = None) -> None:
        super().__init__(node_id, "bundle", source_ast)
        self.entity_id = entity_id

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"IREntityOutput({self.node_id}: bundle = {self.entity_id}.output)"


# Memory type constants
MEMORY_TYPE_STANDARD = "standard"
MEMORY_TYPE_RS_LATCH = "rs_latch"
MEMORY_TYPE_SR_LATCH = "sr_latch"


class IRMemCreate(IREffect):
    """Memory cell creation.

    Supports three memory types:
    - standard: Write-gated latch (2 decider combinators)
    - rs_latch: Reset-priority latch (1 decider, S > R condition)
    - sr_latch: Set-priority latch / hysteresis (1 decider, multi-condition)
    """

    def __init__(
        self,
        memory_id: str,
        signal_type: str,
        source_ast: ASTNode | None = None,
        memory_type: str = MEMORY_TYPE_STANDARD,
    ) -> None:
        super().__init__(f"mem_create_{memory_id}", source_ast)
        self.memory_id = memory_id
        self.signal_type = signal_type
        self.memory_type = memory_type

    def __str__(self) -> str:  # pragma: no cover - debug helper
        type_suffix = f" ({self.memory_type})" if self.memory_type != MEMORY_TYPE_STANDARD else ""
        return f"IRMemCreate({self.memory_id}: {self.signal_type}{type_suffix})"


class IRMemWrite(IREffect):
    """Memory write operation."""

    def __init__(
        self,
        memory_id: str,
        data_signal: ValueRef,
        write_enable: ValueRef,
        source_ast: ASTNode | None = None,
    ) -> None:
        super().__init__(f"mem_write_{memory_id}", source_ast)
        self.memory_id = memory_id
        self.data_signal = data_signal
        self.write_enable = write_enable

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"IRMemWrite({self.memory_id} <- {self.data_signal} when {self.write_enable})"


class IRLatchWrite(IREffect):
    """Latch write operation.

    Creates a single-combinator latch with set/reset behavior:
    - SR latch (set priority): condition S >= R, outputs when set >= reset
    - RS latch (reset priority): condition S > R, outputs when set > reset

    The latch_type field determines the priority:
    - MEMORY_TYPE_SR_LATCH: set wins on tie (S >= R)
    - MEMORY_TYPE_RS_LATCH: reset wins on tie (S > R)
    """

    def __init__(
        self,
        memory_id: str,
        value: ValueRef,
        set_signal: ValueRef,
        reset_signal: ValueRef,
        latch_type: str,  # MEMORY_TYPE_SR_LATCH or MEMORY_TYPE_RS_LATCH
        source_ast: ASTNode | None = None,
    ) -> None:
        super().__init__(f"latch_write_{memory_id}", source_ast)
        self.memory_id = memory_id
        self.value = value
        self.set_signal = set_signal
        self.reset_signal = reset_signal
        self.latch_type = latch_type

    def __str__(self) -> str:  # pragma: no cover - debug helper
        priority = "SR" if self.latch_type == MEMORY_TYPE_SR_LATCH else "RS"
        return f"IRLatchWrite({self.memory_id} <- {self.value}, {priority}, set={self.set_signal}, reset={self.reset_signal})"


class IRPlaceEntity(IREffect):
    """Entity placement in a blueprint."""

    def __init__(
        self,
        entity_id: str,
        prototype: str,
        x: int | ValueRef,
        y: int | ValueRef,
        properties: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"place_{entity_id}")
        self.entity_id = entity_id
        self.prototype = prototype
        self.x = x
        self.y = y
        self.properties = properties or {}

    def __str__(self) -> str:  # pragma: no cover - debug helper
        props_str = f", {self.properties}" if self.properties else ""
        return (
            f"IRPlaceEntity({self.entity_id}: {self.prototype} at ({self.x}, {self.y}){props_str})"
        )


class IREntityPropWrite(IREffect):
    """Entity property write operation."""

    def __init__(self, entity_id: str, property_name: str, value: ValueRef) -> None:
        super().__init__(f"prop_write_{entity_id}_{property_name}")
        self.entity_id = entity_id
        self.property_name = property_name
        self.value = value
        # Optional: inline bundle condition for all()/any() inlining
        self.inline_bundle_condition: dict[str, Any] | None = None

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"IREntityPropWrite({self.entity_id}.{self.property_name} <- {self.value})"


__all__ = [
    "SignalRef",
    "BundleRef",
    "ValueRef",
    "IRNode",
    "IRValue",
    "IREffect",
    "IRConst",
    "IRArith",
    "DeciderCondition",
    "IRDecider",
    "IRWireMerge",
    "IRMemRead",
    "IREntityPropRead",
    "IREntityOutput",
    "IRMemCreate",
    "IRMemWrite",
    "IRLatchWrite",
    "IRPlaceEntity",
    "IREntityPropWrite",
    # Memory type constants
    "MEMORY_TYPE_STANDARD",
    "MEMORY_TYPE_RS_LATCH",
    "MEMORY_TYPE_SR_LATCH",
]
