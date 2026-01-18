"""IR builder utilities for constructing Facto IR."""

from __future__ import annotations

from typing import Any

from dsl_compiler.src.ast.statements import ASTNode
from dsl_compiler.src.common.signal_registry import SignalTypeRegistry

from .nodes import (
    MEMORY_TYPE_STANDARD,
    BundleRef,
    DeciderCondition,
    IRArith,
    IRConst,
    IRDecider,
    IRMemCreate,
    IRMemRead,
    IRMemWrite,
    IRNode,
    IRPlaceEntity,
    IRValue,
    IRWireMerge,
    SignalRef,
    ValueRef,
)


class IRBuilder:
    """Builder for constructing IR from AST nodes."""

    def __init__(self, signal_registry: SignalTypeRegistry | None = None) -> None:
        self.operations: list[IRNode] = []
        self.node_counter = 0
        if signal_registry is None:
            self.signal_registry = SignalTypeRegistry()
        else:
            self.signal_registry = signal_registry
        self._operation_index: dict[str, IRNode] = {}

    @property
    def signal_type_map(self) -> dict[str, Any]:
        """Get signal type map from registry."""
        return self.signal_registry.get_all_mappings()

    def next_id(self, prefix: str = "ir") -> str:
        """Generate the next unique IR node identifier."""

        self.node_counter += 1
        return f"{prefix}_{self.node_counter}"

    def add_operation(self, op: IRNode) -> IRNode:
        """Add an operation to the IR and register it for lookups."""

        self.operations.append(op)
        self._operation_index[op.node_id] = op
        return op

    def get_operation(self, node_id: str) -> IRNode | None:
        """Retrieve an operation previously registered with the builder."""

        return self._operation_index.get(node_id)

    def annotate_signal(
        self,
        signal: ValueRef,
        *,
        label: str | None = None,
        source_ast: ASTNode | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Attach debug metadata to the producing IR node backing a signal."""
        if not isinstance(signal, SignalRef):
            return

        op = self.get_operation(signal.source_id)
        if isinstance(op, IRValue):
            if label:
                op.debug_label = op.debug_label or label
            if source_ast and op.source_ast is None:
                op.source_ast = source_ast
            if metadata:
                op.debug_metadata.update(metadata)

        if label:
            signal.debug_label = label
        if source_ast:
            signal.source_ast = source_ast
        if metadata:
            signal.debug_metadata.update(metadata)

    def const(self, signal_type: str, value: int, source_ast: ASTNode | None = None) -> SignalRef:
        """Create a constant signal value."""

        node_id = self.next_id("const")
        op = IRConst(node_id, signal_type, source_ast)
        op.value = value
        self.add_operation(op)
        return SignalRef(signal_type, node_id, source_ast=source_ast)

    def arithmetic(
        self,
        op: str,
        left: ValueRef,
        right: ValueRef,
        output_type: str,
        source_ast: ASTNode | None = None,
    ) -> SignalRef:
        """Create an arithmetic operation."""

        node_id = self.next_id("arith")
        arith_op = IRArith(node_id, output_type, source_ast)
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
        output_value: ValueRef | int,
        output_type: str,
        source_ast: ASTNode | None = None,
        copy_count_from_input: bool = False,
    ) -> SignalRef:
        """Create a decider combinator operation.

        Args:
            test_op: Comparison operator (==, !=, <, <=, >, >=)
            left: Left operand of comparison
            right: Right operand of comparison
            output_value: Value to output when condition is true (signal or constant)
            output_type: Signal type for output
            source_ast: Source AST node for debugging
            copy_count_from_input: If True and output_value is a signal,
                                   copy the signal's count from input rather than outputting a constant
        """
        node_id = self.next_id("decider")
        decider_op = IRDecider(node_id, output_type, source_ast)
        decider_op.test_op = test_op
        decider_op.left = left
        decider_op.right = right
        decider_op.output_value = output_value
        decider_op.copy_count_from_input = copy_count_from_input
        self.add_operation(decider_op)
        return SignalRef(output_type, node_id, source_ast=source_ast)

    def decider_multi(
        self,
        conditions: list[tuple],
        combine_type: str,
        output_value: ValueRef | int,
        output_type: str,
        source_ast: ASTNode | None = None,
        copy_count_from_input: bool = False,
    ) -> SignalRef:
        """Create a multi-condition decider combinator.

        This is used for condition folding optimization where multiple comparisons
        in a logical AND/OR chain are combined into a single decider.

        Args:
            conditions: List of (comparator, left_operand, right_operand) tuples.
                       Each tuple represents one condition row.
            combine_type: How conditions are combined ("and" or "or").
                         All conditions use the same combine type.
            output_value: Value to output when combined condition is true.
            output_type: Signal type for output.
            source_ast: Source AST node for debugging.
            copy_count_from_input: If True, copy signal value instead of constant.

        Returns:
            SignalRef pointing to the output of this decider.
        """
        node_id = self.next_id("decider")
        decider_op = IRDecider(node_id, output_type, source_ast)
        decider_op.output_value = output_value
        decider_op.copy_count_from_input = copy_count_from_input

        for i, (comparator, left, right) in enumerate(conditions):
            cond = DeciderCondition(
                comparator=comparator,
                # First condition's compare_type is ignored by Factorio
                compare_type=combine_type if i > 0 else "or",
                first_operand=left,
                second_operand=right,
            )
            decider_op.conditions.append(cond)

        self.add_operation(decider_op)
        return SignalRef(output_type, node_id, source_ast=source_ast)

    def wire_merge(
        self,
        sources: list[ValueRef],
        output_type: str,
        source_ast: ASTNode | None = None,
    ) -> SignalRef:
        """Create a virtual wire merge combining multiple sources."""

        node_id = self.next_id("wire_merge")
        merge_op = IRWireMerge(node_id, output_type, source_ast)
        for source in sources:
            merge_op.add_source(source)
        self.add_operation(merge_op)
        return SignalRef(output_type, node_id, source_ast=source_ast)

    def memory_create(
        self,
        memory_id: str,
        signal_type: str,
        source_ast: ASTNode | None = None,
        memory_type: str = MEMORY_TYPE_STANDARD,
    ) -> None:
        """Create a memory cell declaration."""

        op = IRMemCreate(memory_id, signal_type, source_ast, memory_type)
        self.add_operation(op)

    def memory_read(
        self, memory_id: str, signal_type: str, source_ast: ASTNode | None = None
    ) -> SignalRef:
        """Read from a memory cell."""

        node_id = self.next_id("mem_read")
        op = IRMemRead(node_id, signal_type, source_ast)
        op.memory_id = memory_id
        self.add_operation(op)
        return SignalRef(signal_type, node_id, source_ast=source_ast)

    def memory_write(
        self,
        memory_id: str,
        data_signal: ValueRef,
        write_enable: ValueRef,
        source_ast: ASTNode | None = None,
    ) -> IRMemWrite:
        """Write to a memory cell (standard write-gated latch)."""

        op = IRMemWrite(memory_id, data_signal, write_enable, source_ast)
        self.add_operation(op)
        return op

    def latch_write(
        self,
        memory_id: str,
        value: ValueRef,
        set_signal: ValueRef,
        reset_signal: ValueRef,
        latch_type: str,
        source_ast: ASTNode | None = None,
        *,
        set_condition: tuple[ValueRef, str, int] | None = None,
        reset_condition: tuple[ValueRef, str, int] | None = None,
    ) -> Any:
        """Write to a memory cell using latch mode (single combinator).

        Args:
            memory_id: The memory cell ID
            value: The value to output when latch is ON
            set_signal: Signal that turns latch ON
            reset_signal: Signal that turns latch OFF
            latch_type: MEMORY_TYPE_SR_LATCH or MEMORY_TYPE_RS_LATCH
            source_ast: Source AST node for diagnostics
            set_condition: Optional inline condition (signal, op, constant) for set
            reset_condition: Optional inline condition (signal, op, constant) for reset
        """
        from .nodes import IRLatchWrite

        op = IRLatchWrite(
            memory_id,
            value,
            set_signal,
            reset_signal,
            latch_type,
            source_ast,
            set_condition=set_condition,
            reset_condition=reset_condition,
        )
        self.add_operation(op)
        return op

    def place_entity(
        self,
        entity_id: str,
        prototype: str,
        x: int | ValueRef,
        y: int | ValueRef,
        properties: dict[str, Any] | None = None,
        source_ast: ASTNode | None = None,
    ) -> None:
        """Emit an entity placement."""

        op = IRPlaceEntity(entity_id, prototype, x, y, properties)
        op.source_ast = source_ast  # Attach source AST for line number tracking
        self.add_operation(op)

    def allocate_implicit_type(self) -> str:
        """Allocate a new implicit signal type name and record the mapping."""
        return self.signal_registry.allocate_implicit()

    def bundle_const(
        self,
        signals: dict[str, int],
        source_ast: ASTNode | None = None,
    ) -> BundleRef:
        """Create a constant combinator with multiple signals (bundle).

        Args:
            signals: Dictionary mapping signal type names to integer values
            source_ast: Source AST node for debugging

        Returns:
            BundleRef pointing to the constant combinator
        """
        node_id = self.next_id("bundle_const")
        # Use "signal-each" as the nominal output type for bundle constants
        op = IRConst(node_id, "signal-each", source_ast)
        op.signals = signals.copy()
        self.add_operation(op)
        return BundleRef(set(signals.keys()), node_id, source_ast=source_ast)

    def bundle_arithmetic(
        self,
        op: str,
        bundle: BundleRef,
        operand: ValueRef,
        source_ast: ASTNode | None = None,
    ) -> BundleRef:
        """Create an arithmetic operation on a bundle using 'each'.

        Args:
            op: Arithmetic operator (+, -, *, /, %, **, <<, >>, AND, OR, XOR)
            bundle: The bundle to operate on
            operand: Signal or constant to use as right operand
            source_ast: Source AST node for debugging

        Returns:
            BundleRef with same signal types as input bundle
        """
        node_id = self.next_id("bundle_arith")
        # Use "signal-each" for both input and output
        arith_op = IRArith(node_id, "signal-each", source_ast)
        arith_op.op = op
        arith_op.left = SignalRef("signal-each", bundle.source_id)
        arith_op.right = operand

        # If the right operand is a signal (not a constant), we need wire separation
        # to prevent the scalar signal from being processed by "each"
        if isinstance(operand, SignalRef):
            arith_op.needs_wire_separation = True

        self.add_operation(arith_op)
        return BundleRef(bundle.signal_types.copy(), node_id, source_ast=source_ast)

    def get_ir(self) -> list[IRNode]:
        """Return a copy of the currently built IR operations."""

        return self.operations.copy()


__all__ = ["IRBuilder"]
