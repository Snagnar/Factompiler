"""IR builder utilities for constructing Factorio Circuit DSL IR."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from dsl_compiler.src.ast.statements import ASTNode
from dsl_compiler.src.common.signal_registry import SignalTypeRegistry

from .nodes import (
    IRNode,
    IRValue,
    IR_Arith,
    IR_Const,
    IR_Decider,
    IR_MemCreate,
    IR_MemRead,
    IR_MemWrite,
    IR_PlaceEntity,
    IR_WireMerge,
    SignalRef,
    BundleRef,
    ValueRef,
)


class IRBuilder:
    """Builder for constructing IR from AST nodes."""

    def __init__(self, signal_registry: Optional[SignalTypeRegistry] = None) -> None:
        self.operations: List[IRNode] = []
        self.node_counter = 0
        if signal_registry is None:
            self.signal_registry = SignalTypeRegistry()
        else:
            self.signal_registry = signal_registry
        self._operation_index: Dict[str, IRNode] = {}

    @property
    def signal_type_map(self) -> Dict[str, Any]:
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

    def const(
        self, signal_type: str, value: int, source_ast: Optional[ASTNode] = None
    ) -> SignalRef:
        """Create a constant signal value."""

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
        decider_op = IR_Decider(node_id, output_type, source_ast)
        decider_op.test_op = test_op
        decider_op.left = left
        decider_op.right = right
        decider_op.output_value = output_value
        decider_op.copy_count_from_input = copy_count_from_input
        self.add_operation(decider_op)
        return SignalRef(output_type, node_id, source_ast=source_ast)

    def wire_merge(
        self,
        sources: List[ValueRef],
        output_type: str,
        source_ast: Optional[ASTNode] = None,
    ) -> SignalRef:
        """Create a virtual wire merge combining multiple sources."""

        node_id = self.next_id("wire_merge")
        merge_op = IR_WireMerge(node_id, output_type, source_ast)
        for source in sources:
            merge_op.add_source(source)
        self.add_operation(merge_op)
        return SignalRef(output_type, node_id, source_ast=source_ast)

    def memory_create(
        self,
        memory_id: str,
        signal_type: str,
        source_ast: Optional[ASTNode] = None,
    ) -> None:
        """Create a memory cell declaration."""

        op = IR_MemCreate(memory_id, signal_type, source_ast)
        self.add_operation(op)

    def memory_read(
        self, memory_id: str, signal_type: str, source_ast: Optional[ASTNode] = None
    ) -> SignalRef:
        """Read from a memory cell."""

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
    ) -> IR_MemWrite:
        """Write to a memory cell."""
        node_id = self.next_id(f"mem_write_{memory_id}")
        op = IR_MemWrite(node_id, memory_id, data_signal, write_enable, source_ast)
        self.add_operation(op)
        return op

    def place_entity(
        self,
        entity_id: str,
        prototype: str,
        x: Union[int, ValueRef],
        y: Union[int, ValueRef],
        properties: Optional[Dict[str, Any]] = None,
        source_ast: Optional[ASTNode] = None,
    ) -> None:
        """Emit an entity placement."""

        op = IR_PlaceEntity(entity_id, prototype, x, y, properties)
        op.source_ast = source_ast  # Attach source AST for line number tracking
        self.add_operation(op)

    def allocate_implicit_type(self) -> str:
        """Allocate a new implicit signal type name and record the mapping."""
        return self.signal_registry.allocate_implicit()

    def bundle_const(
        self,
        signals: Dict[str, int],
        source_ast: Optional[ASTNode] = None,
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
        op = IR_Const(node_id, "signal-each", source_ast)
        op.signals = signals.copy()
        self.add_operation(op)
        return BundleRef(set(signals.keys()), node_id, source_ast=source_ast)

    def bundle_arithmetic(
        self,
        op: str,
        bundle: BundleRef,
        operand: ValueRef,
        source_ast: Optional[ASTNode] = None,
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
        arith_op = IR_Arith(node_id, "signal-each", source_ast)
        arith_op.op = op
        arith_op.left = SignalRef("signal-each", bundle.source_id)
        arith_op.right = operand
        
        # If the right operand is a signal (not a constant), we need wire separation
        # to prevent the scalar signal from being processed by "each"
        if isinstance(operand, SignalRef):
            arith_op.needs_wire_separation = True
            
        self.add_operation(arith_op)
        return BundleRef(bundle.signal_types.copy(), node_id, source_ast=source_ast)

    def get_ir(self) -> List[IRNode]:
        """Return a copy of the currently built IR operations."""

        return self.operations.copy()


__all__ = ["IRBuilder"]
