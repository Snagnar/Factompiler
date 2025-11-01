"""IR builder utilities for constructing Factorio Circuit DSL IR."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from dsl_compiler.src.ast import ASTNode
from dsl_compiler.src.common import SignalTypeRegistry

from .nodes import (
    IRNode,
    IRValue,
    IR_Arith,
    IR_Const,
    IR_Decider,
    IR_FuncCall,
    IR_FuncDecl,
    IR_MemCreate,
    IR_MemRead,
    IR_MemWrite,
    IR_PlaceEntity,
    IR_WireMerge,
    SignalRef,
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
        """Backward compatibility: get signal type map from registry."""
        # Return the signal mappings as-is (dict format with name and type)
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
                current = getattr(op, "debug_label", None)
                if current in (None, label):
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

        op = IR_MemWrite(memory_id, data_signal, write_enable, source_ast)
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
        """Create a function call placeholder that will be inlined later."""

        node_id = self.next_id("call")
        op = IR_FuncCall(node_id, func_name, args, result_ref, source_ast)
        self.add_operation(op)
        return op

    def allocate_implicit_type(self) -> str:
        """Allocate a new implicit signal type name and record the mapping."""
        return self.signal_registry.allocate_implicit()

    def get_ir(self) -> List[IRNode]:
        """Return a copy of the currently built IR operations."""

        return self.operations.copy()

    def print_ir(self) -> None:  # pragma: no cover - debug helper
        """Print the IR in a human-readable form for debugging."""

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


__all__ = ["IRBuilder"]
