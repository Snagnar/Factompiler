"""IR optimization passes for the Factorio Circuit DSL."""

from __future__ import annotations

from typing import Dict, List

from dsl_compiler.src.ir import (
    IRNode,
    IR_Arith,
    IR_Decider,
    IR_MemWrite,
    SignalRef,
)


class CSEOptimizer:
    """Common subexpression elimination for IR nodes."""

    def __init__(self) -> None:
        self.expr_cache: Dict[str, str] = {}  # expression key -> canonical node id
        self.replacements: Dict[str, str] = {}  # removed node id -> canonical node id

    def optimize(self, ir_operations: List[IRNode]) -> List[IRNode]:
        """Eliminate redundant arithmetic and decider operations."""

        optimized: List[IRNode] = []

        for op in ir_operations:
            if isinstance(op, (IR_Arith, IR_Decider)):
                key = self._make_key(op)
                if key and key in self.expr_cache:
                    self.replacements[op.node_id] = self.expr_cache[key]
                    continue
                if key:
                    self.expr_cache[key] = op.node_id

            optimized.append(op)

        if self.replacements:
            optimized = self._update_references(optimized)

        return optimized

    def _make_key(self, op: IRNode) -> str:
        if isinstance(op, IR_Arith):
            left_key = self._value_key(op.left)
            right_key = self._value_key(op.right)
            return f"arith:{op.op}:{left_key}:{right_key}:{op.output_type}"

        if isinstance(op, IR_Decider):
            left_key = self._value_key(op.left)
            right_key = self._value_key(op.right)
            output_key = self._value_key(op.output_value)
            return (
                "decider:"
                f"{op.test_op}:"
                f"{left_key}:"
                f"{right_key}:"
                f"{output_key}:"
                f"{op.output_type}"
            )

        return ""

    def _value_key(self, value) -> str:
        if isinstance(value, SignalRef):
            source_id = self.replacements.get(value.source_id, value.source_id)
            return f"sig:{source_id}:{value.signal_type}"

        if isinstance(value, int):
            return f"int:{value}"

        if isinstance(value, str):
            return f"str:{value}"

        return repr(value)

    def _update_references(self, operations: List[IRNode]) -> List[IRNode]:
        for op in operations:
            if isinstance(op, IR_Arith):
                op.left = self._update_value(op.left)
                op.right = self._update_value(op.right)
            elif isinstance(op, IR_Decider):
                op.left = self._update_value(op.left)
                op.right = self._update_value(op.right)
                op.output_value = self._update_value(op.output_value)
            elif isinstance(op, IR_MemWrite):
                op.data_signal = self._update_value(op.data_signal)
                op.write_enable = self._update_value(op.write_enable)

        return operations

    def _update_value(self, value):
        if isinstance(value, SignalRef):
            canonical = self.replacements.get(value.source_id)
            if canonical:
                return SignalRef(
                    value.signal_type,
                    canonical,
                    debug_label=value.debug_label,
                    source_ast=value.source_ast,
                    metadata=value.debug_metadata,
                )
        return value
