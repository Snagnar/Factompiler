from __future__ import annotations

from typing import Any

from dsl_compiler.src.ast.expressions import ReadExpr, WriteExpr
from dsl_compiler.src.ast.statements import ASTNode, MemDecl
from dsl_compiler.src.ir.builder import SignalRef, ValueRef
from dsl_compiler.src.ir.nodes import (
    MEMORY_TYPE_RS_LATCH,
    MEMORY_TYPE_SR_LATCH,
    MEMORY_TYPE_STANDARD,
    IR_Const,
    IR_Decider,
)
from dsl_compiler.src.semantic.analyzer import SignalValue
from dsl_compiler.src.semantic.type_system import get_signal_type_name

"""Memory-related lowering helpers for the Facto."""


class MemoryLowerer:
    """Handles lowering of memory declarations and operations."""

    def __init__(self, parent: Any) -> None:
        self.parent = parent

    @property
    def ir_builder(self):
        return self.parent.ir_builder

    @property
    def semantic(self):
        return self.parent.semantic

    @property
    def diagnostics(self):
        return self.parent.diagnostics

    def _error(self, message: str, node: ASTNode | None = None) -> None:
        """Add a lowering error diagnostic."""
        self.diagnostics.error(message, stage="lowering", node=node)

    def lower_mem_decl(self, stmt: MemDecl) -> None:
        """Lower memory declaration to IR.

        Memory type (standard vs latch) is determined later when we see the write.
        At declaration time, we just create a placeholder.
        """
        memory_id = f"mem_{stmt.name}"
        self.parent.memory_refs[stmt.name] = memory_id

        signal_type: str | None = None
        mem_info = getattr(self.semantic, "memory_types", {}).get(stmt.name)

        if mem_info:
            signal_type = mem_info.signal_type
        else:
            symbol = self.semantic.symbol_table.lookup(stmt.name)
            if symbol:
                signal_type = get_signal_type_name(symbol.value_type)

        if signal_type is None:
            self._error(
                f"Memory '{stmt.name}' must have an explicit signal type.",
                stmt,
            )
            signal_type = "signal-0"

        self.parent.ensure_signal_registered(signal_type)

        self.parent.memory_types[stmt.name] = signal_type

        # Create memory with standard type initially - will be upgraded to latch
        # if a latch write is encountered
        self.ir_builder.memory_create(memory_id, signal_type, stmt, MEMORY_TYPE_STANDARD)

    def lower_read_expr(self, expr: ReadExpr) -> SignalRef:
        memory_name = expr.memory_name

        if memory_name not in self.parent.memory_refs:
            self._error(f"Undefined memory: {memory_name}", expr)
            signal_type = self.ir_builder.allocate_implicit_type()
            return self.ir_builder.const(signal_type, 0, expr)

        memory_id = self.parent.memory_refs[memory_name]

        signal_type = self._memory_signal_type(memory_name)
        if signal_type is None:
            signal_type = self.ir_builder.allocate_implicit_type()

        self.parent.ensure_signal_registered(signal_type)

        ref = self.ir_builder.memory_read(memory_id, signal_type, expr)
        # Attach expression context to memory read
        self.parent.expr_lowerer._attach_expr_context(ref.source_id, expr)
        return ref

    def lower_write_expr(self, expr: WriteExpr) -> SignalRef:
        """Lower memory write expression.

        Dispatches to standard write or latch write based on the expression type.
        """
        memory_name = expr.memory_name

        if memory_name not in self.parent.memory_refs:
            self._error(f"Undefined memory: {memory_name}", expr)
            return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)

        # Dispatch to latch write if set/reset signals are present
        if expr.is_latch_write():
            return self._lower_latch_write(expr)

        return self._lower_standard_write(expr)

    def _lower_latch_write(self, expr: WriteExpr) -> SignalRef:
        """Lower latch write: mem.write(value, set=s, reset=r)

        Creates a latch with:
        - SR latch (set priority): multi-condition with feedback
        - RS latch (reset priority): condition S > R

        The latch is a single decider combinator with green wire self-feedback.

        Value handling:
        - value=1: Latch outputs 1 directly, no multiplier needed
        - value=N (constant): Latch outputs 1, multiplier scales to N
        - value=signal: Latch outputs 1, multiplier uses signal × 1 (passthrough)
        """
        memory_name = expr.memory_name
        memory_id = self.parent.memory_refs[memory_name]

        # Lower the value expression
        # Can be an integer constant or a SignalRef
        self.parent.push_expr_context(f"write({memory_name}).value", expr)
        value_ref = self.parent.expr_lowerer.lower_expr(expr.value)
        self.parent.pop_expr_context()

        # The value is passed directly to IR - can be int or SignalRef
        # The layout phase will handle creating a multiplier if needed
        latch_value = value_ref

        # Lower set and reset signals
        self.parent.push_expr_context(f"write({memory_name}).set", expr)
        set_ref = self.parent.expr_lowerer.lower_expr(expr.set_signal)
        self.parent.pop_expr_context()

        self.parent.push_expr_context(f"write({memory_name}).reset", expr)
        reset_ref = self.parent.expr_lowerer.lower_expr(expr.reset_signal)
        self.parent.pop_expr_context()

        # CRITICAL: For latches, the output signal MUST be the set signal type
        # so that feedback participates in the S comparison.
        # Extract the set signal's type and use it as the memory's effective signal type.
        if isinstance(set_ref, SignalRef):
            set_signal_type = set_ref.signal_type
        else:
            self._error(
                f"Latch set signal must be a signal reference, got {type(set_ref).__name__}",
                expr,
            )
            set_signal_type = self.ir_builder.allocate_implicit_type()

        # Get the originally declared memory signal type
        declared_signal_type = self._memory_signal_type(memory_name)

        # Determine if we need a multiplier (value != 1 or value is signal)
        # The multiplier outputs on the DECLARED memory signal type
        # The latch outputs on the SET signal type (for feedback)
        needs_multiplier = not isinstance(latch_value, int) or latch_value != 1

        if needs_multiplier:
            # With multiplier: memory output is on declared signal type
            # Don't change memory_types - keep the declared type
            if declared_signal_type and declared_signal_type != set_signal_type:
                self.diagnostics.info(
                    f"Latch '{memory_name}': latch outputs on '{set_signal_type}' for feedback, "
                    f"multiplier converts to '{declared_signal_type}' for memory output.",
                    stage="lowering",
                    node=expr,
                )
        else:
            # Without multiplier: memory output is directly from latch
            # Latch outputs on the memory's declared signal type
            pass

        self.parent.ensure_signal_registered(set_signal_type)
        if declared_signal_type:
            self.parent.ensure_signal_registered(declared_signal_type)

        # Determine latch type based on set_priority
        # SR latch (set priority): multi-condition
        # RS latch (reset priority): S > R
        memory_type = MEMORY_TYPE_SR_LATCH if expr.set_priority else MEMORY_TYPE_RS_LATCH

        # Create IR latch write operation with the constant value
        # The latch_value is an integer, not an IR node - it's internal to the decider
        self.ir_builder.latch_write(
            memory_id,
            latch_value,  # Integer constant, not a SignalRef
            set_ref,
            reset_ref,
            memory_type,
            expr,
        )

        # Return a SignalRef that points to the memory's output
        # The output is provided by the latch combinator, not by this write
        # Use the set signal type (which is now the memory's effective type)
        return SignalRef(set_signal_type, memory_id)

    def _lower_standard_write(self, expr: WriteExpr) -> SignalRef:
        """Lower standard memory write: mem.write(value) or mem.write(value, when=cond)"""
        memory_name = expr.memory_name
        memory_id = self.parent.memory_refs[memory_name]

        # Push context for memory write so intermediates know their purpose
        self.parent.push_expr_context(f"write({memory_name})", expr)
        data_ref = self.parent.expr_lowerer.lower_expr(expr.value)
        self.parent.pop_expr_context()

        expected_signal_type = self._memory_signal_type(memory_name)
        if expected_signal_type is None:
            self._error(
                f"Memory '{memory_name}' does not have a resolved signal type during lowering.",
                expr,
            )
            expected_signal_type = self.ir_builder.allocate_implicit_type()

        coerced_data_ref = self._coerce_to_signal_type(data_ref, expected_signal_type, expr)

        if expr.when is not None:
            # Push context for the when condition
            self.parent.push_expr_context(f"write({memory_name}).when", expr)
            write_enable = self.parent.expr_lowerer.lower_expr(expr.when)
            self.parent.pop_expr_context()

            is_const_one = False
            if isinstance(write_enable, int) and write_enable == 1:
                is_const_one = True
            elif isinstance(write_enable, SignalRef):
                const_node = self.ir_builder.get_operation(write_enable.source_id)
                if isinstance(const_node, IR_Const) and const_node.value == 1:
                    is_const_one = True

            if not is_const_one:
                if isinstance(write_enable, SignalRef):
                    source_node = self.ir_builder.get_operation(write_enable.source_id)
                    if isinstance(source_node, IR_Decider):
                        self.parent.ensure_signal_registered("signal-W")
                        source_node.output_type = "signal-W"
                        write_enable.signal_type = "signal-W"
                    elif write_enable.signal_type != "signal-W":
                        self.parent.ensure_signal_registered("signal-W")
                        write_enable = self.ir_builder.arithmetic(
                            "+", write_enable, 0, "signal-W", expr
                        )
        else:
            self.parent.ensure_signal_registered("signal-W")
            write_enable = self.ir_builder.const("signal-W", 1, expr)

        self.ir_builder.memory_write(memory_id, coerced_data_ref, write_enable, expr)

        return coerced_data_ref

    def _memory_signal_type(self, memory_name: str) -> str | None:
        if memory_name in self.parent.memory_types:
            return self.parent.memory_types[memory_name]

        mem_info = getattr(self.semantic, "memory_types", {}).get(memory_name)
        if mem_info and getattr(mem_info, "signal_type", None):
            return mem_info.signal_type

        symbol = self.semantic.symbol_table.lookup(memory_name)
        if symbol and isinstance(symbol.value_type, SignalValue):
            signal_info = symbol.value_type.signal_type
            if signal_info and getattr(signal_info, "name", None):
                return signal_info.name

        return None

    def _coerce_to_signal_type(
        self, value_ref: ValueRef, signal_type: str, node: ASTNode
    ) -> SignalRef:
        self.parent.ensure_signal_registered(signal_type)

        if isinstance(value_ref, SignalRef):
            if value_ref.signal_type == signal_type:
                return value_ref

            source_type = getattr(value_ref, "signal_type", None) or "<unknown>"

            # Always emit warning for type mismatches
            self._warning(
                "Type mismatch in memory write:\n"
                f"  Expected: '{signal_type}'\n"
                f"  Got: '{source_type}'\n"
                f'  Fix: Use projection: value | "{signal_type}"',
                node,
            )

            return self.ir_builder.arithmetic("+", value_ref, 0, signal_type, node)

        if isinstance(value_ref, int):
            return self.ir_builder.const(signal_type, value_ref, node)

        if (
            hasattr(value_ref, "signal_type")
            and getattr(value_ref, "signal_type", None) == signal_type
        ):
            return value_ref  # type: ignore[return-value]

        self._error(
            f"Cannot convert value of type '{type(value_ref).__name__}' to signal '{signal_type}' for memory write.",
            node,
        )
        return self.ir_builder.const(signal_type, 0, node)
