from __future__ import annotations

from typing import Any

from dsl_compiler.src.ast.expressions import (
    BinaryOp,
    IdentifierExpr,
    ReadExpr,
    SignalLiteral,
    WriteExpr,
)
from dsl_compiler.src.ast.literals import Identifier, NumberLiteral
from dsl_compiler.src.ast.statements import ASTNode, MemDecl
from dsl_compiler.src.ir.builder import SignalRef, ValueRef
from dsl_compiler.src.ir.nodes import (
    MEMORY_TYPE_RS_LATCH,
    MEMORY_TYPE_SR_LATCH,
    MEMORY_TYPE_STANDARD,
    IRConst,
    IRDecider,
)
from dsl_compiler.src.semantic.analyzer import SignalValue
from dsl_compiler.src.semantic.type_system import get_signal_type_name

"""Memory-related lowering helpers for the Facto."""

# Comparison operators that can be inlined into latch conditions
COMPARISON_OPS = {"<", "<=", ">", ">=", "==", "!="}


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

        OPTIMIZATION: When set and reset are simple comparisons on the same signal
        (e.g., set=battery < 20, reset=battery >= 80), we inline the comparisons
        directly into the latch combinator, eliminating the need for separate
        comparison deciders.

        Value handling:
        - value=1: Latch outputs 1 directly, no multiplier needed
        - value=N (constant): Latch outputs 1, multiplier scales to N
        - value=signal: Latch outputs 1, multiplier uses signal × 1 (passthrough)
        """
        memory_name = expr.memory_name
        memory_id = self.parent.memory_refs[memory_name]

        # Lower the value expression
        self.parent.push_expr_context(f"write({memory_name}).value", expr)
        value_ref = self.parent.expr_lowerer.lower_expr(expr.value)
        self.parent.pop_expr_context()

        latch_value = value_ref

        # Try to extract inline conditions for optimization
        set_condition, reset_condition = self._try_extract_inline_conditions(
            expr.set_signal, expr.reset_signal, memory_name
        )

        if set_condition and reset_condition:
            # OPTIMIZED PATH: Inline conditions directly into latch
            return self._lower_latch_write_inlined(
                expr, memory_id, memory_name, latch_value, set_condition, reset_condition
            )

        # FALLBACK PATH: Lower set/reset as separate expressions
        return self._lower_latch_write_standard(expr, memory_id, memory_name, latch_value)

    def _try_extract_inline_conditions(
        self,
        set_expr: Any,
        reset_expr: Any,
        memory_name: str,
    ) -> tuple[tuple[SignalRef, str, int] | None, tuple[SignalRef, str, int] | None]:
        """Try to extract inline conditions from set/reset expressions.

        For optimization, both must be:
        - Simple comparisons (signal <op> constant)
        - Comparing the same signal

        Returns (set_condition, reset_condition) or (None, None) if not optimizable.
        """
        set_cond = self._extract_simple_comparison(set_expr)
        reset_cond = self._extract_simple_comparison(reset_expr)

        if set_cond is None or reset_cond is None:
            self.diagnostics.info(
                f"Latch '{memory_name}': cannot inline conditions - "
                f"set or reset is not a simple comparison.",
                stage="lowering",
            )
            return None, None

        set_signal_name, set_op, set_const = set_cond
        reset_signal_name, reset_op, reset_const = reset_cond

        # Both must compare the same signal
        if set_signal_name != reset_signal_name:
            self.diagnostics.info(
                f"Latch '{memory_name}': cannot inline conditions - "
                f"set compares '{set_signal_name}' but reset compares '{reset_signal_name}'",
                stage="lowering",
            )
            return None, None

        # Lower the signal reference (will create a const combinator for input signals)
        # We only need to lower it once since both use the same signal
        self.parent.push_expr_context(f"write({memory_name}).condition_signal", None)
        signal_ref = self.parent.expr_lowerer.lower_expr(
            IdentifierExpr(set_signal_name, set_expr.line, set_expr.column)
        )
        self.parent.pop_expr_context()

        if not isinstance(signal_ref, SignalRef):
            return None, None

        self.diagnostics.info(
            f"Latch '{memory_name}': inlining conditions - "
            f"set=({set_signal_name} {set_op} {set_const}), "
            f"reset=({reset_signal_name} {reset_op} {reset_const})",
            stage="lowering",
        )

        return (signal_ref, set_op, set_const), (signal_ref, reset_op, reset_const)

    def _extract_simple_comparison(self, expr: Any) -> tuple[str, str, int] | None:
        """Extract a simple comparison from an expression.

        Returns (signal_name, operator, constant) if the expression is:
        - BinaryOp with comparison operator
        - Left is IdentifierExpr (signal reference) or Identifier
        - Right is NumberLiteral or SignalLiteral containing NumberLiteral (constant)

        Returns None otherwise.
        """
        if not isinstance(expr, BinaryOp):
            return None

        if expr.op not in COMPARISON_OPS:
            return None

        # Left must be an identifier (signal reference)
        # Can be IdentifierExpr (expression context) or Identifier (lvalue context)
        if isinstance(expr.left, (IdentifierExpr, Identifier)):
            signal_name = expr.left.name
        else:
            return None

        # Right must be a constant
        # Can be NumberLiteral directly or SignalLiteral containing NumberLiteral
        if isinstance(expr.right, NumberLiteral):
            constant = expr.right.value
        elif isinstance(expr.right, SignalLiteral):
            # SignalLiteral wraps the actual value
            if isinstance(expr.right.value, NumberLiteral):
                constant = expr.right.value.value
            else:
                return None
        else:
            return None

        return (signal_name, expr.op, constant)

    def _lower_latch_write_inlined(
        self,
        expr: WriteExpr,
        memory_id: str,
        memory_name: str,
        latch_value: ValueRef,
        set_condition: tuple[SignalRef, str, int],
        reset_condition: tuple[SignalRef, str, int],
    ) -> SignalRef:
        """Lower latch write with inlined conditions (optimized path).

        The conditions are passed directly to the IR, and the layout phase
        will generate a single multi-condition decider combinator.
        """
        signal_ref = set_condition[0]  # Same as reset_condition[0]

        # Get the memory's declared signal type for the output
        declared_signal_type = self._memory_signal_type(memory_name)
        if declared_signal_type is None:
            declared_signal_type = signal_ref.signal_type

        self.parent.ensure_signal_registered(declared_signal_type)

        # Determine latch type based on set_priority
        memory_type = MEMORY_TYPE_SR_LATCH if expr.set_priority else MEMORY_TYPE_RS_LATCH

        # Create IR latch write with inline conditions
        # The set_signal/reset_signal are still set to signal_ref for compatibility,
        # but the layout phase will use set_condition/reset_condition instead
        self.ir_builder.latch_write(
            memory_id,
            latch_value,
            signal_ref,  # For compatibility
            signal_ref,  # For compatibility
            memory_type,
            expr,
            set_condition=set_condition,
            reset_condition=reset_condition,
        )

        return SignalRef(declared_signal_type, memory_id)

    def _lower_latch_write_standard(
        self,
        expr: WriteExpr,
        memory_id: str,
        memory_name: str,
        latch_value: ValueRef,
    ) -> SignalRef:
        """Lower latch write without inlined conditions (fallback path).

        This is the original implementation that creates separate deciders
        for set and reset conditions.
        """
        # Lower set and reset signals
        self.parent.push_expr_context(f"write({memory_name}).set", expr)
        set_ref = self.parent.expr_lowerer.lower_expr(expr.set_signal)
        self.parent.pop_expr_context()

        self.parent.push_expr_context(f"write({memory_name}).reset", expr)
        reset_ref = self.parent.expr_lowerer.lower_expr(expr.reset_signal)
        self.parent.pop_expr_context()

        # CRITICAL: For latches, the output signal MUST be the set signal type
        # so that feedback participates in the S comparison.
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
        needs_multiplier = not isinstance(latch_value, int) or latch_value != 1

        if needs_multiplier and declared_signal_type and declared_signal_type != set_signal_type:
            self.diagnostics.info(
                f"Latch '{memory_name}': latch outputs on '{set_signal_type}' for feedback, "
                f"multiplier converts to '{declared_signal_type}' for memory output.",
                stage="lowering",
                node=expr,
            )

        self.parent.ensure_signal_registered(set_signal_type)
        if declared_signal_type:
            self.parent.ensure_signal_registered(declared_signal_type)

        # Determine latch type based on set_priority
        memory_type = MEMORY_TYPE_SR_LATCH if expr.set_priority else MEMORY_TYPE_RS_LATCH

        # Create IR latch write operation
        self.ir_builder.latch_write(
            memory_id,
            latch_value,
            set_ref,
            reset_ref,
            memory_type,
            expr,
        )

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
                if isinstance(const_node, IRConst) and const_node.value == 1:
                    is_const_one = True

            if not is_const_one and isinstance(write_enable, SignalRef):
                source_node = self.ir_builder.get_operation(write_enable.source_id)
                if isinstance(source_node, IRDecider):
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

            if hasattr(self.parent, "diagnostics") and self.parent.diagnostics:
                self.parent.diagnostics.warning(
                    "Type mismatch in memory write:\n"
                    f"  Expected: '{signal_type}'\n"
                    f"  Got: '{source_type}'\n"
                    f'  Fix: Use projection: value | "{signal_type}"',
                    stage="lowering",
                    node=node,
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
