"""Memory-related lowering helpers for the Factorio Circuit DSL."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from dsl_compiler.src.ast import ASTNode, MemDecl, ReadExpr, WriteExpr
from dsl_compiler.src.ir import SignalRef, ValueRef
from dsl_compiler.src.semantic import SignalValue

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .lowerer import ASTLowerer


class MemoryLowerer:
    """Handles lowering of memory declarations and operations."""

    def __init__(self, parent: "ASTLowerer") -> None:
        self.parent = parent

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def ir_builder(self):
        return self.parent.ir_builder

    @property
    def semantic(self):
        return self.parent.semantic

    @property
    def diagnostics(self):
        return self.parent.diagnostics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lower_mem_decl(self, stmt: MemDecl) -> None:
        memory_id = f"mem_{stmt.name}"
        self.parent.memory_refs[stmt.name] = memory_id

        signal_type: Optional[str] = None
        mem_info = getattr(self.semantic, "memory_types", {}).get(stmt.name)

        if mem_info:
            signal_type = mem_info.signal_type
        else:
            symbol = self.semantic.symbol_table.lookup(stmt.name)
            if symbol and isinstance(symbol.value_type, SignalValue):
                signal_type = symbol.value_type.signal_type.name

        if signal_type is None:
            self.diagnostics.error(
                f"Memory '{stmt.name}' must have an explicit signal type.",
                stmt,
            )
            signal_type = "signal-0"

        self.parent.ensure_signal_registered(signal_type)

        self.ir_builder.memory_create(memory_id, signal_type, stmt)

        if stmt.init_expr is not None:
            init_value = self.parent.expr_lowerer.lower_expr(stmt.init_expr)
            coerced_init = self._coerce_to_signal_type(init_value, signal_type, stmt)

            once_enable = self._lower_once_enable(stmt)
            write_op = self.ir_builder.memory_write(
                memory_id, coerced_init, once_enable, stmt
            )
            write_op.is_one_shot = True

    def lower_read_expr(self, expr: ReadExpr) -> SignalRef:
        memory_name = expr.memory_name

        if memory_name not in self.parent.memory_refs:
            self.diagnostics.error(f"Undefined memory: {memory_name}", expr)
            signal_type = self.ir_builder.allocate_implicit_type()
            return self.ir_builder.const(signal_type, 0, expr)

        memory_id = self.parent.memory_refs[memory_name]

        symbol = self.semantic.symbol_table.lookup(memory_name)
        if symbol and isinstance(symbol.value_type, SignalValue):
            signal_type = symbol.value_type.signal_type.name
        else:
            signal_type = self.ir_builder.allocate_implicit_type()

        self.parent.ensure_signal_registered(signal_type)

        return self.ir_builder.memory_read(memory_id, signal_type, expr)

    def lower_write_expr(self, expr: WriteExpr) -> SignalRef:
        memory_name = expr.memory_name

        if memory_name not in self.parent.memory_refs:
            self.diagnostics.error(f"Undefined memory: {memory_name}", expr)
            return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)

        memory_id = self.parent.memory_refs[memory_name]
        data_ref = self.parent.expr_lowerer.lower_expr(expr.value)

        expected_signal_type = self._memory_signal_type(memory_name)
        if expected_signal_type is None:
            self.diagnostics.error(
                f"Memory '{memory_name}' does not have a resolved signal type during lowering.",
                expr,
            )
            expected_signal_type = self.ir_builder.allocate_implicit_type()

        coerced_data_ref = self._coerce_to_signal_type(data_ref, expected_signal_type, expr)

        is_once = getattr(expr, "when_once", False)

        if is_once:
            write_enable = self._lower_once_enable(expr)
        elif expr.when is not None:
            write_enable = self.parent.expr_lowerer.lower_expr(expr.when)
        else:
            self.parent.ensure_signal_registered("signal-W")
            write_enable = self.ir_builder.const("signal-W", 1, expr)

        write_op = self.ir_builder.memory_write(
            memory_id, coerced_data_ref, write_enable, expr
        )

        if is_once:
            write_op.is_one_shot = True

        return coerced_data_ref

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _memory_signal_type(self, memory_name: str) -> Optional[str]:
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

            if getattr(self.semantic, "strict_types", False):
                self.diagnostics.error(
                    "Type mismatch in memory write:\n"
                    f"  Expected: '{signal_type}'\n"
                    f"  Got: '{source_type}'\n"
                    f"  Fix: Use projection: value | \"{signal_type}\"",
                    node,
                )

            return self.ir_builder.arithmetic("+", value_ref, 0, signal_type, node)

        if isinstance(value_ref, int):
            return self.ir_builder.const(signal_type, value_ref, node)

        if hasattr(value_ref, "signal_type") and getattr(value_ref, "signal_type", None) == signal_type:
            return value_ref  # type: ignore[return-value]

        self.diagnostics.error(
            f"Cannot convert value of type '{type(value_ref).__name__}' to signal '{signal_type}' for memory write.",
            node,
        )
        return self.ir_builder.const(signal_type, 0, node)

    def _lower_once_enable(self, node: ASTNode) -> SignalRef:
        self.parent._once_counter += 1
        flag_name = f"__once_flag_{self.parent._once_counter}"
        flag_memory_id = f"mem_{flag_name}"

        flag_signal_type = "signal-W"
        self.parent.ensure_signal_registered(flag_signal_type)

        self.ir_builder.memory_create(flag_memory_id, flag_signal_type, node)

        flag_read = self.ir_builder.memory_read(flag_memory_id, flag_signal_type, node)
        condition = self.ir_builder.decider("==", flag_read, 0, 1, flag_signal_type, node)

        one_const = self.ir_builder.const(flag_signal_type, 1, node)
        flag_write = self.ir_builder.memory_write(
            flag_memory_id, one_const, condition, node
        )
        flag_write.is_one_shot = True

        return condition
