from __future__ import annotations
from typing import Any
from dsl_compiler.src.ir.builder import (
    IR_Const,
    SignalRef,
    ValueRef,
)
from dsl_compiler.src.ir.nodes import IR_EntityPropWrite
from dsl_compiler.src.semantic.analyzer import SignalValue
from dsl_compiler.src.semantic.type_system import IntValue
from dsl_compiler.src.ast.statements import (
    AssignStmt,
    DeclStmt,
    ExprStmt,
    FuncDecl,
    ImportStmt,
    MemDecl,
    ReturnStmt,
    Statement,
)
from dsl_compiler.src.ast.expressions import (
    CallExpr,
)
from dsl_compiler.src.ast.literals import (
    Identifier,
    PropertyAccess,
)


class StatementLowerer:
    """Handles lowering of statements to IR operations."""

    def __init__(self, parent: Any) -> None:
        self.parent = parent

    @property
    def ir_builder(self):
        return self.parent.ir_builder

    @property
    def diagnostics(self):
        return self.parent.diagnostics

    @property
    def semantic(self):
        return self.parent.semantic

    def lower_statement(self, stmt: Statement) -> None:
        handlers = {
            DeclStmt: self.lower_decl_stmt,
            AssignStmt: self.lower_assign_stmt,
            MemDecl: self.parent.mem_lowerer.lower_mem_decl,
            ExprStmt: self.lower_expr_stmt,
            ReturnStmt: self.lower_return_stmt,
            FuncDecl: self.lower_func_decl,
            ImportStmt: self.lower_import_stmt,
        }

        handler = handlers.get(type(stmt))
        if handler:
            handler(stmt)  # type: ignore[arg-type]
        else:
            self._error(f"Unknown statement type: {type(stmt)}", stmt)

    def lower_decl_stmt(self, stmt: DeclStmt) -> None:
        if isinstance(stmt.value, CallExpr) and stmt.value.name == "place":
            self.parent.push_expr_context(stmt.name, stmt)
            entity_id, value_ref = (
                self.parent.expr_lowerer.lower_place_call_with_tracking(stmt.value)
            )
            self.parent.pop_expr_context()
            self.parent.entity_refs[stmt.name] = entity_id
            self.parent.signal_refs[stmt.name] = value_ref
            self.parent.annotate_signal_ref(stmt.name, value_ref, stmt)
            return

        if isinstance(stmt.value, CallExpr):
            self.parent.returned_entity_id = None
            self.parent.push_expr_context(stmt.name, stmt)
            value_ref = self.parent.expr_lowerer.lower_expr(stmt.value)
            self.parent.pop_expr_context()

            if self.parent.returned_entity_id is not None:
                self.parent.entity_refs[stmt.name] = self.parent.returned_entity_id
                self.parent.returned_entity_id = None

            self.parent.signal_refs[stmt.name] = value_ref
            self.parent.annotate_signal_ref(stmt.name, value_ref, stmt)
            return

        self.parent.push_expr_context(stmt.name, stmt)
        value_ref = self.parent.expr_lowerer.lower_expr(stmt.value)
        self.parent.pop_expr_context()

        if isinstance(value_ref, SignalRef):
            # Mark user-declared constants to prevent inappropriate suppression
            const_op = self.ir_builder.get_operation(value_ref.source_id)
            if isinstance(const_op, IR_Const):
                const_op.debug_metadata["user_declared"] = True
                # Only set declared_name if not already set (preserve original)
                if "declared_name" not in const_op.debug_metadata:
                    const_op.debug_metadata["declared_name"] = stmt.name
                const_op.debug_label = stmt.name

            if isinstance(const_op, IR_Const) and const_op.debug_metadata.get(
                "folded_from"
            ):
                const_op.debug_label = stmt.name
                if not const_op.debug_metadata.get("name"):
                    const_op.debug_metadata["name"] = stmt.name

            self.parent.signal_refs[stmt.name] = value_ref
            self.parent.annotate_signal_ref(stmt.name, value_ref, stmt)
            return

        if isinstance(value_ref, int):
            symbol = self.semantic.symbol_table.lookup(stmt.name)

            # For int type variables, store the raw integer value.
            # These are compile-time constants that get inlined when used.
            # They do NOT create IR_Const nodes (no combinator).
            if symbol and isinstance(symbol.value_type, IntValue):
                self.parent.signal_refs[stmt.name] = value_ref
                return

            # For Signal type with integer value, create a constant combinator
            if symbol and isinstance(symbol.value_type, SignalValue):
                signal_type = symbol.value_type.signal_type.name
            else:
                signal_type = self.ir_builder.allocate_implicit_type()

            const_ref = self.ir_builder.const(signal_type, value_ref, stmt)

            const_op = self.ir_builder.get_operation(const_ref.source_id)
            if isinstance(const_op, IR_Const):
                const_op.debug_metadata["user_declared"] = True
                const_op.debug_metadata["declared_name"] = stmt.name
                const_op.debug_label = stmt.name

            self.parent.signal_refs[stmt.name] = const_ref
            self.parent.annotate_signal_ref(stmt.name, const_ref, stmt)

    def lower_assign_stmt(self, stmt: AssignStmt) -> None:
        # Determine target name for expression context
        if isinstance(stmt.target, Identifier):
            target_name = stmt.target.name
        elif isinstance(stmt.target, PropertyAccess):
            target_name = f"{stmt.target.object_name}.{stmt.target.property_name}"
        else:
            target_name = None

        if isinstance(stmt.target, Identifier):
            if isinstance(stmt.value, CallExpr) and stmt.value.name == "place":
                self.parent.push_expr_context(target_name, stmt)
                entity_id, value_ref = (
                    self.parent.expr_lowerer.lower_place_call_with_tracking(stmt.value)
                )
                self.parent.pop_expr_context()
                self.parent.entity_refs[stmt.target.name] = entity_id
                self.parent.signal_refs[stmt.target.name] = value_ref
                self.parent.annotate_signal_ref(stmt.target.name, value_ref, stmt)
                return

            if isinstance(stmt.value, CallExpr):
                self.parent.returned_entity_id = None
                self.parent.push_expr_context(target_name, stmt)
                value_ref = self.parent.expr_lowerer.lower_expr(stmt.value)
                self.parent.pop_expr_context()

                if self.parent.returned_entity_id is not None:
                    self.parent.entity_refs[stmt.target.name] = (
                        self.parent.returned_entity_id
                    )
                    self.parent.returned_entity_id = None

                self.parent.signal_refs[stmt.target.name] = value_ref
                self.parent.annotate_signal_ref(stmt.target.name, value_ref, stmt)
                return

        self.parent.push_expr_context(target_name, stmt)
        value_ref = self.parent.expr_lowerer.lower_expr(stmt.value)
        self.parent.pop_expr_context()

        if isinstance(stmt.target, Identifier):
            if isinstance(value_ref, SignalRef):
                const_op = self.ir_builder.get_operation(value_ref.source_id)
                if isinstance(const_op, IR_Const):
                    const_op.debug_metadata["user_declared"] = True
                    # Only set declared_name if not already set (preserve original)
                    if "declared_name" not in const_op.debug_metadata:
                        const_op.debug_metadata["declared_name"] = stmt.target.name
                    const_op.debug_label = stmt.target.name

                if isinstance(const_op, IR_Const) and const_op.debug_metadata.get(
                    "folded_from"
                ):
                    const_op.debug_label = stmt.target.name
                    if not const_op.debug_metadata.get("name"):
                        const_op.debug_metadata["name"] = stmt.target.name

                self.parent.signal_refs[stmt.target.name] = value_ref
                self.parent.annotate_signal_ref(stmt.target.name, value_ref, stmt)
                return

            if isinstance(value_ref, int):
                symbol = self.semantic.symbol_table.lookup(stmt.target.name)
                if symbol and isinstance(symbol.value_type, SignalValue):
                    signal_type = symbol.value_type.signal_type.name
                else:
                    signal_type = self.ir_builder.allocate_implicit_type()

                const_ref = self.ir_builder.const(signal_type, value_ref, stmt)

                const_op = self.ir_builder.get_operation(const_ref.source_id)
                if isinstance(const_op, IR_Const):
                    const_op.debug_metadata["user_declared"] = True
                    const_op.debug_metadata["declared_name"] = stmt.target.name
                    const_op.debug_label = stmt.target.name

                self.parent.signal_refs[stmt.target.name] = const_ref
                self.parent.annotate_signal_ref(stmt.target.name, const_ref, stmt)
                return

        if isinstance(stmt.target, PropertyAccess):
            entity_name = stmt.target.object_name
            prop_name = stmt.target.property_name
            if entity_name in self.parent.entity_refs:
                entity_id = self.parent.entity_refs[entity_name]
                prop_write_op = IR_EntityPropWrite(entity_id, prop_name, value_ref)
                self.ir_builder.add_operation(prop_write_op)

    def lower_expr_stmt(self, stmt: ExprStmt) -> None:
        self.parent.expr_lowerer.lower_expr(stmt.expr)

    def lower_return_stmt(self, stmt: ReturnStmt) -> ValueRef | None:
        if stmt.expr:
            return self.parent.expr_lowerer.lower_expr(stmt.expr)
        return None

    def lower_func_decl(self, stmt: FuncDecl) -> None:
        return None

    def lower_import_stmt(self, stmt: ImportStmt) -> None:
        self._error(
            "Import statement found in AST - file not found during preprocessing: "
            f"{stmt.path}",
            stmt,
        )
