"""Statement lowering utilities for the Factorio Circuit DSL."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dsl_compiler.src.ast import (
    AssignStmt,
    CallExpr,
    DeclStmt,
    ExprStmt,
    FuncDecl,
    Identifier,
    ImportStmt,
    MemDecl,
    PropertyAccess,
    ReturnStmt,
    Statement,
)
from dsl_compiler.src.ir import IR_EntityPropWrite, SignalRef, ValueRef
from dsl_compiler.src.semantic import SignalValue

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .lowerer import ASTLowerer


class StatementLowerer:
    """Handles lowering of statements to IR operations."""

    def __init__(self, parent: "ASTLowerer") -> None:
        self.parent = parent

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def ir_builder(self):
        return self.parent.ir_builder

    @property
    def diagnostics(self):
        return self.parent.diagnostics

    @property
    def semantic(self):
        return self.parent.semantic

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            self.diagnostics.error(f"Unknown statement type: {type(stmt)}", stmt)

    def lower_decl_stmt(self, stmt: DeclStmt) -> None:
        if isinstance(stmt.value, CallExpr) and stmt.value.name == "place":
            entity_id, value_ref = self.parent.expr_lowerer.lower_place_call_with_tracking(stmt.value)
            self.parent.entity_refs[stmt.name] = entity_id
            self.parent.signal_refs[stmt.name] = value_ref
            self.parent.annotate_signal_ref(stmt.name, value_ref, stmt)
            return

        if isinstance(stmt.value, CallExpr):
            self.parent.returned_entity_id = None
            value_ref = self.parent.expr_lowerer.lower_expr(stmt.value)

            if self.parent.returned_entity_id is not None:
                self.parent.entity_refs[stmt.name] = self.parent.returned_entity_id
                self.parent.returned_entity_id = None

            self.parent.signal_refs[stmt.name] = value_ref
            self.parent.annotate_signal_ref(stmt.name, value_ref, stmt)
            return

        value_ref = self.parent.expr_lowerer.lower_expr(stmt.value)

        if isinstance(value_ref, SignalRef):
            self.parent.signal_refs[stmt.name] = value_ref
            self.parent.annotate_signal_ref(stmt.name, value_ref, stmt)
            return

        if isinstance(value_ref, int):
            symbol = self.semantic.symbol_table.lookup(stmt.name)
            if symbol and isinstance(symbol.value_type, SignalValue):
                signal_type = symbol.value_type.signal_type.name
            else:
                signal_type = self.ir_builder.allocate_implicit_type()

            const_ref = self.ir_builder.const(signal_type, value_ref, stmt)
            self.parent.signal_refs[stmt.name] = const_ref
            self.parent.annotate_signal_ref(stmt.name, const_ref, stmt)

    def lower_assign_stmt(self, stmt: AssignStmt) -> None:
        if isinstance(stmt.target, Identifier):
            if isinstance(stmt.value, CallExpr) and stmt.value.name == "place":
                entity_id, value_ref = self.parent.expr_lowerer.lower_place_call_with_tracking(stmt.value)
                self.parent.entity_refs[stmt.target.name] = entity_id
                self.parent.signal_refs[stmt.target.name] = value_ref
                self.parent.annotate_signal_ref(stmt.target.name, value_ref, stmt)
                return

            if isinstance(stmt.value, CallExpr):
                self.parent.returned_entity_id = None
                value_ref = self.parent.expr_lowerer.lower_expr(stmt.value)

                if self.parent.returned_entity_id is not None:
                    self.parent.entity_refs[stmt.target.name] = self.parent.returned_entity_id
                    self.parent.returned_entity_id = None

                self.parent.signal_refs[stmt.target.name] = value_ref
                self.parent.annotate_signal_ref(stmt.target.name, value_ref, stmt)
                return

        value_ref = self.parent.expr_lowerer.lower_expr(stmt.value)

        if isinstance(stmt.target, Identifier):
            if isinstance(value_ref, SignalRef):
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
        # Function declarations are inlined on demand, nothing to emit now.
        return None

    def lower_import_stmt(self, stmt: ImportStmt) -> None:
        self.diagnostics.error(
            "Import statement found in AST - file not found during preprocessing: "
            f"{stmt.path}",
            stmt,
        )
