from __future__ import annotations

from typing import Any

from dsl_compiler.src.ast.expressions import (
    BinaryOp,
    BundleAllExpr,
    BundleAnyExpr,
    CallExpr,
    SignalLiteral,
)
from dsl_compiler.src.ast.literals import (
    Identifier,
    NumberLiteral,
    PropertyAccess,
)
from dsl_compiler.src.ast.statements import (
    AssignStmt,
    DeclStmt,
    ExprStmt,
    ForStmt,
    FuncDecl,
    ImportStmt,
    MemDecl,
    ReturnStmt,
    Statement,
)
from dsl_compiler.src.ir.builder import (
    BundleRef,
    IRConst,
    SignalRef,
    ValueRef,
)
from dsl_compiler.src.ir.nodes import IREntityPropWrite
from dsl_compiler.src.semantic.type_system import IntValue, get_signal_type_name


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
            ForStmt: self.lower_for_stmt,
        }

        handler = handlers.get(type(stmt))
        if handler:
            handler(stmt)  # type: ignore[arg-type]
        else:
            self._error(f"Unknown statement type: {type(stmt)}", stmt)

    def lower_decl_stmt(self, stmt: DeclStmt) -> None:
        if isinstance(stmt.value, CallExpr) and stmt.value.name == "place":
            self.parent.push_expr_context(stmt.name, stmt)
            entity_id, value_ref = self.parent.expr_lowerer.lower_place_call_with_tracking(
                stmt.value
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
            if isinstance(const_op, IRConst):
                const_op.debug_metadata["user_declared"] = True
                # Only set declared_name if not already set (preserve original)
                if "declared_name" not in const_op.debug_metadata:
                    const_op.debug_metadata["declared_name"] = stmt.name
                const_op.debug_label = stmt.name

            if isinstance(const_op, IRConst) and const_op.debug_metadata.get("folded_from"):
                const_op.debug_label = stmt.name
                if not const_op.debug_metadata.get("name"):
                    const_op.debug_metadata["name"] = stmt.name

            self.parent.signal_refs[stmt.name] = value_ref
            self.parent.annotate_signal_ref(stmt.name, value_ref, stmt)
            return

        # Handle Bundle type declarations
        if isinstance(value_ref, BundleRef):
            source_op = self.ir_builder.get_operation(value_ref.source_id)
            if source_op:
                source_op.debug_metadata["user_declared"] = True
                source_op.debug_metadata["declared_name"] = stmt.name
                source_op.debug_label = stmt.name

            self.parent.signal_refs[stmt.name] = value_ref
            return

        if isinstance(value_ref, int):
            symbol = self.semantic.symbol_table.lookup(stmt.name)

            # For int type variables, store the raw integer value.
            # These are compile-time constants that get inlined when used.
            # They do NOT create IRConst nodes (no combinator).
            if symbol and isinstance(symbol.value_type, IntValue):
                self.parent.signal_refs[stmt.name] = value_ref
                return

            # For Signal type with integer value, create a constant combinator
            signal_type = get_signal_type_name(symbol.value_type) if symbol else None
            if signal_type is None:
                signal_type = self.ir_builder.allocate_implicit_type()

            const_ref = self.ir_builder.const(signal_type, value_ref, stmt)

            const_op = self.ir_builder.get_operation(const_ref.source_id)
            if isinstance(const_op, IRConst):
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
                entity_id, value_ref = self.parent.expr_lowerer.lower_place_call_with_tracking(
                    stmt.value
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
                    self.parent.entity_refs[stmt.target.name] = self.parent.returned_entity_id
                    self.parent.returned_entity_id = None

                self.parent.signal_refs[stmt.target.name] = value_ref
                self.parent.annotate_signal_ref(stmt.target.name, value_ref, stmt)
                return

        # Early check for inlinable bundle conditions on entity.enable
        # This avoids creating unnecessary decider combinators
        if isinstance(stmt.target, PropertyAccess):
            entity_name = stmt.target.object_name
            prop_name = stmt.target.property_name
            if (
                entity_name in self.parent.entity_refs
                and prop_name == "enable"
                and self._is_inlinable_bundle_condition(stmt.value)
            ):
                entity_id = self.parent.entity_refs[entity_name]
                self._lower_inlined_bundle_condition(entity_id, stmt.value, stmt, None)
                return

        self.parent.push_expr_context(target_name, stmt)
        value_ref = self.parent.expr_lowerer.lower_expr(stmt.value)
        self.parent.pop_expr_context()

        if isinstance(stmt.target, Identifier):
            # Handle BundleRef (bundle assignments)
            if isinstance(value_ref, BundleRef):
                source_op = self.ir_builder.get_operation(value_ref.source_id)
                if source_op:
                    source_op.debug_metadata["user_declared"] = True
                    if "declared_name" not in source_op.debug_metadata:
                        source_op.debug_metadata["declared_name"] = stmt.target.name
                    source_op.debug_label = stmt.target.name

                self.parent.signal_refs[stmt.target.name] = value_ref
                self.parent.annotate_signal_ref(stmt.target.name, value_ref, stmt)
                return

            if isinstance(value_ref, SignalRef):
                const_op = self.ir_builder.get_operation(value_ref.source_id)
                if isinstance(const_op, IRConst):
                    const_op.debug_metadata["user_declared"] = True
                    # Only set declared_name if not already set (preserve original)
                    if "declared_name" not in const_op.debug_metadata:
                        const_op.debug_metadata["declared_name"] = stmt.target.name
                    const_op.debug_label = stmt.target.name

                if isinstance(const_op, IRConst) and const_op.debug_metadata.get("folded_from"):
                    const_op.debug_label = stmt.target.name
                    if not const_op.debug_metadata.get("name"):
                        const_op.debug_metadata["name"] = stmt.target.name

                self.parent.signal_refs[stmt.target.name] = value_ref
                self.parent.annotate_signal_ref(stmt.target.name, value_ref, stmt)
                return

            if isinstance(value_ref, int):
                symbol = self.semantic.symbol_table.lookup(stmt.target.name)
                signal_type = get_signal_type_name(symbol.value_type) if symbol else None
                if signal_type is None:
                    signal_type = self.ir_builder.allocate_implicit_type()

                const_ref = self.ir_builder.const(signal_type, value_ref, stmt)

                const_op = self.ir_builder.get_operation(const_ref.source_id)
                if isinstance(const_op, IRConst):
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

                # Note: Inlinable bundle conditions are handled early to avoid creating deciders
                prop_write_op = IREntityPropWrite(entity_id, prop_name, value_ref)
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
            f"Import statement found in AST - file not found during preprocessing: {stmt.path}",
            stmt,
        )

    def _resolve_constant(self, name: str) -> int:
        """Resolve a variable name to its compile-time constant integer value.

        Used for resolving for loop range bounds that use variable references.
        """
        if name not in self.parent.signal_refs:
            raise ValueError(f"Variable '{name}' is not defined")
        value = self.parent.signal_refs[name]
        if not isinstance(value, int):
            raise ValueError(
                f"Variable '{name}' must be a compile-time integer constant "
                f"for use in for loop range, got {type(value).__name__}"
            )
        return value

    def lower_for_stmt(self, stmt: ForStmt) -> None:
        """Lower a for loop by unrolling all iterations.

        For loops are compile-time constructs. Each iteration:
        1. Saves current signal_refs state
        2. Registers the iterator as a compile-time integer constant
        3. Lowers all body statements
        4. Restores signal_refs to remove body-scoped variables
        """
        iteration_values = stmt.get_iteration_values(constant_resolver=self._resolve_constant)

        for value in iteration_values:
            # Save current signal_refs state to implement scope isolation
            # We only save the keys, not a deep copy
            saved_signal_refs_keys = set(self.parent.signal_refs.keys())
            saved_entity_refs_keys = set(self.parent.entity_refs.keys())

            # Register the iterator variable as a compile-time integer constant
            # This allows expressions like place("lamp", i, 0) to use the literal value
            self.parent.signal_refs[stmt.iterator_name] = value

            # Lower each statement in the body
            for body_stmt in stmt.body:
                self.lower_statement(body_stmt)

            # Remove variables defined in this iteration (restore scope)
            # Keep only the keys that existed before this iteration
            new_signal_refs = {
                k: v for k, v in self.parent.signal_refs.items() if k in saved_signal_refs_keys
            }
            new_entity_refs = {
                k: v for k, v in self.parent.entity_refs.items() if k in saved_entity_refs_keys
            }
            self.parent.signal_refs = new_signal_refs
            self.parent.entity_refs = new_entity_refs

    def _is_inlinable_bundle_condition(self, expr: Any) -> bool:
        """Check if expression is all(bundle) OP const or any(bundle) OP const.

        These patterns can be inlined directly to entity circuit conditions
        using signal-everything or signal-anything.
        """
        if not isinstance(expr, BinaryOp):
            return False
        if expr.op not in ("<", "<=", ">", ">=", "==", "!="):
            return False
        if not isinstance(expr.left, (BundleAnyExpr, BundleAllExpr)):
            return False
        # Right side must be a constant integer
        return self._is_constant(expr.right)

    def _is_constant(self, expr: Any) -> bool:
        """Check if expression is a compile-time constant."""
        if isinstance(expr, NumberLiteral):
            return True
        if (
            isinstance(expr, SignalLiteral)
            and expr.signal_type is None
            and isinstance(expr.value, NumberLiteral)
        ):
            # SignalLiteral with no type and a NumberLiteral value is effectively a constant
            return True
        if isinstance(expr, Identifier):
            # Check if it's a compile-time constant in symbol table
            symbol = self.semantic.symbol_table.lookup(expr.name)
            if symbol and isinstance(symbol.value_type, IntValue):
                return True
        return False

    def _extract_constant(self, expr: Any) -> int:
        """Extract constant value from an expression."""
        if isinstance(expr, NumberLiteral):
            return expr.value
        if isinstance(expr, SignalLiteral) and isinstance(expr.value, NumberLiteral):
            # SignalLiteral with NumberLiteral value
            return expr.value.value
        if isinstance(expr, Identifier):
            # Try to get value from signal_refs (compile-time constant)
            ref = self.parent.signal_refs.get(expr.name)
            if isinstance(ref, int):
                return ref
            # Fall back to symbol table
            symbol = self.semantic.symbol_table.lookup(expr.name)
            if symbol and hasattr(symbol.value_type, "value"):
                return symbol.value_type.value
        raise ValueError(f"Cannot extract constant from {expr}")

    def _lower_inlined_bundle_condition(
        self, entity_id: str, expr: BinaryOp, stmt: AssignStmt, value_ref: ValueRef
    ) -> None:
        """Inline all(bundle) < N directly to entity condition.

        Instead of creating a decider combinator, we set the entity's circuit
        condition directly to use signal-everything or signal-anything.
        """
        func_name = "all" if isinstance(expr.left, BundleAllExpr) else "any"
        special_signal = "signal-everything" if func_name == "all" else "signal-anything"

        # Lower the bundle argument to get the source connection
        bundle_arg = expr.left.bundle
        bundle_ref = self.parent.expr_lowerer.lower_expr(bundle_arg)

        # Get the constant value
        constant = self._extract_constant(expr.right)

        # Create property write with inline bundle condition metadata
        prop_write_op = IREntityPropWrite(entity_id, "enable", value_ref)
        prop_write_op.inline_bundle_condition = {
            "signal": special_signal,
            "operator": expr.op,
            "constant": constant,
            "input_source": bundle_ref,
        }
        self.ir_builder.add_operation(prop_write_op)
