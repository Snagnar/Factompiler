"""Expression lowering utilities for the Factorio Circuit DSL."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from dsl_compiler.src.ast import (
    BinaryOp,
    CallExpr,
    DictLiteral,
    Expr,
    IdentifierExpr,
    NumberLiteral,
    ProjectionExpr,
    PropertyAccess,
    PropertyAccessExpr,
    ReadExpr,
    ReturnStmt,
    SignalLiteral,
    StringLiteral,
    UnaryOp,
    WriteExpr,
)
from dsl_compiler.src.ir import (
    IR_Const,
    IR_EntityPropRead,
    IR_WireMerge,
    SignalRef,
    ValueRef,
)
from dsl_compiler.src.semantic import IntValue, SignalValue, ValueInfo

from .constant_folder import ConstantFolder

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .lowerer import ASTLowerer


class ExpressionLowerer:
    """Handles lowering of expressions to IR operations."""

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
    # Core dispatch
    # ------------------------------------------------------------------

    def lower_expr(self, expr: Expr) -> ValueRef:
        """Lower an expression to IR, returning a ValueRef."""

        if isinstance(expr, NumberLiteral):
            return expr.value
        if isinstance(expr, StringLiteral):
            return expr.value

        handlers = {
            IdentifierExpr: self.lower_identifier,
            BinaryOp: self.lower_binary_op,
            UnaryOp: self.lower_unary_op,
            ProjectionExpr: self.lower_projection_expr,
            CallExpr: self.lower_call_expr,
            PropertyAccess: self.lower_property_access,
            PropertyAccessExpr: self.lower_property_access_expr,
            SignalLiteral: self.lower_signal_literal,
            DictLiteral: self.lower_dict_literal,
        }

        if type(expr) in handlers:
            return handlers[type(expr)](expr)  # type: ignore[arg-type]

        if isinstance(expr, ReadExpr):
            return self.parent.mem_lowerer.lower_read_expr(expr)
        if isinstance(expr, WriteExpr):
            return self.parent.mem_lowerer.lower_write_expr(expr)

        self.diagnostics.error(f"Unknown expression type: {type(expr)}", expr)
        return 0

    # ------------------------------------------------------------------
    # Identifier handling
    # ------------------------------------------------------------------

    def lower_identifier(self, expr: IdentifierExpr) -> ValueRef:
        name = expr.name

        if name in self.parent.param_values:
            return self.parent.param_values[name]
        if name in self.parent.signal_refs:
            return self.parent.signal_refs[name]

        self.diagnostics.error(f"Undefined identifier: {name}", expr)
        return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)

    # ------------------------------------------------------------------
    # Binary operations
    # ------------------------------------------------------------------

    def lower_binary_op(self, expr: BinaryOp) -> ValueRef:
        left_type = self.semantic.get_expr_type(expr.left)
        right_type = self.semantic.get_expr_type(expr.right)
        result_type = self.semantic.get_expr_type(expr)

        left_signal_type = (
            left_type.signal_type.name if isinstance(left_type, SignalValue) else None
        )

        if isinstance(result_type, SignalValue):
            output_type = result_type.signal_type.name
        elif isinstance(result_type, IntValue):
            output_type = self.ir_builder.allocate_implicit_type()
        else:
            output_type = self.ir_builder.allocate_implicit_type()

        left_const = ConstantFolder.extract_constant_int(expr.left, self.diagnostics)
        right_const = ConstantFolder.extract_constant_int(expr.right, self.diagnostics)

        if left_const is not None and right_const is not None:
            folded = ConstantFolder.fold_binary_operation(
                expr.op, left_const, right_const, expr, self.diagnostics
            )
            if folded is not None:
                if isinstance(result_type, SignalValue):
                    self.parent.ensure_signal_registered(output_type, left_signal_type)
                    return self.ir_builder.const(output_type, folded, expr)
                return folded

        left_ref = self.lower_expr(expr.left)
        right_ref = self.lower_expr(expr.right)

        merge_candidate = self._attempt_wire_merge(expr, left_ref, right_ref, result_type)
        if merge_candidate is not None:
            return merge_candidate

        self.parent.ensure_signal_registered(output_type, left_signal_type)

        if expr.op in ["+", "-", "*", "/", "%"]:
            return self.ir_builder.arithmetic(expr.op, left_ref, right_ref, output_type, expr)
        if expr.op in ["==", "!=", "<", "<=", ">", ">=", "&&", "||"]:
            return self.lower_comparison_op(expr, left_ref, right_ref, output_type)

        self.diagnostics.error(f"Unknown binary operator: {expr.op}", expr)
        return self.ir_builder.const(output_type, 0, expr)

    def lower_comparison_op(
        self,
        expr: BinaryOp,
        left_ref: ValueRef,
        right_ref: ValueRef,
        output_type: Optional[str] = None,
    ) -> ValueRef:
        if not output_type:
            result_type = self.semantic.get_expr_type(expr)
            if isinstance(result_type, SignalValue):
                output_type = result_type.signal_type.name
            else:
                output_type = self.ir_builder.allocate_implicit_type()

        if expr.op in ["==", "!=", "<", "<=", ">", ">="]:
            return self.ir_builder.decider(expr.op, left_ref, right_ref, 1, output_type, expr)
        if expr.op == "&&":
            return self.ir_builder.arithmetic("*", left_ref, right_ref, output_type, expr)
        if expr.op == "||":
            sum_ref = self.ir_builder.arithmetic("+", left_ref, right_ref, output_type, expr)
            return self.ir_builder.decider(">", sum_ref, 0, 1, output_type, expr)

        self.diagnostics.error(f"Unknown binary operator: {expr.op}", expr)
        return self.ir_builder.const(output_type, 0, expr)

    # ------------------------------------------------------------------
    # Unary operations and projections
    # ------------------------------------------------------------------

    def lower_unary_op(self, expr: UnaryOp) -> ValueRef:
        operand_ref = self.lower_expr(expr.expr)
        result_type = self.semantic.get_expr_type(expr)
        if isinstance(result_type, SignalValue):
            output_type = result_type.signal_type.name
        else:
            output_type = self.ir_builder.allocate_implicit_type()

        operand_signal_type = (
            result_type.signal_type.name if isinstance(result_type, SignalValue) else None
        )
        self.parent.ensure_signal_registered(output_type, operand_signal_type)

        if expr.op == "+":
            return operand_ref
        if expr.op == "-":
            neg_one = self.ir_builder.const(output_type, -1, expr)
            return self.ir_builder.arithmetic("*", operand_ref, neg_one, output_type, expr)
        if expr.op == "!":
            return self.ir_builder.decider("==", operand_ref, 0, 1, output_type, expr)

        self.diagnostics.error(f"Unknown unary operator: {expr.op}", expr)
        return operand_ref

    def lower_projection_expr(self, expr: ProjectionExpr) -> SignalRef:
        source_ref = self.lower_expr(expr.expr)
        target_type = expr.target_type

        if isinstance(source_ref, SignalRef):
            if getattr(source_ref, "signal_type", None) == target_type:
                return source_ref
            self.parent.ensure_signal_registered(target_type, getattr(source_ref, "signal_type", None))
            return self.ir_builder.arithmetic("+", source_ref, 0, target_type, expr)

        if isinstance(source_ref, int):
            self.parent.ensure_signal_registered(target_type)
            return self.ir_builder.const(target_type, source_ref, expr)

        self.diagnostics.error(f"Cannot project {type(source_ref)} to {target_type}", expr)
        self.parent.ensure_signal_registered(target_type)
        return self.ir_builder.const(target_type, 0, expr)

    # ------------------------------------------------------------------
    # Literals and dictionary lowering
    # ------------------------------------------------------------------

    def lower_signal_literal(self, expr: SignalLiteral) -> SignalRef:
        if expr.signal_type is not None:
            signal_name = expr.signal_type
            output_type = expr.signal_type
            self.parent.ensure_signal_registered(signal_name)
            value_ref = self.lower_expr(expr.value)
            if isinstance(value_ref, int):
                ref = self.ir_builder.const(output_type, value_ref, expr)
            else:
                ref = self.ir_builder.const(output_type, 0, expr)
            ref.signal_type = signal_name
            ref.output_type = signal_name
            return ref

        signal_name = self.ir_builder.allocate_implicit_type()
        output_type = signal_name
        self.parent.ensure_signal_registered(signal_name)
        value_ref = self.lower_expr(expr.value)
        if isinstance(value_ref, int):
            ref = self.ir_builder.const(output_type, value_ref, expr)
        else:
            ref = self.ir_builder.const(output_type, 0, expr)
        ref.signal_type = signal_name
        ref.output_type = signal_name
        return ref

    def lower_dict_literal(self, expr: DictLiteral) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        for key, value_expr in expr.entries.items():
            if isinstance(value_expr, NumberLiteral):
                properties[key] = value_expr.value
            elif isinstance(value_expr, StringLiteral):
                properties[key] = value_expr.value
            elif isinstance(value_expr, SignalLiteral):
                inner_value = value_expr.value
                if isinstance(inner_value, NumberLiteral):
                    properties[key] = inner_value.value
                elif isinstance(inner_value, StringLiteral):
                    properties[key] = inner_value.value
                else:
                    lowered = self.lower_expr(inner_value)
                    if isinstance(lowered, (int, str)):
                        properties[key] = lowered
                    else:
                        self.diagnostics.error(
                            f"Unsupported value for property '{key}' in place() call",
                            value_expr,
                        )
            else:
                lowered = self.lower_expr(value_expr)
                if isinstance(lowered, (int, str)):
                    properties[key] = lowered
                else:
                    self.diagnostics.error(
                        f"Unsupported value for property '{key}' in place() call",
                        value_expr,
                    )
        return properties

    # ------------------------------------------------------------------
    # Entity property access
    # ------------------------------------------------------------------

    def lower_property_access(self, expr: PropertyAccess) -> ValueRef:
        entity_name = expr.object_name
        prop_name = expr.property_name

        if entity_name in self.parent.entity_refs:
            entity_id = self.parent.entity_refs[entity_name]
            signal_type = self.ir_builder.allocate_implicit_type()
            read_op = IR_EntityPropRead(f"prop_read_{entity_id}_{prop_name}", signal_type, expr)
            read_op.entity_id = entity_id
            read_op.property_name = prop_name
            self.ir_builder.add_operation(read_op)
            return SignalRef(signal_type, read_op.node_id)

        self.diagnostics.error(f"Undefined entity: {entity_name}", expr)
        return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)

    def lower_property_access_expr(self, expr: PropertyAccessExpr) -> ValueRef:
        entity_name = expr.object_name
        prop_name = expr.property_name

        if entity_name in self.parent.entity_refs:
            entity_id = self.parent.entity_refs[entity_name]
            signal_type = self.ir_builder.allocate_implicit_type()
            read_op = IR_EntityPropRead(f"prop_read_{entity_id}_{prop_name}", signal_type, expr)
            read_op.entity_id = entity_id
            read_op.property_name = prop_name
            self.ir_builder.add_operation(read_op)
            return SignalRef(signal_type, read_op.node_id)

        self.diagnostics.error(f"Undefined entity: {entity_name}", expr)
        return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)

    # ------------------------------------------------------------------
    # Function and special call handling
    # ------------------------------------------------------------------

    def lower_call_expr(self, expr: CallExpr) -> ValueRef:
        if expr.name == "place":
            return self.lower_place_call(expr)
        if expr.name == "memory":
            return self.lower_memory_call(expr)
        return self.lower_function_call_inline(expr)

    def lower_place_call(self, expr: CallExpr) -> SignalRef:
        _, ref = self._lower_place_core(expr)
        return ref

    def lower_place_call_with_tracking(self, expr: CallExpr) -> tuple[str, ValueRef]:
        return self._lower_place_core(expr)

    def lower_memory_call(self, expr: CallExpr) -> ValueRef:
        if not expr.args:
            self.diagnostics.error("memory() requires an initial value", expr)
            return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)
        return self.lower_expr(expr.args[0])

    def lower_function_call_inline(self, expr: CallExpr) -> ValueRef:
        func_symbol = self.semantic.current_scope.lookup(expr.name)
        if (
            not func_symbol
            or func_symbol.symbol_type != "function"
            or not func_symbol.function_def
        ):
            self.diagnostics.error(f"Cannot inline function: {expr.name}", expr)
            return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)

        func_def = func_symbol.function_def

        if len(expr.args) != len(func_def.params):
            self.diagnostics.error(
                f"Function {expr.name} expects {len(func_def.params)} arguments, got {len(expr.args)}",
                expr,
            )
            return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)

        param_values = {
            param_name: self.lower_expr(arg_expr)
            for param_name, arg_expr in zip(func_def.params, expr.args)
        }

        old_param_values = {
            param_name: self.parent.param_values[param_name]
            for param_name in func_def.params
            if param_name in self.parent.param_values
        }

        self.parent.param_values.update(param_values)

        try:
            old_signal_refs = self.parent.signal_refs.copy()
            old_entity_refs = self.parent.entity_refs.copy()

            return_value: Optional[ValueRef] = None
            for stmt in func_def.body:
                if isinstance(stmt, ReturnStmt) and stmt.expr:
                    if isinstance(stmt.expr, IdentifierExpr):
                        var_name = stmt.expr.name
                        if var_name in self.parent.entity_refs:
                            self.parent.returned_entity_id = self.parent.entity_refs[var_name]
                    return_value = self.lower_expr(stmt.expr)
                    break
                else:
                    self.parent.stmt_lowerer.lower_statement(stmt)

            created_entities = {
                name: entity_id
                for name, entity_id in self.parent.entity_refs.items()
                if name not in old_entity_refs
            }

            self.parent.signal_refs = old_signal_refs
            self.parent.entity_refs = old_entity_refs
            self.parent.entity_refs.update(created_entities)

            if return_value is not None:
                return return_value
            return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)
        finally:
            for param_name in func_def.params:
                if param_name in old_param_values:
                    self.parent.param_values[param_name] = old_param_values[param_name]
                else:
                    self.parent.param_values.pop(param_name, None)

    # ------------------------------------------------------------------
    # Wire merge optimisation helpers
    # ------------------------------------------------------------------

    def _is_simple_source_ref(self, value_ref: ValueRef) -> bool:
        if isinstance(value_ref, int):
            return False
        if not isinstance(value_ref, SignalRef):
            return False
        source_op = self.ir_builder.get_operation(value_ref.source_id)
        if source_op is None:
            return False
        if isinstance(source_op, (IR_Const, IR_EntityPropRead, IR_WireMerge)):
            return True
        return False

    def _gather_merge_sources_from_ref(self, value_ref: ValueRef) -> Optional[List[SignalRef]]:
        if not isinstance(value_ref, SignalRef):
            return None

        source_op = self.ir_builder.get_operation(value_ref.source_id)
        if isinstance(source_op, IR_WireMerge):
            return [src for src in source_op.sources if isinstance(src, SignalRef)]
        if self._is_simple_source_ref(value_ref):
            return [value_ref]
        return None

    def _attempt_wire_merge(
        self,
        expr: BinaryOp,
        left_ref: ValueRef,
        right_ref: ValueRef,
        result_type: Optional[ValueInfo],
    ) -> Optional[SignalRef]:
        if expr.op != "+":
            return None
        if not isinstance(result_type, SignalValue):
            return None

        left_sources = self._gather_merge_sources_from_ref(left_ref)
        right_sources = self._gather_merge_sources_from_ref(right_ref)
        if left_sources is None or right_sources is None:
            return None

        combined_sources = left_sources + right_sources
        if len(combined_sources) < 2:
            return None

        output_type = result_type.signal_type.name
        if not output_type:
            return None

        for source in combined_sources:
            if getattr(source, "signal_type", None) != output_type:
                return None

        unique_ids = {source.source_id for source in combined_sources}
        if len(unique_ids) != len(combined_sources):
            return None

        self.parent.ensure_signal_registered(output_type, combined_sources[0].signal_type)

        merge_op_to_reuse: Optional[IR_WireMerge] = None
        reuse_ref: Optional[SignalRef] = None

        if isinstance(left_ref, SignalRef):
            left_source = self.ir_builder.get_operation(left_ref.source_id)
            if isinstance(left_source, IR_WireMerge) and left_source.source_ast is expr.left:
                merge_op_to_reuse = left_source
                reuse_ref = left_ref

        if merge_op_to_reuse is None and isinstance(right_ref, SignalRef):
            right_source = self.ir_builder.get_operation(right_ref.source_id)
            if isinstance(right_source, IR_WireMerge) and right_source.source_ast is expr.right:
                merge_op_to_reuse = right_source
                reuse_ref = right_ref

        if merge_op_to_reuse is not None and reuse_ref is not None:
            merge_op_to_reuse.sources = list(combined_sources)
            merge_op_to_reuse.source_ast = expr
            merge_op_to_reuse.output_type = output_type
            reuse_ref.signal_type = output_type
            self.diagnostics.info(
                f"Optimized addition to wire-merge ({len(combined_sources)} sources)",
                expr,
            )
            return reuse_ref

        merge_ref = self.ir_builder.wire_merge(combined_sources, output_type, expr)
        self.diagnostics.info(
            f"Optimized addition to wire-merge ({len(combined_sources)} sources)",
            expr,
        )
        return merge_ref

    # ------------------------------------------------------------------
    # place() helpers
    # ------------------------------------------------------------------

    def _extract_coordinate(self, coord_expr: Expr) -> Union[int, ValueRef]:
        if isinstance(coord_expr, SignalLiteral) and isinstance(coord_expr.value, NumberLiteral):
            return coord_expr.value.value
        if isinstance(coord_expr, NumberLiteral):
            return coord_expr.value
        return self.lower_expr(coord_expr)

    def _lower_place_core(self, expr: CallExpr) -> tuple[str, SignalRef]:
        if len(expr.args) < 3:
            self.diagnostics.error("place() requires at least 3 arguments: (prototype, x, y)", expr)
            dummy = self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)
            return "error_entity", dummy

        prototype_expr = expr.args[0]
        x_expr = expr.args[1]
        y_expr = expr.args[2]

        if isinstance(prototype_expr, StringLiteral):
            prototype = prototype_expr.value
        else:
            self.diagnostics.error("place() prototype must be a string literal", prototype_expr)
            dummy = self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)
            return "error_entity", dummy

        x_coord = self._extract_coordinate(x_expr)
        y_coord = self._extract_coordinate(y_expr)

        properties: Optional[Dict[str, Any]] = None
        if len(expr.args) >= 4:
            prop_expr = expr.args[3]
            if isinstance(prop_expr, DictLiteral):
                properties = self.lower_dict_literal(prop_expr)
            else:
                self.diagnostics.error(
                    "place() properties argument must be a dictionary literal",
                    prop_expr,
                )

        entity_id = f"entity_{self.ir_builder.next_id()}"
        self.ir_builder.place_entity(
            entity_id,
            prototype,
            x_coord,
            y_coord,
            properties=properties,
            source_ast=expr,
        )

        result_ref = self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)
        const_op = self.ir_builder.get_operation(result_ref.source_id)
        if isinstance(const_op, IR_Const):
            const_op.debug_metadata["suppress_materialization"] = True
        return entity_id, result_ref
