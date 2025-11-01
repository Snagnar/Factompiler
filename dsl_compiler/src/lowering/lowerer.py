"""
AST to IR lowering pass for the Factorio Circuit DSL.

This module orchestrates the lowering helpers that convert semantic-analyzed
AST nodes into IR operations following the compiler specification.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from dsl_compiler.src.ast import *  # noqa: F401,F403 - re-exported API
from dsl_compiler.src.ir import *  # noqa: F401,F403 - re-exported API
from dsl_compiler.src.semantic import (
    DiagnosticCollector,
    SemanticAnalyzer,
    render_source_location,
)
from draftsman.data import signals as signal_data

from .expression_lowerer import ExpressionLowerer
from .memory_lowerer import MemoryLowerer
from .statement_lowerer import StatementLowerer


class ASTLowerer:
    """Facade that coordinates specialised lowering helpers."""

    def __init__(self, semantic_analyzer: SemanticAnalyzer):
        self.semantic = semantic_analyzer
        self.ir_builder = IRBuilder()
        self.diagnostics = DiagnosticCollector()

        # Symbol tables for IR references
        self.signal_refs: Dict[str, SignalRef] = {}
        self.memory_refs: Dict[str, str] = {}
        self.entity_refs: Dict[str, str] = {}
        self.param_values: Dict[str, ValueRef] = {}

        # Share signal type registry with semantic analyzer
        self.ir_builder.signal_registry = self.semantic.signal_registry

        # Hidden structures for compiler-generated helpers
        self._once_counter = 0
        self.returned_entity_id: Optional[str] = None

        # Modular lowering helpers
        self.mem_lowerer = MemoryLowerer(self)
        self.expr_lowerer = ExpressionLowerer(self)
        self.stmt_lowerer = StatementLowerer(self)

    # ------------------------------------------------------------------
    # Debug metadata helpers
    # ------------------------------------------------------------------

    def annotate_signal_ref(self, name: str, ref: ValueRef, node: ASTNode) -> None:
        """Attach debug metadata for a lowered signal reference."""

        if not isinstance(ref, SignalRef):
            return

        location = render_source_location(node, getattr(node, "source_file", None))
        metadata = {"name": name}
        if location:
            metadata["location"] = location

        semantic_info = None
        if hasattr(self.semantic, "get_signal_debug_payload"):
            semantic_info = self.semantic.get_signal_debug_payload(name)

        source_ast = node
        if semantic_info:
            semantic_payload = semantic_info.as_dict()
            for key, value in semantic_payload.items():
                if value is None:
                    continue
                if key == "source_ast":
                    source_ast = value or source_ast
                    continue
                metadata.setdefault(key, value)

            factorio_signal = semantic_payload.get("factorio_signal")
            if factorio_signal:
                metadata.setdefault("factorio_signal", factorio_signal)
                metadata.setdefault("resolved_signal", factorio_signal)

            signal_key = semantic_payload.get("signal_key")
            if signal_key:
                metadata.setdefault("signal_key", signal_key)

        self.ir_builder.annotate_signal(
            ref, label=name, source_ast=source_ast, metadata=metadata
        )

    def _infer_signal_category(self, signal_type: Optional[str]) -> str:
        """Infer the Factorio signal category for the given identifier."""
        if not signal_type:
            return "virtual"

        if signal_data is not None:
            if signal_type in signal_data.type_of:
                types = signal_data.type_of.get(signal_type, [])
                if types:
                    return types[0]
            if signal_type in signal_data.raw:
                info = signal_data.raw.get(signal_type, {})
                prototype_type = info.get("type")
                if prototype_type == "virtual-signal":
                    return "virtual"
                if prototype_type in {
                    "item",
                    "fluid",
                    "recipe",
                    "entity",
                    "space-location",
                    "asteroid-chunk",
                    "quality",
                    "virtual",
                }:
                    return prototype_type

        mapped = self.ir_builder.signal_type_map.get(signal_type)
        if isinstance(mapped, dict):
            return mapped.get("type", "virtual")
        if isinstance(mapped, str):
            if signal_data is not None and mapped in signal_data.raw:
                prototype_type = signal_data.raw[mapped].get("type", "virtual")
                return (
                    "virtual" if prototype_type == "virtual-signal" else prototype_type
                )
            if mapped.startswith("signal-"):
                return "virtual"

        if signal_type.startswith("__v"):
            return "virtual"

        return "virtual"

    def ensure_signal_registered(
        self, signal_type: Optional[str], source_signal_type: Optional[str] = None
    ) -> None:
        """Ensure that a signal identifier is known to the emitter."""

        if not signal_type:
            return

        if signal_data is not None and signal_type in signal_data.raw:
            return

        if signal_type in self.ir_builder.signal_type_map:
            return

        category = (
            self._infer_signal_category(source_signal_type)
            if source_signal_type
            else None
        )
        if not category:
            category = self._infer_signal_category(signal_type)

        self.ir_builder.signal_type_map[signal_type] = {
            "name": signal_type,
            "type": category or "virtual",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lower_program(self, program: Program) -> List[IRNode]:
        for stmt in program.statements:
            self.lower_statement(stmt)
        return self.ir_builder.get_ir()

    def lower_statement(self, stmt: Statement) -> None:
        self.stmt_lowerer.lower_statement(stmt)

    def lower_decl_stmt(self, stmt: DeclStmt) -> None:
        self.stmt_lowerer.lower_decl_stmt(stmt)

    def lower_assign_stmt(self, stmt: AssignStmt) -> None:
        self.stmt_lowerer.lower_assign_stmt(stmt)

    def lower_mem_decl(self, stmt: MemDecl) -> None:
        self.mem_lowerer.lower_mem_decl(stmt)

    def lower_expr_stmt(self, stmt: ExprStmt) -> None:
        self.stmt_lowerer.lower_expr_stmt(stmt)

    def lower_return_stmt(self, stmt: ReturnStmt) -> Optional[ValueRef]:
        return self.stmt_lowerer.lower_return_stmt(stmt)

    def lower_func_decl(self, stmt: FuncDecl) -> None:
        self.stmt_lowerer.lower_func_decl(stmt)

    def lower_import_stmt(self, stmt: ImportStmt) -> None:
        self.stmt_lowerer.lower_import_stmt(stmt)

    def lower_expr(self, expr: Expr) -> ValueRef:
        return self.expr_lowerer.lower_expr(expr)

    def lower_identifier(self, expr: IdentifierExpr) -> ValueRef:
        return self.expr_lowerer.lower_identifier(expr)

    def lower_binary_op(self, expr: BinaryOp) -> ValueRef:
        return self.expr_lowerer.lower_binary_op(expr)

    def lower_comparison_op(
        self,
        expr: BinaryOp,
        left_ref: ValueRef,
        right_ref: ValueRef,
        output_type: Optional[str] = None,
    ) -> ValueRef:
        return self.expr_lowerer.lower_comparison_op(
            expr, left_ref, right_ref, output_type
        )

    def lower_unary_op(self, expr: UnaryOp) -> ValueRef:
        return self.expr_lowerer.lower_unary_op(expr)

    def lower_signal_literal(self, expr: SignalLiteral) -> SignalRef:
        return self.expr_lowerer.lower_signal_literal(expr)

    def lower_dict_literal(self, expr: DictLiteral) -> Dict[str, object]:
        return self.expr_lowerer.lower_dict_literal(expr)

    def lower_read_expr(self, expr: ReadExpr) -> SignalRef:
        return self.mem_lowerer.lower_read_expr(expr)

    def lower_write_expr(self, expr: WriteExpr) -> SignalRef:
        return self.mem_lowerer.lower_write_expr(expr)

    def lower_projection_expr(self, expr: ProjectionExpr) -> SignalRef:
        return self.expr_lowerer.lower_projection_expr(expr)

    def lower_call_expr(self, expr: CallExpr) -> ValueRef:
        return self.expr_lowerer.lower_call_expr(expr)

    def lower_place_call(self, expr: CallExpr) -> SignalRef:
        return self.expr_lowerer.lower_place_call(expr)

    def lower_place_call_with_tracking(self, expr: CallExpr) -> tuple[str, ValueRef]:
        return self.expr_lowerer.lower_place_call_with_tracking(expr)

    def lower_memory_call(self, expr: CallExpr) -> ValueRef:
        return self.expr_lowerer.lower_memory_call(expr)

    def lower_property_access(self, expr: PropertyAccess) -> ValueRef:
        return self.expr_lowerer.lower_property_access(expr)

    def lower_property_access_expr(self, expr: PropertyAccessExpr) -> ValueRef:
        return self.expr_lowerer.lower_property_access_expr(expr)

    def lower_function_call_inline(self, expr: CallExpr) -> ValueRef:
        return self.expr_lowerer.lower_function_call_inline(expr)


def lower_program(
    program: Program, semantic_analyzer: SemanticAnalyzer
) -> tuple[List[IRNode], DiagnosticCollector, Dict[str, str]]:
    """Lower a semantic-analyzed program to IR.

    Args:
        program: The AST program node to lower
        semantic_analyzer: Semantic analyzer containing type information and signal registry

    Returns:
        Tuple of (IR operations list, diagnostics, signal type map)
    """
    lowerer = ASTLowerer(semantic_analyzer)
    ir_operations = lowerer.lower_program(program)
    return ir_operations, lowerer.diagnostics, lowerer.ir_builder.signal_type_map
