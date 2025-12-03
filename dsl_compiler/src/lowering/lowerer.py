"""
AST to IR lowering pass for the Factorio Circuit DSL.

This module orchestrates the lowering helpers that convert semantic-analyzed
AST nodes into IR operations following the compiler specification.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from dsl_compiler.src.ast.statements import (
    ASTNode,
    Statement,
    Program,
)
from dsl_compiler.src.ir.nodes import IRNode, ValueRef, SignalRef
from dsl_compiler.src.ir.builder import IRBuilder
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer
from dsl_compiler.src.common.source_location import SourceLocation
from draftsman.data import signals as signal_data

from .expression_lowerer import ExpressionLowerer
from .memory_lowerer import MemoryLowerer
from .statement_lowerer import StatementLowerer


class ASTLowerer:
    """Facade that coordinates specialised lowering helpers."""

    def __init__(
        self, semantic_analyzer: SemanticAnalyzer, diagnostics: ProgramDiagnostics
    ):
        self.semantic = semantic_analyzer
        self.ir_builder = IRBuilder()
        self.diagnostics = diagnostics
        self.diagnostics.default_stage = "lowering"

        self.signal_refs: Dict[str, SignalRef] = {}
        self.memory_refs: Dict[str, str] = {}
        self.memory_types: Dict[
            str, str
        ] = {}  # Track memory signal types during lowering
        self.entity_refs: Dict[str, str] = {}
        self.param_values: Dict[str, ValueRef] = {}

        self.ir_builder.signal_registry = self.semantic.signal_registry

        self.returned_entity_id: Optional[str] = None

        self.mem_lowerer = MemoryLowerer(self)
        self.expr_lowerer = ExpressionLowerer(self)
        self.stmt_lowerer = StatementLowerer(self)

    def _error(self, message: str, node: Optional[ASTNode] = None) -> None:
        """Add a lowering error diagnostic."""
        self.diagnostics.error(message, stage="lowering", node=node)

    def annotate_signal_ref(self, name: str, ref: SignalRef, node: ASTNode) -> None:
        """Attach debug metadata for a lowered signal reference."""

        if not isinstance(ref, SignalRef):
            return

        location = SourceLocation.render(node, getattr(node, "source_file", None))
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
            if mapped in signal_data.raw:
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

        # Don't register implicit signals (__v1, __v2, etc.) here - let the layout
        # phase allocate real Factorio signals for them when needed
        if signal_type.startswith("__v"):
            return

        if signal_type in signal_data.raw:
            return

        if self.ir_builder.signal_registry.resolve(signal_type) is not None:
            return

        category = (
            self._infer_signal_category(source_signal_type)
            if source_signal_type
            else None
        )
        if not category:
            category = self._infer_signal_category(signal_type)

        self.ir_builder.signal_registry.register(
            signal_type, signal_type, category or "virtual"
        )

    def lower_program(self, program: Program) -> List[IRNode]:
        for stmt in program.statements:
            self.lower_statement(stmt)
        return self.ir_builder.get_ir()

    def lower_statement(self, stmt: Statement) -> None:
        self.stmt_lowerer.lower_statement(stmt)
