"""
Shared fixtures and helpers for lowering tests.

This module provides common test utilities used across multiple lowering test files.
"""

from unittest.mock import MagicMock

import pytest

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.common.source_location import SourceLocation
from dsl_compiler.src.ir.builder import IRBuilder
from dsl_compiler.src.ir.nodes import BundleRef, SignalRef
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer


def compile_to_ir(source: str):
    """Helper to compile source to IR operations.

    Returns:
        Tuple of (ir_ops, lowerer, diagnostics)
    """
    parser = DSLParser()
    diagnostics = ProgramDiagnostics()
    analyzer = SemanticAnalyzer(diagnostics)
    program = parser.parse(source)
    analyzer.visit(program)
    lowerer = ASTLowerer(analyzer, diagnostics)
    ir_ops = lowerer.lower_program(program)
    return ir_ops, lowerer, diagnostics


def lower_program(program, semantic_analyzer):
    """Lower a parsed program to IR.

    Returns:
        Tuple of (ir_ops, diagnostics, signal_type_map)
    """
    diagnostics = ProgramDiagnostics()
    lowerer = ASTLowerer(semantic_analyzer, diagnostics)
    ir_ops = lowerer.lower_program(program)
    return ir_ops, lowerer.diagnostics, lowerer.ir_builder.signal_type_map


def make_loc() -> SourceLocation:
    """Create a dummy source location for testing."""
    return SourceLocation("<test>", 1, 0)


def create_mock_parent():
    """Create a mock parent lowerer for direct unit testing.

    Returns:
        Tuple of (parent_mock, ir_builder, diagnostics)
    """
    diagnostics = ProgramDiagnostics()
    ir_builder = IRBuilder()
    semantic = MagicMock()
    semantic.symbol_table = MagicMock()
    semantic.symbol_table.lookup = MagicMock(return_value=None)

    parent = MagicMock()
    parent.param_values = {}
    parent.signal_refs = {}
    parent.entity_refs = {}
    parent.memory_refs = {}
    parent.referenced_signal_names = set()
    parent.get_expr_context = MagicMock(return_value=None)
    parent.push_expr_context = MagicMock()
    parent.pop_expr_context = MagicMock()
    parent.ensure_signal_registered = MagicMock()
    parent.annotate_signal_ref = MagicMock()
    parent.ir_builder = ir_builder
    parent.semantic = semantic
    parent.diagnostics = diagnostics
    parent.returned_entity_id = None
    parent._inlining_stack = set()
    parent.memory_types = {}
    parent.expr_lowerer = MagicMock()

    return parent, ir_builder, diagnostics


@pytest.fixture
def parser():
    """Provide a DSLParser instance."""
    return DSLParser()


@pytest.fixture
def diagnostics():
    """Provide a fresh ProgramDiagnostics instance."""
    return ProgramDiagnostics()


@pytest.fixture
def ir_builder():
    """Provide a fresh IRBuilder instance."""
    return IRBuilder()
