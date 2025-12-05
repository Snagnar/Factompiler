"""
Test helper utilities for the Factorio Circuit DSL compiler tests.

Provides convenience functions for common test patterns.
"""

from typing import Dict, List, Tuple, Any

from draftsman.blueprintable import Blueprint

from dsl_compiler.src.ast.statements import Program
from dsl_compiler.src.ir.nodes import IRNode
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer
from dsl_compiler.src.lowering.lowerer import ASTLowerer
from dsl_compiler.src.layout.planner import LayoutPlanner
from dsl_compiler.src.emission.emitter import BlueprintEmitter


def lower_program(
    program: Program, semantic_analyzer: SemanticAnalyzer
) -> Tuple[List[IRNode], ProgramDiagnostics, Dict[str, Any]]:
    """Lower a semantic-analyzed program to IR.

    Args:
        program: The AST program node to lower
        semantic_analyzer: Semantic analyzer containing type information and signal registry

    Returns:
        Tuple of (IR operations list, diagnostics, signal type map)
    """
    diagnostics = ProgramDiagnostics()
    lowerer = ASTLowerer(semantic_analyzer, diagnostics)
    ir_operations = lowerer.lower_program(program)
    return ir_operations, lowerer.diagnostics, lowerer.ir_builder.signal_type_map


def emit_blueprint(
    ir_operations: List[IRNode],
    label: str = "DSL Generated",
    signal_type_map: Dict[str, Any] = None,
    *,
    power_pole_type: str = None,
) -> Tuple[Blueprint, ProgramDiagnostics]:
    """Convert IR operations to Factorio blueprint.

    Args:
        ir_operations: List of IR nodes to emit
        label: Label for the blueprint
        signal_type_map: Signal type mappings
        power_pole_type: Optional power pole type to add

    Returns:
        Tuple of (Blueprint, diagnostics)
    """
    signal_type_map = signal_type_map or {}

    emitter_diagnostics = ProgramDiagnostics()
    emitter = BlueprintEmitter(emitter_diagnostics, signal_type_map)

    planner_diagnostics = ProgramDiagnostics()
    planner = LayoutPlanner(
        signal_type_map,
        diagnostics=planner_diagnostics,
        power_pole_type=power_pole_type,
        max_wire_span=9.0,
    )

    layout_plan = planner.plan_layout(
        ir_operations,
        blueprint_label=label,
        blueprint_description="",
    )

    combined_diagnostics = ProgramDiagnostics()
    combined_diagnostics.diagnostics.extend(planner.diagnostics.diagnostics)

    if planner.diagnostics.has_errors():
        return Blueprint(), combined_diagnostics

    blueprint = emitter.emit_from_plan(layout_plan)

    combined_diagnostics.diagnostics.extend(emitter.diagnostics.diagnostics)

    return blueprint, combined_diagnostics


def emit_blueprint_string(
    ir_operations: List[IRNode],
    label: str = "DSL Generated",
    signal_type_map: Dict[str, Any] = None,
    *,
    power_pole_type: str = None,
) -> Tuple[str, ProgramDiagnostics]:
    """Convert IR operations to Factorio blueprint string.

    Args:
        ir_operations: List of IR nodes to emit
        label: Label for the blueprint
        signal_type_map: Signal type mappings
        power_pole_type: Optional power pole type to add

    Returns:
        Tuple of (blueprint string, diagnostics)
    """
    blueprint, diagnostics = emit_blueprint(
        ir_operations,
        label,
        signal_type_map,
        power_pole_type=power_pole_type,
    )

    try:
        blueprint_string = blueprint.to_string()
        return blueprint_string, diagnostics
    except Exception as e:
        diagnostics.error(f"Blueprint string generation failed: {e}")
        return "", diagnostics
