"""Semantic analysis package for the Factorio Circuit DSL."""

from .analyzer import SemanticAnalyzer, analyze_program, analyze_file
from .diagnostics import (
    DiagnosticCollector,
    Diagnostic,
    DiagnosticLevel,
    SemanticError,
)
from .symbol_table import SymbolTable, Symbol
from .type_system import (
    SignalValue,
    IntValue,
    FunctionValue,
    ValueInfo,
    SignalTypeInfo,
    MemoryInfo,
    SignalDebugInfo,
)
from .signal_allocator import SignalAllocator
from .validators import render_source_location, EXPLAIN_MODE

__all__ = [
    "SemanticAnalyzer",
    "analyze_program",
    "analyze_file",
    "DiagnosticCollector",
    "Diagnostic",
    "DiagnosticLevel",
    "SemanticError",
    "SymbolTable",
    "Symbol",
    "SignalValue",
    "IntValue",
    "FunctionValue",
    "ValueInfo",
    "SignalTypeInfo",
    "MemoryInfo",
    "SignalDebugInfo",
    "SignalAllocator",
    "render_source_location",
    "EXPLAIN_MODE",
]
