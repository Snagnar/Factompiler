"""Semantic analysis package for the Factorio Circuit DSL."""

from dsl_compiler.src.common import SourceLocation
from .analyzer import SemanticAnalyzer, analyze_program, analyze_file
from .exceptions import SemanticError
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

# Compatibility alias
render_source_location = SourceLocation.render

__all__ = [
    "SemanticAnalyzer",
    "analyze_program",
    "analyze_file",
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
]
