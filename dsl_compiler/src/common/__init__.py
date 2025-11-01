from .diagnostics import ProgramDiagnostics, DiagnosticSeverity
from .source_location import SourceLocation
from .signal_registry import SignalTypeRegistry
from .entity_data import EntityDataHelper, get_entity_footprint, get_entity_alignment
from .symbol_types import SymbolType
from .constants import *

"""Common utilities shared across compiler stages."""


__all__ = [
    "ProgramDiagnostics",
    "DiagnosticSeverity",
    "SourceLocation",
    "SignalTypeRegistry",
    "SymbolType",
    "EntityDataHelper",
    "get_entity_footprint",
    "get_entity_alignment",
    # Constants
    "MAX_WIRE_SPAN",
    "MAX_IMPLICIT_VIRTUAL_SIGNALS",
]
