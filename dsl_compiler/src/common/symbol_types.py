from enum import Enum

"""Constants for symbol types used in symbol table."""


class SymbolType(Enum):
    """Symbol types in the DSL."""

    VARIABLE = "variable"
    MEMORY = "memory"
    FUNCTION = "function"
    PARAMETER = "parameter"
    ENTITY = "entity"
    MODULE = "module"
