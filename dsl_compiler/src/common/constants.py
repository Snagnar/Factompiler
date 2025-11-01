"""Shared constants across the compiler."""

# Circuit network constraints
MAX_WIRE_SPAN = 9.0
WIRE_RELAY_ENTITY = "medium-electric-pole"

# Signal allocation limits
MAX_IMPLICIT_VIRTUAL_SIGNALS = 26  # A-Z

# Layout defaults
DEFAULT_ENTITY_SPACING = 1
DEFAULT_ROW_HEIGHT = 2

# Power pole types
POWER_POLE_TYPES = {
    "small": "small-electric-pole",
    "medium": "medium-electric-pole",
    "big": "big-electric-pole",
    "substation": "substation",
}

# Wire colors
WIRE_COLORS = ("red", "green")
