"""
Factorio Circuit DSL Compiler - User-Configurable Constants

This module contains all user-tunable settings for the compiler.
Import DEFAULT_CONFIG for default values, or create a custom CompilerConfig.
"""

from dataclasses import dataclass


@dataclass
class CompilerConfig:
    """User-tunable compiler settings.

    Attributes:
        layout_solver_time_limit: Time limit in seconds for the constraint
            solver. Higher values may produce better layouts but increase
            compilation time.

        max_layout_coordinate: Maximum grid size for entity placement.
            Increase for very large circuits that don't fit in default area.

        acceptable_layout_violations: Number of wire span violations
            acceptable before the solver gives up on finding a perfect layout
            and returns the best solution found.

        default_blueprint_label: Default label applied to generated blueprints
            when no explicit name is provided.

        default_blueprint_description: Default description text appended to
            generated blueprints.

        default_power_pole_type: Power pole type used when --power-poles flag
            is specified without an explicit type value.

        wire_span_safety_margin: Safety margin (in tiles) subtracted from
            maximum wire span when placing relay poles. Lower values place
            relays closer to the wire limit, higher values are more
            conservative.
    """

    # Layout Optimization
    layout_solver_time_limit: int = 25
    max_layout_coordinate: int = 200
    acceptable_layout_violations: int = 5

    # Blueprint Output
    default_blueprint_label: str = "DSL Generated"
    default_blueprint_description: str = ""

    # Power Infrastructure
    default_power_pole_type: str = "medium"

    # Wire Routing
    wire_span_safety_margin: float = 1.8


# Global default instance - import this for default behavior
DEFAULT_CONFIG = CompilerConfig()
