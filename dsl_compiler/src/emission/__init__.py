import sys
from pathlib import Path

# Ensure draftsman package is available when importing emission helpers
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent / "factorio-draftsman")
)

from .emitter import (
    BlueprintEmitter,
    emit_blueprint,
    emit_blueprint_string,
    format_entity_description,
)
from .entity_emitter import PlanEntityEmitter

__all__ = [
    # Main API
    "BlueprintEmitter",
    "emit_blueprint",
    "emit_blueprint_string",
    # Subsystems (for advanced use)
    "PlanEntityEmitter",
    "format_entity_description",
]
