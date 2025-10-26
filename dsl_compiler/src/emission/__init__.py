import sys
from pathlib import Path

# Ensure draftsman package is available when importing emission helpers
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent / "factorio-draftsman")
)

from .signals import (  # noqa: F401
    EntityPlacement,
    SignalGraph,
    SignalMaterializer,
    SignalUsageEntry,
)
from .layout import LayoutEngine  # noqa: F401
from .memory import MemoryCircuitBuilder  # noqa: F401
from .emitter import (  # noqa: F401
    BlueprintEmitter,
    WireRelayOptions,
    MAX_CIRCUIT_WIRE_SPAN,
    emit_blueprint,
    emit_blueprint_string,
)

__all__ = [
    "EntityPlacement",
    "SignalGraph",
    "SignalMaterializer",
    "SignalUsageEntry",
    "LayoutEngine",
    "MemoryCircuitBuilder",
    "BlueprintEmitter",
    "WireRelayOptions",
    "MAX_CIRCUIT_WIRE_SPAN",
    "emit_blueprint",
    "emit_blueprint_string",
]
