from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from draftsman.classes.entity import Entity


@dataclass
class EntityPlacement:
    """Information about placed entity for wiring."""

    entity: Entity
    entity_id: str
    position: Tuple[int, int]
    output_signals: Dict[str, str]
    input_signals: Dict[str, str]
    role: Optional[str] = None
    zone: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SignalGraph:
    """Minimal signal connectivity graph for emitter wiring decisions."""

    def __init__(self) -> None:
        self._sources: Dict[str, str] = {}
        self._sinks: Dict[str, List[str]] = defaultdict(list)

    def reset(self) -> None:
        self._sources.clear()
        self._sinks.clear()

    def set_source(self, signal_id: str, entity_id: str) -> None:
        self._sources[signal_id] = entity_id

    def get_source(self, signal_id: str) -> Optional[str]:
        return self._sources.get(signal_id)

    def add_sink(self, signal_id: str, entity_id: str) -> None:
        sinks = self._sinks[signal_id]
        if entity_id not in sinks:
            sinks.append(entity_id)

    def has_sink(self, signal_id: str, entity_id: str) -> bool:
        return entity_id in self._sinks.get(signal_id, [])

    def iter_sinks(self, signal_id: str) -> List[str]:
        return list(self._sinks.get(signal_id, []))

    def signals(self) -> Set[str]:
        return set(self._sinks.keys()) | set(self._sources.keys())

    def iter_edges(self):
        for signal_id, sinks in self._sinks.items():
            yield signal_id, self._sources.get(signal_id), list(sinks)

    def iter_source_sink_pairs(self):
        for signal_id, sinks in self._sinks.items():
            source = self._sources.get(signal_id)
            for sink in sinks:
                yield signal_id, source, sink


__all__ = [
    "EntityPlacement",
    "SignalGraph",
]
