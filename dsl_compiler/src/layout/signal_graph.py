"""Signal connectivity graph for layout planning."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set


class SignalGraph:
    """Minimal signal connectivity graph tracking producers and consumers."""

    def __init__(self) -> None:
        self._sources: Dict[str, str] = {}
        self._sinks: Dict[str, List[str]] = defaultdict(list)

    def reset(self) -> None:
        """Clear all tracked producers and consumers."""

        self._sources.clear()
        self._sinks.clear()

    def set_source(self, signal_id: str, entity_id: str) -> None:
        """Mark ``entity_id`` as the producer for ``signal_id``."""

        self._sources[signal_id] = entity_id

    def get_source(self, signal_id: str) -> Optional[str]:
        """Return the producer entity id for ``signal_id`` if known."""

        return self._sources.get(signal_id)

    def add_sink(self, signal_id: str, entity_id: str) -> None:
        """Register that ``entity_id`` consumes ``signal_id``."""

        sinks = self._sinks[signal_id]
        if entity_id not in sinks:
            sinks.append(entity_id)

    def has_sink(self, signal_id: str, entity_id: str) -> bool:
        """Return True if ``entity_id`` already consumes ``signal_id``."""

        return entity_id in self._sinks.get(signal_id, [])

    def iter_sinks(self, signal_id: str) -> List[str]:
        """Return a snapshot of sink ids for ``signal_id``."""

        return list(self._sinks.get(signal_id, []))

    def signals(self) -> Set[str]:
        """Return the set of all tracked signal identifiers."""

        return set(self._sinks.keys()) | set(self._sources.keys())

    def iter_edges(self):
        """Yield triples of (signal_id, source_id, sink_ids)."""

        for signal_id, sinks in self._sinks.items():
            yield signal_id, self._sources.get(signal_id), list(sinks)

    def iter_source_sink_pairs(self):
        """Yield triples of (signal_id, source_id, sink_id)."""

        for signal_id, sinks in self._sinks.items():
            source = self._sources.get(signal_id)
            for sink in sinks:
                yield signal_id, source, sink


__all__ = ["SignalGraph"]
