from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Set

"""Signal connectivity graph for layout planning."""


class SignalGraph:
    """Minimal signal connectivity graph tracking producers and consumers.

    ✅ FIX: Now supports multiple sources per signal to handle memory feedback loops
    where both gates (write_gate and hold_gate) output the same signal.
    """

    def __init__(self) -> None:
        self._sources: Dict[str, List[str]] = defaultdict(list)  # ✅ Changed to List
        self._sinks: Dict[str, List[str]] = defaultdict(list)

    def reset(self) -> None:
        """Clear all tracked producers and consumers."""

        self._sources.clear()
        self._sinks.clear()

    def set_source(self, signal_id: str, entity_id: str) -> None:
        """Mark ``entity_id`` as a producer for ``signal_id``.

        ✅ FIX: Now supports multiple sources per signal by appending instead of replacing.
        """
        sources = self._sources[signal_id]
        if entity_id not in sources:
            sources.append(entity_id)

    def get_source(self, signal_id: str) -> Optional[str]:
        """Return the first producer entity id for ``signal_id`` if known.

        ✅ NOTE: For signals with multiple sources (like memory feedback),
        this returns the first one. Use get_sources() for all sources.
        """
        sources = self._sources.get(signal_id, [])
        return sources[0] if sources else None

    def get_sources(self, signal_id: str) -> List[str]:
        """Return all producer entity ids for ``signal_id``.

        ✅ NEW: Added to support multiple sources per signal.
        """
        return list(self._sources.get(signal_id, []))

    def add_sink(self, signal_id: str, entity_id: str) -> None:
        """Register that ``entity_id`` consumes ``signal_id``."""

        sinks = self._sinks[signal_id]
        if entity_id not in sinks:
            sinks.append(entity_id)

    def remove_sink(self, signal_id: str, entity_id: str) -> None:
        """Remove ``entity_id`` from the consumers of ``signal_id``."""

        sinks = self._sinks.get(signal_id, [])
        if entity_id in sinks:
            sinks.remove(entity_id)

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
        """Yield triples of (signal_id, source_ids, sink_ids).

        ✅ FIX: Now yields ALL sources for each signal (list instead of single value).
        """
        for signal_id, sinks in self._sinks.items():
            yield signal_id, self._sources.get(signal_id, []), list(sinks)

    def iter_source_sink_pairs(self):
        """Yield triples of (signal_id, source_id, sink_id).

        ✅ FIX: Now iterates over ALL sources per signal, creating edges for each
        (source, sink) combination. This is necessary for memory feedback loops.
        """
        for signal_id, sinks in self._sinks.items():
            sources = self._sources.get(signal_id, [])
            for source in sources:  # ✅ Iterate over ALL sources
                for sink in sinks:
                    yield signal_id, source, sink


__all__ = ["SignalGraph"]
