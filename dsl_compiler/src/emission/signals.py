from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from draftsman.classes.entity import Entity
from draftsman.data import signals as signal_data

from ..ir import IRValue, IR_Const, SignalRef
from ..semantic import DiagnosticCollector


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


@dataclass
class SignalUsageEntry:
    """Metadata describing how a logical signal is produced and consumed."""

    signal_id: str
    signal_type: Optional[str] = None
    producer: Optional[IRValue] = None
    consumers: Set[str] = field(default_factory=set)
    export_targets: Set[str] = field(default_factory=set)
    output_entities: Set[str] = field(default_factory=set)
    resolved_outputs: Dict[str, str] = field(default_factory=dict)
    source_ast: Optional[Any] = None
    literal_value: Optional[int] = None
    literal_declared_type: Optional[str] = None
    is_typed_literal: bool = False
    debug_label: Optional[str] = None
    debug_metadata: Dict[str, Any] = field(default_factory=dict)
    export_anchor_id: Optional[str] = None
    should_materialize: bool = True
    resolved_signal_name: Optional[str] = None
    resolved_signal_type: Optional[str] = None


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


class SignalMaterializer:
    """Helper for deciding when and how to materialize logical signals."""

    def __init__(
        self,
        signal_usage: Dict[str, SignalUsageEntry],
        signal_type_map: Dict[str, Any],
        diagnostics: DiagnosticCollector,
    ) -> None:
        self.signal_usage = signal_usage
        self.signal_type_map = signal_type_map
        self.diagnostics = diagnostics

    def finalize(self) -> None:
        """Populate materialization decisions and physical signal metadata."""

        for entry in self.signal_usage.values():
            self._decide_materialization(entry)
            self._resolve_signal_identity(entry)

    def should_materialize(self, signal_id: str) -> bool:
        entry = self.signal_usage.get(signal_id)
        if not entry:
            return True
        return entry.should_materialize

    def can_inline_constant(self, signal_ref: SignalRef) -> bool:
        entry = self.signal_usage.get(signal_ref.source_id)
        if not entry:
            return False
        if not isinstance(entry.producer, IR_Const):
            return False
        if entry.should_materialize:
            return False
        return entry.literal_value is not None

    def inline_value(self, signal_ref: SignalRef) -> Optional[int]:
        if self.can_inline_constant(signal_ref):
            entry = self.signal_usage[signal_ref.source_id]
            return entry.literal_value
        return None

    def resolve_signal_name(
        self,
        signal_type: Optional[str],
        entry: Optional[SignalUsageEntry] = None,
    ) -> str:
        lookup_entry = entry
        if lookup_entry is None and signal_type:
            lookup_entry = self.signal_usage.get(signal_type)

        if lookup_entry and lookup_entry.resolved_signal_name:
            return lookup_entry.resolved_signal_name

        return self._resolve_via_mapping(signal_type, lookup_entry)

    def resolve_signal_type(
        self, signal_type: Optional[str], entry: Optional[SignalUsageEntry] = None
    ) -> Optional[str]:
        lookup_entry = entry
        if lookup_entry is None and signal_type:
            lookup_entry = self.signal_usage.get(signal_type)

        if lookup_entry and lookup_entry.resolved_signal_type:
            return lookup_entry.resolved_signal_type

        self._resolve_signal_identity(lookup_entry, force=True)
        if lookup_entry and lookup_entry.resolved_signal_type:
            return lookup_entry.resolved_signal_type
        return None

    def _decide_materialization(self, entry: SignalUsageEntry) -> None:
        producer = entry.producer
        if isinstance(producer, IR_Const):
            named_metadata = False
            if entry.debug_metadata:
                metadata = entry.debug_metadata
                named_metadata = any(
                    key in metadata for key in ("name", "declared_type", "source_ast")
                )

            has_user_label = (
                bool(entry.debug_label) and entry.debug_label != entry.signal_id
            )

            entry.should_materialize = bool(
                entry.is_typed_literal
                or entry.export_targets
                or not entry.consumers
                or named_metadata
                or has_user_label
            )
        else:
            entry.should_materialize = True

    def _resolve_signal_identity(
        self, entry: Optional[SignalUsageEntry], force: bool = False
    ) -> None:
        if not entry:
            return
        if not force and entry.resolved_signal_name:
            return

        candidates: List[str] = []
        if entry.literal_declared_type:
            candidates.append(entry.literal_declared_type)
        if entry.signal_type:
            candidates.append(entry.signal_type)
        if entry.debug_label and entry.debug_label not in candidates:
            candidates.append(entry.debug_label)

        name: Optional[str] = None
        category: Optional[str] = None

        for candidate in candidates:
            if not candidate:
                continue
            mapped = self.signal_type_map.get(candidate)
            if isinstance(mapped, dict):
                name = mapped.get("name", candidate)
                category = mapped.get("type")
                break
            if isinstance(mapped, str):
                name = mapped
                category = self._infer_category_from_name(mapped)
                break

            if signal_data is not None and candidate in signal_data.raw:
                name = candidate
                category = signal_data.raw[candidate].get("type", "virtual")
                if category == "virtual-signal":
                    category = "virtual"
                break

        if name is None and candidates:
            name = candidates[0]
            category = self._infer_category_from_name(name)

        if name is None:
            name = "signal-0"
            category = "virtual"

        entry.resolved_signal_name = name
        entry.resolved_signal_type = category

        # Ensure the resolved signal exists in the Draftsman registry
        if signal_data is not None and name:
            existing = signal_data.raw.get(name)
            target_type = category or "virtual"
            try:
                if existing is None:
                    signal_data.add_signal(name, target_type)
                else:
                    existing_type = (
                        existing.get("type") if isinstance(existing, dict) else None
                    )
                    if (
                        existing_type
                        and target_type == "virtual"
                        and existing_type != "virtual-signal"
                    ):
                        # Avoid clobbering real prototype types with virtual overrides
                        target_type = existing_type
            except Exception as exc:
                self.diagnostics.warning(
                    f"Could not register signal '{name}' as {target_type}: {exc}"
                )

    def _resolve_via_mapping(
        self, signal_type: Optional[str], entry: Optional[SignalUsageEntry]
    ) -> str:
        if entry and entry.resolved_signal_name:
            return entry.resolved_signal_name

        if signal_type in self.signal_type_map:
            mapped = self.signal_type_map[signal_type]
            if isinstance(mapped, dict):
                return mapped.get("name", signal_type)
            if isinstance(mapped, str):
                return mapped

        if signal_type and signal_data is not None and signal_type in signal_data.raw:
            return signal_type

        if (
            signal_type
            and signal_data is not None
            and signal_type not in signal_data.raw
        ):
            try:
                signal_data.add_signal(signal_type, "virtual")
            except Exception as exc:
                self.diagnostics.warning(
                    f"Could not register signal '{signal_type}' as virtual: {exc}"
                )
            return signal_type

        return "signal-0"

    def _infer_category_from_name(self, name: str) -> str:
        if signal_data is not None and name in signal_data.raw:
            proto_type = signal_data.raw[name].get("type", "virtual")
            return "virtual" if proto_type == "virtual-signal" else proto_type
        if name.startswith("signal-"):
            return "virtual"
        return "virtual"


__all__ = [
    "EntityPlacement",
    "SignalUsageEntry",
    "SignalGraph",
    "SignalMaterializer",
]
