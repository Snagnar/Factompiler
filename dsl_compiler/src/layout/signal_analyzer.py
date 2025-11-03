from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from draftsman.data import signals as signal_data  # type: ignore[import-not-found]
from dsl_compiler.src.ast import SignalLiteral
from dsl_compiler.src.common import ProgramDiagnostics
from dsl_compiler.src.ir import (
    IRNode,
    IRValue,
    IR_Const,
    IR_Arith,
    IR_Decider,
    IR_MemCreate,
    IR_MemWrite,
    IR_PlaceEntity,
    IR_EntityPropWrite,
    IR_ConnectToWire,
    IR_WireMerge,
    SignalRef,
)


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


class SignalAnalyzer:
    """Analyzes IR to determine signal usage patterns."""

    def __init__(self, diagnostics: ProgramDiagnostics):
        self.diagnostics = diagnostics
        self.signal_usage: Dict[str, SignalUsageEntry] = {}

    def analyze(self, ir_operations: List[IRNode]) -> Dict[str, SignalUsageEntry]:
        """Analyze IR operations to build a signal usage index."""

        usage: Dict[str, SignalUsageEntry] = {}

        def ensure_entry(
            signal_id: str, signal_type: Optional[str] = None
        ) -> SignalUsageEntry:
            entry = usage.get(signal_id)
            if entry is None:
                entry = SignalUsageEntry(signal_id=signal_id)
                usage[signal_id] = entry
            if signal_type and not entry.signal_type:
                entry.signal_type = signal_type
            return entry

        def record_consumer(ref: Any, consumer_id: str) -> None:
            if isinstance(ref, SignalRef):
                entry = ensure_entry(ref.source_id, ref.signal_type)
                entry.consumers.add(consumer_id)
            elif isinstance(ref, (list, tuple)):
                for item in ref:
                    record_consumer(item, consumer_id)

        def record_export(ref: Any, export_label: str) -> None:
            if isinstance(ref, SignalRef):
                entry = ensure_entry(ref.source_id, ref.signal_type)
                entry.export_targets.add(export_label)

        for op in ir_operations:
            if isinstance(op, IRValue):
                entry = ensure_entry(op.node_id, getattr(op, "output_type", None))
                entry.producer = op
                if op.source_ast and not entry.source_ast:
                    entry.source_ast = op.source_ast
                if getattr(op, "debug_label", None):
                    entry.debug_label = op.debug_label
                elif not entry.debug_label:
                    entry.debug_label = op.node_id
                if getattr(op, "debug_metadata", None):
                    entry.debug_metadata.update(op.debug_metadata)

                if isinstance(op, IR_Const):
                    entry.literal_value = op.value
                    if isinstance(op.source_ast, SignalLiteral):
                        declared_type = getattr(op.source_ast, "signal_type", None)
                        if declared_type:
                            entry.is_typed_literal = True
                            entry.literal_declared_type = declared_type
                        else:
                            entry.literal_declared_type = getattr(
                                op, "output_type", None
                            )

            if isinstance(op, IR_Arith):
                record_consumer(op.left, op.node_id)
                record_consumer(op.right, op.node_id)
            elif isinstance(op, IR_Decider):
                record_consumer(op.left, op.node_id)
                record_consumer(op.right, op.node_id)
                record_consumer(op.output_value, op.node_id)
            elif isinstance(op, IR_MemCreate):
                if hasattr(op, "initial_value") and op.initial_value is not None:
                    record_consumer(op.initial_value, op.node_id)
            elif isinstance(op, IR_MemWrite):
                entry = ensure_entry(op.node_id)
                if not entry.debug_label:
                    entry.debug_label = op.memory_id
                record_consumer(op.data_signal, op.node_id)
                record_consumer(op.write_enable, op.node_id)
            elif isinstance(op, IR_PlaceEntity):
                record_consumer(op.x, op.node_id)
                record_consumer(op.y, op.node_id)
                for prop_value in op.properties.values():
                    record_consumer(prop_value, op.node_id)
            elif isinstance(op, IR_EntityPropWrite):
                record_consumer(op.value, op.node_id)
                record_export(op.value, f"entity:{op.entity_id}.{op.property_name}")
            elif isinstance(op, IR_ConnectToWire):
                record_export(op.signal, f"wire:{op.channel}")
            elif isinstance(op, IR_WireMerge):
                record_consumer(op.sources, op.node_id)

        # Second pass: Mark output signals (top-level, no consumers)
        for signal_id, entry in usage.items():
            if isinstance(entry.producer, IRValue):
                # Check if this is a named signal with no consumers
                if entry.debug_label and entry.debug_label != signal_id:
                    if not entry.consumers:
                        # This is an output signal
                        entry.debug_metadata["is_output"] = True

        self.signal_usage = usage
        return usage


class SignalMaterializer:
    """Decides when and how to materialize logical signals."""

    def __init__(
        self,
        signal_usage: Dict[str, SignalUsageEntry],
        signal_type_map: Dict[str, Any],
        diagnostics: ProgramDiagnostics,
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
        
        # Check suppression flag first (but respect user declarations)
        if entry.debug_metadata.get("suppress_materialization"):
            # If user-declared, override suppression
            if producer and hasattr(producer, "debug_metadata"):
                if producer.debug_metadata.get("user_declared"):
                    entry.should_materialize = True
                    return
            entry.should_materialize = False
            return
        
        # Check producer's metadata for suppression
        if producer and hasattr(producer, "debug_metadata"):
            # User-declared constants always materialize
            if producer.debug_metadata.get("user_declared"):
                entry.should_materialize = True
                return
            
            if producer.debug_metadata.get("suppress_materialization"):
                entry.should_materialize = False
                return

        if isinstance(producer, IR_Const):
            # Check for user declaration via debug_label
            is_user_declared = False
            if hasattr(producer, "debug_metadata"):
                is_user_declared = producer.debug_metadata.get("user_declared", False)
            
            # User-declared constants always materialize
            if is_user_declared:
                entry.should_materialize = True
                return
            
            # Check if this is marked as an output signal
            if entry.debug_metadata.get("is_output"):
                entry.should_materialize = True
                return
            
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
