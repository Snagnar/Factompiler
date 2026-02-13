from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from draftsman.data import signals as signal_data  # type: ignore[import-not-found]

from dsl_compiler.src.ast.expressions import SignalLiteral
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.common.signals import (
    AVAILABLE_VIRTUAL_SIGNALS,
    RESERVED_SIGNALS,
    WILDCARD_SIGNALS,
)
from dsl_compiler.src.ir.builder import (
    BundleRef,
    IRArith,
    IRConst,
    IRDecider,
    IRMemCreate,
    IRMemWrite,
    IRNode,
    IRPlaceEntity,
    IRValue,
    IRWireMerge,
    SignalRef,
)
from dsl_compiler.src.ir.nodes import IREntityPropWrite


@dataclass
class SignalUsageEntry:
    """Metadata describing how a logical signal is produced and consumed."""

    signal_id: str
    signal_type: str | None = None
    producer: IRValue | None = None
    consumers: set[str] = field(default_factory=set)
    export_targets: set[str] = field(default_factory=set)
    output_entities: set[str] = field(default_factory=set)
    source_ast: Any | None = None
    literal_value: int | None = None
    literal_declared_type: str | None = None
    is_typed_literal: bool = False
    debug_label: str | None = None
    debug_metadata: dict[str, Any] = field(default_factory=dict)
    should_materialize: bool = True
    resolved_signal_name: str | None = None
    resolved_signal_type: str | None = None
    # Track all variable names that alias to this signal (for output detection)
    alias_names: set[str] = field(default_factory=set)
    # Track which aliases are output-only (not consumed by other code)
    output_aliases: set[str] = field(default_factory=set)


class SignalAnalyzer:
    """Analyzes IR to determine signal usage patterns, materialization decisions, and signal name resolution."""

    def __init__(
        self,
        diagnostics: ProgramDiagnostics,
        signal_type_map: dict[str, Any],
        signal_refs: dict[str, SignalRef] | None = None,
        referenced_signal_names: set[str] | None = None,
    ):
        self.diagnostics = diagnostics
        self.signal_type_map = signal_type_map
        self.signal_refs = signal_refs or {}
        self.referenced_signal_names = referenced_signal_names or set()
        self.signal_usage: dict[str, SignalUsageEntry] = {}
        # Pre-populate allocated signals from existing mappings
        self._allocated_signals: set[str] = set()
        for mapping in signal_type_map.values():
            if isinstance(mapping, dict):
                signal_name = mapping.get("name")
                if signal_name is not None:
                    self._allocated_signals.add(str(signal_name))
            elif isinstance(mapping, str):
                self._allocated_signals.add(mapping)

        # Build pool of available virtual signals for allocation
        self._available_signal_pool = self._build_available_signal_pool()
        self._signal_pool_index = 0
        self._warned_signal_reuse = False

    def analyze(self, ir_operations: list[IRNode]) -> dict[str, SignalUsageEntry]:
        """Analyze IR operations to build a signal usage index."""

        usage: dict[str, SignalUsageEntry] = {}

        # Build a map from node_id to operation for quick lookup
        op_by_id: dict[str, IRNode] = {op.node_id: op for op in ir_operations}

        # Build alias map: source_id -> set of variable names that reference it
        # Skip entries whose producer has suppress_materialization (e.g., entity placeholders)
        source_to_names: dict[str, set[str]] = {}
        for name, ref in self.signal_refs.items():
            if isinstance(ref, SignalRef):
                # Check if this source has suppress_materialization
                producer = op_by_id.get(ref.source_id)
                if (
                    producer
                    and hasattr(producer, "debug_metadata")
                    and producer.debug_metadata.get("suppress_materialization")
                ):
                    continue  # Skip entity placeholders
                if ref.source_id not in source_to_names:
                    source_to_names[ref.source_id] = set()
                source_to_names[ref.source_id].add(name)

        def ensure_entry(signal_id: str, signal_type: str | None = None) -> SignalUsageEntry:
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
            elif isinstance(ref, BundleRef):
                entry = ensure_entry(ref.source_id)
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

                if isinstance(op, IRConst):
                    entry.literal_value = op.value
                    if isinstance(op.source_ast, SignalLiteral):
                        declared_type = getattr(op.source_ast, "signal_type", None)
                        if declared_type:
                            entry.is_typed_literal = True
                            entry.literal_declared_type = declared_type
                        else:
                            entry.literal_declared_type = getattr(op, "output_type", None)

            if isinstance(op, IRArith):
                record_consumer(op.left, op.node_id)
                record_consumer(op.right, op.node_id)
            elif isinstance(op, IRDecider):
                # Legacy single-condition mode
                record_consumer(op.left, op.node_id)
                record_consumer(op.right, op.node_id)
                record_consumer(op.output_value, op.node_id)
                # Multi-condition mode (Factorio 2.0)
                for cond in op.conditions:
                    record_consumer(cond.first_operand, op.node_id)
                    record_consumer(cond.second_operand, op.node_id)
            elif isinstance(op, IRMemCreate):
                if hasattr(op, "initial_value") and op.initial_value is not None:
                    record_consumer(op.initial_value, op.node_id)
                # Exclude memory signal types from auto-allocation pool to
                # prevent collisions between memory cells and auto-allocated signals
                if op.signal_type and op.signal_type not in self._allocated_signals:
                    self._allocated_signals.add(op.signal_type)
                    with suppress(ValueError):
                        self._available_signal_pool.remove(op.signal_type)
            elif isinstance(op, IRMemWrite):
                entry = ensure_entry(op.node_id)
                if not entry.debug_label:
                    entry.debug_label = op.memory_id
                record_consumer(op.data_signal, op.node_id)
                record_consumer(op.write_enable, op.node_id)
            elif isinstance(op, IRPlaceEntity):
                record_consumer(op.x, op.node_id)
                record_consumer(op.y, op.node_id)
                for prop_value in op.properties.values():
                    record_consumer(prop_value, op.node_id)
            elif isinstance(op, IREntityPropWrite):
                record_consumer(op.value, op.node_id)
                record_export(op.value, f"entity:{op.entity_id}.{op.property_name}")
                # Handle inlined bundle conditions - the bundle is consumed by the entity
                if hasattr(op, "inline_bundle_condition") and op.inline_bundle_condition:
                    input_source = op.inline_bundle_condition.get("input_source")
                    if input_source:
                        record_consumer(input_source, op.entity_id)
            elif isinstance(op, IRWireMerge):
                record_consumer(op.sources, op.node_id)

        # Populate alias information and detect output aliases
        # An alias is an "output alias" if it was declared but never referenced
        # (i.e., never used as input to any operation)
        for signal_id, entry in usage.items():
            if signal_id in source_to_names:
                entry.alias_names = source_to_names[signal_id]
                # Find aliases that are NOT referenced (output-only aliases)
                for alias_name in entry.alias_names:
                    if alias_name not in self.referenced_signal_names:
                        entry.output_aliases.add(alias_name)

        for signal_id, entry in usage.items():
            if isinstance(entry.producer, IRValue):
                # Mark as output if:
                # 1. Original behavior: has debug_label, no consumers
                # 2. New behavior: has output-only aliases
                if entry.debug_label and entry.debug_label != signal_id and not entry.consumers:
                    entry.debug_metadata["is_output"] = True
                if entry.output_aliases:
                    entry.debug_metadata["is_output"] = True
                    entry.debug_metadata["output_aliases"] = list(entry.output_aliases)

        self.signal_usage = usage

        self.finalize_materialization()

        return usage

    def finalize_materialization(self) -> None:
        """Populate materialization decisions and physical signal metadata."""
        for entry in self.signal_usage.values():
            self._decide_materialization(entry)
            self._resolve_signal_identity(entry)

    def should_materialize(self, signal_id: str) -> bool:
        """Check if a signal should be materialized as a constant combinator."""
        entry = self.signal_usage.get(signal_id)
        if not entry:
            return True
        return entry.should_materialize

    def can_inline_constant(self, signal_ref: SignalRef) -> bool:
        """Check if a signal reference can be inlined as a literal value."""
        entry = self.signal_usage.get(signal_ref.source_id)
        if not entry:
            return False
        if not isinstance(entry.producer, IRConst):
            return False
        if entry.should_materialize:
            return False
        return entry.literal_value is not None

    def inline_value(self, signal_ref: SignalRef) -> int | None:
        """Get the inlined literal value for a signal reference if possible."""
        if self.can_inline_constant(signal_ref):
            entry = self.signal_usage[signal_ref.source_id]
            return entry.literal_value
        return None

    def resolve_signal_name(
        self,
        signal_type: str | None,
        entry: SignalUsageEntry | None = None,
    ) -> str:
        """Resolve a signal type to a Factorio signal name."""
        # If signal_type is a concrete Factorio signal name (not an implicit __v variable),
        # return it directly without looking up the entry's resolved name.
        # This handles bundle selection like resources["iron-plate"] where signal_type is
        # "iron-plate" but entry might be for the bundle with resolved_signal_name="signal-each".
        if (
            signal_type
            and not signal_type.startswith("__")
            and (signal_type in signal_data.raw or signal_type.startswith("signal-"))
        ):
            return signal_type

        lookup_entry = entry
        if lookup_entry is None and signal_type:
            lookup_entry = self.signal_usage.get(signal_type)

        if lookup_entry and lookup_entry.resolved_signal_name:
            return lookup_entry.resolved_signal_name

        return self._resolve_via_mapping(signal_type, lookup_entry)

    def resolve_signal_type(
        self, signal_type: str | None, entry: SignalUsageEntry | None = None
    ) -> str | None:
        """Resolve a signal type to its category (virtual, item, fluid)."""
        lookup_entry = entry
        if lookup_entry is None and signal_type:
            lookup_entry = self.signal_usage.get(signal_type)

        if lookup_entry and lookup_entry.resolved_signal_type:
            return lookup_entry.resolved_signal_type

        self._resolve_signal_identity(lookup_entry, force=True)
        if lookup_entry and lookup_entry.resolved_signal_type:
            return lookup_entry.resolved_signal_type
        return None

    def get_signal_name(self, operand: Any) -> str:
        """Get Factorio signal name for any operand type.

        This is a high-level API that handles SignalRef, str, int, and other types.
        """
        if isinstance(operand, int):
            return "signal-0"

        if isinstance(operand, SignalRef):
            entry = self.signal_usage.get(operand.source_id)
            return self.resolve_signal_name(operand.signal_type, entry)

        if hasattr(operand, "signal_type"):
            signal_type = operand.signal_type
        else:
            signal_type = str(operand).split("@")[0]

        return self._resolve_via_mapping(signal_type, None)

    def get_operand_for_combinator(self, operand: Any) -> str | int:
        """Resolve an operand for combinator use (inline constants or signal names)."""
        if isinstance(operand, int):
            return operand

        if isinstance(operand, SignalRef):
            inlined = self.inline_value(operand)
            if inlined is not None:
                return inlined
            usage_entry = self.signal_usage.get(operand.source_id)
            resolved = self.resolve_signal_name(operand.signal_type, usage_entry)
            if resolved is not None:
                return resolved
            return self.get_signal_name(operand.signal_type)

        if isinstance(operand, str):
            return self.get_signal_name(operand)

        return self.get_signal_name(str(operand))

    def _decide_materialization(self, entry: SignalUsageEntry) -> None:
        producer = entry.producer

        # Check suppression flag first (but respect user declarations)
        if entry.debug_metadata.get("suppress_materialization"):
            if (
                producer
                and hasattr(producer, "debug_metadata")
                and producer.debug_metadata.get("user_declared")
            ):
                entry.should_materialize = True
                return
            entry.should_materialize = False
            return

        # Check producer's metadata for suppression
        if producer and hasattr(producer, "debug_metadata"):
            if producer.debug_metadata.get("user_declared"):
                entry.should_materialize = True
                return

            if producer.debug_metadata.get("suppress_materialization"):
                entry.should_materialize = False
                return

        if isinstance(producer, IRConst):
            # Check for user declaration via debug_label
            is_user_declared = False
            if hasattr(producer, "debug_metadata"):
                is_user_declared = producer.debug_metadata.get("user_declared", False)

            if is_user_declared:
                entry.should_materialize = True
                return

            if entry.debug_metadata.get("is_output"):
                entry.should_materialize = True
                return

            named_metadata = False
            if entry.debug_metadata:
                metadata = entry.debug_metadata
                named_metadata = any(
                    key in metadata for key in ("name", "declared_type", "source_ast")
                )

            has_user_label = bool(entry.debug_label) and entry.debug_label != entry.signal_id

            entry.should_materialize = bool(
                entry.is_typed_literal
                or entry.export_targets
                or not entry.consumers
                or named_metadata
                or has_user_label
            )
        else:
            entry.should_materialize = True

    def _resolve_signal_identity(self, entry: SignalUsageEntry | None, force: bool = False) -> None:
        if not entry:
            return
        if not force and entry.resolved_signal_name:
            return

        candidates: list[str] = []
        if entry.literal_declared_type:
            candidates.append(entry.literal_declared_type)
        if entry.signal_type:
            candidates.append(entry.signal_type)
        if entry.debug_label and entry.debug_label not in candidates:
            candidates.append(entry.debug_label)

        name: str | None = None
        category: str | None = None

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

            if candidate in signal_data.raw:
                name = candidate
                category = signal_data.raw[candidate].get("type", "virtual")
                if category == "virtual-signal":
                    category = "virtual"
                break

        # Handle unmapped implicit signals - allocate fresh Factorio signals
        if name is None and candidates:
            first_candidate = candidates[0]
            if first_candidate and first_candidate.startswith("__v"):
                factorio_signal = self._allocate_factorio_virtual_signal()
                self.signal_type_map[first_candidate] = {
                    "name": factorio_signal,
                    "type": "virtual",
                }
                name = factorio_signal
                category = "virtual"
            else:
                name = first_candidate
                category = self._infer_category_from_name(name)

        if name is None:
            name = "signal-0"
            category = "virtual"

        entry.resolved_signal_name = name
        entry.resolved_signal_type = category

        if name:
            existing = signal_data.raw.get(name)
            target_type = category or "virtual"
            try:
                if existing is None:
                    signal_data.add_signal(name, target_type)
                else:
                    existing_type = existing.get("type") if isinstance(existing, dict) else None
                    if (
                        existing_type
                        and target_type == "virtual"
                        and existing_type != "virtual-signal"
                    ):
                        target_type = existing_type
            except Exception as exc:
                self.diagnostics.info(f"Could not register signal '{name}' as {target_type}: {exc}")

    def _resolve_via_mapping(self, signal_type: str | None, entry: SignalUsageEntry | None) -> str:
        if entry and entry.resolved_signal_name:
            return entry.resolved_signal_name

        if not signal_type:
            return "signal-0"

        if signal_type in self.signal_type_map:
            mapped = self.signal_type_map[signal_type]
            if isinstance(mapped, dict):
                return mapped.get("name", signal_type)
            if isinstance(mapped, str):
                return mapped

        if signal_type in signal_data.raw:
            return signal_type

        # Handle unmapped implicit signals
        if signal_type.startswith("__v"):
            factorio_signal = self._allocate_factorio_virtual_signal()
            self.signal_type_map[signal_type] = {
                "name": factorio_signal,
                "type": "virtual",
            }
            return factorio_signal

        # Register as virtual if not known
        with suppress(ValueError):
            # Already registered
            signal_data.add_signal(signal_type, "virtual")
        return signal_type

    def _infer_category_from_name(self, name: str) -> str:
        if name in signal_data.raw:
            proto_type = signal_data.raw[name].get("type", "virtual")
            return "virtual" if proto_type == "virtual-signal" else proto_type
        if name.startswith("signal-"):
            return "virtual"
        return "virtual"

    def _build_available_signal_pool(self) -> list[str]:
        """
        Build pool of signals available for implicit allocation.

        Excludes:
        - Signals already used in signal_type_map
        - Signals explicitly referenced by user code
        - Reserved compiler signals
        - Wildcard signals
        """
        excluded: set[str] = set()

        # Add reserved and wildcard signals
        excluded.update(RESERVED_SIGNALS)
        excluded.update(WILDCARD_SIGNALS)

        # Add signals already allocated or mapped
        excluded.update(self._allocated_signals)

        # Add user-referenced signals (from program source)
        excluded.update(self.referenced_signal_names)

        # Build the pool excluding used signals
        return [s for s in AVAILABLE_VIRTUAL_SIGNALS if s not in excluded]

    def _allocate_factorio_virtual_signal(self) -> str:
        """
        Allocate a fresh Factorio virtual signal from the available pool.

        Uses signals in priority order (letters, digits, colors, then symbols).
        When exhausted, warns once and starts reusing from the beginning.
        """
        if not self._available_signal_pool:
            # No signals available at all - fall back to signal-0
            self.diagnostics.info("No virtual signals available for allocation. Using signal-0.")
            return "signal-0"

        if self._signal_pool_index >= len(self._available_signal_pool):
            if not self._warned_signal_reuse:
                self.diagnostics.warning(
                    f"Exhausted all {len(self._available_signal_pool)} available virtual signals. "
                    "Some internal signals will be reused - this may cause signal collisions."
                )
                self._warned_signal_reuse = True
            self._signal_pool_index = 0

        signal_name = self._available_signal_pool[self._signal_pool_index]
        self._signal_pool_index += 1
        self._allocated_signals.add(signal_name)
        return signal_name
