from __future__ import annotations
from typing import Any, Dict, Optional, Union
from draftsman.data import signals as signal_data  # type: ignore[import-not-found]
from dsl_compiler.src.ir.builder import SignalRef
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from .signal_analyzer import SignalMaterializer, SignalUsageEntry

"""Signal resolution helpers for layout planning."""


class SignalResolver:
    """Resolve logical signal identifiers into Factorio signal names.

    Resolves signal keys to Factorio signal names for blueprint emission.
    Handles inlining and materialization decisions.
    Used during layout and emission.

    See dsl_compiler.src.common.signal_types for the complete architecture overview.
    """

    def __init__(
        self,
        signal_type_map: Dict[str, Any],
        diagnostics: ProgramDiagnostics,
        *,
        materializer: Optional[SignalMaterializer] = None,
        signal_usage: Optional[Dict[str, SignalUsageEntry]] = None,
    ) -> None:
        self.signal_type_map = signal_type_map
        self.diagnostics = diagnostics
        self.materializer = materializer
        self.signal_usage = signal_usage or {}

    def update_context(
        self,
        *,
        materializer: Optional[SignalMaterializer] = None,
        signal_usage: Optional[Dict[str, SignalUsageEntry]] = None,
    ) -> None:
        if materializer is not None:
            self.materializer = materializer
        if signal_usage is not None:
            self.signal_usage = signal_usage

    def get_signal_name(self, operand: Union[str, int, SignalRef, object]) -> str:
        materializer = self.materializer
        usage_index = self.signal_usage or {}

        if materializer:
            entry: Optional[SignalUsageEntry] = None
            operand_key = operand
            if isinstance(operand, SignalRef):
                entry = usage_index.get(operand.source_id)
                operand_key = operand.signal_type
            if isinstance(operand_key, str):
                resolved = materializer.resolve_signal_name(operand_key, entry)
                if resolved:
                    return resolved

        if isinstance(operand, int):
            return "signal-0"

        if hasattr(operand, "signal_type"):
            operand_str = getattr(operand, "signal_type")
        else:
            operand_str = str(operand)

        clean_name = operand_str.split("@")[0]

        mapped_signal = self.signal_type_map.get(clean_name)
        if mapped_signal is not None:
            if isinstance(mapped_signal, dict):
                signal_name = mapped_signal.get("name", clean_name)
                signal_type = mapped_signal.get("type", "virtual")
                if signal_data is not None and signal_name not in signal_data.raw:
                    try:
                        signal_data.add_signal(signal_name, signal_type)
                    except Exception as exc:  # pragma: no cover - draftsman errors
                        self.diagnostics.warning(
                            f"Could not register custom signal '{signal_name}': {exc}"
                        )
                return signal_name
            if signal_data is not None and mapped_signal not in signal_data.raw:
                try:
                    signal_data.add_signal(mapped_signal, "virtual")
                except Exception as exc:  # pragma: no cover - draftsman errors
                    self.diagnostics.warning(
                        f"Could not register signal '{mapped_signal}' as virtual: {exc}"
                    )
            return str(mapped_signal)

        if signal_data is not None and clean_name in signal_data.raw:
            return clean_name

        if clean_name.startswith("__v"):
            try:
                index = int(clean_name[3:])
                letter = chr(ord("A") + (index - 1) % 26)
                return f"signal-{letter}"
            except ValueError:
                pass

        if signal_data is not None and clean_name not in signal_data.raw:
            try:
                signal_data.add_signal(clean_name, "virtual")
            except Exception as exc:  # pragma: no cover - draftsman errors
                self.diagnostics.warning(
                    f"Could not register signal '{clean_name}' as virtual: {exc}"
                )

        return clean_name

    def get_operand_for_combinator(
        self, operand: Union[str, int, SignalRef, object]
    ) -> Union[str, int]:
        if isinstance(operand, int):
            return operand

        if isinstance(operand, SignalRef):
            materializer = self.materializer
            if materializer:
                inlined = materializer.inline_value(operand)
                if inlined is not None:
                    return inlined
                usage_entry = (self.signal_usage or {}).get(operand.source_id)
                resolved = materializer.resolve_signal_name(
                    operand.signal_type, usage_entry
                )
                if resolved is not None:
                    return resolved
            return self.get_signal_name(operand.signal_type)

        if isinstance(operand, str):
            return self.get_signal_name(operand)

        return self.get_signal_name(str(operand))

    def get_operand_value(self, operand: Union[str, int, SignalRef, object]):
        if isinstance(operand, int):
            return operand

        if isinstance(operand, SignalRef):
            materializer = self.materializer
            if materializer:
                inlined = materializer.inline_value(operand)
                if inlined is not None:
                    return inlined
                usage_entry = (self.signal_usage or {}).get(operand.source_id)
                resolved = materializer.resolve_signal_name(
                    operand.signal_type, usage_entry
                )
                if resolved:
                    return resolved
            if hasattr(operand, "signal_type"):
                return self.get_signal_name(operand.signal_type)
            return str(operand)

        if isinstance(operand, str):
            return self.get_signal_name(operand)

        return str(operand)


__all__ = ["SignalResolver"]
