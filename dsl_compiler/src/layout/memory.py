from typing import Dict

from draftsman.blueprintable import Blueprint
from draftsman.entity import DeciderCombinator

from dsl_compiler.src.layout.layout_plan import EntityPlacement


class MemoryCircuitBuilder:
    """Build orbit-ready memory circuits using the layout engine and Draftsman."""

    def __init__(self, layout_engine, blueprint: Blueprint):
        self.layout = layout_engine
        self.blueprint = blueprint
        self.memory_modules: Dict[str, Dict[str, EntityPlacement]] = {}

    def build_sr_latch(
        self, memory_id: str, signal_type: str
    ) -> Dict[str, EntityPlacement]:
        """Construct the 2-combinator SR latch used for DSL memories."""

        placements: Dict[str, EntityPlacement] = {}

        write_gate = DeciderCombinator()
        write_pos = self.layout.get_next_position(
            footprint=(write_gate.tile_width, write_gate.tile_height)
        )
        write_gate.tile_position = write_pos
        write_gate.conditions = [
            DeciderCombinator.Condition(
                first_signal="signal-W",
                comparator=">",
                constant=0,
                first_signal_networks={"green": True, "red": False},
            )
        ]
        write_output = DeciderCombinator.Output(
            signal=signal_type,
            copy_count_from_input=True,
            networks={"green": False, "red": True},
        )
        write_gate.outputs = [write_output]
        self.blueprint.entities.append(write_gate, copy=False)

        hold_gate = DeciderCombinator()
        hold_pos = self.layout.get_next_position(
            footprint=(hold_gate.tile_width, hold_gate.tile_height)
        )
        hold_gate.tile_position = hold_pos
        hold_gate.conditions = [
            DeciderCombinator.Condition(
                first_signal="signal-W",
                comparator="=",
                constant=0,
                first_signal_networks={"green": True, "red": False},
            )
        ]
        hold_output = DeciderCombinator.Output(
            signal=signal_type,
            copy_count_from_input=True,
            networks={"green": False, "red": True},
        )
        hold_gate.outputs = [hold_output]
        self.blueprint.entities.append(hold_gate, copy=False)

        placements["write_gate"] = EntityPlacement(
            entity=write_gate,
            entity_id=f"{memory_id}_write_gate",
            position=write_pos,
            output_signals={signal_type: "red"},
            input_signals={signal_type: "red", "signal-W": "green"},
        )

        placements["hold_gate"] = EntityPlacement(
            entity=hold_gate,
            entity_id=f"{memory_id}_hold_gate",
            position=hold_pos,
            output_signals={signal_type: "red"},
            input_signals={"signal-W": "green", signal_type: "red"},
        )

        self.memory_modules[memory_id] = placements
        return placements

    def wire_sr_latch(self, memory_id: str) -> None:
        if memory_id not in self.memory_modules:
            return

        module = self.memory_modules[memory_id]
        write_gate = module.get("write_gate")
        hold_gate = module.get("hold_gate")

        try:
            if write_gate and hold_gate:
                self.blueprint.add_circuit_connection(
                    "red",
                    write_gate.entity,
                    hold_gate.entity,
                    side_1="output",
                    side_2="input",
                )

            if hold_gate:
                self.blueprint.add_circuit_connection(
                    "red",
                    hold_gate.entity,
                    hold_gate.entity,
                    side_1="output",
                    side_2="input",
                )

            if write_gate and hold_gate:
                self.blueprint.add_circuit_connection(
                    "green",
                    write_gate.entity,
                    hold_gate.entity,
                    side_1="input",
                    side_2="input",
                )
        except Exception as exc:
            print(f"Warning: failed to wire memory cell {memory_id}: {exc}")


__all__ = ["MemoryCircuitBuilder"]
