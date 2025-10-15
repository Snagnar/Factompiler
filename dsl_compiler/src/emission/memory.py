from typing import Dict

from draftsman.blueprintable import Blueprint
from draftsman.entity import ArithmeticCombinator, ConstantCombinator, DeciderCombinator

from .signals import EntityPlacement


class MemoryCircuitBuilder:
    """Builds proper memory circuits based on real Factorio designs."""

    def __init__(self, layout_engine, blueprint: Blueprint):
        self.layout = layout_engine
        self.blueprint = blueprint
        self.memory_modules: Dict[str, Dict[str, EntityPlacement]] = {}

    def build_sr_latch(
        self, memory_id: str, signal_type: str, initial_value: int = 0
    ) -> Dict[str, EntityPlacement]:
        """Build the gated memory cell used for DSL memory operations."""

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
            signal="signal-each",
            copy_count_from_input=True,
            networks={"red": True},
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
            signal="signal-each",
            copy_count_from_input=True,
            networks={"red": True},
        )
        hold_gate.outputs = [hold_output]
        self.blueprint.entities.append(hold_gate, copy=False)

        converter = ArithmeticCombinator()
        converter_pos = self.layout.get_next_position(
            footprint=(converter.tile_width, converter.tile_height)
        )
        converter.tile_position = converter_pos
        converter.first_operand = signal_type
        converter.second_operand = 1
        converter.operation = "*"
        converter.output_signal = signal_type
        self.blueprint.entities.append(converter, copy=False)

        init_injector = DeciderCombinator()
        init_injector_pos = self.layout.get_next_position(
            footprint=(init_injector.tile_width, init_injector.tile_height)
        )
        init_injector.tile_position = init_injector_pos
        init_injector.conditions = [
            DeciderCombinator.Condition(
                first_signal="signal-W",
                comparator="=",
                constant=0,
                first_signal_networks={"green": True, "red": False},
            )
        ]
        init_injector.outputs = [
            DeciderCombinator.Output(
                signal="signal-each",
                copy_count_from_input=True,
                networks={"red": True},
            )
        ]
        self.blueprint.entities.append(init_injector, copy=False)

        init_combinator = ConstantCombinator()
        init_pos = self.layout.get_next_position(
            footprint=(init_combinator.tile_width, init_combinator.tile_height)
        )
        init_combinator.tile_position = init_pos
        if initial_value != 0:
            section = init_combinator.add_section()
            section.set_signal(index=0, name=signal_type, count=initial_value)
        else:
            init_combinator.add_section()
        self.blueprint.entities.append(init_combinator, copy=False)

        placements["write_gate"] = EntityPlacement(
            entity=write_gate,
            entity_id=f"{memory_id}_write_gate",
            position=write_pos,
            output_signals={signal_type: "red"},
            input_signals={signal_type: "green", "signal-W": "green"},
        )

        placements["hold_gate"] = EntityPlacement(
            entity=hold_gate,
            entity_id=f"{memory_id}_hold_gate",
            position=hold_pos,
            output_signals={signal_type: "red"},
            input_signals={"signal-W": "green", signal_type: "red"},
        )

        placements["output_converter"] = EntityPlacement(
            entity=converter,
            entity_id=f"{memory_id}_converter",
            position=converter_pos,
            output_signals={signal_type: "red"},
            input_signals={signal_type: "red"},
        )

        placements["init_injector"] = EntityPlacement(
            entity=init_injector,
            entity_id=f"{memory_id}_init_injector",
            position=init_injector_pos,
            output_signals={signal_type: "red"},
            input_signals={signal_type: "red", "signal-W": "green"},
        )

        placements["init_combinator"] = EntityPlacement(
            entity=init_combinator,
            entity_id=f"{memory_id}_init",
            position=init_pos,
            output_signals={signal_type: "red"},
            input_signals={},
        )

        self.memory_modules[memory_id] = placements
        return placements

    def wire_sr_latch(self, memory_id: str) -> None:
        if memory_id not in self.memory_modules:
            return

        module = self.memory_modules[memory_id]
        write_gate = module.get("write_gate")
        hold_gate = module.get("hold_gate")
        converter = module.get("output_converter")
        init_gate = module.get("init_injector")
        init_comb = module.get("init_combinator")

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

            if converter and hold_gate:
                self.blueprint.add_circuit_connection(
                    "red",
                    hold_gate.entity,
                    converter.entity,
                    side_1="output",
                    side_2="input",
                )

            if init_comb and init_gate:
                self.blueprint.add_circuit_connection(
                    "red",
                    init_comb.entity,
                    init_gate.entity,
                    side_1="output",
                    side_2="input",
                )

            if init_gate and hold_gate:
                self.blueprint.add_circuit_connection(
                    "red",
                    init_gate.entity,
                    hold_gate.entity,
                    side_1="output",
                    side_2="input",
                )
                self.blueprint.add_circuit_connection(
                    "green",
                    init_gate.entity,
                    hold_gate.entity,
                    side_1="input",
                    side_2="input",
                )

            if init_gate and write_gate:
                self.blueprint.add_circuit_connection(
                    "green",
                    init_gate.entity,
                    write_gate.entity,
                    side_1="input",
                    side_2="input",
                )

            # Initializer shares enable wiring and injects data while idle
        except Exception as exc:
            print(f"Warning: failed to wire memory cell {memory_id}: {exc}")


__all__ = ["MemoryCircuitBuilder"]
