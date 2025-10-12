from typing import Dict

from draftsman.blueprintable import Blueprint

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
        """Build a 3-combinator memory cell that handles negatives and burst signals."""

        from draftsman.entity import ArithmeticCombinator, DeciderCombinator

        placements: Dict[str, EntityPlacement] = {}

        input_pos = self.layout.get_next_position()
        input_combinator = DeciderCombinator(tile_position=input_pos)

        input_condition = DeciderCombinator.Condition(
            first_signal="signal-R",
            comparator="!=",
            constant=0,
        )
        input_combinator.conditions = [input_condition]

        input_output = DeciderCombinator.Output(
            signal="signal-R",
            copy_count_from_input=False,
            constant=1,
        )
        input_combinator.outputs.append(input_output)

        self.blueprint.entities.append(input_combinator, copy=False)

        output_pos = self.layout.get_next_position()
        output_combinator = DeciderCombinator(tile_position=output_pos)

        output_condition = DeciderCombinator.Condition(
            first_signal="signal-R",
            comparator=">",
            constant=0,
        )
        output_combinator.conditions = [output_condition]

        output_output = DeciderCombinator.Output(
            signal="signal-M",
            copy_count_from_input=True,
        )
        output_combinator.outputs.append(output_output)

        self.blueprint.entities.append(output_combinator, copy=False)

        memory_pos = self.layout.get_next_position()
        memory_combinator = DeciderCombinator(tile_position=memory_pos)

        memory_condition = DeciderCombinator.Condition(
            first_signal="signal-R",
            comparator="=",
            constant=0,
        )
        memory_combinator.conditions = [memory_condition]

        memory_output = DeciderCombinator.Output(
            signal="signal-M",
            copy_count_from_input=True,
        )
        memory_combinator.outputs.append(memory_output)

        from draftsman.entity import ConstantCombinator

        init_pos = self.layout.get_next_position()
        init_combinator = ConstantCombinator(tile_position=init_pos)
        section = init_combinator.add_section()
        section.set_signal(index=0, name="signal-M", count=initial_value)
        self.blueprint.entities.append(init_combinator, copy=False)

        placements["init_combinator"] = EntityPlacement(
            entity=init_combinator,
            entity_id=f"{memory_id}_init",
            position=init_pos,
            output_signals={"signal-M": "red"},
            input_signals={},
        )

        self.blueprint.entities.append(memory_combinator, copy=False)

        converter_pos = self.layout.get_next_position()
        converter = ArithmeticCombinator(tile_position=converter_pos)
        converter.first_operand = "signal-M"
        converter.second_operand = 1
        converter.operation = "*"
        converter.output_signal = signal_type
        self.blueprint.entities.append(converter, copy=False)

        placements["input_combinator"] = EntityPlacement(
            entity=input_combinator,
            entity_id=f"{memory_id}_input",
            position=input_pos,
            output_signals={"signal-R": "red"},
            input_signals={signal_type: "green", "signal-R": "red"},
        )

        placements["output_combinator"] = EntityPlacement(
            entity=output_combinator,
            entity_id=f"{memory_id}_output",
            position=output_pos,
            output_signals={"signal-M": "green"},
            input_signals={"signal-R": "red", signal_type: "red"},
        )

        placements["memory_combinator"] = EntityPlacement(
            entity=memory_combinator,
            entity_id=f"{memory_id}_memory",
            position=memory_pos,
            output_signals={"signal-M": "green"},
            input_signals={"signal-R": "red", "signal-M": "green"},
        )

        placements["output_converter"] = EntityPlacement(
            entity=converter,
            entity_id=f"{memory_id}_converter",
            position=converter_pos,
            output_signals={signal_type: "red"},
            input_signals={"signal-M": "green"},
        )

        self.memory_modules[memory_id] = placements
        return placements

    def wire_sr_latch(self, memory_id: str) -> None:
        if memory_id not in self.memory_modules:
            return

        module = self.memory_modules[memory_id]
        input_comb = module.get("input_combinator")
        output_comb = module.get("output_combinator")
        memory_comb = module.get("memory_combinator")
        converter = module.get("output_converter")
        init_comb = module.get("init_combinator")

        try:
            if input_comb and output_comb:
                self.blueprint.add_circuit_connection(
                    "red",
                    input_comb.entity,
                    output_comb.entity,
                    side_1="output",
                    side_2="input",
                )

            if input_comb and memory_comb:
                self.blueprint.add_circuit_connection(
                    "red",
                    input_comb.entity,
                    memory_comb.entity,
                    side_1="output",
                    side_2="input",
                )

            if memory_comb:
                self.blueprint.add_circuit_connection(
                    "green",
                    memory_comb.entity,
                    memory_comb.entity,
                    side_1="output",
                    side_2="input",
                )

            if output_comb and memory_comb:
                self.blueprint.add_circuit_connection(
                    "green",
                    output_comb.entity,
                    memory_comb.entity,
                    side_1="output",
                    side_2="input",
                )

            if converter and memory_comb:
                self.blueprint.add_circuit_connection(
                    "green",
                    memory_comb.entity,
                    converter.entity,
                    side_1="output",
                    side_2="input",
                )

            if converter and output_comb:
                self.blueprint.add_circuit_connection(
                    "green",
                    output_comb.entity,
                    converter.entity,
                    side_1="output",
                    side_2="input",
                )

            if init_comb and memory_comb:
                self.blueprint.add_circuit_connection(
                    "green",
                    init_comb.entity,
                    memory_comb.entity,
                    side_1="output",
                    side_2="input",
                )

            if init_comb and converter:
                self.blueprint.add_circuit_connection(
                    "green",
                    init_comb.entity,
                    converter.entity,
                    side_1="output",
                    side_2="input",
                )
        except Exception as exc:
            print(f"Warning: failed to wire memory cell {memory_id}: {exc}")


__all__ = ["MemoryCircuitBuilder"]
