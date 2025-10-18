"""
Tests for emit.py - Blueprint emission functionality.
"""

import math
from typing import Tuple

import pytest
from dsl_compiler.src.parser import DSLParser
from dsl_compiler.src.semantic import SemanticAnalyzer, analyze_program
from dsl_compiler.src.lowerer import lower_program
from dsl_compiler.src.emit import (
    BlueprintEmitter,
    MAX_CIRCUIT_WIRE_SPAN,
    WireRelayOptions,
)
from draftsman.entity import new_entity  # type: ignore[import-not-found]
from dsl_compiler.src.emission.signals import EntityPlacement, SignalUsageEntry
from dsl_compiler.src.emission.debug_format import format_entity_description


def _make_far_endpoints(
    emitter: BlueprintEmitter,
    *,
    sink_target: Tuple[int, int] = (40, 0),
    sink_prototype: str = "arithmetic-combinator",
) -> Tuple[EntityPlacement, EntityPlacement]:
    source_entity = new_entity("constant-combinator")
    source_pos = emitter.layout.get_next_position()
    source_entity.tile_position = source_pos
    source_entity = emitter._add_entity(source_entity)
    source = EntityPlacement(
        entity=source_entity,
        entity_id="source",
        position=source_pos,
        output_signals={},
        input_signals={},
    )
    emitter.entities["source"] = source

    sink_entity = new_entity(sink_prototype)
    sink_pos = emitter.layout.reserve_near(sink_target, max_radius=200)
    sink_entity.tile_position = sink_pos
    sink_entity = emitter._add_entity(sink_entity)
    sink = EntityPlacement(
        entity=sink_entity,
        entity_id="sink",
        position=sink_pos,
        output_signals={},
        input_signals={},
    )
    emitter.entities["sink"] = sink

    return source, sink


class TestBlueprintEmitter:
    def test_entity_descriptions_include_names_and_locations(self, parser, analyzer):
        """Test that entity descriptions contain variable names and source line numbers when available."""
        import os

        file_path = "tests/sample_programs/01_basic_arithmetic.fcdsl"
        assert os.path.exists(file_path)
        with open(file_path, "r") as f:
            code = f.read()
        program = parser.parse(code, filename=file_path)
        analyze_program(program, strict_types=False, analyzer=analyzer)
        ir_operations, _, signal_map = lower_program(program, analyzer)
        emitter = BlueprintEmitter(signal_type_map=signal_map)
        blueprint = emitter.emit_blueprint(ir_operations)

        # Check entity descriptions
        described_entities = [
            e
            for e in emitter.entities.values()
            if getattr(e.entity, "player_description", "")
        ]
        assert described_entities, (
            "At least one entity should expose player_description"
        )

        descriptions = [e.entity.player_description for e in described_entities]

        assert any("signal" in desc for desc in descriptions), (
            "Descriptions should include 'signal' keyword"
        )
        assert any("(" in desc for desc in descriptions), (
            "Descriptions should include resolved signal hints when available"
        )

        expected_names = {
            "a",
            "b",
            "c",
            "sum",
            "diff",
            "product",
            "quotient",
            "remainder",
            "output_val",
        }
        assert any(
            any(name in desc for name in expected_names) for desc in descriptions
        ), "At least one description should contain a variable name"

        assert any("@ 01_basic_arithmetic.fcdsl:" in desc for desc in descriptions), (
            "Descriptions should include file and line information"
        )

    @pytest.mark.parametrize(
        "sample_file",
        [
            "tests/sample_programs/02_mixed_types.fcdsl",
            "tests/sample_programs/03_bundles.fcdsl",
        ],
    )
    def test_chained_arithmetic_no_extra_constants(self, parser, analyzer, sample_file):
        """Regression: Chained arithmetic should not emit extra constants when reusing outputs."""
        import os

        assert os.path.exists(sample_file)
        with open(sample_file, "r") as f:
            code = f.read()
        program = parser.parse(code)
        analyze_program(program, strict_types=False, analyzer=analyzer)
        ir_operations, _, signal_map = lower_program(program, analyzer)
        emitter = BlueprintEmitter(signal_type_map=signal_map)
        blueprint = emitter.emit_blueprint(ir_operations)

        # Count constant combinators
        constant_combinators = [
            e
            for e in emitter.entities.values()
            if hasattr(e.entity, "name") and e.entity.name == "constant-combinator"
        ]
        # Should not exceed number of explicit literals and export anchors
        explicit_literals = [
            line for line in code.splitlines() if "Signal" in line and '("' in line
        ]
        # Allow for anchors (blank combinators)
        anchor_combinators = []
        for e in constant_combinators:
            is_anchor = True
            for sec in getattr(e.entity, "sections", []):
                filters = getattr(sec, "filters", None)
                if filters and any(
                    getattr(s, "count", 0) != 0 for s in filters.values()
                ):
                    is_anchor = False
            if is_anchor:
                anchor_combinators.append(e)
        # The number of non-anchor constant combinators should match explicit literals
        non_anchor_constants = [
            c for c in constant_combinators if c not in anchor_combinators
        ]
        assert len(non_anchor_constants) == len(explicit_literals), (
            f"Should only emit constants for explicit literals: found {len(non_anchor_constants)}, expected {len(explicit_literals)}"
        )

    def test_basic_arithmetic_literal_and_export_anchor(self, parser, analyzer):
        """Regression: 01_basic_arithmetic.fcdsl should emit one combinator for typed literal and blank anchor for exported signal."""
        import os

        file_path = "tests/sample_programs/01_basic_arithmetic.fcdsl"
        assert os.path.exists(file_path)
        with open(file_path, "r") as f:
            code = f.read()
        program = parser.parse(code)
        analyze_program(program, strict_types=False, analyzer=analyzer)
        ir_operations, _, signal_map = lower_program(program, analyzer)
        emitter = BlueprintEmitter(signal_type_map=signal_map)
        blueprint = emitter.emit_blueprint(ir_operations)

        # Find constant combinators for explicit typed literal (iron-plate)
        iron_plate_combinators = []
        for e in emitter.entities.values():
            if hasattr(e.entity, "name") and e.entity.name == "constant-combinator":
                for sec in getattr(e.entity, "sections", []):
                    filters = getattr(sec, "filters", None)
                    if filters:
                        for s in filters.values():
                            if (
                                getattr(s, "name", None) == "iron-plate"
                                and getattr(s, "count", 0) == 50
                            ):
                                iron_plate_combinators.append(e)
        assert len(iron_plate_combinators) == 1, (
            "Should emit exactly one combinator for typed literal 'c'"
        )

        # Check for blank anchor combinator for exported signal (output_val)
        anchor_combinators = []
        for e in emitter.entities.values():
            if hasattr(e.entity, "name") and e.entity.name == "constant-combinator":
                is_anchor = True
                for sec in getattr(e.entity, "sections", []):
                    filters = getattr(sec, "filters", None)
                    if filters and any(
                        getattr(s, "count", 0) != 0 for s in filters.values()
                    ):
                        is_anchor = False
                if is_anchor:
                    anchor_combinators.append(e)
        assert len(anchor_combinators) >= 1, (
            "Should emit at least one blank anchor for exported signal"
        )

    def test_constant_roles_align_with_zones(self, parser, analyzer):
        """Literals should occupy the north edge, export anchors the south edge."""

        code = """
        Signal iron = ("iron-plate", 25);
        Signal copper = ("copper-plate", 50);
        Signal total = iron + copper;
        """

        program = parser.parse(code)
        analyze_program(program, strict_types=False, analyzer=analyzer)
        ir_operations, _, signal_map = lower_program(program, analyzer)

        emitter = BlueprintEmitter(signal_type_map=signal_map)
        emitter.emit_blueprint(ir_operations)

        literal_placements = [
            placement
            for placement in emitter.entities.values()
            if placement.role == "literal"
        ]

        assert literal_placements, "Expected literal placements along the north edge"

        expected_north_y = -emitter.layout.row_height * 3
        literal_rows = {placement.position[1] for placement in literal_placements}

        assert literal_rows == {expected_north_y}, (
            f"Literal combinators should align on row {expected_north_y}, got {literal_rows}"
        )
        assert all(
            placement.zone == "north_literals" for placement in literal_placements
        ), "Literal combinators should be tagged with the north_literals zone"

        anchor_placements = [
            placement
            for placement in emitter.entities.values()
            if placement.role == "export_anchor"
        ]

        assert anchor_placements, "Expected at least one export anchor on south edge"

        expected_south_y = emitter.layout.row_height * 3
        anchor_rows = {placement.position[1] for placement in anchor_placements}

        assert anchor_rows == {expected_south_y}, (
            f"Export anchors should align on row {expected_south_y}, got {anchor_rows}"
        )
        assert all(
            placement.zone == "south_exports" for placement in anchor_placements
        ), "Export anchors should be tagged with the south_exports zone"

    """Test blueprint emission functionality."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    @pytest.fixture
    def analyzer(self):
        return SemanticAnalyzer()

    def test_emitter_initialization(self):
        """Test emitter can be initialized."""
        emitter = BlueprintEmitter()
        assert emitter is not None

    def test_basic_emission(self, parser, analyzer):
        """Test basic blueprint emission."""
        program = parser.parse("Signal x = 42;")
        analyze_program(program, strict_types=False, analyzer=analyzer)
        ir_operations, _, signal_map = lower_program(program, analyzer)

        emitter = BlueprintEmitter(signal_type_map=signal_map)
        blueprint = emitter.emit_blueprint(ir_operations)

        assert blueprint is not None
        assert len(emitter.entities) > 0

    def test_emission_sample_files(self, parser, analyzer):
        """Test emission on sample files."""
        import os

        sample_files = [
            "tests/sample_programs/01_basic_arithmetic.fcdsl",
            "tests/sample_programs/04_memory.fcdsl",
        ]

        for file_path in sample_files:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    code = f.read()
                program = parser.parse(code)
                analyze_program(program, strict_types=False, analyzer=analyzer)
                ir_operations, _, signal_map = lower_program(program, analyzer)

                emitter = BlueprintEmitter(signal_type_map=signal_map)
                blueprint = emitter.emit_blueprint(ir_operations)

                assert blueprint is not None
                assert len(emitter.entities) > 0

    def test_memory_write_emits_signal_w_enable(self, parser, analyzer):
        """Memory writes should emit a signal-W enable combinator and writers wired on green."""

        code = """
        Memory counter: "iron-plate";
        write(("iron-plate", 0), counter, when=once);
        Signal enable = ("iron-plate", 1);
        write(read(counter) + ("iron-plate", 1), counter, when=enable);
        """

        program = parser.parse(code)
        analyze_program(program, strict_types=False, analyzer=analyzer)
        ir_operations, diagnostics, signal_map = lower_program(program, analyzer)

        assert not diagnostics.has_errors(), diagnostics.get_messages()

        emitter = BlueprintEmitter(signal_type_map=signal_map)
        blueprint = emitter.emit_blueprint(ir_operations)

        assert blueprint is not None

        enable_outputs = [
            placement
            for placement in emitter.entities.values()
            if placement.output_signals.get("signal-W") == "green"
        ]

        assert enable_outputs, "Expected a signal-W enable combinator in the blueprint"

        injectors = [
            placement
            for placement in emitter.entities.values()
            if "_write_data_" in placement.entity_id
        ]

        assert len(injectors) >= 2, (
            "Expected a dedicated write injector for each write() call"
        )

        for placement in injectors:
            assert getattr(placement.entity, "name", "") == "decider-combinator"
            assert placement.input_signals.get("signal-W") == "green", (
                "Write injector should read signal-W on the green network"
            )
            assert any(color == "red" for color in placement.output_signals.values()), (
                "Write injector should drive the memory data line on the red network"
            )

    def test_memory_outputs_restrict_input_networks(self, parser, analyzer):
        """Memory gates must read from distinct networks to avoid summing feedback and writes."""

        code = """
        Memory counter: "iron-plate";
        write(("iron-plate", 0), counter, when=once);
        write(read(counter) + ("iron-plate", 1), counter, when=1);
        """

        program = parser.parse(code)
        analyze_program(program, strict_types=False, analyzer=analyzer)
        ir_operations, diagnostics, signal_map = lower_program(program, analyzer)

        assert not diagnostics.has_errors(), diagnostics.get_messages()

        emitter = BlueprintEmitter(signal_type_map=signal_map)
        emitter.emit_blueprint(ir_operations)

        memory_modules = emitter.memory_builder.memory_modules
        assert memory_modules, "Expected memory modules to be registered during emission"

        for memory_id, module in memory_modules.items():
            write_gate = module["write_gate"].entity
            hold_gate = module["hold_gate"].entity

            write_outputs = getattr(write_gate, "outputs", [])
            hold_outputs = getattr(hold_gate, "outputs", [])

            assert write_outputs, f"{memory_id} write gate should define outputs"
            assert hold_outputs, f"{memory_id} hold gate should define outputs"

            write_selection = write_outputs[0].networks
            hold_selection = hold_outputs[0].networks

            write_green = getattr(write_selection, "green", None)
            write_red = getattr(write_selection, "red", None)
            hold_green = getattr(hold_selection, "green", None)
            hold_red = getattr(hold_selection, "red", None)

            assert write_red is True, (
                f"{memory_id} write gate outputs must ride the red network"
            )
            assert write_green is False, (
                f"{memory_id} write gate must not drive the green network"
            )
            assert hold_red is True, (
                f"{memory_id} hold gate outputs must ride the red network"
            )
            assert hold_green is False, (
                f"{memory_id} hold gate must not drive the green network"
            )
            write_condition = write_gate.conditions[0]
            hold_condition = hold_gate.conditions[0]

            write_enable_networks = getattr(
                write_condition, "first_signal_networks", None
            )
            hold_enable_networks = getattr(
                hold_condition, "first_signal_networks", None
            )

            assert write_enable_networks is not None, (
                f"{memory_id} write gate condition should expose network selection"
            )
            assert hold_enable_networks is not None, (
                f"{memory_id} hold gate condition should expose network selection"
            )

            write_enable_green = getattr(write_enable_networks, "green", None)
            write_enable_red = getattr(write_enable_networks, "red", None)
            hold_enable_green = getattr(hold_enable_networks, "green", None)
            hold_enable_red = getattr(hold_enable_networks, "red", None)

            assert write_enable_green is True and write_enable_red is False, (
                f"{memory_id} write gate must read signal-W from the green network only"
            )
            assert hold_enable_green is True and hold_enable_red is False, (
                f"{memory_id} hold gate must read signal-W from the green network only"
            )

    def test_unconditional_memory_increment_uses_feedback_loop(self, parser, analyzer):
        """Unconditional counter increments should reuse the arithmetic combinator via feedback."""

        code = """
        Memory counter: "signal-A";
        write(read(counter) + ("signal-A", 1), counter, when=1);
        """

        program = parser.parse(code)
        analyze_program(program, strict_types=False, analyzer=analyzer)
        ir_operations, diagnostics, signal_map = lower_program(program, analyzer)

        assert not diagnostics.has_errors(), diagnostics.get_messages()

        emitter = BlueprintEmitter(signal_type_map=signal_map)
        emitter.emit_blueprint(ir_operations)

        memory_id = "mem_counter"

        feedback_entities = [
            placement
            for placement in emitter.entities.values()
            if placement.metadata.get("feedback_loop")
            and placement.metadata.get("memory_id") == memory_id
        ]

        assert feedback_entities, (
            "Expected arithmetic combinator to advertise feedback loop metadata"
        )

        feedback_entity = feedback_entities[0]
        feedback_entity_id = feedback_entity.entity_id

        assert emitter.signal_graph.get_source(memory_id) == feedback_entity_id, (
            "Memory source should be redirected to the feedback combinator"
        )
        assert not any(
            entity_id.startswith(f"{memory_id}_write_data_")
            for entity_id in emitter.entities
        ), "Feedback loop strategy must avoid creating gated write injectors"
        assert not any(
            entity_id.startswith(f"{memory_id}_write_enable_")
            for entity_id in emitter.entities
        ), "Feedback loop strategy must avoid creating enable pulse combinators"

        assert feedback_entity.metadata.get("feedback_signal") is not None, (
            "Feedback metadata should expose the propagated signal identifier"
        )

        read_nodes = emitter._memory_reads_by_memory.get(memory_id, [])

        assert read_nodes, "Expected at least one memory read to be indexed for feedback wiring"

        assert all(
            emitter.signal_graph.get_source(read_node_id) == feedback_entity_id
            for read_node_id in read_nodes
        ), "Every memory read should redirect to the feedback combinator"

    def test_signal_name_resolution(self):
        """Test signal name resolution with mapping."""
        signal_map = {"__v1": "signal-A", "__v2": "signal-B"}
        emitter = BlueprintEmitter(signal_type_map=signal_map)

        # Test that implicit signals are resolved
        resolved = emitter._get_signal_name("__v1")
        assert resolved == "signal-A"

        # Test that explicit signals pass through
        resolved = emitter._get_signal_name("iron-plate")
        assert resolved == "iron-plate"

    def test_wire_color_planner_assigns_opposite_colors(self):
        emitter = BlueprintEmitter()
        emitter._reset_for_emit()

        class DummyEntity:
            dual_circuit_connectable = False

            def __init__(self, name: str):
                self.name = name

        source_a = EntityPlacement(
            entity=DummyEntity("constant-combinator"),
            entity_id="source_a",
            position=(0, 0),
            output_signals={},
            input_signals={},
        )
        source_b = EntityPlacement(
            entity=DummyEntity("constant-combinator"),
            entity_id="source_b",
            position=(2, 0),
            output_signals={},
            input_signals={},
        )
        sink = EntityPlacement(
            entity=DummyEntity("arithmetic-combinator"),
            entity_id="sink",
            position=(1, 1),
            output_signals={},
            input_signals={},
        )

        emitter.entities = {
            "source_a": source_a,
            "source_b": source_b,
            "sink": sink,
        }

        entry_a = SignalUsageEntry(signal_id="sig_a")
        entry_a.resolved_signal_name = "signal-A"
        entry_b = SignalUsageEntry(signal_id="sig_b")
        entry_b.resolved_signal_name = "signal-A"

        emitter.signal_usage = {
            "sig_a": entry_a,
            "sig_b": entry_b,
        }

        emitter.signal_graph.set_source("sig_a", "source_a")
        emitter.signal_graph.add_sink("sig_a", "sink")

        emitter.signal_graph.set_source("sig_b", "source_b")
        emitter.signal_graph.add_sink("sig_b", "sink")

        emitter._track_signal_source("sig_a", "source_a")
        emitter._track_signal_source("sig_b", "source_b")

        emitter._prepare_wiring_plan()

        # Ensure the planner chose distinct colors for the conflicting producers
        color_a = emitter._edge_color_map[("source_a", "sink", "signal-A")]
        color_b = emitter._edge_color_map[("source_b", "sink", "signal-A")]

        assert color_a != color_b


def test_format_entity_description_round_trip():
    """Ensure the description formatter produces deterministic output."""

    debug_info = {
        "name": "sum",
        "factorio_signal": "signal-A",
        "location": "example.fcdsl:12",
    }

    description = format_entity_description(debug_info)
    assert description == "signal sum (signal-A) @ example.fcdsl:12"


class TestWarningAnalysis:
    """Test for analyzing and addressing warnings."""

    def test_no_memory_signal_warnings(self):
        """Test that memory signal warnings are eliminated."""
        from dsl_compiler.src.parser import DSLParser
        from dsl_compiler.src.semantic import SemanticAnalyzer, analyze_program
        from dsl_compiler.src.lowerer import lower_program
        from dsl_compiler.src.emit import BlueprintEmitter

        code = """
        Memory counter: "iron-plate";
        write(("iron-plate", 0), counter, when=once);
        Signal current = read(counter);
        write(current + ("iron-plate", 1), counter, when=1);
        """

        parser = DSLParser()
        analyzer = SemanticAnalyzer()

        program = parser.parse(code)
        analyze_program(program, strict_types=False, analyzer=analyzer)
        ir_operations, _, signal_map = lower_program(program, analyzer)

        emitter = BlueprintEmitter(signal_type_map=signal_map)
        blueprint = emitter.emit_blueprint(ir_operations)

        # Check for warnings
        warnings = []
        if hasattr(emitter.diagnostics, "diagnostics"):
            warnings = [
                d for d in emitter.diagnostics.diagnostics if d.level.name == "WARNING"
            ]

        # Should not have memory signal warnings
        memory_warnings = [
            w for w in warnings if "Unknown signal type for memory" in w.message
        ]
        assert len(memory_warnings) == 0, (
            f"Found memory signal warnings: {[w.message for w in memory_warnings]}"
        )


def test_wire_relays_inserted_for_long_spans():
    """Connections longer than the wire span should insert relay poles."""

    emitter = BlueprintEmitter()
    emitter._reset_for_emit()

    source, sink = _make_far_endpoints(emitter, sink_target=(40, 0))

    relays = emitter._insert_wire_relays_if_needed(source, sink)

    assert relays, "Expected relay placements for long-span wiring"

    distance = emitter._compute_wire_distance(source, sink)
    expected_count = math.ceil(distance / MAX_CIRCUIT_WIRE_SPAN) - 1

    assert len(relays) == expected_count, (
        f"Expected {expected_count} relays for span {distance}, got {len(relays)}"
    )
    assert all(relay.role == "wire_relay" for relay in relays)
    assert all(relay.zone == "infrastructure" for relay in relays)
    assert all(relay.entity_id in emitter.entities for relay in relays)


def test_wire_relays_respect_disabled_option():
    emitter = BlueprintEmitter(wire_relay_options=WireRelayOptions(enabled=False))
    emitter._reset_for_emit()

    source, sink = _make_far_endpoints(emitter, sink_target=(40, 0))

    relays = emitter._insert_wire_relays_if_needed(source, sink)

    assert relays == [], "No relays should be inserted when auto wiring is disabled"
    assert emitter.diagnostics.warning_count == 0


def test_wire_relays_manhattan_strategy_inserts_extra_relays():
    emitter = BlueprintEmitter(
        wire_relay_options=WireRelayOptions(placement_strategy="manhattan")
    )
    emitter._reset_for_emit()

    source, sink = _make_far_endpoints(emitter, sink_target=(6, 6))

    relays = emitter._insert_wire_relays_if_needed(source, sink)

    assert relays, "Manhattan strategy should require at least one relay on diagonal span"
    assert len(relays) == 1


def test_wire_relays_max_limit_triggers_warning():
    emitter = BlueprintEmitter(wire_relay_options=WireRelayOptions(max_relays=0))
    emitter._reset_for_emit()

    source, sink = _make_far_endpoints(emitter, sink_target=(40, 0))

    relays = emitter._insert_wire_relays_if_needed(source, sink)

    assert relays == []
    warnings = [
        diag.message
        for diag in emitter.diagnostics.diagnostics
        if diag.level.name == "WARNING"
    ]
    assert any("requires" in msg for msg in warnings)
