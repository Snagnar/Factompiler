"""
Tests for emit.py - Blueprint emission functionality.
"""

import pytest
from dsl_compiler.src.parser import DSLParser
from dsl_compiler.src.semantic import SemanticAnalyzer, analyze_program
from dsl_compiler.src.lowerer import lower_program
from dsl_compiler.src.emit import BlueprintEmitter
from dsl_compiler.src.emission.signals import EntityPlacement, SignalUsageEntry
from dsl_compiler.src.emission.debug_format import format_entity_description


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
        Memory counter = 0;
        Signal current = read(counter);
        write(counter, current + 1);
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
