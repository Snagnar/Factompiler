"""Tests for emitter.py - Blueprint emission."""

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.emission.emitter import EDGE_LAYOUT_NOTE, BlueprintEmitter
from dsl_compiler.src.layout.layout_plan import LayoutPlan, WireConnection


class TestBlueprintEmitterInit:
    """Tests for BlueprintEmitter initialization."""

    def test_init_creates_emitter(self):
        """Test BlueprintEmitter can be initialized."""
        diagnostics = ProgramDiagnostics()
        emitter = BlueprintEmitter(diagnostics)

        assert emitter.diagnostics is diagnostics
        assert emitter.signal_type_map == {}
        assert emitter.blueprint is not None

    def test_init_with_signal_type_map(self):
        """Test BlueprintEmitter uses provided signal type map."""
        diagnostics = ProgramDiagnostics()
        signal_map = {"__v1": "signal-A", "__v2": "signal-B"}
        emitter = BlueprintEmitter(diagnostics, signal_type_map=signal_map)

        assert emitter.signal_type_map == signal_map


class TestBlueprintEmitterEmitFromPlan:
    """Tests for BlueprintEmitter.emit_from_plan()."""

    def test_emit_empty_plan(self):
        """Test emitting an empty layout plan."""
        diagnostics = ProgramDiagnostics()
        emitter = BlueprintEmitter(diagnostics)
        layout_plan = LayoutPlan()

        blueprint = emitter.emit_from_plan(layout_plan)

        assert blueprint is not None
        assert len(blueprint.entities) == 0

    def test_emit_plan_with_single_entity(self):
        """Test emitting a plan with a single entity."""
        diagnostics = ProgramDiagnostics()
        emitter = BlueprintEmitter(diagnostics)
        layout_plan = LayoutPlan()

        layout_plan.create_and_add_placement(
            ir_node_id="const_1",
            entity_type="constant-combinator",
            position=(0.5, 0.5),
            footprint=(1, 1),
            role="constant",
        )

        blueprint = emitter.emit_from_plan(layout_plan)

        assert len(blueprint.entities) == 1

    def test_emit_plan_with_wire_connection(self):
        """Test emitting a plan with wire connections."""
        diagnostics = ProgramDiagnostics()
        emitter = BlueprintEmitter(diagnostics)
        layout_plan = LayoutPlan()

        layout_plan.create_and_add_placement(
            ir_node_id="const_1",
            entity_type="constant-combinator",
            position=(0.5, 0.5),
            footprint=(1, 1),
            role="constant",
        )
        layout_plan.create_and_add_placement(
            ir_node_id="arith_1",
            entity_type="arithmetic-combinator",
            position=(3.0, 0.5),
            footprint=(2, 1),
            role="arithmetic",
        )

        layout_plan.add_wire_connection(
            WireConnection(
                source_entity_id="const_1",
                sink_entity_id="arith_1",
                signal_name="signal-A",
                wire_color="red",
                source_side=None,
                sink_side=1,
            )
        )

        blueprint = emitter.emit_from_plan(layout_plan)

        assert len(blueprint.entities) == 2

    def test_emit_plan_sets_blueprint_label(self):
        """Test emitting a plan sets the blueprint label."""
        diagnostics = ProgramDiagnostics()
        emitter = BlueprintEmitter(diagnostics)
        layout_plan = LayoutPlan()
        layout_plan.blueprint_label = "Test Blueprint"

        blueprint = emitter.emit_from_plan(layout_plan)

        assert blueprint.label == "Test Blueprint"

    def test_emit_plan_sets_blueprint_description(self):
        """Test emitting a plan sets the blueprint description."""
        diagnostics = ProgramDiagnostics()
        emitter = BlueprintEmitter(diagnostics)
        layout_plan = LayoutPlan()
        layout_plan.blueprint_description = "Test description"

        blueprint = emitter.emit_from_plan(layout_plan)

        # Should include the edge layout note
        assert "Test description" in blueprint.description
        assert EDGE_LAYOUT_NOTE in blueprint.description


class TestBlueprintEmitterApplyMetadata:
    """Tests for BlueprintEmitter._apply_blueprint_metadata()."""

    def test_apply_metadata_empty_description(self):
        """Test applying metadata to empty description."""
        diagnostics = ProgramDiagnostics()
        emitter = BlueprintEmitter(diagnostics)
        layout_plan = LayoutPlan()
        layout_plan.blueprint_description = ""

        blueprint = emitter.emit_from_plan(layout_plan)

        assert blueprint.description == EDGE_LAYOUT_NOTE

    def test_apply_metadata_appends_to_existing(self):
        """Test applying metadata appends to existing description."""
        diagnostics = ProgramDiagnostics()
        emitter = BlueprintEmitter(diagnostics)
        layout_plan = LayoutPlan()
        layout_plan.blueprint_description = "Custom description"

        blueprint = emitter.emit_from_plan(layout_plan)

        assert "Custom description" in blueprint.description
        assert EDGE_LAYOUT_NOTE in blueprint.description

    def test_apply_metadata_does_not_duplicate_note(self):
        """Test applying metadata does not duplicate the edge layout note."""
        diagnostics = ProgramDiagnostics()
        emitter = BlueprintEmitter(diagnostics)
        layout_plan = LayoutPlan()
        layout_plan.blueprint_description = EDGE_LAYOUT_NOTE

        blueprint = emitter.emit_from_plan(layout_plan)

        # Should not have duplicated the note
        assert blueprint.description.count(EDGE_LAYOUT_NOTE) == 1


class TestBlueprintEmitterMaterializeConnections:
    """Tests for BlueprintEmitter._materialize_connections()."""

    def test_missing_entity_logs_warning(self):
        """Test that missing entity in connection logs warning."""
        diagnostics = ProgramDiagnostics()
        emitter = BlueprintEmitter(diagnostics)
        layout_plan = LayoutPlan()

        # Add only one entity
        layout_plan.create_and_add_placement(
            ir_node_id="const_1",
            entity_type="constant-combinator",
            position=(0.5, 0.5),
            footprint=(1, 1),
            role="constant",
        )

        # But try to connect to a non-existent entity
        layout_plan.add_wire_connection(
            WireConnection(
                source_entity_id="const_1",
                sink_entity_id="nonexistent",
                signal_name="signal-A",
                wire_color="red",
            )
        )

        emitter.emit_from_plan(layout_plan)

        # Should have logged a warning about missing entity
        assert diagnostics.warning_count() > 0


class TestBlueprintEmitterEnsureSignalMapRegistered:
    """Tests for BlueprintEmitter._ensure_signal_map_registered()."""

    def test_registers_simple_signal(self):
        """Test registering a simple signal string."""
        diagnostics = ProgramDiagnostics()
        signal_map = {"__v1": "signal-test-custom"}
        emitter = BlueprintEmitter(diagnostics, signal_type_map=signal_map)

        # The signal registration should not raise
        assert emitter.signal_type_map == signal_map

    def test_registers_dict_signal(self):
        """Test registering a signal dict with name and type."""
        diagnostics = ProgramDiagnostics()
        signal_map = {
            "__v1": {"name": "signal-custom", "type": "virtual"},
        }
        emitter = BlueprintEmitter(diagnostics, signal_type_map=signal_map)

        assert emitter.signal_type_map == signal_map


class TestEdgeLayoutNote:
    """Tests for the EDGE_LAYOUT_NOTE constant."""

    def test_edge_layout_note_content(self):
        """Test that EDGE_LAYOUT_NOTE has expected content."""
        assert "Edge layout" in EDGE_LAYOUT_NOTE
        assert "constants" in EDGE_LAYOUT_NOTE
        assert "north" in EDGE_LAYOUT_NOTE
        assert "south" in EDGE_LAYOUT_NOTE
