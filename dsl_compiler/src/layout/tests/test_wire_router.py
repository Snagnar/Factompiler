"""
Tests for layout/wire_router.py - Wire color planning and routing.
"""

from dsl_compiler.src.layout.signal_graph import SignalGraph
from dsl_compiler.src.layout.wire_router import (
    CircuitEdge,
    ColoringResult,
    ConflictEdge,
    _resolve_entity_type,
    collect_circuit_edges,
    detect_merge_color_conflicts,
    plan_wire_colors,
)


def make_edge(
    source: str, sink: str, signal: str, originating_merge_id: str | None = None
) -> CircuitEdge:
    """Helper to create a CircuitEdge."""
    return CircuitEdge(
        logical_signal_id=f"{source}->{sink}",
        resolved_signal_name=signal,
        source_entity_id=source,
        sink_entity_id=sink,
        originating_merge_id=originating_merge_id,
    )


# === Tests for CircuitEdge ===


class TestCircuitEdge:
    """Tests for CircuitEdge dataclass."""

    def test_circuit_edge_creation(self):
        """Test creating a basic CircuitEdge."""
        edge = CircuitEdge(
            logical_signal_id="sig1",
            resolved_signal_name="signal-A",
            source_entity_id="src",
            sink_entity_id="sink",
        )
        assert edge.logical_signal_id == "sig1"
        assert edge.resolved_signal_name == "signal-A"
        assert edge.source_entity_id == "src"
        assert edge.sink_entity_id == "sink"

    def test_circuit_edge_with_types(self):
        """Test CircuitEdge with entity types and role."""
        edge = CircuitEdge(
            logical_signal_id="sig1",
            resolved_signal_name="signal-A",
            source_entity_id="src",
            sink_entity_id="sink",
            source_entity_type="constant-combinator",
            sink_entity_type="arithmetic-combinator",
            sink_role="arithmetic",
        )
        assert edge.source_entity_type == "constant-combinator"
        assert edge.sink_entity_type == "arithmetic-combinator"
        assert edge.sink_role == "arithmetic"

    def test_circuit_edge_with_originating_merge(self):
        """Test CircuitEdge with originating_merge_id."""
        edge = CircuitEdge(
            logical_signal_id="sig1",
            resolved_signal_name="signal-A",
            source_entity_id="src",
            sink_entity_id="sink",
            originating_merge_id="merge_1",
        )
        assert edge.originating_merge_id == "merge_1"


# === Tests for ConflictEdge ===


class TestConflictEdge:
    """Tests for ConflictEdge dataclass."""

    def test_conflict_edge_creation(self):
        """Test creating a ConflictEdge."""
        edge = ConflictEdge(
            nodes=(("src1", "signal-A"), ("src2", "signal-A")),
        )
        assert edge.nodes == (("src1", "signal-A"), ("src2", "signal-A"))
        assert edge.sinks == set()

    def test_conflict_edge_with_sinks(self):
        """Test ConflictEdge with sinks."""
        edge = ConflictEdge(
            nodes=(("src1", "signal-A"), ("src2", "signal-A")),
            sinks={"sink1", "sink2"},
        )
        assert edge.sinks == {"sink1", "sink2"}


# === Tests for ColoringResult ===


class TestColoringResult:
    """Tests for ColoringResult dataclass."""

    def test_coloring_result_creation(self):
        """Test creating a ColoringResult."""
        result = ColoringResult(
            assignments={("src1", "signal-A"): "red"},
            conflicts=[],
            is_bipartite=True,
        )
        assert result.assignments == {("src1", "signal-A"): "red"}
        assert result.conflicts == []
        assert result.is_bipartite is True


# === Tests for _resolve_entity_type ===


class TestResolveEntityType:
    """Tests for _resolve_entity_type function."""

    def test_resolve_entity_type_none(self):
        """Test resolving None placement."""
        assert _resolve_entity_type(None) is None

    def test_resolve_entity_type_with_entity_type(self):
        """Test resolving placement with entity_type attribute."""

        class MockPlacement:
            entity_type = "constant-combinator"

        result = _resolve_entity_type(MockPlacement())
        assert result == "constant-combinator"

    def test_resolve_entity_type_with_entity(self):
        """Test resolving placement with entity attribute."""

        class MockEntity:
            pass

        class MockPlacement:
            entity_type = None
            entity = MockEntity()

        result = _resolve_entity_type(MockPlacement())
        assert result == "MockEntity"

    def test_resolve_entity_type_with_prototype(self):
        """Test resolving placement with prototype attribute."""

        class MockPlacement:
            entity_type = None
            entity = None
            prototype = "test-prototype"

        result = _resolve_entity_type(MockPlacement())
        assert result == "test-prototype"

    def test_resolve_entity_type_no_attributes(self):
        """Test resolving placement with no recognizable attributes."""

        class MockPlacement:
            entity_type = None
            entity = None
            prototype = None

        result = _resolve_entity_type(MockPlacement())
        assert result is None


# === Tests for collect_circuit_edges ===


class TestCollectCircuitEdges:
    """Tests for collect_circuit_edges function."""

    def test_collect_circuit_edges_empty(self):
        """Test collecting edges with empty signal graph."""
        signal_graph = SignalGraph()
        edges = collect_circuit_edges(signal_graph, {}, {})
        assert edges == []

    def test_collect_circuit_edges_basic(self):
        """Test collecting edges with basic signal graph."""
        signal_graph = SignalGraph()
        signal_graph.set_source("sig1", "src1")
        signal_graph.add_sink("sig1", "sink1")

        edges = collect_circuit_edges(signal_graph, {}, {})
        assert len(edges) == 1
        assert edges[0].logical_signal_id == "sig1"
        assert edges[0].source_entity_id == "src1"
        assert edges[0].sink_entity_id == "sink1"

    def test_collect_circuit_edges_with_usage(self):
        """Test collecting edges with signal usage info."""
        signal_graph = SignalGraph()
        signal_graph.set_source("sig1", "src1")
        signal_graph.add_sink("sig1", "sink1")

        class MockUsage:
            resolved_signal_name = "signal-A"

        usage = {"sig1": MockUsage()}

        edges = collect_circuit_edges(signal_graph, usage, {})
        assert len(edges) == 1
        assert edges[0].resolved_signal_name == "signal-A"

    def test_collect_circuit_edges_with_entities(self):
        """Test collecting edges with entity placements."""
        signal_graph = SignalGraph()
        signal_graph.set_source("sig1", "src1")
        signal_graph.add_sink("sig1", "sink1")

        class MockPlacement:
            entity_type = "constant-combinator"
            role = "literal"

        entities = {"src1": MockPlacement(), "sink1": MockPlacement()}

        edges = collect_circuit_edges(signal_graph, {}, entities)
        assert len(edges) == 1
        assert edges[0].source_entity_type == "constant-combinator"
        assert edges[0].sink_entity_type == "constant-combinator"
        assert edges[0].sink_role == "literal"

    def test_collect_circuit_edges_export_anchor(self):
        """Test collecting edges detects export anchors."""
        signal_graph = SignalGraph()
        signal_graph.set_source("sig1", "src1")
        signal_graph.add_sink("sig1", "output_export_anchor")

        edges = collect_circuit_edges(signal_graph, {}, {})
        assert len(edges) == 1
        assert edges[0].sink_role == "export"

    def test_collect_circuit_edges_multiple_sources(self):
        """Test collecting edges with multiple sources for a signal."""
        signal_graph = SignalGraph()
        signal_graph.set_source("sig1", "src1")
        signal_graph.set_source("sig1", "src2")  # Second source
        signal_graph.add_sink("sig1", "sink1")

        edges = collect_circuit_edges(signal_graph, {}, {})
        # Should create edges for both sources
        assert len(edges) == 2
        source_ids = {e.source_entity_id for e in edges}
        assert source_ids == {"src1", "src2"}


# === Tests for plan_wire_colors ===


class TestPlanWireColors:
    """Tests for plan_wire_colors function."""

    def test_assigns_two_colors_when_possible(self):
        """Conflicting producers should receive opposite wire colors."""
        edges = [
            make_edge("src_a", "sink_1", "signal-A"),
            make_edge("src_b", "sink_1", "signal-A"),
        ]

        result = plan_wire_colors(edges)

        assert result.is_bipartite is True
        color_a = result.assignments[("src_a", "signal-A")]
        color_b = result.assignments[("src_b", "signal-A")]
        assert color_a != color_b, "Conflicting producers should receive opposite wire colors"

    def test_detects_non_bipartite_conflict(self):
        """Non-bipartite graphs should be detected and report conflicts."""
        edges = [
            make_edge("src_a", "sink_1", "signal-A"),
            make_edge("src_b", "sink_1", "signal-A"),
            make_edge("src_b", "sink_2", "signal-A"),
            make_edge("src_c", "sink_2", "signal-A"),
            make_edge("src_c", "sink_3", "signal-A"),
            make_edge("src_a", "sink_3", "signal-A"),
        ]

        result = plan_wire_colors(edges)

        assert result.is_bipartite is False
        assert result.conflicts, "Non-bipartite graphs should record conflicts"
        conflict_nodes = {node for edge in result.conflicts for node in edge.nodes}
        assert len(conflict_nodes) >= 2

    def test_respects_locked_colors(self):
        """Pre-locked colors should be respected."""
        edges = [
            make_edge("src_a", "sink_1", "signal-A"),
            make_edge("src_b", "sink_1", "signal-A"),
        ]

        result = plan_wire_colors(edges, locked_colors={("src_a", "signal-A"): "green"})

        assert result.assignments[("src_a", "signal-A")] == "green"
        assert result.assignments[("src_b", "signal-A")] == "red"

    def test_single_edge_no_conflict(self):
        """Single edge should not create conflicts."""
        edges = [make_edge("src_a", "sink_1", "signal-A")]

        result = plan_wire_colors(edges)

        assert result.is_bipartite is True
        assert result.conflicts == []
        assert ("src_a", "signal-A") in result.assignments

    def test_same_merge_no_conflict(self):
        """Edges from same merge should not conflict."""
        edges = [
            make_edge("src_a", "sink_1", "signal-A", originating_merge_id="merge1"),
            make_edge("src_b", "sink_1", "signal-A", originating_merge_id="merge1"),
        ]

        result = plan_wire_colors(edges)

        # Same merge ID means they should be on the same wire (no conflict edge)
        assert result.is_bipartite is True

    def test_different_merges_conflict(self):
        """Edges from different merges to same sink should conflict."""
        edges = [
            make_edge("src_a", "sink_1", "signal-A", originating_merge_id="merge1"),
            make_edge("src_b", "sink_1", "signal-A", originating_merge_id="merge2"),
        ]

        result = plan_wire_colors(edges)

        # Different merge IDs to same sink means potential conflict
        color_a = result.assignments[("src_a", "signal-A")]
        color_b = result.assignments[("src_b", "signal-A")]
        assert color_a != color_b

    def test_empty_edges(self):
        """Empty edge list should return empty result."""
        result = plan_wire_colors([])

        assert result.assignments == {}
        assert result.conflicts == []
        assert result.is_bipartite is True

    def test_edge_without_source(self):
        """Edge without source entity should be skipped."""
        edge = CircuitEdge(
            logical_signal_id="sig1",
            resolved_signal_name="signal-A",
            source_entity_id=None,  # No source
            sink_entity_id="sink1",
        )

        result = plan_wire_colors([edge])

        assert result.assignments == {}
        assert result.is_bipartite is True

    def test_locked_color_conflict_detected(self):
        """Locked colors that conflict should be detected."""
        edges = [
            make_edge("src_a", "sink_1", "signal-A"),
            make_edge("src_b", "sink_1", "signal-A"),
        ]

        # Lock both to the same color - this creates a conflict
        result = plan_wire_colors(
            edges,
            locked_colors={
                ("src_a", "signal-A"): "red",
                ("src_b", "signal-A"): "red",
            },
        )

        assert result.is_bipartite is False
        assert len(result.conflicts) > 0


# === Tests for detect_merge_color_conflicts ===


class TestDetectMergeColorConflicts:
    """Tests for detect_merge_color_conflicts function."""

    def test_detect_no_conflicts_single_merge(self):
        """Test no conflicts when source is only in one merge."""
        merge_membership = {"src1": {"merge1"}}
        signal_graph = SignalGraph()

        result = detect_merge_color_conflicts(merge_membership, signal_graph)
        assert result == {}

    def test_detect_no_conflicts_no_common_sinks(self):
        """Test no conflicts when merges don't share sinks."""
        merge_membership = {"src1": {"merge1", "merge2"}}

        # Create a mock signal graph with get_sinks method
        class MockSignalGraph:
            def get_sinks(self, merge_id: str) -> list[str]:
                if merge_id == "merge1":
                    return ["sink1"]
                elif merge_id == "merge2":
                    return ["sink2"]  # Different sink
                return []

        result = detect_merge_color_conflicts(merge_membership, MockSignalGraph())
        assert result == {}

    def test_detect_conflicts_common_sink(self):
        """Test conflicts detected when merges share a sink."""
        merge_membership = {"src1": {"merge1", "merge2"}}

        # Create a mock signal graph with get_sinks method
        class MockSignalGraph:
            def get_sinks(self, merge_id: str) -> list[str]:
                if merge_id == "merge1":
                    return ["common_sink"]
                elif merge_id == "merge2":
                    return ["common_sink"]  # Same sink!
                return []

        result = detect_merge_color_conflicts(merge_membership, MockSignalGraph())

        # Should have locked colors for both merges
        assert len(result) == 2
        assert ("merge1", "src1") in result
        assert ("merge2", "src1") in result
        # Colors should be different
        assert result[("merge1", "src1")] != result[("merge2", "src1")]

    def test_detect_empty_membership(self):
        """Test empty merge membership returns no conflicts."""
        result = detect_merge_color_conflicts({}, SignalGraph())
        assert result == {}

    def test_detect_conflicts_with_none_signal_graph(self):
        """Test that None signal graph results in no conflicts."""
        merge_membership = {"src1": {"merge1", "merge2"}}
        result = detect_merge_color_conflicts(merge_membership, None)
        assert result == {}
