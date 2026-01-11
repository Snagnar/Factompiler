"""
Tests for layout/wire_router.py - Wire color planning and routing.
"""

from dsl_compiler.src.layout.wire_router import CircuitEdge, plan_wire_colors


def make_edge(source: str, sink: str, signal: str) -> CircuitEdge:
    """Helper to create a CircuitEdge."""
    return CircuitEdge(
        logical_signal_id=f"{source}->{sink}",
        resolved_signal_name=signal,
        source_entity_id=source,
        sink_entity_id=sink,
    )


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
