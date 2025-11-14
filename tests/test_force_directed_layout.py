"""Tests for force-directed layout optimization."""
import math

import pytest
from dsl_compiler.src.layout.force_directed_layout import (
    ForceDirectedLayoutEngine,
    LayoutConstraints,
)
from dsl_compiler.src.layout.signal_graph import SignalGraph
from dsl_compiler.src.layout.layout_plan import EntityPlacement
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics


def test_force_directed_basic():
    """Test basic force-directed optimization."""
    # Create simple graph: A -> B -> C
    signal_graph = SignalGraph()
    signal_graph.set_source("sig1", "A")
    signal_graph.add_sink("sig1", "B")
    signal_graph.set_source("sig2", "B")
    signal_graph.add_sink("sig2", "C")

    # Create entity placements
    placements = {
        "A": EntityPlacement("A", "arithmetic-combinator", None, {"footprint": (1, 2)}),
        "B": EntityPlacement("B", "arithmetic-combinator", None, {"footprint": (1, 2)}),
        "C": EntityPlacement("C", "arithmetic-combinator", None, {"footprint": (1, 2)}),
    }

    # Optimize
    diagnostics = ProgramDiagnostics()
    engine = ForceDirectedLayoutEngine(signal_graph, placements, diagnostics)
    positions = engine.optimize(population_size=3, parallel=False)

    # Verify all entities positioned
    assert len(positions) == 3
    assert "A" in positions
    assert "B" in positions
    assert "C" in positions

    # Verify positions are valid tuples
    for pos in positions.values():
        assert isinstance(pos, tuple)
        assert len(pos) == 2
        assert isinstance(pos[0], (int, float))
        assert isinstance(pos[1], (int, float))


def test_force_directed_with_fixed_position():
    """Test that fixed positions are respected."""
    signal_graph = SignalGraph()
    signal_graph.set_source("sig1", "A")
    signal_graph.add_sink("sig1", "B")

    placements = {
        "A": EntityPlacement(
            "A",
            "arithmetic-combinator",
            (0.0, 0.0),  # Fixed position
            {"footprint": (1, 2), "user_specified_position": True},
        ),
        "B": EntityPlacement("B", "arithmetic-combinator", None, {"footprint": (1, 2)}),
    }

    diagnostics = ProgramDiagnostics()
    engine = ForceDirectedLayoutEngine(signal_graph, placements, diagnostics)
    positions = engine.optimize(population_size=3, parallel=False)

    # Verify A stayed at fixed position (within floating point tolerance)
    assert abs(positions["A"][0] - 0.0) < 1e-6
    assert abs(positions["A"][1] - 0.0) < 1e-6

    # Verify B was positioned
    assert "B" in positions
    assert positions["B"] != (0.0, 0.0)  # Should be different from A


def test_force_directed_empty():
    """Test force-directed optimization with no entities."""
    signal_graph = SignalGraph()
    placements = {}

    diagnostics = ProgramDiagnostics()
    engine = ForceDirectedLayoutEngine(signal_graph, placements, diagnostics)
    positions = engine.optimize(population_size=1, parallel=False)

    assert len(positions) == 0


def test_force_directed_disconnected():
    """Test force-directed optimization with disconnected entities."""
    signal_graph = SignalGraph()
    # No connections between entities

    placements = {
        "A": EntityPlacement("A", "arithmetic-combinator", None, {"footprint": (1, 2)}),
        "B": EntityPlacement("B", "arithmetic-combinator", None, {"footprint": (1, 2)}),
    }

    diagnostics = ProgramDiagnostics()
    engine = ForceDirectedLayoutEngine(signal_graph, placements, diagnostics)
    positions = engine.optimize(population_size=2, parallel=False)

    # Verify all entities positioned
    assert len(positions) == 2
    assert "A" in positions
    assert "B" in positions

    # Verify they are not at the same position (repulsion should separate them)
    assert positions["A"] != positions["B"]


def test_force_directed_constraints():
    """Test that constraints are applied."""
    signal_graph = SignalGraph()
    signal_graph.set_source("sig1", "A")
    signal_graph.add_sink("sig1", "B")

    placements = {
        "A": EntityPlacement("A", "arithmetic-combinator", None, {"footprint": (1, 2)}),
        "B": EntityPlacement("B", "arithmetic-combinator", None, {"footprint": (1, 2)}),
    }

    # Use very restrictive wire span constraint
    constraints = LayoutConstraints(max_wire_span=5.0, entity_spacing=0.5)

    diagnostics = ProgramDiagnostics()
    engine = ForceDirectedLayoutEngine(
        signal_graph, placements, diagnostics, constraints
    )
    positions = engine.optimize(population_size=3, parallel=False)

    # Verify positions respect wire span

    dist = math.dist(positions["A"], positions["B"])
    assert (
        dist <= constraints.max_wire_span * 1.1
    )  # Allow small tolerance for optimization


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
