"""
Tests for layout/signal_graph.py - Signal connectivity graph.
"""

from dsl_compiler.src.layout.signal_graph import SignalGraph


class TestSignalGraphInit:
    """Tests for SignalGraph initialization."""

    def test_init_creates_empty_graph(self):
        """Test SignalGraph can be initialized."""
        graph = SignalGraph()
        assert graph is not None
        assert graph.signals() == set()


class TestSignalGraphSetSource:
    """Tests for SignalGraph.set_source method."""

    def test_set_source_single(self):
        """Test setting a single source for a signal."""
        graph = SignalGraph()
        graph.set_source("signal-A", "entity-1")
        assert graph.get_source("signal-A") == "entity-1"

    def test_set_source_multiple_sources(self):
        """Test setting multiple sources for a signal."""
        graph = SignalGraph()
        graph.set_source("signal-A", "entity-1")
        graph.set_source("signal-A", "entity-2")
        # get_source returns the first one
        assert graph.get_source("signal-A") == "entity-1"

    def test_set_source_same_entity_twice(self):
        """Test setting the same source twice doesn't duplicate."""
        graph = SignalGraph()
        graph.set_source("signal-A", "entity-1")
        graph.set_source("signal-A", "entity-1")
        # Should still work, no duplicates
        assert graph.get_source("signal-A") == "entity-1"


class TestSignalGraphGetSource:
    """Tests for SignalGraph.get_source method."""

    def test_get_source_unknown_signal(self):
        """Test getting source for unknown signal returns None."""
        graph = SignalGraph()
        assert graph.get_source("unknown") is None

    def test_get_source_existing_signal(self):
        """Test getting source for existing signal."""
        graph = SignalGraph()
        graph.set_source("signal-A", "entity-1")
        assert graph.get_source("signal-A") == "entity-1"


class TestSignalGraphAddSink:
    """Tests for SignalGraph.add_sink method."""

    def test_add_sink_single(self):
        """Test adding a single sink."""
        graph = SignalGraph()
        graph.add_sink("signal-A", "entity-1")
        sinks = graph.iter_sinks("signal-A")
        assert sinks == ["entity-1"]

    def test_add_sink_multiple(self):
        """Test adding multiple sinks for same signal."""
        graph = SignalGraph()
        graph.add_sink("signal-A", "entity-1")
        graph.add_sink("signal-A", "entity-2")
        sinks = graph.iter_sinks("signal-A")
        assert set(sinks) == {"entity-1", "entity-2"}

    def test_add_sink_same_entity_twice(self):
        """Test adding same sink twice doesn't duplicate."""
        graph = SignalGraph()
        graph.add_sink("signal-A", "entity-1")
        graph.add_sink("signal-A", "entity-1")
        sinks = graph.iter_sinks("signal-A")
        assert sinks == ["entity-1"]


class TestSignalGraphRemoveSink:
    """Tests for SignalGraph.remove_sink method."""

    def test_remove_sink_existing(self):
        """Test removing an existing sink."""
        graph = SignalGraph()
        graph.add_sink("signal-A", "entity-1")
        graph.add_sink("signal-A", "entity-2")
        graph.remove_sink("signal-A", "entity-1")
        sinks = graph.iter_sinks("signal-A")
        assert sinks == ["entity-2"]

    def test_remove_sink_nonexistent(self):
        """Test removing a nonexistent sink does nothing."""
        graph = SignalGraph()
        graph.add_sink("signal-A", "entity-1")
        graph.remove_sink("signal-A", "entity-2")  # doesn't exist
        sinks = graph.iter_sinks("signal-A")
        assert sinks == ["entity-1"]

    def test_remove_sink_unknown_signal(self):
        """Test removing sink for unknown signal does nothing."""
        graph = SignalGraph()
        graph.remove_sink("unknown", "entity-1")  # should not raise
        assert graph.iter_sinks("unknown") == []


class TestSignalGraphIterSinks:
    """Tests for SignalGraph.iter_sinks method."""

    def test_iter_sinks_empty(self):
        """Test iter_sinks for unknown signal returns empty list."""
        graph = SignalGraph()
        assert graph.iter_sinks("unknown") == []

    def test_iter_sinks_returns_snapshot(self):
        """Test iter_sinks returns a snapshot (copy)."""
        graph = SignalGraph()
        graph.add_sink("signal-A", "entity-1")
        sinks = graph.iter_sinks("signal-A")
        sinks.append("entity-2")  # modify the returned list
        # Original should be unchanged
        assert graph.iter_sinks("signal-A") == ["entity-1"]


class TestSignalGraphSignals:
    """Tests for SignalGraph.signals method."""

    def test_signals_empty(self):
        """Test signals on empty graph."""
        graph = SignalGraph()
        assert graph.signals() == set()

    def test_signals_from_sources(self):
        """Test signals includes source signals."""
        graph = SignalGraph()
        graph.set_source("signal-A", "entity-1")
        assert "signal-A" in graph.signals()

    def test_signals_from_sinks(self):
        """Test signals includes sink signals."""
        graph = SignalGraph()
        graph.add_sink("signal-B", "entity-2")
        assert "signal-B" in graph.signals()

    def test_signals_combined(self):
        """Test signals returns union of sources and sinks."""
        graph = SignalGraph()
        graph.set_source("signal-A", "entity-1")
        graph.add_sink("signal-B", "entity-2")
        graph.set_source("signal-C", "entity-3")
        graph.add_sink("signal-C", "entity-4")  # signal-C has both
        assert graph.signals() == {"signal-A", "signal-B", "signal-C"}


class TestSignalGraphIterEdges:
    """Tests for SignalGraph.iter_edges method."""

    def test_iter_edges_empty(self):
        """Test iter_edges on empty graph."""
        graph = SignalGraph()
        edges = list(graph.iter_edges())
        assert edges == []

    def test_iter_edges_with_source_and_sinks(self):
        """Test iter_edges returns triples."""
        graph = SignalGraph()
        graph.set_source("signal-A", "source-1")
        graph.add_sink("signal-A", "sink-1")
        graph.add_sink("signal-A", "sink-2")
        edges = list(graph.iter_edges())
        assert len(edges) == 1
        signal_id, sources, sinks = edges[0]
        assert signal_id == "signal-A"
        assert sources == ["source-1"]
        assert set(sinks) == {"sink-1", "sink-2"}

    def test_iter_edges_multiple_sources(self):
        """Test iter_edges with multiple sources."""
        graph = SignalGraph()
        graph.set_source("signal-A", "source-1")
        graph.set_source("signal-A", "source-2")
        graph.add_sink("signal-A", "sink-1")
        edges = list(graph.iter_edges())
        assert len(edges) == 1
        signal_id, sources, sinks = edges[0]
        assert set(sources) == {"source-1", "source-2"}

    def test_iter_edges_sorted_by_signal(self):
        """Test iter_edges is sorted by signal id."""
        graph = SignalGraph()
        graph.add_sink("signal-Z", "sink-1")
        graph.add_sink("signal-A", "sink-2")
        graph.add_sink("signal-M", "sink-3")
        edges = list(graph.iter_edges())
        signal_ids = [e[0] for e in edges]
        assert signal_ids == ["signal-A", "signal-M", "signal-Z"]


class TestSignalGraphIterSourceSinkPairs:
    """Tests for SignalGraph.iter_source_sink_pairs method."""

    def test_iter_source_sink_pairs_empty(self):
        """Test iter_source_sink_pairs on empty graph."""
        graph = SignalGraph()
        pairs = list(graph.iter_source_sink_pairs())
        assert pairs == []

    def test_iter_source_sink_pairs_single(self):
        """Test iter_source_sink_pairs with single source and sink."""
        graph = SignalGraph()
        graph.set_source("signal-A", "source-1")
        graph.add_sink("signal-A", "sink-1")
        pairs = list(graph.iter_source_sink_pairs())
        assert pairs == [("signal-A", "source-1", "sink-1")]

    def test_iter_source_sink_pairs_multiple_sinks(self):
        """Test iter_source_sink_pairs with multiple sinks."""
        graph = SignalGraph()
        graph.set_source("signal-A", "source-1")
        graph.add_sink("signal-A", "sink-1")
        graph.add_sink("signal-A", "sink-2")
        pairs = list(graph.iter_source_sink_pairs())
        assert len(pairs) == 2
        assert ("signal-A", "source-1", "sink-1") in pairs
        assert ("signal-A", "source-1", "sink-2") in pairs

    def test_iter_source_sink_pairs_multiple_sources(self):
        """Test iter_source_sink_pairs with multiple sources (memory feedback)."""
        graph = SignalGraph()
        graph.set_source("signal-A", "write-gate")
        graph.set_source("signal-A", "hold-gate")
        graph.add_sink("signal-A", "consumer-1")
        pairs = list(graph.iter_source_sink_pairs())
        assert len(pairs) == 2
        # Both sources should produce pairs with the consumer
        assert ("signal-A", "write-gate", "consumer-1") in pairs
        assert ("signal-A", "hold-gate", "consumer-1") in pairs

    def test_iter_source_sink_pairs_cartesian_product(self):
        """Test iter_source_sink_pairs produces cartesian product."""
        graph = SignalGraph()
        graph.set_source("signal-A", "source-1")
        graph.set_source("signal-A", "source-2")
        graph.add_sink("signal-A", "sink-1")
        graph.add_sink("signal-A", "sink-2")
        pairs = list(graph.iter_source_sink_pairs())
        # 2 sources x 2 sinks = 4 pairs
        assert len(pairs) == 4

    def test_iter_source_sink_pairs_no_source(self):
        """Test iter_source_sink_pairs with no sources yields nothing."""
        graph = SignalGraph()
        graph.add_sink("signal-A", "sink-1")
        pairs = list(graph.iter_source_sink_pairs())
        assert pairs == []

    def test_iter_source_sink_pairs_sorted(self):
        """Test iter_source_sink_pairs is sorted by signal id."""
        graph = SignalGraph()
        graph.set_source("signal-Z", "source-z")
        graph.set_source("signal-A", "source-a")
        graph.add_sink("signal-Z", "sink-z")
        graph.add_sink("signal-A", "sink-a")
        pairs = list(graph.iter_source_sink_pairs())
        signal_ids = [p[0] for p in pairs]
        assert signal_ids == ["signal-A", "signal-Z"]
