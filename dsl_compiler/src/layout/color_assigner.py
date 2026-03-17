"""Wire color assignment: constraint collection and solving.

Determines wire colors (red/green) for all signal graph edges based on:
- Hard constraints (user-specified, memory feedback, bundle separation)
- Separation constraints (same signal at same sink, isolation, transitive merge)
- Merge constraints (edges in same merge group share color)

This runs BEFORE layout optimization so the placement solver knows
which connections share wire networks.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.ir.builder import BundleRef, SignalRef

from .layout_plan import LayoutPlan
from .signal_analyzer import SignalUsageEntry
from .wire_router import (
    WIRE_COLORS,
    ColorAssignment,
    WireColorSolver,
    WireEdge,
)


@dataclass
class WireColorResult:
    """Result of wire color assignment."""

    wire_edges: list[WireEdge]
    edge_colors: dict[tuple[str, str, str], str]  # (src, snk, sig) → "red"|"green"
    network_ids: dict[tuple[str, str, str], int]  # (src, snk, sig) → network_id
    is_bipartite: bool
    isolated_entities: set[str] = field(default_factory=set)


class WireColorAssigner:
    """Assigns wire colors to all signal graph edges.

    Runs before layout optimization. Wire colors depend on signal graph
    topology (separation, merge, isolation constraints), not on positions.
    """

    def __init__(
        self,
        layout_plan: LayoutPlan,
        signal_usage: dict[str, SignalUsageEntry],
        diagnostics: ProgramDiagnostics,
        memory_modules: dict[str, Any],
    ) -> None:
        self.layout_plan = layout_plan
        self.signal_usage = signal_usage
        self.diagnostics = diagnostics
        self._memory_modules = memory_modules
        self._isolated_entities: set[str] = set()

    def assign_colors(
        self,
        signal_graph: Any,
        wire_merge_junctions: dict[str, Any] | None = None,
        merge_membership: dict[str, set] | None = None,
    ) -> WireColorResult:
        """Assign wire colors to all signal edges.

        Returns WireColorResult with edge_colors mapping.
        """
        # Phase 1: collect edges
        edges = self._collect_edges(
            signal_graph, self.layout_plan.entity_placements, wire_merge_junctions
        )

        # Phase 2: build solver with all constraints
        solver = self._build_solver(
            edges,
            self.layout_plan.entity_placements,
            merge_membership or {},
            signal_graph,
        )

        # Phase 3: solve colors
        result = solver.solve()
        edge_colors = self._apply_color_result(result, edges)

        if not result.is_bipartite:
            for c in result.conflicts:
                self.diagnostics.info(
                    f"Wire coloring conflict: {c.reason} — "
                    f"{c.edge_a.signal_name} ({c.edge_a.source_entity_id}→{c.edge_a.sink_entity_id}) vs "
                    f"{c.edge_b.signal_name} ({c.edge_b.source_entity_id}→{c.edge_b.sink_entity_id})"
                )

        # Phase 4: compute network IDs
        network_ids = self._compute_network_ids(edges, edge_colors)

        return WireColorResult(
            wire_edges=edges,
            edge_colors=edge_colors,
            network_ids=network_ids,
            is_bipartite=result.is_bipartite,
            isolated_entities=set(self._isolated_entities),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Phase 1: edge collection
    # ──────────────────────────────────────────────────────────────────────

    def _collect_edges(
        self,
        signal_graph: Any,
        entities: dict[str, Any],
        wire_merge_junctions: dict[str, Any] | None,
    ) -> list[WireEdge]:
        """Collect all WireEdge instances from signal graph, expanding merges."""
        raw_edges: list[WireEdge] = []

        for logical_id, source_id, sink_id in signal_graph.iter_source_sink_pairs():
            usage = self.signal_usage.get(logical_id)
            resolved = (
                usage.resolved_signal_name if usage and usage.resolved_signal_name else logical_id
            )

            if self._is_internal_feedback_signal(resolved):
                continue

            if source_id and self._is_memory_feedback_edge(source_id, sink_id, resolved):
                continue

            raw_edges.append(
                WireEdge(
                    source_entity_id=source_id or "",
                    sink_entity_id=sink_id,
                    signal_name=resolved,
                    logical_signal_id=logical_id,
                )
            )

        if wire_merge_junctions:
            raw_edges = self._expand_merges(raw_edges, wire_merge_junctions, entities, signal_graph)

        return [e for e in raw_edges if e.source_entity_id]

    def _expand_merges(
        self,
        edges: list[WireEdge],
        junctions: dict[str, Any],
        entities: dict[str, Any],
        signal_graph: Any,
    ) -> list[WireEdge]:
        """Replace merge-junction edges with direct source→sink edges tagged with merge_group."""
        expanded: list[WireEdge] = []

        for edge in edges:
            if edge.sink_entity_id in junctions:
                continue

            merge_info = junctions.get(edge.source_entity_id)
            if not merge_info:
                expanded.append(edge)
                continue

            merge_group = edge.source_entity_id

            for source_ref in merge_info.get("inputs", []):
                if isinstance(source_ref, (SignalRef, BundleRef)):
                    ir_source = source_ref.source_id
                else:
                    continue

                actual_source = ir_source
                if signal_graph is not None:
                    resolved_entity = signal_graph.get_source(ir_source)
                    if resolved_entity:
                        actual_source = resolved_entity

                expanded.append(
                    WireEdge(
                        source_entity_id=actual_source,
                        sink_entity_id=edge.sink_entity_id,
                        signal_name=edge.signal_name,
                        logical_signal_id=edge.logical_signal_id,
                        merge_group=merge_group,
                    )
                )

        return expanded

    # ──────────────────────────────────────────────────────────────────────
    # Phase 2: constraint collection + solver setup
    # ──────────────────────────────────────────────────────────────────────

    def _build_solver(
        self,
        edges: list[WireEdge],
        entities: dict[str, Any],
        merge_membership: dict[str, set],
        signal_graph: Any,
    ) -> WireColorSolver:
        solver = WireColorSolver()

        for e in edges:
            solver.add_edge(e)

        self._add_hard_constraints(solver, edges, entities, signal_graph)
        self._collect_isolated_entities(entities)
        self._add_merge_constraints(solver, edges)
        self._add_separation_constraints(solver, edges, entities, merge_membership, signal_graph)

        return solver

    def _add_hard_constraints(
        self,
        solver: WireColorSolver,
        edges: list[WireEdge],
        entities: dict[str, Any],
        signal_graph: Any,
    ) -> None:
        """Add hard color constraints (user-specified, memory, feedback, bundle separation)."""

        # User-specified wire colors (highest priority, first-writer-wins)
        for entity_id, placement in self.layout_plan.entity_placements.items():
            wire_color = placement.properties.get("wire_color")
            if wire_color:
                for e in edges:
                    if e.source_entity_id == entity_id:
                        solver.add_hard_constraint(e, wire_color, "user-specified")

        # signal-W → GREEN (write-enable for gated memories)
        for e in edges:
            if e.signal_name == "signal-W":
                solver.add_hard_constraint(e, "green", "signal-W is memory write-enable")

        # Data signals feeding into gated memory gates → RED
        from .memory_builder import MemoryModule

        for module in self._memory_modules.values():
            if not isinstance(module, MemoryModule):
                continue
            if module.archetype == "gated" and module.secondary:
                gate_id = module.secondary.ir_node_id
                for e in edges:
                    if e.sink_entity_id == gate_id and e.signal_name == module.signal_type:
                        solver.add_hard_constraint(
                            e, "red", f"data signal to memory gate ({module.signal_type})"
                        )

        # Self-feedback → RED (for accumulator archetypes)
        for entity_id, placement in self.layout_plan.entity_placements.items():
            if placement.properties.get("has_self_feedback"):
                fb_signal = placement.properties.get("feedback_signal")
                if fb_signal:
                    self._lock_edges(
                        solver,
                        edges,
                        source=entity_id,
                        signal=fb_signal,
                        color="red",
                        reason="self-feedback",
                    )
                    for e in edges:
                        if e.sink_entity_id == entity_id and e.source_entity_id != entity_id:
                            solver.add_hard_constraint(
                                e,
                                "green",
                                f"accumulator external input separated from self-feedback ({fb_signal})",
                            )

        # Latch inputs → RED (GREEN reserved for self-loop feedback)
        for entity_id, placement in self.layout_plan.entity_placements.items():
            if getattr(placement, "role", None) == "latch":
                for e in edges:
                    if e.sink_entity_id == entity_id:
                        solver.add_hard_constraint(
                            e, "red", "latch input (GREEN reserved for self-loop)"
                        )

        # Bundle wire separation: needs_wire_separation
        for entity_id, placement in self.layout_plan.entity_placements.items():
            if not placement.properties.get("needs_wire_separation"):
                continue

            if placement.entity_type == "arithmetic-combinator":
                right_signal_id = placement.properties.get("right_operand_signal_id")
                right_operand = placement.properties.get("right_operand")
                if (
                    right_signal_id
                    and isinstance(right_operand, str)
                    and hasattr(right_signal_id, "source_id")
                ):
                    source_id = right_signal_id.source_id
                    actual = signal_graph.get_source(source_id) if signal_graph else source_id
                    if actual is None:
                        actual = source_id
                    for e in edges:
                        if e.source_entity_id == actual and e.sink_entity_id == entity_id:
                            solver.add_hard_constraint(e, "green", "bundle: scalar operand")
                    left_signal_id = placement.properties.get("left_operand_signal_id")
                    if left_signal_id and hasattr(left_signal_id, "source_id"):
                        left_source = left_signal_id.source_id
                        actual_left = (
                            signal_graph.get_source(left_source) if signal_graph else left_source
                        )
                        if actual_left is None:
                            actual_left = left_source
                        for e in edges:
                            if e.source_entity_id == actual_left and e.sink_entity_id == entity_id:
                                solver.add_hard_constraint(e, "red", "bundle: each operand")

            elif placement.entity_type == "decider-combinator":
                ov_signal_id = placement.properties.get("output_value_signal_id")
                if ov_signal_id and hasattr(ov_signal_id, "source_id"):
                    bundle_ir = ov_signal_id.source_id
                    actual_src = signal_graph.get_source(bundle_ir) if signal_graph else bundle_ir
                    if actual_src is None:
                        actual_src = bundle_ir
                    for e in edges:
                        if e.source_entity_id == actual_src and e.sink_entity_id == entity_id:
                            solver.add_hard_constraint(
                                e, "green", "bundle gating: bundle to decider"
                            )

        # Input bundle constants — heuristic color assignment
        self._add_bundle_const_heuristic(solver, edges, entities)

    def _add_bundle_const_heuristic(
        self,
        solver: WireColorSolver,
        edges: list[WireEdge],
        entities: dict[str, Any],
    ) -> None:
        """Assign heuristic colors to bundle constant combinators."""
        bundle_consts: list[tuple[str, bool]] = []
        for eid, placement in self.layout_plan.entity_placements.items():
            if (
                placement.entity_type == "constant-combinator"
                and getattr(placement, "role", None) == "bundle_const"
            ):
                signals = placement.properties.get("signals", {})
                has_nonzero = (
                    any(v != 0 for v in signals.values()) if isinstance(signals, dict) else False
                )
                bundle_consts.append((eid, has_nonzero))

        if not bundle_consts:
            return

        color_map: dict[str, str] = {}
        if len(bundle_consts) == 1:
            eid, has_nonzero = bundle_consts[0]
            color_map[eid] = "green" if has_nonzero else "red"
        elif len(bundle_consts) >= 2:
            nonzero = [eid for eid, nz in bundle_consts if nz]
            zero = [eid for eid, nz in bundle_consts if not nz]
            if nonzero and zero:
                for eid in nonzero:
                    color_map[eid] = "green"
                for eid in zero:
                    color_map[eid] = "red"
            else:
                # Sort by position if available, otherwise by entity ID for determinism
                sorted_b = sorted(
                    bundle_consts,
                    key=lambda x: (
                        (self.layout_plan.entity_placements[x[0]].position or (0, 0))[0],
                        x[0],
                    ),
                )
                colors = ["red", "green"]
                for i, (eid, _) in enumerate(sorted_b):
                    color_map[eid] = colors[i % 2]

        for e in edges:
            if e.source_entity_id in color_map:
                solver.add_hard_constraint(
                    e, color_map[e.source_entity_id], "bundle constant heuristic"
                )

    def _collect_isolated_entities(self, entities: dict[str, Any]) -> None:
        """Identify user-defined input constants and output anchors as isolated."""
        self._isolated_entities = set()
        for eid, placement in self.layout_plan.entity_placements.items():
            if (
                placement.properties.get("is_input")
                or placement.properties.get("is_output")
                or getattr(placement, "role", None) == "output_anchor"
            ):
                self._isolated_entities.add(eid)

    def _add_merge_constraints(
        self,
        solver: WireColorSolver,
        edges: list[WireEdge],
    ) -> None:
        """Group edges by merge_group and add merge constraints."""
        groups: dict[str, list[WireEdge]] = defaultdict(list)
        for e in edges:
            if e.merge_group:
                groups[e.merge_group].append(e)
        for merge_id, group_edges in sorted(groups.items()):
            if len(group_edges) >= 2:
                solver.add_merge(group_edges, merge_id)

    def _add_separation_constraints(
        self,
        solver: WireColorSolver,
        edges: list[WireEdge],
        entities: dict[str, Any],
        merge_membership: dict[str, set],
        signal_graph: Any,
    ) -> None:
        """Add separation constraints: same-signal-same-sink + isolation + transitive merge."""
        # 1. Same signal, same sink, different sources (not in same merge group) → separate
        sink_signal_groups: dict[tuple[str, str], list[WireEdge]] = defaultdict(list)
        for e in edges:
            sink_signal_groups[(e.sink_entity_id, e.signal_name)].append(e)

        for (_sink, _sig), group in sorted(sink_signal_groups.items()):
            if len(group) <= 1:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    if a.source_entity_id == b.source_entity_id:
                        continue
                    if a.merge_group and a.merge_group == b.merge_group:
                        continue
                    solver.add_separation(a, b, f"same signal '{_sig}' at sink '{_sink}'")

        # 2. Same-signal operand conflict (both operands read same Factorio signal)
        for eid, placement in self.layout_plan.entity_placements.items():
            left_signal = placement.properties.get("left_operand")
            right_signal = placement.properties.get("right_operand")
            if not left_signal or not right_signal:
                continue
            if isinstance(left_signal, int) or isinstance(right_signal, int):
                continue
            if left_signal != right_signal:
                continue
            left_id = placement.properties.get("left_operand_signal_id")
            right_id = placement.properties.get("right_operand_signal_id")
            if not left_id or not right_id:
                continue
            left_src = self._resolve_source_entity(left_id, signal_graph)
            right_src = self._resolve_source_entity(right_id, signal_graph)
            if not left_src or not right_src or left_src == right_src:
                continue
            left_edge = self._find_edge(edges, left_src, eid)
            right_edge = self._find_edge(edges, right_src, eid)
            if left_edge and right_edge:
                solver.add_separation(
                    left_edge,
                    right_edge,
                    f"same-signal operand conflict ({left_signal}) at {eid}",
                )

        # 2b. Multi-condition decider: same signal from different sources across conditions
        for eid, placement in self.layout_plan.entity_placements.items():
            conditions = placement.properties.get("conditions")
            if not conditions or len(conditions) < 2:
                continue
            cond_sources: list[tuple[str, str]] = []
            for cond in conditions:
                sig = cond.get("first_signal")
                sid = cond.get("first_operand_signal_id")
                if not sig or isinstance(sig, int) or not sid:
                    continue
                src = self._resolve_source_entity(sid, signal_graph)
                if src:
                    cond_sources.append((sig, src))
            for i in range(len(cond_sources)):
                for j in range(i + 1, len(cond_sources)):
                    sig_i, src_i = cond_sources[i]
                    sig_j, src_j = cond_sources[j]
                    if sig_i == sig_j and src_i != src_j:
                        edge_i = self._find_edge(edges, src_i, eid)
                        edge_j = self._find_edge(edges, src_j, eid)
                        if edge_i and edge_j:
                            solver.add_separation(
                                edge_i,
                                edge_j,
                                f"multi-condition same signal '{sig_i}' at {eid}",
                            )

        # 3. Stray signal contamination from multi-signal sources.
        # When source A is a bundle (entity output/wire merge producing multiple signals)
        # and another edge carries a specific signal to the same sink, they must be
        # separated. Otherwise the bundle's stray signals contaminate the intended
        # specific signal on the same wire, potentially creating feedback loops.
        # Also applies when source A has edges carrying different specific signals
        # (to other sinks) that collide with another edge's signal at the same sink.
        source_output_signals: dict[str, set[str]] = defaultdict(set)
        for e in edges:
            source_output_signals[e.source_entity_id].add(e.signal_name)

        sink_groups: dict[str, list[WireEdge]] = defaultdict(list)
        for e in edges:
            sink_groups[e.sink_entity_id].append(e)

        for sink_id, incoming in sorted(sink_groups.items()):
            if len(incoming) <= 1:
                continue
            for i in range(len(incoming)):
                for j in range(i + 1, len(incoming)):
                    a, b = incoming[i], incoming[j]
                    if a.source_entity_id == b.source_entity_id:
                        continue
                    if a.merge_group and a.merge_group == b.merge_group:
                        continue

                    # Case 1: bundle edge vs specific-signal edge — bundle may contain
                    # the specific signal, causing contamination
                    a_is_bundle = a.signal_name == "bundle"
                    b_is_bundle = b.signal_name == "bundle"
                    if (a_is_bundle and not b_is_bundle) or (b_is_bundle and not a_is_bundle):
                        bundle_src = a.source_entity_id if a_is_bundle else b.source_entity_id
                        specific = b if a_is_bundle else a
                        solver.add_separation(
                            a,
                            b,
                            f"stray signal: bundle source '{bundle_src}' "
                            f"may contaminate '{specific.signal_name}' at sink '{sink_id}'",
                        )
                        continue

                    # Case 2: both specific signals — check if source A produces
                    # signal B carries (or vice versa) via its other edges
                    a_stray = b.signal_name in source_output_signals.get(a.source_entity_id, set())
                    b_stray = a.signal_name in source_output_signals.get(b.source_entity_id, set())
                    if a_stray or b_stray:
                        stray_src = a.source_entity_id if a_stray else b.source_entity_id
                        stray_sig = b.signal_name if a_stray else a.signal_name
                        solver.add_separation(
                            a,
                            b,
                            f"stray signal: '{stray_src}' also produces "
                            f"'{stray_sig}' at sink '{sink_id}'",
                        )

        # 4. Isolation: user-defined inputs/outputs must not carry stray signals
        for e in edges:
            if e.merge_group:
                continue
            if e.source_entity_id in self._isolated_entities:
                for other in edges:
                    if other is e:
                        continue
                    if other.sink_entity_id != e.sink_entity_id:
                        continue
                    if other.merge_group and other.merge_group == e.merge_group:
                        continue
                    if other.source_entity_id == e.source_entity_id:
                        continue
                    solver.add_separation(
                        e,
                        other,
                        f"isolation: user input/output {e.source_entity_id}",
                    )
            if e.sink_entity_id in self._isolated_entities:
                for other in edges:
                    if other is e:
                        continue
                    if other.sink_entity_id != e.sink_entity_id:
                        continue
                    if other.source_entity_id == e.source_entity_id:
                        continue
                    solver.add_separation(
                        e,
                        other,
                        f"isolation: output anchor {e.sink_entity_id}",
                    )

        # 5. Transitive merge conflicts
        self._add_transitive_merge_constraints(solver, edges, merge_membership, signal_graph)

    def _add_transitive_merge_constraints(
        self,
        solver: WireColorSolver,
        edges: list[WireEdge],
        merge_membership: dict[str, set],
        signal_graph: Any,
    ) -> None:
        """When a source participates in multiple merges with transitive paths, separate them."""
        merge_to_sources: dict[str, set[str]] = defaultdict(set)
        merge_to_sinks: dict[str, set[str]] = defaultdict(set)
        for e in edges:
            if e.merge_group:
                merge_to_sources[e.merge_group].add(e.source_entity_id)
                merge_to_sinks[e.merge_group].add(e.sink_entity_id)

        for source_id, merge_ids in merge_membership.items():
            if len(merge_ids) <= 1:
                continue

            actual_source = source_id
            if signal_graph is not None:
                resolved = signal_graph.get_source(source_id)
                if resolved:
                    actual_source = resolved

            merge_list = sorted(merge_ids)
            has_conflict = False
            for i, m1 in enumerate(merge_list):
                sinks1 = merge_to_sinks.get(m1, set())
                for m2 in merge_list[i + 1 :]:
                    sources2 = merge_to_sources.get(m2, set())
                    sinks2 = merge_to_sinks.get(m2, set())
                    sources1 = merge_to_sources.get(m1, set())
                    if (sinks1 & sources2) or (sinks2 & sources1):
                        has_conflict = True
                        break
                if has_conflict:
                    break

            if not has_conflict:
                continue

            source_edges_by_merge: dict[str, list[WireEdge]] = defaultdict(list)
            for e in edges:
                if (
                    e.source_entity_id == actual_source
                    and e.merge_group is not None
                    and e.merge_group in merge_ids
                ):
                    source_edges_by_merge[e.merge_group].append(e)

            sorted_merges = sorted(source_edges_by_merge.keys())
            for i, m1 in enumerate(sorted_merges):
                for m2 in sorted_merges[i + 1 :]:
                    e1_list = source_edges_by_merge[m1]
                    e2_list = source_edges_by_merge[m2]
                    if e1_list and e2_list:
                        color_idx = sorted_merges.index(m1)
                        solver.add_hard_constraint(
                            e1_list[0],
                            WIRE_COLORS[color_idx % 2],
                            f"transitive merge conflict ({m1})",
                        )
                        color_idx2 = sorted_merges.index(m2)
                        solver.add_hard_constraint(
                            e2_list[0],
                            WIRE_COLORS[color_idx2 % 2],
                            f"transitive merge conflict ({m2})",
                        )

    # ──────────────────────────────────────────────────────────────────────
    # Phase 3: apply color result
    # ──────────────────────────────────────────────────────────────────────

    def _apply_color_result(
        self, result: ColorAssignment, edges: list[WireEdge]
    ) -> dict[tuple[str, str, str], str]:
        """Build edge_colors dict from the solver result."""
        edge_colors: dict[tuple[str, str, str], str] = {}
        for edge, color in result.edge_colors.items():
            edge_colors[edge.key] = color
            rev_key = (edge.sink_entity_id, edge.source_entity_id, edge.signal_name)
            if rev_key not in edge_colors:
                edge_colors[rev_key] = color

        color_counts = Counter(result.edge_colors.values())
        parts = [f"{c} {clr}" for clr, c in sorted(color_counts.items())]
        if parts:
            self.diagnostics.info("Wire color assignments: " + ", ".join(parts))

        return edge_colors

    # ──────────────────────────────────────────────────────────────────────
    # Phase 4: network IDs
    # ──────────────────────────────────────────────────────────────────────

    def _compute_network_ids(
        self,
        edges: list[WireEdge],
        edge_colors: dict[tuple[str, str, str], str],
    ) -> dict[tuple[str, str, str], int]:
        """Compute network IDs for relay isolation."""
        next_id = 1
        source_color_map: dict[tuple[str, str], int] = {}
        network_ids: dict[tuple[str, str, str], int] = {}
        for e in edges:
            color = edge_colors.get(e.key, "red")
            sc_key = (e.source_entity_id, color)
            if sc_key not in source_color_map:
                source_color_map[sc_key] = next_id
                next_id += 1
            network_ids[e.key] = source_color_map[sc_key]
        return network_ids

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_source_entity(self, signal_id: Any, signal_graph: Any) -> str | None:
        """Resolve a signal reference to a physical entity ID."""
        candidate = None
        signal_key = None

        if isinstance(signal_id, str) and "@" in signal_id:
            parts = signal_id.split("@")
            candidate = parts[1]
            signal_key = candidate
        elif hasattr(signal_id, "source_id"):
            candidate = signal_id.source_id
            signal_key = candidate

        if candidate and candidate in self.layout_plan.entity_placements:
            return candidate

        if signal_key and signal_graph is not None:
            all_sources = signal_graph._sources.get(signal_key, [])
            for src in all_sources:
                if src in self.layout_plan.entity_placements:
                    return src

        return None

    def _find_edge(self, edges: list[WireEdge], source_id: str, sink_id: str) -> WireEdge | None:
        for e in edges:
            if e.source_entity_id == source_id and e.sink_entity_id == sink_id:
                return e
        return None

    @staticmethod
    def _lock_edges(
        solver: WireColorSolver,
        edges: list[WireEdge],
        source: str,
        signal: str,
        color: str,
        reason: str,
    ) -> None:
        """Lock all edges from `source` carrying `signal` to a color."""
        for e in edges:
            if e.source_entity_id == source and e.signal_name == signal:
                solver.add_hard_constraint(e, color, reason)

    def _is_memory_feedback_edge(self, source_id: str, sink_id: str, signal_name: str) -> bool:
        """Check if an edge is a memory feedback connection (handled by explicit wires)."""
        from .memory_builder import MemoryModule

        if self._is_internal_feedback_signal(signal_name):
            return True

        placement = self.layout_plan.get_placement(source_id)
        if (
            placement
            and source_id == sink_id
            and placement.properties.get("has_self_feedback")
            and placement.properties.get("feedback_signal") == signal_name
        ):
            return True

        for module in self._memory_modules.values():
            if not isinstance(module, MemoryModule) or module.archetype != "gated":
                continue
            gate_id = module.secondary.ir_node_id if module.secondary else None
            storage_id = module.primary.ir_node_id if module.primary else None
            if (
                source_id in (gate_id, storage_id)
                and sink_id == storage_id
                and signal_name == module.signal_type
            ):
                return True
        return False

    @staticmethod
    def _is_internal_feedback_signal(signal_name: str) -> bool:
        """Any __feedback_ prefixed signal is an internal layout-only edge."""
        return signal_name.startswith("__feedback_")
