"""Connection planning: constraint collection, color solving, MST, relay routing.

Orchestrates the full wire connection pipeline:
1. Collect WireEdge instances from the signal graph
2. Collect constraints (hard color, separation, merge, isolation)
3. Solve wire colors via WireColorSolver
4. Optimize fan-out routing via MST
5. Route long-distance connections through relay poles
6. Inject operand wire colors into combinator placements
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from dsl_compiler.src.common.constants import DEFAULT_CONFIG, CompilerConfig
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.common.entity_data import is_dual_circuit_connectable
from dsl_compiler.src.common.signals import WILDCARD_SIGNALS
from dsl_compiler.src.ir.builder import BundleRef, SignalRef

from .layout_plan import LayoutPlan, WireConnection
from .signal_analyzer import SignalUsageEntry
from .tile_grid import TileGrid
from .wire_router import (
    WIRE_COLORS,
    ColorAssignment,
    WireColorSolver,
    WireEdge,
)

# ──────────────────────────────────────────────────────────────────────────
# Relay infrastructure  (kept from old implementation, mostly unchanged)
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class RelayNode:
    """A relay pole for routing circuit signals."""

    position: tuple[float, float]
    entity_id: str
    pole_type: str
    networks_red: set[int] = field(default_factory=set)
    networks_green: set[int] = field(default_factory=set)

    def can_route_network(self, network_id: int, wire_color: str) -> bool:
        networks = self.networks_red if wire_color == "red" else self.networks_green
        return len(networks) == 0 or network_id in networks

    def add_network(self, network_id: int, wire_color: str) -> None:
        if wire_color == "red":
            self.networks_red.add(network_id)
        else:
            self.networks_green.add(network_id)


class RelayNetwork:
    """Manages shared relay infrastructure for inter-cluster routing."""

    def __init__(
        self,
        tile_grid: TileGrid,
        max_span: float,
        layout_plan: LayoutPlan,
        diagnostics: ProgramDiagnostics,
        config: CompilerConfig = DEFAULT_CONFIG,
    ):
        self.tile_grid = tile_grid
        self.max_span = max_span
        self.layout_plan = layout_plan
        self.diagnostics = diagnostics
        self.config = config
        self.relay_nodes: dict[tuple[int, int], RelayNode] = {}
        self._relay_counter = 0

    @property
    def span_limit(self) -> float:
        return float(self.max_span)

    def add_relay_node(
        self, position: tuple[float, float], entity_id: str, pole_type: str
    ) -> RelayNode:
        tile_pos = (int(math.floor(position[0])), int(math.floor(position[1])))
        if tile_pos in self.relay_nodes:
            return self.relay_nodes[tile_pos]
        node = RelayNode(position, entity_id, pole_type)
        self.relay_nodes[tile_pos] = node
        return node

    def route_signal(
        self,
        source_pos: tuple[float, float],
        sink_pos: tuple[float, float],
        signal_name: str,
        wire_color: str,
        network_id: int = 0,
    ) -> list[tuple[str, str]] | None:
        """Find or create relay path.  Returns list of (relay_id, wire_color)."""
        distance = math.dist(source_pos, sink_pos)
        if distance <= self.span_limit:
            return []

        existing_path = self._find_existing_path(source_pos, sink_pos, wire_color, network_id)
        if existing_path is not None:
            for relay_id, relay_color in existing_path:
                node = self._get_node_by_id(relay_id)
                if node:
                    node.add_network(network_id, relay_color)
            return existing_path

        return self._create_relay_path(source_pos, sink_pos, signal_name, wire_color, network_id)

    # -- internal helpers ---------------------------------------------------

    def _find_existing_path(
        self,
        source_pos: tuple[float, float],
        sink_pos: tuple[float, float],
        wire_color: str,
        network_id: int,
    ) -> list[tuple[str, str]] | None:
        import heapq

        nodes: dict[str, tuple[float, float]] = {
            "__source__": source_pos,
            "__sink__": sink_pos,
        }
        for node in self.relay_nodes.values():
            if node.can_route_network(network_id, wire_color):
                nodes[node.entity_id] = node.position

        open_set: list[tuple[float, float, str, list[str]]] = [(0.0, 0.0, "__source__", [])]
        visited: set[str] = set()

        while open_set:
            _, g, current, path = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            current_pos = nodes[current]

            if math.dist(current_pos, sink_pos) <= self.span_limit:
                return [(rid, wire_color) for rid in path]

            for node_id, node_pos in nodes.items():
                if node_id in ("__source__", "__sink__") or node_id in visited:
                    continue
                dist = math.dist(current_pos, node_pos)
                if dist > self.span_limit:
                    continue
                new_g = g + dist
                heapq.heappush(
                    open_set,
                    (new_g + math.dist(node_pos, sink_pos), new_g, node_id, path + [node_id]),
                )
        return None

    def _create_relay_path(
        self,
        source_pos: tuple[float, float],
        sink_pos: tuple[float, float],
        signal_name: str,
        wire_color: str,
        network_id: int,
    ) -> list[tuple[str, str]] | None:
        distance = math.dist(source_pos, sink_pos)
        step_size = self.span_limit * 0.8
        path: list[tuple[str, str]] = []
        current_pos = source_pos

        for _ in range(int(math.ceil(distance / step_size)) + 5):
            remaining = math.dist(current_pos, sink_pos)
            if remaining <= self.span_limit:
                break

            dir_x = (sink_pos[0] - current_pos[0]) / remaining
            dir_y = (sink_pos[1] - current_pos[1]) / remaining
            step = min(step_size, remaining * 0.6)
            ideal = (current_pos[0] + dir_x * step, current_pos[1] + dir_y * step)

            relay = self._find_or_create_relay(
                ideal, current_pos, sink_pos, signal_name, wire_color, network_id
            )
            if relay is None:
                return None
            if math.dist(current_pos, relay.position) > self.span_limit:
                return None

            relay.add_network(network_id, wire_color)
            path.append((relay.entity_id, wire_color))
            current_pos = relay.position

        if math.dist(current_pos, sink_pos) > self.span_limit:
            return None
        return path

    def _find_or_create_relay(
        self,
        ideal: tuple[float, float],
        source_pos: tuple[float, float],
        sink_pos: tuple[float, float],
        signal_name: str,
        wire_color: str,
        network_id: int,
    ) -> RelayNode | None:
        # Try existing relays near ideal position
        for node in self.relay_nodes.values():
            if (
                math.dist(node.position, ideal) <= 3.0
                and math.dist(node.position, source_pos) <= self.span_limit
                and node.can_route_network(network_id, wire_color)
            ):
                return node

        return self._create_relay(ideal, source_pos, sink_pos, signal_name)

    def _create_relay(
        self,
        ideal: tuple[float, float],
        source_pos: tuple[float, float],
        sink_pos: tuple[float, float],
        signal_name: str,
    ) -> RelayNode | None:
        tile = (int(round(ideal[0])), int(round(ideal[1])))
        if self.tile_grid.reserve_exact(tile, footprint=(1, 1)):
            return self._finalize(tile, signal_name, ideal)

        candidates: list[tuple[float, tuple[int, int]]] = []
        for dx in range(-6, 7):
            for dy in range(-6, 7):
                if dx == 0 and dy == 0:
                    continue
                pos = (tile[0] + dx, tile[1] + dy)
                if not self.tile_grid.is_available(pos, footprint=(1, 1)):
                    continue
                center = (pos[0] + 0.5, pos[1] + 0.5)
                if math.dist(center, source_pos) > self.span_limit:
                    continue
                score = math.dist(center, ideal) + 2.0 * math.dist(center, sink_pos)
                candidates.append((score, pos))

        candidates.sort()
        for _, cpos in candidates:
            if self.tile_grid.reserve_exact(cpos, footprint=(1, 1)):
                return self._finalize(cpos, signal_name, ideal)
        return None

    def _finalize(
        self, tile: tuple[int, int], signal_name: str, ideal: tuple[float, float]
    ) -> RelayNode:
        self._relay_counter += 1
        relay_id = f"__relay_{self._relay_counter}"
        center = (tile[0] + 0.5, tile[1] + 0.5)
        self.diagnostics.info(
            f"Creating relay {relay_id} at {center} for {signal_name} (ideal {ideal})"
        )
        node = self.add_relay_node(center, relay_id, "medium-electric-pole")
        self.layout_plan.create_and_add_placement(
            ir_node_id=relay_id,
            entity_type="medium-electric-pole",
            position=center,
            footprint=(1, 1),
            role="wire_relay",
            debug_info={
                "variable": f"relay_{self._relay_counter}",
                "operation": "infrastructure",
                "details": "wire_relay",
                "role": "relay",
            },
        )
        return node

    def _get_node_by_id(self, relay_id: str) -> RelayNode | None:
        for node in self.relay_nodes.values():
            if node.entity_id == relay_id:
                return node
        return None


# ──────────────────────────────────────────────────────────────────────────
# ConnectionPlanner — the main coordinator
# ──────────────────────────────────────────────────────────────────────────


class ConnectionPlanner:
    """Plan all wire connections for a blueprint.

    Single entry point: ``plan_connections()``.

    Internally orchestrates:
    1. Edge collection  (signal graph → WireEdge list)
    2. Constraint collection  (hard / separation / merge / isolation)
    3. Color solving  (WireColorSolver)
    4. MST optimization  (fan-out routing)
    5. Relay routing  (long-distance connections)
    6. Operand wire injection  (combinator wire filters)
    """

    def __init__(
        self,
        layout_plan: LayoutPlan,
        signal_usage: dict[str, SignalUsageEntry],
        diagnostics: ProgramDiagnostics,
        tile_grid: TileGrid,
        *,
        max_wire_span: float = 9.0,
        power_pole_type: str | None = None,
        config: CompilerConfig = DEFAULT_CONFIG,
        use_mst_optimization: bool = True,
    ) -> None:
        self.layout_plan = layout_plan
        self.signal_usage = signal_usage
        self.diagnostics = diagnostics
        self.tile_grid = tile_grid
        self.max_wire_span = max_wire_span
        self.power_pole_type = power_pole_type
        self.config = config
        self.use_mst_optimization = use_mst_optimization

        # Populated during plan_connections
        self._wire_edges: list[WireEdge] = []
        self._edge_wire_colors: dict[tuple[str, str, str], str] = {}
        self._edge_network_ids: dict[tuple[str, str, str], int] = {}
        self._routing_failed = False
        self._isolated_entities: set[str] = set()
        self._memory_modules: dict[str, Any] = {}

        self.relay_network = RelayNetwork(
            tile_grid,
            max_wire_span,
            layout_plan,
            diagnostics,
            config,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def plan_connections(
        self,
        signal_graph: Any,
        entities: dict[str, Any],
        wire_merge_junctions: dict[str, Any] | None = None,
        merge_membership: dict[str, set] | None = None,
    ) -> bool:
        """Compute all wire connections.  Returns True on success."""
        self._register_power_poles_as_relays()
        self._add_self_feedback_connections()

        preserved = list(self.layout_plan.wire_connections)
        self.layout_plan.wire_connections.clear()
        self._wire_edges = []
        self._edge_wire_colors = {}
        self._routing_failed = False

        # Phase 1: collect edges
        edges = self._collect_edges(signal_graph, entities, wire_merge_junctions)
        self._wire_edges = edges

        # Phase 2: build solver with all constraints
        solver = self._build_solver(edges, entities, merge_membership or {}, signal_graph)

        # Phase 3: solve colors
        result = solver.solve()
        self._apply_color_result(result, edges)

        if not result.is_bipartite:
            for c in result.conflicts:
                self.diagnostics.info(
                    f"Wire coloring conflict: {c.reason} — "
                    f"{c.edge_a.signal_name} ({c.edge_a.source_entity_id}→{c.edge_a.sink_entity_id}) vs "
                    f"{c.edge_b.signal_name} ({c.edge_b.source_entity_id}→{c.edge_b.sink_entity_id})"
                )

        # Phase 4+5: MST optimization + relay routing → physical connections
        self._compute_network_ids()
        self._create_physical_connections()

        if preserved:
            self.layout_plan.wire_connections.extend(preserved)

        # Phase 6: operand wire injection
        self._inject_operand_wires(signal_graph)

        self._validate_relay_coverage()
        return not self._routing_failed

    def get_wire_color_for_edge(
        self, source_entity_id: str, sink_entity_id: str, signal_name: str
    ) -> str:
        """Lookup wire color for a specific edge.  Default: "red"."""
        return self._edge_wire_colors.get((source_entity_id, sink_entity_id, signal_name), "red")

    def get_wire_color_for_entity_pair(
        self, source_entity_id: str, sink_entity_id: str
    ) -> str | None:
        """Lookup wire color for ANY edge between two entities."""
        for (src, snk, _), color in self._edge_wire_colors.items():
            if src == source_entity_id and snk == sink_entity_id:
                return color
        return None

    def edge_color_map(self) -> dict[tuple[str, str, str], str]:
        return dict(self._edge_wire_colors)

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

            # Skip internal feedback signals
            if self._is_internal_feedback_signal(resolved):
                continue

            # Skip memory feedback edges (handled separately)
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

        # Expand merge junctions
        if wire_merge_junctions:
            raw_edges = self._expand_merges(raw_edges, wire_merge_junctions, entities, signal_graph)

        # Filter out edges without a real source
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
            # Skip edges whose sink IS a merge junction (they'll be replaced)
            if edge.sink_entity_id in junctions:
                continue

            # Check if the source is a merge junction
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

        # 2a: hard constraints
        self._add_hard_constraints(solver, edges, entities, signal_graph)

        # 2b: isolation constraints (collect isolated entity set)
        self._collect_isolated_entities(entities)

        # 2c: merge constraints
        self._add_merge_constraints(solver, edges)

        # 2d: separation constraints (including isolation-aware ones)
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
        from .memory_builder import MemoryModule

        # -- User-specified wire colors (highest priority, first-writer-wins) --
        for entity_id, placement in self.layout_plan.entity_placements.items():
            wire_color = placement.properties.get("wire_color")
            if wire_color:
                for e in edges:
                    if e.source_entity_id == entity_id:
                        solver.add_hard_constraint(e, wire_color, "user-specified")

        # -- Memory data signals → RED, signal-W → GREEN --
        for module in self._memory_modules.values():
            if isinstance(module, MemoryModule) and module.optimization is None:
                if module.write_gate:
                    self._lock_edges(
                        solver,
                        edges,
                        source=module.write_gate.ir_node_id,
                        signal=module.signal_type,
                        color="red",
                        reason="memory data (write gate)",
                    )
                if module.hold_gate:
                    self._lock_edges(
                        solver,
                        edges,
                        source=module.hold_gate.ir_node_id,
                        signal=module.signal_type,
                        color="red",
                        reason="memory data (hold gate)",
                    )

        # Pass-through memories: output signal locked to GREEN
        for module in self._memory_modules.values():
            if isinstance(module, MemoryModule) and module.optimization == "pass_through":  # noqa: SIM102
                if module.output_node_id:
                    self._lock_edges(
                        solver,
                        edges,
                        source=module.output_node_id,
                        signal=module.signal_type,
                        color="green",
                        reason="pass-through memory output",
                    )

        # signal-W → GREEN
        for e in edges:
            if e.signal_name == "signal-W":
                solver.add_hard_constraint(e, "green", "signal-W is memory write-enable")

        # Data signals feeding into write gates → RED
        for module in self._memory_modules.values():
            if not isinstance(module, MemoryModule) or module.optimization is not None:
                continue
            if not module.write_gate or not module.hold_gate:
                continue
            write_gate_id = module.write_gate.ir_node_id
            hold_gate_id = module.hold_gate.ir_node_id
            data_signal = module.signal_type
            for e in edges:
                if (
                    e.sink_entity_id == write_gate_id
                    and e.signal_name == data_signal
                    and e.source_entity_id != write_gate_id
                    and e.source_entity_id != hold_gate_id
                ):
                    solver.add_hard_constraint(
                        e, "red", f"data signal to write gate ({data_signal})"
                    )

        # Self-feedback → RED
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

        # Bundle wire separation: needs_wire_separation
        for entity_id, placement in self.layout_plan.entity_placements.items():
            if not placement.properties.get("needs_wire_separation"):
                continue

            if placement.entity_type == "arithmetic-combinator":
                # Lock the right (scalar) operand to GREEN
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
                    # Lock left (bundle) edges to RED
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
                # Lock the output_value (bundle) to GREEN
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

        # Assign colors
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
                sorted_b = sorted(
                    bundle_consts,
                    key=lambda x: (self.layout_plan.entity_placements[x[0]].position or (0, 0))[0],
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
            # Build non-merge pairs
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    if a.source_entity_id == b.source_entity_id:
                        continue
                    if a.merge_group and a.merge_group == b.merge_group:
                        continue  # same merge → they SHOULD be on same wire
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
            # Resolve to entity IDs
            left_src = self._resolve_source_entity(left_id, signal_graph)
            right_src = self._resolve_source_entity(right_id, signal_graph)
            if not left_src or not right_src or left_src == right_src:
                continue
            # Find the corresponding edges
            left_edge = self._find_edge(edges, left_src, eid)
            right_edge = self._find_edge(edges, right_src, eid)
            if left_edge and right_edge:
                solver.add_separation(
                    left_edge,
                    right_edge,
                    f"same-signal operand conflict ({left_signal}) at {eid}",
                )

        # 3. Isolation: user-defined inputs/outputs must not carry stray signals
        for e in edges:
            if e.merge_group:
                continue  # merge edges are exempt from isolation
            if e.source_entity_id in self._isolated_entities:
                # This edge originates from an isolated entity.
                # Separate it from all other edges arriving at the same sink
                # on ANY signal (not just same signal).
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
                # This edge goes to an isolated entity (output anchor).
                # Separate it from all other edges arriving at the same sink.
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

        # 4. Transitive merge conflicts
        self._add_transitive_merge_constraints(solver, edges, merge_membership, signal_graph)

    def _add_transitive_merge_constraints(
        self,
        solver: WireColorSolver,
        edges: list[WireEdge],
        merge_membership: dict[str, set],
        signal_graph: Any,
    ) -> None:
        """When a source participates in multiple merges with transitive paths, separate them."""
        # Build maps
        merge_to_sources: dict[str, set[str]] = defaultdict(set)
        merge_to_sinks: dict[str, set[str]] = defaultdict(set)
        for e in edges:
            if e.merge_group:
                merge_to_sources[e.merge_group].add(e.source_entity_id)
                merge_to_sinks[e.merge_group].add(e.sink_entity_id)

        # For each source in multiple merges, check for transitive paths
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

            # Separate edges from this source across different merge groups
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
                    # Separate the first edge of each group (representative)
                    e1_list = source_edges_by_merge[m1]
                    e2_list = source_edges_by_merge[m2]
                    if e1_list and e2_list:
                        # Hard-lock to alternating colors for determinism
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

    def _apply_color_result(self, result: ColorAssignment, edges: list[WireEdge]) -> None:
        """Populate _edge_wire_colors from the solver result."""
        for edge, color in result.edge_colors.items():
            self._edge_wire_colors[edge.key] = color
            # Also store reverse for bidirectional lookups
            rev_key = (edge.sink_entity_id, edge.source_entity_id, edge.signal_name)
            if rev_key not in self._edge_wire_colors:
                self._edge_wire_colors[rev_key] = color

        color_counts = Counter(result.edge_colors.values())
        parts = [f"{c} {clr}" for clr, c in sorted(color_counts.items())]
        if parts:
            self.diagnostics.info("Wire color assignments: " + ", ".join(parts))

    # ──────────────────────────────────────────────────────────────────────
    # Phase 4+5: physical connection creation (MST + relay)
    # ──────────────────────────────────────────────────────────────────────

    def _compute_network_ids(self) -> None:
        """Compute network IDs for relay isolation."""
        next_id = 1
        source_color_map: dict[tuple[str, str], int] = {}
        for e in self._wire_edges:
            color = self._edge_wire_colors.get(e.key, "red")
            sc_key = (e.source_entity_id, color)
            if sc_key not in source_color_map:
                source_color_map[sc_key] = next_id
                next_id += 1
            self._edge_network_ids[e.key] = source_color_map[sc_key]

    def _create_physical_connections(self) -> None:
        """Group edges by (signal, color), apply MST where beneficial, route through relays."""
        # Group edges
        signal_groups: dict[tuple[str, str], list[WireEdge]] = defaultdict(list)
        for e in self._wire_edges:
            color = self._edge_wire_colors.get(e.key, "red")
            signal_groups[(e.signal_name, color)].append(e)

        for (_sig, wire_color), group_edges in sorted(signal_groups.items()):
            # Find bidirectional pairs
            pair_set = {(e.source_entity_id, e.sink_entity_id) for e in group_edges}
            bidir = set()
            for e in group_edges:
                if (e.sink_entity_id, e.source_entity_id) in pair_set:
                    bidir.add((e.source_entity_id, e.sink_entity_id))
                    bidir.add((e.sink_entity_id, e.source_entity_id))

            # Group by source
            by_source: dict[str, list[WireEdge]] = defaultdict(list)
            for e in group_edges:
                by_source[e.source_entity_id].append(e)

            for source_id, src_edges in sorted(by_source.items()):
                safe = [e for e in src_edges if (e.source_entity_id, e.sink_entity_id) not in bidir]
                bidir_edges = [
                    e for e in src_edges if (e.source_entity_id, e.sink_entity_id) in bidir
                ]

                mst_ok = False
                if self.use_mst_optimization and len(safe) >= 2:
                    safe_sinks = [e.sink_entity_id for e in safe]
                    mst_ok = self._try_mst(source_id, safe_sinks, _sig, wire_color)

                if not mst_ok:
                    for e in safe:
                        self._route_edge(e, wire_color)

                for e in bidir_edges:
                    self._route_edge(e, wire_color)

    def _try_mst(
        self, source_id: str, sink_ids: list[str], signal_name: str, wire_color: str
    ) -> bool:
        """Build MST for fan-out, excluding isolated entities as intermediates."""
        all_entities = [source_id] + sorted(set(sink_ids))
        mst_edges = self._build_mst(all_entities)

        if not mst_edges:
            return False

        # Validate source is connected
        if not any(source_id in edge for edge in mst_edges):
            return False

        # Pre-validate: all MST edges within span
        span = self.relay_network.span_limit
        for a, b in mst_edges:
            pa = self.layout_plan.get_placement(a)
            pb = self.layout_plan.get_placement(b)
            if not pa or not pb or not pa.position or not pb.position:
                return False
            if math.dist(pa.position, pb.position) > span:
                return False

        # Check isolation: MST may route through an isolated entity as intermediate.
        # Isolated entities should only be leaf nodes in the MST.
        for a, b in mst_edges:
            # Check if an intermediate node (not source, not a final sink) is isolated
            if a != source_id and a not in sink_ids and a in self._isolated_entities:
                return False
            if b != source_id and b not in sink_ids and b in self._isolated_entities:
                return False

        # Register logical edges
        for sid in sink_ids:
            self._edge_wire_colors[(source_id, sid, signal_name)] = wire_color
            if (sid, source_id, signal_name) not in self._edge_wire_colors:
                self._edge_wire_colors[(sid, source_id, signal_name)] = wire_color

        # Route MST edges
        network_id = self._edge_network_ids.get((source_id, sink_ids[0], signal_name), 0)
        all_ok = True
        for a, b in mst_edges:
            side_a = self._get_connection_side(a, is_source=(a == source_id))
            side_b = self._get_connection_side(b, is_source=(b == source_id))
            # Store MST edge colors (don't overwrite existing)
            for k in ((a, b, signal_name), (b, a, signal_name)):
                if k not in self._edge_wire_colors:
                    self._edge_wire_colors[k] = wire_color
            if not self._route_connection(
                a, b, signal_name, wire_color, side_a, side_b, network_id
            ):
                all_ok = False

        return all_ok

    def _build_mst(self, entity_ids: list[str]) -> list[tuple[str, str]]:
        """Prim's MST over entities."""
        if len(entity_ids) <= 1:
            return []

        positions: dict[str, tuple[float, float]] = {}
        for eid in entity_ids:
            p = self.layout_plan.get_placement(eid)
            if p and p.position:
                positions[eid] = p.position

        valid = [e for e in entity_ids if e in positions]
        if len(valid) <= 1:
            return []

        in_tree = {valid[0]}
        result: list[tuple[str, str]] = []

        while len(in_tree) < len(valid):
            best: tuple[str, str] | None = None
            best_dist = float("inf")
            for t in sorted(in_tree):
                for c in valid:
                    if c in in_tree:
                        continue
                    d = math.dist(positions[t], positions[c])
                    if d < best_dist or (d == best_dist and (t, c) < (best or ("", ""))):
                        best_dist = d
                        best = (t, c)
            if best is None:
                break
            result.append(best)
            in_tree.add(best[1])

        return result

    def _route_edge(self, edge: WireEdge, wire_color: str) -> None:
        """Route a single edge directly."""
        self._edge_wire_colors[edge.key] = wire_color
        source_side = self._get_connection_side(edge.source_entity_id, is_source=True)
        sink_side = self._get_connection_side(edge.sink_entity_id, is_source=False)
        network_id = self._edge_network_ids.get(edge.key, 0)
        self._route_connection(
            edge.source_entity_id,
            edge.sink_entity_id,
            edge.signal_name,
            wire_color,
            source_side,
            sink_side,
            network_id,
        )

    def _route_connection(
        self,
        source_id: str,
        sink_id: str,
        signal_name: str,
        wire_color: str,
        source_side: str | None,
        sink_side: str | None,
        network_id: int = 0,
    ) -> bool:
        """Route connection, using relays if needed."""
        src = self.layout_plan.get_placement(source_id)
        snk = self.layout_plan.get_placement(sink_id)
        if not src or not snk or not src.position or not snk.position:
            return True  # skip silently

        relay_path = self.relay_network.route_signal(
            src.position,
            snk.position,
            signal_name,
            wire_color,
            network_id,
        )
        if relay_path is None:
            self.diagnostics.warning(
                f"Relay routing failed for '{signal_name}' between {source_id} and {sink_id}"
            )
            self._routing_failed = True
            return False

        self._create_relay_chain(
            source_id, sink_id, signal_name, wire_color, relay_path, source_side, sink_side
        )
        return True

    def _create_relay_chain(
        self,
        source_id: str,
        sink_id: str,
        signal_name: str,
        wire_color: str,
        relay_path: list[tuple[str, str]],
        source_side: str | None = None,
        sink_side: str | None = None,
    ) -> None:
        if not relay_path:
            self.layout_plan.add_wire_connection(
                WireConnection(
                    source_entity_id=source_id,
                    sink_entity_id=sink_id,
                    signal_name=signal_name,
                    wire_color=wire_color,
                    source_side=source_side,
                    sink_side=sink_side,
                )
            )
        else:
            cur_id = source_id
            cur_side = source_side
            for relay_id, relay_color in relay_path:
                self.layout_plan.add_wire_connection(
                    WireConnection(
                        source_entity_id=cur_id,
                        sink_entity_id=relay_id,
                        signal_name=signal_name,
                        wire_color=relay_color,
                        source_side=cur_side,
                        sink_side=None,
                    )
                )
                cur_id = relay_id
                cur_side = None
            self.layout_plan.add_wire_connection(
                WireConnection(
                    source_entity_id=cur_id,
                    sink_entity_id=sink_id,
                    signal_name=signal_name,
                    wire_color=relay_path[-1][1],
                    source_side=None,
                    sink_side=sink_side,
                )
            )

    # ──────────────────────────────────────────────────────────────────────
    # Phase 6: operand wire injection
    # ──────────────────────────────────────────────────────────────────────

    def _inject_operand_wires(self, signal_graph: Any) -> None:
        """Set operand wire filters on combinator placements."""
        injected = 0
        for placement in self.layout_plan.entity_placements.values():
            if placement.entity_type not in ("arithmetic-combinator", "decider-combinator"):
                continue

            injected += self._inject_operand(placement, "left", signal_graph)
            injected += self._inject_operand(placement, "right", signal_graph)
            injected += self._inject_condition_wires(placement, signal_graph)

            if placement.entity_type == "decider-combinator":
                injected += self._inject_output_value(placement, signal_graph)

        self.diagnostics.info(f"Wire color injection: {injected} operands configured")

    def _inject_operand(self, placement: Any, side: str, signal_graph: Any) -> int:
        signal = placement.properties.get(f"{side}_operand")
        signal_id = placement.properties.get(f"{side}_operand_signal_id")
        if not signal or isinstance(signal, int) or not signal_id:
            return 0

        source = self._resolve_source_entity(signal_id, signal_graph)
        if not source:
            placement.properties[f"{side}_operand_wires"] = {"red", "green"}
            return 0

        color = self._lookup_color(source, placement.ir_node_id, signal)
        placement.properties[f"{side}_operand_wires"] = {color}
        return 1

    def _inject_output_value(self, placement: Any, signal_graph: Any) -> int:
        if not placement.properties.get("copy_count_from_input", False):
            return 0

        output_value = placement.properties.get("output_value")
        signal_id = placement.properties.get("output_value_signal_id")
        if not output_value or isinstance(output_value, int) or not signal_id:
            return 0

        source = self._resolve_source_entity(signal_id, signal_graph)
        if not source:
            placement.properties["output_value_wires"] = {"red", "green"}
            return 0

        # Try output_value signal, then bundle signal names
        lookup_signals = [output_value]
        if output_value == "signal-everything":
            lookup_signals.append("signal-each")

        color = None
        for sig in lookup_signals:
            key = (source, placement.ir_node_id, sig)
            if key in self._edge_wire_colors:
                color = self._edge_wire_colors[key]
                break

        if color is None:
            color = self.get_wire_color_for_entity_pair(source, placement.ir_node_id) or "red"

        placement.properties["output_value_wires"] = {color}
        return 1

    def _inject_condition_wires(self, placement: Any, signal_graph: Any) -> int:
        conditions = placement.properties.get("conditions")
        if not conditions:
            return 0

        count = 0
        for cond in conditions:
            for key in ("first_signal", "second_signal"):
                sig = cond.get(key)
                sid = cond.get(f"{key.replace('signal', 'operand')}_signal_id")
                if not sig or isinstance(sig, int) or not sid:
                    continue
                source = self._resolve_source_entity(sid, signal_graph)
                if not source:
                    continue
                color = self._lookup_color(source, placement.ir_node_id, sig)
                cond[f"{key}_wires"] = {color}
                count += 1
        return count

    def _lookup_color(self, source_id: str, sink_id: str, signal: str) -> str:
        """Look up wire color with wildcard + entity-pair fallback."""
        if signal in WILDCARD_SIGNALS:
            return self.get_wire_color_for_entity_pair(source_id, sink_id) or "red"

        color = self.get_wire_color_for_edge(source_id, sink_id, signal)
        # Fall back to entity-pair if exact lookup returned default
        pair_color = self.get_wire_color_for_entity_pair(source_id, sink_id)
        if pair_color and pair_color != color:
            color = pair_color
        return color

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

    def _get_connection_side(self, entity_id: str, is_source: bool) -> str | None:
        placement = self.layout_plan.get_placement(entity_id)
        if not placement:
            return None
        if is_dual_circuit_connectable(placement.entity_type):
            return "output" if is_source else "input"
        return None

    def _is_memory_feedback_edge(self, source_id: str, sink_id: str, signal_name: str) -> bool:
        from .memory_builder import MemoryModule

        if self._is_internal_feedback_signal(signal_name):
            return True

        for module in self._memory_modules.values():
            if not isinstance(module, MemoryModule) or module.optimization is not None:
                continue
            if not module.write_gate or not module.hold_gate:
                continue
            w = module.write_gate.ir_node_id
            h = module.hold_gate.ir_node_id
            if source_id == w and sink_id == h and signal_name == module.signal_type:
                return True
            if source_id == h and sink_id == w and signal_name == module.signal_type:
                return True
        return False

    def _is_internal_feedback_signal(self, signal_name: str) -> bool:
        if not signal_name.startswith("__feedback_"):
            return False
        from .memory_builder import MemoryModule

        for module in self._memory_modules.values():
            if not isinstance(module, MemoryModule):
                continue
            if (
                hasattr(module, "_feedback_signal_ids")
                and signal_name in module._feedback_signal_ids
            ):
                return True
        # Fallback: any __feedback_ prefixed signal is internal
        return True

    def _register_power_poles_as_relays(self) -> None:
        from .power_planner import POWER_POLE_CONFIG

        for entity_id, placement in self.layout_plan.entity_placements.items():
            if not placement.properties.get("is_power_pole"):
                continue
            if placement.position is None:
                continue
            pole_type = placement.properties.get("pole_type", "medium")
            config = POWER_POLE_CONFIG.get(pole_type.lower())
            if config:
                self.relay_network.add_relay_node(
                    placement.position, entity_id, str(config["prototype"])
                )

    def _add_self_feedback_connections(self) -> None:
        for entity_id, placement in self.layout_plan.entity_placements.items():
            if placement.properties.get("has_self_feedback"):
                fb = placement.properties.get("feedback_signal")
                if fb:
                    self.layout_plan.add_wire_connection(
                        WireConnection(
                            source_entity_id=entity_id,
                            sink_entity_id=entity_id,
                            signal_name=fb,
                            wire_color="red",
                            source_side="output",
                            sink_side="input",
                        )
                    )

    def _validate_relay_coverage(self) -> None:
        span = self.relay_network.span_limit
        violations = 0
        for conn in self.layout_plan.wire_connections:
            src = self.layout_plan.get_placement(conn.source_entity_id)
            snk = self.layout_plan.get_placement(conn.sink_entity_id)
            if not src or not snk or not src.position or not snk.position:
                continue
            if math.dist(src.position, snk.position) > span + 1e-6:
                violations += 1
                if violations <= 5:
                    self.diagnostics.warning(
                        f"Wire exceeds span: {conn.source_entity_id}→{conn.sink_entity_id} "
                        f"({conn.signal_name}) dist={math.dist(src.position, snk.position):.1f}"
                    )
        if violations > 5:
            self.diagnostics.warning(f"Total {violations} wire span violations")

        relays = sum(
            1
            for p in self.layout_plan.entity_placements.values()
            if getattr(p, "role", None) == "wire_relay"
        )
        if relays:
            self.diagnostics.warning(f"Blueprint required {relays} wire relay pole(s)")
