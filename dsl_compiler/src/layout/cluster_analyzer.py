"""Simple connectivity-based clustering for wire distance guarantees."""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import math

from dsl_compiler.src.ir.nodes import (
    IRNode,
    IR_Const,
    IR_Arith,
    IR_Decider,
    IR_MemCreate,
    IR_MemRead,
    IR_MemWrite,
    IR_PlaceEntity,
)
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics


@dataclass
class Cluster:
    """Bounded region containing tightly-connected entities."""

    cluster_id: str
    entity_ids: Set[str] = field(default_factory=set)
    center: Tuple[float, float] = (0.0, 0.0)  # Will be computed
    bounds: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)

    # Layout constraints
    MAX_DIMENSION = 5  # 5×5 tiles = 7.07 diagonal < 9 wire limit


class ClusterAnalyzer:
    """Partitions IR operations into bounded clusters by signal connectivity."""

    def __init__(self, diagnostics: ProgramDiagnostics):
        self.diagnostics = diagnostics
        self.clusters: List[Cluster] = []
        self.entity_to_cluster: Dict[str, int] = {}  # entity_id -> cluster_index

    def analyze(self, ir_operations: List[IRNode], signal_graph) -> List[Cluster]:
        """Main entry: partition IR into connected clusters."""

        # Step 1: Build entity list and adjacency
        entities, adjacency = self._build_adjacency(ir_operations, signal_graph)

        # Step 2: Find connected components
        components = self._find_connected_components(entities, adjacency)

        # Step 3: Split oversized components
        final_clusters = []
        for i, component in enumerate(components):
            sub_clusters = self._split_if_needed(component, adjacency, i)
            final_clusters.extend(sub_clusters)

        self.clusters = final_clusters
        self._build_entity_to_cluster_map()

        self.diagnostics.info(
            f"Clustered {len(entities)} entities into {len(final_clusters)} bounded regions"
        )

        return self.clusters

    def _build_adjacency(self, ir_operations: List[IRNode], signal_graph):
        """Build adjacency list from signal graph."""

        # Collect all entities that need placement
        entities = set()
        for op in ir_operations:
            if isinstance(
                op,
                (IR_Const, IR_Arith, IR_Decider, IR_MemCreate, IR_MemRead, IR_MemWrite),
            ):
                entities.add(op.node_id)
            elif isinstance(op, IR_PlaceEntity):
                entities.add(op.entity_id)

        # Build adjacency from signal connections
        adjacency = {eid: set() for eid in entities}

        for signal_id, source_id, sink_ids in signal_graph.iter_edges():
            if source_id in entities:
                for sink_id in sink_ids:
                    if sink_id in entities:
                        adjacency[source_id].add(sink_id)
                        adjacency[sink_id].add(source_id)

        return entities, adjacency

    def _find_connected_components(
        self, entities: Set[str], adjacency: Dict[str, Set[str]]
    ) -> List[List[str]]:
        """Standard DFS to find connected components."""

        visited = set()
        components = []

        for entity in entities:
            if entity in visited:
                continue

            component = []
            stack = [entity]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue

                visited.add(current)
                component.append(current)

                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited:
                        stack.append(neighbor)

            components.append(component)

        return components

    def _split_if_needed(
        self, component: List[str], adjacency: Dict[str, Set[str]], base_id: int
    ) -> List[Cluster]:
        """Split component if it would exceed size limit."""

        # Estimate: 1.5 tiles per entity (accounts for spacing)
        estimated_area = len(component) * 1.5
        estimated_dimension = math.sqrt(estimated_area)

        if estimated_dimension <= Cluster.MAX_DIMENSION:
            # Small enough - create single cluster
            cluster = Cluster(
                cluster_id=f"cluster_{base_id}", entity_ids=set(component)
            )
            return [cluster]

        # Too large - split by min-cut
        self.diagnostics.info(
            f"Splitting large component ({len(component)} entities, ~{estimated_dimension:.1f} tiles)"
        )

        return self._recursive_split(component, adjacency, base_id, depth=0)

    def _recursive_split(
        self,
        component: List[str],
        adjacency: Dict[str, Set[str]],
        base_id: int,
        depth: int,
    ) -> List[Cluster]:
        """Recursively split component until all parts fit."""

        if depth > 10:  # Safety limit
            self.diagnostics.warning(
                "Max split depth reached - creating cluster anyway"
            )
            return [Cluster(f"cluster_{base_id}_deep", set(component))]

        # Simple split: partition by graph cut
        sub1, sub2 = self._bisect_component(component, adjacency)

        # Check if both parts are small enough now
        result = []

        for i, sub in enumerate([sub1, sub2]):
            estimated_area = len(sub) * 1.5
            estimated_dim = math.sqrt(estimated_area)

            if estimated_dim <= Cluster.MAX_DIMENSION:
                result.append(Cluster(f"cluster_{base_id}_{depth}_{i}", set(sub)))
            else:
                # Still too large - recurse
                result.extend(self._recursive_split(sub, adjacency, base_id, depth + 1))

        return result

    def _bisect_component(
        self, component: List[str], adjacency: Dict[str, Set[str]]
    ) -> Tuple[List[str], List[str]]:
        """Split component into two parts with minimal connections between them."""

        if len(component) <= 1:
            return component, []

        # Simple heuristic: BFS from arbitrary start, take first half vs second half
        # This tends to produce spatially-coherent groups

        start = component[0]
        visited = []
        queue = [start]
        seen = {start}

        while queue and len(visited) < len(component):
            current = queue.pop(0)
            visited.append(current)

            for neighbor in adjacency.get(current, []):
                if neighbor in component and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)

        # Split at midpoint of BFS order
        split_point = len(visited) // 2
        return visited[:split_point], visited[split_point:]

    def _build_entity_to_cluster_map(self):
        """Build reverse mapping for lookup."""
        for cluster_idx, cluster in enumerate(self.clusters):
            for entity_id in cluster.entity_ids:
                self.entity_to_cluster[entity_id] = cluster_idx

    def get_cluster_for_entity(self, entity_id: str) -> int:
        """Get cluster index for an entity."""
        return self.entity_to_cluster.get(entity_id, 0)
