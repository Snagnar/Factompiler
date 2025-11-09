"""Unified clustering and entity packing.

This module handles both cluster analysis (finding connected components)
and entity packing (positioning entities within cluster bounds) in a single pass.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.layout.cluster_split import partition_component_spectral
from .layout_plan import LayoutPlan, EntityPlacement
from .signal_graph import SignalGraph


@dataclass
class PackingResult:
    """Result of clustering and packing operation."""

    clusters: List["ClusterInfo"]
    entity_to_cluster: Dict[str, int]  # entity_id -> cluster_index


@dataclass
class ClusterInfo:
    """Information about a packed cluster."""

    cluster_id: int
    entity_ids: Set[str]
    bounds: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[float, float]


class ClusterPacker:
    """Unified clustering and entity packing.

    Takes unpositioned entities with footprints and connectivity information,
    produces clustered and positioned entities in one pass.

    This replaces the previous multi-step process:
    - Old: Place unconstrained → Analyze clusters → Rearrange
    - New: Analyze connectivity → Pack directly into clusters
    """

    CLUSTER_SIZE = 5  # 5x5 tiles per cluster
    CLUSTER_SPACING = 2  # 2 tiles between clusters
    CLUSTERS_PER_ROW = 5

    def __init__(
        self,
        layout_plan: LayoutPlan,
        signal_graph: SignalGraph,
        diagnostics: ProgramDiagnostics,
    ):
        self.layout_plan = layout_plan
        self.signal_graph = signal_graph
        self.diagnostics = diagnostics

    def pack(self) -> PackingResult:
        """Main entry point: analyze connectivity and pack entities.

        Uses iterative pack-and-split approach:
        1. Initial clustering based on connectivity
        2. Pack entities into clusters
        3. If overflows detected, split overflowing clusters
        4. Repeat until no overflows (max 4 iterations)

        Returns:
            PackingResult with cluster metadata and entity-to-cluster mapping
        """
        # Get all entities that need positioning
        entities = self.layout_plan.entity_placements

        if not entities:
            self.diagnostics.info("No entities to cluster")
            return PackingResult(clusters=[], entity_to_cluster={})

        # 1. Find connected components based on signal connectivity
        components = self._find_connected_components(entities)

        # 2. Initial optimization (split large, merge small)
        current_clusters = self._optimize_cluster_sizes(components)

        # 3. ITERATIVE PACK-AND-SPLIT
        max_iterations = 8  # Increased from 4 to handle larger programs
        for iteration in range(max_iterations):
            self.diagnostics.info(f"Pack iteration {iteration + 1}/{max_iterations}")

            # Calculate grid positions for current clusters
            cluster_bounds = self._calculate_cluster_grid(len(current_clusters))

            # Pack each cluster and detect overflows
            overflowing_cluster_indices = []

            for cluster_idx, (entity_ids, bounds) in enumerate(
                zip(current_clusters, cluster_bounds)
            ):
                has_overflow = self._pack_cluster(cluster_idx, entity_ids, bounds)
                if has_overflow:
                    overflowing_cluster_indices.append(cluster_idx)

            # Check if we have any overflows
            if not overflowing_cluster_indices:
                self.diagnostics.info(
                    f"No overflows detected, packing complete after {iteration + 1} iteration(s)"
                )
                break

            self.diagnostics.warning(
                f"Iteration {iteration + 1}: {len(overflowing_cluster_indices)} cluster(s) overflowed: {overflowing_cluster_indices}"
            )

            # If this is the last iteration, we can't split anymore
            if iteration == max_iterations - 1:
                self.diagnostics.error(
                    f"Max iterations ({max_iterations}) reached with {len(overflowing_cluster_indices)} "
                    f"overflowing clusters. Some entities will overlap!"
                )
                break

            # Split overflowing clusters and rebuild cluster list
            new_clusters = []
            for cluster_idx, entity_ids in enumerate(current_clusters):
                if cluster_idx in overflowing_cluster_indices:
                    # Split this cluster
                    self.diagnostics.info(
                        f"Splitting overflowing cluster {cluster_idx} with {len(entity_ids)} entities"
                    )
                    adjacency = self._build_adjacency_map(entity_ids)
                    sub_clusters = self._split_cluster(entity_ids, adjacency)
                    new_clusters.extend(sub_clusters)
                    self.diagnostics.info(
                        f"  Split into {len(sub_clusters)} sub-clusters"
                    )
                else:
                    # Keep this cluster as-is
                    new_clusters.append(entity_ids)

            current_clusters = new_clusters

            # Clear positions from all entities for re-packing
            for entity_id in entities:
                entities[entity_id].position = None

        # 4. Build final result after successful packing
        cluster_infos = []
        entity_to_cluster = {}
        cluster_bounds = self._calculate_cluster_grid(len(current_clusters))

        for cluster_idx, (entity_ids, bounds) in enumerate(
            zip(current_clusters, cluster_bounds)
        ):
            x1, y1, x2, y2 = bounds
            cluster_infos.append(
                ClusterInfo(
                    cluster_id=cluster_idx,
                    entity_ids=set(entity_ids),
                    bounds=bounds,
                    center=((x1 + x2) / 2, (y1 + y2) / 2),
                )
            )

            # Map entities to cluster
            for entity_id in entity_ids:
                entity_to_cluster[entity_id] = cluster_idx

        self.diagnostics.info(
            f"Final result: {len(entities)} entities in {len(cluster_infos)} clusters"
        )

        return PackingResult(
            clusters=cluster_infos,
            entity_to_cluster=entity_to_cluster,
        )

    def _find_connected_components(
        self, entities: Dict[str, "EntityPlacement"]
    ) -> List[List[str]]:
        """Find connected components using signal graph.

        Args:
            entities: Dictionary of entity_id -> EntityPlacement

        Returns:
            List of components, where each component is a list of entity IDs
        """
        entity_ids = set(entities.keys())

        # Build adjacency from signal graph
        adjacency = {eid: set() for eid in entity_ids}

        for signal_id, source_id, sink_ids in self.signal_graph.iter_edges():
            if source_id in entity_ids:
                for sink_id in sink_ids:
                    if sink_id in entity_ids:
                        adjacency[source_id].add(sink_id)
                        adjacency[sink_id].add(source_id)

        # DFS to find connected components
        visited = set()
        components = []

        for entity_id in entity_ids:
            if entity_id in visited:
                continue

            # Start new component
            component = []
            stack = [entity_id]

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

    def _optimize_cluster_sizes(self, components: List[List[str]]) -> List[List[str]]:
        """Split large components and merge small ones.

        Ported from cluster_analyzer.py to properly handle clusters that
        don't fit in fixed 5x5 cluster bounds.

        Args:
            components: Initial connected components

        Returns:
            Optimized list of entity ID lists that fit in cluster bounds
        """
        MAX_DIMENSION = self.CLUSTER_SIZE  # 5 tiles

        # Build adjacency map for splitting
        adjacency = self._build_adjacency()

        # Split large clusters
        optimized = []

        for component in components:
            split_clusters = self._split_if_needed(component, adjacency)
            optimized.extend(split_clusters)

        # Try merging small clusters
        optimized = self._try_merge_small_clusters(optimized, MAX_DIMENSION)

        return optimized

    def _build_adjacency(self) -> Dict[str, List[str]]:
        """Build adjacency map for all entities.

        Args:
            components: Not used, kept for consistency with cluster_analyzer

        Returns:
            Adjacency map: entity_id -> list of connected entity_ids
        """
        adjacency = {}

        # Initialize adjacency for all entities
        for entity_id in self.layout_plan.entity_placements.keys():
            adjacency[entity_id] = []

        # Build connections from signal graph
        for signal_id, source_id, sink_id in self.signal_graph.iter_source_sink_pairs():
            if source_id and sink_id:
                # Add bidirectional edge
                if source_id not in adjacency:
                    adjacency[source_id] = []
                if sink_id not in adjacency:
                    adjacency[sink_id] = []

                if sink_id not in adjacency[source_id]:
                    adjacency[source_id].append(sink_id)
                if source_id not in adjacency[sink_id]:
                    adjacency[sink_id].append(source_id)

        return adjacency

    def _compute_cluster_area(self, entity_ids: List[str]) -> float:
        """Simulate packing to get ACTUAL space needed, not just sum of footprints.

        This accounts for row-based packing waste.

        Args:
            entity_ids: List of entity IDs

        Returns:
            Actual area needed when packed row-by-row
        """
        if not entity_ids:
            return 0.0

        # Simulate packing with same algorithm as _pack_cluster
        entities = self.layout_plan.entity_placements
        sorted_ids = sorted(
            entity_ids,
            key=lambda eid: (
                entities[eid].properties.get("footprint", (1, 1))[0]
                * entities[eid].properties.get("footprint", (1, 1))[1]
            ),
        )

        # Simulate row-based packing to find required dimensions
        current_x = 0
        current_y = 0
        row_max_height = 0
        max_x_used = 0
        max_y_used = 0

        for entity_id in sorted_ids:
            entity = entities[entity_id]
            footprint = entity.properties.get("footprint", (1, 1))
            width, height = footprint

            # Check if entity fits in current row (assume max width of CLUSTER_SIZE)
            if current_x + width > self.CLUSTER_SIZE:
                # Move to next row
                if row_max_height > 0:
                    current_y += row_max_height
                current_x = 0
                row_max_height = 0

            # Track maximum extents
            max_x_used = max(max_x_used, current_x + width)
            max_y_used = max(max_y_used, current_y + height)

            # Advance
            current_x += width
            row_max_height = max(row_max_height, height)

        # Return actual bounding box area
        return float(max_x_used * max_y_used)

    def _split_if_needed(
        self,
        component: List[str],
        adjacency: Dict[str, List[str]],
    ) -> List[List[str]]:
        """Split component if it won't fit in cluster bounds.

        Args:
            component: Entity IDs in component
            adjacency: Connectivity map
            base_id: Base cluster ID for naming
            max_dimension: Maximum dimension (5 tiles)

        Returns:
            List of sub-clusters that fit in bounds
        """
        # Determine how many sub-clusters needed
        area = self._compute_cluster_area(component)
        max_area = self.CLUSTER_SIZE * self.CLUSTER_SIZE
        if area <= max_area:
            return [component]

        # Use ceiling division and add 1 extra cluster for safety margin
        # This ensures we have enough capacity even with packing inefficiencies
        num_parts = int(math.ceil(area / max_area)) + 1

        # Partition using spectral clustering
        sub_clusters = partition_component_spectral(
            component,
            adjacency,
            num_parts,
            max_area,
            self.layout_plan.entity_placements,
        )
        return sub_clusters

    def _try_merge_small_clusters(
        self, clusters: List[List[str]], max_dimension: int
    ) -> List[List[str]]:
        """Merge small clusters that can be combined.

        Args:
            clusters: List of clusters
            max_dimension: Maximum dimension

        Returns:
            Merged cluster list
        """
        if len(clusters) <= 1:
            return clusters

        # Sort by size for better merging
        sorted_clusters = sorted(clusters, key=len)
        merged = []
        skip_indices = set()

        for i, cluster1 in enumerate(sorted_clusters):
            if i in skip_indices:
                continue

            # Try to merge with subsequent clusters
            merged_cluster = cluster1.copy()

            for j in range(i + 1, len(sorted_clusters)):
                if j in skip_indices:
                    continue

                cluster2 = sorted_clusters[j]
                combined_area = self._compute_cluster_area(merged_cluster + cluster2)

                if combined_area <= max_dimension * max_dimension:
                    # Can merge
                    merged_cluster.extend(cluster2)
                    skip_indices.add(j)

            merged.append(merged_cluster)

        return merged

    def _calculate_cluster_grid(
        self, num_clusters: int
    ) -> List[Tuple[int, int, int, int]]:
        """Calculate grid positions for clusters with fixed 5x5 sizing.

        All clusters are CLUSTER_SIZE x CLUSTER_SIZE (5x5) to reserve space
        for relay poles in the spacing areas between clusters.

        Args:
            num_clusters: Number of clusters to position

        Returns:
            List of (x1, y1, x2, y2) bounds for each cluster
        """
        bounds_list = []
        cluster_stride = self.CLUSTER_SIZE + self.CLUSTER_SPACING  # 5 + 2 = 7

        for idx in range(num_clusters):
            row = idx // self.CLUSTERS_PER_ROW
            col = idx % self.CLUSTERS_PER_ROW

            x1 = col * cluster_stride
            y1 = row * cluster_stride
            x2 = x1 + self.CLUSTER_SIZE
            y2 = y1 + self.CLUSTER_SIZE

            bounds_list.append((x1, y1, x2, y2))

        return bounds_list

    def _build_adjacency_map(self, entity_ids: List[str]) -> Dict[str, List[str]]:
        """Build adjacency map for given entity IDs.

        Args:
            entity_ids: List of entity IDs to build adjacency for

        Returns:
            Dict mapping entity_id -> list of connected entity_ids
        """
        adjacency = {eid: [] for eid in entity_ids}
        entity_set = set(entity_ids)

        # Build connections from signal graph - iterate over all source-sink pairs
        for signal_id, source_id, sink_id in self.signal_graph.iter_source_sink_pairs():
            # Only include connections where both entities are in our set
            if (
                source_id in entity_set
                and sink_id in entity_set
                and source_id != sink_id
            ):
                # Add bidirectional edge
                if sink_id not in adjacency[source_id]:
                    adjacency[source_id].append(sink_id)
                if source_id not in adjacency[sink_id]:
                    adjacency[sink_id].append(source_id)

        return adjacency

    def _split_cluster(
        self, entity_ids: List[str], adjacency: Dict[str, List[str]]
    ) -> List[List[str]]:
        """Split a cluster into smaller sub-clusters.

        Uses recursive splitting to ensure all sub-clusters fit in bounds.

        Args:
            entity_ids: Entity IDs in the cluster to split
            adjacency: Adjacency map for connectivity

        Returns:
            List of sub-clusters (each is a list of entity IDs)
        """
        return self._split_if_needed(entity_ids, adjacency)

    def _pack_cluster(
        self,
        cluster_idx: int,
        entity_ids: List[str],
        bounds: Tuple[int, int, int, int],
    ) -> bool:
        """Pack entities into cluster using simple greedy row-based algorithm.

        Algorithm (as suggested by user):
        1. Start at top-left (x1, y1)
        2. Sort entities by footprint (smallest first for better packing)
        3. Place left-to-right until hitting right bound
        4. Track max height in current row
        5. Move to next row when needed
        6. Repeat until all entities placed

        Args:
            cluster_idx: Index of this cluster
            entity_ids: List of entity IDs to pack
            bounds: (x1, y1, x2, y2) bounds for this cluster

        Returns:
            True if overflow occurred, False otherwise
        """
        if not entity_ids:
            return False

        x1, y1, x2, y2 = bounds
        entities = self.layout_plan.entity_placements

        # Sort by footprint area (smallest first for better packing)
        sorted_ids = sorted(
            entity_ids,
            key=lambda eid: (
                entities[eid].properties.get("footprint", (1, 1))[0]
                * entities[eid].properties.get("footprint", (1, 1))[1]
            ),
        )

        # Pack entities row by row
        current_x = x1
        current_y = y1
        row_max_height = 0
        has_overflow = False

        for entity_id in sorted_ids:
            entity = entities[entity_id]
            footprint = entity.properties.get("footprint", (1, 1))
            width, height = footprint

            # Check if entity fits in current row
            if current_x + width > x2:
                # Move to next row
                if row_max_height > 0:
                    current_y += row_max_height  # No spacing - entities touch
                current_x = x1
                row_max_height = 0

            # Check if we're overflowing vertically
            if current_y + height > y2:
                has_overflow = True
                print(
                    f"[OVERFLOW] Cluster {cluster_idx}: entity '{entity_id}' at ({current_x}, {current_y}) with height {height} exceeds y2={y2}"
                )
                self.diagnostics.warning(
                    f"Cluster {cluster_idx} overflow: entity '{entity_id}' "
                    f"at tile ({current_x}, {current_y}) exceeds bounds {bounds}. "
                    f"Entity height {height} would place it at y={current_y + height}, but y2={y2}"
                )

            # Assign position to entity (convert tile position to center position)
            # For draftsman, positions are CENTER positions
            center_x = current_x + width / 2.0
            center_y = current_y + height / 2.0
            entity.position = (center_x, center_y)

            # Advance horizontally - entities touch directly, no spacing
            current_x += width
            row_max_height = max(row_max_height, height)

        if not has_overflow:
            self.diagnostics.info(
                f"Packed {len(entity_ids)} entities in cluster {cluster_idx} "
                f"within bounds {bounds}"
            )

        return has_overflow
