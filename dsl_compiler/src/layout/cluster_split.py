from typing import List, Dict
import logging
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import kmeans_plusplus
from ortools.sat.python import cp_model

from dsl_compiler.src.layout.layout_plan import EntityPlacement


def partition_component_spectral(
    component: List[str],
    adjacency: Dict[str, List[str]],
    num_parts: int,
    max_area: int,
    entities: Dict[str, "EntityPlacement"],
) -> List[List[str]]:
    """Partition using spectral clustering with optimal capacitated assignment.

    Uses:
    - Scipy for spectral embedding (eigenvectors of Laplacian)
    - Sklearn for k-means++ center initialization
    - Google OR-Tools CP-SAT for optimal assignment with capacity constraints

    Args:
        component: Entity IDs to partition
        adjacency: Connectivity map
        num_parts: Target number of partitions (may be increased if needed)

    Returns:
        List of sub-components respecting capacity constraints
    """

    # Calculate how many clusters we actually need based on area
    k = int(num_parts)

    if len(component) <= k:
        return [[eid] for eid in component]

    # Build sparse adjacency matrix
    node_to_idx = {node: i for i, node in enumerate(component)}
    n = len(component)

    rows, cols = [], []
    for i, node in enumerate(component):
        for neighbor in adjacency.get(node, []):
            if neighbor in node_to_idx:
                j = node_to_idx[neighbor]
                rows.append(i)
                cols.append(j)

    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

    # Compute Laplacian: L = D - A
    degrees = np.array(A.sum(axis=1)).flatten()
    D = sparse.diags(degrees)
    L = D - A

    # Get k smallest eigenvectors (skip the first, which is trivial)
    # eigsh returns smallest eigenvalues by default
    eigenvalues, eigenvectors = eigsh(L, k=k, which="SM")

    # Use eigenvectors 2 through k (skip first trivial eigenvector)
    Y = eigenvectors[:, 1:k]  # Shape: (n, k-1)

    # Initialize cluster centers using k-means++

    centers, _ = kmeans_plusplus(Y, n_clusters=k, random_state=0)

    # Compute costs: squared distance from each point to each center
    costs = np.zeros((n, k))
    for j in range(k):
        diff = Y - centers[j]
        costs[:, j] = np.sum(diff * diff, axis=1)

    # Get footprints for each entity (as integers)
    footprints = np.array(
        [
            entities[component[i]].properties.get("footprint", (1, 1))[0]
            * entities[component[i]].properties.get("footprint", (1, 1))[1]
            for i in range(n)
        ],
        dtype=int,
    )

    # Solve capacitated assignment problem using OR-Tools CP-SAT
    assignment = _solve_capacitated_assignment(costs, footprints, max_area, k)

    # Build partitions from assignment
    partitions = [[] for _ in range(k)]
    for i, cluster_idx in enumerate(assignment):
        partitions[cluster_idx].append(component[i])

    # Remove empty partitions
    partitions = [p for p in partitions if p]

    return partitions


def _solve_capacitated_assignment(
    costs: "np.ndarray",
    footprints: "np.ndarray",
    capacity: int,
    k: int,
    time_limit_seconds: int = 30,
) -> "np.ndarray":
    """Solve capacitated assignment using Google OR-Tools CP-SAT.

    Finds optimal assignment of n items to k clusters where:
    - Each item has a footprint (integer area)
    - Each cluster has maximum capacity
    - Assignment minimizes total cost

    Args:
        costs: (n, k) matrix of assignment costs
        footprints: (n,) array of integer footprints
        capacity: Maximum capacity per cluster
        k: Number of clusters
        time_limit_seconds: Solver time limit

    Returns:
        Array of length n with cluster assignments (0 to k-1)
    """

    n = len(footprints)

    # Create CP-SAT model
    model = cp_model.CpModel()

    # Binary variables: x[i,j] = 1 if item i assigned to cluster j
    x = {}
    for i in range(n):
        for j in range(k):
            x[(i, j)] = model.NewBoolVar(f"x_{i}_{j}")

    # Constraint: each item assigned to exactly one cluster
    for i in range(n):
        model.Add(sum(x[(i, j)] for j in range(k)) == 1)

    # Constraint: cluster capacity not exceeded
    for j in range(k):
        model.Add(sum(int(footprints[i]) * x[(i, j)] for i in range(n)) <= capacity)

    # Objective: minimize total cost
    # Scale costs to integers for CP-SAT
    cost_scale = 1000
    costs_int = np.round(costs * cost_scale).astype(int)

    objective = []
    for i in range(n):
        for j in range(k):
            objective.append(costs_int[i, j] * x[(i, j)])

    model.Minimize(sum(objective))

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 4
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Fallback: greedy assignment if no solution found
        logging.warning(
            "CP-SAT could not find solution, falling back to greedy assignment"
        )
        return _greedy_assignment(costs, footprints, capacity, k)

    # Extract assignment
    assignment = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(k):
            if solver.Value(x[(i, j)]) == 1:
                assignment[i] = j
                break

    return assignment


def _greedy_assignment(
    costs: "np.ndarray", footprints: "np.ndarray", capacity: int, k: int
) -> "np.ndarray":
    """Greedy fallback assignment if CP-SAT fails.

    Assigns items to clusters in order of increasing cost.

    Args:
        costs: (n, k) cost matrix
        footprints: (n,) footprint array
        capacity: Maximum capacity per cluster
        k: Number of clusters

    Returns:
        Assignment array
    """

    n = len(footprints)
    assignment = np.zeros(n, dtype=int)
    cluster_used = np.zeros(k, dtype=int)

    # Create priority queue: (cost, item, cluster)
    items = []
    for i in range(n):
        for j in range(k):
            items.append((costs[i, j], i, j))

    items.sort()
    assigned = [False] * n

    # Greedy assignment
    for cost, i, j in items:
        if assigned[i]:
            continue

        if cluster_used[j] + footprints[i] <= capacity:
            assignment[i] = j
            cluster_used[j] += footprints[i]
            assigned[i] = True

    # Force assign any remaining items
    for i in range(n):
        if not assigned[i]:
            # Find cluster with most space
            j = np.argmin(cluster_used)
            assignment[i] = j
            cluster_used[j] += footprints[i]

    return assignment
