import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Combinatorics
# ---------------------------------------------------------------------------

def factorial(n: int) -> int:
    if n == 1:
        return 1
    return n * factorial(n - 1)


# ---------------------------------------------------------------------------
# Core Weitzman scoring
# ---------------------------------------------------------------------------

def evaluate_removal_sequence(
    removal_sequence: tuple[int, ...] | list[int],
    d_matrix: NDArray[np.float64],
) -> float:
    """
    Score a permutation under Weitzman's recursive definition:
        W(A) = max_i { W(A \\ {i}) + D(i, A \\ {i}) }

    This function evaluates one specific removal order rather than
    finding the maximum over all orders. The total score is the sum
    of point-to-set distances at each removal step.

    Parameters
    ----------
    removal_sequence : ordered permutation of {0, ..., n-1}
        Index 0 is removed first; the last index contributes 0
        (it is the only remaining element, so its set is empty).
    d_matrix : (n, n) symmetric distance matrix.

    Returns
    -------
    float : Weitzman diversity value for this specific removal order.
    """
    w = 0.0

    # 'remaining' starts as the full set and shrinks as elements are removed.
    remaining = set(range(len(removal_sequence)))

    for element in removal_sequence[:-1]:   # last element contributes 0 - skip it
        remaining.remove(element)           # element leaves the set before we measure its distance

        # D(i, A\{i}): distance from the removed element to the nearest
        # point still in the set (Weitzman uses the minimum distance).
        _, neighbour = point_to_set_distance(element, remaining, d_matrix, kind="min")

        # accumulate the contribution of this removal step
        w += d_matrix[element, neighbour]   
    return w


# ---------------------------------------------------------------------------
# Distance utilities
# ---------------------------------------------------------------------------

def compute_distance_matrix(lattice: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the (n, n) pairwise Euclidean distance matrix for an (n, d) array.

    Uses broadcasting to avoid an explicit double loop.
    """
    # diff[i, j, :] = lattice[i] - lattice[j]  - shape (n, n, d)
    diff = lattice[:, np.newaxis, :] - lattice[np.newaxis, :, :]
    # sum of squared components along the last axis, then square root
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def point_to_set_distance(
    current: int,
    elements: set[int],
    d_matrix: NDArray[np.float64],
    kind: str = "min",
) -> tuple[float, int]:
    """
    Distance from point `current` to the nearest (kind='min') or
    farthest (kind='max') point in `elements`.

    Returns (distance, index_of_best_candidate).
    """
    # A multiplier of -1 inverts all distances so argmax always finds the min.
    # This avoids branching inside the hot loop.
    multiplier = -1 if kind == "min" else 1

    best_dist = float("-inf")   # worst possible signed value - any real distance beats it
    best_idx = -1

    for idx in elements:
        if idx == current:      # a point is never its own neighbour
            continue
        d = d_matrix[current, idx] * multiplier
        if d > best_dist:
            best_dist = d
            best_idx = idx

    # Undo the sign flip before returning so the caller always gets a positive distance.
    return best_dist * multiplier, best_idx


# ---------------------------------------------------------------------------
# Euler circuit / Hamiltonian path helpers
# (shared by the MST-based heuristics: TwiceAround and Christofides)
# ---------------------------------------------------------------------------

AdjList = dict[int, list[list]]  # {node: [[neighbour, weight], ...]}

WeightedAdjList = dict[int, list[list]]
EdgeIDAdjList   = dict[int, list[tuple[int, float, int, int]]]

def multigraph_to_edgeid_adjlist(multigraph) -> tuple[EdgeIDAdjList, int]:
    """
    Convert a NetworkX MultiGraph to a weighted adjacency list with edge IDs.

    Each undirected multigraph edge receives a unique edge_id.
    Each edge is stored twice:
        u -> v
        v -> u

    local_order preserves the insertion order of each directed copy inside
    that endpoint's adjacency list.

    Returns
    -------
    adj: {node: [(neighbor, weight, edge_id, local_order), ...]}

    num_edges: Number of undirected multigraph edges.
    """
    adj: EdgeIDAdjList = {u: [] for u in multigraph.nodes()}
    edge_id = 0

    for u, v, data in multigraph.edges(data=True):
        w = float(data["weight"])

        order_u = len(adj[u])
        order_v = len(adj[v])

        adj[u].append((v, w, edge_id, order_u))
        adj[v].append((u, w, edge_id, order_v))

        edge_id += 1
    return adj, edge_id


def sort_edgeid_adjlist(
    adj: EdgeIDAdjList,
    mode: str = "max"
) -> EdgeIDAdjList:
    """
    Return a sorted copy of an edge-ID adjacency list.

    mode = 'max': choose largest weight,
        ties resolved by first original adjacency occurence.

    mode = 'min': choose smallest weight,
        ties resolved by first original adjacency occurence.
    """
    if mode not in {"min", "max"}:
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

    sorted_adj: EdgeIDAdjList = {}

    for v, edges in adj.items():
        if mode == "max":
            sorted_edges = sorted(edges, key=lambda item: (-item[1], item[3]))
        else:
            sorted_edges = sorted(edges, key=lambda item: (item[1], item[3]))

        sorted_adj[v] = sorted_edges
    return sorted_adj


def greedy_euler_circuit_sorted_ptr(
    sorted_adj: EdgeIDAdjList,
    start: int,
    num_edges: int,
) -> list[int]:
    """
    Greedy Hierholzer traversal using sorted adjacency lists.

    Parameters
    ----------
    sorted_adj: Edge-ID adjacency list already sorted according to the desired mode.

    start: Starting vertex.

    num_edges: Number of undirected multigraph edges. This is the number of
        unique edge IDs.

    Returns
    -------
    circuit: Euler circuit as a vertex sequence.

    Complexity
    ----------
    O(n)
    """
    if start not in sorted_adj:
        raise ValueError(f"start vertex {start!r} is not in the adjacency list")

    used = [False] * num_edges
    ptr = {v: 0 for v in sorted_adj}

    stack = [start]
    euler_circuit: list[int] = []

    while stack:
        v = stack[-1]
        edges = sorted_adj[v]
        p = ptr[v]

        # Skip stale edges already consumed
        while p < len(edges) and used[edges[p][2]]:
            p += 1

        ptr[v] = p
        if p < len(edges):
            u, _w, edge_id, _local_order = edges[p]

            used[edge_id] = True
            ptr[v] += 1
            stack.append(u)
        else:
            euler_circuit.append(stack.pop())

    return euler_circuit[::-1]


def prepare_sorted_greedy_euler_adjacency(
    multigraph,
    mode: str = "max",
) -> tuple[EdgeIDAdjList, int]:
    """
    Prepare the sorted adjacency structure used by the greedy Hierholzer
    Euler traversal.
    """
    edgeid_adj, num_edges = multigraph_to_edgeid_adjlist(multigraph)
    sorted_adj = sort_edgeid_adjlist(edgeid_adj, mode=mode)
    return sorted_adj, num_edges


def run_greedy_euler_circuit(
    sorted_adj: EdgeIDAdjList,
    start: int,
    num_edges: int,
) -> list[int]:
    """
    Thin wrapper around the pointer-based implementation.
    """
    return greedy_euler_circuit_sorted_ptr(
        sorted_adj=sorted_adj,
        start=start,
        num_edges=num_edges,
    )


def shortcut_to_hamiltonian(euler: list[int]) -> list[int]:
    """
    Convert an Euler circuit to a Hamiltonian path by skipping
    any vertex that has already been visited (triangle-inequality shortcut).

    This is valid in metric spaces: the direct edge (u, w) is never
    longer than the detour (u, v, w) when v was a repeated visit.
    """
    visited: set[int] = set()
    path: list[int] = []
    for v in euler:
        if v not in visited:
            visited.add(v)
            path.append(v)   # keep only the first occurrence of each vertex
    return path


# ---------------------------------------------------------------------------
# Unweighted Euler (kept for completeness)
# ---------------------------------------------------------------------------

def _euler_cycle(adj: dict[int, list[int]], start: int = 0) -> list[int]:
    """Unweighted Hierholzer's algorithm (no greedy edge choice)."""
    local_adj = {u: list(vs) for u, vs in adj.items()}
    if start not in local_adj:
        local_adj[start] = []

    stack = [start]
    cycle: list[int] = []

    while stack:
        u = stack[-1]
        if local_adj.get(u):
            v = local_adj[u].pop()
            stack.append(v)
        else:
            cycle.append(stack.pop())

    return cycle[::-1]

# ---------------------------------------------------------------------------
# Deprecated functions. A better implementation was found or not in use.
# ---------------------------------------------------------------------------

def _multigraph_to_adjlist(multigraph) -> AdjList:
    """
    Convert a NetworkX MultiGraph to a plain weighted adjacency list.

    Each undirected edge (u, v, w) is stored in both adj[u] and adj[v]
    so that the Euler circuit traversal can consume edges from either end.
    """
    adj: AdjList = {u: [] for u in multigraph.nodes()}
    for u, v, data in multigraph.edges(data=True):
        w = float(data["weight"])
        adj[u].append([v, w])   # forward direction
        adj[v].append([u, w])   # reverse direction (undirected graph)
    return adj


def _greedy_euler_circuit(
    adj: AdjList, start: int, mode: str = "max"
) -> list[int]:
    """
    DEPRECATED: Naive implementation with worst case O(n^2)
    on star MSTs.

    Hierholzer's algorithm on a weighted Eulerian multigraph.

    At every step the next edge is chosen greedily by weight
    (largest for mode='max', smallest for mode='min').  This
    greedy bias influences which Hamiltonian path is produced
    after the shortcut step, connecting the edge selection strategy
    to the quality of the resulting Weitzman sequence.

    The input multigraph must be Eulerian (all degrees even).
    """
    if mode not in {"min", "max"}:
        raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

    # Work on a local copy to avoid mutating the adjacency lists while traversing.
    local_adj = {u: neighbors[:] for u, neighbors in adj.items()}

    stack = [start]
    circuit: list[int] = []

    while stack:
        v = stack[-1]
        nbrs = local_adj.get(v, [])

        if nbrs:
            # Greedy edge selection: pick the heaviest (or lightest) available edge.
            if mode == "max":
                idx = max(range(len(nbrs)), key=lambda i: nbrs[i][1])
            else:
                idx = min(range(len(nbrs)), key=lambda i: nbrs[i][1])

            u, _ = nbrs.pop(idx)    # remove the chosen edge from v's list

            # Remove the reverse copy from u's list to maintain consistency
            # (each undirected edge is stored twice, one per endpoint).
            for j, item in enumerate(local_adj[u]):
                if item[0] == v:
                    local_adj[u].pop(j)
                    break
            else:
                raise RuntimeError(f"Missing reverse edge {u} -> {v}")

            stack.append(u)
        else:
            # v has no more edges: it belongs to the circuit at this position.
            circuit.append(stack.pop())

    return circuit[::-1]   # reverse to get the circuit in traversal order


# ---------------------------------------------------------------------------
# Spanning tree (Prim's algorithm) NOT IN USE
# ---------------------------------------------------------------------------

def _better_than(
    new: NDArray[np.float64], old: NDArray[np.float64], mode: str
) -> NDArray[np.bool_]:
    """Element-wise comparison used during the MST key update."""
    if mode == "min":
        return new < old
    if mode == "max":
        return new > old
    raise ValueError("mode must be 'min' or 'max'.")


def _spanning_tree(
    d_matrix: NDArray[np.float64],
    start: int = 0,
    mode: str = "min",
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Prim's algorithm for a dense complete graph.

    mode='min' -> minimum spanning tree (MST).
    mode='max' -> maximum spanning tree (MaxST).

    Returns
    -------
    parent : (n,) array where parent[v] is the MST predecessor of v
             (-1 for the root).
    key    : (n,) array of the edge weights used to attach each vertex.

    NOTE: CURRENTLY NOT IN USE
    """
    n = d_matrix.shape[0]

    # key[v] holds the best (min or max) edge weight seen so far that
    # connects v to the current tree. Initialised to the worst possible
    # value so every real edge will improve it on the first update.
    key = np.full(n, np.inf if mode == "min" else -np.inf)

    parent = np.full(n, -1, dtype=int)    # -1 marks the root / not yet assigned
    in_mst = np.zeros(n, dtype=bool)      # tracks which vertices are already in the tree

    key[start] = 0.0          # the root costs nothing to add
    worst = np.inf if mode == "min" else -np.inf

    for _ in range(n):
        # Among all vertices not yet in the MST, pick the one with the best key.
        # Masking in-tree vertices with `worst` prevents them from being selected.
        masked = np.where(in_mst, worst, key)
        u = np.argmin(masked) if mode == "min" else np.argmax(masked)

        in_mst[u] = True   # commit vertex u to the tree

        # For every out-of-tree vertex v, check whether the edge (u, v) is
        # better than the best edge seen so far connecting v to the tree.
        not_in = ~in_mst
        better = _better_than(d_matrix[u], key, mode)    # vectorised comparison
        mask = not_in & better                          # only update out-of-tree vertices that improve
        key[mask] = d_matrix[u, mask]                   # record the new best edge weight
        parent[mask] = u                                # u is now the tentative parent of those vertices

    return parent, key
