"""
"""
import numpy as np
from numpy.typing import NDArray


def factorial(n: int) -> int:
    if n == 1:
        return 1
    return n * factorial(n-1)


def evaluate_removal_sequence(removal_sequence: tuple[int], d_matrix: NDArray[np.float64]) -> float:
    """
    Algorithm to evaluate a removal path from the Weitzman's tree which has a branching-factor sequence (n, n-1, n-2, ..., 2, 1).
    """

    # Initialize value as zero
    w = 0.0

    # Get all elements
    set_of_elements = set([i for i in range(len(removal_sequence))])

    for element in removal_sequence[:-1]:

        # Remove element from the set
        set_of_elements = set_of_elements - {element}

        # Find the closest neighbour to it
        _, neighbour = point_to_set_distance(element, set_of_elements, d_matrix, kind="min")

        # Get the distance between element and its neighbour
        distance = d_matrix[element, neighbour]

        # Add the value to the total sum
        w += distance

    return w


def compute_distance_matrix(lattice: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    lattice: n x m matrix, with n being the number of points and m the number of objectives
    """

    # Compute the n x n distance matrix using the 2-norm
    diff = lattice[:, np.newaxis, :] - lattice[np.newaxis, :, :]
    d_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    return d_matrix


def point_to_set_distance(current_index: int, elements: set[int],
                        d_matrix: NDArray[np.float64],
                        kind: str = "min") -> tuple[float, int]:
    """
    Get the min/max distance from element to a set A
    """
    
    # Allows to change the comparison
    multiplier = 1 if kind != "min" else -1

    # Start variables
    best_distance = float("-inf")
    best_candidate = -1

    # Loop over the set of elements
    for list_index in elements:

        # Sanity check
        if current_index == list_index:
            continue

        # Get the distance between current point and new point using the distance matrix
        current_distance: NDArray[np.float64] = d_matrix[current_index, list_index]

        # Compare against best distance
        if best_distance < current_distance * multiplier:
            best_distance = current_distance.item() * multiplier
            best_candidate = list_index

    # Multiply best_distance by multiplier to fix the sign (in case its minimize)
    best_distance *= multiplier

    return best_distance, best_candidate


def better_than(new: NDArray[np.float64], old: NDArray[np.float64],
                mode: str) -> NDArray[np.bool]:
    """
    Comparison of NDArrays
    
    Returns: NDArray with bool values.
    """
    if mode == "min":
        return new < old
    elif mode == "max":
        return new > old
    
    raise ValueError("Mode must be 'min' or 'max' only.")


def spanning_tree(
        d_matrix: NDArray[np.float64],
        start: int = 0,
        mode: str = "min"
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Prim's algorithm for dense (fully connected) graphs.
    """

    # Copy adjacent matrix
    dist = d_matrix
    
    # Get the number of nodes/vertices
    n = dist.shape[0]

    # --- Initialize arrays ---
    ## Key values used to pick min/max weight edge in cut
    key = np.full(n, np.inf if mode == "min" else -np.inf)

    ## Array to store constructured MST
    parent = np.full(n, -1, dtype=int)
    
    ## Set all vertices as not in MST
    in_mst: NDArray[np.bool] = np.zeros(n).astype(np.bool)

    # Start from vertex start, set node as root (-1)
    key[start] = 0.0
    parent[start] = -1

    # Choose worst value based on min or max
    worst_value = np.inf if mode == "min" else -np.inf
    
    for _ in range(n):
        # Pick vertex not in MST with best (min or max) distance
        ## Mask out vertices already in MST by setting key -> worst_value
        masked_keys = np.where(in_mst, worst_value, key)
        u = np.argmin(masked_keys) if mode == "min" else np.argmax(masked_keys)

        # Put the min/max distance vertex in the MST
        in_mst[u] = True

        # Update distance value of the adjacent vertices not in MST
        # only if the current distance is better than new distance
        not_in_mst: NDArray[np.bool] = ~in_mst
        better: NDArray[np.bool] = better_than(dist[u], key, mode)
        update_mask = not_in_mst & better

        # Update
        key[update_mask] = dist[u, update_mask]
        parent[update_mask] = u

    # Print Parents
    #print(f"Starting vertex: {start}")
    #for i in range(n):
    #    if i == start:
    #        continue
    #    print(parent[i], "-", i, "\t", dist[parent[i], i])
    
    return parent, key



def euler_cycle(adj: dict[int, list[int]], start: int = 0) -> list[int]:
    """
    Compute an Euler cycle using Hierholzer's algorithm.
    """

    # Verify each vertex exists as a key
    local_adj = {u: list(vs) for u, vs in adj.items()}
    if start not in local_adj:
        local_adj[start] = []

    stack = [start]
    cycle = []

    while stack:
        u = stack[-1]
        nbrs = local_adj.get(u, [])

        # While edges remain
        if nbrs:
            v = nbrs.pop() # Remove one
            stack.append(v)
        else:
            cycle.append(stack.pop())

    return cycle[::-1]


def shortcut_cycle(euler_path: list[int]) -> list[int]:
    """
    Remove repeats while preserving order.
    Produces a Hamiltonian cycle.
    """
    visited = set()
    tour = []

    for node in euler_path:
        if node not in visited:
            visited.add(node)
            tour.append(node)

    return tour


