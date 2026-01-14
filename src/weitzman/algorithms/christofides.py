"""
"""

import os
import numpy as np
import networkx as nx

from pathlib import Path
from numpy.typing import NDArray
from collections import defaultdict

from weitzman.io.loaders import load_lattice
from weitzman.io.writers import write_values, make_save_folder
from weitzman.utils.operations import evaluate_removal_sequence
from weitzman.utils.operations import shortcut_cycle, euler_cycle
from weitzman.utils.operations import compute_distance_matrix, spanning_tree


def min_weight_perfect_matching_odd_vertices(
        odd_vertices: list[int],
        d_matrix: NDArray[np.float64],
) -> list[tuple[int, int]]:
    """
    """

    if len(odd_vertices) == 0:
        return []
    if len(odd_vertices) % 2 != 0:
        raise ValueError(f"Odd vertex count must be even.")

    G = nx.Graph()
    G.add_nodes_from(odd_vertices)

    # Complete graph on odd vertices with metric weights
    # O(k^2), k = |odd_vertices|
    for i in range(len(odd_vertices)):
        u = odd_vertices[i]
        for j in range(i + 1, len(odd_vertices)):
            v = odd_vertices[j]
            G.add_edge(u, v, weight=float(d_matrix[u, v]))

    # min_weight_matching
    matching = nx.algorithms.matching.min_weight_matching(G, weight="weight")
    matching_edges = [(u, v) for (u, v) in matching]

    # Validate it
    covered = set()
    for u, v in matching_edges:
        covered.add(u)
        covered.add(v)

    if covered != set(odd_vertices):
        missing = sorted(set(odd_vertices) - covered)
        raise ValueError("Matching is not perfect.")

    return matching_edges


def make_eulerian_multigraph_adj(
    n: int,
    base_edges: Iterable[tuple[int, int]],
    extra_edges: Iterable[tuple[int, int]] = (),
) -> dict[int, list[int]]:
    """
    """
    all_edges = list(base_edges) + list(extra_edges)

    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    deg = [0] * n

    for u, v in all_edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"Edge ({u}, {v}) has vertex outside [0, {n-1}]")
        if u == v:
            raise ValueError(f"Self-loop ({u}, {v}) is unexpected for TSP metric graphs.")

        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1

    odd = [i for i, d in enumerate(deg) if d % 2 == 1]
    if odd:
        raise ValueError(
            f"Graph is not Eulerian: odd-degree vertices exist: {odd}. "
            "This typically means the matching was not perfect."
            )

    return adj


def christofides(lattice_path: Path, labeled: str, start_mst: int, start_euler: int,
                 mst_mode: str = "min", reverse: bool = False) -> None:

    # Load lattice
    lattice, _ = load_lattice(str(lattice_path), labeled=labeled)

    # Compute distance matrix
    d_matrix = compute_distance_matrix(lattice)
    #print(d_matrix)

    # Get n
    n = d_matrix.shape[0]

    # Build an MST from the set of vertices
    tree, _ = spanning_tree(d_matrix, start=start_mst, mode=mst_mode)
    tree = tree.tolist()
    #print(tree)

    if sum(1 for x in tree if x != -1) != n - 1:
        raise ValueError("MST parent array does not have n-1 edges.")

    # MST edges
    mst_edges = [(tree[i], i) for i in range(n) if tree[i] != -1]

    # Degrees in MST
    deg = [0] * n
    for u, v in mst_edges:
        deg[u] += 1
        deg[v] += 1

    # Odd-degree vertices
    odd_vertices = [i for i in range(n) if deg[i] % 2 == 1]

    # Minimum-weight matching on odd vertices
    matching_edges = min_weight_perfect_matching_odd_vertices(odd_vertices, d_matrix)

    # Eulerian multigraph adjacency
    adj = make_eulerian_multigraph_adj(n, mst_edges, matching_edges)

    ## Make an Euler cycle
    euler_path = euler_cycle(adj, start=start_euler)
    #print(euler_path)

    # Euler tour must use every arc exactly once -> length m + 1
    m = len(mst_edges) + len(matching_edges)
    if len(euler_path) != 2 * m + 1:
        # Explanation: adjacency stores each undirected edge twice (u->v and v->u),
        # so the Euler walk consumes 2m directed arcs => vertex sequence length 2m+1.
        raise ValueError(
            f"Euler tour length mismatch: len(euler_path)={len(euler_path)} expected={2*m+1}. "
            "Graph likely malformed or euler_cycle did not traverse all arcs."
        )

    # Traverse the cycle, take shortcuts when repeating a vertex
    removal_sequence = shortcut_cycle(euler_path)

    if len(set(removal_sequence)) != n or len(removal_sequence) != n:
        raise ValueError(f"Shortcut did not produce a Hamiltonian cycle: got {len(removal_sequence)} unique={len(set(removal_sequence))}, expected {n}.")

    if any((u < 0 or u >= n) for u in removal_sequence):
        raise ValueError("Tour contains an out-of-range vertex id")

    if reverse:
        removal_sequence = removal_sequence[::-1]
        #removal_sequence = rotate_to_start(removal_sequence, start_euler)

    # Evaluate the removal sequence
    w = evaluate_removal_sequence(removal_sequence, d_matrix)

    return removal_sequence, w


def run(lattice_params: dict[str, str | Path | bool], outdir: Path, **kwargs) -> None:

    # Extract parameters
    lattice_dir = lattice_params["lattice_dir"]
    lattice_ext = lattice_params["lattice_ext"]
    lattice_labeled = lattice_params["lattice_labeled"]

    # Load lattices
    lattices = sorted(os.listdir(str(lattice_dir)))
    lattices = [lattice for lattice in lattices if lattice.endswith(lattice_ext)]

    ## Get algorithm parameters
    mst_mode = kwargs.get("mst_mode", "min")
    reverse_seq = kwargs.get("reverse", True)

    print(f"Christofides: mst_mode: {mst_mode}, reverse: {reverse_seq}.\n")

    # Solve all instances
    for lattice in lattices:

        # Get n
        n = int(lattice.split("_")[2])

        # Make lattice path
        lattice_path = lattice_dir / lattice

        # To store values
        all_values: list[float] = []
        all_sequences: list[tuple[int]] = []

        # For all starting points
        for start in range(0, n):

            # Execute christofides heuristic
            removal_sequence, value = christofides(lattice_path=lattice_path, labeled=lattice_labeled,
                                                    start_mst=start, start_euler=start,
                                                    mst_mode=mst_mode, reverse=reverse_seq)

            all_values.append(value)
            all_sequences.append(removal_sequence)

        # To numpy
        vector = np.array(all_values)
        sequence_matrix = np.array(all_sequences, np.int16)

        # Get original filename - no extension
        original_name = lattice.rsplit(".", 1)[0]

        # Write values
        file_name = f"values_{original_name}.npy"
        file_path = str(outdir / "values" / file_name)
        write_values(vector, file_path)

        # Write sequences
        file_name = f"sequences_{original_name}.npy"
        file_path = str(outdir / "sequences" / file_name)
        write_values(sequence_matrix, file_path)

