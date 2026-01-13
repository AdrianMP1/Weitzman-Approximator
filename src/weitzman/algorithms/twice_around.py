
import os
import numpy as np

from pathlib import Path
from numpy.typing import NDArray
from collections import defaultdict

from weitzman.io.loaders import load_lattice
from weitzman.io.writers import write_values, make_save_folder
from weitzman.utils.operations import evaluate_removal_sequence
from weitzman.utils.operations import shortcut_cycle, euler_cycle
from weitzman.utils.operations import compute_distance_matrix, spanning_tree


def rotate_to_start(tour: list[int], start: int) -> list[int]:
    if start not in tour:
        return tour
    k = tour.index(start)
    return tour[k:] + tour[:k]


def build_directed_adj_from_parent(parent: list[int], n: int) -> tuple[dict[int, list[int]], int]:
    edges = [(parent[i], i) for i in range(n) if parent[i] != -1]
    arcs = edges + [(v,u) for (u,v) in edges] # both directions

    adj = {i: [] for i in range(n)}
    for u, v in arcs:
        adj[u].append(v)

    return adj, len(arcs)


def twice_around(lattice_path: Path, labeled: str, n: int, start_mst: int, start_euler: int,
                 mst_mode: str = "min", reverse: bool = True) -> tuple[tuple[int], float]:

    # Load lattice
    lattice, _ = load_lattice(str(lattice_path), labeled=labeled)

    # Compute distance matrix
    d_matrix = compute_distance_matrix(lattice)
    #print(d_matrix)

    # Build an MST from the set of vertices
    tree, _ = spanning_tree(d_matrix, start=start_mst, mode=mst_mode)
    tree = tree.tolist()
    #print(tree)

    if sum(1 for x in tree if x != -1) != n - 1:
        raise ValueError("MST parent array does not have n-1 edges.")

    # Duplicate all edges to construct an Euler cycle
    edges = [(tree[i], i) for i in range(0, len(tree)) if tree[i] != -1]
    edges_duplicated = edges + [(v, u) for u, v in edges] # Add the reverse edges

    #print(edges)
    #print(edges_duplicated)

    ## Make an adjacency list
    adj, m = build_directed_adj_from_parent(tree, n)
    #print(adj)

    ## Make an Euler cycle
    euler_path = euler_cycle(adj, start=start_euler)
    #print(euler_path)

    # Euler tour must use every arc exactly once -> length m + 1
    if len(euler_path) != m + 1:
        raise ValueError(
            f"Euler tour did not consume all edges: len(path)={len(euler_path)} expected={m+1}."
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

    lattices = sorted(os.listdir(str(lattice_dir)))
    lattices = [lattice for lattice in lattices if lattice.endswith(lattice_ext)]

    mst_mode = kwargs.get("mst_mode", "min")
    reverse_seq = kwargs.get("reverse", True)

    print(f"Twice Around: mst_mode: {mst_mode}, reverse: {reverse_seq}.\n")

    for lattice in lattices:

        # Get n
        n = int(lattice.split("_")[2])

        # Make lattice path
        lattice_path = lattice_dir / lattice

        # To store values
        all_values = []
        all_sequences = []

        # For all starting points
        for start in range(0, n):

            # Execute twice around heuristic
            removal_sequence, weitzman_value = twice_around(lattice_path=lattice_path, labeled=lattice_labeled, n=n,
                                                    start_mst=start, start_euler=start,
                                                    mst_mode=mst_mode, reverse=reverse_seq)

            #print(removal_sequence, " -> ", weitzman_value)

            all_values.append(weitzman_value)
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

