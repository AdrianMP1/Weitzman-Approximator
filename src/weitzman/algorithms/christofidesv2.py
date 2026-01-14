"""
Christofides algorithm adaptation for (approx.) Weitzman computation
"""
import os
import numpy as np
import networkx as nx

from numpy.typing import NDArray
from itertools import combinations

from weitzman.io.loaders import load_lattice
from weitzman.io.writers import write_values, make_save_folder
from weitzman.utils.operations import point_to_set_distance
from weitzman.utils.operations import evaluate_removal_sequence
from weitzman.utils.operations import shortcut_cycle, euler_cycle
from weitzman.utils.operations import compute_distance_matrix, spanning_tree


def eulerian_circuit_max_weight(adj, start):
    """
    Eulerian circuit with greedy construction; requires a starting node and a dictionary of adjacency lists
    """
    adj = {u: neighbors.copy() for u, neighbors in adj.items()}

    stack = [start]
    circuit = []

    while stack:
        v = stack[-1]

        if adj[v]:
            idx = max(range(len(adj[v])), key=lambda i: adj[v][i][1])
            u, w = adj[v].pop(idx)

            for j, (x, _) in enumerate(adj[u]):
                if x == v:
                    adj[u].pop(j)
                    break

            stack.append(u)
        else:
            circuit.append(stack.pop())

    return circuit[::-1]


def multigraph_to_adjlist(multigraph):
    adj = {u: [] for u in multigraph.nodes}

    for u, v, data in multigraph.edges(data=True):
        w = data["weight"]
        adj[u].append([v, w])
        adj[v].append([u, w])

    return adj


def christofides_weitzman(lattice_path: Path, labeled: str, n: int,
                          mst_mode: str = "max", reverse: bool = False) -> None:
    """
    Heuristic inspired on christofides algorithm, repurposed for weitzman
    Note: This implementation uses NetworkX library; when kind == 'min' this implementation is equivalent to the original Christofides algorithm for the tsp
    """
    # Load lattice
    lattice, element_mapping = load_lattice(str(lattice_path), labeled=labeled)

    # Compute distance matrix
    d_matrix = compute_distance_matrix(lattice = lattice)

    # Get the unique index for each lattice point
    vertices: NDArray[np.int64] = np.array(list(element_mapping.keys()))

    # Initialize a complete graph using NetworkX Graphs
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=d_matrix[i,j])

    # Compute the maximum/minimum spanning tree
    if mst_mode == "max":
        T = nx.maximum_spanning_tree(G, weight="weight")
    elif mst_mode == "min":
        T = nx.minimum_spanning_tree(G, weight="weight")
    else:
        raise ValueError(f"mst_mode must be either max/min. {mst_mode} was given.")

    # Get the odd nodes from the tree
    odd_nodes = [v for v, d in T.degree() if d % 2 == 1]

    # Compute the maximum-/minimum-weight perfect matching on the odd nodes (consider the complete graph edges)
    M = G.subgraph(odd_nodes).copy()
    if mst_mode == "max":
        matching = nx.algorithms.matching.max_weight_matching(M, weight="weight")
    elif mst_mode == "min":
        matching = nx.algorithms.matching.min_weight_matching(M, weight="weight")

    # Add the matching edges to T, to generate a multigraph
    multigraph = nx.MultiGraph()
    multigraph.add_edges_from(T.edges(data=True))
    for u, v in matching:
        multigraph.add_edge(u, v, weight=d_matrix[u,v])

    # Get dictionary of adj lists
    adj_dict = multigraph_to_adjlist(multigraph)

    # To store data
    sequences = []

    for i in range(n):
        euler = eulerian_circuit_max_weight(adj_dict, start=i)

        visited = set()
        sequence = []
        for v in euler:
            if v not in visited:
                visited.add(v)
                sequence.append(v)

        sequences.append(sequence[::-1])

    # --- NEW ---
    values = []
    for sequence in sequences:
        w = evaluate_removal_sequence(sequence, d_matrix)
        values.append(w)

    return sequences, values


def run(lattice_params: dict[str, str | Path | bool], outdir: Path, **kwargs) -> None:
#def run(lattice_path: str, kind: str = "max"):

# Extract parameters
    lattice_dir = lattice_params["lattice_dir"]
    lattice_ext = lattice_params["lattice_ext"]
    lattice_labeled = lattice_params["lattice_labeled"]

    # Load the lattices
    lattices = sorted(os.listdir(str(lattice_dir)))
    lattices = [lattice for lattice in lattices if lattice.endswith(lattice_ext)]

    ## Get algorithm parameters
    mst_mode = kwargs.get("mst_mode", "max")
    reverse_seq = kwargs.get("reverse", True)

    print(f"Christofides: mst_mode: {mst_mode}, reverse: {reverse_seq}.\n")

    # Solve all lattices
    for lattice in lattices:

        # Get n
        n = int(lattice.split("_")[2])

        # Make lattice path
        lattice_path = lattice_dir / lattice

        # Execute Christofides
        all_sequences, all_values = christofides_weitzman(lattice_path=lattice_path, labeled=lattice_labeled, n=n,
                                                   mst_mode=mst_mode, reverse=reverse_seq)

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

        # Compute stats
        print(f"Chr-Heuristic, n = {n}")
        print("Mean: ", np.mean(vector))
        print("Std: ", np.std(vector))
        print("[min, q1, median, q3, max] ->", np.percentile(vector, [0, 25, 50, 75, 100]))
        print("\n")
