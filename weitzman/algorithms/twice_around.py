"""
Twice-Around-the-Tree heuristic for the Weitzman diversity.

Strategy
--------
Adapts the classical Twice-Around-the-Tree TSP heuristic to produce
Weitzman removal sequences:

  1. Build a spanning tree T of the complete distance graph
     (max spanning tree with mst_mode='max', min with 'min').
  2. Double every edge of T to obtain an Eulerian multigraph
     (all vertex degrees become even, guaranteeing an Euler circuit exists).
  3. Traverse the multigraph with an Euler circuit starting from
     each vertex - at each step the next edge is chosen by weight
     (largest for 'max', smallest for 'min').
  4. Shortcut repeated vertices to obtain a Hamiltonian path
     (triangle-inequality shortcut: skip any vertex already visited).
  5. Reverse the path to obtain a removal sequence (reverse=True, the
     algorithm-correct default).  The starting vertex becomes the final
     singleton e_n (contributing 0); e_n-1 forms
     the last pair whose contribution d(e_n-1, e_n) the heuristic aims
     to maximize.  reverse=False is an experimental variant not present
     in the published pseudocode.

The procedure is repeated from every vertex as a starting point for
the Euler traversal, producing n candidate sequences per instance.

Complexity: O(n^2 * log(n)) dominated by NetworkX Kruskal spanning tree.
"""

import os
import json
import time
import logging
import numpy as np
import networkx as nx

from tqdm import tqdm
from pathlib import Path

from weitzman.io.loaders import load_lattice
from weitzman.io.writers import write_values

from weitzman.utils.operations import (
    compute_distance_matrix,
    evaluate_removal_sequence,
    shortcut_to_hamiltonian,
    prepare_sorted_greedy_euler_adjacency,
    run_greedy_euler_circuit,
)

logger = logging.getLogger(__name__)

def _solve_instance(
    lattice_path: Path,
    labeled: bool,
    mst_mode: str,
    reverse: bool,
) -> tuple[list[list[int]], list[float]]:
    """
    Run the full Twice-Around pipeline on one instance.

    Returns all n candidate sequences and their Weitzman values.
    """
    lattice, _ = load_lattice(str(lattice_path), labeled=labeled)
    d_matrix = compute_distance_matrix(lattice)
    n = d_matrix.shape[0]

    # Build a complete graph and extract its spanning tree
    # NetworkX is used here because it provides robust max/min spanning
    # tree implementations.  The complete graph has n(n-1)/2 edges.
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=float(d_matrix[i, j]))

    if mst_mode == "max":
        T = nx.maximum_spanning_tree(G, weight="weight")
    elif mst_mode == "min":
        T = nx.minimum_spanning_tree(G, weight="weight")
    else:
        raise ValueError(f"mst_mode must be 'max' or 'min', got '{mst_mode}'")

    # Double all tree edges -> Eulerian multigraph
    # Doubling each edge makes every vertex degree even (a necessary and
    # sufficient condition for an Euler circuit to exist on a connected graph).
    multigraph = nx.MultiGraph()
    for u, v, data in T.edges(data=True):
        w = float(data["weight"])
        multigraph.add_edge(u, v, weight=w)   # original edge
        multigraph.add_edge(u, v, weight=w)   # duplicate edge

    # Make a sorted adjacency list
    sorted_adj, num_edges = prepare_sorted_greedy_euler_adjacency(multigraph, mst_mode)

    # Euler circuit -> Hamiltonian shortcut -> removal order
    sequences: list[list[int]] = []
    values: list[float] = []

    for start in range(n):
        # Greedy Euler circuit starting from `start`.
        # Different starting vertices explore different parts of the multigraph
        # first, yielding different shortcuts and therefore different sequences.
        euler = run_greedy_euler_circuit(
                sorted_adj=sorted_adj,
                start=start, num_edges=num_edges,
        )

        # Shortcut: the first visit to each vertex is kept, the rest skipped.
        # This produces a Hamiltonian path (visits every vertex exactly once).
        sequence = shortcut_to_hamiltonian(euler)

        if reverse:
            # Reversing places the vertices encountered *last* in the Euler
            # tour at the front of the removal order.
            # For a max spanning tree, large edges are consumed first by the
            # greedy traversal, so late-visited vertices tend to be central (high-degree);
            # reversing consequently pushes isolated points toward the end
            # of the removal sequence, where the last pair (e_n-1, e_n) provides
            # the largest Weitzman contribution.
            sequence = sequence[::-1]

        if len(set(sequence)) != n or len(sequence) != n:
            raise ValueError(
                f"Shortcut did not produce a valid Hamiltonian path "
                f"(got {len(sequence)} vertices, {len(set(sequence))} unique)."
            )

        sequences.append(sequence)
        values.append(evaluate_removal_sequence(sequence, d_matrix))

    return sequences, values


def run(lattice_params: dict, outdir: Path, seed: int, **kwargs) -> None:
    mst_mode = kwargs.get("mst_mode", "max")
    reverse  = kwargs.get("reverse", True)

    lattice_dir     = Path(lattice_params["lattice_dir"])
    lattice_ext     = lattice_params["lattice_ext"]
    lattice_labeled = lattice_params["lattice_labeled"]

    instances = sorted(f for f in os.listdir(lattice_dir) if f.endswith(lattice_ext))
    logger.info("TwiceAround | mst_mode=%s reverse=%s | %d instances",
                mst_mode, reverse, len(instances))

    timing: dict[str, float] = {}
    for name in tqdm(instances, desc="TAT", unit="inst"):
        t0 = time.perf_counter()
        sequences, values = _solve_instance(
            lattice_dir / name, lattice_labeled, mst_mode, reverse
        )
        original_name = (lattice_dir / name).stem
        write_values(np.array(values),
                     str(outdir / "values"    / f"values_{original_name}.npy"))
        write_values(np.array(sequences, dtype=np.int16),
                     str(outdir / "sequences" / f"sequences_{original_name}.npy"))
        timing[original_name] = time.perf_counter() - t0
    (outdir / "timing.json").write_text(json.dumps(timing, indent=2))
