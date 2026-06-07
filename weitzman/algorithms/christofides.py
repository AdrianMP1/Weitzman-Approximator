"""
Christofides-inspired heuristic for the Weitzman diversity index.

Strategy
--------
Adapts the classical Christofides TSP algorithm to produce Weitzman
removal sequences.  Compared to Twice-Around, the key difference is
in how the Eulerian multigraph is constructed:

  1. Build a spanning tree T (max or min, controlled by mst_mode).
  2. Identify the *odd-degree* vertices of T.
     (An Euler circuit requires all degrees to be even; Twice-Around
     fixes this by doubling all edges - Christofides only adds a
     perfect matching on the odd-degree subset.)
  3. Compute a max- or min-weight perfect matching M on the odd vertices.
  4. Combine T and M into an Eulerian multigraph.
  5. Reverse the path to obtain a removal sequence (reverse=True, the
     algorithm-correct default; reverse=False is an experimental variant
     not present in the published pseudocode).

The procedure is repeated from every vertex as the Euler starting point.

Complexity: O(n^3) total - dominated by perfect matching and odd-degree vertices.
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
    _multigraph_to_adjlist,
    _greedy_euler_circuit,
    shortcut_to_hamiltonian,
    prepare_sorted_greedy_euler_adjacency,
    run_greedy_euler_circuit,
)

logger = logging.getLogger(__name__)


def compare_old_vs_fast_greedy_euler(
    multigraph,
    mode: str = "max",
    verbose: bool = True,
) -> None:
    """
    Compare the original greedy Euler implementation against the optimized
    sorted-pointer implementation for every possible starting vertex.

    Raises AssertionError if a mismatch is found.
    """
    old_adj = _multigraph_to_adjlist(multigraph)

    sorted_adj, num_edges = prepare_sorted_greedy_euler_adjacency(
        multigraph,
        mode=mode,
    )

    nodes = list(multigraph.nodes())

    iterator = tqdm(
        nodes,
        desc=f"Comparing Euler circuits ({mode}) CHR",
        unit="start",
        disable=not verbose,
        leave=True,
    )

    for start in iterator:
        old_euler = _greedy_euler_circuit(
            old_adj,
            start=start,
            mode=mode,
        )

        fast_euler = run_greedy_euler_circuit(
            sorted_adj=sorted_adj,
            start=start,
            num_edges=num_edges,
        )

        if old_euler != fast_euler:
            iterator.close()

            print(
                "[compare_old_vs_fast_greedy_euler] MISMATCH FOUND\n"
                f"mode: {mode}\n"
                f"start: {start}\n"
                f"old : {old_euler}\n"
                f"fast: {fast_euler}"
            )

            raise AssertionError(
                "Mismatch found\n"
                f"mode: {mode}\n"
            )


def _solve_instance(
    lattice_path: Path,
    labeled: bool,
    mst_mode: str,
    reverse: bool,
) -> tuple[list[list[int]], list[float]]:
    """
    Run the full Christofides pipeline on one instance.

    Returns all n candidate sequences and their Weitzman values.
    """
    lattice, _ = load_lattice(str(lattice_path), labeled=labeled)
    d_matrix = compute_distance_matrix(lattice)
    n = d_matrix.shape[0]

    # --- Step 1: complete graph and spanning tree ---
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

    # --- Step 2: identify odd-degree vertices ---
    # In any spanning tree of n vertices, exactly those vertices with
    # odd degree need extra edges to become even.
    odd_nodes = [v for v, deg in T.degree() if deg % 2 == 1]

    # --- Step 3: matching on the odd-degree subgraph ---
    # Build a subgraph restricted to the odd vertices; the matching
    # only needs to cover those nodes.
    M_graph = G.subgraph(odd_nodes).copy()

    if mst_mode == "max":
        # Maximum-weight matching: adds the heaviest edges possible,
        # consistent with the max-spanning-tree strategy.
        matching = nx.algorithms.matching.max_weight_matching(
            M_graph, weight="weight", maxcardinality=True
        )
    else:
        # Minimum-weight matching: adds the lightest edges - classical
        # Christofides choice for TSP approximation.
        matching = nx.algorithms.matching.min_weight_matching(
            M_graph, weight="weight"
        )

    # --- Step 4: merge tree + matching into an Eulerian multigraph ---
    # After adding the matching edges, every previously odd-degree vertex
    # gains one more edge and becomes even-degree.
    multigraph = nx.MultiGraph()
    multigraph.add_edges_from(T.edges(data=True))   # all spanning tree edges
    for u, v in matching:
        multigraph.add_edge(u, v, weight=float(d_matrix[u, v]))  # matching edges

    # --- DELETE THIS ---
    compare_old_vs_fast_greedy_euler(multigraph, mst_mode)

    # Make a sorted adjacency list
    sorted_adj, num_edges = prepare_sorted_greedy_euler_adjacency(
            multigraph, mst_mode
    )

    # --- Step 5: Euler circuit -> Hamiltonian shortcut -> removal order ---
    sequences: list[list[int]] = []
    values: list[float] = []

    for start in range(n):
        # Greedy Euler circuit: at each step, take the heaviest (or lightest)
        # available edge from the current vertex.  The greedy choice steers
        # the shortcut toward sequences with larger (or smaller) early steps.
        euler = run_greedy_euler_circuit(
                    sorted_adj=sorted_adj,
                    start=start,
                    num_edges=num_edges,
        )

        # Skip repeated vertices to collapse the Euler circuit into a
        # Hamiltonian path that visits every point exactly once.
        sequence = shortcut_to_hamiltonian(euler)

        if reverse:
            # Same reversal rationale as Twice-Around: flipping the sequence
            # moves late-tour (often central) vertices to early removal positions,
            # leaving isolated peripheral points to survive longest.
            sequence = sequence[::-1]

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
    logger.info("Christofides | mst_mode=%s reverse=%s | %d instances",
                mst_mode, reverse, len(instances))

    timing: dict[str, float] = {}
    for name in tqdm(instances, desc="CHR", unit="inst"):
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
