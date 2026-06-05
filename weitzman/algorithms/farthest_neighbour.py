"""
Farthest-Neighbour heuristic for the Weitzman diversity approximation.

Strategy
--------
In Weitzman's removal-permutation scoring, every element e_i contributes
D(e_i, {e_i+1, ..., e_n}) - its nearest-neighbour distance to the points
still in the set. The last element (the final singleton) always contributes 0.
The largest potential contribution comes from the last pair (e_n-1, e_n):
it equals d(e_n-1, e_n), the direct distance between the two surviving points.

This heuristic greedily maximizes that last-pair distance at every step,
treating every current position as the beginning of a new last-pair scenario:

  1. Choose a seed s (tried exhaustively - every vertex is used once as seed).
  2. Greedily insert the unvisited point farthest from the current vertex
     (kind='max'), then advance the current vertex to that point.
  3. Reverse the insertion order to obtain the removal sequence \rho.

After reversal, s becomes e_n (contributes 0) and the first-chosen point
(farthest from s) becomes e_n-1, so the last-pair distance d(e_n, e_n-1)
is maximized by the opening greedy step.  Applying the same rule at each
subsequent step extends this assumption recursively down the chain.
The best permutation across all seeds is retained at evaluation time.

Note: kind='max' does NOT maximize the Weitzman value directly because
D(e_1, S) is a point-to-set distance over the remaining set, not a single max
pairwise distance

Complexity: O(n^3) total - n seeds x O(n^2) per seed.
Per seed dominated by: n insertions steps x O(n) argmax scan.
"""

import os
import logging
import json
import time
import numpy as np

from tqdm import tqdm
from pathlib import Path
from numpy.typing import NDArray

from weitzman.io.loaders import load_lattice
from weitzman.io.writers import write_values

from weitzman.utils.operations import (
    evaluate_removal_sequence,
    compute_distance_matrix,
    point_to_set_distance,
)

logger = logging.getLogger(__name__)


def _build_sequence(
    vertices: NDArray[np.int64],
    start: int,
    d_matrix: NDArray[np.float64],
    kind: str,
) -> tuple[int, ...]:
    """
    Build one removal sequence starting from `start`.

    The sequence is constructed as an insertion order and then reversed,
    so index 0 of the returned tuple is the first point to be *removed*.
    """
    visited = np.zeros(len(vertices), dtype=bool)
    current = start
    visited[current] = True
    sequence = [current]             # insertion order: start is inserted first

    while not np.all(visited):
        # Identify which vertices have not yet been inserted.
        unvisited_set = set(vertices[~visited].tolist())

        # Pick the unvisited point with the max (or min) distance from `current`.
        # This is the greedy step: it decides which point is "next" in the tour.
        _, next_v = point_to_set_distance(current, unvisited_set, d_matrix, kind)

        current = next_v
        visited[current] = True
        sequence.append(current)     # append in insertion order

    # Reverse: the first point *inserted* (the seed) becomes the last one
    # to be removed (e_n), contributing 0 as a final singleton.  The second
    # inserted point (chosen as the farthest from the seed) becomes e_n-1,
    # whose contribution d(e_n-1, e_n) is exactly the last-pair distance the
    # opening greedy step maximized.

    # An optimal Weitzman removal sequence tend to remove the farthest points
    # at the last steps. This heuristic tries to achieve this.
    return tuple(sequence[::-1])


def _run_instance(lattice_path: Path, labeled: bool, outdir: Path, kind: str) -> None:
    """Run the heuristic for every possible starting vertex on one instance."""
    lattice, element_mapping = load_lattice(str(lattice_path), labeled=labeled)
    d_matrix = compute_distance_matrix(lattice)
    vertices = np.array(list(element_mapping.keys()), dtype=np.int64)

    sequences: list[tuple[int, ...]] = []
    values: list[float] = []

    # Exhaustive starting vertices: each vertex is tried as the seed once.
    # The best value across all starts is taken during analysis/plotting.
    for vertex in vertices:
        seq = _build_sequence(vertices, int(vertex), d_matrix, kind)
        sequences.append(seq)
        values.append(evaluate_removal_sequence(seq, d_matrix))

    original_name = lattice_path.stem
    write_values(np.array(values),
                 str(outdir / "values" / f"values_{original_name}.npy"))
    write_values(np.array(sequences, dtype=np.int16),
                 str(outdir / "sequences" / f"sequences_{original_name}.npy"))


def run(lattice_params: dict, outdir: Path, seed: int, **kwargs) -> None:
    """Entry point called by the experiment runner."""
    kind = kwargs.get("kind", "max")
    lattice_dir = Path(lattice_params["lattice_dir"])
    lattice_ext = lattice_params["lattice_ext"]
    lattice_labeled = lattice_params["lattice_labeled"]

    instances = sorted(f for f in os.listdir(lattice_dir) if f.endswith(lattice_ext))
    logger.info("FarthestNeighbour | kind=%s | %d instances", kind, len(instances))

    timing: dict[str, float] = {}
    for name in tqdm(instances, desc="FN", unit="inst"):
        t0 = time.perf_counter()
        _run_instance(lattice_dir / name, lattice_labeled, outdir, kind)
        timing[Path(name).stem] = time.perf_counter() - t0
    (outdir / "timing.json").write_text(json.dumps(timing, indent=2))
