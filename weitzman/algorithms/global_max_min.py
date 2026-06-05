"""
Global Max-Min heuristic for the Weitzman diversity index.

Strategy
--------
Builds an insertion sequence Q greedily by always selecting the point
outside Q that maximises the minimum distance to any point already in Q:

    i* = argmax_{u \\in A\\Q} ( min_{v \\in Q} D(u, v) )

Initialisation: Q <- {seed vertex a}.
At each step: select i*, append it to the insertion list L, add it to Q.
Repeat until |Q| = |A|.  Reverse L to obtain the removal sequence.

The rule is a "max of mins": it picks whichever unselected point is
most isolated from the current set Q - maximising the point-to-set
distance at every insertion step.

The heuristic is run exhaustively over all n possible seed vertices;
the best resulting Weitzman value across all seeds is used in analysis.

Complexity: O(n^3) - n insertion steps, each scanning O(n) incremental
update of the nearest-Q distance vector, n starting points.
"""

import os
import json
import time
import logging
import numpy as np

from tqdm import tqdm
from pathlib import Path
from numpy.typing import NDArray

from weitzman.io.loaders import load_lattice
from weitzman.io.writers import write_values
from weitzman.utils.operations import evaluate_removal_sequence, compute_distance_matrix

logger = logging.getLogger(__name__)


def _build_sequence(
    n: int,
    start: int,
    d_matrix: NDArray[np.float64],
) -> tuple[int, ...]:
    """
    Build one insertion sequence starting from seed `start` and return
    it reversed as a removal sequence.

    Parameters
    ----------
    n       : number of points (vertices are 0 ... n-1).
    start   : index of the seed vertex (first point added to Q).
    d_matrix: (n, n) pairwise distance matrix.
    """
    in_Q = np.zeros(n, dtype=bool)
    in_Q[start] = True              # Q <- {start}
    insertion_list = [start]        # L <- (start,)

    # min_dist[u] = minimum distance from u to any point currently in Q.
    # Seed is the only Q member at start.
    min_dist = d_matrix[:, start].copy()

    while len(insertion_list) < n:
        # i* = argmax over unvisited of that nearest-Q distance.
        # Among all candidate points, pick the one most isolated from Q.
        i_star = int(np.argmax(np.where(~in_Q, min_dist, -np.inf)))

        in_Q[i_star] = True              # Q <- Q U {i*}
        insertion_list.append(i_star)    # L <- (L, i*)

        # Incremental update: i_star just joined Q.
        # For every point u, check whether d(u, i_star) improves its nearest-Q distance.
        np.minimum(min_dist, d_matrix[:, i_star], out=min_dist)

    # Reverse: the last point inserted (most globally isolated at that step)
    # whose min-distance to Q was maximized at that step, becomes the first removed,
    # contributing the largest point-to-set distance when scored by Weitzman.
    return tuple(insertion_list[::-1])


def _run_instance(lattice_path: Path, labeled: bool, outdir: Path) -> None:
    """Run the heuristic for every possible seed vertex on one instance."""
    lattice, element_mapping = load_lattice(str(lattice_path), labeled=labeled)
    d_matrix = compute_distance_matrix(lattice)
    n = len(element_mapping)

    sequences: list[tuple[int, ...]] = []
    values: list[float] = []

    # Exhaustive seed: each of the n vertices is tried as starting point.
    # Different seeds yield different insertion orders and hence different
    # Weitzman values; the distribution across seeds is saved in full.
    for start in range(n):
        seq = _build_sequence(n, start, d_matrix)
        sequences.append(seq)
        values.append(evaluate_removal_sequence(seq, d_matrix))

    original_name = lattice_path.stem
    write_values(np.array(values),
                 str(outdir / "values"    / f"values_{original_name}.npy"))
    write_values(np.array(sequences, dtype=np.int16),
                 str(outdir / "sequences" / f"sequences_{original_name}.npy"))


def run(lattice_params: dict, outdir: Path, seed: int, **kwargs) -> None:
    lattice_dir   = Path(lattice_params["lattice_dir"])
    lattice_ext   = lattice_params["lattice_ext"]
    lattice_labeled = lattice_params["lattice_labeled"]

    instances = sorted(f for f in os.listdir(lattice_dir) if f.endswith(lattice_ext))
    logger.info("GlobalMaxMin | %d instances", len(instances))

    timing: dict[str, float] = {}
    for name in tqdm(instances, desc="GMM", unit="inst"):
        t0 = time.perf_counter()
        _run_instance(lattice_dir / name, lattice_labeled, outdir)
        timing[Path(name).stem] = time.perf_counter() - t0
    (outdir / "timing.json").write_text(json.dumps(timing, indent=2))
