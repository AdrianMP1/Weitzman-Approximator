"""
Brute-force exact solver for the Weitzman diversity index.

Enumerates all n! removal sequences and evaluates each one.
The maximum value found is the true Weitzman diversity W(A).

This is used to:
  - Obtain ground-truth W(A) for small instances (n ≤ 12).
  - Characterise the full distribution of Weitzman values over all
    permutations — revealing the search-space structure that heuristics
    must navigate.
  - Identify the best and worst removal sequences for visualisation.

Complexity: O(n!) — feasible only for small n.
Uses multiprocessing to parallelise permutation evaluation across CPU cores.
"""

import gc
import logging
import itertools
import numpy as np

from tqdm import tqdm
from pathlib import Path
from numpy.typing import NDArray
from multiprocessing import Pool, cpu_count

from weitzman.io.loaders import load_lattice
from weitzman.utils.operations import compute_distance_matrix, factorial, evaluate_removal_sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------

# The distance matrix is stored as a module-level global so it is shared
# (read-only) across worker processes without being serialised per task.
_distance_matrix: NDArray[np.float64]


def _init_worker(distance_matrix: NDArray[np.float64]) -> None:
    """Initialise each worker process with the shared distance matrix."""
    global _distance_matrix
    _distance_matrix = distance_matrix


def _worker_evaluate(perm: tuple[int, ...]) -> tuple[tuple[int, ...], float]:
    """Evaluate one permutation; called in a worker process."""
    value = evaluate_removal_sequence(perm, _distance_matrix)
    return perm, value


# ---------------------------------------------------------------------------
# Core enumeration
# ---------------------------------------------------------------------------

def enumerate_all_paths(
    distance_matrix: NDArray[np.float64],
    results_dir: Path,
    batch_size: int = 50_000,
) -> None:
    """
    Evaluate all n! permutations of {0, …, n-1} and save:
      - values/values_{n:03d}_points.npy       : all n! Weitzman scores
      - best_sequences/best_sequences_{n:03d}_points.npy  : optimal sequences
      - worst_sequences/worst_sequences_{n:03d}_points.npy: worst sequences

    Parameters
    ----------
    distance_matrix : (n, n) pairwise distance matrix for the instance.
    results_dir     : root directory where subdirectories are expected to
                      already exist (values/, best_sequences/, worst_sequences/).
    batch_size      : chunksize passed to imap_unordered; controls the
                      granularity of work distributed to each worker.
    """
    n = distance_matrix.shape[0]
    total = factorial(n)

    # Use (cpu_count − 2) workers to leave the OS and main process headroom.
    n_workers = max(1, cpu_count() - 2)

    # Avoid materialising all n! tuples in memory simultaneously.
    perms_iter = itertools.permutations(range(n))

    best_value = 0.0
    worst_value = np.inf
    best_sequences: list[tuple[int, ...]] = []
    worst_sequences: list[tuple[int, ...]] = []

    # Pre-allocate the full values array (n!).
    all_values = np.empty(total, dtype=np.float64)
    counter = 0

    logger.info("Enumerating %d permutations for n=%d using %d workers",
                total, n, n_workers)

    with Pool(processes=n_workers,
              initializer=_init_worker,
              initargs=(distance_matrix,)) as pool:

        for perm, value in tqdm(
            pool.imap_unordered(_worker_evaluate, perms_iter, chunksize=batch_size),
            total=total,
        ):
            # Track the best value and all sequences that achieve it.
            if value > best_value:
                best_value = value
                best_sequences = [perm]           # new best found — reset list
            elif value == best_value:
                best_sequences.append(perm)       # tie - keep all optimal sequences

            # Track the worst value symmetrically.
            if value < worst_value:
                worst_value = value
                worst_sequences = [perm]
            elif value == worst_value:
                worst_sequences.append(perm)

            all_values[counter] = value
            counter += 1

    # --- Save results ---
    stem = f"{n:03d}_points"

    np.save(results_dir / "values"          / f"values_{stem}.npy",          all_values)
    np.save(results_dir / "best_sequences"  / f"best_sequences_{stem}.npy",
            np.array(best_sequences,  dtype=np.int16))
    np.save(results_dir / "worst_sequences" / f"worst_sequences_{stem}.npy",
            np.array(worst_sequences, dtype=np.int16))

    logger.info("n=%d | best=%.4f | worst=%.4f", n, best_value, worst_value)

    # Explicitly release the large arrays before moving to the next n.
    del all_values, best_sequences, worst_sequences
    gc.collect()
