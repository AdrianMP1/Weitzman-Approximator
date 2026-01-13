"""
Script to compute recursively the Weitzman indicator with its original formulation.
It returns the value that maximizes the weitzman formulation.

Complexity: O(n!) [both cases]
"""

import os
import gc
import time
import logging
import itertools
import numpy as np

from tqdm import tqdm
from pathlib import Path
from numpy.typing import NDArray
from multiprocessing import Pool, cpu_count

from weitzman.io.writers import save_resolved_config
from weitzman.io.loaders import load_lattice, load_config_file
from weitzman.utils.operations import compute_distance_matrix, factorial, evaluate_removal_sequence


logger = logging.getLogger(__name__)


def _init_worker(distance_matrix: NDArray[np.float64]) -> None:
    # Each worker will init this at startup.
    global _distance_matrix
    _distance_matrix = distance_matrix


def _worker_solve(perm: tuple[int]):
    """
    Worker Procedures 
    """

    value = evaluate_removal_sequence(perm, _distance_matrix)
    return perm, value


def enumerate_weitzman_paths(distance_matrix: NDArray[np.float64], results_dir: Path, batch_size: int = 50_000):
    """
    Hypothesis:
    Weitzman explores n! paths and returns the one with maximum value.

    By evaluating the n! removal paths we can observe the distribution of values (similar to a search space). 
    """
    
    # Get the number of elements
    n = distance_matrix.shape[0]
    total_paths = factorial(n)

    # Create a generator to produce the n! permutations.
    perms_iter = itertools.permutations(range(n))

    # Get max number of workers
    max_workers = cpu_count()
    n_workers = max_workers - 2

    # Chunksize (reduces overhead)
    chunksize = batch_size

    # Min and Max values found
    best_value = 0
    worst_value = np.inf

    # To store best and worst sequences
    best_sequences = []
    worst_sequences = []

    # Store all values (RAM EXPENSIVE)
    counter = 0
    all_values = np.empty((total_paths,), dtype=np.float64)

    with Pool(processes=n_workers,
              initializer=_init_worker,
              initargs=(distance_matrix,)) as pool:
        
        for perm, value in tqdm(pool.imap_unordered(_worker_solve, perms_iter, chunksize=chunksize), total=total_paths):
            
            if value > best_value:
                best_value = value
                best_sequences = [perm]

            elif value == best_value:
                best_sequences.append(perm)

            if value < worst_value:
                worst_value = value
                worst_sequences = [perm]
            
            elif value == worst_value:
                worst_sequences.append(perm)

            all_values[counter] = value
            counter += 1

    print("Computation Done!\nSaving Files...")
    
    # Make NPY file path
    values_path = results_dir / "values" / f"values_{n:03d}_points.npy"
    best_sequences_path = results_dir / "best_sequences" / f"best_sequences_{n:03d}_points.npy"
    worst_sequences_path = results_dir / "worst_sequences" / f"worst_sequences_{n:03d}_points.npy"
    
    # Save distribution values
    np.save(values_path, all_values)

    # Save sequences as integer matrices
    best_sequences_matrix = np.array(best_sequences, dtype=np.int16)
    np.save(best_sequences_path, best_sequences_matrix)

    worst_sequences_matrix = np.array(worst_sequences, dtype=np.int16)
    np.save(worst_sequences_path, worst_sequences_matrix)

    # Release memory
    del all_values, best_sequences, worst_sequences
    del best_sequences_matrix, worst_sequences_matrix
    gc.collect()


def run(args) -> int:

    # Make save dirs
    results_dir = args.outdir.parent.parent / "factorial_results"
    results_dir.mkdir(parents=False, exist_ok=True)

    values_dir = results_dir / "values"
    values_dir.mkdir(parents=False, exist_ok=True)

    best_seq_dir = results_dir / "best_sequences"
    best_seq_dir.mkdir(parents=False, exist_ok=True)

    worst_seq_dir = results_dir / "worst_sequences"
    worst_seq_dir.mkdir(parents=False, exist_ok=True)

    # Make the configuration file path
    config_file: str = args.config
    config_path: str = str(args.config_path / config_file) # "dir/" + "name.ext"
    logger.info("Config path: %s", config_path)

    # Load configuration dict
    config = load_config_file(config_path)
    ## Write configuration file into run dir
    save_resolved_config(config, args.outdir)

    ## --- Build Lattice Path ---
    lattice_path = Path(config["data"]["instances_dir"])
    lattices = sorted(os.listdir(str(lattice_path)))

    for j, i in enumerate(range(4, 13)):

        # Permutation size
        n = i

        # Load the lattice
        lattice, _ = load_lattice(str(lattice_path / lattices[j]), labeled=True)

        # Compute distance matrix
        d_matrix = compute_distance_matrix(lattice = lattice)

        # Compute all paths of weitzman
        logger.info(f"Starting computation for n! (n={n}) cases.")
        t0 = time.time()
        enumerate_weitzman_paths(d_matrix, results_dir, batch_size=5000)
        logger.info(f"n = {n}, {round(time.time() - t0, 4)} seconds.\n")

    return 0
