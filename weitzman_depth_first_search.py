import os
import gc
import time
import itertools
import numpy as np
from tqdm import tqdm
from numpy.typing import NDArray
from multiprocessing import Pool, cpu_count

from auxiliar import point_to_set_distance, load_lattice_deprecated, compute_distance_matrix

# ----- Declare a global variable for all workers -----
_distance_matrix: NDArray[np.float64] = np.array([], dtype=np.float64)


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


def enumerate_weitzman_paths(distance_matrix: NDArray[np.float64], batch_size: int = 50_000):
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
    main_path = "Weitzman_factorial_results/"
    values_path = os.path.join(main_path, "values", f"values_{n:03d}_points.npy")
    best_sequences_path = os.path.join(main_path, "best_sequences", f"best_sequences_{n:03d}_points.npy")
    worst_sequences_path = os.path.join(main_path, "worst_sequences", f"worst_sequences_{n:03d}_points.npy")
    
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


def main():

    for i in range(4, 13):

        # Permutation size
        n = i

        # Load lattice
        lattice, _ = load_lattice_deprecated(f"Small_Linear_Lattices/Linear_3D_{n:03d}_1.00.txt")

        # Compute distance matrix
        distance_matrix = compute_distance_matrix(lattice)

        # Compute all paths of Weitzman
        print(f"Starting computation for n! (n={n}) cases.")
        t0 = time.time()
        enumerate_weitzman_paths(distance_matrix, batch_size=5000)
        print(f"n = {n}, {round(time.time() - t0, 4)} seconds.\n")

if __name__ == "__main__":
    main()
