"""
Compute an heuristic based on nearest neighbour.

Note:   This heuristic is a greedy method that starts backwards on the Weitzman paths,
        that is, it assumes that at every moment it is at the leaves level on the Weitzman tree.
        
        The assumption implies that it constructs a sequence of insertions, not removals,
        hence it is required to inverse the list/vector/permutation produced.
"""

import os
import numpy as np
from pathlib import Path
from numpy.typing import NDArray

from weitzman.io.writers import write_values, make_save_folder
from weitzman.io.loaders import load_lattice

from weitzman.utils.operations import evaluate_removal_sequence
from weitzman.utils.operations import compute_distance_matrix, point_to_set_distance


def distance_neighbour(vertices: NDArray[np.int64], starting_vertex: np.int64, 
                       d_matrix: NDArray[np.float64],
                       element_mapping: dict[int, NDArray[np.float64]],
                       kind: str) -> tuple[int]:
    """
    Heuristic inspired on nearest neighbour.

    Note:   The sum of distances at each step of the heuristic under kind = "max" is not equivalent to Weitzman.
            Adding the distances like this violates the core definition of point to set established by Weitzman which grants someof its properties.
    """
    # Initialize all vertices as unvisited
    visited = np.array([False] * len(vertices))

    # Set current vertex & mark as visited
    current = starting_vertex.item()
    visited[current] = True

    # Make the sequence from bottom to top
    sequence = [current]

    # Loop until all vertices were visited
    while not(all(visited)):

        # Find the shortest/farthest edge connecting the current vertex
        _, next_vertex = point_to_set_distance(current, vertices[~visited],
                                                     d_matrix, kind)

        # Set next_vertex as current & mark as visited
        current = next_vertex
        visited[current] = True

        # Add it to the sequence
        sequence.append(current)

    sequence = tuple(sequence[::-1])

    return sequence


def exhaustive_run(lattice_path: Path, labeled: bool, outdir: Path = Path("results/"), kind: str = "max"):
    """
    Run the heuristic for each vertex of the lattice.
    """

    # Vertices
    ## Load the lattice
    lattice, element_mapping = load_lattice(str(lattice_path), labeled=labeled)

    # Compute distance matrix
    d_matrix = compute_distance_matrix(lattice = lattice)

    # Get the unique index for each lattice point
    vertices: NDArray[np.int64] = np.array(list(element_mapping.keys())) 

    # Store all sequences (|vertices| tuples)
    all_sequences: list[tuple[int]] = []
    weitzman_values: list[float] = []

    # Loop over vertices and execute the heuristic for each starting point
    for vertex in vertices:

        # Run the heuristic (The sequence is already reversed, i.e, a removal sequence)
        removal_sequence = distance_neighbour(vertices, starting_vertex=vertex,
                           d_matrix=d_matrix, element_mapping=element_mapping,
                           kind=kind)

        # Compute its Weitzman value.
        w_value = evaluate_removal_sequence(removal_sequence, d_matrix=d_matrix)

        # Store the generated sequence and its value
        all_sequences.append(removal_sequence)
        weitzman_values.append(w_value)

    # Compute stats
    n = d_matrix.shape[0]
    vector = np.array(weitzman_values)
    print(f"NN-Heuristic, n = {n}")
    print("Mean: ", np.mean(vector))
    print("Std: ", np.std(vector))
    print("[min, q1, median, q3, max] ->", np.percentile(vector, [0, 25, 50, 75, 100]))

    # Get original file name - no extension
    original_name = lattice_path.name.rsplit(".", 1)[0]

    # Write values
    file_name = f"values_{original_name}.npy"
    file_path = str(outdir / "values" / file_name)
    write_values(vector, file_path)

    # Write sequences
    sequences = np.array(all_sequences, np.int16)
    file_name = f"sequences_{original_name}.npy"
    file_path = str(outdir / "sequences" / file_name)
    write_values(sequences, file_path)

    print("\n")


def run(lattice_params: dict[str, str | Path | bool], outdir: Path, **kwargs) -> None:
    """
    Execute the heuristic for each file in the lattice_path folder.
    """

    # Extract parameters
    kind = kwargs.get("kind", "max")

    lattice_path = lattice_params["lattice_path"]
    lattice_ext = lattice_params["lattice_ext"]
    lattice_labeled = lattice_params["lattice_labeled"]

    # Access the file names
    instance_names = sorted(os.listdir(lattice_path))
    instance_names = [instance for instance in instance_names if instance.endswith(lattice_ext)]

    for instance_name in instance_names:

        instance = lattice_path / instance_name
        exhaustive_run(instance, lattice_labeled, outdir, kind=kind)

