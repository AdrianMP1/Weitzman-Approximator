"""
Compute an heuristic based on nearest neighbour.

Note:   This heuristic is a greedy method that starts backwards on the Weitzman paths,
        that is, it assumes that at every moment it is at the leaves level on the Weitzman tree.
        
        The assumption implies that it constructs a sequence of insertions, not removals,
        hence it is required to inverse the list/vector/permutation produced.
"""

import numpy as np
from numpy.typing import NDArray
from weitzman_depth_first_search import evaluate_removal_sequence 
from auxiliar import load_lattice, load_lattice_deprecated, compute_distance_matrix, point_to_set_distance 


def distance_neighbour(vertices: NDArray[np.int64], starting_vertex: np.int64, 
                       d_matrix: NDArray[np.float64],
                       element_mapping: dict[int, NDArray[np.float64]],
                       kind: str) -> tuple[int]:
    """
    Heuristic inspired on nearest neighbour.

    Note:   The sum of distances at each step of the heuristic under kind = "max" is not equivalent to Weitzman.
            Adding the distances like this violates the core definition of point to set established by Weitzman which grants the link property.
    """
    # Initialize all vertices as unvisited
    visited = np.array([False] * len(vertices))

    # Set current vertex & mark as visited
    current = starting_vertex.item()
    visited[current] = True

    # Make the sequence from bottom to top
    sequence = [current]

    # Start Weitzman value
    #w_value = 0.0

    # Loop until all vertices were visited
    while not(all(visited)):

        # Find the shortest/farthest edge connecting the current vertex
        distance, next_vertex = point_to_set_distance(current, vertices[~visited],
                                                     d_matrix, kind)

        # Set next_vertex as current & mark as visited
        current = next_vertex
        visited[current] = True

        # Add it to the sequence
        sequence.append(current)

        # Sum the distance
        #w_value += distance
    
    sequence = tuple(sequence[::-1])

    return sequence


def run_nearest_neighbour(lattice_path: str, kind: str = "max"):
    """
    Run the heuristic for each vertex of the lattice.
    """
    # Vertices
    ## Load the lattice
    #lattice, element_mapping = load_lattice(lattice_path)
    lattice, element_mapping = load_lattice_deprecated(lattice_path)

    # Compute distance matrix
    d_matrix = compute_distance_matrix(lattice = lattice)

    # Get the unique index for each lattice point
    vertices: NDArray[np.int64] = np.array(list(element_mapping.keys())) 

    # Store all sequences
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

        #print(sequence, " -> ", value, "\n")

    # Compare all against all [optional]
    """
    for i in range(len(all_sequences)):
        a = all_sequences[i]
        for j in range(i+1, len(all_sequences)):
            b = all_sequences[j]

            if all([a[k] == b[k] for k in range(len(vertices))]):
                print("WARNING: REPEATED")
    """

    # Compute stats
    vector = np.array(weitzman_values)
    print(f"NN-Heuristic, n = {d_matrix.shape[0]}")
    print("Mean: ", np.mean(vector))
    print("Std: ", np.std(vector))
    print("[min, q1, median, q3, max] ->", np.percentile(vector, [0, 25, 50, 75, 100]))
    print("\n")
    
    # Write values
    file_path = f"Weitzman_heuristic_results/farthest_neighbour/values_{n:03d}_points.npy"
    np.save(file_path, vector)


if __name__ == "__main__":
    # Run for multiple n values
    for n in range(4, 13):
        run_nearest_neighbour(f"Small_Linear_Lattices/Linear_3D_{n:03d}_1.00.txt", kind="max")

