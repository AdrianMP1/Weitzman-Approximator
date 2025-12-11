"""
Script to compute recursively the Weitzman indicator with its original formulation.
It returns the value that maximizes the weitzman formulation.

Complexity: O(n!) [both cases]
"""

import numpy as np
from tqdm import tqdm
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

from auxiliar import point_to_set_distance, compute_distance_matrix, load_lattice_deprecated

def weitzman_diversity(points: NDArray[np.float64]) -> float:
    if len(points) <= 1:
        return 0.0
    dist_matrix = squareform(pdist(points, metric='euclidean'))
    max_diversity = 0.0
    for i in range(len(points)):
        mask = np.ones(len(points), dtype=bool)
        mask[i] = False
        subset_points = points[mask]
        min_dist = np.min(dist_matrix[i, mask])
        diversity = weitzman_diversity(subset_points) + min_dist
        max_diversity = max(max_diversity, diversity)
    return max_diversity


# W(A) = max{W(A \ i) + D(i, A \ i)}
def recursive_weitzman(vertices: set[int], d_matrix: NDArray[np.float64], 
                       element_mapping: dict[int, NDArray[np.float64]]) -> float:
    ...

    # Base case
    if len(vertices) == 1:
        return 0
    
    # Store weitzman values
    weitzman_values: list[float] = []

    # Loop over each element
    for element in vertices:

        # Remove the element
        sub_vertices = vertices - {element}

        # Point to set distance
        min_distance, _ = point_to_set_distance(element, sub_vertices,
                                                 d_matrix, kind="min")

        # Recursively compute
        weitzman_values.append(recursive_weitzman(sub_vertices, d_matrix, element_mapping) + min_distance)

    w = max(weitzman_values)

    return w


def main(lattice_path: str) -> None:
    # Vertices
    ## Load the lattice
    #lattice, element_mapping = load_lattice(lattice_path)
    lattice, element_mapping = load_lattice_deprecated(lattice_path)

    # Compute distance matrix
    d_matrix = compute_distance_matrix(lattice = lattice)

    # Get the unique index for each lattice point
    vertices: NDArray[np.int64] = np.array(list(element_mapping.keys()))

    # Recursive computation
    weitzman_value = recursive_weitzman(set(vertices), d_matrix, element_mapping)

    # Other method
    weitzman_value_matrix = weitzman_diversity(lattice)

    #return weitzman_value, weitzman_value_matrix
    print("Recursive: ", weitzman_value)
    print("Second method: ", weitzman_value_matrix)

if __name__ == "__main__":
    # Number of points
    n = 4

    # Run
    main("Small_Linear_Lattices/Linear_3D_{n:03d}_1.00.txt")
