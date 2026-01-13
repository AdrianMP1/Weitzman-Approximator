
import numpy as np

from scipy.spatial.distance import pdist, squareform

from weitzman.utils.operations import point_to_set_distance


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
