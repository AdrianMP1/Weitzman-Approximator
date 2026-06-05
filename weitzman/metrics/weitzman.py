import numpy as np
from numpy.typing import NDArray

from weitzman.utils.operations import point_to_set_distance


def recursive_weitzman(
    vertices: set[int],
    d_matrix: NDArray[np.float64],
) -> float:
    """
    Exact Weitzman diversity via the original recursive formulation:
        W(A) = max_{i in A} { W(A \\ {i}) + D(i, A \\ {i}) }

    Complexity: O(n!) - only feasible for small n (up to ~12).
    For large sets use the heuristic algorithms instead.
    """
    if len(vertices) == 1:
        return 0.0

    best = 0.0
    for element in vertices:
        sub = vertices - {element}
        min_dist, _ = point_to_set_distance(element, sub, d_matrix, kind="min")
        value = recursive_weitzman(sub, d_matrix) + min_dist
        if value > best:
            best = value

    return best
