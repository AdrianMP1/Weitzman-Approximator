import numpy as np
from numpy.typing import NDArray

def read_front(file_path: str) -> NDArray[np.float64]:
    
    data = np.loadtxt(file_path)
    return data


def find_front(geometry: str, coverage: str) -> str:

    path = "PFAs"

    if geometry == "linear":
        path += "_Linear/Linear_3D"
    elif geometry == "Convex":
        path += "_Convex/Convex_3D"
    else:
        path += "_Concave/Concave_3D"
    

    path += f"_105_{coverage}.POF"

    return path


def load_lattice(lattice_path: str) -> tuple[NDArray[np.float64], dict[int, NDArray[np.float64]]]:
    """
    ...
    """

    data_points = read_front(lattice_path)
    element_mapping = {i: e for i, e in enumerate(data_points)}

    return data_points, element_mapping


def load_lattice_deprecated(lattice_path: str) -> tuple[NDArray[np.float64], dict[int, NDArray[np.float64]]]:

    labels: list[str] = []
    array_points: list[list[float]] = []

    with open(lattice_path, "r") as f:
        for line in f:
            
            points, label = line.split(" -> ")

            labels.append(label.strip())
            array_points.append(list(map(float, points.split(" "))))

    lattice = np.array(array_points)
    element_mapping = {i: e for i, e in enumerate(lattice)}

    return lattice, element_mapping


def compute_distance_matrix(lattice: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    lattice: n x m matrix, with n being the number of points and m the number of objectives
    """

    # Compute the n x n distance matrix using the 2-norm
    diff = lattice[:, np.newaxis, :] - lattice[np.newaxis, :, :]
    d_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    return d_matrix


def point_to_set_distance(current_index: int, elements: set[int],
                        d_matrix: NDArray[np.float64],
                        kind: str = "min") -> tuple[float, int]:
    """
    Get the min/max distance from element to a set A
    """
    
    # Allows to change the comparison
    multiplier = 1 if kind != "min" else -1

    # Start variables
    best_distance = float("-inf")
    best_candidate = -1

    # Loop over the set of elements
    for list_index in elements:

        # Sanity check
        if current_index == list_index:
            continue

        # Get the distance between current point and new point using the distance matrix
        current_distance: NDArray[np.float64] = d_matrix[current_index, list_index]

        # Compare against best distance
        if best_distance < current_distance * multiplier:
            best_distance = current_distance.item() * multiplier
            best_candidate = list_index

    # Multiply best_distance by multiplier to fix the sign (in case its minimize)
    best_distance *= multiplier

    return best_distance, best_candidate


def better_than(new: NDArray[np.float64], old: NDArray[np.float64],
                mode: str) -> NDArray[np.bool]:
    """
    Comparison of NDArrays
    
    Returns: NDArray with bool values.
    """
    if mode == "min":
        return new < old
    elif mode == "max":
        return new > old
    
    raise ValueError("Mode must be 'min' or 'max' only.")


def spanning_tree(
        d_matrix: NDArray[np.float64],
        start: int = 0,
        mode: str = "min"
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Prim's algorithm for dense (fully connected) graphs.
    """

    # Copy adjacent matrix
    dist = d_matrix
    
    # Get the number of nodes/vertices
    n = dist.shape[0]

    # --- Initialize arrays ---
    ## Key values used to pick min/max weight edge in cut
    key = np.full(n, np.inf if mode == "min" else -np.inf)

    ## Array to store constructured MST
    parent = np.full(n, -1, dtype=int)
    
    ## Set all vertices as not in MST
    in_mst: NDArray[np.bool] = np.zeros(n).astype(np.bool)

    # Start from vertex start, set node as root (-1)
    key[start] = 0.0
    parent[start] = -1

    # Choose worst value based on min or max
    worst_value = np.inf if mode == "min" else -np.inf
    
    for _ in range(n):
        # Pick vertex not in MST with best (min or max) distance
        ## Mask out vertices already in MST by setting key -> worst_value
        masked_keys = np.where(in_mst, worst_value, key)
        u = np.argmin(masked_keys) if mode == "min" else np.argmax(masked_keys)

        # Put the min/max distance vertex in the MST
        in_mst[u] = True

        # Update distance value of the adjacent vertices not in MST
        # only if the current distance is better than new distance
        not_in_mst: NDArray[np.bool] = ~in_mst
        better: NDArray[np.bool] = better_than(dist[u], key, mode)
        update_mask = not_in_mst & better

        # Update
        key[update_mask] = dist[u, update_mask]
        parent[update_mask] = u

    #for i in range(1, n):
    #    print(parent[i], "-", i, "\t", dist[parent[i], i])

    return parent, key

