# ------------------------------
# Diversity indicators 
# ------------------------------
import numpy as np
from scipy.spatial.distance import pdist, squareform

def weitzman_diversity(points):
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

def max_min_diversity(points):
    dist_matrix = squareform(pdist(points, metric='euclidean'))
    np.fill_diagonal(dist_matrix, np.inf) 
    return np.min(dist_matrix)

def solow_polasky_diversity(points, theta=10):
    dist_matrix = squareform(pdist(points, metric='euclidean'))
    M = np.exp(-theta * dist_matrix)
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        M_inv = np.linalg.pinv(M)
    return np.sum(M_inv)

def riesz_s_energy(points):
    m = points.shape[1]
    s = m + 1                      
    dist_matrix = squareform(pdist(points, metric='euclidean'))
    dist_matrix[dist_matrix == 0] = np.inf
    return np.sum(1.0 / dist_matrix**s)