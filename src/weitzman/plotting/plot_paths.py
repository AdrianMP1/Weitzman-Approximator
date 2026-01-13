
"""
Visualize the removal sequence for Weitzman
"""

import os
import string
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pathlib import Path
from scipy.spatial import Delaunay

from weitzman.io.loaders import load_lattice

from auxiliar import load_lattice_deprecated


def make_3d_mesh(ax, lattice_points, fully_connected: bool = False):

    pts = lattice_points

    if fully_connected:
        # Fully connected graph in 3D...
        n = pts.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                p1 = pts[i]
                p2 = pts[j]
                ax.plot([p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]],
                        color='k', linewidth=0.8, alpha=0.6)
    else:
        # Fit plane (PCA)
        centroid = pts.mean(axis=0)
        U, S, Vt = np.linalg.svd(pts - centroid)
        plane_basis = Vt[:2] # Two orthonormal vectors spanning the plane

        # Project to 2D
        proj_2d = (pts - centroid) @ plane_basis.T # Shape (n, 2)
        proj_2d_jittered = proj_2d + 1e-8 * np.random.randn(*proj_2d.shape)

        # Triangulate
        tri = Delaunay(proj_2d_jittered, qhull_options="QJ")

        # Wire plot
        for simplex in tri.simplices:
            tri_pts = pts[simplex]
            tri_pts = np.vstack([tri_pts, tri_pts[0]])

            ax.plot(tri_pts[:,0], tri_pts[:,1], tri_pts[:,2],
                    color="k", linewidth=0.8)

        # Build Poly3DCollection from the triangulation
        faces = [pts[simplex] for simplex in tri.simplices]

        # Create the surface
        poly = Poly3DCollection(
            faces,
            alpha=0.1,
            edgecolor="k"
        )

        poly.set_facecolor("grey")
        ax.add_collection3d(poly)    

    return ax


# ----------------------------
# Helper: curved segment via quadratic Bezier
# ----------------------------
def arch_curve_3d(p_start, p_end, height=0.5, n_points=50):
    """
    Returns points along a curved 3D path between p_start and p_end.
    Uses a simple quadratic Bezier curve with a control point lifted in z.

    p_start, p_end: np.array shape (3,)
    height: how much to lift the curve above the plane (z-direction)
    n_points: resolution of the curve
    """
    p_start = np.asarray(p_start, dtype=float)
    p_end = np.asarray(p_end, dtype=float)

    # Midpoint between start and end
    p_mid = 0.5 * (p_start + p_end)

    # Lift the midpoint in the z-direction (assuming plane is roughly z=0)
    p_mid[2] += height

    t = np.linspace(0, 1, n_points)
    # Quadratic Bezier: B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
    curve = (
        (1 - t)[:, None] ** 2 * p_start
        + 2 * (1 - t)[:, None] * t[:, None] * p_mid
        + t[:, None] ** 2 * p_end
    )
    return curve  # shape (n_points, 3)


def plot_arrows(ax, path_points, use_curved_arrows, arch_height, arrow_length_ratio):

    for i in range(len(path_points) - 1):
        p_start = path_points[i]
        p_end = path_points[i + 1]

        if use_curved_arrows:

            # Draw a curved arch
            curve = arch_curve_3d(p_start, p_end, height=arch_height, n_points=50)
            line, = ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], linewidth=2)
            color = line.get_color()

            # Choose a middle index for the arrow
            mid_idx = len(curve) // 2

            if mid_idx == 0:
                mid_idx = 1
            if mid_idx == len(curve) - 1:
                mid_idx = len(curve) - 2

            # Small arrowhead near the middle of the curve
            p_mid  = curve[mid_idx]
            p_prev = curve[mid_idx - 2]
            p_next = curve[mid_idx + 2]

            # Local direction along the curve (tangent)
            direction = p_next - p_prev

            # Small arrowhead near the end of the curve
            # Take last two points on the curve to get direction
            #p_prev = curve[-2]
            #p_last = curve[-1]
            #direction = p_last - p_prev
            ax.quiver(
                p_mid[0], p_mid[1], p_mid[2],
                direction[0], direction[1], direction[2],
                length = 0.25,
                normalize = True,
                arrow_length_ratio = 0.75,
                color = color,
            )

            """ ax.quiver(
                p_prev[0], p_prev[1], p_prev[2],
                direction[0], direction[1], direction[2],
                length=0.25,
                normalize=True,
                arrow_length_ratio=arrow_length_ratio
            ) """
        else:
            # Straight arrow using quiver
            direction = p_end - p_start
            line, = ax.plot(
                [p_start[0], p_end[0]],
                [p_start[1], p_end[1]],
                [p_start[2], p_end[2]],
                linewidth=2,
                linestyle='-'
            )
            color = line.get_color()

            # Choose a middle index for the arrow
            p_mid = (p_start + p_end) / 2
            ax.quiver(
                p_mid[0], p_mid[1], p_mid[2],
                direction[0], direction[1], direction[2],
                length=0.1,
                normalize=True,
                arrow_length_ratio=1.25,
                color=color,
            )

    return ax


def plot_removal_sequence(fig, subplot_position: int, solution: tuple[str], lattice_path: str):

    # Number of points
    n_elements = len(solution)

    # Load lattice
    lattice_points, element_mapping = load_lattice_deprecated(lattice_path)

    # Get the name of each vertex
    labels = list(string.ascii_uppercase[:n_elements])

    # Build mapping: label -> index in lattice points
    label_to_idx = {labels[i]: i for i in range(n_elements)}

    # ----------------------------
    # Plotting
    # ----------------------------
    n_axes = len(fig.axes)
    ax = fig.add_subplot(subplot_position, projection='3d')

    # ----------------------------
    # Build the path given by the solution
    # ----------------------------
    path_points = np.array([lattice_points[label_to_idx[l]] for l in solution])

    # Plot the lattice points
    xs, ys, zs = path_points[:, 0], path_points[:, 1], path_points[:, 2]
    ax.scatter(xs[1:-1], ys[1:-1], zs[1:-1], s=40, alpha=1.0)       # Middle removables
    ax.scatter(xs[0], ys[0], zs[0], s=40, alpha=1.0, color="k")     # The first to be removed
    ax.scatter(xs[-1], ys[-1], zs[-1], s=40, alpha=1.0, color="r")  # The last to be removed

    # Add text labels for each lattice point
    for i, label in enumerate(labels):
        x, y, z = lattice_points[i]
        ax.text(x, y, z + 0.05, label, fontsize=10, ha='center')

    # Parameters
    use_curved_arrows = True   # Set False to use straight arrows
    arch_height = 0.5          # Height of arcs above the plane
    arrow_length_ratio = 0.2   # For arrowheads when using quiver

    # Plot path arrows
    ax = plot_arrows(ax, path_points, use_curved_arrows, arch_height, arrow_length_ratio)

    # Make 3D Mesh
    ax = make_3d_mesh(ax, lattice_points, fully_connected=True if n_elements < 8 else False)

    # ----------------------------
    # Styling / view
    # ----------------------------
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set equal-ish aspect ratio
    max_range = (xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min())
    max_range = max(max_range) if max_range != (0, 0, 0) else 1
    mid_x = 0.5 * (xs.max() + xs.min())
    mid_y = 0.5 * (ys.max() + ys.min())
    mid_z = 0.5 * (zs.max() + zs.min())
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

    ax.view_init(elev=30, azim=45)  # adjust angle as you like
    return fig, ax


def plot_removal_sequences(n: int, lattice_dir: str,
                           best_dir: Path, worst_dir: Path,
                           result_dir: Path):

    # Search for lattice
    lattice_path = lattice_dir / f"Linear_3D_{n:03d}_1.00.txt"

    # Search for sequences
    best_path = best_dir / f"best_sequences_{n:03d}_points.npy"
    worst_path = worst_dir / f"worst_sequences_{n:03d}_points.npy"
    result_path = result_dir / f"sequences_{n:03d}_points.npy"

    # Load sequences
    best_matrix = np.load(best_path)
    worst_matrix = np.load(worst_path)
    result_matrix = np.load(result_path)

    # Get 2 random solutions
    m = 2
    rng = np.random.default_rng(seed=42)
    best_matrix = best_matrix[rng.integers(0, len(best_matrix), size=m),:]
    worst_matrix = worst_matrix[rng.integers(0, len(worst_matrix), size=m),:]
    result_matrix = result_matrix[rng.integers(0, len(result_matrix), size=m),:]

    # Transform solutions into ascii letters
    ascii_letters = list(string.ascii_uppercase) # ["A", "B", ..., "Z"]
    n_letters = len(ascii_letters)
    best_sequences = [tuple(ascii_letters[int(x) % n_letters] for x in row) for row in best_matrix]
    worst_sequences = [tuple(ascii_letters[int(x) % n_letters] for x in row) for row in worst_matrix]
    result_sequences = [tuple(ascii_letters[int(x) % n_letters] for x in row) for row in result_matrix]    

    # Make figure
    fig = plt.figure(figsize=(12,8))

    # Plot best sequences
    cont = 1
    for sequence in best_sequences:
        subplot_position = int(f"23{cont}")
        fig, ax = plot_removal_sequence(fig, subplot_position, sequence, lattice_path)

        if cont == 1:
            ax.set_title("Best Sequences")
        cont += 3

    # Plot worst sequences
    cont = 2
    for sequence in worst_sequences:
        subplot_position = int(f"23{cont}")
        fig, ax = plot_removal_sequence(fig, subplot_position, sequence, lattice_path)

        if cont == 2:
            ax.set_title("Worst Sequences")
        cont += 3

    # Plot result sequences
    cont = 3
    for sequence in result_sequences:
        subplot_position = int(f"23{cont}")
        fig, ax = plot_removal_sequence(fig, subplot_position, sequence, lattice_path)

        if cont == 3:
            ax.set_title("Heuristic Sequences")
        cont += 3

    return fig


def get_n_solved(results_dir: Path):

    files = sorted(os.listdir(str(results_dir)))

    n_values: list[int] = []

    for file in files: 
        # Extract the n number following nomenclature "values_XYZ_points.ext"
        n = int(file.split("_")[1])
        n_values.append(n)

    return n_values


def run_paths(run_dir: Path, outdir: Path, **kwargs) -> None:

    # Read parameters
    dpi: int = kwargs.get("dpi", 200)
    show: bool = kwargs.get("show", False)
    results_dir: Path = kwargs["results_path"] / "sequences"
    heuristic_name: str = kwargs.get("heuristic_name", "Unknown heuristic")

    lattice_dir: str = kwargs["lattice_path"]

    precomputed_dir: Path = results_dir.parent.parent.parent.parent / "factorial_results"
    best_dir: Path = precomputed_dir / "best_sequences"
    worst_dir: Path = precomputed_dir / "worst_sequences"

    if not(precomputed_dir.exists()):
        raise FileNotFoundError("The directory results/factorial_results does not exist.")

    # Get lattice solved
    n_values = get_n_solved(results_dir)

    # Make save dir
    save_dir = outdir / "plots"
    save_dir.mkdir(parents=False, exist_ok=True)

    for n in n_values:
        fig = plot_removal_sequences(n, lattice_dir,
                               best_dir, worst_dir,
                               results_dir)

        fig.suptitle(f"Removal sequences, n={n}")
        save_path = save_dir / f"Removal_Sequence_{n:03d}_{heuristic_name}.png"
        fig.savefig(save_path, dpi=dpi)

        if show:
            plt.show()
        plt.close(fig)
