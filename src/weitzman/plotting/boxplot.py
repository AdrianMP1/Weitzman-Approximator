
import os
import gc
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)

def get_n_solved(results_dir: Path):

    files = sorted(os.listdir(str(results_dir)))

    n_values: list[int] = []

    for file in files: 
        # Extract the n number following nomenclature "values_XYZ_points.ext"
        n = int(file.split("_")[1])
        n_values.append(n)

    return n_values


def unpack_data(**kwargs):

    # Read parameters
    dpi: int = kwargs.get("dpi", 200)
    show: bool = kwargs.get("show", False)
    results_dir: Path = kwargs["results_path"] / "values"
    heuristic_name: str = kwargs.get("heuristic_name", "Unknown heuristic")

    return dpi, show, results_dir, heuristic_name


def factorial_boxplot(run_dir: Path, outdir: Path, **kwargs):

    # Read parameters
    dpi, show, results_dir, heuristic_name = unpack_data(**kwargs)

    # To extract factorial data
    precomputed_dir: Path = results_dir.parent.parent.parent.parent / "factorial_results" / "values"
    if not(precomputed_dir.exists()):
        raise FileNotFoundError("The directory results/factorial_results does not exist.")
    
    # Get lattice solved
    n_values = get_n_solved(results_dir)
    file_names = sorted(os.listdir(str(results_dir)))

    # To store stats
    box_data_weitzman: list[dict] = []
    box_data_heuristic: list[dict] = []
    real_weitzman_value: list[float] = []

    # Load and compute stats for each n-value
    for k, n in enumerate(n_values):

        # Make both paths
        precomputed_file_path = precomputed_dir / f"values_{n:03d}_points.npy"
        result_file_path = results_dir / file_names[k]#f"values_{n:03d}_points.npy"

        # Load npy file (HIGH RAM USAGE)
        weitzman_dist = np.load(precomputed_file_path)

        ## Compute stats
        min_value, q1, median, q3, max_value = np.percentile(weitzman_dist, [0, 25, 50, 75, 100]) 

        ## Add data to boxplot dictionaries
        box_data_weitzman.append(
            {'label': f"n={n}", 'whislo': min_value, 'q1': q1, 'med': median,
             'q3': q3, 'whishi': max_value})

        ## Save the real Weitzman value
        real_weitzman_value.append(max_value)

        # Load heuristic results
        heuristic_dist = np.load(result_file_path)

        ## Compute stats
        min_value, q1, median, q3, max_value = np.percentile(heuristic_dist, [0, 25, 50, 75, 100]) 

        ## Add data to boxplot dictionaries
        box_data_heuristic.append(
            {'label': f"n={n}", 'whislo': min_value, 'q1': q1, 'med': median,
             'q3': q3, 'whishi': max_value})

        # Release memory
        del weitzman_dist
        gc.collect()

    # Make figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ## Define positions
    n_experiments = len(box_data_weitzman)
    positions_A = [i - 0.15 for i in range(n_experiments)]
    positions_B = [i + 0.15 for i in range(n_experiments)]

    # Plot a trend line using the real Weitzman value
    x_positions = range(n_experiments)
    ax.plot(x_positions, real_weitzman_value, marker = "D", color="red", markersize="6", label="Weitzman Value")

    # Annotate text above the markers
    for x, y in zip(x_positions, real_weitzman_value):
        ax.annotate(f"{y:.2f}", (x,y), textcoords="offset points", xytext=(0,8), ha="center", fontsize=9)

    # Plot both groups
    weitzman_boxplots = ax.bxp(box_data_weitzman, positions=positions_A, widths=0.25, showfliers=False, patch_artist=True)
    heuristic_boxplots = ax.bxp(box_data_heuristic, positions=positions_B, widths=0.25, showfliers=False, patch_artist=True)

    # Add color to the boxplots
    for patch in weitzman_boxplots["boxes"]:
        patch.set_facecolor("#1f77b4")

    for patch in heuristic_boxplots["boxes"]:
        patch.set_facecolor("#ff7f0e")

    # Median lines
    for med in weitzman_boxplots['medians'] + heuristic_boxplots['medians']:
        med.set_color("black")
        med.set_linewidth(2)
        med.set_linestyle("--")

    # Legends
    patchA = mpatches.Patch(color="#1f77b4", label="Weitzman Dist")
    patchB = mpatches.Patch(color="#ff7f0e", label="Heuristic Dist")
    max_line = Line2D([0],[0], color="red", marker="D", markersize=6, linewidth=1.5, label="Weitzman Value")
    ax.legend(handles=[patchA, patchB, max_line])

    # Ticks format
    ax.set_xticks(range(n_experiments))
    ax.set_xticklabels([f"n={i}" for i in n_values])

    # Set title
    ax.set_title(f'Heuristic - {heuristic_name}')

    save_dir = outdir / f"plots"
    save_dir.mkdir(parents=False, exist_ok=True)
    save_path = save_dir / f"Boxplot_Factorial_Heuristic_{heuristic_name}.png"
    plt.savefig(save_path, dpi=dpi)

    if show:
        plt.show()

    plt.close()


def comparison_boxplot(run_dir: Path, outdir: Path, **kwargs):

    # Read parameters
    dpi, show, results_dir, heuristic_name = unpack_data(**kwargs)

    # Get lattice solved
    file_names = sorted(os.listdir(str(results_dir)))

    # To store data
    box_data: list[dict] = []
    max_values: list[float] = []
    degrees: list[float] = []

    for filename in file_names:

        # Make path
        result_file_path = results_dir / filename

        ## Get diversity degree
        degree = filename.rsplit("_", 1)[1].rsplit(".", 1)[0]
        degrees.append(degree)

        # Load npy file
        heuristic_dist = np.load(result_file_path)

        ## Compute stats
        min_val, q1, median, q3, max_val = np.percentile(heuristic_dist, [0, 25, 50, 75, 100]) 

       ## Add data to boxplot dictionaries
        box_data.append(
            {'label': f"deg={degree}", 'whislo': min_val, 'q1': q1, 'med': median,
             'q3': q3, 'whishi': max_val})

        ## Save the maximum value found
        max_values.append(max_val)

        # Release memory
        del heuristic_dist
        gc.collect()

    # Make figure
    fig, ax = plt.subplots(1, 1, figsize=(12,6))

    # Define positions
    n_experiments = len(box_data)

    # Plot a trend line using the max value
    x_positions = range(n_experiments)
    ax.plot(x_positions, max_values, marker = "D", color="red", markersize="6", label="Max Values")

    # Annotate text above the markers
    for x, y in zip(x_positions, max_values):
        ax.annotate(f"{y:.2f}", (x,y), textcoords="offset points", xytext=(0,8), ha="center", fontsize=9)

    # Plot
    heuristic_boxplots = ax.bxp(box_data, positions=x_positions, widths=0.25, showfliers=False, patch_artist=True)

    # Add color to the boxplots
    for patch in heuristic_boxplots["boxes"]:
        patch.set_facecolor("#1f77b4")

    # Median lines
    for med in heuristic_boxplots['medians']:
        med.set_color("black")
        med.set_linewidth(2)
        med.set_linestyle("--")

    # Legends
    patchA = mpatches.Patch(color="#1f77b4", label="Heuristic Dist")
    max_line = Line2D([0],[0], color="red", marker="D", markersize=6, linewidth=1.5, label="Max Value Found.")
    ax.legend(handles=[patchA, max_line])

    # Ticks format
    ax.set_xticks(range(n_experiments))
    ax.set_xticklabels([f"deg={i}" for i in degrees])

    # Set title
    print(filename)
    geometry, dimension, n_points = filename.split("_")[1:4]
    ax.set_title(f'{dimension} {geometry}, n={n_points} - {heuristic_name}')

    save_dir = outdir / f"plots"
    save_dir.mkdir(parents=False, exist_ok=True)
    save_path = save_dir / f"Boxplot_Self_Heuristic_{heuristic_name}.png"
    plt.savefig(save_path, dpi=dpi)

    if show:
        plt.show()

    plt.close()


def comparison_overlapped(run_dir: Path, outdir: Path, **kwargs):
    """Overlapped heuristics on the same geometry"""

    # Read parameters
    dpi, show, results_dir, heuristic_name = unpack_data(**kwargs)

    # Get heuristic names
    heuristics = [heuristic for heuristic in os.listdir(str(run_dir)) if heuristic not in ["logs", "config", "plots"]]
    n_heuristics = len(heuristics)

    # Make boxplot positions
    n_experiments = len(os.listdir(str(run_dir / heuristics[0] / "values")))
    centers = np.arange(n_experiments)
    offsets = np.linspace(-0.6/2, 0.6/2, n_heuristics)
    positions = centers[None, :] + offsets[:, None]
    print(positions)

    # Make figure
    fig, ax = plt.subplots(1, 1, figsize=(12,6))

    # To store data
    patches = []
    max_values = [0.0] * n_experiments
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]

    for k, heuristic in enumerate(heuristics):
        # Make the directory path
        results_dir = run_dir / heuristic / "values"

        # Get filenames
        file_names = sorted(os.listdir(str(results_dir)))

        # To store data
        box_data: list[dict] = []
        degrees: list[float] = []

        for i, filename in enumerate(file_names):
            # Make path
            result_file_path = results_dir / filename

            # Get diversity degree
            degree = filename.rsplit("_", 1)[1].rsplit(".", 1)[0]
            degrees.append(degree)

            # Load npy file
            heuristic_dist = np.load(result_file_path)

            ## Compute stats
            min_val, q1, median, q3, max_val = np.percentile(heuristic_dist, [0, 25, 50, 75, 100]) 

            ## Add data to boxplot dictionaries
            box_data.append(
                {'label': f"deg={degree}", 'whislo': min_val, 'q1': q1, 'med': median,
                 'q3': q3, 'whishi': max_val})

            # Update maximum values
            if max_val > max_values[i]:
                max_values[i] = max_val

        # Plot boxplots
        heuristic_boxplots = ax.bxp(box_data, positions=positions[k,:], widths=0.25, showfliers=False, patch_artist=True)

        # Add color to the boxplots
        for patch in heuristic_boxplots["boxes"]:
            patch.set_facecolor(colors[k])

        # Median lines
        for med in heuristic_boxplots['medians']:
            med.set_color("black")
            med.set_linewidth(2)
            med.set_linestyle("--")

        # Legends
        patchZ = mpatches.Patch(color=colors[k], label=f"{heuristic} Dist")
        patches.append(patchZ)

    # Plot a trend line using the max value
    ax.plot(centers, max_values, marker="D", color="red", markersize="6", label="Max Values")

    # Annotate text above the markers
    for x, y in zip(centers, max_values):
        ax.annotate(f"{y:.2f}", (x,y), textcoords="offset points", xytext=(0,8), ha="center", fontsize=9)

    # Plot an horizontal line at max value height
    for i in range(positions.shape[1]):
        ax.plot([positions[0,i], positions[-1,i]], [max_values[i], max_values[i]], "k--")

    # Legends
    max_line = Line2D([0],[0], color="red", marker="D", markersize=6, linewidth=1.5, label="Max Value Found.")
    ax.legend(handles=[*patches, max_line])

    # Ticks format
    ax.set_xticks(centers)
    ax.set_xticklabels([f"deg={i}" for i in degrees])

    # Set title
    geometry, dimension, n_points = filename.split("_")[1:4]
    ax.set_title(f'{dimension} {geometry}, n={n_points}')

    save_dir = outdir / f"plots"
    save_dir.mkdir(parents=False, exist_ok=True)
    save_path = save_dir / f"Boxplot_Overlapped_Heuristics.png"
    print(save_path)
    plt.savefig(save_path, dpi=dpi)

    if show:
        plt.show()

    plt.close()


def run_boxplot(run_dir: Path, outdir: Path, **kwargs):

    print(run_dir, outdir)

    # First try the factorial plot
    try:
        factorial_boxplot(run_dir, outdir, **kwargs)
    except Exception as e:
        logger.info("Factorial boxplot failed.")
        logger.debug(e)

    # Try the self comparison
    try:
        comparison_boxplot(run_dir, outdir, **kwargs)
    except Exception as e:
        logger.info("Self comparison boxplot failed.")
        logger.debug(e)

    # Try comparison_overlap
    try:
        comparison_overlapped(run_dir, outdir, **kwargs)
    except Exception as e:
        logger.info("Comparison overlapped boxplot failed.")
        logger.debug(e)

