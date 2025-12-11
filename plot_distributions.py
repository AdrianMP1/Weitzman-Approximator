"""
Makes a figure with boxplots that contain the distribution values of the Weitzman tree against the performance of some heuristic.
"""

import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib.lines import Line2D


def main():
    
    
    # To store stats
    box_data_weitzman: list[dict] = []
    box_data_heuristic: list[dict] = []
    real_weitzman_value: list[float] = []

    # Load and compute stats for each n-value
    for n in range(4, 12 + 1):
        
        # Make both paths
        npy_path = f"Weitzman_factorial_results/values/values_{n:03d}_points.npy"
        heuristic_results_path = f"Weitzman_heuristic_results/farthest_neighbour/values_{n:03d}_points.npy"
    
        # Load npy file (HIGH RAM USAGE)
        weitzman_dist = np.load(npy_path)

        ## Compute stats
        min_value, q1, median, q3, max_value = np.percentile(weitzman_dist, [0, 25, 50, 75, 100]) 
        
        ## Add data to boxplot dictionaries
        box_data_weitzman.append(
            {'label': f"n={n}", 'whislo': min_value, 'q1': q1, 'med': median,
             'q3': q3, 'whishi': max_value})

        ## Save the real Weitzman value
        real_weitzman_value.append(max_value)
        
        # Load heuristic results
        heuristic_dist = np.load(heuristic_results_path)

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
    ax.set_xticklabels([f"n={4 + i}" for i in range(n_experiments)])

    # Set title
    ax.set_title('Heuristic - X')

    plt.savefig("Heuristic_X.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
