import os
import gc
import logging
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from typing import Any
from pathlib import Path

from weitzman.io.loaders import load_lattice
from weitzman.metrics.pure_diversity import PD

logger = logging.getLogger(__name__)

COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

# Canonical display order for algorithms.
_DISPLAY_ORDER = [
    "farthest_neighbour", "twice_around", "christofides",
    "global_max_min",
]


def _load_stats(file_path: Path) -> dict:
    """Load a .npy values file and return a bxp-compatible stats dict."""
    data = np.load(file_path)
    lo, q1, med, q3, hi = np.percentile(data, [0, 25, 50, 75, 100])
    del data
    gc.collect()
    return {"whislo": lo, "q1": q1, "med": med, "q3": q3, "whishi": hi}


def _sorted_value_files(values_dir: Path) -> list[Path]:
    return sorted(values_dir.glob("values_*.npy"))


# ---------------------------------------------------------------------------
# Plot 1: distribution of all n! Weitzman values (brute-force only)
# ---------------------------------------------------------------------------

def factorial_distribution_boxplot(
    run_dir: Path,
    outdir: Path,
    plot_params: dict[str, Any],
    factorial_dir: Path,
) -> None:
    """
    Boxplot of the full Weitzman value distribution across all n! permutations
    for each n, with the true maximum (W(A)) annotated.
    """
    values_dir = factorial_dir / "values"
    files = _sorted_value_files(values_dir)
    if not files:
        raise FileNotFoundError(f"No value files found in {values_dir}")

    box_data, max_vals, min_vals, n_labels = [], [], [], []

    for f in files:
        n = int(f.stem.split("_")[1])   # "values_007_points" -> 7
        stats = _load_stats(f)
        stats["label"] = f"n={n}"
        box_data.append(stats)
        max_vals.append(stats["whishi"])
        min_vals.append(stats["whislo"])
        n_labels.append(n)

    fig, ax = plt.subplots(figsize=(12, 6))
    xs = range(len(box_data))

    bxp = ax.bxp(box_data, positions=list(xs), widths=0.25,
                 showfliers=False, patch_artist=True)
    for patch in bxp["boxes"]:
        patch.set_facecolor(COLORS[0])
    for med in bxp["medians"]:
        med.set(color="black", linewidth=2, linestyle="--")

    ax.plot(xs, max_vals, marker="D", color="red", markersize=6, label="W(A) (true maximum)")
    for x, y in zip(xs, max_vals):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9)
    for x, y in zip(xs, min_vals):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -10),
                    ha="center", fontsize=9)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"n={n}" for n in n_labels])
    ax.grid(axis="y", ls="--", lw=0.75, alpha=0.7)
    ax.set_title("Weitzman Value Distribution (all n! permutations)")
    ax.legend(handles=[
        mpatches.Patch(color=COLORS[0], label="Value distribution"),
        Line2D([0], [0], color="red", marker="D", markersize=6, label="W(A)"),
    ])

    save_dir = outdir / "plots"
    save_dir.mkdir(exist_ok=True)
    fig.savefig(save_dir / "boxplot_factorial_distribution.png", dpi=plot_params["dpi"])
    if plot_params.get("show"):
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: heuristics vs. factorial distribution
# ---------------------------------------------------------------------------

def heuristics_vs_factorial_boxplot(
    run_dir: Path,
    outdir: Path,
    heuristics: list[str],
    plot_params: dict[str, Any],
    factorial_dir: Path,
) -> None:
    """
    Side-by-side boxplots: factorial distribution + each heuristic's
    value distribution, grouped by n.  The true W(A) is overlaid.
    """
    factorial_values_dir = factorial_dir / "values"
    ordered = [h for h in _DISPLAY_ORDER if h in heuristics]
    n_h = len(ordered)

    # Determine n values from the first heuristic's output files.
    ref_files = _sorted_value_files(run_dir / ordered[0] / "values")
    n_experiments = len(ref_files)

    if any(int(f.stem.split("_")[1]) > 12 for f in
           _sorted_value_files(factorial_values_dir)):
        raise ValueError("Factorial data contains n > 12; plot outside scope.")

    # Positions: groups separated by `spacing`, boxes within each group
    # spread across `group_width`.
    spacing = 1.8
    group_width = 1.2
    centers = np.arange(n_experiments) * spacing
    offsets = np.linspace(-group_width / 2, group_width / 2, n_h + 1)
    positions = centers[None, :] + offsets[:, None]

    fig, ax = plt.subplots(figsize=(12, 6))
    patches = []
    max_vals = [0.0] * n_experiments

    # --- Factorial distribution (column 0 in each group) ---
    fact_files = _sorted_value_files(factorial_values_dir)
    fact_box, n_labels = [], []
    for f in fact_files:
        n = int(f.stem.split("_")[1])
        stats = _load_stats(f)
        stats["label"] = f"n={n}"
        fact_box.append(stats)
        max_vals[len(fact_box) - 1] = stats["whishi"]
        n_labels.append(n)

    bxp = ax.bxp(fact_box, positions=positions[0], widths=0.25,
                 showfliers=False, patch_artist=True)
    for patch in bxp["boxes"]:
        patch.set_facecolor(COLORS[0])
    for med in bxp["medians"]:
        med.set(color="black", linewidth=2)
    patches.append(mpatches.Patch(color=COLORS[0], label="n! distribution"))

    # --- One heuristic per remaining column ---
    for k, h_name in enumerate(ordered):
        h_files = _sorted_value_files(run_dir / h_name / "values")
        h_box = []
        for f in h_files:
            stats = _load_stats(f)
            stats["label"] = f.stem
            h_box.append(stats)

        bxp = ax.bxp(h_box, positions=positions[k + 1], widths=0.25,
                     showfliers=False, patch_artist=True)
        for patch in bxp["boxes"]:
            patch.set_facecolor(COLORS[(k + 1) % len(COLORS)])
        for med in bxp["medians"]:
            med.set(color="black", linewidth=2)
        patches.append(mpatches.Patch(color=COLORS[(k + 1) % len(COLORS)], label=h_name))

    ax.plot(centers, max_vals, linestyle="-.", marker="D", color=COLORS[0],
            markersize=6, zorder=10, label="W(A)")
    for x, y in zip(centers, max_vals):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9)
    for i in range(n_experiments):
        ax.plot([positions[0, i], positions[-1, i]],
                [max_vals[i], max_vals[i]], "k--", lw=0.8)

    ax.set_xticks(centers)
    ax.set_xticklabels([f"n={n}" for n in n_labels])
    ax.grid(axis="y", ls="--", lw=0.75, alpha=0.7)
    ax.set_title("Heuristic Performance vs. Full Search Space")
    ax.legend(handles=[
        Line2D([0], [0], linestyle="-.", color=COLORS[0], marker="D",
               markersize=6, label="W(A)"),
        *patches,
    ])

    save_dir = outdir / "plots"
    save_dir.mkdir(exist_ok=True)
    fig.savefig(save_dir / "boxplot_heuristics_vs_factorial.png", dpi=plot_params["dpi"])
    if plot_params.get("show"):
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: all heuristics overlapped with PD reference line
# ---------------------------------------------------------------------------

def heuristics_comparison_boxplot(
    run_dir: Path,
    outdir: Path,
    heuristics: list[str],
    plot_params: dict[str, Any],
    lattice_info: dict[str, Any],
) -> None:
    """
    Overlapped boxplots for all heuristics on the same set of instances,
    with the best observed value and the PD metric as reference lines.
    """
    ordered = [h for h in _DISPLAY_ORDER if h in heuristics]
    n_h = len(ordered)

    ref_files = _sorted_value_files(run_dir / ordered[0] / "values")
    n_experiments = len(ref_files)

    centers = np.arange(n_experiments, dtype=float)
    offsets = np.linspace(-0.3, 0.3, n_h)
    positions = centers[None, :] + offsets[:, None]

    fig, ax = plt.subplots(figsize=(12, 6))
    patches = []
    max_vals = [0.0] * n_experiments

    # Compute PD for every lattice instance (used as a diversity reference).
    lattice_dir = lattice_info["dir"]
    lattice_ext = lattice_info["ext"]
    lattice_labeled = lattice_info["labeled"]
    pd_values = []
    for lf in sorted(f for f in os.listdir(lattice_dir) if f.endswith(lattice_ext)):
        pts, _ = load_lattice(str(lattice_dir / lf), labeled=lattice_labeled)
        pd_values.append(PD(pts))

    x_labels = []
    for k, h_name in enumerate(ordered):
        h_files = _sorted_value_files(run_dir / h_name / "values")
        h_box = []
        for i, f in enumerate(h_files):
            stats = _load_stats(f)
            stats["label"] = f.stem
            h_box.append(stats)
            if stats["whishi"] > max_vals[i]:
                max_vals[i] = stats["whishi"]
            if k == 0:
                x_labels.append(f.stem)

        bxp = ax.bxp(h_box, positions=positions[k], widths=0.12,
                     showfliers=False, patch_artist=True)
        for patch in bxp["boxes"]:
            patch.set_facecolor(COLORS[k % len(COLORS)])
        for med in bxp["medians"]:
            med.set(color="black", linewidth=2, linestyle="--")
        patches.append(mpatches.Patch(color=COLORS[k % len(COLORS)], label=h_name))

    ax.plot(centers, max_vals, marker="D", color="red", markersize=6, label="Best found")
    ax.plot(centers, pd_values, marker="s", color=COLORS[n_h % len(COLORS)],
            markersize=6, label="PD value")

    for x, y in zip(centers, max_vals):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9)
    for x, y in zip(centers, pd_values):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -10),
                    ha="center", fontsize=9)

    for i in range(n_experiments):
        ax.plot([positions[0, i], positions[-1, i]],
                [max_vals[i], max_vals[i]], "k--", lw=0.8)

    ax.set_xticks(centers)
    ax.set_xticklabels(x_labels, rotation=15, ha="right", fontsize=8)
    ax.grid(axis="y", ls="--", lw=0.75, alpha=0.7)
    ax.set_title("Heuristic Comparison")
    ax.legend(handles=[
        *patches,
        Line2D([0], [0], color="red", marker="D", markersize=6, label="Best found"),
        Line2D([0], [0], color=COLORS[n_h % len(COLORS)], marker="s",
               markersize=6, label="PD value"),
    ])

    save_dir = outdir / "plots"
    save_dir.mkdir(exist_ok=True)
    fig.savefig(save_dir / "boxplot_heuristics_comparison.png", dpi=plot_params["dpi"])
    if plot_params.get("show"):
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_boxplot(
    run_dir: Path,
    outdir: Path,
    heuristics: list[str],
    plot_params: dict[str, Any],
    lattice_info: dict[str, Any],
    factorial_dir: Path | None = None,
) -> None:
    """
    Run whichever boxplot variants are applicable given the available data.

    - Single heuristic + factorial data  -> factorial distribution plot.
    - Multiple heuristics + factorial    -> heuristics vs. factorial.
    - Multiple heuristics, no factorial  -> heuristics comparison + PD.
    """
    n_h = len(heuristics)

    if factorial_dir is not None and factorial_dir.exists():
        try:
            factorial_distribution_boxplot(run_dir, outdir, plot_params, factorial_dir)
        except Exception as e:
            logger.warning("factorial_distribution_boxplot failed: %s", e)

        if n_h > 1:
            try:
                heuristics_vs_factorial_boxplot(
                    run_dir, outdir, heuristics, plot_params, factorial_dir
                )
            except Exception as e:
                logger.warning("heuristics_vs_factorial_boxplot failed: %s", e)

    if n_h > 1:
        try:
            heuristics_comparison_boxplot(
                run_dir, outdir, heuristics, plot_params, lattice_info
            )
        except Exception as e:
            logger.warning("heuristics_comparison_boxplot failed: %s", e)
