"""
Publisher-quality trendline figures for the Weitzman diversity paper.

Each figure shows how each heuristic's best Weitzman approximation changes with
the coverage/uniformity degree (lattice_deg) for a fixed (kind, geom, card) cell.
PD is plotted as a reference line; W(A) is shown when n is small enough for the
brute-force ground truth to have been computed.

Usage (standalone):
    python weitzman/plotting/trendlines.py --data results/data.csv
                                           --outdir results/figures

Usage via main.py:
    python main.py trendline [--kind coverage] [--geom Linear] [--card 21]
"""

import shutil
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layout constants (IEEE/Springer-style single-column)
# ---------------------------------------------------------------------------

_MM_TO_IN          = 1 / 25.4
_PT_TO_MM          = 0.35146
_TEXTWIDTH_MM      = 347.12354 * _PT_TO_MM   # full text width of the paper
_ASPECT            = 21 / 9


def _fig_size(width_mm: float = _TEXTWIDTH_MM, aspect: float = _ASPECT):
    w = width_mm * _MM_TO_IN
    return w, w / aspect


# ---------------------------------------------------------------------------
# Algorithm display names and paper subset
# ---------------------------------------------------------------------------

DISPLAY_NAMES: dict[str, str] = {
    "farthest_neighbour": "FN",
    "twice_around":       "TAT",
    "christofides":       "CHR",
    "global_max_min":     "GMM",
}

# Algorithms shown in the paper
PAPER_ALGORITHMS: list[str] = [
    "farthest_neighbour",
    "twice_around",
    "christofides",
    "global_max_min",
]

# ---------------------------------------------------------------------------
# Line styles and grayscale palette
# ---------------------------------------------------------------------------

_STYLES: list[tuple] = [
    ((0, (2, 2)),              "o"),   # FN:  short-dashed
    ((0, (3, 1, 1, 1)),        "s"),   # TAT: dense dash-dot
    ((0, (5, 1, 1, 1, 1, 1)), "^"),   # CHR: dash-dot-dot
    ((0, (1, 2)),              "D"),   # GMM: dotted
    ((0, (8, 2, 1, 2)),        "v"),   # PD:  long-dash + dot
    ("-",                      "*"),   # W(A): solid
]

_GRAY_PALETTE: list[str] = [str(x) for x in np.linspace(0, 0.6, 6)][::-1]


# ---------------------------------------------------------------------------
# Publisher rcParams
# ---------------------------------------------------------------------------

def _set_publisher_rcparams() -> None:
    mpl.rcParams.update({
        # --- Fonts ---
        "font.family":    "serif",
        "font.size":      8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,

        # --- Lines ---
        "lines.linewidth": 1.0,
        "lines.markersize": 4.0,
        "axes.linewidth":  0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # --- Figure size ---
        "figure.dpi":      100,
        "savefig.dpi":     300,

        # --- PDF / SVG ---
        "pdf.fonttype":    42,
        "ps.fonttype":     42,
        "svg.fonttype":    "none",

        # --- LaTeX Render ---
        **(
            {
                "text.usetex":   True,
                "pgf.texsystem": "pdflatex",
                "pgf.rcfonts":   False,
                "pgf.preamble":  r"\providecommand{\mathdefault}[1]{#1}",
            }
            if shutil.which("pdflatex") is not None
            else {}
        ),

        # --- Colors ---
        "image.cmap":      "gray",
        "axes.prop_cycle": mpl.cycler(color=["0.0", "0.35", "0.6"]),

        # --- Layout to avoid clipped labels ---
        "figure.constrained_layout.use": True,
    })
    if shutil.which("pdflatex") is None:
        logger.warning(
            "pdflatex not found — falling back to matplotlib mathtext. "
            "Install a LaTeX distribution for publisher-quality output."
        )


# ---------------------------------------------------------------------------
# Core plotting function
# ---------------------------------------------------------------------------

def plot_case(
    kind: str,
    geom: str,
    card: int,
    df: pd.DataFrame,
    algorithms: list[str] | None = None,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Plot one (kind, geom, card) trendline figure.

    Parameters
    ----------
    kind       : "coverage" or "uniformity"
    geom       : "Concave", "Convex", or "Linear"
    card       : number of Pareto front points (e.g. 15, 21, …)
    df         : full data.csv loaded as a DataFrame
    algorithms : algorithms to include; defaults to PAPER_ALGORITHMS
    """
    if algorithms is None:
        algorithms = PAPER_ALGORITHMS

    data = df[(df["kind"] == kind) & (df["geom"] == geom) & (df["card"] == card)]
    data = data[data["algorithm"].isin(algorithms)]

    degrees = sorted(data["lattice_deg"].unique().tolist(), reverse=True)
    if kind == "Uniformity":
        # Due to the Uniformity Generator, the maximum degree value is consider the ground-truth
        # Map it into 1.0 (e.g, 0.7 is maximum, then it is relabed to 1.0)
        temp_cte = 1.0 - degrees[0]
        map_degrees = {deg: round(deg + temp_cte, 2) for deg in degrees}
    else:
        map_degrees = {deg: deg for deg in degrees}

    xticks = range(len(degrees))
    lw = 1.25

    # Collect per-algorithm max values and reference lines
    alg_vals: dict[str, list[float]] = defaultdict(list)
    w_vals:   list[float] = []
    pd_vals:  list[float] = []

    for deg in degrees:
        subset = data[data["lattice_deg"] == deg]
        for alg in algorithms:
            row = subset[subset["algorithm"] == alg]
            alg_vals[alg].append(float(row["max"].iloc[0]) if len(row) else float("nan"))
        ref = subset.iloc[0] if len(subset) else None
        w_vals.append( float(ref["W-value"]) if ref is not None else float("nan"))
        pd_vals.append(float(ref["PD"])      if ref is not None else float("nan"))

    fig, ax = plt.subplots(figsize=_fig_size())

    for i, alg in enumerate(algorithms):
        ax.plot(xticks, alg_vals[alg],
                color=_GRAY_PALETTE[i],
                lw=lw, ls=_STYLES[i][0], marker=_STYLES[i][1], ms=4,
                label=DISPLAY_NAMES.get(alg, alg))

    n_algs = len(algorithms)
    ax.plot(xticks, pd_vals,
            color=_GRAY_PALETTE[n_algs],
            lw=lw, ls=_STYLES[n_algs][0], marker=_STYLES[n_algs][1], ms=4,
            label="PD")

    if card <= 36:
        ax.plot(xticks, w_vals,
                color=_GRAY_PALETTE[n_algs + 1],
                lw=lw, ls=_STYLES[n_algs + 1][0], marker=_STYLES[n_algs + 1][1], ms=6,
                label=r"$\mathcal{W}(\mathcal{A})$")

    ax.set_xlabel(f"{kind.capitalize()} Loss")
    ax.set_ylabel(r"$\widetilde{\mathcal{W}}(\mathcal{A}, \rho)$")
    ax.set_xticks(list(xticks))
    ax.set_xticklabels([str(map_degrees.get(d, d)) for d in degrees])
    ax.grid(axis="y", ls="-", lw=0.5, color="0.6")
    ax.legend(
        loc="center left", bbox_to_anchor=(0.987, 0.5),
        handlelength=2.5, markerscale=0.9, numpoints=1, frameon=False,
    )

    return fig, ax


# ---------------------------------------------------------------------------
# Batch export
# ---------------------------------------------------------------------------

def plot_all(
    data_csv: Path,
    outdir: Path,
    algorithms: list[str] | None = None,
    fmt: tuple[str, ...] = ("pdf", "png"),
) -> None:
    """Generate one trendline figure per (kind, geom, card) cell."""
    _set_publisher_rcparams()
    df    = pd.read_csv(data_csv)
    cells = df[["kind", "geom", "card"]].drop_duplicates()
    outdir.mkdir(parents=True, exist_ok=True)

    for _, row in cells.iterrows():
        kind, geom, card = row["kind"], row["geom"], int(row["card"])
        fig, _ = plot_case(kind, geom, card, df, algorithms)
        cell_dir = outdir / kind / geom
        cell_dir.mkdir(parents=True, exist_ok=True)
        stem = f"trendline_{kind}_{geom.lower()}_{card}"
        for ext in fmt:
            dpi = 1200 if ext == "png" else 400
            fig.savefig(cell_dir / f"{stem}.{ext}", dpi=dpi)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate trendline figures from aggregated data.csv."
    )
    parser.add_argument("--data",   type=Path, default=Path("results/data.csv"),
                        help="Aggregated CSV produced by 'python main.py aggregate'.")
    parser.add_argument("--outdir", type=Path, default=Path("results/figures"),
                        help="Output directory for figures.")
    parser.add_argument("--kind",   default=None,
                        help="Restrict to one kind ('coverage' or 'uniformity').")
    parser.add_argument("--geom",   default=None,
                        help="Restrict to one geometry ('Concave', 'Convex', 'Linear').")
    parser.add_argument("--card",   type=int, default=None,
                        help="Restrict to one cardinality (number of points).")
    args = parser.parse_args(argv)

    _set_publisher_rcparams()
    df = pd.read_csv(args.data)

    if args.kind:
        df = df[df["kind"] == args.kind]
    if args.geom:
        df = df[df["geom"] == args.geom]
    if args.card:
        df = df[df["card"] == args.card]

    cells = df[["kind", "geom", "card"]].drop_duplicates()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for _, row in cells.iterrows():
        kind, geom, card = row["kind"], row["geom"], int(row["card"])
        fig, _ = plot_case(kind, geom, card, df)
        cell_dir = args.outdir / kind / geom
        cell_dir.mkdir(parents=True, exist_ok=True)
        stem = f"trendline_{kind}_{geom.lower()}_{card}"
        fig.savefig(cell_dir / f"{stem}.pdf", dpi=400)
        fig.savefig(cell_dir / f"{stem}.png", dpi=1200)
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
