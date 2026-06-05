"""
Compute Kendall \\tau rank-correlation for each (kind, geom, card, algorithm).

For each group the values at degrees [0.6, 0.7, 0.8, 0.9, 1.0] are expected to be
monotonically increasing.  Kendall \\tau = 1 means perfect monotone increase; any
deviation from 1 is flagged in the paper table.

PD is treated as an additional "algorithm" column, deduplicated per
(kind, geom, card, lattice_deg) before computing τ.

Outputs
-------
results/kendall_tau_<kind>_pivot.csv - pivoted table matching the paper layout,
                                       one file per kind value found in the data.

Usage (standalone):
    python experiments/compute_kendall_tau.py [--data results/data.csv]

Usage via main.py:
    python main.py kendall [--data results/data.csv]
"""

import argparse
import pandas as pd

from pathlib import Path
from scipy.stats import kendalltau

# Column order that matches the paper table
_ALG_ORDER = ["PD", "farthest_neighbour", "twice_around", "christofides", "global_max_min"]
_DISPLAY   = {
    "farthest_neighbour": "FN",
    "twice_around":       "TAT",
    "christofides":       "CHR",
    "global_max_min":     "GMM",
    "PD":                 "PD",
}
_GEOM_ORDER = ["Linear", "Concave", "Convex"]


def _tau(series: pd.Series) -> float:
    """Kendall \\tau between index position (ascending) and values, rounded to 1 dp."""
    vals = series.sort_index().values
    tau, _ = kendalltau(range(len(vals)), vals)
    return round(float(tau), 1)


def _long(df: pd.DataFrame) -> pd.DataFrame:
    """Build intermediate long-form \\tau values (not written to disk)."""
    rows = []

    # Heuristic algorithms: use the `max` column per (kind, geom, card, lattice_deg)
    for (kind, geom, card, alg), grp in df.groupby(["kind", "geom", "card", "algorithm"]):
        pivot = grp.set_index("lattice_deg")["max"]
        rows.append({
            "kind":        kind,
            "geom":        geom,
            "card":        int(card),
            "algorithm":   alg,
            "kendall_tau": _tau(pivot),
        })

    # PD: deduplicate then compute τ
    pd_df = df[["kind", "geom", "card", "lattice_deg", "PD"]].drop_duplicates(
        subset=["kind", "geom", "card", "lattice_deg"]
    )
    for (kind, geom, card), grp in pd_df.groupby(["kind", "geom", "card"]):
        pivot = grp.set_index("lattice_deg")["PD"]
        rows.append({
            "kind":        kind,
            "geom":        geom,
            "card":        int(card),
            "algorithm":   "PD",
            "kendall_tau": _tau(pivot),
        })

    return pd.DataFrame(rows).sort_values(
        ["kind", "geom", "card", "algorithm"], ignore_index=True
    )


def pivot_table(long_df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    Produce a pivoted table for one kind matching the paper layout:

        card | (Linear, PD) | (Linear, FN) | ... | (Convex, GMM)
    """
    sub = long_df[long_df["kind"] == kind].copy()

    # Keep only algorithms that appear in the paper order
    sub = sub[sub["algorithm"].isin(_ALG_ORDER)]

    # Rename algorithms to display names
    sub["algorithm"] = sub["algorithm"].map(_DISPLAY)
    alg_display_order = [_DISPLAY[a] for a in _ALG_ORDER if _DISPLAY[a] in sub["algorithm"].unique()]

    pivoted = sub.pivot_table(
        index="card",
        columns=["geom", "algorithm"],
        values="kendall_tau",
    )

    # Reorder columns to (geom_order × alg_order)
    col_order = [
        (g, a)
        for g in _GEOM_ORDER if g in pivoted.columns.get_level_values("geom")
        for a in alg_display_order if (g, a) in pivoted.columns
    ]
    pivoted = pivoted[col_order]
    pivoted.index.name = "Set size"
    return pivoted


def run(data_csv: Path, outdir: Path) -> None:
    """Compute and save pivoted Kendall \\tau tables for all kinds in data_csv."""
    df   = pd.read_csv(data_csv)
    long = _long(df)
    outdir.mkdir(parents=True, exist_ok=True)

    for kind in sorted(long["kind"].unique()):
        out_path = outdir / f"kendall_tau_{kind.lower()}_pivot.csv"
        pivot_table(long, kind).to_csv(out_path)
        print(f"Saved Kendall \\tau ({kind}) -> {out_path}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute Kendall \\tau coefficients from aggregated data.csv."
    )
    parser.add_argument("--data",   type=Path, default=Path("results/data.csv"),
                        help="Aggregated CSV produced by 'python main.py aggregate'.")
    parser.add_argument("--outdir", type=Path, default=Path("results"),
                        help="Output directory for pivot CSVs.")
    args = parser.parse_args(argv)

    run(args.data, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
