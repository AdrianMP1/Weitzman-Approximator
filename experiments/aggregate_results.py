"""
Aggregate batch results into a single data.csv.

Scans results/batch/, computes per-row statistics (min, q1, median, q3, max),
joins the exact W(A) value from brute-force factorial results (when available),
and computes the PD metric from the original .POF files.

Usage (standalone):
    python experiments/aggregate_results.py [--batch-dir results/batch]
                                            [--data-dir data]
                                            [--out results/data.csv]

Usage via main.py:
    python main.py aggregate
"""

import json
import logging
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path

from weitzman.io.loaders import load_lattice
from weitzman.metrics.pure_diversity import PD
from weitzman.utils.logging_utils import configure_logging, get_logger

logger = logging.getLogger(__name__)

_KIND_MAP = {"CovLoss": "Coverage", "UnifLoss": "Uniformity"}


def discover_cells(data_root: Path) -> dict[tuple[str, str, str], Path]:
    """Return {(kind, geom, card_name): leaf_path} for every data cell."""
    cells: dict[tuple[str, str, str], Path] = {}
    for kind_dir in sorted(data_root.iterdir()):
        if not kind_dir.is_dir():
            continue
        kind = next((v for k, v in _KIND_MAP.items() if k in kind_dir.name), None)
        if kind is None:
            continue
        for geom_dir in sorted(kind_dir.iterdir()):
            if not geom_dir.is_dir():
                continue
            geom = geom_dir.name.rsplit("_", 1)[-1]
            for card_dir in sorted(geom_dir.iterdir(), key=lambda d: int(d.name.split("p")[1]) if "p" in d.name else 0):
                if card_dir.is_dir():
                    cells[(kind, geom, card_dir.name)] = card_dir
    return cells


def _parse_cell_name(name: str) -> tuple[str, str, str]:
    """'Coverage_Concave_m3_p4' -> ('Coverage', 'Concave', 'm3_p4')"""
    parts = name.split("_")
    kind = parts[0]
    geom = parts[1]
    card = "_".join(parts[2:])
    return kind, geom, card


def _load_timing(alg_dir: Path) -> dict[str, float]:
    """Return {pof_stem: elapsed_seconds} from timing.json, or empty dict."""
    p = alg_dir / "timing.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _load_exact_w_values(exact_dir: Path) -> dict[str, float]:
    """
    Return {pof_stem: W(A)} from BB exact solver results.

    Files are named values_<pof_stem>.npy (shape (1,)), one per instance,
    mirroring the heuristic values naming convention.
    """
    w: dict[str, float] = {}
    val_dir = exact_dir / "values"
    if not val_dir.exists():
        return w
    for f in val_dir.glob("values_*.npy"):
        # stem: "values_Linear_3D_015_0.50"  ->  pof_stem: "Linear_3D_015_0.50"
        pof_stem = f.stem[len("values_"):]
        try:
            w[pof_stem] = float(np.load(f)[0])
        except Exception:
            logger.warning("Cannot load exact W from %s", f.name)
    return w


def aggregate(batch_dir: Path, data_dir: Path) -> pd.DataFrame:
    cells_map  = discover_cells(data_dir)
    cell_dirs  = sorted(d for d in batch_dir.iterdir() if d.is_dir())
    rows: list[dict] = []

    print(f"\nAggregating {len(cell_dirs)} cells from {batch_dir} ...", flush=True)

    for cell_dir in tqdm(cell_dirs, desc="Aggregate", unit="cell"):
        try:
            kind, geom, card_name = _parse_cell_name(cell_dir.name)
        except Exception:
            logger.debug("Skipping non-cell dir: %s", cell_dir.name)
            continue

        data_path = cells_map.get((kind, geom, card_name))
        exact_w   = _load_exact_w_values(cell_dir / "exact")

        for alg_dir in sorted(cell_dir.iterdir()):
            if not alg_dir.is_dir() or alg_dir.name in ("factorial", "exact"):
                continue
            alg_name   = alg_dir.name
            values_dir = alg_dir / "values"
            if not values_dir.exists():
                continue
            alg_timing = _load_timing(alg_dir)

            for npy_file in sorted(values_dir.glob("values_*.npy")):
                # npy stem: "values_Linear_3D_015_0.50"
                # strip "values_" prefix to recover the original POF stem
                pof_stem = npy_file.stem[len("values_"):]   # "Linear_3D_015_0.50"
                parts = pof_stem.split("_")
                try:
                    card_n      = int(parts[-2])       # "015" -> 15
                    lattice_deg = float(parts[-1])     # "0.50" -> 0.5
                except (ValueError, IndexError):
                    logger.warning("Cannot parse card/deg from %s", npy_file.name)
                    continue

                values = np.load(npy_file)
                lo, q1, med, q3, hi = np.percentile(values, [0, 25, 50, 75, 100])

                w_val  = exact_w.get(pof_stem, float("nan"))

                pd_val = float("nan")
                if data_path is not None:
                    orig_path = data_path / (pof_stem + ".POF")
                    if orig_path.exists():
                        try:
                            pts, _ = load_lattice(str(orig_path), labeled=False)
                            pd_val = PD(pts)
                        except Exception as exc:
                            logger.warning("PD failed for %s: %s", orig_path.name, exc)

                rows.append({
                    "kind":        kind,
                    "geom":        geom,
                    "card":        card_n,
                    "lattice_deg": lattice_deg,
                    "algorithm":   alg_name,
                    "min":         lo,
                    "q1":          q1,
                    "median":      med,
                    "q3":          q3,
                    "max":         hi,
                    "W-value":     w_val,
                    "PD":          pd_val,
                    "time_s":      alg_timing.get(pof_stem, float("nan")),
                })
                logger.info("  %-12s | %-8s | %-6s | %-20s | deg=%.2f | max=%.4f",
                            kind, geom, card_name, alg_name, lattice_deg, hi)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["kind", "geom", "card", "algorithm", "lattice_deg"],
            ignore_index=True,
        )
    return df


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate batch results into data.csv."
    )
    parser.add_argument("--batch-dir", type=Path, default=Path("results/batch"),
                        help="Root directory produced by 'python main.py batch'.")
    parser.add_argument("--data-dir",  type=Path, default=Path("data"),
                        help="Root data directory (needed to locate original .POF files).")
    parser.add_argument("--out",       type=Path, default=Path("results/data.csv"),
                        help="Output CSV path.")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    configure_logging(verbosity=args.verbose)
    log = get_logger(__name__)
    log.info("Aggregating from %s", args.batch_dir)

    df = aggregate(args.batch_dir, args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log.info("Wrote %d rows to %s", len(df), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
