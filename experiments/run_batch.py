"""
Batch runner: execute heuristics + brute force + exact on every data cell, then aggregate.

Discovers all (kind, geom, card) leaf directories under data/, runs each cell,
writes a consolidated results/data.csv, figures, and Kendall Tau values at the end.
Cells whose output already exists are skipped automatically unless --force is given.

Usage (standalone):
    python experiments/run_batch.py --config experiment.yaml [-v] [--force]

Usage via main.py:
    python main.py batch --config experiment.yaml [-v] [--force]
"""

import time
import math
import copy
import logging
import argparse

from pathlib import Path

from weitzman.io.loaders import load_config_file, load_lattice
from weitzman.utils.operations import compute_distance_matrix
from weitzman.utils.logging_utils import configure_logging, get_logger

from weitzman.metrics.bb_solver import WeitzmanBBSolver

from weitzman.algorithms import ALGORITHMS
from weitzman.algorithms.brute_force import enumerate_all_paths

from experiments.aggregate_results import discover_cells

logger = logging.getLogger(__name__)

_W = 62   # header width


def _card_to_n(card: str) -> int:
    """'m3_p7' -> 36  (Das-Dennis: C(p+2, 2))"""
    try:
        p = int(card.split("p")[1])
        return math.comb(p + 2, 2)
    except Exception:
        return 0


def _hdr(text: str) -> None:
    print(f"\n{'─' * _W}\n{text}\n{'─' * _W}", flush=True)


def _run_cell_heuristics(
    base_config: dict,
    data_path: Path,
    cell_dir: Path,
    seed: int,
    force: bool,
) -> None:
    """Run all configured heuristics on one data cell."""
    from experiments.run_heuristics import _run_algorithms

    alg_names = base_config["algorithms"]["names"]
    if alg_names == ["all"]:
        alg_names = list(ALGORITHMS.keys())

    if not force:
        all_done = all(
            (cell_dir / name / "values").exists()
            and any((cell_dir / name / "values").glob("*.npy"))
            for name in alg_names
        )
        if all_done:
            logger.info("  [skip heuristics] all results present")
            return

    cell_config = copy.deepcopy(base_config)
    cell_config["data"]["instances_dir"] = str(data_path)
    _run_algorithms(cell_config, cell_dir, seed)


def _run_cell_exact_solver(
    base_config: dict,
    data_path: Path,
    cell_dir: Path,
    force: bool,
) -> None:
    """Run the BB exact solver for instances with n ≤ n_max in one data cell."""
    from experiments.run_exact_solver import run_exact_on_dir

    n_max: int = base_config.get("exact_solver", {}).get("n_max", 36)
    exact_dir  = cell_dir / "exact"
    val_dir    = exact_dir / "values"

    if not force and val_dir.exists() and any(val_dir.glob("*.npy")):
        logger.info("  [skip exact solver] results present")
        return

    run_exact_on_dir(
        lattice_dir     = data_path,
        lattice_ext     = base_config["data"]["instance_pattern"],
        lattice_labeled = base_config["data"]["labeled"],
        n_max           = n_max,
        out_dir         = exact_dir,
        force           = force,
    )


def _run_cell_brute_force(
    base_config: dict,
    data_path: Path,
    cell_dir: Path,
    force: bool,
) -> None:
    """Run brute-force enumeration for small instances in one data cell."""
    factorial_dir = cell_dir / "factorial"
    fact_values   = factorial_dir / "values"

    if not force and fact_values.exists() and any(fact_values.glob("*.npy")):
        logger.info("  [skip brute force] results present")
        return

    for sub in ("values", "best_sequences", "worst_sequences"):
        (factorial_dir / sub).mkdir(parents=True, exist_ok=True)

    lattice_ext     = base_config["data"]["instance_pattern"]
    lattice_labeled = base_config["data"]["labeled"]
    n_range: list[int] = base_config["brute_force"]["n_range"]

    instances = sorted(f for f in data_path.iterdir() if f.name.endswith(lattice_ext))
    for inst_path in instances:
        lattice, _ = load_lattice(str(inst_path), labeled=lattice_labeled)
        n = lattice.shape[0]
        if n < n_range[0] or n > n_range[1]:
            logger.debug("  [skip bf] n=%d outside %s (%s)", n, n_range, inst_path.name)
            continue
        d_matrix = compute_distance_matrix(lattice)
        logger.info("  brute force n=%d (%s)", n, inst_path.name)
        t0 = time.time()
        enumerate_all_paths(d_matrix, factorial_dir)
        logger.info("  n=%d done in %.1f s", n, time.time() - t0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch runner across all (kind, geom, card) data cells."
    )
    parser.add_argument("--config",       required=True,
                        help="Config filename (looked up in --config-dir).")
    parser.add_argument("--config-dir",   type=Path, default=Path("configs"))
    parser.add_argument("--data-dir",     type=Path, default=Path("data"),
                        help="Root data directory.")
    parser.add_argument("--batch-dir",    type=Path, default=Path("results/batch"),
                        help="Output root for per-cell results.")
    parser.add_argument("--force",        action="store_true",
                        help="Re-run cells even when output already exists.")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="Skip CSV aggregation at the end.")
    parser.add_argument("--no-plots",     action="store_true",
                        help="Skip trendline figure generation at the end.")
    parser.add_argument("--p-min", type=int, default=None,
                        help="Only process cells with p >= P_MIN (e.g. 4).")
    parser.add_argument("--p-max", type=int, default=None,
                        help="Only process cells with p <= P_MAX (e.g. 10).")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    config = load_config_file(str(args.config_dir / args.config))
    seed   = config["experiment"]["seed"]

    args.batch_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(verbosity=args.verbose,
                      log_file=args.batch_dir / "batch.log")
    log = get_logger(__name__, run_id="batch")

    cells      = discover_cells(args.data_dir)
    cells_list = list(cells.items())

    if args.p_min is not None or args.p_max is not None:
        cells_list = [
            (key, path) for key, path in cells_list
            if (args.p_min is None or int(key[2].split("p")[1]) >= args.p_min)
            and (args.p_max is None or int(key[2].split("p")[1]) <= args.p_max)
        ]
        print(f"Cardinality filter: p_min={args.p_min}, p_max={args.p_max} "
              f"-> {len(cells_list)} cells selected", flush=True)

    total      = len(cells_list)
    log.info("Found %d cells under %s", total, args.data_dir)
    print(f"\nBatch: {total} cells found under {args.data_dir}", flush=True)

    batch_t0 = time.time()

    for idx, ((kind, geom, card), data_path) in enumerate(cells_list, start=1):
        cell_name = f"{kind}_{geom}_{card}"
        cell_dir  = args.batch_dir / cell_name
        cell_dir.mkdir(parents=True, exist_ok=True)

        n = _card_to_n(card)
        _hdr(f"[{idx}/{total}]  {kind} | {geom} | {card}  (n={n})")
        log.info("[%s]", cell_name)

        cell_t0 = time.time()
        _run_cell_heuristics(config, data_path, cell_dir, seed, args.force)
        _run_cell_exact_solver(config, data_path, cell_dir, args.force)
        _run_cell_brute_force(config, data_path, cell_dir, args.force)
        print(f"  Cell done in {time.time() - cell_t0:.1f} s", flush=True)

    print(f"\n{'═' * _W}", flush=True)
    print(f"All cells done in {time.time() - batch_t0:.1f} s", flush=True)
    print(f"{'═' * _W}", flush=True)

    data_csv = args.batch_dir.parent / "data.csv"

    if not args.no_aggregate:
        from experiments.aggregate_results import main as _agg_main
        agg_argv = [
            "--batch-dir", str(args.batch_dir),
            "--data-dir",  str(args.data_dir),
            "--out",       str(data_csv),
        ]
        agg_argv += ["-v"] * args.verbose
        _agg_main(agg_argv)

    if not args.no_aggregate and not args.no_plots and data_csv.exists():
        print(f"\nGenerating trendline figures ...", flush=True)
        from weitzman.plotting.trendlines import main as _tl_main
        _tl_main([
            "--data",   str(data_csv),
            "--outdir", str(args.batch_dir.parent / "figures"),
        ])
        print(f"Figures saved to {args.batch_dir.parent / 'figures'}", flush=True)

        print(f"\nComputing Kendall \\tau tables ...", flush=True)
        from experiments.compute_kendall_tau import run as _kt_run
        _kt_run(data_csv, args.batch_dir.parent)

    log.info("Batch complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
