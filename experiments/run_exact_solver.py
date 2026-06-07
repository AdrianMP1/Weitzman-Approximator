"""
Exact Weitzman solver for instances up to n \\approx 36 using branch-and-bound.

For each .POF instance whose point count n falls within exact_solver.n_max,
runs WeitzmanBBSolver and saves the optimal W(A) value.

Unlike run_brute_force.py, this does NOT enumerate the full permutation
distribution, it only computes the maximum W(A).

Usage (standalone):
    python experiments/run_exact_solver.py --config experiment.yaml

Usage via main.py:
    python main.py exact --config experiment.yaml
"""

import json
import time
import logging
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path

from weitzman.io.writers import save_resolved_config
from weitzman.io.loaders import load_config_file, load_lattice

from weitzman.utils.run_context import create_run_context
from weitzman.utils.logging_utils import configure_logging, get_logger
from weitzman.utils.operations import compute_distance_matrix

from weitzman.metrics.bb_solver import WeitzmanBBSolver

logger = logging.getLogger(__name__)

def run_exact_on_dir(
    lattice_dir: Path,
    lattice_ext: str,
    lattice_labeled: bool,
    n_max: int,
    out_dir: Path,
    force: bool = False,
) -> None:
    """
    Run the BB exact solver on every eligible instance in lattice_dir.

    Results are written to:
        out_dir/values/values_<NNN>_points.npy   shape (1,) = [W(A)]
        out_dir/sequences/sequences_<NNN>_points.npy   shape (n,) = optimal removal order

    Parameters
    ----------
    lattice_dir     : directory containing .POF files.
    lattice_ext     : file extension to match (e.g. ".POF").
    lattice_labeled : whether .POF files have "-> label" suffixes.
    n_max           : only instances with n ≤ n_max are processed.
    out_dir         : output root (subdirs values/ and sequences/ created here).
    force           : recompute even if output already exists.
    """
    for sub in ("values", "sequences"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    instances = sorted(
        f for f in lattice_dir.iterdir() if f.name.endswith(lattice_ext)
    )

    timing: dict[str, float] = {}
    for inst_path in tqdm(instances, desc="Exact(BB)", unit="inst"):
        lattice, _ = load_lattice(str(inst_path), labeled=lattice_labeled)
        n = lattice.shape[0]

        if n > n_max:
            logger.info("Skipping %s (n=%d > n_max=%d)", inst_path.name, n, n_max)
            continue

        stem     = inst_path.stem
        val_path = out_dir / "values"    / f"values_{stem}.npy"
        seq_path = out_dir / "sequences" / f"sequences_{stem}.npy"

        if not force and val_path.exists():
            logger.info("Skipping %s (already computed)", inst_path.name)
            continue

        d_matrix = compute_distance_matrix(lattice)
        logger.info("Solving n=%d (%s)", n, inst_path.name)
        t0 = time.perf_counter()

        solver = WeitzmanBBSolver(d_matrix)
        w_opt  = solver.solve()
        seq    = solver.reconstruct_optimal_sequence()
        elapsed = time.perf_counter() - t0

        logger.info(
            "n=%d | W(A)=%.4f | nodes=%d | memo=%d | %.1f s",
            n, w_opt, solver._nodes_visited, len(solver._memo), elapsed,
        )

        np.save(val_path, np.array([w_opt], dtype=float))
        np.save(seq_path, np.array(seq,    dtype=np.int16))
        timing[stem] = elapsed

    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Exact Weitzman solver (branch-and-bound) for instances up to n_max."
    )
    parser.add_argument("--config",     required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--force",      action="store_true",
                        help="Recompute even when output already exists.")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    config_path = args.config_dir / args.config
    config      = load_config_file(str(config_path))

    ctx = create_run_context(results_root=Path("results"), prefix="exact")
    configure_logging(verbosity=args.verbose,
                      log_file=ctx.run_dir / "logs" / "run.log")
    log = get_logger(__name__, run_id=ctx.run_id)

    save_resolved_config(config, ctx.run_dir)

    lattice_dir     = Path(config["data"]["instances_dir"])
    lattice_ext     = config["data"]["instance_pattern"]
    lattice_labeled = config["data"]["labeled"]
    n_max: int      = config.get("exact_solver", {}).get("n_max", 36)

    # Stable output path (outside timestamped run dir) so batch can find results
    exact_dir = Path("results") / "exact_results"

    log.info("Running exact solver on %s (n_max=%d)", lattice_dir, n_max)
    run_exact_on_dir(
        lattice_dir, lattice_ext, lattice_labeled,
        n_max, exact_dir, force=args.force,
    )

    ctx.mark_success()
    log.info("Exact solver complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
