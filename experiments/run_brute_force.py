"""
Exhaustive O(n!) Weitzman solver for small instances.

Enumerates all n! removal sequences for each instance in the configured
data directory, records the full value distribution, and saves the
optimal (best) and worst sequences.

Usage (standalone):
    python experiments/run_brute_force.py --config experiment.yaml

Usage via main.py:
    python main.py brute --config experiment.yaml

WARNING: runtime grows as n!.  Practical limit is n <= 12 on modern hardware.
"""

import time
import argparse
import logging

from pathlib import Path

from weitzman.io.writers import save_resolved_config
from weitzman.io.loaders import load_config_file, load_lattice

from weitzman.utils.run_context import create_run_context
from weitzman.utils.operations import compute_distance_matrix
from weitzman.utils.logging_utils import configure_logging, get_logger

from weitzman.algorithms.brute_force import enumerate_all_paths

logger = logging.getLogger(__name__)

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Brute-force O(n!) Weitzman enumeration for small instances."
    )
    parser.add_argument("--config",     required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    config_path = args.config_dir / args.config
    config = load_config_file(str(config_path))

    ctx = create_run_context(results_root=Path("results"), prefix="brute")
    configure_logging(verbosity=args.verbose,
                      log_file=ctx.run_dir / "logs" / "run.log")
    log = get_logger(__name__, run_id=ctx.run_id)

    save_resolved_config(config, ctx.run_dir)

    # Results are stored outside the timestamped run directory so that
    # heuristic runs can locate them by a stable path.
    factorial_dir = Path("results") / "factorial_results"
    for sub in ("values", "best_sequences", "worst_sequences"):
        (factorial_dir / sub).mkdir(parents=True, exist_ok=True)

    lattice_dir     = Path(config["data"]["instances_dir"])
    lattice_ext     = config["data"]["instance_pattern"]
    lattice_labeled = config["data"]["labeled"]
    n_range: list[int] = config["brute_force"]["n_range"]   # e.g. [4, 12]

    instance_files = sorted(
        f for f in lattice_dir.iterdir() if f.suffix == lattice_ext or f.name.endswith(lattice_ext)
    )

    for instance_path in instance_files:
        lattice, _ = load_lattice(str(instance_path), labeled=lattice_labeled)
        n = lattice.shape[0]

        if n < n_range[0] or n > n_range[1]:
            log.info("Skipping %s (n=%d outside range %s)", instance_path.name, n, n_range)
            continue

        d_matrix = compute_distance_matrix(lattice)
        log.info("Starting n=%d  (%s)", n, instance_path.name)
        t0 = time.time()
        enumerate_all_paths(d_matrix, factorial_dir)
        log.info("n=%d done in %.1f s", n, time.time() - t0)

    ctx.mark_success()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
