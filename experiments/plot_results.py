"""
Generate plots from an existing heuristic run directory.

Useful when you want to regenerate or tweak figures without re-running
the algorithms.

Usage (standalone):
    python experiments/plot_results.py --run-dir results/runs/run_XXX --config experiment.yaml

Usage via main.py:
    python main.py plot --run-dir results/runs/run_XXX --config experiment.yaml
"""

import argparse
import logging

from pathlib import Path

from weitzman.io.loaders import load_config_file
from weitzman.utils.logging_utils import configure_logging

from weitzman.plotting.paths import run_paths
from weitzman.plotting.boxplot import run_boxplot

logger = logging.getLogger(__name__)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate plots from an existing heuristic run."
    )
    parser.add_argument("--run-dir",    type=Path, required=True,
                        help="Path to the run directory (e.g. results/runs/run_XXX).")
    parser.add_argument("--config",     required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    configure_logging(verbosity=args.verbose)

    config = load_config_file(str(args.config_dir / args.config))
    run_dir = args.run_dir.resolve()

    if not run_dir.exists():
        logger.error("Run directory does not exist: %s", run_dir)
        return 1

    plot_params = config["plots"]
    lattice_info = {
        "dir":     Path(config["data"]["instances_dir"]),
        "ext":     config["data"]["instance_pattern"],
        "labeled": config["data"]["labeled"],
    }

    heuristics = sorted(
        d.name for d in run_dir.iterdir()
        if d.is_dir() and d.name not in ("logs", "config", "plots")
    )
    if not heuristics:
        logger.error("No algorithm result directories found in %s", run_dir)
        return 1

    factorial_dir = run_dir.parent.parent / "factorial_results"
    (run_dir / "plots").mkdir(exist_ok=True)

    run_boxplot(run_dir, run_dir, heuristics, plot_params, lattice_info,
                factorial_dir=factorial_dir if factorial_dir.exists() else None)

    logger.info("Plots saved to %s/plots/", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
