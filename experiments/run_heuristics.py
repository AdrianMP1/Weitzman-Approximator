"""
Run heuristic algorithms on a set of Pareto front instances.

Usage (standalone):
    python experiments/run_heuristics.py --config experiment.yaml

Usage via main.py:
    python main.py run --config experiment.yaml
"""

import logging
import argparse

from pathlib import Path

from weitzman.io.loaders import load_config_file
from weitzman.io.writers import save_resolved_config

from weitzman.utils.run_context import create_run_context
from weitzman.utils.logging_utils import configure_logging, get_logger

from weitzman.algorithms import ALGORITHMS
from weitzman.plotting.boxplot import run_boxplot

logger = logging.getLogger(__name__)

def _run_algorithms(config: dict, run_dir: Path, seed: int) -> None:
    algorithm_names: list[str] = config["algorithms"]["names"]
    if algorithm_names == ["all"]:
        algorithm_names = list(ALGORITHMS.keys())

    algorithm_configs: dict = config["algorithms"].get("config", {})
    lattice_params = {
        "lattice_dir":     Path(config["data"]["instances_dir"]),
        "lattice_ext":     config["data"]["instance_pattern"],
        "lattice_labeled": config["data"]["labeled"],
    }

    for name in algorithm_names:
        if name not in ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{name}'. Available: {sorted(ALGORITHMS.keys())}"
            )
        alg_dir = run_dir / name
        (alg_dir / "values").mkdir(parents=True, exist_ok=True)
        (alg_dir / "sequences").mkdir(parents=True, exist_ok=True)

        params = algorithm_configs.get(name, {})
        logger.info("Running %s", name)
        ALGORITHMS[name](lattice_params, alg_dir, seed, **params)


def _run_plots(config: dict, run_dir: Path) -> None:
    plot_params = config["plots"]
    lattice_info = {
        "dir":     Path(config["data"]["instances_dir"]),
        "ext":     config["data"]["instance_pattern"],
        "labeled": config["data"]["labeled"],
    }

    heuristics = sorted(
        d.name for d in run_dir.iterdir()
        if d.is_dir() and d.name not in ("logs", "config")
    )
    if not heuristics:
        logger.warning("No algorithm result directories found in %s", run_dir)
        return

    # Brute-force results live two levels above the run directory:
    # results/runs/<run_id>/ -> results/factorial_results/
    factorial_dir = run_dir.parent.parent / "factorial_results"

    (run_dir / "plots").mkdir(exist_ok=True)
    run_boxplot(run_dir, run_dir, heuristics, plot_params, lattice_info,
                factorial_dir=factorial_dir if factorial_dir.exists() else None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Weitzman heuristic algorithms on Pareto front instances."
    )
    parser.add_argument("--config",      required=True,
                        help="Config filename (looked up inside --config-dir).")
    parser.add_argument("--config-dir",  type=Path, default=Path("configs"),
                        help="Directory containing config files.")
    parser.add_argument("--seed",        type=int,  default=None,
                        help="Random seed (overrides value in config).")
    parser.add_argument("--no-plots",    action="store_true",
                        help="Skip plot generation.")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    config_path = args.config_dir / args.config
    config = load_config_file(str(config_path))
    seed = args.seed if args.seed is not None else config["experiment"]["seed"]

    ctx = create_run_context(results_root=Path("results"), prefix="run")
    configure_logging(verbosity=args.verbose,
                      log_file=ctx.run_dir / "logs" / "run.log")
    log = get_logger(__name__, run_id=ctx.run_id)
    log.info("Run started: %s", ctx.run_dir)

    save_resolved_config(config, ctx.run_dir)
    _run_algorithms(config, ctx.run_dir, seed)

    if not args.no_plots:
        _run_plots(config, ctx.run_dir)

    ctx.mark_success()
    log.info("Run completed: %s", ctx.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
