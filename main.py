"""
Weitzman Diversity - main entry point.

Commands
--------
  batch      Run all (kind, geom, card) cells, compute brute force, write data.csv.
  aggregate  Re-aggregate results/batch/ into results/data.csv (skips computation).
  trendline  Generate trendline figures from results/data.csv.
  run        Run heuristics on a single data directory (development / ad-hoc).
  brute      Exhaustive O(n!) enumeration on a single data directory.
  plot       Regenerate boxplots from an existing run directory.

Typical workflow
----------------
  1. python main.py batch     --config experiment.yaml -v
     (runs everything: heuristics + brute force + CSV aggregation)

  2. python main.py trendline --outdir results/figures
     (one PDF + PNG per (kind, geom, card) cell)

  3. python main.py trendline --kind coverage --geom Linear --card 21
     (single figure)

  4. python main.py aggregate
     (re-build data.csv without re-running algorithms)
"""

import sys
import argparse

from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Weitzman diversity heuristics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    if not sys.argv[1:]:
        parser.print_help()
        sys.exit(0)

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- batch ----
    batch_p = sub.add_parser(
        "batch",
        help="Run all data cells (heuristics + brute force) and aggregate.",
    )
    batch_p.add_argument("--config",       required=True,
                         help="Config filename inside --config-dir.")
    batch_p.add_argument("--config-dir",   type=Path, default=Path("configs"))
    batch_p.add_argument("--data-dir",     type=Path, default=Path("data"))
    batch_p.add_argument("--batch-dir",    type=Path, default=Path("results/batch"))
    batch_p.add_argument("--force",        action="store_true",
                         help="Re-run even when output already exists.")
    batch_p.add_argument("--no-aggregate", action="store_true",
                         help="Skip CSV aggregation at the end.")
    batch_p.add_argument("--no-plots",     action="store_true",
                         help="Skip trendline figure generation at the end.")
    batch_p.add_argument("--p-min", type=int, default=None,
                         help="Only process cells with p >= P_MIN.")
    batch_p.add_argument("--p-max", type=int, default=None,
                         help="Only process cells with p <= P_MAX.")
    batch_p.add_argument("-v", "--verbose", action="count", default=0)

    # ---- aggregate ----
    agg_p = sub.add_parser(
        "aggregate",
        help="Re-aggregate results/batch/ into results/data.csv.",
    )
    agg_p.add_argument("--batch-dir", type=Path, default=Path("results/batch"))
    agg_p.add_argument("--data-dir",  type=Path, default=Path("data"))
    agg_p.add_argument("--out",       type=Path, default=Path("results/data.csv"))
    agg_p.add_argument("-v", "--verbose", action="count", default=0)

    # ---- trendline ----
    tl_p = sub.add_parser(
        "trendline",
        help="Generate trendline figures from results/data.csv.",
    )
    tl_p.add_argument("--data",   type=Path, default=Path("results/data.csv"))
    tl_p.add_argument("--outdir", type=Path, default=Path("results/figures"))
    tl_p.add_argument("--kind",   default=None,
                      help="Restrict to one kind ('coverage' or 'uniformity').")
    tl_p.add_argument("--geom",   default=None,
                      help="Restrict to one geometry ('Concave', 'Convex', 'Linear').")
    tl_p.add_argument("--card",   type=int, default=None,
                      help="Restrict to one cardinality (number of points).")

    # ---- exact ----
    exact_p = sub.add_parser("exact", help="BB exact solver on a single data directory.")
    exact_p.add_argument("--config",     required=True)
    exact_p.add_argument("--config-dir", type=Path, default=Path("configs"))
    exact_p.add_argument("--force",      action="store_true")
    exact_p.add_argument("-v", "--verbose", action="count", default=0)

    # ---- run ----
    run_p = sub.add_parser("run", help="Run heuristics on a single data directory.")
    run_p.add_argument("--config",      required=True,
                       help="Config filename inside --config-dir.")
    run_p.add_argument("--config-dir",  type=Path, default=Path("configs"))
    run_p.add_argument("--seed",        type=int,  default=None,
                       help="Random seed (overrides config value).")
    run_p.add_argument("--no-plots",    action="store_true")
    run_p.add_argument("-v", "--verbose", action="count", default=0)

    # ---- brute ----
    brute_p = sub.add_parser("brute", help="O(n!) exhaustive enumeration.")
    brute_p.add_argument("--config",      required=True)
    brute_p.add_argument("--config-dir",  type=Path, default=Path("configs"))
    brute_p.add_argument("-v", "--verbose", action="count", default=0)

    # ---- plot ----
    plot_p = sub.add_parser("plot", help="Regenerate boxplots from a finished run.")
    plot_p.add_argument("--run-dir",    type=Path, required=True)
    plot_p.add_argument("--config",     required=True)
    plot_p.add_argument("--config-dir", type=Path, default=Path("configs"))
    plot_p.add_argument("-v", "--verbose", action="count", default=0)

    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "batch":
        from experiments.run_batch import main as _main
        fwd = ["--config", args.config,
               "--config-dir", str(args.config_dir),
               "--data-dir",   str(args.data_dir),
               "--batch-dir",  str(args.batch_dir)]
        if args.force:
            fwd += ["--force"]
        if args.no_aggregate:
            fwd += ["--no-aggregate"]
        if args.no_plots:
            fwd += ["--no-plots"]
        if args.p_min is not None:
            fwd += ["--p-min", str(args.p_min)]
        if args.p_max is not None:
            fwd += ["--p-max", str(args.p_max)]
        fwd += ["-v"] * args.verbose
        return _main(fwd)

    if args.command == "aggregate":
        from experiments.aggregate_results import main as _main
        fwd = ["--batch-dir", str(args.batch_dir),
               "--data-dir",  str(args.data_dir),
               "--out",       str(args.out)]
        fwd += ["-v"] * args.verbose
        return _main(fwd)

    if args.command == "trendline":
        from weitzman.plotting.trendlines import main as _main
        fwd = ["--data",   str(args.data),
               "--outdir", str(args.outdir)]
        if args.kind:
            fwd += ["--kind", args.kind]
        if args.geom:
            fwd += ["--geom", args.geom]
        if args.card is not None:
            fwd += ["--card", str(args.card)]
        return _main(fwd)

    if args.command == "exact":
        from experiments.run_exact_solver import main as _main
        fwd = ["--config", args.config, "--config-dir", str(args.config_dir)]
        if args.force:
            fwd += ["--force"]
        fwd += ["-v"] * args.verbose
        return _main(fwd)

    if args.command == "run":
        from experiments.run_heuristics import main as _main
        fwd = ["--config", args.config,
               "--config-dir", str(args.config_dir)]
        if args.seed is not None:
            fwd += ["--seed", str(args.seed)]
        if args.no_plots:
            fwd += ["--no-plots"]
        fwd += ["-v"] * args.verbose
        return _main(fwd)

    if args.command == "brute":
        from experiments.run_brute_force import main as _main
        fwd = ["--config", args.config,
               "--config-dir", str(args.config_dir)]
        fwd += ["-v"] * args.verbose
        return _main(fwd)

    if args.command == "plot":
        from experiments.plot_results import main as _main
        fwd = ["--run-dir", str(args.run_dir),
               "--config", args.config,
               "--config-dir", str(args.config_dir)]
        fwd += ["-v"] * args.verbose
        return _main(fwd)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
