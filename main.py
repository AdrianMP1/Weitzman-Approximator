"""
main.py - Project entry point.

Usage examples:
    python main.py run ...
"""

INTRO = """
=================================================
 Weitzman CLI
=================================================
Run experiments, generate plots, and build reports.

Examples:
  python main.py run train --config cfg.yaml
python main.py plot curves --run-id run_001
  python main.py report summary --runs r1 r2

Use -h or --help for full options.
=================================================
"""

import sys
import logging
import argparse
import importlib
import subprocess

from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional

from weitzman.utils.run_context import create_run_context
from weitzman.utils.logging_utils import configure_logging, get_logger

# --- Logging / Startup ---

# - Make a global logger
logger = logging.getLogger("main")


# --- Project paths Auxiliar ---
def project_root() -> Path:
    """
    Returns the project root.
    """
    return Path(__file__).resolve().parent


# --- Dispatch ---
# - Metadata
## Avoids importing everything at startup - Lazy loading
@dataclass(frozen=True)
class DispatchTarget:
    module: str
    func: str

# - Maps for modules
RUN_TARGETS: dict[str, DispatchTarget] = {
    "heuristic": DispatchTarget(module="scripts.heuristic", func="run"),
    "evaluate": DispatchTarget(module="scripts.evaluate",   func="run"),
    "brute-force": DispatchTarget(module="scripts.brute_evaluator", func="run"),
}

PLOT_TARGETS: dict[str, DispatchTarget] = {
    "curves": DispatchTarget(module="scripts.plot_curves",  func="run"),
    "pareto": DispatchTarget(module="scripts.plot_pareto",  func="run"),
    "all"   : DispatchTarget(module="scripts.plot_all",     func="run"),
}

REPORT_TARGETS: dict[str, DispatchTarget] = {
    "summary": DispatchTarget(module="scripts.report_summary", func="run"),
}

# - In-process dispatch utilities
def load_callable(target: DispatchTarget) -> Callable[[argparse.Namespace], int]:
    """
    Make use of DispatchTarget metadata to dynamically import modules & functions
    """
    mod = importlib.import_module(target.module)
    fn = getattr(mod, target.func, None)
    
    if fn is None or not callable(fn):
        raise AtributeError(f"Target {target.module}.{target.func} not found or not callable.")

    return fn


def dispatch_in_process(group: str, action: str, args: argparse.Namespace) -> int:
    """
    group: "run" | "plot" | "report"
    action: e.g. "heuristic", "all", "summary"
    """
    registry = {"run": RUN_TARGETS, "plot": PLOT_TARGETS, "report": REPORT_TARGETS}.get(group)
    
    if registry is None:
        raise ValueError(f"Unknown group: {group}")

    if action not in registry:
        raise ValueError(f"Unknown {group} action: {action}")

    target = registry[action]
    fn = load_callable(target)
    
    logger.debug("Dispatching in-process: %s %s -> %s.%s", group, action, target.module, target.func)
    return int(fn(args))


# --- Subprocess, isolated executions ---
def dispatch_subprocess(script_path: Path, passthrough_args: list[str]) -> int:
    """
    Execute another script as a subprocess.
    Strict isolation or separate env behavior
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    cmd = [sys.executable, str(script_path), *passthrough_args]
    logger.debug("Running subprocess: %s", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(project_root()))
    return int(completed.returncode)


# --- CLI definition ---
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Project entry point..."
    )

    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v=INFO, -vv=DEBUG)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and show what would run, but do not execute."
    )
    parser.add_argument(
        "--dispatch",
        choices=["inproc", "subprocess"],
        default="inproc",
        help="Dispatch mode. Use 'inproc' or 'subprocess'."
    )
    parser.add_argument(
        "--keep-run-dir",
        action="store_true",
        default=True,
        help="Keep the run folder after execution",
    )
    parser.add_argument(
        "--cleanup-on-success",
        dest="keep_run_dir",
        action="store_false",
        help="Delete the run folder if the run succeeds.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Avoid all plots if the run succeeds.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # -- run --
    run_p = sub.add_parser("run", help="Run a pipeline (heuristic/evaluate/...).")
    run_sub = run_p.add_subparsers(dest="action", required=True)

    heuristic_p = run_sub.add_parser("heuristic", help="Execute a heuristic over a set of lattices.")
    heuristic_p.add_argument("--config", type=str, required=True, help="Config file name.")
    heuristic_p.add_argument("--config_path", type=Path, default=project_root() / "configs/", help="Path to configs directory.")
    heuristic_p.add_argument("--seed", type=int, default=42, help="Random seed.")
    heuristic_p.add_argument("--outdir", type=Path, default=project_root() / "results/run_latest", help="Output directory.")

    eval_p = run_sub.add_parser("evaluate", help="Evaluate a prior run.")
    eval_p.add_argument("--run-id", type=str, required=True, help="Run identifier (folder name or ID).")
    #eval_p.add_argument("--metrics", nargs="+", default=["f1"], help="Metrics to compute.")
    eval_p.add_argument("--outdir", type=str, default="results/eval_latest", help="Output directory.")

    brute_p = run_sub.add_parser("brute-force", help="Execute n! paths of Weitzman.")
    brute_p.add_argument("--config", type=str, required=True, help="Config file name.")
    brute_p.add_argument("--config_path", type=Path, default=project_root() / "configs/", help="Path to configs directory.")
    brute_p.add_argument("--outdir", type=Path, default=project_root() / "results/run_latest", help="Output directory.")

    # ---- plot ----
    plot_p = sub.add_parser("plot", help="Generate plots from results.")
    plot_sub = plot_p.add_subparsers(dest="action", required=True)

    curves_p = plot_sub.add_parser("curves", help="Plot training/eval curves.")
    curves_p.add_argument("--run-id", type=str, required=True, help="Run identifier.")
    curves_p.add_argument("--outdir", type=str, default=project_root() / "results/run_latest", help="Output image path.")
    curves_p.add_argument("--show", action="store_true", help="Show the plot interactively.")
    curves_p.add_argument("--config", type=str, required=True, help="Config file name.")
    curves_p.add_argument("--config_path", type=Path, default=project_root() / "configs/", help="Path to configs directory.")

    pareto_p = plot_sub.add_parser("pareto", help="Plot Pareto front (example).")
    pareto_p.add_argument("--run-id", type=str, required=True)
    pareto_p.add_argument("--out", type=str, default="results/pareto.png")
    pareto_p.add_argument("--show", action="store_true")

    # ---- report ----
    rep_p = sub.add_parser("report", help="Generate reports / summaries.")
    rep_sub = rep_p.add_subparsers(dest="action", required=True)

    summary_p = rep_sub.add_parser("summary", help="Create a consolidated summary report.")
    summary_p.add_argument("--runs", nargs="+", required=True, help="List of run directories/IDs.")
    summary_p.add_argument("--out", type=str, default="results/summary.json", help="Output report path.")

    return parser


# -----------------
# Dispatch commands
# -----------------
def dispatch_main_commands(group: str, action: str, args: argparse.Namespace) -> int:

    # Execute main instructions
    try:
        # Dispatch decision
        if args.dispatch == "inproc":
            exit_code = dispatch_in_process(group=group, action=action, args=args)

        else:
            # Subprocess mode dictionary
            script_map = {
                ("run", "train"): project_root() / "scripts" / "train.py",
                ("run", "evaluate"): project_root() / "scripts" / "evaluate.py",
                ("plot", "curves"): project_root() / "scripts" / "plot_curves.py",
                ("plot", "pareto"): project_root() / "scripts" / "plot_pareto.py",
                ("report", "summary"): project_root() / "scripts" / "report_summary.py",
            }

            script = script_map.get((group, action))
            if script is None:
                logger.exception("Fatal error: %s", f"No subprocess mapping for {group} {action}")
                raise ValueError(f"No subprocess mapping for {group} {action}")

            # Pass through only the subcommand-specific args to the script.
            # Simple approach: rebuild from sys.argv (minus main.py + group/action flags).
            # Robust approach: define per-script args; for boilerplate, keep it straightforward.
            passthrough = sys.argv[3:]  # assumes: python main.py <group> <action> ...
            exit_code = dispatch_subprocess(script, passthrough)

        if exit_code != 0:
            # Fatal Error
            return exit_code

    except Exception:
        # Failure
        raise

    else:
        print("Main Computation Terminated...")
        return exit_code

# -----------------
# Dispatch plots
# -----------------
def dispatch_all_plots(args: argparse.Namespace) -> int:
    # Run all plots
    try:
        if args.no_plot:
            exit_code = 0
            return exit_code

        exit_code = dispatch_in_process(group="plot", action="all", args=args)

    except Exception:
        # Failure
        raise

    else:
        print("Plot Scripts Terminated...")

# -----------------
# Main
# -----------------
def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print(INTRO)
        #build_parser().print_help()
        return 0

    # Build parser
    parser = build_parser()
    args = parser.parse_args(argv)

    ## Get attributes from args
    group = args.command
    action = args.action

    # Create run folder
    results_root = Path(getattr(args, "results_root", "results"))

    if getattr(args, "run_id", None) is None:
        print("HELLO")
        args.run_id = None

    keep = getattr(args, "keep_run_dir", True)
    ctx = create_run_context(results_root=results_root, prefix=group, keep=keep, run_id=args.run_id)

    ## Start cleanup for early crashes
    ctx.register_cleanup()

    # Configure logging to console + file inside run directory
    log_path = ctx.run_dir / "logs" / "run.log"
    configure_logging(verbosity=args.verbose, log_file=log_path)

    # Use an adapter that injects run_id
    logger = get_logger("main", run_id=ctx.run_id)
    logger.info("Run ID: %s", ctx.run_id)
    logger.info("Run started: %s", ctx.run_dir)
    logger.debug("Parsed args: %s", args)

    # Dry-run
    if args.dry_run:
        logger.warning("DRY RUN: would execute %s %s (dispatch=%s)", group, action, args.dispatch)
        return 0

    # Make run_dir available to pipelines via args
    args.run_id = ctx.run_id
    args.run_dir = str(ctx.run_dir)

    # Configure outdir to be run_id
    if str(args.outdir).endswith("run_latest"):
        args.outdir = Path(args.run_dir)

    # Execute main instructions
    exit_code = dispatch_main_commands(group, action, args)
    if exit_code != 0:
        # Fatal Error
        return exit_code

    if group in ["plot", "evaluate"]:
        # Mark as successful
        ctx.mark_success()
        return 0

    # Run all plots
    exit_code = dispatch_all_plots(args)
    if exit_code != 0:
        # Fatal Error
        return exit_code

    # Mark as successful
    ctx.mark_success()

    # Exit function and return 0
    return 0

if __name__ == "__main__":

    try:
        # Run the program
        raise SystemExit(main())

    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        raise SystemExit(130)

    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        raise SystemExit(1)
