
import logging
from pathlib import Path
from typing import Any

from weitzman.io.loaders import load_config_file
from weitzman.utils.registry import build_component
from weitzman.pipelines.registries import HEURISTIC_REGISTRY, METRICS_REGISTRY, PLOTTER_REGISTRY

logger = logging.getLogger(__name__)

def execute_heuristic(heuristic_name: str, heuristic_params: dict[str, Any],
                        lattice_dir: Path, lattice_ext: str, lattice_labeled: str,
                        seed: int, outdir: Path) -> None:
    # Start the process
    logger.info("Starting execute_heuristic pipeline. outdir=%s", outdir)
    logger.info("Instances to solve: %s. | ext = %s", lattice_dir, lattice_ext)

    # Building heuristic
    logger.info("Building the heuristic %s.", heuristic_name)
    heuristic = build_component(HEURISTIC_REGISTRY, heuristic_name, builder="callable")

    # Make save folders
    logger.info("Creating directories to save results...")
    heuristic_dir = outdir / heuristic_name
    heuristic_dir.mkdir(parents=False, exist_ok=True)

    values_dir = heuristic_dir / "values"
    values_dir.mkdir(parents=False, exist_ok=True)
    sequences_dir = heuristic_dir / "sequences"
    sequences_dir.mkdir(parents=False, exist_ok=True)

    # Pack data
    lattice_params = {"lattice_dir": lattice_dir,
                      "lattice_ext": lattice_ext,
                      "lattice_labeled": lattice_labeled}

    # Run heuristic
    logger.info("Executing the heuristic...")
    heuristic(lattice_params, heuristic_dir, **heuristic_params)

    logger.info("Execute heuristic pipeline completed.")

