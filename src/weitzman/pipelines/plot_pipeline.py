
import logging
from pathlib import Path
from typing import Any

from weitzman.io.loaders import load_config_file
from weitzman.utils.registry import build_component
from weitzman.pipelines.registries import HEURISTIC_REGISTRY, METRICS_REGISTRY, PLOTTER_REGISTRY

logger = logging.getLogger(__name__)

def execute_plot(plot_name: str, plot_parameters: dict[str, Any],
                 run_dir: Path, outdir: Path) -> None:
    # Start the process
    logger.info("Starting execute_plot pipeline. outdir=%s", outdir)
    logger.info("Executing plot called: %s.", plot_name)

    # Make the plot
    logger.info("Rendering the figure...")
    plot = build_component(PLOTTER_REGISTRY, plot_name, builder="callable")
    plot(run_dir, outdir, **plot_parameters)

    logger.info("Execute_plot pipeline completed.")
