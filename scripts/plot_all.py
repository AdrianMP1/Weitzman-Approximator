
import logging
from pathlib import Path

from weitzman.io.writers import save_resolved_config
from weitzman.io.loaders import load_config_file

from weitzman.pipelines.plot_pipeline import execute_plot

logger = logging.getLogger(__name__)


def run(args) -> int:

    logger.info("CLI plot all called. config=%s | outdir=%s", args.config, args.outdir)

    # Make the configuration file path
    config_file: str = args.config
    config_path: str = str(args.config_path / config_file) # "dir/" + "name.ext"
    logger.info("Config path: %s", config_path)

    # Load configuration dict
    config = load_config_file(config_path)
    ## Write configuration file into run dir
    save_resolved_config(config, args.outdir)

    ## Get dpi
    fig_dpi = config["plotting"]["dpi"]

    ## Show flag
    fig_show = config["plotting"]["show"]

    ## Get heuristic name
    heuristic_name = config["algorithm"]["name"]

    ## Results path
    results_path = Path(args.run_dir) / heuristic_name

    ## Build Lattice Path
    lattice_path = Path(config["data"]["instances_dir"])
    lattice_ext = config["data"]["instance_pattern"]

    # Make plot parameters dict
    parameters = {"dpi": fig_dpi, "show": fig_show,
                  "heuristic_name": heuristic_name,
                  "results_path": results_path,
                  "lattice_path": lattice_path,
                  "lattice_ext": lattice_ext}

    # Run boxplot script
    try:
        execute_plot("boxplot", parameters, 
        run_dir=Path(args.run_dir), outdir=args.outdir)
    except Exception as e:
        logger.info("Error while performing the boxplot: %s.", e)

    # Run paths script
    try:
        execute_plot("paths", parameters, 
                     run_dir=Path(args.run_dir), outdir=args.outdir)
    except Exception as e:
        logger.info("Error while performing the path plot: %s.", e)

    # Notify
    logger.info("CLI plot scripts finished.")

    return 0
