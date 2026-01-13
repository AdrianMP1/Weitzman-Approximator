
import logging
from pathlib import Path

from weitzman.io.writers import save_resolved_config
from weitzman.io.loaders import load_config_file
from weitzman.pipelines.heuristic_pipeline import execute_heuristic

logger = logging.getLogger(__name__)

"""
---------------------------------------------------
NOTE: Already running nearest neighbour.
NOTE: Extend for plotting or try another heuristic.
---------------------------------------------------
"""


def run(args) -> int:

    logger.info("CLI heuristic called. config=%s | outdir=%s", args.config, args.outdir)
    #logger.info("CLI heuristic called. config=%s | seed=%s | outdir=%s", args.config, args.seed, args.outdir)

    # Make the configuration file path
    config_file: str = args.config
    config_path: str = str(args.config_path / config_file) # "dir/" + "name.ext"
    logger.info("Config path: %s", config_path)

    # Load configuration dict
    config = load_config_file(config_path)
    ## Write configuration file into run dir
    save_resolved_config(config, args.outdir)

    ## Get random seed
    seed = config["experiment"]["seed"]

    ## Get heuristic name
    heuristic_name = config["algorithm"]["name"]
    algorithm_parameters = config["algorithm"]["parameters"]

    ## --- Build Lattice Path ---
    lattice_path = Path(config["data"]["instances_dir"])
    lattice_ext = config["data"]["instance_pattern"]
    lattice_labeled = config["data"]["labeled"]

    # Run the heuristic
    execute_heuristic(heuristic_name, algorithm_parameters,
                      lattice_path, lattice_ext, lattice_labeled,
                      seed=seed, outdir=args.outdir)

    # Notify
    logger.info("CLI heuristic finished.")

    return 0
