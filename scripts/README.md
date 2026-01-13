# Scripts Folder
CLI adapters with certain rules.

* One script is equivalent to a single CLI action
* It must expose def run(args) -> int, 0 for successful, 1 for error.
* The scripts must call pipelines and the library code, save outputs and keep them simple.

For example:

from weitzman.pipelines.run_heuristic import action

def run(args):
    output = action(config=args.config, seed=args.seed, outdir=args.outdir)
    return 0
