from weitzman.utils.registry import LazyTarget

HEURISTIC_REGISTRY = {
    "nearest_neighbour": LazyTarget("weitzman.algorithms.nearest_neighbour", "run", kind="heuristic"),
    "twice_around": LazyTarget("weitzman.algorithms.twice_around", "run", kind="heuristic"),
    "christofides": LazyTarget("weitzman.algorithms.christofides", "run", kind="heuristic")
}

METRICS_REGISTRY = {
    "hypervolume": LazyTarget("weitzman.metrics.multiobjective", "hypervolume", kind="metric"),
    "riesz-energy": LazyTarget("weitzman.metrics.diversity", "energy", kind="metric"),
    "weitzman": LazyTarget("weitzman.metrics.diversity", "weitzman", kind="metric"), 
    "statistics": LazyTarget("weitzman.metrics.stats", "summary", kind="metric")
}

PLOTTER_REGISTRY = {
    "boxplot": LazyTarget("weitzman.plotting.boxplot", "run_boxplot", kind="plot"),
    "paths": LazyTarget("weitzman.plotting.plot_paths", "run_paths", kind="plot"),
}
