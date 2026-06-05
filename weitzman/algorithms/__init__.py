from weitzman.algorithms import (
    farthest_neighbour,
    twice_around,
    christofides,
    global_max_min,
)

ALGORITHMS: dict = {
    "farthest_neighbour": farthest_neighbour.run,
    "twice_around":       twice_around.run,
    "christofides":       christofides.run,
    "global_max_min":     global_max_min.run,
}
