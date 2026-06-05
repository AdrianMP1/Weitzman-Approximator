# Weitzman Diversity on Pareto Fronts

Source code accompanying the paper:<br>
**A Permutation-Based Reformulation for Approximating Weitzman Diversity for Benchmarking in Multi-Objective Optimization** <br>
[Parallel Problem Solving From Nature, 2026]

> **[Authors]**<br>
> Mahboubeh Nezhadmoghaddam<br>
> Adrián Isaí Morales-Paredes<br>
> Julio Juárez<br>
> Jesús Guillermo Falcón-Cardona<br>
> Víctor Adrián Sosa Hernández<br>

This repository contains the implementations of four heuristic algorithms for approximating the Weitzman diversity indicator for Pareto fronts (euclidean distance is used), along with a branch & bound implementation for the ground truth (n<= 36) and all scripts required to reproduce the paper's results.

---

## Table of Contents

- [Background](#background)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data](#data)
- [Reproducing the paper results](#reproducing-the-paper-results)
- [Configuration reference](#configuration-reference)
- [Algorithms](#algorithms)
- [Output format](#output-format)

---

## Background

The **Weitzman diversity** of a finite set $A$ is defined recursively as

$$W(A) = \max_{a_i \in A} \bigl[ W(A \setminus \{a_i\}) + d(a_i,\, A \setminus \{a_i\}) \bigr]$$

where $d(a_i, Q) = \min_{a_j \in Q} d(a_i, a_j)$ is the minimum distance from point $a_i$ to the remaining set $Q$, and $W(\{a_i\}) = 0$.

Computing $W(A)$ requires evaluating all $n!$ removal sequences and is only feasible for small values of $n$. This paper studies four heuristic algorithms that approximate $W(A)$ efficiently by constructing removal sequences.

---

## Repository structure

```
Clean_Weitzman/
├── main.py                          ← single entry point (run / batch / plot)
├── pyproject.toml
├── configs/
│   ├── experiment.yaml              ← full experiment (all algorithms, all instances)
│   └── debug.yaml                   ← fast single-algorithm sanity check
├── data/
│   ├── CovLoss(Concave-Convex-Linear)/
│   │   ├── PFAs_CovLoss_Concave/m3_p4 … m3_p19/
│   │   ├── PFAs_CovLoss_Convex/m3_p4 … m3_p19/
│   │   └── PFAs_CovLoss_Linear/m3_p4 … m3_p19/
│   └── UnifLoss(Concave-Convex-Linear)/
│       ├── PFAs_UnifLoss_Concave/m3_p4 … m3_p19/
│       ├── PFAs_UnifLoss_Convex/m3_p4 … m3_p19/
│       └── PFAs_UnifLoss_Linear/m3_p4 … m3_p19/
├── experiments/
│   ├── run_batch.py                 ← orchestrates the full pipeline end-to-end
│   ├── run_heuristics.py            ← run all configured algorithms (used by run_batch)
│   ├── run_brute_force.py           ← O(n!) exact solver for small n
│   ├── run_exact_solver.py          ← B&B exact solver for small n
│   ├── aggregate_results.py         ← collects per-instance .npy to produce a data.csv
│   ├── compute_kendall_tau.py       ← Kendall tau correlation table
│   └── plot_results.py              ← regenerate figures from a finished run
└── weitzman/
    ├── algorithms/                  ← one module per heuristic + brute force
    ├── metrics/                     ← Weitzman computation (B&B), Pure Diversity
    ├── plotting/                    ← trendlines, box plots
    ├── io/                          ← .POF loaders, config loader, writers
    └── utils/                       ← core math, run context, logging
```

**Naming convention for data subfolders:** `m3_pX` denotes instances on a
three-objective ($m = 3$) and a parameter $p = X$.
The number of points is $\binom{p+2}{2}$ (e.g. `m3_p4` -> 15 points,
`m3_p10` -> 66 points, `m3_p19` -> 210 points).

---

## Requirements

- Python >= 3.10
- NumPy >= 1.24, SciPy >= 1.10, Matplotlib >= 3.7
- NetworkX >= 3.0, PyYAML >= 6.0, tqdm >= 4.64, pandas >= 2.0

All dependencies are declared in `pyproject.toml` and installed automatically.

---

## Installation

```bash
# Clone the repository
git clone <repo-url> Weitzman_Project
cd Weitzman_Project

# (Recommended) activate your environment first
conda activate <your-env>       # or: source .venv/bin/activate

# Install the weitzman package in editable mode
pip install -e .
```

Verify the installation:

```bash
python -c "import weitzman; print('OK')"
```

---

## Data

The `data/` directory is organised by Coverage and Uniformity, then for Pareto front geometry:

| Group | Geometries | Instances per geometry |
|-------|-----------|----------------------|
| `CovLoss` | Concave, Convex, Linear | 16 sizes × 6 coverage values |
| `UnifLoss` | Concave, Convex, Linear | 16 sizes × 6 coverage values |

Each `.POF` file contains one three-objective Pareto front. Files at size `m3_p4`
have 15 points; files at `m3_p19` have 210 points.

---

## Reproducing the paper results

All commands must be run from the `Weitzman_Project/` directory.

### Run the full batch [Everything with a single execution]

Runs all four algorithms on every `(kind, geom, card)` data cell, computes the brute-force ground truth for small instances ($n \le 12$ by default) while running B&B for instances $n \leq 28$ (it can be set up to 36, but the execution time is significant), and writes the aggregated `results/data.csv` automatically.

```bash
python main.py batch --config experiment.yaml
```

Results are saved to `results/batch/<kind>_<geom>_<card>/` (one directory per
cell) and produces the `results/figures/` along with the `.csv` files.  The B&B solver is a bottleneck on instances with $n \geq 36$, by default it is set as `n_max: 28` on `exact_solver` on **configs/experiment.yaml**. The $n=36$ is feasible, but takes some hours.

Cells that already have results are skipped automatically; use `--force` to
re-run everything from scratch.

### [Optional] - Re-aggregate without re-running

If you need to rebuild `data.csv` without re-running any algorithms
(e.g. after adding a new metric):

```bash
python main.py aggregate
```

### [Extra] Ad-hoc single-directory run (development)

For quick checks on one data folder without the full batch:

```bash
python main.py run --config debug.yaml -v
```

**Useful flags:**

| Flag | Effect |
|------|--------|
| `--config debug.yaml` | Fast single-algorithm run on `m3_p4` |
| `--seed N` | Override the random seed in the config |
| `--no-plots` | Skip figure generation (run only) |
| `--force` | Re-run even if output exists (batch only) |
| `-v` / `-vv` | Increase log verbosity |

---

## Configuration reference

Both configs (`experiment.yaml` and `debug.yaml`) share the same schema.

```yaml
experiment:
  name: "weitzman_baseline"
  seed: 42

data:
  instances_dir: "data/CovLoss(Concave-Convex-Linear)/PFAs_CovLoss_Linear/m3_p4"
  instance_pattern: ".POF"
  labeled: false          # true for files with " -> label" suffix on each line

algorithms:
  # ["all"] runs every registered algorithm.
  # Alternatively list any subset:
  # ["farthest_neighbour", "twice_around", "christofides", "global_max_min"]
  names: ["all"]

  config:
    farthest_neighbour:
      kind: "max"         # "max" = farthest-first, "min" = nearest-first
      reverse: true

    twice_around:
      mst_mode: "max"     # "max" = maximum spanning tree
      reverse: true

    christofides:
      mst_mode: "max"
      reverse: true

    global_max_min:       # No parameters

exact_solver:
  n_max: 28               # run B&B exact solver for instances with n <= n_max

brute_force:
  n_range: [4, 12]        # only instances with n in [low, high] are processed

plots:
  show: false
  dpi: 200
  fontsize: 12
```

---

## Algorithms

| Key | Full name | Complexity | Description |
|-----|-----------|-----------|-------------|
| `farthest_neighbour` | Farthest-Neighbour | $O(n^3)$ | Greedy farthest-first insertion; exhaustive starting vertices (paper: FN) |
| `twice_around` | Twice-Around-the-Tree | $O(n^2 \log n)$ | Maximum spanning tree -> doubled edges -> Euler circuit -> Hamiltonian shortcut |
| `christofides` | Christofides-inspired | $O(n^3)$ | Maximum spanning tree -> odd-degree matching -> Euler circuit -> Hamiltonian shortcut |
| `global_max_min` | Global Max-Min | $O(n^3)$ | Greedy sequence: $i^* = \arg\max_{u \notin Q} \min_{v \in Q} d(u, v)$; exhaustive starting vertices |

The brute-force solver enumerates all $n!$ sequences and serves as ground truth (B&B too); it is not a heuristic and is not included in the algorithm registry.

---

## Output format

After `python main.py batch`, the results tree is:

```
results/
├── data.csv                             ← aggregated results (all cells, all algorithms)
├── batch/
│   └── <kind>_<geom>_<card>/            ← e.g. coverage_Linear_m3_p4
│       ├── <algorithm>/
│       │   ├── values/
│       │   │   └── values_<instance>.npy ← shape (n,): W score per starting vertex
│       │   ├─── sequences/
│       │   │     └── sequences_<instance>.npy
│       │   └── timing.json              ← execution time summary
│       ├── exact/
│       │   ├── values/
│       │   │   └── values_<instance>.npy ← shape (n,): W score per starting vertex
│       │   ├── sequences/
│       │   │   └── sequences_<instance>.npy
│       │   └── timing.json              ← execution time summary
│       └── factorial/                   ← brute-force (only for n ≤ n_range[1])
│           ├── values/
│           │   └── values_<NNN>_points.npy  ← shape (n!,): all W values
│           ├── best_sequences/
│           └── worst_sequences/
└── figures/                             ← after python main.py trendline or batch
    ├── Coverage/
    │   ├── Linear/
    │   │   └── trendline_<kind>_<geom>_<card>.{pdf,png}
    │   ├── Concave/
    │   └── Convex/
    └── Uniformity/
```

**`data.csv` columns:**

| Column | Description |
|--------|-------------|
| `kind` | `coverage` or `uniformity` |
| `geom` | `Concave`, `Convex`, or `Linear` |
| `card` | number of Pareto front points |
| `lattice_deg` | coverage/uniformity parameter (0.6 – 1.0) |
| `algorithm` | registry key (e.g. `farthest_neighbour`) |
| `min`, `q1`, `median`, `q3`, `max` | distribution of values across starting vertices |
| `W-value` | exact W(A) from brute force (NaN or empty when n > n_max) |
| `PD` | Pure Diversity metric with Euclidean distance |
| `time_s` | execution time in seconds |

For ad-hoc single-directory runs (`python main.py run`), output goes to
`results/runs/run_<YYYYMMDD_HHMMSS_hostname>/` with the same per-algorithm layout.
