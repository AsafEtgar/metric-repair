# Metric Repair (Sage)

Metric-repair code and experiments. SageMath 10.8 kernel. This repo holds **only**
metric-repair material; hyperbolicity/slimness/etc. studies were moved to sibling folders
(`../average_hyperbolicity/`, `../misc_metric_repair_heuristics/`).

## The core files

| File | Role |
|------|------|
| **`graph_models.sage`** | Random weighted-graph generators (`random_geometric_weighted_graph`, …) + `seed_all`. |
| **`metric_repair.sage`** | The repair algorithms (`domr_alg`, `shortest_path_cover`, `pivot_heuristic`, `l1_min_heuristic`, …) and exactly the support they use (encoding/weights, cycle matrix, `complete`, `verifier`). |
| **`metric_extras.sage`** | Auxiliary helpers **not** used by any repair algorithm: metric/coherence diagnostics (`is_metric`, `cumulative_coherence`), the triangle/cycle matrices, `get_subdivided_graph`, and deprecated wrappers/aliases. Depends on `metric_repair.sage`. |
| **`run_experiments.sage`** | Headless, parametrized experiment runner. One task → one tidy CSV in `results/`. |
| **`process_results.py`** | Merges the per-task CSVs in `results/` into one table for analysis/plotting. |

`graph_models.sage` and `metric_repair.sage` are **independent** — each loads on its own.
`metric_extras.sage` depends on `metric_repair.sage`, so load that first.

## Running it

**Interactively (local Jupyter / sanity checks):**
```python
%run Packages_and_Functions.ipynb     # loads graph_models + metric_repair + metric_extras
# or directly:  load("graph_models.sage"); load("metric_repair.sage"); load("metric_extras.sage")

seed_all(0)
G = random_geometric_weighted_graph(20, 0.5)
pivot_heuristic(G)
```

**Batch / cluster:**
```bash
sage run_experiments.sage --n 100 --p 0.5 --reps 30 --algo all --seed 0   # one task -> results/
sage -python process_results.py                                           # merge results/*.csv
```
Define your actual experiment in `run_one()` inside `run_experiments.sage` (same code runs locally
and on the cluster). For a job array, give each task a distinct `--seed`; `seed_all` seeds Sage,
NumPy and Python's `random` together, and the seed is recorded in every output row.

`Packages_and_Functions.ipynb` is a thin loader kept so existing `%run Packages_and_Functions.ipynb`
cells still work. The pre-split all-in-one notebook, old experiment rounds, and the (unused)
vendored hitting-set solver were moved out of this repo to `../metric_repair_archive/` for separate
auditing.

## Other notebooks (metric repair)
- `EXPERIMENT - Average Metric Repair Experiments.ipynb` — main repair experiments. Tidy/parallel
  harness: `run_instance` → `build_tasks` → `run_sweep` (fork Pool) → long-format rows with
  per-algorithm runtime + `valid`; `save_results` writes git-stamped `.csv.gz` to `results/`,
  `summarize` aggregates. (`...BACKUP-2026-06-25.ipynb` is the pre-rewrite version.)
- `PAPER_PLOTS.ipynb` — paper figures; reads/writes `res_finalle/` and `plots_paper/`.
- `EXPONENTIAL_PAPER_PLOTS.ipynb` — exponential-weight plots (writes `expon_*.csv`).

## Data & support

| Folder | Contents |
|--------|----------|
| `results/` | Per-task experiment CSVs (gitignored; the dir is kept via `.gitkeep`). |
| `res_finalle/` | Result CSVs consumed by `PAPER_PLOTS.ipynb`. **Do not rename** (relative path). |
| `plots_paper/` | Paper figures consumed by `PAPER_PLOTS.ipynb`. **Do not rename.** |

Moved out of the repo to `../metric_repair_archive/` (pending a manual audit): `Archive/` (old
experiment rounds), `vendor/minihit/` (unused hitting-set solver), and the pre-split backup notebook.

## TODO / follow-ups
- Fold the plotting/aggregation functions from `PAPER_PLOTS.ipynb` into `process_results.py`.

## Resolved
- `left_edge_heuristic` / `pivot_heuristic` failing `verifier` (2026-06-25): **not** a model
  mismatch. `MVD_Pivot` and `Gilbert_Jain_IOMR` built their cover from the *position-indexed*
  adjacency matrix but returned position pairs, while generators drop isolated vertices so vertex
  labels are non-contiguous — the verifier read positions as labels and rejected valid covers (worse
  on sparser graphs). Fixed by mapping positions → `Kn.vertices(sort=True)`. (A separate symmetry bug
  in the `MVD_Pivot` recursion — writing `X[j,k]` but not `X[k,j]` — was fixed at the same time.)
  `verifier` also gained a float tolerance (`D[u][v] < w - tol`) so non-integer-weight generators
  aren't tripped by rounding; integer-weight runs are unaffected.
