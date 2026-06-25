# Metric Repair (Sage)

Metric-repair code and experiments. SageMath 10.8 kernel. This repo holds **only**
metric-repair material; hyperbolicity/slimness/etc. studies were moved to sibling folders
(`../average_hyperbolicity/`, `../misc_metric_repair_heuristics/`).

## The four core files

| File | Role |
|------|------|
| **`graph_models.sage`** | Random weighted-graph generators (`random_geometric_weighted_graph`, …) + `seed_all`. |
| **`metric_repair.sage`** | The repair algorithms (`domr_alg`, `shortest_path_cover`, `pivot_heuristic`, `l1_min_heuristic`, …) and everything they need (encoding/weights, cycles, `complete`, `verifier`, coherence). |
| **`run_experiments.sage`** | Headless, parametrized experiment runner. One task → one tidy CSV in `results/`. |
| **`process_results.py`** | Merges the per-task CSVs in `results/` into one table for analysis/plotting. |

`graph_models.sage` and `metric_repair.sage` are **independent** — each loads on its own.

## Running it

**Interactively (local Jupyter / sanity checks):**
```python
%run Packages_and_Functions.ipynb     # loads graph_models.sage + metric_repair.sage
# or directly:  load("graph_models.sage"); load("metric_repair.sage")

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
cells still work. `Packages_and_Functions.BACKUP-2026-06-24.ipynb` is the pre-split all-in-one notebook.

## Other notebooks (metric repair)
- `EXPERIMENT - Average Metric Repair Experiments.ipynb` — main repair experiments (writes `alg_p03.csv`).
- `PAPER_PLOTS.ipynb` — paper figures; reads/writes `res_finalle/` and `plots_paper/`.
- `EXPONENTIAL_PAPER_PLOTS.ipynb` — exponential-weight plots (writes `expon_*.csv`).

## Data & support

| Folder | Contents |
|--------|----------|
| `results/` | Per-task experiment CSVs (gitignored; the dir is kept via `.gitkeep`). |
| `res_finalle/` | Result CSVs consumed by `PAPER_PLOTS.ipynb`. **Do not rename** (relative path). |
| `plots_paper/` | Paper figures consumed by `PAPER_PLOTS.ipynb`. **Do not rename.** |
| `vendor/minihit/` | Vendored minimal-hitting-set solver (github.com/TheMatjaz/minihit). Currently unused; re-enable with `import sys; sys.path.append("vendor")` before `import minihit`. |
| `Archive/` | Old split library, old tests and earlier result rounds. |

## TODO / follow-ups
- Fold the plotting/aggregation functions from `PAPER_PLOTS.ipynb` into `process_results.py`.
- `left_edge_heuristic` / `pivot_heuristic` sometimes fail `verifier` — likely a repair-model
  mismatch (`verifier` heavies covered edges to max); worth resolving before a large sweep.
