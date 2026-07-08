# Metric Repair (pure Python)

Metric-repair code and experiments. The library is now **pure Python** (numpy / scipy / networkx);
the original **Sage** implementation is preserved under [`sage_version/`](sage_version/) and the two
are proven to agree in [`equivalence/`](equivalence/). This repo holds **only** metric-repair
material; hyperbolicity/slimness/etc. studies live under `../metric_repair_archive/` (e.g.
`../metric_repair_archive/average_hyperbolicity/`, `../metric_repair_archive/misc_metric_repair_heuristics/`).

Why the port: easy cluster deployment (`pip install numpy scipy networkx`, no Sage), no Sage-Integer
pitfalls, and a ~100x faster `shortest_path_cover` via `scipy.sparse.csgraph`.

## The core files (Python)

| File | Role |
|------|------|
| **`graph_models.py`** | Random weighted-graph generators (`random_geometric_weighted_graph`, …) + `seed_all`. |
| **`metric_repair.py`** | The repair algorithms (`domr_alg`, `shortest_path_cover`, `pivot_heuristic`, `left_edge_heuristic`, `l1_min_heuristic`) and exactly the support they use (encoding/weights, cycle matrix, `complete`, `verifier`, `iomr_verifier`). |
| **`metric_extras.py`** | Auxiliary helpers **not** used by any repair algorithm (`is_metric`, `cumulative_coherence`, triangle/cycle matrices, `get_subdivided_graph`, deprecated wrappers/aliases). Imports from `metric_repair.py`. |
| **`run_experiments.py`** | The experiment harness **and** a headless CLI. Importable (used by the notebook) and runnable (`python run_experiments.py …`) → one tidy CSV in `results/`. |
| **`process_results.py`** | Merges the per-task CSVs in `results/` into one table for analysis. |

A weighted graph is a `networkx.Graph` with a numeric `weight` on each edge; an edge is a sorted
`(u, v)` tuple and a cover is a `set` of those. `graph_models.py` and `metric_repair.py` are
independent; `metric_extras.py` imports from `metric_repair.py`.

## Running it

**Interactively (Jupyter / Python ≥ 3.9 with numpy, scipy, networkx, pandas):**
```python
from graph_models import seed_all, random_geometric_weighted_graph
from metric_repair import pivot_heuristic, verifier

seed_all(0)
G = random_geometric_weighted_graph(20, 0.5)
S = pivot_heuristic(G)
verifier(G, S)        # 1 == valid cover
```

**Experiments (notebook):** `EXPERIMENT - Average Metric Repair Experiments.ipynb` imports the harness
from `run_experiments.py` — `build_tasks` → `run_sweep` (parallel across instances) → tidy long-format
rows (cover size, per-algorithm runtime, `valid`); `summarize` aggregates, `save_results` writes a
git-stamped `.csv.gz` to `results/`. Plus the broken-cycle threshold experiments.

**Batch / cluster:**
```bash
python run_experiments.py --generator geometric --n 100 --p 0.3 --reps 30 --seed 0   # one task -> results/
python process_results.py                                                            # merge results/*.csv
```
For a SLURM job array give each task a distinct `--seed` (e.g. `$SLURM_ARRAY_TASK_ID`); `seed_all`
seeds NumPy and Python's `random`, and the seed is recorded in every output row. Pin threads to 1
(`OMP_NUM_THREADS=1`, …) so co-located single-core tasks don't oversubscribe.

## Equivalence to the Sage version

[`equivalence/`](equivalence/) proves the Python library matches Sage on everything **except graph
generation** (which uses a different RNG by design): deterministic functions are bit-for-bit identical
on the same input graph; `shortest_path_cover` and the `l1` support differ only by shortest-path
tie-breaking / LP degeneracy (always valid, sizes within a few %). Run:
```bash
sage equivalence/export_reference.sage        # dump Sage reference outputs
sage -python equivalence/check_equivalence.py # diff the Python library against them  ->  RESULT: PASS
```

## Sage original

[`sage_version/`](sage_version/) keeps the original `.sage` library, the `Packages_and_Functions.ipynb`
loader, and the pre-port experiment / sanity notebooks. It still runs under SageMath 10.8 and is the
reference the equivalence proof checks against. **One intentional behavioural difference:** because the
generators use different RNGs, the same seed gives different graphs in the two libraries — regenerate
results with the Python library rather than expecting to replay specific Sage numbers.

## Other notebooks
- `PAPER_PLOTS.ipynb` — paper figures; reads/writes `res_finalle/` and `plots_paper/`.
- `EXPONENTIAL_PAPER_PLOTS.ipynb` — exponential-weight plots.

(These are plotting/analysis notebooks over result CSVs; they were not part of the port.)

## Data & support

| Folder | Contents |
|--------|----------|
| `results/` | Per-task experiment CSVs (gitignored; the dir is kept via `.gitkeep`). |
| `res_finalle/` | Result CSVs consumed by `PAPER_PLOTS.ipynb`. **Do not rename** (relative path). |
| `plots_paper/` | Paper figures consumed by `PAPER_PLOTS.ipynb`. **Do not rename.** |

Moved out of the repo to `../metric_repair_archive/` (legacy, pending a manual audit): `Archive/` (old
experiment rounds), `vendor/minihit/` (unused hitting-set solver), the pre-split backup notebook, and the
consolidated pre-port legacy folders — `Average Metric Graphs/`, `Sage Script/`, `Random Geometric
Weighted Graphs (Julia)/`, `misc_metric_repair_heuristics/`, and the separate `average_hyperbolicity/` study.

## History
- 2026-06-25: ported the library Sage → pure Python (numpy/scipy/networkx); Sage kept in
  `sage_version/`; equivalence proven in `equivalence/`.
- Earlier (in `sage_version/` history): fixed `MVD_Pivot` / `Gilbert_Jain_IOMR` returning
  position-indexed covers as if they were vertex labels (generators drop isolated vertices, so labels
  are non-contiguous), plus an `MVD_Pivot` symmetry bug and a `verifier` float tolerance. The
  position↔label discipline is preserved in the Python port (`graph_to_matrix` / `positions_to_labels`).
