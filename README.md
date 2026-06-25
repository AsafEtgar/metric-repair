# Average Metric Repair (Sage)

Code for the metric-repair experiments. SageMath 10.8 kernel.

## Library

All shared functions live in one place:

| File | Role |
|------|------|
| **`packages_and_functions.sage`** | The library (graph generators, metric-repair algorithms, cycle/coherence helpers, weight utilities). Edit **here** — it's plain text and diff-friendly. |
| `Packages_and_Functions.ipynb` | Thin loader: one cell, `load("packages_and_functions.sage")`. Kept so existing `%run Packages_and_Functions.ipynb` cells work unchanged. |
| `Packages_and_Functions.BACKUP-2026-06-24.ipynb` | Pre-cleanup all-in-one notebook (backup). |

Use it from any notebook with:

```python
%run Packages_and_Functions.ipynb     # or, directly:  load("packages_and_functions.sage")
```

Library sections (inside the `.sage` file): imports · edge encoding & weights · metric utilities ·
coherence · cycle functions · metric-repair algorithms · graph generators · backwards-compatible
aliases · scratch/examples.

## Notebooks by role

**Experiments**
- `EXPERIMENT - Average Metric Repair Experiments.ipynb` — main repair-algorithm experiments (writes `alg_p03.csv`).
- `EXPERIMENT_SHORTEST_PATH_DIST.ipynb` — shortest-path distance experiments.

**Paper plots**
- `PAPER_PLOTS.ipynb` — figures for the paper; reads/writes `res_finalle/` and `plots_paper/`.
- `EXPONENTIAL_PAPER_PLOTS.ipynb` — exponential-weight plots (writes `expon_*.csv`).

**Graph-property studies**
- `AVG_HYPERBOLICITY.ipynb` — average hyperbolicity (writes `AVG_hyperbolicity_*.pdf`, see note below).
- `AVG_SLIMNESS.ipynb` — average slimness.
- `OUTERPLANAR.ipynb` — outerplanar-graph study.
- `TEST-Diameter and Hyperbolicity.ipynb` — diameter/hyperbolicity checks (loader fixed to the consolidated library).

**Algorithms / scratch**
- `SHAPLEY.ipynb` — Shapley-value heuristic.
- `PLAYGROUND.ipynb`, `PLAYGROUND Random Hitting Set.ipynb` — scratch / exploration.

## Data & outputs

| Folder | Contents |
|--------|----------|
| `res_finalle/` | Result CSVs (consumed by `PAPER_PLOTS.ipynb`). **Do not rename** — referenced by relative path. |
| `plots_paper/` | Paper figures (consumed by `PAPER_PLOTS.ipynb`). **Do not rename.** |
| `outputs/` | Generated hyperbolicity PDFs (tidied here). |
| `vendor/minihit/` | Vendored minimal-hitting-set solver (from github.com/TheMatjaz/minihit). Currently unused. |
| `Archive/` | Old split library (`Packages`, `Cycle_Mtx_functions`, `Graph_Generators`, `Metric_Repair_Algorithms`), old tests and earlier result rounds. |

### Notes
- **Re-running `AVG_HYPERBOLICITY.ipynb`** writes new PDFs to the folder root (its `savefig` uses bare
  filenames). Move them into `outputs/` afterwards, or change the `savefig(...)` paths to `outputs/...`.
- **Re-enabling `minihit`**: it now lives in `vendor/`, so add `import sys; sys.path.append("vendor")`
  before `import minihit`, or move the package back to the folder root.
