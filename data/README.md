# Datasets

Real data lives here. Loaders in [`../datasets.py`](../datasets.py) turn each supported file into the
same weighted `networkx.Graph` (`G[u][v]["weight"]`) that the generators produce, so every repair
algorithm, the verifier, and the experiment harness work on real data unchanged.

## Layout

| folder | tracked in git? | what goes here |
|---|---|---|
| `raw/` | **no** (gitignored) | datasets exactly as obtained. Large/private — kept local, not committed. |
| `processed/` | **no** (gitignored) | cleaned / derived graphs (e.g. saved edge lists from `save_edgelist`). Regenerable. |
| `examples/` | **yes** | tiny synthetic samples documenting each format (used by the smoke test). |

Only `raw/` and `processed/` are ignored (a `.gitkeep` keeps the empty dirs in git). Drop your data in
`raw/`, write any cleaned version to `processed/`, and commit a small sample to `examples/` if it helps
others run the code.

## Supported formats

### 1. Edge list — `examples/example_edgelist.csv`
A CSV with columns `u, v, weight`, one row per edge. Best for sparse / partially-observed data.

```python
from datasets import load_edgelist
G = load_edgelist("data/examples/example_edgelist.csv")
```

### 2. Distance / dissimilarity matrix — `examples/example_distance_matrix.csv`
A square, symmetric matrix of pairwise dissimilarities, with an optional label row+column (a blank
corner cell). Zero diagonal. This is the canonical metric-repair input: the matrix may **violate the
triangle inequality**, which is exactly what the repair algorithms fix.

```python
from datasets import load_distance_matrix
G = load_distance_matrix("data/examples/example_distance_matrix.csv")          # complete graph
G = load_distance_matrix("data/examples/example_distance_matrix.csv", threshold=2)  # keep near pairs only
```

`.npy` arrays are also accepted (`load_distance_matrix("...npy")`, no labels). NaN/inf entries mean
"not measured" → no edge; pass `zero_is_missing=True` if your file encodes a missing pair as 0.

### 3. MATLAB `.mat` — `examples/example_mat_matrix.mat`
A dissimilarity matrix stored in a MATLAB file (a dict of named variables). The single square variable
is picked automatically; name it with `var=` when there are several.

```python
from datasets import load_mat_matrix
G = load_mat_matrix("data/examples/example_mat_matrix.mat")            # auto-pick the matrix variable
G = load_mat_matrix("data/raw/mydata.mat", var="D", threshold=2)      # name it; same matrix options
```

Reads MATLAB v4–v7 via `scipy.io.loadmat`. **v7.3** files are HDF5-based and unreadable by loadmat —
the loader raises a clear error pointing at `pip install mat73` (or load with h5py and pass the array
to `graph_from_matrix`).

## Inspect a loaded dataset

```python
from datasets import describe
describe(G)
# {'n_vertices': 4, 'n_edges': 5, 'connected': True, 'weight_min': 1, 'weight_max': 3,
#  'cycle_length_bound': 3, 'is_metric': False, 'n_broken_cycles': 1}
```

`cycle_length_bound` is `metric_repair.broken_cycle_length_bound(G)` — the cap that makes broken-cycle
enumeration (and the exact ILP) tractable. It is small when weights are integers or span a narrow
dynamic range, and large otherwise (see that function's docstring).

## Practical notes / best practices

- **Keep raw data immutable.** Never edit files in `raw/`; write cleaned versions to `processed/` so a
  run can always be traced back to the original.
- **Record provenance.** For each dataset add a short note (source URL, date, license, units of the
  dissimilarity) next to it or in this README.
- **Weights ≥ 1 / small dynamic range** make the exact ILP and rounding heuristics fast. If your
  dissimilarities are real-valued and wildly scaled, the cycle bound degrades; prefer the heuristics,
  or (once built) the separation-oracle LP, which never enumerates cycles.
- **Integer-ish data** (counts, ranks) round-trips losslessly; genuinely continuous data stays float
  and the verifier uses a tolerance.
