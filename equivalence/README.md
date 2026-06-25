# Equivalence proof: pure-Python library vs the original Sage library

This folder proves that the pure-Python translation (`../*.py`) computes the **same thing** as the
original Sage library (`../sage_version/*.sage`) — everything **except graph generation**, which uses
a different RNG on purpose (see below).

## How to run it

```bash
# from the repo root
sage equivalence/export_reference.sage        # STEP 1: run the Sage library, dump reference.json
sage -python equivalence/check_equivalence.py # STEP 2: run the Python library, diff against it
#   (step 2 imports no Sage; any Python with numpy/scipy/networkx works:  python equivalence/check_equivalence.py)
```

`export_reference.sage` generates a batch of graphs with the Sage library and records, for each one,
the **input edge list** plus the **reference output** of every function. `check_equivalence.py`
rebuilds the *same* graphs from those edge lists and runs the Python library on them, so the
comparison is apples-to-apples (it never compares two independently *generated* graphs).

## What is proven

**EXACT — bit-for-bit identical to Sage on the same input graph:**
`complete`, `domr_alg`, `left_edge_heuristic` (Gilbert–Jain), `verifier`, `iomr_verifier`,
`is_metric`, the **L1 LP optimum value**, and `MVD_Pivot` / `pivot_heuristic` (reproducible given the
same NumPy seed — the pivot kernel is the same NumPy code as Sage).

**DISTRIBUTIONAL — same problem, non-unique answer (validated as *valid* + size within a few %):**

| function | why it isn't bit-identical |
|---|---|
| `shortest_path_cover` (general & iomr) | when several shortest paths tie, scipy and Sage's Boost pick different ones, so the greedy cover differs slightly |
| `l1_min_heuristic` (the *support*) | the L1 LP is degenerate — many supports achieve the minimum. The **optimum value is identical** (checked exactly above); only which optimal support the solver returns differs |

Every cover produced by the Python library is checked with the verifier and is **valid**.

## The one intentional difference: graph generation

`random_*_weighted_graph` build their structure with `networkx.fast_gnp_random_graph` instead of
Sage's `graphs.RandomGNP`. These draw from different RNG streams, so **the same seed yields a
different graph** in the two libraries. The edge-weight distributions are unchanged, so the
statistical model is the same — but you should **regenerate** results with the Python library rather
than expect to reproduce a specific historical Sage number. Everything *downstream* of a given graph
is faithful, which is exactly what this proof establishes.
