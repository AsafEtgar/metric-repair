# Metric-Repair POC Experiments — exact specification

A reproducible comparison of the metric-repair algorithms across graph size and density, on the
**geometric** weight model. Implemented in `experiments/` (`harness.py` is the single source of truth;
this document is the human-readable record). Run instructions: `experiments/RUN.md`.

---

## 1. Goal

For each generated graph, measure **per algorithm**: cover **size**, **CPU + wall time**, **peak memory**,
validity, and (for the exact/LP methods) convergence, cutting-plane rounds, and the fractional lower bound.
The scientific quantity is the **approximation ratio** = size / reference-optimum. Two axes:

- **Experiment 1** — how the algorithms scale with **n** (graph size), at fixed density.
- **Experiment 2** — how they behave across **density** `p = n^{-α}` at fixed n=500, on two fronts:
  the **non-metricity onset** (2a) and the **connectivity / density transition** (2b).

---

## 2. Weight models (and why only geometric)

Weights and edge density are coupled in the standard generators, which makes some models degenerate:

- **uniform** (`w > 1−p` on `K_n`): weights sit in a razor-thin band, so the graph is **always metric**
  (nothing to repair) — **dropped**.
- **exponential** and **coupled geometric**: as `p→0` the weights collapse to ≈1, so the graph turns
  metric — usable only at fixed, non-trivial `p`. Exponential is also float-valued (no exact rsp oracle).
  **Exponential dropped**; **geometric kept**.

We use two geometric variants:

- **Coupled geometric** — `random_geometric_weighted_graph(n, p)`: edges `G(n,p)`, weights
  `Geometric(1−p)`. Weight spread tracks `p`. **Provable onset:** for `p = n^{-α}` the expected number of
  broken triangles is `Θ(n^{3−5α})`, so the graph is non-metric w.h.p. **iff α < 3/5**.
- **Decoupled geometric** — `random_decoupled_geometric_weighted_graph(n, p, weight_p=0.5)`: edges
  `G(n,p)`, weights `Geometric(1−0.5)` **fixed** (independent of `p`, ≈ the coupled model "as if p=0.5").
  Stays non-metric at every density, so it can carry a full density sweep.

---

## 3. The three experiments

40 sampled graphs per point; one **task** = one graph = the full algorithm suite → one CSV.

| Experiment | model | n | density parameter | points | samples | graphs |
|---|---|---|---|---|---|---|
| **Exp 1** (size sweep) | coupled geometric | **100 … 500**, 20 values `linspace(100,500,20)` | fixed **p ∈ {0.3, 0.5}** | 2×20 = 40 | 40 | 1600 |
| **Exp 2a** (non-metricity onset) | coupled geometric | 500 | **α ∈ [1/2, 2/3]**, 20 values; `p = 500^{−α}` | 20 | 40 | 800 |
| **Exp 2b** (density transition) | decoupled geometric | 500 | **α ∈ [1/2, 4/5]**, 20 values; `p = 500^{−α}` | 20 | 40 | 800 |
| | | | | | **total** | **3200** |

- **Exp 2a** brackets the **3/5 onset** (α=0.6): near-zero OPT above it, lifting off as α→½; all instances
  are **connected** (connectivity threshold for n=500 is α≈0.706 > 2/3). OPT is small (single- to
  low-double-digit) and finite-size-smeared — the interest is *critical-regime behaviour*.
- **Exp 2b** sweeps density with a fixed weight spread, so OPT ranges from a few (α=0.8, sparse) to
  thousands (α=0.5, dense). It **crosses the connectivity threshold** (α≈0.706): at the sparse end the
  graph is a giant component plus fragments. Metric repair decomposes over components (see §7).

**n grid (Exp 1):** 100, 121, 142, 163, 184, 205, 226, 247, 268, 289, 311, 332, 353, 374, 395, 416, 437,
458, 479, 500.

---

## 4. Algorithm suite (18 variants)

Run **per connected component** (results aggregated, §7). `n≤k` = runs only when the giant component has
`≤ k` vertices; `|H|≤200` = runs only when the total broken-edge count is small.

| algo id | MR variant | role | oracle | gate |
|---|---|---|---|---|
| `domr` | DOMR | exact DOMR (also a valid GMR cover; gives \|H\|) | — | — |
| `gmr_lp_rsp` | GMR | **exact GMR** LP value + (integral) cover | rsp | — |
| `gmr_lp_naive` | GMR | GMR LP **lower bound** (looser) | naive | — |
| `gmr_ilp` | GMR | **exact GMR** cover (separation ILP; `converged`) | — | — |
| `iomr_ilp` | IOMR | **exact IOMR** cover (separation ILP; `converged`) | — | — |
| `iomr_lp_naive` | IOMR | IOMR LP **lower bound** (scalable) | naive | — |
| `iomr_lp_rsp` | IOMR | IOMR LP **lower bound** (tight) | rsp | **n≤150** |
| `iomr_thr_naive` | IOMR | threshold rounding (heuristic) | naive | — |
| `iomr_thr_rsp` | IOMR | threshold rounding (**guaranteed f-approx**) | rsp | **n≤150** |
| `iomr_bestofk` | IOMR | multi-vertex threshold (`best_of_k=12`) | naive | — |
| `iomr_rand` | IOMR | randomized rounding | naive | — |
| `iomr_regiongrow` | IOMR | GVY region-growing multicut | naive | **\|H\|≤200** |
| `l1sep_gmr` | GMR | cutting-plane L1 (on G) | — | — |
| `l1sep_iomr` | IOMR | cutting-plane L1 (on G) | — | — |
| `spc_gmr` | GMR | shortest-path cover | — | — |
| `spc_iomr` | IOMR | shortest-path cover | — | — |
| `pivot` | GMR | MVD pivot (completion) | — | — |
| `left_edge` | IOMR | Gilbert–Jain (completion) | — | — |

**Reference optimum for the ratio:** exact OPT from `gmr_ilp` / `iomr_ilp` when `converged`, else the
tightest LP lower bound (`*_lp_rsp` where available, else `*_lp_naive`).

**Gates rationale:** `iomr_lp_rsp`/`iomr_thr_rsp` — the rsp IOMR LP does not converge past ~n=150 (the
integrality gap forces many rounds); `region growing` — its multicut is `O(V·E)` per heavy pair.

---

## 5. Recorded columns

**Per instance:** `task, exp, model, n, p, alpha, sample, seed, V, E, w_min, w_max, n_components, giant, H`.

**Per (instance, algorithm):** `algo, variant, status` (ok / timeout / oom / killed / skipped_n /
skipped_H / skipped_time), `size, valid, lp_bound, exact_opt, converged, rounds, cuts, cpu, wall, peak_mb,
oracle, guaranteed, full_separation, min_pair_dist`.

Lower-bound rows (`gmr_lp_naive`, `iomr_lp_*`) have empty `size`/`valid` and report `lp_bound`.

---

## 6. Caps & resources

- **Time:** 30 min per (algorithm, instance) [`TIMEOUT_S`, fork-enforced]; 120 min per task [`TASK_BUDGET_S`;
  later algorithms marked `skipped_time`]. SLURM `--time 02:30:00`, references run first so they're never
  starved.
- **Memory:** 1 GB per task (`--mem-per-cpu 1g`) — heavily over-provisioned; real peak ≈ 16–90 MB.
  `peak_mb` recorded per algorithm. Single-threaded BLAS (`OMP_NUM_THREADS=1`) — fork-safe, deterministic
  timing.
- **Concurrency:** 64 tasks (`--max-jobs 64`), Bouchet `day` partition (well under its 1024-core cap).
  Estimated wall-clock ≈ 3–6 h (the spread is the exact IOMR sep-ILP at n=500, which is unmeasured; it's
  cap-bounded at 30 min/algorithm).

---

## 7. Connected components & aggregation

Each algorithm runs on **every non-metric component** (a component needs repair iff `domr(CC) ≠ ∅`; metric
fragments contribute 0). Metric repair decomposes exactly over components, so aggregation is exact:
`size, cpu, wall, rounds, cuts, lp_bound, exact_opt` **sum**; `peak_mb` **max**;
`valid, converged, guaranteed, full_separation` **AND**; `min_pair_dist` **min**; `status` = worst.
Below the connectivity threshold (Exp 2b sparse end) only the giant component is non-metric in practice.

---

## 8. Reproducibility

Every task's seed is `crc32("exp|model|n|p|alpha|sample")` — deterministic, distinct, and recorded in the
`seed` column. `seed_all(seed)` seeds NumPy and Python's `random` (the graph structure comes from
`networkx.fast_gnp_random_graph`, which uses Python's RNG). Re-running `run_task.py --task-index K`
reproduces task K bit-for-bit.

---

## 9. Known caveats (expected, not bugs)

- **Exact IOMR sep-ILP at n=500** is unmeasured — may not converge within the 30-min cap (recorded via
  `converged=False`); it's the main runtime uncertainty.
- **Region growing** degenerates in this weight regime (full separation never holds → `full_separation`
  will read `False`, `min_pair_dist ≈ 0`); it stays valid via the oracle top-up but carries no ratio.
- **naive-oracle LP bounds** (`gmr_lp_naive`, `iomr_lp_naive`) are loose lower bounds, not covers.
