# RGG Metric-Repair Experiments — exact specification

Ground-truth (planted) metric-repair experiments on **Random Geometric Graphs**, which the `G(n,p)`
weight models can't support because they have no known corrupted set / no coordinates. Two parts:

- **Part 1 — regime characterization:** how the RGG repair problem behaves as every knob moves
  (size, density, break direction/magnitude/fraction, jitter). One-factor-at-a-time (OFAT) sweeps.
- **Part 2 — kNN recovery:** does repair restore true neighborhoods? Build kNN of the true graph (**T**),
  the corrupted graph (**C**), and the repaired graph (**F**); compare all three stages.

`n=500` and `40` seeds throughout, to line up with the POC experiments. Implemented as a **sibling module**
`experiments/rgg_harness.py` reusing the POC harness's fork-isolation, seeding, per-component aggregation,
CSV, and dSQ machinery. The POC harness (`harness.py`) is left frozen.

---

## 0. Weight model — float (exactly metric)

RGG edge weights are **float Euclidean distances** (`weight_scale=None`), so the base is *exactly* metric
and every broken cycle comes from the planted break — `corrupted` is exactly the perturbed set, zero
rounding floor. Cost: the three **rsp** methods (`gmr_lp_rsp`, `iomr_lp_rsp`, `iomr_thr_rsp`) need integer
weights and are **dropped** → **15 of 18 algorithms run**. This keeps clean ground truth *and* both exact
references (`gmr_ilp`, `iomr_ilp` use combinatorial separation, float-safe) and the naive-oracle LP bounds.

Rationale (why not integer): rsp costs `O(w_max·n²)` with `w_max = C·radius`; a C large enough to shrink
the rounding floor makes `w_max` huge and rsp intractable, while a C small enough for rsp leaves a
~50–100-edge floor. Float removes the dilemma. See the decision log in git history.

---

## 1. RGG base + breaks (from `graph_models.py`)

- `random_geometric_metric_graph(n, mode, radius/k, dim, weight_scale=None)` — metric base, stores `pos`.
- `break_metric_graph(G, frac_q, direction, magnitude)` → `(H, corrupted)` — edge reweighting
  (`inflate` = clean planted OPT; `deflate` = shortcut, planted≠OPT; `mixed`).
- `jitter_points(G, n_jitter, jitter, subset_s)` → `(H, corrupted, jittered)` — sensor-drift break
  (partial re-measurement of an independent set of moved points).

`corrupted` is the ground-truth edit set for both break kinds; `jittered` is the extra ground truth for
localization. Every algorithm's cover is scored against `corrupted` (edit precision/recall).

---

## 2. Part 1 — regime characterization (OFAT)

**Baseline instance:** `n=500`, `mode=radius`, mean degree ≈ 12 (`radius = sqrt(deg/(π·n)) ≈ 0.087`),
`dim=2`, float weights. **Baseline break:** `inflate`, `frac_q=0.10`, `magnitude=3`. Jitter magnitudes are
quoted as multiples of `radius` (≈ the local edge length), so they stay meaningful as density changes.

Each sweep varies **one** knob from the baseline. `radius(deg,n) = sqrt(deg/(π·n))`.

| id | knob | values (proposed) | holds fixed | isolates |
|---|---|---|---|---|
| **S1 size** | `n` | 100, 200, 300, 500, 800 (radius ∝ 1/√n → deg≈12: r∈{.195,.138,.113,.087,.069}) | density | size vs density |
| **S2 density (radius)** | mean degree → `radius` | deg ∈ {4,8,12,20,30,40} → r∈{.050,.071,.087,.113,.138,.160} | n=500 | incompleteness/density |
| **S2′ density (knn)** | `k` | 4, 8, 12, 20, 30 (mode=knn) | n=500 | topology variant |
| **S3 inflate mag** | `magnitude` | 1.2, 1.5, 2, 3, 5, 10 (`direction=inflate`) | frac_q=.1 | severity, planted=OPT |
| **S3′ deflate mag** | `magnitude` | 1.2, 1.5, 2, 3, 5, 10 (`direction=deflate`) | frac_q=.1 | severity, planted≠OPT |
| **S4 fraction** | `frac_q` | .01, .02, .05, .10, .20, .30 (inflate & deflate) | magnitude=3 | # broken edges, gap growth |
| **S5a jitter count** | `n_jitter` | 1, 2, 4, 8, 16 | jitter=1.5·r≈.13, s=.5 | localized vs spread |
| **S5b jitter mag** | `jitter` | {0.5,1,1.5,2.5,4}·r = {.044,.087,.131,.218,.350} | n_jitter=4, s=.5 | drift vs edge length |
| **S5c jitter subset** | `subset_s` | .1, .25, .5, .75, .9 | n_jitter=4, jitter=1.5·r | metric↔non-metric dial |
| **S6 interaction** | `magnitude × frac_q` | {1.5,3,6} × {.05,.1,.2} (inflate) | n=500 | severity×placement coupling |

**Samples:** **40 seeds** per config (matching the POC's `N_SAMPLES`). ~64 configs × 40 ≈ **2560 tasks**.

**Recorded per (instance, algorithm, seed):** the POC harness's fields (size, valid, lp_bound, converged,
rounds, cuts, cpu, wall, peak_mb, oracle, guaranteed, full_separation, min_pair_dist) **plus**
- instance: `model=rgg`, `mode`, `radius`, `k`, `dim`, `n`, `V`, `E`, `giant`, `H` (=broken/|DOMR|),
  `break_type`, `direction`, `magnitude`, `frac_q`, `n_jitter`, `jitter`, `subset_s`, `n_corrupted`;
- ground truth: **`edit_precision`** = |cover ∩ corrupted| / |cover|, **`edit_recall`** =
  |cover ∩ corrupted| / |corrupted| (cover = the algorithm's edge set).

Derived post-hoc (as in `analyze.py`): `ratio` = size/OPT, `ratio_domr` = size/|H|. Note at `n=500` the
exact **IOMR** reference (`iomr_ilp`) mostly won't converge within its 3-min cap (as in the POC), so IOMR
ratios fall back to the looser `iomr_lp_naive` bound; the exact **GMR** reference (`gmr_ilp`) still closes.

---

## 3. Part 2 — kNN recovery (jitter only)

The realistic break only. One instance = one RGG + one jitter; per algorithm we materialize the repaired
graph and compare kNN neighborhoods across the three stages.

### 3.1 Pipeline

1. **T** — true RGG. `D_T` = all-pairs shortest-path distances; `kNN_T(v)` = the k nearest of each v.
2. **C** — jittered graph. `D_C` = shortest paths under corrupted weights; `kNN_C(v)`.
3. **F** (per algorithm) — run repair → cover `S` → **repaired distance `D_F = D_{G∖S}`** (shortest paths
   that avoid the flagged edges, using the trusted weights); `kNN_F(v)`. Uniform across algorithms for
   comparability (L1's own weights are an available alternative but we standardize on `D_{G∖S}`).

kNN uses **graph shortest-path distance** in every stage — only the weights differ (true / augmented /
repaired). Within a connected component (or the giant component for sparse instances).

### 3.2 Metrics (per k, averaged over nodes; all three stages kept)

- `jaccard_TC` = mean Jaccard(kNN_T, kNN_C) — how far corruption pushed neighborhoods (baseline),
- `jaccard_TF` = mean Jaccard(kNN_T, kNN_F) — how close repair got,
- `recall_TF` = mean |kNN_T ∩ kNN_F| / k,
- **`lift`** = `jaccard_TF − jaccard_TC` — the headline (positive ⇒ repair restored neighborhoods),
- `triplet_acc_C`, `triplet_acc_F` — distance-ordering agreement with T over sampled triples (no embedding).

Keeping C explicitly lets us read every leg: true→corrupted (damage), corrupted→repaired (repair gain),
true→repaired (residual error).

`k ∈ {5, 10, 20, 50}`; triplet sample = 20·V random triples per instance.

### 3.3 F-producing algorithms

Any method that yields a cover or a repaired weight function (**13**): `domr`, `gmr_ilp`, `iomr_ilp`,
`iomr_thr_naive`, `iomr_bestofk`, `iomr_rand`, `iomr_regiongrow`, `l1sep_gmr`, `l1sep_iomr`, `spc_gmr`,
`spc_iomr`, `pivot`, `left_edge`. The naive LP **bounds** (`gmr_lp_naive`, `iomr_lp_naive`) return no
cover → excluded from F (they still appear in Part 1 as lower bounds).

### 3.4 Sweep (OFAT on jitter severity) + CSV layout

Baseline RGG (n=500, radius≈0.087, float). Sweep, **40 seeds** each:
- `subset_s ∈ {.1,.25,.5,.75,.9}` (n_jitter=8, jitter=2·r≈.175),
- `jitter ∈ {0.5,1,2,3,4}·r = {.044,.087,.175,.262,.350}` (n_jitter=8, subset_s=.5),
- `n_jitter ∈ {2,4,8,16,32}` (jitter=2·r, subset_s=.5).

~15 configs × 40 seeds ≈ **600 tasks**, each × 13 algorithms × 4 k-values.

**CSV layout — long:** one row per `(instance, algorithm, k)`, carrying the Part-1 fields **plus**
`k, jaccard_TC, jaccard_TF, recall_TF, lift, triplet_acc_C, triplet_acc_F`. (Long chosen over wide: tidy
lift-vs-k faceting, drops straight into `analyze.py`'s groupby.)

---

## 4. Infrastructure

- **`experiments/rgg_harness.py`** — sibling of `harness.py`: reuses `run_isolated` (fork + per-algo
  timeout), `seed_all`, per-component split/aggregate, CSV schema. Defines the Part-1 and Part-2 task
  lists and `run_one_rgg_task(index)`.
- **`experiments/run_rgg_task.py`**, **`experiments/make_rgg_joblist.py`** — mirror the POC runners; dSQ
  submission reuses `submit_dsq.sh` (parameterized for the RGG joblist).
- **Caps:** same `TIMEOUT_S` (30 min/algo) and `ALGO_TIMEOUT` (`iomr_ilp` 3 min); RGG capped at `n≤800`.
- **Reproducibility:** `seed = crc32("rgg|part|sweep|config|sample")`, recorded in the `seed` column.
- **Total:** ~2560 (Part 1) + ~600 (Part 2) ≈ **3160 tasks**. Est. wall-clock **~8–12 h** at 64-wide
  (n=500 baseline is heavier than the POC), cap-bounded; dSQ, `day` partition.
- **Analysis:** a new pass (extend `analyze.py`/`plots.py` or an `rgg_analyze.py`) for the edit-precision/
  recall curves and the kNN lift curves — separate step, after data lands.

---

## 5. Sign-off

Resolved: **n=500**, **40 seeds**, float weights, sibling module, 15/18 algorithms (13 F-producing in
Part 2), `D_{G∖S}` repair, all three stages T/C/F kept, **long** CSV for per-k kNN metrics.

Still-tweakable proposals (flag any): mean degree ≈ 12 and `dim=2` at baseline; the sweep ranges
(especially jitter multiples of `radius`, the `n` ceiling of 800, and `frac_q` up to 0.30); `k ∈ {5,10,20,50}`
and the 20·V triplet-sample size.

---

## 6. Running on the cluster

Two grids: **`poc`** (n=100..250 step 10, 30 seeds, ~960 tasks, 4 GB/task) for fast data + analysis, and
**`full`** (n=500 sweeps, 40 seeds, 3160 tasks, 8 GB/task). Same commands, swap `poc`↔`full`. From the repo
root with the conda env from `RUN.md` step 0:

```bash
git pull
module load miniconda && conda activate metricrepair
python experiments/run_rgg_task.py --grid poc --count            # -> 960

# build joblist + batch script:  submit_rgg_dsq.sh <env> <PI_netid> [grid] [mem]
bash experiments/submit_rgg_dsq.sh metricrepair <PI_netid> poc   # -> rgg_poc_joblist.txt, dsq_rgg_poc_submit.sh

# smoke ONE task exactly as dSQ runs it (verifies env activation + pipeline):
bash -c "$(head -1 rgg_poc_joblist.txt)"
echo "exit: $?"                                                  # want 0
head -3 results_rgg_poc/task_000000.csv

# submit + monitor
sbatch dsq_rgg_poc_submit.sh
squeue --me
ls results_rgg_poc/*.csv | wc -l                                # -> 960
dSQAutopsy dsq_rgg_poc_submit.sh rgg_poc_joblist.txt

# collect + health-check (rgg_check.py runs LOCALLY, needs pandas)
python experiments/collect.py --indir results_rgg_poc --out results_rgg_poc_all.csv
sage -python experiments/rgg_check.py --results results_rgg_poc_all.csv
```

**Memory:** 4 GB/task (poc) / 8 GB/task (full) — the earlier **1 GB OOM'd at n=500** (2878 tasks). Override
with a 4th arg, e.g. `submit_rgg_dsq.sh metricrepair <PI_netid> full 12g`. `day` partition, 1 core/task,
`--time 02:30:00`, `--max-jobs 64`; each joblist line self-activates the env. POC wall-clock ~1 h.
