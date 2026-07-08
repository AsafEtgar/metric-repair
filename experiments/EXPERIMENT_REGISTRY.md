# Experiment Registry & Large-Scale (n ≤ 3000) Design

The master log of **every metric-repair experiment campaign** — what ran, with which parameters, where the
outputs live, the cost, and the headline findings — plus the **design for the next, larger-scale (n ≤ 3000)**
suite on the RGG and geometric models. Pairs with the per-campaign specs it points to
(`EXPERIMENTS.md` geometric, `RGG_EXPERIMENTS.md`, `REAL_EXPERIMENTS.md`) and the theory/decisions handoff
(`STATUS.md`, `OVERVIEW.md`). If it isn't in this table, we didn't run it.

---

## 1. Campaigns run so far

| # | campaign | model / grid | n range | configs × seeds = tasks | algos | output dir → combined | compute | status |
|---|---|---|---|---|---|---|---|---|
| 1 | **Geometric small** | geometric (`grid=small`) | exp1 100–300 (step 10); exp2a/2b @ n=300 | 82 × 30 = **2460** | 21 (incl. rsp + ILP) | `results_small/` → `results_small_all.csv` | ~454 core-h¹ | ✅ complete |
| 2 | **RGG poc** | RGG float (`grid=poc`) | 100–250 (step 10) | 32 × 30 = **960** | 15 (rsp dropped) | `results_rgg_poc/` → `results_rgg_poc_all.csv` | ~16 core-h | ✅ complete |
| 3 | **RGG full** | RGG float (`grid=full`) | size sweeps 100–500 (step 20); OFAT baseline n=300 | 116 × 40 = **4640** | 15 (rsp dropped) | `results_rgg/` → `results_rgg_full_all.csv` | ~55 core-h | ✅ complete |
| 4 | **Real datasets — heur** | 19 real graphs | n=75–5000 (per graph) | 19×(1 det + 30 rand) = **589** | 18 (rsp dropped) | `results_real/` (+ `results_real_covers/`) | ~4h array setup | 🔄 re-running² |
| 5 | **Real datasets — ILP** | 16 dist-sensible graphs | same graphs | 16 × 2 = **32** | gmr_ilp / iomr_ilp | `results_real/` | 17 h/task cap | ⏳ once-ever³ |

¹ The geometric-small ILP ran under the **old 180 s cap** (later tightened to 45 s), so its ILP wall — and the
454 core-h total — is inflated; the true drivers there were `l1sep_iomr` (38%) and `gmr_lp_rsp` (27%), not ILP.
² Being re-run on commit `bd429b7` (the `run_isolated` pipe-deadlock fix — see §3 note). The first attempt
truncated to ~286/589 CSVs; the big-H graphs (ripe/flycns/bct/…) will now return real covers.
³ ILP array is independent and capped at 17 h/graph; "we only need to run this once, ever."

**Where each campaign is specified in full:** geometric → `EXPERIMENTS.md`; RGG → `RGG_EXPERIMENTS.md`
(sweeps S1–S6 / P2*); real → `REAL_EXPERIMENTS.md`. Cluster runbook → `RUN.md`.

### Headline findings (so the next design is informed)

- **Compute is dominated by ILP + the separation-oracle LPs.** RGG-full: ILP = **47%** (`iomr_ilp` 44%).
  Geometric-small: `l1sep_iomr` 38% + `gmr_lp_rsp` 27% + rsp 4% ≫ ILP 19%. Neither ILP nor the `rsp`/`l1sep`
  methods survive to large n → the large-scale suite drops them (§3).
- **ILP convergence:** `gmr_ilp` closes to ~n=1000 (GMR); `iomr_ilp` converges on **sparse RGG at small n**
  but **never on the geometric grid, even at n=100** (64% of geometric `iomr_ilp` runs hit the cap) → for
  IOMR at scale there is no exact optimum, only the LP lower bound and `ratio_domr = size/|H|`.
- **The clean, scale-free metric is `ratio_domr = size/|H|`** (needs no ILP; consistent across n). Use it as
  the primary axis at large n; keep `size/OPT` only where the ILP actually converges (small n).
- **Regime findings to re-test at scale:** on **inflate** GMR-OPT = |H| (`ratio_domr`→1) and IOMR-OPT ≈ 2.2·|H|;
  on **jitter** GMR can beat DOMR (`ratio_domr` 0.9). `pivot` (GMR) and `left_edge` (IOMR) — the "fix an
  arbitrary edge per broken cycle" heuristics — **blow up on jitter** (51× / 59×). `l1sep` is **bimodal** at
  higher break density (median oscillates 1.0↔2.4). kNN/triplet **repair gain is ≤0** on mild jitter.

---

## 2. Scale-tiered algorithm suite (from the measured cost table)

Grounded in `STATUS.md §3` + this session's runtime fit (`log cpu ~ size_exp·log n`) and the real-graph runs
at n=1000–5000. **The n ≤ 3000 design uses the "scale suite" column.**

| algorithm | variant | n-scaling (measured) | keep at n ≤ 3000? |
|---|---|---|---|
| `domr` | DOMR (ref \|H\|) | ~n^1.2, one APSP | ✅ always (cheap; defines \|H\|) |
| `gmr_lp_naive`, `iomr_lp_naive` | LP lower bounds | mild | ✅ always (the references) |
| `spc_gmr`, `spc_iomr` | shortest-path cover | ~n^1.2–1.5 | ✅ always |
| `left_edge` | IOMR (Gilbert–Jain) | ~n^1.8 | ✅ always |
| `pivot` | GMR (MVD pivot) | ~n^2.3 → ~2 min @3000 | ✅ (heaviest cheap one) |
| `gmr_thr_naive`/`bestofk`/`rand`, `iomr_thr_naive`/`bestofk`/`rand` | covering-LP rounding | weight-dominated; ~20 s @500 → mins @3000 | ✅ (watch memory: `bestofk` = k roundings) |
| `iomr_regiongrow` | IOMR region-grow | O(V·E)/pair | ✅ but region-gated (skips when \|H\|>200) |
| `l1sep_gmr`, `l1sep_iomr` | L1 cutting-plane | expensive, ~6 s @500 | ⚠️ keep but **cap** (was 38% of geo compute) |
| `gmr_ilp`, `iomr_ilp` | exact | NP-hard; `iomr` rarely converges | ⚠️ **gate to n ≤ 800** + 60 s cap (exact ref where it closes; auto-skip above) |
| `gmr_lp_rsp`, `iomr_lp_rsp`, `iomr_thr_rsp` | rsp weight-budget | **O(w_max·n²)** | ❌ **drop** (already dropped on float RGG) |
| `MVD_Pivot`/`Gilbert_Jain` completion variants | — | O(n³) completion | ❌ not in the suite at scale |

> **Enabling change needed (small):** ILP gating requires setting `n_max=800` on `gmr_ilp`/`iomr_ilp` (today
> only the rsp methods carry an `n_max`). A tighter `l1sep` cap and the ILP `n_max` are the only harness edits
> the large-scale suite needs — the fork/budget/region-gate machinery already exists.

**Memory:** the APSP dict-of-dicts is O(n²): ~1 GB at n=3000 (fine), but `bestofk`/covering-LP structures
pushed RGG-full to **8 GB even at n=500**, so budget **16–24 GB/task** at n=3000 and **probe first** (§4.3).
n=3000 is the ceiling *without* the `all_pairs_distances` matrix refactor (deferred); n≈5000 would OOM the dict.

---

## 3. Large-scale design (n = 1000…3000)  — PLANNED, not yet run

A **dedicated large-n campaign**, not an extension of the small grid. **No ILP at all** (excluded outright —
it can't converge at this scale), so the suite is **heuristics + LP-bound references + `domr`**; `rsp` dropped
throughout. Primary axes: **`ratio_domr = size/|H|` (quality) and wall-time (cost) vs n**. Size ladder:

```
n ∈ {1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000}   # step 200 → 11 points
```

### 3.1 The grids

| campaign | model / weights | sweep | fixed | measures |
|---|---|---|---|---|
| **RGG size / P1** | RGG float, `radius`, deg=12, dim=2 | `n` ladder | inflate, frac_q=.10, mag=3 | `ratio_domr`, edit prec/recall vs n |
| **RGG size / P2** | " | `n` ladder | jitter nj=8, jitter=2·r, s=.5 | kNN recovery (lift, triplet) vs n |
| **RGG density (radius)** | RGG float, `radius` | mean deg ∈ {4,8,12,20,30,40} | **n=2000**, inflate | `ratio_domr` vs density |
| **RGG density (knn)** | RGG float, **`knn`** | k ∈ {8,12,20,30} | **n=2000**, inflate | knn-topology density variant |
| **GEO / exp1** | coupled geometric, int | `n` ladder | **p ∈ {0.3, 0.5, 0.8}** | `ratio_domr` vs n, planted breaks |
| **GEO / exp2** | decoupled geometric, int | **α: 4/5 → 1/3** (`p = 2·n^−α`) | **n=1500** | density onset + algo separation |

**exp2 detail:** `p = 2·n^{−α}`, α **decreasing** from 4/5 to 1/3 over ~16 points (linspace). The ×2 keeps the
graph connected further into the sweep, and lowering α densifies it: at n=1500 this runs **p ≈ 0.006 → 0.175**
(mean degree ≈ 9 → ≈ 260) — the dense end has many more broken cycles, to expose algorithm separation.

**Config count:** 11 (RGG-P1) + 11 (RGG-P2) + 6 (RGG deg) + 4 (RGG knn) + 33 (geo exp1: 3 p × 11 n) +
16 (geo exp2) = **81 configs**.

### 3.2 Suite, seeds, cost, memory, outputs

- **Suite:** the scale suite (§2) **minus the two ILP entries entirely** → RGG ~13 algos, geometric ~13
  (rsp also dropped). References = `domr` (|H|), `gmr_lp_naive`, `iomr_lp_naive`.
- **Seeds: 20 (flat)** — all points are large, so no tapering; drop to 15 if the probe says it's pricey.
- **Cost:** ≈ **140–200 core-h** (up from the pure-ladder estimate: exp1 is now 3 densities and exp2's
  **dense end — n=1500, p≈0.175, ~200k edges — is the single heaviest slice**: APSP + covering-LP on that many
  edges is minutes/instance). *Extrapolated; the probe (§4.3) — which must include the exp2 dense end — sets
  the real number.*
- **Memory: 16 GB/task**, bump to **24–32 GB** if the probe shows the dense exp2 / `bestofk` blowing up.
  `day` partition, 1 core/task, `--max-jobs 64`, walltime **04:00:00**.
- **Outputs:** `results_rgg_large/` and `results_geo_large/` (+ `analysis/summary_*_large.csv`,
  `analysis/figs/{rgg,geometric}_large/`); add both dirs to `.gitignore` like the others.

---

## 4. Before launching

### 4.1 Harness edits required (small, scoped)
1. **`rgg_harness.py` `large` grid:** `LARGE_NS = range(1000, 3001, 200)` size sweep (P1 inflate + P2 jitter),
   plus density at n=2000 in **both** `radius` (deg∈{4,8,12,20,30,40}) and `knn` (k∈{8,12,20,30}) modes.
2. **`harness.py` `_points_large`:** exp1 coupled-geometric over the ladder × **p∈{.3,.5,.8}**; exp2
   decoupled-geometric at **n=1500** with **p=2·n^{−α}**, α = `linspace(4/5, 1/3, 16)`.
3. **Exclude `gmr_ilp`/`iomr_ilp` outright** from the large suite; `rsp` dropped (float RGG + geometric here).
4. New runners/dirs: `--grid large` → `results_rgg_large/` / `results_geo_large/`; `.gitignore` entries.

### 4.2 Open decisions to confirm (flag any before I wire it)
- **11 points** (keep n=3000) or **10** (drop 3000)?
- **20 seeds** (flat) or 15, given the heavier cost (exp1×3 densities + the dense exp2 tail)?
- exp2: **16 α-points** across 4/5→1/3 ok, and confirm the **×2** factor on `p` (keeps it connected + denser)?

### 4.3 Probe first (per the STATUS §3 practice)
Run **one instance** at n ∈ {1000, 2000, 3000} through the scale suite, logging per-algo wall + `peak_mb`,
before committing cluster hours. This replaces the §3.2 estimate with measured numbers and sets the true
memory request. (The real-graph runs already give n=1000–5000 timings for `domr` and several heuristics, but
not the RGG-sparse covering-LP family at n ≥ 1500.)
