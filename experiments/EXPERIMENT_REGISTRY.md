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
| 3 | **RGG full** | RGG float (`grid=full`) | size sweeps 100–500 (step 20); OFAT baseline n=300 | now **125 × 40 = 5000** (+P2df/P2dm) | 18 (rsp dropped) | `results_rgg/` → `results_rgg_full_all.csv` | ~55–75 core-h | ⟳ re-run: adds P2df/P2dm kNN-recovery, `light_frac`, and FIXES the old no-op deflate (S3d/S4d) |
| 4 | **Real datasets — heur** | 19 real graphs | n=75–5000 (per graph) | 19×(1 det + 30 rand) = **589** | 18 (rsp dropped) | `results_real/` (+ `results_real_covers/`) | ~4h array setup | 🔄 re-running² |
| 5 | **Real datasets — ILP** | 16 dist-sensible graphs | same graphs | 16 × 2 = **32** | gmr_ilp / iomr_ilp | `results_real/` | 17 h/task cap | ⏳ once-ever³ |

¹ The geometric-small ILP ran under the **old 180 s cap** (later tightened to 45 s), so its ILP wall — and the
454 core-h total — is inflated; the true drivers there were `l1sep_iomr` (38%) and `gmr_lp_rsp` (27%), not ILP.
² Being re-run on commit `bd429b7` (the `run_isolated` pipe-deadlock fix — see §3 note). The first attempt
truncated to ~286/589 CSVs; the big-H graphs (ripe/flycns/bct/…) will now return real covers.
³ ILP array is independent and capped at 17 h/graph; "we only need to run this once, ever."

**Where each campaign is specified in full:** geometric → `EXPERIMENTS.md`; RGG → `RGG_EXPERIMENTS.md`
(sweeps S1–S6 / P2*); real → `REAL_EXPERIMENTS.md`. Cluster runbook → `RUN.md`.

### Campaign map — what each experiment checks, and which algorithms it drops

Full suite = **21**: refs `domr`, `gmr_lp_naive`/`iomr_lp_naive`; exact `gmr_ilp`/`iomr_ilp`; covering-LP rounding
`{gmr,iomr}_{thr_naive,bestofk,rand}`; `iomr_regiongrow`; `l1sep_{gmr,iomr}`; `spc_{gmr,iomr}`; `pivot`;
`left_edge`; and 3 weight-budget `*_rsp`. Standing drops: **`*_rsp`** (need integer weights, cost O(w_max·n²) —
dropped on every FLOAT graph and at large scale); **`gmr_ilp`/`iomr_ilp`** (NP-hard — kept only where they can
converge, i.e. small n; dropped entirely at n≥1000); **`iomr_regiongrow`** never dropped but auto-skips when |H|>200.

| graph model — experiment | what it checks | suite (drops) |
|---|---|---|
| **Geometric small** — exp1 inflate (n≤300) | approx ratio (size/OPT) & runtime vs n; GMR exact = OPT here | **21** (integer → rsp+ILP kept) |
| ” — exp2a onset (n=300) | non-metricity **onset** vs edge density (α≈3/5) | 21 |
| ” — exp2b density (n=300) | ratio vs edge density on decoupled geometric | 21 |
| **Geometric large** — exp1 inflate (p{.3,.5}, n=1000–1500) | ratio_domr & `light_frac` vs n on dense geometric | **16** (−rsp −ILP) |
| ” — exp2 density onset (n=2000, α 4/5→1/3) | algo **separation** as graphs densify / break more | 16 |
| **RGG poc** — size (n≤250, inflate+jitter) | fast smoke: edit prec/recall, ratio_domr, kNN | **18** (−rsp) |
| **RGG full** — S1 inflate size | ratio_domr / edit-precision-recall vs n | 18 |
| ” — S2 / S2k density (radius / knn) | density & topology effect | 18 |
| ” — S3/S3d, S4i/S4d inflate & **deflate** mag/frac | corruption severity & direction (deflate now fixed) | 18 |
| ” — S5a/b/c jitter | edit metrics under sensor-drift jitter | 18 |
| ” — P2size/s/j/n jitter kNN | kNN neighborhood recovery under **jitter** | 18 (LP-bounds have no cover → out of kNN) |
| ” — **P2df / P2dm deflate kNN** (new) | **does repair restore kNN under shortcuts** + variant split | 18 |
| **RGG large** — S1 / S1d inflate / **deflate** size | ratio_domr & `light_frac` vs n→3000; GMR ≠ DOMR | **16** (−rsp −ILP) |
| ” — S2 / S2k density (n=2000) | density & topology at scale | 16 |
| ” — P2size jitter kNN vs n | kNN recovery scaling | 16 |
| ” — **P2df / P2dm deflate kNN** (n=1000) | **where repair helps kNN** + variant dissociation | 16 |
| **Real — heur** (19 graphs) | ratio_domr / edit / `light_frac` on real data | **18** (−rsp) |
| **Real — ILP** (16 graphs, 17 h) | exact GMR/IOMR reference where it converges | **2** (the ILPs only) |

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

> **Implemented (commit adds a `large` grid):** the large suite drops `gmr_ilp`/`iomr_ilp` **and** the rsp
> methods via a **per-grid filter** — `DROP_LARGE` in `harness.py`, `drop_ilp=True` in `build_suite_rgg` — NOT a
> global `n_max=800`, which would also disable the **real-data ILP array** (its graphs have giant > 800). The
> existing fork/budget/region-gate machinery is reused unchanged.

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

| campaign | sweep-id | model / weights | sweep | fixed | measures |
|---|---|---|---|---|---|
| **RGG size / P1 inflate** | S1 | RGG float, `radius`, deg=12, dim=2 | `n` ladder | inflate, frac_q=.10, mag=3 | `ratio_domr`, `light_frac`, edit vs n |
| **RGG size / P1 deflate** | S1d | " | `n` ladder | **deflate**, frac_q=.10, mag=3 | `light_frac`, `ratio_domr` (GMR ≠ DOMR) |
| **RGG size / P2 jitter** | P2size | " | `n` ladder | jitter nj=8, jitter=2·r, s=.5 | kNN recovery (lift, triplet) vs n |
| **RGG density (radius)** | S2 | RGG float, `radius` | deg ∈ {4,8,12,20,30,40} | **n=2000**, inflate | `ratio_domr` vs density |
| **RGG density (knn)** | S2k | RGG float, **`knn`** | k ∈ {8,12,20,30} | **n=2000**, inflate | knn-topology density variant |
| **RGG kNN recovery** ★ | P2df, P2dm | RGG float, `radius` | **deflate** frac_q ∈ {.02..0.3} · mag ∈ {2..10} | **n=1000** | **kNN lift by variant** — where repair HELPS |
| **GEO / exp1** | exp1 | coupled geometric, int | **10-pt mesh n=1000…1500** | **p ∈ {0.3, 0.5}** | `ratio_domr`, `light_frac` vs n |
| **GEO / exp2** | exp2b | decoupled geometric, int | **α: 4/5→1/3** (16 pts, `p=2·n^−α`) | **n=2000** | density onset + algo separation |

★ **The kNN-recovery arm is the "does repair help downstream?" experiment.** kNN/triplet run on shortest-path
distance, so a too-long (inflate) edge is invisible to them and repairing it can't change kNN — that's why every
earlier lift was ≤0 (we only ran inflate + mild jitter). SHORTCUT (deflate) corruption is the regime where
repair helps: measured **exact-GMR lift +0.41** at severe deflate (n=300), and a clean **variant dissociation** —
GMR/IOMR recover (they raise/drop the light shortcut) while **DOMR gives exactly 0** (decrease-only touches only
the heavy victims, off the shortest paths). `light_frac` also predicts recovery quality (low-light covers can hurt).

**exp2 detail:** `p = 2·n^{−α}`, α **decreasing** from 4/5 to 1/3 (16 points) at **fixed n=2000**. The ×2 keeps
it connected further in; lowering α densifies it (**p ≈ 0.005 → 0.16**, ~9k → ~317k edges). Only the dense end
(α≈1/3) is heavy — heuristics fall back to the LP bound there, but `domr` (≈9 s) and `ratio_domr` still land.

**GEO exp1 is capped at n ≤ 1500** (10-point mesh 1000→1500) because its `p` is an edge probability (dense):
edges ≈ p·n²/2, so the heaviest config is p=0.5,n=1500 ≈ **562k edges** (≈ ripe-scale, tractable); n=2000,p=0.5
would be ~1M and was dropped (even `domr` times out there — probe finding).

**Config count:** RGG-large = 11 (S1) + 11 (S1d) + 11 (P2size) + 6 (S2) + 4 (S2k) + 5 (P2df) + 4 (P2dm) =
**52 configs / 1040 tasks**. GEO-large = 20 (exp1: 2 p × 10 n) + 16 (exp2b α @ n=2000) = **36 configs / 720
tasks**. Total **88 configs / 1760 tasks** @ 20 seeds.

> **Probe-measured cost/memory** (per instance, at the heavy end): RGG scales cleanly to **n=3000** (~890 MB
> peak, `domr` ~3 s; the P2/kNN pass ~17 min). GEO edge counts (≈ p·n²/2) are the constraint — exp1 capped at
> **n≤1500** (heaviest p=0.5,n=1500 ≈ 562k edges; the dropped n=3000,p=0.5 was ~2.25M where even `domr` times
> out), exp2's dense corner (n=2000, α≈1/3 ≈ 317k edges, `domr` ~9 s) heuristics fall back to the LP bound.
> Request **16 GB / 4 h**; the dense GEO tasks are the slow ones (per-task budget caps them so the CSV lands).

### 3.2 Suite, seeds, cost, memory, outputs

- **Suite:** the scale suite (§2) **minus the two ILP entries entirely** → RGG ~13 algos, geometric ~13
  (rsp also dropped). References = `domr` (|H|), `gmr_lp_naive`, `iomr_lp_naive`.
- **Seeds: 20 (flat)** — all points are large, so no tapering; drop to 15 if the probe says it's pricey.
- **Cost:** ≈ **150–220 core-h** (exp1 is 3 densities; exp2's **dense end — n=2000, p≈0.16, ~320k edges (near
  ripe_atlas) — is the single heaviest slice**: APSP + covering-LP on that many edges is minutes/instance and
  some heuristics will hit the cap). *Extrapolated; the probe (§4.3) — which must include the exp2 dense end —
  sets the real number.*
- **Memory: 16 GB/task**, bump to **24–32 GB** if the probe shows the dense exp2 / `bestofk` blowing up.
  `day` partition, 1 core/task, `--max-jobs 64`, walltime **04:00:00**.
- **Outputs:** `results_rgg_large/` and `results_large/` (geometric) (+ `analysis/summary_*_large.csv`,
  `analysis/figs/{rgg,geometric}_large/`); add both dirs to `.gitignore` like the others.

---

## 4. Before launching

### 4.1 Harness wiring — DONE
Both grids are wired and verified to enumerate: **RGG-large = 52 configs / 1040 tasks** (S1 + S1d + P2size size
ladders, S2 + S2k density at n=2000, P2df + P2dm deflate-kNN); **GEO-large = 36 configs / 720 tasks** (exp1
2 p × 10 n capped at n≤1500; exp2b 16 α at n=2000). These match §3.1 and the live `--count` output. Suite =
**16 algos** (no ILP, no rsp) in both, via the per-grid `DROP_LARGE` / `drop_ilp` filter. `--grid large` on both
runners; `submit_{dsq,rgg_dsq}.sh large` request **16 GB / 4 h**; outputs gitignored. **Current knobs:** the
exp1 mesh `np.linspace(1000, 1500, 10)` in `harness._points_large` (NOT the dead `LARGE_NS`), 20 seeds
(`SAMPLES/SAMPLE_COUNT['large']`), 16 α-points — edit those to change.

> Earlier revisions of this section claimed 640 / 980 tasks. That predated the exp1 mesh change (3 p × 11 n →
> 2 p × 10 n, capped at n≤1500 by the probe) and the RGG deflate/kNN additions. Trust `--count`, not prose.

### 4.2 Cluster run (after the probe)
```bash
git pull && module load miniconda && conda activate metricrepair
python experiments/run_rgg_task.py --grid large --count      # -> 1040
python experiments/run_task.py     --grid large --count      # -> 720
bash experiments/submit_rgg_dsq.sh metricrepair <PI_netid> large   # -> results_rgg_large/ (16g, 4h)
bash experiments/submit_dsq.sh     metricrepair <PI_netid> large   # -> results_large/
sbatch dsq_rgg_large_submit.sh ; sbatch dsq_large_submit.sh
# collect + reuse the existing analysis on the new dirs:
python experiments/collect.py --indir results_rgg_large --out results_rgg_large_all.csv
python experiments/collect.py --indir results_large     --out results_large_all.csv
```

### 4.3 Probe first (per the STATUS §3 practice)
Measure the heaviest instances — RGG P1/P2 at n=3000 and the **dense exp2 (n=2000, ~320k edges)** — logging
per-algo wall + `peak_mb`, before committing cluster hours; this replaces the §3.2 estimate and sets the true
memory request. (Results folded back into §3.2 once measured.)
