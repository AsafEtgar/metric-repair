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

## 3. Large-scale design (n ≤ 3000)  — PLANNED, not yet run

Two campaigns, same **scale suite** (§2): references + all heuristics; `rsp` dropped; ILP gated to n ≤ 800.
Primary metric **`ratio_domr = size/|H|`** throughout (ILP `size/OPT` only where it converges). Float weights
on RGG (clean planted set); integer weights on geometric (planted breaks / density onset).

### 3.1 RGG-LARGE (float, `mode=radius`, `dim=2`, deg=12 baseline)

| sweep | knob | values | fixed | part |
|---|---|---|---|---|
| **LS-size** | `n` | 100, 300, 500, 800, 1200, 1800, 2400, **3000** | deg=12, inflate frac=.1 mag=3 | P1 (edit + ratio) |
| **LS-size-knn** | `n` (same ladder) | " | jitter nj=8, jr=2·r, s=.5 | P2 (kNN recovery) |
| **LS-density** | mean degree | 4, 8, 12, 20, 30, 40 | **n=1500**, inflate | P1 |
| **LS-frac** | `frac_q` | .01, .05, .10, .20, .30 | n=1500, inflate mag=3 | P1 |
| **LS-mag** | `magnitude` | 1.5, 3, 5, 10 | n=1500, inflate frac=.1 | P1 |
| **LS-jitter** | `n_jitter`×`jitter`×`subset_s` | nj∈{2,8,32}, jr∈{1,2,4}·r, s∈{.25,.5,.9} (OFAT, 3 sub-sweeps ≈ 9) | n=1500 | P2 |

Overlaps the RGG-full ladder at n∈{100,300,500} so the curves **connect** to existing data, then extend to
3000. `~29 configs` (8+8 size + 6+5+4 OFAT + 9 jitter). Also worth a **`mode=knn` density variant** at n=1500
(k∈{8,12,20,30}) — optional.

### 3.2 GEO-LARGE (integer weights)

| sweep | model | knob | values | fixed |
|---|---|---|---|---|
| **LG-size** | coupled geometric | `n` (exp1) | 100, 300, 500, 800, 1200, 1800, 2400, **3000** | p ∈ {0.3, 0.5} |
| **LG-density@scale** | decoupled geometric | `alpha` (exp2b) | linspace(0.5, 0.8, 15) → p = n^−α (crosses connectivity ~0.706) | **n=1500** |

`LG-size` = 2 p × 8 n = 16 configs; `LG-density@scale` = 15 configs. (exp2a onset is an n=500 story — leave as
run; the scaling question is exp1 + the exp2b density sweep at a larger fixed n.)

### 3.3 Seeds, cost, memory, outputs

- **Tapered seeds** (large-n instances are pricier and less variable): **30** at n ≤ 500, **20** at 800–1200,
  **12** at n ≥ 1800. A single knob in the harness; total is dominated by the large-n tail either way.
- **Cost estimate:** RGG-LARGE ≈ **45–70 core-h**, GEO-LARGE ≈ **40–60 core-h** → **~100–130 core-h** total,
  comparable to RGG-full. *This is an extrapolation from n ≤ 500 — treat as provisional until the probe (§4.3).*
- **Memory:** request **16 GB/task** (bump to 24–32 GB if the probe shows `bestofk`/covering-LP blowing up at
  n ≥ 2000). `day` partition, 1 core/task, `--max-jobs 64`; per-task walltime **04:00:00** (matches the real
  heur cap; the n=3000 suite is ~10–15 min/instance).
- **Outputs (proposed):** `results_rgg_large/` and `results_geo_large/` (+ `analysis/summary_*_large.csv`,
  `analysis/figs/{rgg,geometric}_large/`). Add the dirs to `.gitignore` like the others.

---

## 4. Before launching

### 4.1 Harness edits required (small, scoped)
1. Add a **`large` grid** to `rgg_harness.py` (the LS-* sweeps; extend `SIZE_NS`, add a large OFAT baseline)
   and to `harness.py` (`_points_large`: exp1 ladder + exp2b @ n=1500).
2. **Gate ILP:** `n_max=800` on `gmr_ilp`/`iomr_ilp` in `build_suite` (or a large-grid override), 60 s cap.
3. Optional: tighten the `l1sep` per-algo cap; wire the tapered-seed map.

### 4.2 Open decisions to confirm (flag any before I wire it)
- n ladder `{100,300,500,800,1200,1800,2400,3000}` and the **n=1500 OFAT baseline** — OK?
- **Tapered seeds** (30/20/12) vs a flat count?
- Keep `gmr_ilp` as a small-n exact GMR reference (gated ≤800), or drop ILP entirely and rely on
  `ratio_domr` + LP bounds throughout?
- Include the `mode=knn` density variant and the GEO-LARGE exp2b-at-n=1500, or keep the first pass minimal?

### 4.3 Probe first (per the STATUS §3 practice)
Run **one instance** at n ∈ {1000, 2000, 3000} through the scale suite, logging per-algo wall + `peak_mb`,
before committing cluster hours. This replaces the §3.3 estimates with measured numbers and sets the true
memory request. (The real-graph runs already give n=1000–5000 timings for `domr` and several heuristics, but
not the RGG-sparse covering-LP family at n ≥ 1500.)
