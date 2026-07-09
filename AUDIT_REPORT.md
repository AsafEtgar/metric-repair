# Code audit — metric repair
**Date:** 2026-07-09 · **Scope:** all first-party code (`data/raw/fish1/**` is vendored, excluded).
**Method:** six parallel read-only audits (exact/LP/ILP core; heuristics; graph models; synthetic harnesses;
real pipeline; analysis/plots), each finding then independently verified or refuted by direct experiment.

**No bugs were fixed.** Nothing in this report has been applied to the code.

## UPDATE — fix pass, 2026-07-09

Applied (all verified; the fixed pipeline reproduces the paper's Table 3 to two decimals):

| id | fix | file |
|---|---|---|
| **A2** | reference gate: an ILP is `exact` only if `converged` **and** `status=="ok"` **and** `valid==1` | `real_analyze.py`, `analyze.py`, `rgg_analyze.py` |
| **A3** | invalid covers (`valid==0`) dropped before aggregation, and the drop is printed | all three analyzers |
| **A5** | `except (EOFError, OSError)` — a child killed mid-send no longer destroys the task | all three harnesses |
| **A7** | `n_ok` (finite-sample count) added beside `n_samples`; skip/timeout counts surfaced | `analyze.py`, `rgg_analyze.py` |
| **A8** | heatmap reddens `lower_bound` **and** `none` columns; annotation contrast fixed under `LogNorm` | `real_plots.py` |
| **A9** | `collect.py` aborts on orphan indices (`--grid`), stale mtimes (`--max-age-days`), and ragged schemas | `collect.py` |
| **new** | `_aggregate` nulls `size/valid/converged/exact_opt` when `status != "ok"` — a partial multi-component result no longer masquerades as complete | `harness.py` |
| **new** | `analyze.aggregate` no longer hard-requires `light_frac` | `analyze.py` |

**A6 is REFUTED — no re-run needed.** `time_limit` is never passed to the ILP; `ALGO_TIMEOUT` is
`run_isolated`'s wall cap, enforced by killing the child. So `milp` always runs to optimality and
`res.status` never signals a time limit. A6 is latent, reachable only if someone passes `time_limit=`.

**What the A6 probe found instead (real, active, now fixed):** `_aggregate` set `status = worst(components)`
while ANDing `converged` over only the components that *reported*, and summing `size`/`exact_opt` over those
same few. A timed-out component reports nothing, so a partially-solved graph emitted `converged=True` with a
**partial optimum**. Measured: **27 ILP rows** in `results_rgg_full_all.csv` carry `status=timeout` **and**
`converged=True`. Verified after the fix: all 26 affected tasks now resolve to `lower_bound`, while 200/200
healthy tasks still resolve to `exact` (no over-rejection).

**Also newly found:** `analyze.aggregate` hard-required `light_frac`, a column the geometric-small campaign
predates — so the analyzer **could not reprocess `results_small_all.csv`, the source of the paper's own
geometric table.** Now fixed.

**Still NOT fixed, deliberately:**
* **A1** — the root cause. Fixing `_apsp_positions` changes algorithm output, so it must not land mid-campaign.
  Apply it after the current jobs drain, then re-run the ILP + LP-rounding family on the four sub-`1e-8` graphs.
* **A4** — `l1_separation` still returns silently at `max_rounds`. Harmless now that A3 filters its invalid
  covers, but it should raise or return a converged flag.
* All **B-series** latent findings.

---

## How to read this

Every finding carries a **status**, which matters more than its severity:

| status | meaning |
|---|---|
| **ACTIVE** | affects data we already have, or data the next run will produce |
| **LATENT** | a real bug with *verified zero* current exposure — it will bite on the next parameter change |
| **REFUTED** | an audit flagged it; direct experiment showed it is not real (recorded so nobody re-finds it) |

Agents proposed findings; the experiments below decided them. Two of the scariest proposals were latent, and
one confident diagnosis was **wrong in a way that would have corrupted the paper** had it been applied.

---

## 0. Headline

**A1 is the finding that matters.** The separation oracle silently deletes every graph edge whose weight is
`<= 1e-8`, because `_apsp_positions` hands a dense `np.zeros` matrix to SciPy, and SciPy's dense→graph
conversion masks near-zero entries as *non-edges* (`np.ma.masked_values`, default `atol=1e-8`). The `_lin`/`_log`
similarity inversions add exactly one edge of weight `EPS = 1e-9`. That single edge is dropped, the oracle
cannot see detours through it, it reports `converged=True`, and the returned "exact optimum" fails the verifier.

This is **not** the "structural incompleteness of the float-weight oracle" we wrote into the paper. It is a
fixable library gotcha. The three excluded real graphs are **recoverable**, and §5 lists the paper text that
must change.

---

## 1. ACTIVE findings — these affect results

### A1 — `_apsp_positions` deletes edges with weight `<= 1e-8` · HIGH
`metric_repair.py:_apsp_positions` (dense `A = np.zeros(...)` → `shortest_path(A, method="FW")`)

The oracle's graph is missing every near-zero edge. Consequences:

* `exact_metric_repair_ilp_separation` returns `converged=True` with a cover the verifier rejects.
* Everything routed through the same oracle inherits it: the covering-LP feasibility check and top-up, hence
  `*_thr_naive`, `*_bestofk`, `*_rand`, `covering_lp_cover(solve="separation")`. This is why *all 30 seeds* of
  `gmr_rand`/`iomr_rand` are invalid on the affected graphs, not just the ILP.
* `verifier` is unaffected (networkx APSP keeps the edge), which is why the two disagree.

**Evidence.** Threshold measured exactly: a 2-hop detour of weight `1e-7` is found; `1e-8`, `9e-9`, `1e-9`,
`3.36e-10` are all dropped. Reproduced end-to-end on `bct_coactivation_lin`: the ILP returns `|S|=87`,
`converged=True` (matching the recorded cluster run exactly), `verifier=0`, with 19 residual violations whose
gaps run `3.9e-3 … 8.8e-2`. Feeding that final cover back into `_violated_cuts` yields **0 cuts** — the oracle
is blind, not merely premature. Tightening the oracle tolerance to `1e-9` changes nothing (identical `|S|`,
identical residuals) on all three graphs.

**Predictive check** (this is what confirms the mechanism): the graphs holding an edge `< 1e-8` are exactly
`bct_coactivation_lin`, `bct_coactivation_log`, `flycns_male_log`, `fish1_ten_log`. The first three are exactly
the graphs with invalid covers. `fish1_ten_log` escapes only because `|H| = 4` — no cut ever needs that edge.

| graph | \|E\| | \|H\| | residual violations | % of \|H\| | max gap |
|---|---:|---:|---:|---:|---:|
| `bct_coactivation_lin` | 18625 | 210 | 19 | 9.0% | 0.088 |
| `bct_coactivation_log` | 18625 | 3273 | 17 | 0.5% | 0.897 |
| `flycns_male_log` | 14025 | 1761 | 56 | 3.2% | 3.95 |

**Fix (not applied).** Build the matrix with `np.inf` for non-edges, or use
`csgraph_from_dense(A, null_value=np.inf)`, or construct a sparse CSR directly. Then re-run the ILP + LP-rounding
family on the four affected graphs and un-exclude them.

**Do not** "fix" this by loosening the verifier — that was tried before and correctly reverted. The verifier is
the correct party here.

---

### A2 — a converged-but-invalid ILP is accepted as the exact optimum · HIGH
`experiments/real_analyze.py:53-60` (`_graph_refs`), same pattern at `experiments/analyze.py:80,88`

`_graph_refs` sets `ref_kind="exact"` on `val("gmr_ilp","converged") is True` and never checks `valid == 1`.
On the three A1 graphs the ILP's *insufficient* cover becomes the denominator of `ratio = size/ref` for every
algorithm on that graph.

**The published Table 3 is correct** — recomputing with those graphs excluded reproduces the paper's numbers to
two decimals for all twelve algorithms (`pivot` 9.75, `left_edge` 70.70, `domr` 1.72, …). The exclusion was
applied when the table was built. **But that filter exists nowhere in the code.**

What `real_analyze.py` produces *today*:

| algo | paper | code today | corrected |
|---|---:|---:|---:|
| `gmr_bestofk` | 1.28 | **30.58** | 1.28 |
| `domr` | 1.72 | **2.41** | 1.72 |
| `spc_gmr` | 4.42 | **4.94** | 4.42 |
| `gmr_thr_naive` | 1.29 | **1.49** | 1.29 |

**This is a live trap.** Regenerating `summary_real.csv` after the cluster jobs land — which is step 2 of the
plan — would silently destroy the "general repair beats decrease-only" claim in §4. **A2 must be fixed before
any regeneration.**

Fix: require `valid == 1` as well as `converged` before treating an ILP row as the exact reference.

---

### A3 — invalid covers are printed, never filtered, in all three analyzers · HIGH
`real_analyze.py:119-122` (prints; `aggregate()` at :134 runs on the full frame; `to_csv` at :139 writes it)
`analyze.py:52-70` (same shape) · `rgg_analyze.py:142-177` (neither filters **nor reports**, and has no
`timeout_rate` column at all)

An invalid cover is an *insufficient* one, so its `|S|` is understated and every quality metric derived from it
is biased **downward — the algorithm looks better than it is.**

Currently flowing into the summaries and figures: 198 invalid rows (real), 329 (geometric: 306 expected
`gmr_lp_rsp` LP-gap + **23 genuine `l1sep_iomr`**, see A4), 5 invalid IOMR covers + 202 unflagged timeouts (RGG).

On the affected real graphs the groups are **100% invalid** (`gmr_rand` 30/30, `iomr_rand` 30/30), so correct
filtering would blank those cells rather than shrink them. Today they display real-looking numbers.

---

### A4 — `l1_separation` silently returns an unconverged cover at `max_rounds` · MEDIUM-HIGH
`metric_repair.py:588-659` (loop bounded at `max_rounds=200`, returns a bare set)
called as `(l1_separation(CC, ...), {})` at `experiments/harness.py:200-201` — the only repair entry point that
returns no `info`/converged signal, so nothing records `rounds` (its `rounds` column is entirely NaN).

**Evidence.** All 23 invalid `l1sep_iomr` rows in the geometric data are `exp1`, all at the *denser* `p=0.5`,
all at the large-`n` end, and every one carries `status="ok"`. Conditioning on the runs that actually completed
makes the trend monotone and terminal:

| n | completed | timed out | invalid among completed |
|---:|---:|---:|---:|
| 220 | 27 | 3 | 0% |
| 240 | 30 | 0 | 10.0% |
| 260 | 27 | 3 | 22.2% |
| 270 | 27 | 3 | 25.9% |
| 280 | 14 | 16 | 14.3% |
| 290 | 1 | 29 | **100%** (1/1) |
| 300 | 0 | 30 | — |

The apparent improvement at `n>=280` is **survivorship**: the 30-minute timeout starts absorbing the hard
instances so their invalid results stop being recorded as `ok`. `l1sep` is reported in the paper as *"within
~4% of optimal"* and consumed 38% of the geometric-small compute.

Fix: return a converged flag (or raise) when the loop exits via `max_rounds`.

---

### A5 — `recv()` catches only `EOFError`; a child killed mid-send loses the **whole task** · HIGH (data loss)
`experiments/harness.py:271` · `experiments/rgg_harness.py:330` · `experiments/real_harness.py` (same line)

`multiprocessing.Connection` frames each message. Measured directly:

| child dies… | exception | caught? |
|---|---|---|
| before sending anything | `EOFError` | ✅ → recorded as `killed` |
| **mid-body** (large cover, partially sent) | **`OSError`** | ❌ uncaught |
| mid-header | `OSError` | ❌ uncaught |

So the guard catches OOM-during-*compute* but not OOM-during-*send*. The `OSError` propagates out of the
algorithm loop, out of `run_one_task`, and out of `run_task.py` (no handler) → **no CSV**, losing every *other*
algorithm's result on that graph too. It also skips the final `p.join()`.

Exposure is exactly where the current jobs live: large covers on memory-pressured graphs (the `exp2b` dense tail,
the `dimacs_ny_big` realrec bases). **It corrupts nothing — it loses tasks.**

Fix: `except (EOFError, OSError): out = {"status": "killed"}`.

---

### A6 — ILP timeout incumbent is trusted as a proven optimum · MEDIUM-HIGH
`metric_repair.py:1044-1051`

`milp` can return a feasible-but-suboptimal incumbent when `time_limit` fires. Only `res.x is None` is guarded;
`res.status` / `res.success` are never inspected (they appear nowhere in the file). If the next oracle pass finds
no new cut, `converged=True` is returned — violating the docstring's contract that `converged ⇒ proven exact
optimum`. The harness uses a 45 s ILP cap, so this is reachable.

---

### A7 — survivorship bias: medians over survivors, `n_samples` over everything · MEDIUM
`analyze.py:134`, `real_analyze.py:92`, `rgg_analyze.py:148,163`

Timeout / `oom` / `killed` / `skipped_*` rows carry `size=NaN`, so `median()` silently drops them, but
`n_samples=("sample","nunique")` counts them. Worse, `skipped_n` (4200 rows) and `skipped_H` (1317) are gated on
per-draw quantities (`giant`, `total_H`), so they drop the **harder** instances of a config → medians biased
**better**. There is no `skip_rate` or finite-sample count anywhere. A4's table is a concrete instance of the
damage.

---

### A8 — the heatmap does not red-flag invalid-reference columns · MEDIUM
`experiments/real_plots.py:78-80`

A column label is painted red only when `ref_kind == "lower_bound"`. The three A1 graphs are tagged `"exact"`,
so `fig:real-heat` presents them as trustworthy exact-reference columns. The table excludes them; the figure
does not.

---

### A9 — `collect.py` blindly concatenates, with a first-file header · MEDIUM
`experiments/collect.py:16,23-31`

`glob("task_*.csv")` with no grid filter and no dedupe; `header = r.fieldnames` from the *first* file feeds a
`DictWriter` whose default `extrasaction='raise'`. A later file with an extra column crashes collect mid-write
(partial output); a file with fewer columns is silently blank-padded.

**Live exposure:** the RGG `full` grid grew 3160 → 5000 tasks and the re-run writes back into the *same*
`results_rgg/`. Old indices are overwritten only by tasks that *succeed*. **Any task that fails leaves the
previous run's CSV in place** — including pre-deflate-fix rows, whose deflate was a no-op. Before collecting,
mtime-audit `results_rgg/` exactly as we did for `results_real/`.

---

## 2. LATENT findings — real bugs, verified zero current exposure

| id | file | finding | why it doesn't bite (verified) |
|---|---|---|---|
| **B1** | `graph_models.py:247` | `m = int(round(frac_q*|E|))` rounds to **0** → corruption is a silent no-op, `corrupted=∅` | min `frac_q·\|E\|` across *all six grids* = **18** |
| **B2** | `build_real_graphs.py:171-174` | DIMACS directed arcs collapsed onto `nx.Graph` with last-weight-wins | **0 / 365,050** asymmetric arc pairs in both `-d` and `-t`. Corollary: `dimacs_ny_t`'s 0.3% non-metricity is *genuine* — a slow segment with a faster highway detour |
| **B3** | `graph_models.py:251-252,274` | integer deflate can round back up to `gap` → no real edit, **yet the edge is marked corrupted** (poisons planted ground truth). Inflate has a strict-increase floor; deflate has none | our deflate arms run on float RGG only |
| **B4** | `harness.py:366`, `rgg_harness.py:467`, `real_harness.py:225` | per-**component** timeout: one algorithm can burn `k × timeout` before the task budget is ever checked (it is only tested *between* algorithms) | every processed graph has exactly one non-metric component today. *Found independently by two agents.* |
| **B5** | `real_harness.py:215` | the ILP path is explicitly **exempt** from `REAL_TASK_BUDGET_S`; its only cap is the per-component 17 h, against a 17.5 h wall | k=1 today |
| **B6** | `metric_repair.py:396` | `_mvd_pivot_rec` draws pivots from the **global** `np.random`; it has no `seed` parameter, unlike every other randomized method | `seed_all` seeds global numpy and nothing touches it before `pivot`. Reorder the suite and every pivot cover silently changes |
| **B7** | `metric_repair.py:353` | `domr_alg` uses exact float equality `w != apsp[u][v]`; `_heavy_pairs:1196` uses `> apsp + 1e-9` | needs a detour that rounds strictly below `w` |
| **B8** | `metric_repair.py:147` | `verifier` raises `ValueError` on an edge-less graph; `iomr_verifier` returns 1 on the same input | harnesses split into components first |
| **B9** | `metric_repair.py:253,260-261` | `broken_cycle_length_bound`: float-floor can under-estimate by one; `int(inf)` raises `OverflowError` | needs an adversarial `wmax/wmin` near an integer |
| **B10** | `metric_repair.py:334` | `complete()` inserts `inf`-weight edges on a disconnected graph → `NaN` in MVD | harnesses split into components first |
| **B11** | `metric_repair.py:682-691` | `find_shortest_path` loops/wraps on scipy's `-9999` predecessor sentinel | every caller first proves `dist < w` |
| **B12** | `rgg_harness.py:257` | `realrec` loads `data/processed/...` via a **CWD-relative** path; every other path resolves off `__file__` | dSQ runs tasks from the repo root |
| **B13** | `real_harness.py:227` | the per-task budget sums the *child-measured* `wall`, which excludes the post-timer `verify_fn` — and `verifier` runs a full dense APSP | timeouts (parent-measured, accurate) dominate |
| **B14** | `real_harness.py:199-203` | an unbudgeted, untimed DOMR pre-pass runs a full APSP per component, then the suite computes `domr` **again** | eats wall/memory slack silently |
| **B15** | `real_check.py:30` | `EXP_ROWS` is defined and **never used** — no per-file row-count validation, so a truncated CSV counts as present | values are correct (det=11, rand=5, ilp=1) |
| **B16** | `submit_dsq.sh:29`, `submit_rgg_dsq.sh:30` | `mkdir -p "$OUTDIR"` runs *before* the grid name is validated → a typo leaves an empty `results_<typo>/` | argparse catches the typo one step later |
| **B17** | `build_real_graphs.py:233` | `build_flycns` computes `1/s` unguarded; `build_bct` clamps with `1e-12`, `build_fish1_ten` is safe by construction | synapse counts are positive |
| **B18** | `graph_models.py:186-209` | radius/kNN RGG has no connectivity guarantee and silently drops isolated vertices (`n_eff < n`) with no signal | current `deg`/`k` are well above the threshold |
| **B19** | `graph_models.py:245` | the `integer` flag is all-or-nothing; one float weight flips the whole graph to float mode | our graphs are homogeneous |
| **B20** | `metric_repair.py:568,573` | `l1_rounding_heuristic`'s fallback `set(support)` is returned unverified though the docstring calls it "always valid" | not invoked by any experiment |
| **B21** | `metric_repair.py:777` vs `:137` | oracle `tol=1e-6` vs verifier `tol=1e-9`, both **absolute** — a real 1000× mismatch | **zero** blind-band breaks in any real graph, and zero among the A1 residuals. Demonstrable only on a synthetic triangle. See §3 |

---

## 3. REFUTED — proposed, then disproved by experiment

These are recorded so they are not re-discovered and acted on.

* **"The invalid covers are caused by the oracle/verifier tolerance mismatch (B21)."**
  The tolerance mismatch is real (minimal counterexample: triangle `0.001, 0.001, 0.0020005`, gap `5e-7`, ILP
  returns `∅` with `converged=True`). But it explains **none** of the real-data failures: tightening the oracle to
  `1e-9` leaves `|S|`, `converged`, `valid`, and every residual gap *bit-identical* on all three graphs; the
  residual gaps are `1e-3 … 3.95`, up to 88,000× the tolerance; and no real graph has a single break in the blind
  band. The agent generalized from synthetic weight-scale-`1e-6` graphs where B21 dominates. **The true cause is
  A1.** Had this been "fixed" as diagnosed, the bug would have survived and we would have declared it closed.

* **"`pivot`'s 9.8× blowup is an implementation error."** No. The recursion→loop conversion was checked against a
  faithful recursion on 200 randomized trials under identical seeds: **0 mismatches** in both the cover and the
  mutated matrix, with identical RNG draw counts. Termination, state threading, and tie-breaking all preserved.

* **"`Gilbert_Jain_IOMR`'s column-only update `A[viol,k] = cand[viol]` is a symmetry bug."** No — it is correct
  and *load-bearing*. Matched a literal scalar implementation on 150 random broken graphs (0 differences);
  forcing the matrix symmetric **enlarges** covers (4→10 edges). The 71× blowup is the algorithm.

* **"`frac_q` rounding to zero silently corrupted a campaign."** Latent only (B1); min `frac_q·|E|` = 18.

* **"The DIMACS travel-time graph's non-metricity is a parsing artifact."** No — 0/365,050 asymmetric arcs.

Also checked clean, no action: `ratio_domr` guards `H==0` in all three analyzers; `BOUND_ALGOS` are correctly
used as references and excluded from cover aggregates; no Jensen's-inequality error (per-instance ratio, then
median); `"True"/"False"` CSV booleans are remapped everywhere; the `restore` recovery branch computes detours in
`G\S` *before* any reweight, with no in-place mutation while iterating; no seed collision exists today; every
grid's task count matches its documentation; `seed_all` seeds every RNG actually consumed; `DROP_LARGE`/`drop_ilp`
remove rows, not columns, so CSV schemas stay uniform; no bare `except:`; no mutable default arguments.

---

## 4. Documentation debt

**Stale comments that actively mislead:**
* `harness.py:97` — `LARGE_NS = range(1000,3001,200)` ("11 points") is **dead code**; the real ladder is
  `linspace(1000,1500,10)`.
* `harness.py:95` — says exp1 sweeps "three densities"; the code uses **two** (0.3, 0.5).
* `real_harness.py:147-148` — *"harness.\_child sends no cover, so its p.join() variant is safe."* Both halves are
  now false: `harness._child` ships the cover (for `light_frac`) and `harness.run_isolated` already uses poll/recv.
* `rgg_harness.py:4-5` — "the 3 rsp methods drop, 15/18 run" does not reconcile (21 in suite, −3 rsp → 18).
* `EXPERIMENT_REGISTRY.md` §4.1/§4.2 — claims "RGG-large = 32 configs / 640 tasks" and "GEO-large = 49 configs /
  980 tasks". Actual, from the harnesses: **1040** and **720** (which is what §3.1 says). §4 predates the exp1
  mesh change and the RGG deflate/kNN additions.
* `metric_repair.py:792-793` — the `_violated_cuts` docstring asserts the integral oracle is *"sound AND complete,
  so a clean pass proves S optimal."* A1 shows it is neither.

**Complicated lines that deserve a comment** (candidates for the pass we owe):
`metric_repair.py:353` (why `!=` is safe), `:536` (the open `#todo: why round for an ilp?` — answer: `milp`
returns near-integer floats), `:904-905` (the `np.minimum.at` scatter-min DP relaxation), `:906-943` (`_rsp_separation`
detour reconstruction), `:1044` (that the `res.x is None` branch does not distinguish a timed-out incumbent),
`:1199-1262` (`_region_growing_multicut` radius selection), `:1361-1366` (`covering_lp_cover` guaranteed-flag logic);
`harness.py:156` (the `0.5` integrality threshold), `:281-285` (`_WORST` severity ranks, and that `skipped_n`/
`skipped_H` rank equal to `ok`), `:56` (`_ints` — no guard against two `linspace` samples colliding);
`real_harness.py:215` (why ILP is exempt from the budget), `:225` (that the timeout is per-component);
`collect.py:26-28` (first-file-wins header); `plots.py:84-88` (the unexplained `0.6` constant);
`rgg_analyze.py:145` (`groupby(...).first()` relies on NaN-skipping); `datasets.py:101`; `build_real_graphs.py:134`.

---

## 5. Corrections the paper needs

1. **The `bct_lin` / `bct_log` / `flycns_log` exclusion is a BUG, not a limitation.** `campaign_summary.tex`
   §7 currently says the naive oracle *"is incomplete … a real limitation of the float-weight oracle, not a
   verifier tolerance."* Both clauses need rewriting: the cause is A1, a SciPy dense-matrix conversion that
   deletes sub-`1e-8` edges, triggered by the `EPS=1e-9` floor our own `_lin`/`_log` inversions introduce. It is
   fixable and the graphs are recoverable.
2. **"up to 37% [of edges] still broken" is unsupported.** Measured residuals are 19 / 17 / 56 edges = 0.10% /
   0.09% / 0.40% of `|E|`, or 9.0% / 0.5% / 3.2% of `|H|`. Replace with the measured figures, and note the max
   residual gap (3.95 on `flycns_male_log`) — these are flat misses, not numerical near-ties.
3. **The `ripe_atlas` OOM paragraph (already added) stands** and is unaffected.
4. **Table 3's numbers are correct**; the *figure* `fig:real-heat` is not (A8) — it shows the three graphs as
   exact-reference columns.

---

## 6. Suggested order (nothing applied)

1. **A5** — one token (`except (EOFError, OSError)`). Do it before re-running anything, so failures on the heavy
   tail record a `killed` row instead of destroying the task.
2. **A1** — the correctness fix. Then re-run the ILP + LP-rounding family on the four sub-`1e-8` graphs.
3. **A2 + A3** — *must* precede any regeneration of `summary_real.csv` or the figures.
4. **A4, A6** — surface non-convergence instead of hiding it.
5. **A7, A8, A9** — reporting honesty: skip/timeout rates, red-flag invalid references, grid-filter `collect.py`.
6. **B4/B5** — before the graph set ever gains a disconnected non-metric graph.
7. **B6** — thread a seed through `MVD_Pivot` before the suite order is ever changed.
