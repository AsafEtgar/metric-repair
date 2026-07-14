# Metric Repair — project context

Read this before doing anything. It is the state of the work as of 2026-07-13, and it travels with the files
(unlike session memory, which is scoped to the directory Claude was launched from).

---

## The thesis, in one paragraph

**Metric repair is two problems, and the literature only ever posed the first.** A repair is a SET of edges
*and* an assignment of WEIGHTS to them. Every implementation closes the second silently, with the same rule
(`restore`: reweight each cover edge to its shortest detour in G∖S), and folds that rule into the *definition*
of the problem — so it becomes invisible, and nobody asks whether it is right.

**Do not let the definition weld them together.** The moment `restore` goes back into the problem statement,
the paper loses its point.

## The four results

1. **Question 1 (which edges) is answered.** The ranking of the 16 algorithms **inverts with density**.
   `gmr_bestofk` gives the best covers when sparse (ρ_H = 0.332) and returns on 4 of 720 dense instances;
   `pivot`/`left_edge` are unremarkable when dense and the two worst in the study when sparse. And on planted
   data **DOMR's cover IS the corrupted set** — precision 1.000, recall 1.000, polynomial time, no oracle.

2. **Question 2 (what weights) is not.** *Lemma:* if every e ∈ S has w(e) ≥ d_{G∖S}(e), then `restore` leaves
   every shortest-path distance unchanged. That is exactly what an inflation corruption produces, and exactly
   what the heavy set *is*. Measured with the exact corrupted set: **max|D_restore − D_obs| = 2.2e-16.** It
   recovers nothing.

3. **The value is in the weights.** A cover of 19.6% precision with the TRUE weights hits the ceiling exactly.
   A cover of 100% precision with canonical weights recovers nothing. Same information, two rules.

4. **On real data, no SMALL set works at all.** Hand every saved cover its true weights on the NMR graphs, and
   what it captures correlates with **|S|/m at r = 0.945** (p = 4.5e-16). What a cover recovers is decided by
   *how many* edges it edits, not *which* ones. The small covers — what metric repair exists to find — capture
   a mean of **−1.3%**, and 15 of 28 make things worse. **The error is not concentrated anywhere. Sparsity and
   recovery are in direct opposition.**

Open problem, to close the paper: find w′ on a given S that is **certified** (metric), **recovering** (moves
toward truth), and **oblivious** (uses only the observed graph). `restore` has 1 and 3; the true weights have
1 and 2. Nothing has all three.

---

## PRIOR ART — cite it, do not claim it

- **Lemma 6.1 / DOMR are already published.** Simas, Brattig Correia & Rocha, *IMA J. Complex Networks* 2021 —
  the **distance backbone**. Their *semi-metric edges* = our heavy set H; their *metric backbone* = our DOMR;
  their **Theorem 3.2** is our lemma. It ships in a package (`distanceclosure`).
  https://arxiv.org/pdf/2103.04668 — **if this goes in as ours, a referee kills it.**
- **Havel's bound smoothing IS our `observed` baseline.** The smoothed *upper* limits are the all-pairs
  shortest paths of the graph whose weights are the bounds — i.e. `apsp(edges)`. So we compare repair against
  what every NMR pipeline already produces, not against raw data. Havel also names our heavy set ("triangle
  inequality violations") and calls bound smoothing *"the most important, and least well-solved, step."*
  https://ti.inf.ethz.ch/ew/courses/GCMB07/material/lecture13/havel-distgeom-review.pdf
- **Tree-metric variant is taken:** L0 fitting to a tree metric = Kipouridis, ESA 2023.

**Framing that works:** *one problem, rediscovered independently in three communities, none of whom cite each
other, each solving a special case, and none of whom asked what the weights should be.* Do **not** say "only
2–3 papers know about this."

---

## Datasets — NMR is the only strong motivation

| dataset | verdict |
|---|---|
| **NMR (atom, residue)** | **STRONG** — the embedding *is* the science; the field already computes our baseline |
| `ripe_atlas` | WEAK — TIVs known 25 yrs, but the field tolerates/discards/shifts, never sparse-repairs |
| `dimacs_ny_d/_t` | **NULL** — metric by construction. **`_t` is NOT real travel-time data** (great-circle ÷ a synthetic 4-value speed table). A planted *base*, not an application. |
| `cassiopeia` | WEAK — phylogeny needs *additivity*, strictly stronger than the triangle inequality |
| connectomes, fMRI | WEAK — Leiden/UMAP/modularity need no metric; the 1/s conversion manufactures the violations |
| `pbmc3k` | WEAK — kNN+geodesic+MDS = Isomap, which the field does not run |

**NMR is also the only graph where the oracle-weights question can be *asked*** — it needs the graph and the
truth to measure the same physical quantity (NOE bounds and fold distances are both in Å). Elsewhere the
substitution is a unit error, and the "ceiling" is circular.

---

## Files

**THE PAPER NO LONGER LIVES IN THIS TREE.** As of 2026-07-13 it moved out of `overleaf_report/` and into the
Overleaf–Dropbox sync folder, where it is a live Overleaf project:

```
PAPER="/Users/asaf/Library/CloudStorage/Dropbox/Apps/Overleaf/Metric Repair in The Wild"
```

Everything in it edits *in place* and syncs to Overleaf on save. The old `overleaf_report/` is gone (it is in
`~/.Trash/overleaf_report_moved_20260713-163552` if anything was missed). Paths below are relative to `$PAPER`.

**RESTRUCTURED 2026-07-14. `ALENEX/` IS GONE.** The submission is now the paper root, and everything that is
not the submission lives in `buildup/`. The root *is* the ALENEX paper:

```
$PAPER/
  story.tex            <- THE MAIN FILE. Compile this. 7pp today; 10pp is the cap.
  intro.tex  set_up.tex  benchmarks.tex  weights.tex     <- the section bodies
  siamproceedings.sty  siamplain.bst  example_references.bib
  figures/             <- 159 figures. \graphicspath{{figures/}}
  tables/              <- ALL generated LaTeX. \input{tables/tab_x}
  buildup/             <- everything else; each doc still compiles
```

- **`story.tex`** — **write from this.** Two-column SIAM. Complete introduction + abstract; most sections are
  a header plus bullets (blue `plan` env). NMR section is fully written. `\input{benchmarks.tex}` is
  **currently commented out** (line 85) — the old `sec_edges.tex` was folded into it. Ask before re-enabling.
- **`weights.tex` is EMPTY (0 bytes).** That is §6, "What weights?" — the title's second half, and the
  biggest unwritten thing in the paper.
- `tables/` — the six generated tables **plus `inversion_macros.tex`**. One rule: **everything generated
  lives in `tables/`.** The generators enforce it (see below).
- `buildup/` — `practice.tex` (54pp report), `paper_plan.tex` (float budget + the four `app_*.tex`),
  `experimental_design.tex`, `alenex.tex` (earlier draft, a parts bin; blue `\F{}` = filled from data), the
  `tab_plan_*`/`tab_*recovery` tables those two use, `archive/`, the `_check_*.py` one-off verifiers, and the
  `.md` notes. All four top-level docs compile **from inside `buildup/`** (54 / 26 / 13 / 8 pp, 0 errors).
  They reach the figures with `\graphicspath{{../figures/}}`; `alenex.tex` has its own copies of the `.sty`
  and `.bst`.

`figures/` and `tables/` must stay **siblings of `story.tex`** at the paper root. Everything resolves off
that, including `buildup/` one level down.

**CAPTIONS ARE SHORT, AND THE AUTHOR EDITS THEM.** Settled 2026-07-14 — it is his stated preference, and the
generators now obey it. The first pass emitted ~2,000-character captions that argued the methodology inside
the float; he commented them out by hand (they cost a page of a ten-page limit), and a regeneration then
silently destroyed those edits. All five captions were cut on 2026-07-14 (9,273 → 4,791 chars, −48%), and
`story.tex` holds at 7pp with them **active**.

A generated caption is now: **bold lead + how to read the table + the one headline number.** Two or three
sentences, then stop. The median convention, the best-not-median argument, the DOMR control, the recall
cliff — all of that is the **section prose's** job. Two things never get cut for brevity:
- **Numbers stay DERIVED.** Shortening is never a licence to type a value into a caption string.
- **A disclosure the table would be dishonest without stays** — but terse. `pbmc3k` k-NN at k ≤ 15 is
  tautological; `dimacs_ny_d` medians are over survivors; `nmr_atom`'s truth covers 343 of 430 nodes.

The caption still lives in the generated `.tex`, so regenerating still overwrites a hand-edit — short captions
just make that a 30-second re-do instead of a lost page.

---

## THE SKELETON — agreed 2026-07-13. Build to this.

The author set this structure. **Do not silently reorganise it.** The one amendment on top of his draft
(which he accepted) is splitting weights from downstream — see §6/§7.

| § | content | floats |
|---|---|---|
| 1 | Introduction | — |
| 2 | **Preliminaries — and BOTH lemmas live here**, stated once, attributed to Simas et al., then *used* by §5 (the DOMR control), §6 (the cross) and §7 (the inflation corollary). Do not introduce theory mid-results. | — |
| 3 | Related work — one problem, three communities | — |
| 4 | **Setup.** The algorithms (**be surgical about which are NEW** — see the trap below), the synthetic families and their breakage, the real datasets. | `tab_datasets` |
| 5 | **Benchmarking.** Accuracy, CPU/wall time, **the ranking flips**. | `tab_invert`, `fig:invert` |
| 6 | **What weights?** The corollary, the 2×2 set×correction cross, "the value is in the weights." **This is the title's second half and it gets its own section.** | `tab_cross` |
| 7 | **What repair buys downstream.** The corruption decides; the hope answered; the oracle lies. | `tab_corruption`, `tab_hope` |
| 8 | **NMR — the crescendo, not an epilogue.** captured ≈ \|S\|/m at r = 0.945; no SMALL set works; sparsity and recovery are in direct opposition. | oracle table + MDS figures |
| 9 | Conclusion — the open problem (certified / recovering / oblivious) | — |
| A+ | Appendices. **The dataset × algorithm grid goes HERE** (19 × 21 ≈ 400 cells — it would eat a page of the main text and nobody would read it). | |

**Why §6 and §7 are separate.** The title is *"Metric Repair Is Two Problems: Which Edges, and What
Weights."* If "what weights" is one bullet inside a section that also carries the downstream probes, the
title promises a symmetry the body does not have. They are also two different claims: §6 says **the repair
rule is wrong**; §7 asks **whether repair buys anything at all**. Answering the first does not discharge the
second.

**Two open risks, flagged and not yet closed:**
1. **Which algorithms are claimed as NEW.** This is the sentence a referee hunts. `DOMR` is **not** ours (see
   PRIOR ART above). Check `spc`, `l1sep` and `bestofk` against the literature before claiming any of them.
2. **Float budget.** Six tables plus NMR figures in ten pages will bind. The §6/§7 split spreads them; the
   dataset × algorithm grid must stay in the appendix.

---

### Generated LaTeX — never hand-edit, never transcribe

**Five gated generators** in `Average Metric Repair Sage/experiments/` (all new, all untracked/uncommitted).
Run each from `Average Metric Repair Sage/`:

```
PAPER="/Users/asaf/Library/CloudStorage/Dropbox/Apps/Overleaf/Metric Repair in The Wild"

sage -python experiments/inversion_table.py  --texdir "$PAPER/tables"    # tab_invert + inversion_macros
sage -python experiments/cross_table.py      --texdir "$PAPER/tables"    # tab_cross      (§6)
sage -python experiments/corruption_table.py --texdir "$PAPER/tables"    # tab_corruption (§7)
sage -python experiments/hope_table.py       --texdir "$PAPER/tables"    # tab_hope       (§7)
sage -python experiments/datasets_table.py   --texdir "$PAPER/tables"    # tab_datasets   (§4, one column)
```

**All five now take the same `--texdir "$PAPER/tables"`.** Every one **gates before it writes** and refuses to
overwrite on failure, leaving the previous `.tex` intact rather than giving the paper untrustworthy numbers.

`paper_dir()` — the same function in all five — locates `story.tex` and accepts **only** the directory holding
it or that directory's `tables/`. It refuses `$PAPER/figures`, `$PAPER/buildup`, and anything outside the
paper. Before 2026-07-14 it accepted any sibling of a directory containing `story.tex`, which meant
`--texdir "$PAPER/figures"` would have silently dumped a table into the figures folder — the exact rot the
gate exists to stop. **Captions are short by design and the author edits them — see the Files section.**

**Why they exist.** The last hand-computed table shipped a **wrong number**: `l1sep_iomr`'s sparse return rate
printed as 100.0% when it is 99.7% — the hand pass filtered on `status` while the caption promised "returned
*and* verified." If a number needs to change, **regenerate; do not retype.**

**The gates that earn their keep** (each can genuinely fail; each has):
- **DOMR is an exact control.** By the decrease-only lemma it changes no shortest path, so it must move
  neither downstream axis. Measured: 4e-16 disparity, exactly 0 on k-NN. A nonzero reading is a bug.
- **Provenance by rebuild.** `cross_table.py` regenerates the seeded planted instance and requires its \|B\|
  to match the CSV's oracle row — the only check that catches a *stale* CSV.
- **The no-op identity.** A cover holding none of the corrupted edges cannot be helped by the true weights,
  so its disparity must equal the observed one *exactly*.

**One editorial column is NOT a measurement** and the code says so: `tab_datasets`' "does a standard tool need
a metric?" is a judgement about six literatures, curated in `VERDICT`. The gate polices its *scope* (every
graph has a verdict; every verdict names a real graph), never its truth.

---

### The COUPLED density sweep — `exp2c`, added 2026-07-14, NOT YET RUN

The large dense grid never varied density under **coupled** weights: `exp1` (coupled) pins p at {0.3, 0.5},
and only `exp2b` (decoupled) sweeps it. So every density statement the paper makes at scale is a statement
about the *decoupled* model. This array fills the gap.

```
experiments/coupled_harness.py     the array  (n=2000, p = 2n^-alpha, alpha 0.500 -> 0.229, 16 pts x 20 seeds)
experiments/submit_coupled_dsq.sh  build joblist + dSQ batch; RUNS THE PREFLIGHT and refuses to submit if it fails
experiments/collect_coupled.py     5 gates, all able to fail
```

**`harness.py` is imported READ-ONLY.** The array carries its own grid and its own runner, reusing the
harness's generator, suite, isolation, cap and `CSV_FIELDS` verbatim — the rows are schema-identical to the
campaign's, which is the whole point. Never add a grid to `harness.py`; it would invalidate 11,965 tasks.

**THE TRAP THIS ARRAY ALMOST FELL INTO, and the gate that stops it.** Under coupling the weights are
Geometric(1−p), so the mean weight is **1/(1−p)**: *density IS the weight spread*. At small p the weights
collapse onto 1, and an all-ones graph is metric by construction. The obvious mirror of `exp2a`
(α ∈ [1/2, 2/3]) therefore puts **every** point at |H|/m ≤ 0.003 — 320 tasks and ~800 core-hours to
benchmark repair algorithms on graphs with **nothing to repair**, producing a table of zeros that would read
as a finding. `--preflight` builds every grid point for real, measures |H|/m, and **refuses to submit** unless
the sweep actually breaks. Run it; it is not optional. (Measured: α 0.500 → 0.229 gives |H|/m 0.0020 → 0.1233,
m 90k → 701k. The sparse end being near-metric is the **onset**, and it is the result.)

**Two `bestofk` methods are NOT run.** They time out on 100% of `exp2b`'s tasks — 1800 s each, every task,
zero data, **38% of the whole budget**. Dropping them is not free: it **forfeits** their return-rate cells
rather than measuring them at 0%, and a return rate is only comparable across sweeps at an equal cap. Their
cells are **absent, not zero**, and `collect_coupled.py` G5 says so out loud. Everything else runs at the same
1800 s cap as `exp1`/`exp2b`, so every other cell IS comparable.

Resources: 24g (exp1 ran 563k edges in 16g; this reaches 701k), 8 h walltime — the per-task budget is
`task_budget("large") = 6 h` and **must stay under the walltime or the CSV is lost outright**
(`harness.py:56`). `MAXJOBS=320`: the array is embarrassingly parallel and only 320 wide.

**HOLD `exp2c` if the dense family is demoted.** It is a *dense-family* experiment. As of 2026-07-14 the
author is considering dropping the dense family and resting the benchmark on the RGG. If that happens, do not
submit it — nothing is lost, the code and the preflight keep.

---

### The RGG SCALE array — `rgg_scale`, added 2026-07-14, NOT YET RUN

The benchmark section's candidate spine: the sparse family pushed to **n = 4000** in **both** corruption
directions, with fraction and magnitude swept in both directions, and the jitter sweep dropped.
**60 points × 30 seeds = 1,800 tasks, ~2,370 core-hours.**

```
experiments/rgg_scale_harness.py     the array   (n=1000..4000 step 200; S1/S1d/S2/S2k/P2df/P2dm/P2if/P2im)
experiments/submit_rgg_scale_dsq.sh  12g, 12h walltime, MAXJOBS=1800; runs the preflight, refuses on failure
experiments/collect_rgg_scale.py     6 gates
```

`rgg_harness.py` is imported **read-only**. `_run()` already takes `budget` as a parameter, so the raise below
costs no edit.

**THE BUDGET IS RAISED TO 9 h, AND IT IS NOT A TUNING KNOB.** The section wants to claim that algorithms hit
their limits even on a sparse family. That claim is only worth making if a timeout is an *algorithmic* fact.
It would not have been. The harness's per-grid budget is **6 h**, and the top rungs approach it (the ladder
was designed to reach n = 5000, ~10 core-h, before it was trimmed) — so the budget can fire, and `_run` then
marks the **remaining** algorithms `skipped_time`,
walking `build_suite_rgg` in order, which ends:

```
... l1sep_gmr, l1sep_iomr, spc_gmr, spc_iomr, PIVOT, LEFT_EDGE
```

**`pivot` and `left_edge` are last — and they are exactly the two whose limitation the section exists to
demonstrate.** Under a 6 h budget their failure would be *true by construction of the suite order*, and
unfalsifiable. The RGG is connected (median 1 component, giant ≥ 99.4%), so a task is hard-bounded at
16 × 1800 s = **8 h**; a **9 h** budget therefore never binds and the per-algorithm cap is the only thing left
that can stop an algorithm. `collect_rgg_scale.py` **G4 gates on `skipped_time == 0`** and refuses to print if
the budget ever fired. (On the published grid `skipped_time` is 0 of 7,040 rows — the hazard is entirely new
to the larger n, which is why it would have been missed.)

**Scale is a VERTEX claim, and that is the stronger one.** At n = 4000 the graph carries **23,470 edges** (the
dense family reaches 563k, so this is no edge-count record) — but `pivot` and `left_edge` **complete** it to
**7,998,000**, a **341× blowup**, at ~3.7 GB peak. They pay Θ(n²) whether the edges are there or not. That is
only visible at large n *on a sparse graph*; the dense family could never have shown it.

**Fraction and magnitude now run in BOTH directions** (`P2if`, `P2im` are new). The published sweeps are
deflate-only, and the corruption direction does not shift the ranking — it **inverts** it. A one-direction
sweep cannot support a claim about fraction or magnitude in a section whose thesis is that the direction
decides.

---

### §5 BENCHMARKS — the structure, agreed 2026-07-14. Build to this.

The RGG is the spine. The author set this order; do not silently reorganise it.

| § | content |
|---|---|
| **5.1** | **Small RGG.** Where the optimum is known — the ILP converges on **4,960 of 5,000** tasks (99.2%, against the dense grid's 69.8% / 36.0%). This is the only place the paper can say *how far from optimal*: \|S\|/OPT = `l1sep_gmr` **1.562**, `spc_gmr` 2.649, `pivot` **8.723**. |
| **5.2** | **Performance, PER DIRECTION.** Never pooled. There is no single "performance on the RGG": `l1sep_gmr` rewrites 12% of the graph under deflation and **80%** under inflation. |
| **5.3** | **Limitations.** Runtime and memory. `pivot`/`left_edge` **complete the graph** — at n=4000 that is 7,998,000 edges for a graph with 23,470, a **341× blowup**, ~3.7 GB. They pay Θ(n²) whether the edges are there or not, and this is **only visible at large n on a SPARSE graph**. |
| **5.4** | **Fraction and magnitude**, in both directions (that is what `P2if`/`P2im` are for). |
| **5.5** | **The corruption decides.** `tab_invert`. |

**Why 5.2 comes before 5.5 and it still works:** the author's structure reports performance *per direction from
the start*, so no number is ever presented pooled and none is retroactively ambiguous. 5.5 then formalises
what the reader has already been seeing. (An earlier draft put the direction finding last while reporting
pooled performance first — that version does not survive, because pooled medians are not statistics of
anything. See the GROUPS note in `inversion_table.py`.)

**The RGG needs no dense family for a density story.** Its own degree sweep (`S2`, n=2000, deg 4→40) spans
m = 3,942 → 37,229 at ~2 core-h/task — and its weights are Euclidean distances, so **density and the weight
model are independent by construction**. That is exactly the pathology that makes a Γ(n,p) density sweep so
awkward (coupled weights ⇒ density *is* brokenness; decoupled weights need a whole second model). The RGG
degree sweep replaces the dense density sweep and does it better.

Figures are still copied into `figures/` by hand from `Average Metric Repair Sage/analysis/figs/`. That is the
one remaining un-gated hop, and it has already gone stale once.

Every number traces to a gated script in `Average Metric Repair Sage/experiments/`. **Nothing is transcribed
by hand — keep it that way.**

---

## Rules

- `sage -python` for everything. System `python3` has no numpy/pandas.
- **NEVER edit** `metric_repair.py`, `graph_models.py`, `datasets.py`,
  `experiments/{harness,rgg_harness,real_harness,run_*,sweeps}.py` — changing task-import-path code
  invalidates the published campaign. Reading is fine.
- **No auto commit/push.** Only when asked, or before a cluster run.
- **Gates must fail CLOSED.** A check that cannot fail is not a check. This has already caught two real bugs:
  a collector that printed "!! VIOLATED" and exited 0, and a gate that compared against two *fabricated*
  constants. **Never hardcode an expected value — read it from the CSV you are checking against.**

## Traps that produce plausible-but-wrong numbers *without crashing*

- **Node ordering:** `load_graph` sorts labels as **strings** ('10' < '2'); `load_edgelist` uses int order.
  Mixing them permutes the truth against the embedding. Go through `load_graph` + `true_distances` end to end.
- **`cpu = 0` sentinel:** a killed run records cpu 0. Filter on `status == "ok"` or the slowest algorithms
  report as the fastest — and vanish from log-axis figures entirely.
- **`nmr_1d3z_atom` truth is partial:** 343 of 430 nodes; MDS core is 340. Disclose it.
- **`pbmc3k` kNN at k ≤ 15 is tautological** (it *is* the 15-NN graph of its own truth). Quote **k = 20**.
- **Median over survivors:** on the planted road net 9 of 15 algorithms time out, so `med` is biased toward
  the cheap combinatorial methods. Say so wherever it is quoted.
- **MDS grids:** panels within one dataset must share one axis box, or a wrecked embedding renders as intact.
