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

- **`overleaf_report/ALENEX/story.tex`** — **write from this.** Two-column SIAM, 10-page target. Complete
  introduction + abstract; every other section is a header plus bullets (blue `plan` env). NMR section is
  fully written.
- `overleaf_report/ALENEX/alenex.tex` — earlier fuller draft; a parts bin. Blue `\F{}` = filled from data.
- `overleaf_report/paper_plan.tex` — float budget + appendices. Not the submission.
- `overleaf_report/practice.tex` — the 54-page technical report. Keep.

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
