#!/bin/bash
# reanalyze_all.sh -- re-collect, health-check and analyze EVERY campaign with the corrected code.
# Run FROM THE REPO ROOT, after all re-runs have finished.
#
#   bash experiments/reanalyze_all.sh [PY]        # PY defaults to `python` (cluster) -- use `sage -python` locally
#
# Why re-analyze everything, not just the new results: the analyzers changed. The reference gate now
# requires status=="ok" AND valid==1 (not just `converged`), and invalid covers are dropped before
# aggregation. Every summary CSV and figure produced before that gate is suspect. See AUDIT_REPORT.md.
#
# STALE-FILE GUARD: collect.py now ABORTS on orphan indices, on CSVs older than --max-age-days, and on
# ragged schemas. If it aborts, that is the point -- inspect and quarantine, do not pass --allow-stale
# reflexively. `results_rgg/` is the known-risky directory: it is REUSED and once held a 3160-task grid.
set -euo pipefail

PY="${1:-python}"
AGE="${STALE_DAYS:-3}"          # CSVs older than this are treated as pre-fix survivors
mkdir -p analysis

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

# ---------------------------------------------------------------- preflight
# Run this on a live campaign and you get a report that LOOKS complete and is not. Two ways it lies:
#   * an empty output dir made collect.py exit 0, so every analyzer read the PREVIOUS run's *_all.csv;
#   * results_real is rewritten IN PLACE, so a half-finished heur array still shows 620/621 files present.
# Neither is detectable downstream. Refuse to start instead. FORCE=1 to override.
banner "preflight"
live=$(squeue --me -h -r 2>/dev/null | wc -l || echo 0)
if [ "${live:-0}" -gt 0 ]; then
    echo "  $live task(s) still queued or running."
    [ "${FORCE:-0}" = "1" ] || { echo "ABORT: wait for the campaign to drain (FORCE=1 to override)."; exit 1; }
    echo "  FORCE=1 -- proceeding on a LIVE campaign. Nothing below is trustworthy."
fi

short=0
while IFS=: read -r dir want; do
    [ -d "$dir" ] || { printf "  %-24s MISSING DIR (expected %s)\n" "$dir" "$want"; short=1; continue; }
    got=$(ls "$dir"/*.csv 2>/dev/null | wc -l)
    if [ "$got" -ne "$want" ]; then printf "  %-24s %5d / %-5d  INCOMPLETE\n" "$dir" "$got" "$want"; short=1
    else                            printf "  %-24s %5d / %-5d  ok\n"         "$dir" "$got" "$want"; fi
done <<EOF
results_small:2460
results_large:720
results_rgg:5000
results_rgg_large:1040
results_rgg_mixed:840
results_rgg_largemix:400
results_rgg_realrec:900
results_real:620
EOF
# results_real is 620, not 621, on purpose: ripe_atlas__gmr_ilp was OOM-killed at 8GB and will not converge
# at 95.3% break density, so re-running it buys a 17h timeout row rather than an optimum. real_check.py still
# reports it as the single missing file; that is the gate, not this count.
if [ "$short" -ne 0 ]; then
    [ "${FORCE:-0}" = "1" ] || { echo "ABORT: at least one campaign is incomplete."; exit 1; }
    echo "  FORCE=1 -- proceeding on an incomplete campaign."
fi

# ---------------------------------------------------------------- collect
banner "collect (aborts on stale / orphan / ragged input)"
$PY experiments/collect.py --indir results_small        --out results_small_all.csv         --harness geometric --grid small
$PY experiments/collect.py --indir results_large        --out results_large_all.csv         --harness geometric --grid large    --max-age-days "$AGE"
$PY experiments/collect.py --indir results_rgg          --out results_rgg_full_all.csv      --harness rgg --grid full      --max-age-days "$AGE"
$PY experiments/collect.py --indir results_rgg_large    --out results_rgg_large_all.csv     --harness rgg --grid large     --max-age-days "$AGE"
$PY experiments/collect.py --indir results_rgg_mixed    --out results_rgg_mixed_all.csv     --harness rgg --grid mixed     --max-age-days "$AGE"
$PY experiments/collect.py --indir results_rgg_largemix --out results_rgg_largemix_all.csv  --harness rgg --grid largemix  --max-age-days "$AGE"
$PY experiments/collect.py --indir results_rgg_realrec  --out results_rgg_realrec_all.csv   --harness rgg --grid realrec   --max-age-days "$AGE"

# ---------------------------------------------------------------- health checks
banner "health checks"
$PY experiments/missing_report.py                       | tee analysis/missing_report.out
$PY experiments/real_check.py --results results_real --covers results_real_covers | tee analysis/real_check.out
for f in results_rgg_full_all results_rgg_large_all results_rgg_mixed_all results_rgg_largemix_all results_rgg_realrec_all; do
    [ -f "$f.csv" ] && $PY experiments/rgg_check.py --results "$f.csv" | tee "analysis/${f}_check.out"
done

# ---------------------------------------------------------------- analyze
# Each analyzer now PRINTS the invalid-cover rows it drops. A nonzero count on the RGG/geometric grids is a
# finding, not noise: it means l1_separation exited at max_rounds without converging (AUDIT_REPORT.md A4).
banner "analyze -- geometric"
$PY experiments/analyze.py     --results results_small_all.csv        --outdir analysis            | tee analysis/analyze_small.out
$PY experiments/analyze.py     --results results_large_all.csv        --outdir analysis/large      | tee analysis/analyze_large.out

banner "analyze -- RGG"
$PY experiments/rgg_analyze.py --results results_rgg_full_all.csv     --outdir analysis/rgg           | tee analysis/analyze_rgg_full.out
$PY experiments/rgg_analyze.py --results results_rgg_large_all.csv    --outdir analysis/rgg_large     | tee analysis/analyze_rgg_large.out
$PY experiments/rgg_analyze.py --results results_rgg_mixed_all.csv    --outdir analysis/rgg_mixed     | tee analysis/analyze_rgg_mixed.out
$PY experiments/rgg_analyze.py --results results_rgg_largemix_all.csv --outdir analysis/rgg_largemix  | tee analysis/analyze_rgg_largemix.out
$PY experiments/rgg_analyze.py --results results_rgg_realrec_all.csv  --outdir analysis/rgg_realrec   | tee analysis/analyze_rgg_realrec.out

banner "analyze -- real"
$PY experiments/real_analyze.py --results results_real --outdir analysis | tee analysis/analyze_real.out

# ---------------------------------------------------------------- figures
banner "figures"
$PY experiments/plots.py      --summary analysis/summary.csv        --outdir analysis/figs/geometric
$PY experiments/plots.py      --summary analysis/large/summary.csv  --outdir analysis/figs/geometric_large
# rgg_plots takes the two summaries separately (--edit / --knn), not one --summary.
for g in rgg rgg_large rgg_mixed rgg_largemix rgg_realrec; do
    [ -f "analysis/$g/summary_edit.csv" ] || continue
    $PY experiments/rgg_plots.py --edit "analysis/$g/summary_edit.csv" \
                                 --knn  "analysis/$g/summary_knn.csv" \
                                 --outdir "analysis/figs/$g"
done
$PY experiments/real_plots.py --summary analysis/summary_real.csv    --outdir analysis/figs/real

# ---------------------------------------------------------------- pure real-data downstream (ripe/nmr)
# Separate experiment (downstream_recovery.py), so it is guarded on its own output existing rather than in
# the preflight. Analyze + plot here so the one big pass produces its figures too, and bundle_analysis then
# carries analysis/summary_downstream.csv and analysis/figs/downstream/* to the Mac like everything else.
if ls results_downstream/*.csv >/dev/null 2>&1; then
    banner "analyze + figures -- pure real-data downstream"
    $PY experiments/downstream_analyze.py --indir results_downstream --outdir analysis | tee analysis/analyze_downstream.out
    $PY experiments/downstream_plots.py   --summary analysis/summary_downstream.csv --outdir analysis/figs/downstream
else
    echo; echo "note: results_downstream/ empty or absent -- skipping the pure real-data downstream figures."
fi

# ---------------------------------------------------------------- MDS geometry recovery (ripe/nmr + broken RGG)
# Self-contained geometric lens (mds_recovery.py): reuses the saved real-graph covers (results_real_covers/,
# auto-detected) and generates its own broken RGGs, so it needs no results_* array beyond those covers. Non-
# fatal: a failure here must not abort the main pass. bundle_analysis carries analysis/summary_mds.csv and
# analysis/figs/mds/* like the rest.
#
# ORDER MATTERS: mds_recovery -> mds_sweep -> mds_plots. mds_plots' per-algorithm grid (fig_mds_grid_*) reads
# the SWEEP's summary_mds_sweep.csv + mds_sweep_embeddings.npz, so plotting before the sweep silently drops
# every grid figure (it warns and carries on). The old order ran the sweep last.
#
# NOTE on the sweep's CSV: mds_sweep MERGES into analysis/summary_mds_sweep.csv rather than truncating it,
# because _iter_saved only sees covers that exist ON THIS MACHINE. ripe's `pivot` and `iomr_rand` covers live
# only on the cluster; a truncating local run would delete those rows permanently (the CSV is gitignored).
# Merged-in rows are stamped source=cluster. It also writes a .bak first. Do not "simplify" that away.
banner "analyze + figures -- MDS geometry recovery"
if $PY experiments/mds_recovery.py --outdir analysis | tee analysis/analyze_mds.out; then
    $PY experiments/mds_sweep.py --only all --plot --outdir analysis \
        || echo "warn: mds_sweep failed (per-algorithm disparity; the grid figures will be skipped)."
    $PY experiments/mds_plots.py --data analysis/summary_mds.csv --emb analysis/mds_embeddings.npz \
        --sweep-data analysis/summary_mds_sweep.csv --sweep-emb analysis/mds_sweep_embeddings.npz \
        --outdir analysis/figs/mds || echo "warn: mds_plots failed (data CSVs still written)."
else
    echo "warn: mds_recovery failed -- skipping MDS figures (does not affect the rest of the pass)."
fi

# ---------------------------------------------------------------- cost law (n / m / |H| scaling exponents)
# The paper's "three algorithms, three cost laws" rests on this and on nothing else -- there was no cost-law
# artifact in analysis/ at all. Guarded and non-fatal like the MDS block: it is a standalone sweep, not a
# consumer of the results_* arrays. Sweep H (the |H| axis) has never been run; if it fails, SAY SO -- the
# paper must then downgrade its |H| claims from measured laws to mechanism arguments.
if [ -f experiments/cost_law.py ]; then
    banner "cost law"
    $PY experiments/cost_law.py --seeds 3 --plot --outdir analysis | tee analysis/cost_law.out \
        || echo "warn: cost_law failed -- analysis/cost_law.csv NOT written. The |H| axis is unmeasured."
fi

banner "done"
echo "Check these before trusting anything:"
echo "  * every collect.py call succeeded (it aborts on stale/orphan/ragged input)"
echo "  * real_check.py: HARD failures == 0, metric control dimacs_ny_d == 0 rows with size>0"
echo "  * real_check.py: invalid covers should now be 0 -- the A1 fix repaired bct_*/flycns_male_log"
echo "  * analyze/rgg_analyze: any 'DROPPING N invalid-cover rows' line is l1_separation non-convergence"
echo "  * downstream_analyze: DOMR self-check must read 'max |lift| = 0.00e+00'"
echo "  * mds_recovery: DOMR self-check must read 'max |gap| = 0.00e+00' (D_F == D_G by Lemma 6.1)"
echo "  * mds_sweep: '0 not measured', and summary_mds_sweep.csv still has ripe's pivot + iomr_rand rows"
echo "    (grep them -- exit 0 is NOT evidence a panel landed; a swallowed wiring bug once looked identical)"
echo "  * mds_plots: fig_mds_grid_*_{gmr,iomr}.png exist, and the domr panel is identical to observed"
