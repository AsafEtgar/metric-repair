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

banner "done"
echo "Check these before trusting anything:"
echo "  * every collect.py call succeeded (it aborts on stale/orphan/ragged input)"
echo "  * real_check.py: HARD failures == 0, metric control dimacs_ny_d == 0 rows with size>0"
echo "  * real_check.py: invalid covers should now be 0 -- the A1 fix repaired bct_*/flycns_male_log"
echo "  * analyze/rgg_analyze: any 'DROPPING N invalid-cover rows' line is l1_separation non-convergence"
