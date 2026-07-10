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

banner "done"
echo "Check these before trusting anything:"
echo "  * every collect.py call succeeded (it aborts on stale/orphan/ragged input)"
echo "  * real_check.py: HARD failures == 0, metric control dimacs_ny_d == 0 rows with size>0"
echo "  * real_check.py: invalid covers should now be 0 -- the A1 fix repaired bct_*/flycns_male_log"
echo "  * analyze/rgg_analyze: any 'DROPPING N invalid-cover rows' line is l1_separation non-convergence"
