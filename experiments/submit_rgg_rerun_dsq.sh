#!/bin/bash
# Re-run the RGG campaign -- small AND large, PLANTED corruptions only -- against the FIXED inflate.
# Run FROM THE REPO ROOT. Builds two dSQ arrays. Does NOT submit.
#
#   bash experiments/submit_rgg_rerun_dsq.sh <conda_env> <pi_netid>
#
# WHY. break_metric_graph's inflate branch carried an ABSOLUTE floor -- max(mu*detour, w0*1.001 + 1) -- meant
# for integer weights. RGG weights are Euclidean distances in the unit square, all below 0.12, so the "+1"
# swamped mu*detour and EVERY inflation landed at ~11.8x the detour regardless of the magnitude requested
# (measured: 11.77 at mu = 1.2, 1.5, 2, 3, 5 and 10 alike). The magnitude knob was inert and the inflate
# magnitude sweep planted one corruption six times. graph_models.py is fixed; every INFLATE row in the
# published RGG campaign was generated against a corruption nobody asked for.
#
# WHAT RUNS.  small  74 points x 40 seeds = 2,960 tasks   (n=100..500;  ILPs ON -- the exact optimum)
#             large  41 points x 20 seeds =   820 tasks   (n=1000..3000; ILPs dropped)
#             3,780 tasks total. Jitter is dropped from both grids, per the author.
#
# WHAT DOES NOT RUN. The dense family (exp1/exp2a/exp2b), the coupling A/B and exp2c are UNAFFECTED --
# harness.py never calls break_metric_graph; its graphs are non-metric by WEIGHT MODEL, not by a planted
# corruption. The planted REAL bases are a separate array: submit_realplanted_dsq.sh.
#
# SAME GRAPHS, NEW CORRUPTION. task_seed hashes the cfg's CONTENT, not the task index, so dropping the jitter
# points renumbers the array without moving a single seed. The preflight checks that against the DELIVERED CSV
# (G4: 2,960 of 2,960 seeds matched) -- this is a REPLACEMENT of the inflate rows, not a different experiment.
#
# BUDGETS ARE THE PUBLISHED ONES (2 h small, 6 h large), and that is deliberate: skipped_time is 0 of 175,218
# rows on the published small grid and 0 of 32,582 on the large one, so the task budget never fired and a
# timeout is an algorithmic fact. The fix also makes inflations WEAKER (mu=3, not 11.8), which shrinks the
# covering LP -- 6,825 broken cycles instead of 9,464 at n=300 -- so there is more slack than before, not less.
# collect_rgg_rerun.py GATES on skipped_time == 0.
#
# WALLTIME IS NOT THE BUDGET. A task stops LAUNCHING algorithms at the budget, but the one already running
# may take TIMEOUT_S = 30 min more. Small: 2 h + 30 min = 2.5 h, so 3 h. Large: 6 h + 30 min = 6.5 h, so 8 h.
# A walltime kill loses the CSV outright -- never shrink one of these without the other.
set -euo pipefail

ENV="${1:-metricrepair}"
NETID="${2:-CHANGE_ME}"

module load miniconda dSQ 2>/dev/null || echo "note: 'module load miniconda dSQ' failed -- load them yourself"
conda activate "$ENV"

echo "PREFLIGHT: is the magnitude knob actually live, and do the seeds still match the campaign?"
python experiments/rgg_rerun_harness.py --preflight || {
    echo
    echo "PREFLIGHT FAILED -- NOT submitting. (If G3 failed, the '+1' floor is back in graph_models.py and"
    echo "submitting would burn the cluster reproducing the bug.)"
    exit 1
}

mkdir -p logs
for GRID in small large; do
    if [ "$GRID" = "small" ]; then MEM=8g;  TIME=03:00:00; else MEM=16g; TIME=08:00:00; fi
    OUTDIR="results_rgg_rerun_$GRID"
    JOB="rgg_rerun_${GRID}_joblist.txt"
    BATCH="dsq_rgg_rerun_${GRID}_submit.sh"
    mkdir -p "$OUTDIR"

    NTASKS=$(python experiments/rgg_rerun_harness.py --grid "$GRID" --count)
    echo
    echo "rgg_rerun/$GRID: tasks=$NTASKS mem=$MEM time=$TIME maxjobs=$NTASKS -> $OUTDIR"

    python - "$NTASKS" "$ENV" "$OUTDIR" "$GRID" <<'PY' > "$JOB"
import sys
n, env, outdir, grid = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
setup = ('module load miniconda && source "$(conda info --base)/etc/profile.d/conda.sh" '
         f'&& conda activate {env}')
for i in range(n):
    print(f"{setup} && python experiments/rgg_rerun_harness.py --grid {grid} "
          f"--task-index {i} --outdir {outdir}")
PY
    echo "  wrote $JOB ($NTASKS lines)"

    dsq --job-file "$JOB" --batch-file "$BATCH" --partition day --account "pi_${NETID}" \
        --cpus-per-task 1 --mem-per-cpu "$MEM" --time "$TIME" --max-jobs "$NTASKS" \
        --output "logs/dsq-rggrerun-${GRID}-%A_%3a.out"
done

echo
echo "Created dsq_rgg_rerun_small_submit.sh and dsq_rgg_rerun_large_submit.sh. Review, then:"
echo "    sbatch dsq_rgg_rerun_small_submit.sh"
echo "    sbatch dsq_rgg_rerun_large_submit.sh"
echo
echo "When they land:"
echo "    sage -python experiments/collect_rgg_rerun.py"
echo "    sage -python experiments/rgg_analyze.py --results results_rgg_rerun_small --outdir analysis/rgg"
echo "    sage -python experiments/rgg_analyze.py --results results_rgg_rerun_large --outdir analysis/rgg_large"
echo "    sage -python experiments/section5.py --texdir \"\$PAPER/tables\" --figdir \"\$PAPER/figures\""
echo
echo "NOTE FOR THE COLLECTOR: H == B under inflation is NO LONGER a grid-wide identity. At mu = 1.2 a weak"
echo "inflation does not survive its own interference (the detour is measured in G but heaviness is judged in"
echo "H), so 157 of 166 planted edges come back heavy. That is a RESULT, not a bug -- the old +1 inflated so"
echo "hard it could never show. section5.py's G2a must be re-scoped to the magnitudes where it holds."
