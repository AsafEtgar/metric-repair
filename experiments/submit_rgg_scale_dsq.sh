#!/bin/bash
# Build a joblist + dSQ batch for the RGG SCALE array. Run FROM THE REPO ROOT. Does NOT submit.
#
#   bash experiments/submit_rgg_scale_dsq.sh <conda_env> <pi_netid> [mem]
#
# WHAT IT IS. The sparse family pushed to n = 5000 in BOTH corruption directions, with the fraction and
# magnitude sweeps run in both directions too, and the jitter sweep dropped. 70 points x 30 seeds = 2,100
# tasks. This is the array the benchmark section rests on if the RGG becomes the spine.
#
# THE TWO THINGS THAT MAKE IT WORTH RUNNING.
#
#   SCALE IS A VERTEX CLAIM, AND THAT IS THE STRONGER ONE. At n = 5000 the graph carries ~29,500 edges --
#   the dense family reaches 563,000, so this is not an edge-count record. But `pivot` and `left_edge`
#   COMPLETE the graph before they start: at n = 5000 that is 12,497,500 edges for a graph that has 29,470,
#   a 424x blowup, and ~5.5 GB of peak memory. "We ran to n = 5000 on a graph with 29k edges, and the
#   graph-completing methods needed 12.5M" is a sharper sentence than any edge count, and it is only
#   visible at large n ON A SPARSE GRAPH.
#
#   THE FRACTION/MAGNITUDE SWEEPS NOW RUN IN BOTH DIRECTIONS. The published grid runs them deflate-only,
#   and the corruption direction does not shift the ranking -- it INVERTS it (l1sep_gmr loses to spc_gmr on
#   100% of inflate tasks and beats it on 100% of deflate). A one-direction sweep cannot support a claim
#   about fraction or magnitude in a section whose thesis is that the direction decides.
#
# THE BUDGET IS RAISED TO 9 h, AND THIS IS NOT A TUNING KNOB. harness's per-grid budget is 6 h. At n = 5000
# the median task needs ~10 core-h, so the budget would fire -- and `_run` then marks the REMAINING
# algorithms `skipped_time`, in build_suite_rgg order, which ends: ... spc_gmr, spc_iomr, PIVOT, LEFT_EDGE.
# Those two are exactly the methods whose limitation the section exists to demonstrate. Under a 6 h budget
# their failure would be TRUE BY CONSTRUCTION OF THE SUITE ORDER, and unfalsifiable. The RGG is connected, so
# a task is hard-bounded at 16 algos x 1800 s = 8 h; a 9 h budget therefore never binds and the per-algorithm
# cap is the only thing left that can stop an algorithm. collect_rgg_scale.py GATES on skipped_time == 0.
#
# NON-DESTRUCTIVE. rgg_harness.py is imported read-only (`_run` already takes `budget` as a parameter, so
# the raise costs no edit). Writes ONLY results_rgg_scale/.
#
# COST. ~4,700 core-hours. Per-task median climbs from 0.09 core-h at n=1000 to ~8 h (the cap ceiling) at
# n=5000. Walltime 12 h covers the 9 h budget plus instance build and the DOMR pass.
set -euo pipefail

ENV="${1:-metricrepair}"
NETID="${2:-CHANGE_ME}"
MEM="${3:-16g}"            # pivot/left_edge peak ~5.5 GB at n=5000 (completion is 12.5M edges). Headroom.
MAXJOBS=2100               # every task at once; the array is embarrassingly parallel
TIME=12:00:00              # > the 9 h budget. Never shrink one without the other.

OUTDIR=results_rgg_scale
JOB="rgg_scale_joblist.txt"
BATCH="dsq_rgg_scale_submit.sh"

module load miniconda dSQ 2>/dev/null || echo "note: 'module load miniconda dSQ' failed -- load them yourself"
conda activate "$ENV"

echo "PREFLIGHT: is the grid what the section will claim it is?"
python experiments/rgg_scale_harness.py --preflight || {
    echo
    echo "PREFLIGHT FAILED -- NOT submitting."
    exit 1
}

mkdir -p "$OUTDIR" logs
NTASKS=$(python experiments/rgg_scale_harness.py --count)
echo
echo "rgg_scale array: tasks=$NTASKS mem=$MEM time=$TIME maxjobs=$MAXJOBS -> $OUTDIR"

python - "$NTASKS" "$ENV" "$OUTDIR" <<'PY' > "$JOB"
import sys
n, env, outdir = int(sys.argv[1]), sys.argv[2], sys.argv[3]
setup = ('module load miniconda && source "$(conda info --base)/etc/profile.d/conda.sh" '
         f'&& conda activate {env}')
for i in range(n):
    print(f"{setup} && python experiments/rgg_scale_harness.py --task-index {i} --outdir {outdir}")
PY
echo "  wrote $JOB ($NTASKS lines)"

dsq --job-file "$JOB" --batch-file "$BATCH" --partition day --account "pi_${NETID}" \
    --cpus-per-task 1 --mem-per-cpu "$MEM" --time "$TIME" --max-jobs "$MAXJOBS" \
    --output "logs/dsq-rggscale-%A_%3a.out"

echo
echo "Created $BATCH.  Review it, then:  sbatch $BATCH"
echo
echo "When it lands:"
echo "    tar czf results_rgg_scale.tgz $OUTDIR/"
echo "    sage -python experiments/collect_rgg_scale.py"
echo
echo "The collector REFUSES to print if any row came back skipped_time -- that would mean the task budget,"
echo "not the algorithm's own cap, decided who failed, and the whole limitations claim would be confounded."
