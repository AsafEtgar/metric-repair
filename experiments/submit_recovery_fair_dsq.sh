#!/bin/bash
# UNBIASED downstream recovery: ONE task per (graph, corruption, algorithm, seed), UNIFORM 2h cap.
# Run FROM THE REPO ROOT. Does NOT submit -- it writes the dSQ batch for you to sbatch.
#
#   bash experiments/submit_recovery_fair_dsq.sh <conda_env> <pi_netid> [mem]
#
# WHY. The published recovery ran the whole suite in ONE task per (graph, corruption), under a 300s/900s
# per-algorithm cap plus a shared task budget. On the n=5000 road net most methods time out, so the median in
# tab:corruption is taken over the cheap combinatorial SURVIVORS and is biased toward them (its own caption
# says so). Here EVERY algorithm gets its OWN task and a uniform 2h wall cap, so a timeout is an ALGORITHMIC
# fact, not a scheduling artefact, and one slow method never starves another.
#
# GRID: {RGG n=3000, dimacs_ny_d n=5000} x {inflate, deflate, mixed} x 5 seeds x (observed + 15 algorithms)
# = 480 tasks. The graph + corruption is seeded by (graph, corruption, seed) and NOT the algorithm, so an
# instance's tasks rebuild the BYTE-IDENTICAL corrupted graph -- the preflight gates it (G4). Embarrassingly
# parallel, and only 480 wide, so MAXJOBS=480 runs the ENTIRE grid at once: maximum parallelism.
#
# RESOURCES. dimacs_ny_d is n=5000 and pivot/left_edge COMPLETE the graph (~12.5M edges), so 24 g -- the size
# the published realrec/recovery arrays used for this graph. Walltime clears the 2h algorithm cap plus the
# instance build + MDS/kNN scoring, with margin. Never shrink cap and walltime independently.
set -euo pipefail

ENV="${1:-metricrepair}"
NETID="${2:-CHANGE_ME}"
MEM="${3:-24g}"            # n=5000 road net + pivot/left_edge graph completion; light algorithms use far less
MAXJOBS=480               # the WHOLE grid at once -- the point is maximum parallelism
TIME=03:00:00             # 2h algorithm cap + ~10 min instance build/score, with margin

OUTDIR=results_recovery_fair
JOB="recovery_fair_joblist.txt"
BATCH="dsq_recovery_fair_submit.sh"

module load miniconda dSQ 2>/dev/null || echo "note: 'module load miniconda dSQ' failed -- load them yourself"
conda activate "$ENV"

echo "PREFLIGHT: fix live, both graphs build+break, determinism holds, 2h cap?"
python experiments/recovery_fair_harness.py --preflight || {
    echo
    echo "PREFLIGHT FAILED -- NOT submitting."
    exit 1
}

mkdir -p "$OUTDIR" logs
NTASKS=$(python experiments/recovery_fair_harness.py --count)
echo
echo "recovery_fair array: tasks=$NTASKS mem=$MEM time=$TIME maxjobs=$MAXJOBS -> $OUTDIR"

python - "$NTASKS" "$ENV" "$OUTDIR" <<'PY' > "$JOB"
import sys
n, env, outdir = int(sys.argv[1]), sys.argv[2], sys.argv[3]
setup = ('module load miniconda && source "$(conda info --base)/etc/profile.d/conda.sh" '
         f'&& conda activate {env}')
for i in range(n):
    print(f"{setup} && python experiments/recovery_fair_harness.py --task-index {i} --outdir {outdir}")
PY
echo "  wrote $JOB ($NTASKS lines)"

dsq --job-file "$JOB" --batch-file "$BATCH" --partition day --account "pi_${NETID}" \
    --cpus-per-task 1 --mem-per-cpu "$MEM" --time "$TIME" --max-jobs "$MAXJOBS" \
    --output "logs/dsq-recovery-fair-%A_%3a.out"

echo
echo "Created $BATCH.  Review it, then:  sbatch $BATCH"
echo
echo "When it lands:"
echo "    sage -python experiments/collect_recovery_fair.py"
