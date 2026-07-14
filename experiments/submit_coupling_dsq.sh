#!/bin/bash
# Build a joblist + dSQ batch for the COUPLING A/B. Run FROM THE REPO ROOT. Does NOT submit.
#
#   bash experiments/submit_coupling_dsq.sh <conda_env> <pi_netid> [mem]
#
# WHAT IT ASKS, AND WHY NOTHING ELSE CAN ANSWER IT. Appendix A reports that spc_gmr and pivot SWAP between
# the dense grid's two sweeps -- spc rewrites 0.225 of exp1 and 0.418 of exp2b; pivot rewrites 0.457 and
# 0.278. That is a real observation and a useless one: exp1 and exp2b differ in n (1000-1500 vs 2000), in p
# (0.3/0.5 vs a sweep) AND in the weight model. The flip cannot be attributed to any of the three.
#
# This array fixes exactly that. n is FIXED at 300; p is FIXED, point for point; the topology distribution is
# identical. THE ONLY THING THAT CHANGES IS THE WEIGHT MODEL:
#
#     coupled     weights Geom(1 - p)   -- the spread tracks the density
#     decoupled   weights Geom(0.5)     -- the spread is held fixed
#
# If the ranking still flips, the coupling alone did it. Nothing in the published campaign runs this.
#
# THE GRID. p = 2 n^-alpha, alpha 0.2 -> 0.7 (11 points), n = 300, 30 seeds, both models = 660 tasks.
# Measured before submission: the coupled sweep collapses from |H|/m = 0.410 to 0.0006 (a 671x collapse,
# crossing the 3/5 metricity onset) while the decoupled control stays broken (0.255 -> 0.093, only 2.7x).
# The two even CROSS near alpha = 0.25 -- coupling makes the graph MORE broken than the fixed model at high
# density and METRIC at low density. That scissors IS the effect, and --preflight refuses to submit without it.
#
# THE BUDGET IS RAISED TO 10 h, AND IT IS NOT A TUNING KNOB. harness's small-grid TASK_BUDGET is 2 h and the
# alpha = 0.2 point costs ~2.25 core-h, so the budget WOULD fire -- and run_one_task then marks the REMAINING
# algorithms `skipped_time`, in build_suite order, which ends: ... spc_gmr, spc_iomr, PIVOT, LEFT_EDGE.
# spc_gmr and pivot are THE TWO METHODS THIS ARRAY EXISTS TO COMPARE. Under the default budget they would come
# back skipped at exactly the broken end where the coupling bites, and the experiment would have destroyed its
# own headline. The suite's ceiling is 9.5 h (the sum of its per-algorithm caps), so 10 h never binds.
# collect_coupling.py GATES on skipped_time == 0.
#
# WHAT IT CANNOT GIVE, STATED UP FRONT. The exact optimum dies exactly where the coupling bites: gmr_ilp
# converges on ~99% of instances below |H| = 200 and on 0% above |H| = 3,000, and the coupled model carries
# |H| ~ 11,600 at alpha = 0.2. So |S|/OPT exists only at the metric end. That is not a flaw in the design --
# it is the dense family's own pathology, the same one Appendix A reports for IOMR ("the optimum exists only
# where there is nothing to repair"), and this array MEASURES it. Below the ILP's reach we report |S|/m.
#
# NON-DESTRUCTIVE. harness.py is imported read-only. Writes ONLY results_coupling/.
#
# COST. ~300 core-hours over 660 tasks, dominated by the alpha = 0.2 rung (m = 28,700). Everything below
# alpha = 0.4 is nearly free.
set -euo pipefail

ENV="${1:-metricrepair}"
NETID="${2:-CHANGE_ME}"
MEM="${3:-8g}"             # n = 300, m <= 28,700 -- small graphs. 8g is generous.
MAXJOBS=660                # every task at once
TIME=12:00:00              # > the 10 h budget. Never shrink one without the other.

OUTDIR=results_coupling
JOB="coupling_joblist.txt"
BATCH="dsq_coupling_submit.sh"

module load miniconda dSQ 2>/dev/null || echo "note: 'module load miniconda dSQ' failed -- load them yourself"
conda activate "$ENV"

echo "PREFLIGHT: are the two models matched, does the coupled sweep cross the onset, is the control flat?"
python experiments/coupling_harness.py --preflight || {
    echo
    echo "PREFLIGHT FAILED -- NOT submitting."
    exit 1
}

mkdir -p "$OUTDIR" logs
NTASKS=$(python experiments/coupling_harness.py --count)
echo
echo "coupling array: tasks=$NTASKS mem=$MEM time=$TIME maxjobs=$MAXJOBS -> $OUTDIR"

python - "$NTASKS" "$ENV" "$OUTDIR" <<'PY' > "$JOB"
import sys
n, env, outdir = int(sys.argv[1]), sys.argv[2], sys.argv[3]
setup = ('module load miniconda && source "$(conda info --base)/etc/profile.d/conda.sh" '
         f'&& conda activate {env}')
for i in range(n):
    print(f"{setup} && python experiments/coupling_harness.py --task-index {i} --outdir {outdir}")
PY
echo "  wrote $JOB ($NTASKS lines)"

dsq --job-file "$JOB" --batch-file "$BATCH" --partition day --account "pi_${NETID}" \
    --cpus-per-task 1 --mem-per-cpu "$MEM" --time "$TIME" --max-jobs "$MAXJOBS" \
    --output "logs/dsq-coupling-%A_%3a.out"

echo
echo "Created $BATCH.  Review it, then:  sbatch $BATCH"
echo
echo "When it lands:"
echo "    tar czf results_coupling.tgz $OUTDIR/"
echo "    sage -python experiments/collect_coupling.py"
echo
echo "The collector REFUSES to print if any row came back skipped_time -- that would mean the budget, not"
echo "the algorithm's own cap, decided who failed, and spc_gmr and pivot are last in the suite order."
