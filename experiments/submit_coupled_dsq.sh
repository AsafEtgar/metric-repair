#!/bin/bash
# Build a joblist + dSQ batch for the COUPLED DENSITY SWEEP (exp2c). Run FROM THE REPO ROOT. Does NOT submit.
#
#   bash experiments/submit_coupled_dsq.sh <conda_env> <pi_netid> [mem]
#
# WHAT IT ASKS. Every density statement the paper makes on the large grid is a statement about the DECOUPLED
# weight model, because exp1 (coupled) pins p at {0.3, 0.5} and only exp2b (decoupled) sweeps it. This array
# supplies the missing cell: density moving, weights COUPLED (Geometric(1-p)).
#
# WHAT IT WILL FIND (a prediction, made before the run, from a 2-seed probe). Under coupling, density IS the
# weight spread -- the mean weight is 1/(1-p) -- so the two cannot be separated. The sweep should therefore
# show non-metricity switching on WITH density rather than independently of it:
#
#     alpha    p = 2n^-alpha       m        |H|/m
#     0.500       0.045          90k        0.002     <-- effectively metric
#     0.370       0.120         239k        0.014
#     0.274       0.250         499k        0.063
#     0.229       0.350         700k        0.123
#
# That entanglement is the result. It is also exactly why graph_models.py carries a decoupled model at all,
# and the array turns a modelling remark into a measurement.
#
# THE GRID WAS NEARLY WRONG. The obvious mirror of exp2a (alpha in [1/2, 2/3]) puts every point at
# |H|/m <= 0.003 -- metric graphs, nothing to repair, 320 tasks and ~800 core-hours to benchmark repair
# algorithms on inputs that need none. `--preflight` exists to make that impossible: it BUILDS each grid point
# and refuses to submit unless the sweep actually gets broken. Run it. It is not optional.
#
# WHAT IS DROPPED. gmr_bestofk and iomr_bestofk time out on 100% of exp2b's tasks -- 1800 s each, every task,
# zero data, 38% of the budget. They do not run here. The cost: we FORFEIT their return-rate cells rather than
# measuring them at 0%, and a return rate is only comparable across sweeps at an equal cap. Every other
# algorithm runs at the same 1800 s cap as exp1/exp2b, so every other cell IS comparable.
#
# NON-DESTRUCTIVE. harness.py is imported read-only and never modified -- changing a task-import path
# mid-campaign would invalidate 11,965 delivered tasks. Writes ONLY results_coupled/.
#
# COST. 320 tasks. Per task the suite costs ~2.6 core-hours at these edge counts, dominated by the 1800 s
# per-algorithm cap rather than by real work. The per-task budget is harness's 6 h; walltime is 8 h, and the
# budget MUST stay under the walltime or the CSV is lost outright (harness.py:56).
set -euo pipefail

ENV="${1:-metricrepair}"
NETID="${2:-CHANGE_ME}"
MEM="${3:-24g}"            # exp1 ran 563k edges in 16g; this array reaches 700k, so give it headroom
MAXJOBS=320                # every task at once -- the array is embarrassingly parallel and only 320 wide
TIME=08:00:00              # > the 6 h per-task budget (harness.py:56). Do not shrink one without the other.

OUTDIR=results_coupled
JOB="coupled_joblist.txt"
BATCH="dsq_coupled_submit.sh"

module load miniconda dSQ 2>/dev/null || echo "note: 'module load miniconda dSQ' failed -- load them yourself"
conda activate "$ENV"

# ---------------------------------------------------------------------------
# THE PREFLIGHT IS THE POINT. It generates every grid point for real and measures |H|/m. A sweep with nothing
# to repair is the one failure mode that costs a full cluster run and produces a table of zeros, and it is
# invisible until the results come back. Fail here, not there.
# ---------------------------------------------------------------------------
echo "PREFLIGHT: does this sweep actually contain broken graphs?"
python experiments/coupled_harness.py --preflight || {
    echo
    echo "PREFLIGHT FAILED -- NOT submitting. The grid does not contain graphs that need repair."
    exit 1
}

mkdir -p "$OUTDIR" logs
NTASKS=$(python experiments/coupled_harness.py --count)
echo
echo "coupled array (exp2c): tasks=$NTASKS mem=$MEM time=$TIME maxjobs=$MAXJOBS -> $OUTDIR"

python - "$NTASKS" "$ENV" "$OUTDIR" <<'PY' > "$JOB"
import sys
n, env, outdir = int(sys.argv[1]), sys.argv[2], sys.argv[3]
setup = ('module load miniconda && source "$(conda info --base)/etc/profile.d/conda.sh" '
         f'&& conda activate {env}')
for i in range(n):
    print(f"{setup} && python experiments/coupled_harness.py --task-index {i} --outdir {outdir}")
PY
echo "  wrote $JOB ($NTASKS lines)"

dsq --job-file "$JOB" --batch-file "$BATCH" --partition day --account "pi_${NETID}" \
    --cpus-per-task 1 --mem-per-cpu "$MEM" --time "$TIME" --max-jobs "$MAXJOBS" \
    --output "logs/dsq-coupled-%A_%3a.out"

echo
echo "Created $BATCH.  Review it, then:  sbatch $BATCH"
echo
echo "When it lands:"
echo "    tar czf results_coupled.tgz $OUTDIR/     # bring it back"
echo "    sage -python experiments/collect_coupled.py"
