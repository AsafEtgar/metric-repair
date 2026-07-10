#!/bin/bash
# Build the joblist and create a Dead-Simple-Queue (dSQ) batch script for the Bouchet cluster.
# Run this FROM THE REPO ROOT.  It does NOT submit -- it prints the sbatch command so you can review first.
#
#   bash experiments/submit_dsq.sh <conda_env> <pi_netid> [grid] [mem]
#     grid = small (Exp1 n=100..300, Exp2 n=300, 30 seeds)  |  full (n up to 500, 40 seeds)
#     mem  = per-task memory (default 4g for small, 8g for full -- the old 1g OOM'd at n=500)
#
# Prereqs: a conda env with numpy/scipy/networkx (the pure-Python port needs no Sage on the cluster).
set -euo pipefail

ENV="${1:-metricrepair}"           # conda env name
NETID="${2:-CHANGE_ME}"            # PI's netid, for -A pi_<netid>
GRID="${3:-small}"
if [ "$GRID" = "small" ]; then DEFMEM=4g; elif [ "$GRID" = "large" ]; then DEFMEM=16g; else DEFMEM=8g; fi
MEM="${4:-$DEFMEM}"
# 8h: harness.TASK_BUDGET gives `large` a 6h per-task budget so pivot/spc/left_edge are actually measured
# instead of starved by the covering-LP family running before them. Budget must stay inside the walltime.
if [ "$GRID" = "large" ]; then TIME=08:00:00; else TIME=02:30:00; fi
MAXJOBS=64

if [ "$GRID" = "full" ]; then
    OUTDIR=results;         JOB=joblist.txt;             BATCH=dsq_submit.sh
else
    OUTDIR="results_$GRID"; JOB="${GRID}_joblist.txt";   BATCH="dsq_${GRID}_submit.sh"
fi

module load miniconda dSQ 2>/dev/null || echo "note: 'module load miniconda dSQ' failed -- load them yourself"
conda activate "$ENV"

mkdir -p "$OUTDIR" logs
NTASKS=$(python experiments/run_task.py --grid "$GRID" --count)
echo "grid=$GRID tasks=$NTASKS mem=$MEM -> $OUTDIR"

# --setup bakes env activation into EVERY task line: dSQ runs each line in a bare non-interactive shell that
# doesn't inherit this env; `conda activate` alone errors there, so source conda.sh first.
CONDA_SETUP='module load miniconda && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate '"$ENV"
python experiments/make_joblist.py --python python --grid "$GRID" --outdir "$OUTDIR" --joblist "$JOB" \
    --setup "$CONDA_SETUP"

# --time is PER TASK; the harness caps each algorithm at 30 min and each task at 120 min, so 2:30:00 leaves
# slack. --mem-per-cpu bumped off 1g because n=500 OOM'd; 4-8g fits.
dsq --job-file "$JOB" \
    --batch-file "$BATCH" \
    --partition day \
    --account "pi_${NETID}" \
    --cpus-per-task 1 \
    --mem-per-cpu "$MEM" \
    --time "$TIME" \
    --max-jobs "$MAXJOBS" \
    --output "logs/dsq-${GRID}-%A_%3a.out"

echo
echo "Created $BATCH.  Review it, then submit with:"
echo "    sbatch $BATCH"
echo "Monitor:   squeue --me    |   dSQAutopsy $BATCH $JOB"
echo "Collect:   python experiments/collect.py --indir $OUTDIR --out results_${GRID}_all.csv"
