#!/bin/bash
# Build an RGG joblist + dSQ batch script for the Bouchet cluster. Sibling of submit_dsq.sh.
# Run FROM THE REPO ROOT. Does NOT submit -- prints the sbatch command so you can review first.
#
#   bash experiments/submit_rgg_dsq.sh <conda_env> <pi_netid> [grid] [mem]
#     grid = poc (n=100..250, 30 seeds, ~960 tasks)  |  full (n=500 sweeps, 40 seeds, 3160 tasks)
#     mem  = per-task memory (default 4g for poc, 8g for full -- the old 1g OOM'd at n=500)
#
# Prereqs: a conda env with numpy/scipy/networkx.
set -euo pipefail

ENV="${1:-metricrepair}"
NETID="${2:-CHANGE_ME}"
GRID="${3:-poc}"
if [ "$GRID" = "poc" ]; then DEFMEM=4g; else DEFMEM=8g; fi
MEM="${4:-$DEFMEM}"
MAXJOBS=64

if [ "$GRID" = "full" ]; then
    OUTDIR=results_rgg;         JOB=rgg_joblist.txt;             BATCH=dsq_rgg_submit.sh
else
    OUTDIR="results_rgg_$GRID"; JOB="rgg_${GRID}_joblist.txt";   BATCH="dsq_rgg_${GRID}_submit.sh"
fi

module load miniconda dSQ 2>/dev/null || echo "note: 'module load miniconda dSQ' failed -- load them yourself"
conda activate "$ENV"

mkdir -p "$OUTDIR" logs
NTASKS=$(python experiments/run_rgg_task.py --grid "$GRID" --count)
echo "rgg grid=$GRID tasks=$NTASKS mem=$MEM -> $OUTDIR"

# --setup bakes env activation into EVERY task line (dSQ runs each line in a bare non-interactive shell that
# doesn't inherit this env; `conda activate` alone errors there, so source conda.sh first).
CONDA_SETUP='module load miniconda && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate '"$ENV"
python experiments/make_rgg_joblist.py --python python --grid "$GRID" --outdir "$OUTDIR" --joblist "$JOB" \
    --setup "$CONDA_SETUP"

dsq --job-file "$JOB" \
    --batch-file "$BATCH" \
    --partition day \
    --account "pi_${NETID}" \
    --cpus-per-task 1 \
    --mem-per-cpu "$MEM" \
    --time 02:30:00 \
    --max-jobs "$MAXJOBS" \
    --output "logs/dsq-rgg-${GRID}-%A_%3a.out"

echo
echo "Created $BATCH.  Review it, then submit with:"
echo "    sbatch $BATCH"
echo "Monitor:   squeue --me    |   dSQAutopsy $BATCH $JOB"
echo "Collect:   python experiments/collect.py --indir $OUTDIR --out results_rgg_${GRID}_all.csv"
