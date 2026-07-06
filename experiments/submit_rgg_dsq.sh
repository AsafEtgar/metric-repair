#!/bin/bash
# Build the RGG joblist and a Dead-Simple-Queue (dSQ) batch script for the Bouchet cluster.
# Sibling of submit_dsq.sh (RGG_EXPERIMENTS.md). Run this FROM THE REPO ROOT.
# It does NOT submit -- it prints the sbatch command so you can review first.
#
#   bash experiments/submit_rgg_dsq.sh <conda_env> <pi_netid>
#
# Prereqs: a conda env with numpy/scipy/networkx (the pure-Python port needs no Sage on the cluster).
set -euo pipefail

ENV="${1:-metricrepair}"           # conda env name
NETID="${2:-CHANGE_ME}"            # your netid, for -A pi_<netid>
MAXJOBS=64                         # concurrent tasks (cores); well under the day-partition cap (1024)
OUTDIR=results_rgg

module load miniconda dSQ 2>/dev/null || echo "note: 'module load miniconda dSQ' failed -- load them yourself"
conda activate "$ENV"

mkdir -p "$OUTDIR" logs
NTASKS=$(python experiments/run_rgg_task.py --count)
echo "rgg tasks: $NTASKS"

# --setup bakes env activation into EVERY task line: dSQ runs each line in a bare, non-interactive shell
# that does NOT inherit this login-shell env, and `conda activate` alone errors there ("Run 'conda init'
# before ...") because module load doesn't install conda's shell functions -- so source conda.sh first.
CONDA_SETUP='module load miniconda && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate '"$ENV"
python experiments/make_rgg_joblist.py --python python --outdir "$OUTDIR" --joblist rgg_joblist.txt \
    --setup "$CONDA_SETUP"

# dsq turns rgg_joblist.txt into an array batch script; --max-jobs throttles concurrency to $MAXJOBS.
# --time is PER TASK (one graph = the whole 15-algorithm suite, + kNN for Part 2); the harness caps each
# algorithm at 30 min and each task at 120 min, so 2:30:00 leaves slack. --mem-per-cpu 1g matches.
dsq --job-file rgg_joblist.txt \
    --batch-file dsq_rgg_submit.sh \
    --partition day \
    --account "pi_${NETID}" \
    --cpus-per-task 1 \
    --mem-per-cpu 1g \
    --time 02:30:00 \
    --max-jobs "$MAXJOBS" \
    --output "logs/dsq-rgg-%A_%3a.out"

echo
echo "Created dsq_rgg_submit.sh.  Review it, then submit with:"
echo "    sbatch dsq_rgg_submit.sh"
echo "Monitor:   squeue --me    |   dSQAutopsy dsq_rgg_submit.sh rgg_joblist.txt"
echo "Collect:   python experiments/collect.py --indir $OUTDIR --out results_rgg_all.csv"
