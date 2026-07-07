#!/bin/bash
# Build the joblist and create a Dead-Simple-Queue (dSQ) batch script for the Bouchet cluster.
# Run this FROM THE REPO ROOT.  It does NOT submit -- it prints the sbatch command so you can review first.
#
#   bash experiments/submit_dsq.sh <conda_env> <pi_netid>
#
# Prereqs: a conda env with numpy/scipy/networkx (the pure-Python port needs no Sage on the cluster).
set -euo pipefail

ENV="${1:-metricrepair}"           # conda env name
NETID="${2:-CHANGE_ME}"            # your netid, for -A pi_<netid>
MAXJOBS=64                         # concurrent tasks (cores); well under the day-partition cap (1024)
OUTDIR=results

module load miniconda dSQ 2>/dev/null || echo "note: 'module load miniconda dSQ' failed -- load them yourself"
conda activate "$ENV"

mkdir -p "$OUTDIR" logs
NTASKS=$(python experiments/run_task.py --count)
echo "tasks: $NTASKS"

# --setup bakes env activation into EVERY task line: dSQ runs each line in a bare, non-interactive shell
# that does NOT inherit this login-shell env, so a plain `python` there is system Python (no numpy) ->
# instant failure. `conda activate` also needs conda's shell functions, which `module load` does NOT
# install in a non-interactive shell (-> "Run 'conda init' before ..."), so we source conda.sh first.
CONDA_SETUP='module load miniconda && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate '"$ENV"
python experiments/make_joblist.py --python python --outdir "$OUTDIR" --joblist joblist.txt \
    --setup "$CONDA_SETUP"

# dSQ turns joblist.txt into an array batch script; --max-jobs throttles concurrency to $MAXJOBS.
# --time is PER TASK (one graph = the whole 19-algorithm suite); the harness caps each algorithm at 30 min
# and each task at 120 min, so 2:30:00 leaves slack.  --mem-per-cpu 1g matches the memory request.
dsq --job-file joblist.txt \
    --batch-file dsq_submit.sh \
    --partition day \
    --account "pi_${NETID}" \
    --cpus-per-task 1 \
    --mem-per-cpu 8g \
    --time 02:30:00 \
    --max-jobs "$MAXJOBS" \
    --output "logs/dsq-%A_%3a.out"

echo
echo "Created dsq_submit.sh.  Review it, then submit with:"
echo "    sbatch dsq_submit.sh"
echo "Monitor:   squeue --me    |   dSQAutopsy dsq_submit.sh joblist.txt"
echo "Collect:   python experiments/collect.py --indir $OUTDIR --out results_all.csv"
