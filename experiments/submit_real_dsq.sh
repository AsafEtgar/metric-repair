#!/bin/bash
# Build a joblist + dSQ batch script for ONE real-data array on the Bouchet cluster. Sibling of
# submit_rgg_dsq.sh. Run FROM THE REPO ROOT. Does NOT submit -- prints the sbatch command to review first.
#
#   bash experiments/submit_real_dsq.sh <conda_env> <pi_netid> <array> [mem]
#     array = heur (589 tasks: det x1 + rand x30 per graph, short walltime)
#           | ilp  (32 tasks: gmr_ilp/iomr_ilp per distance-sensible graph, 17h walltime)
#     mem   = per-task memory (default 8g; dense ripe is the stress case -- bump to 16g if it OOMs)
#
# Prereqs on the cluster: (1) the conda env has numpy/scipy/networkx AND **pandas** (datasets.load_edgelist
# uses pd.read_csv -- the synthetic harness didn't need it); (2) the 19 processed graphs are present under
# data/processed/ (they are gitignored -- scp them up, see REAL_EXPERIMENTS.md runbook).
set -euo pipefail

ENV="${1:-metricrepair}"
NETID="${2:-CHANGE_ME}"
ARRAY="${3:-heur}"
MEM="${4:-8g}"
MAXJOBS=64
if [ "$ARRAY" = "ilp" ]; then TIME=17:30:00; else TIME=02:30:00; fi   # ilp: 17h cap + slack

OUTDIR=results_real
COVERS=results_real_covers
JOB="real_${ARRAY}_joblist.txt"
BATCH="dsq_real_${ARRAY}_submit.sh"

module load miniconda dSQ 2>/dev/null || echo "note: 'module load miniconda dSQ' failed -- load them yourself"
conda activate "$ENV"

# fail early if the input graphs / pandas aren't there (the two things unique to the real-data runs)
python - <<'PY' || { echo "PREREQ CHECK FAILED -- see message above"; exit 1; }
import glob, sys
try:
    import pandas  # noqa: F401
except ImportError:
    sys.exit("  pandas missing in this env -- `pip install pandas` (datasets.load_edgelist needs it)")
n = len(glob.glob("data/processed/*.csv"))
if n < 20:  # 19 graphs + REAL_GRAPHS_REPORT.csv
    sys.exit(f"  only {n} files in data/processed/ -- scp the 19 processed graphs up first")
print(f"  prereqs ok: pandas present, {n} files in data/processed/")
PY

mkdir -p "$OUTDIR" "$COVERS" logs
NTASKS=$(python experiments/run_real_task.py --array "$ARRAY" --count)
echo "real array=$ARRAY tasks=$NTASKS mem=$MEM time=$TIME -> $OUTDIR (covers in $COVERS)"

CONDA_SETUP='module load miniconda && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate '"$ENV"
python experiments/make_real_joblist.py --python python --array "$ARRAY" --outdir "$OUTDIR" \
    --covers "$COVERS" --joblist "$JOB" --setup "$CONDA_SETUP"

dsq --job-file "$JOB" \
    --batch-file "$BATCH" \
    --partition day \
    --account "pi_${NETID}" \
    --cpus-per-task 1 \
    --mem-per-cpu "$MEM" \
    --time "$TIME" \
    --max-jobs "$MAXJOBS" \
    --output "logs/dsq-real-${ARRAY}-%A_%3a.out"

echo
echo "Created $BATCH.  Review it, then submit with:"
echo "    sbatch $BATCH"
echo "Monitor:   squeue --me    |   dSQAutopsy $BATCH $JOB"
echo "Results land in $OUTDIR/ (CSVs) and $COVERS/ (repair covers, for the local GT-recovery pass)."
