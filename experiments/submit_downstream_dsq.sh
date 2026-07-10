#!/bin/bash
# Submit the PURE real-data downstream-recovery experiment (does repair recover the true metric?).
# A tiny post-processing pass over the covers the real campaign already saved -- one task per graph, seconds
# to minutes each. Run FROM THE REPO ROOT. Does NOT submit; prints the sbatch line to review first.
#
#   bash experiments/submit_downstream_dsq.sh <conda_env> <pi_netid> [mem]
#
# PREREQS on the cluster (checked below):
#   1. the saved covers, results_real_covers/<graph>/*.txt  (produced by the real heur array)
#   2. the ground truth, data/processed/gt/{ripe_atlas__coords.csv, nmr_1d3z_*__truedist.npz}
#      -- these were built locally; scp data/processed/gt/ up if absent (it is ~1 MB).
#
# This experiment is light enough to run LOCALLY instead: pull the ripe/nmr cover folders down (a few dozen
# tiny files), then `sage -python experiments/run_downstream_task.py --graph ripe_atlas` etc. The cluster
# path exists only so it lives beside the rest of the pipeline.
set -euo pipefail

ENV="${1:-metricrepair}"
NETID="${2:-CHANGE_ME}"
MEM="${3:-8g}"
MAXJOBS=8
TIME=01:00:00
OUTDIR=results_downstream
JOB=downstream_joblist.txt
BATCH=dsq_downstream_submit.sh

module load miniconda dSQ 2>/dev/null || echo "note: 'module load miniconda dSQ' failed -- load them yourself"
conda activate "$ENV"

# prereq check -- the two things unique to this experiment
python - <<'PY' || { echo "PREREQ CHECK FAILED -- see message above"; exit 1; }
import glob, os, sys
sys.path.insert(0, "experiments")
from downstream_recovery import DOWNSTREAM_GRAPHS
miss = []
for g, kind in DOWNSTREAM_GRAPHS.items():
    gt = f"data/processed/gt/{g}__coords.csv" if kind == "coords" else f"data/processed/gt/{g}__truedist.npz"
    if not os.path.exists(gt):
        miss.append(gt)
    covdir = next((d for d in (f"results_real_covers/{g}", f"results_real/results_real_covers/{g}")
                   if os.path.isdir(d)), None)
    ncov = len(glob.glob(os.path.join(covdir, "*.txt"))) if covdir else 0
    print(f"  {g:<20} gt={'ok' if os.path.exists(gt) else 'MISSING'}  covers={ncov}")
if miss:
    sys.exit("  MISSING ground-truth files: %s\n  scp data/processed/gt/ up from your Mac (~1 MB)." % miss)
PY

mkdir -p "$OUTDIR" logs
N=$(python experiments/run_downstream_task.py --count)
echo "downstream: $N graphs -> $OUTDIR"

CONDA_SETUP='module load miniconda && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate '"$ENV"
: > "$JOB"
for i in $(seq 0 $((N-1))); do
    echo "$CONDA_SETUP && python experiments/run_downstream_task.py --task-index $i --outdir $OUTDIR" >> "$JOB"
done

dsq --job-file "$JOB" --batch-file "$BATCH" --partition day --account "pi_${NETID}" \
    --cpus-per-task 1 --mem-per-cpu "$MEM" --time "$TIME" --max-jobs "$MAXJOBS" \
    --output "logs/dsq-downstream-%A_%3a.out"

echo
echo "Created $BATCH.  Review it, then:  sbatch $BATCH"
echo "Collect + report:  python experiments/downstream_analyze.py --indir $OUTDIR --outdir analysis"
