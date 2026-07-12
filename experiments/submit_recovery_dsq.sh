#!/bin/bash
# Build a joblist + dSQ batch script for the RECOVERY (topology) array. Run FROM THE REPO ROOT.
# Does NOT submit -- it prints the sbatch command so the batch file can be reviewed first.
#
#   bash experiments/submit_recovery_dsq.sh <conda_env> <pi_netid> [mem]
#
# WHAT THIS ARRAY IS. 48 tasks = 3 planted corruptions of the NY road network (inflate / deflate / mixed,
# n=5000, planted on the exactly-metric dimacs_ny_d base) x (1 `observed` baseline + 15 repair algorithms).
# Each task computes, FROM ONE COVER ON ONE REPAIRED DISTANCE MATRIX:
#   * topology       -- k-NN Jaccard against geography, at k = 5, 10, 20
#   * geometry       -- Procrustes disparity against the same geography
#   * cover quality  -- |S|, and precision/recall against the PLANTED edit set
#
# WHY. The recovery table's three planted rows have a geometry number but no topology number, because a
# matched k-NN needs the cover suite re-run at n = 5000. We fill them here rather than borrow a k-NN from the
# earlier realrec downstream pass, which used magnitude 5.0 while the MDS sweep used 3.0 -- quoting one beside
# the other would be a confound dressed as a result. Because both axes come out of one cover in one process,
# a k-NN entry and a disparity entry on the same row describe the same repair of the same graph.
#
# THE GATE. Task 0 is the `observed` row of dimacs_ny_d_inflate and MUST reproduce disp = 0.364054, the value
# in analysis/summary_mds_sweep.csv; every algorithm's disparity must likewise reproduce its stored value. If
# not, the instance rebuild drifted and NOTHING HERE IS VALID. collect_recovery.py refuses to print until it
# passes -- verified locally: task 0 -> 0.364054, task 14 (gmr_thr_naive) -> 0.239225, both exact.
#
# COST. At n=5000 every algorithm hits the HUGE_N cap of 300 s, so the cover is bounded; the rest is the
# instance build (two |H| computations plus a classical MDS of a 5000x5000 truth matrix) and one scoring pass.
# Measured locally: ~10-25 min per task. 2h walltime is ample. Memory: several 5000x5000 float64 matrices live
# at once (~200 MB each), so 16g.
#
# Prereqs: conda env with numpy/scipy/networkx/pandas; data/processed/dimacs_ny_d.csv (the base graph) and
# data/processed/gt/dimacs_ny_d__coords.csv (that file IS the ground truth). Both are gitignored -- scp up.
set -euo pipefail

ENV="${1:-metricrepair}"
NETID="${2:-CHANGE_ME}"
MEM="${3:-16g}"
MAXJOBS=48
TIME=02:00:00

OUTDIR=results_recovery
JOB="recovery_joblist.txt"
BATCH="dsq_recovery_submit.sh"

module load miniconda dSQ 2>/dev/null || echo "note: 'module load miniconda dSQ' failed -- load them yourself"
conda activate "$ENV"

python - <<'PY' || { echo "PREREQ CHECK FAILED -- see message above"; exit 1; }
import os, sys
try:
    import pandas, scipy, networkx  # noqa: F401
except ImportError as e:
    sys.exit(f"  missing package in this env: {e}")
for f in ("data/processed/dimacs_ny_d.csv", "data/processed/gt/dimacs_ny_d__coords.csv"):
    if not os.path.exists(f):
        sys.exit(f"  {f} missing -- scp it up (it is gitignored). The coords file IS the ground truth.")
print("  prereqs ok: packages present, base graph and ground truth on disk")
PY

mkdir -p "$OUTDIR" logs
NTASKS=$(python experiments/recovery_harness.py --count)
echo "recovery array: tasks=$NTASKS mem=$MEM time=$TIME -> $OUTDIR"

CONDA_SETUP='module load miniconda && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate '"$ENV"
python experiments/make_recovery_joblist.py --python python --outdir "$OUTDIR" \
    --joblist "$JOB" --setup "$CONDA_SETUP"

dsq --job-file "$JOB" \
    --batch-file "$BATCH" \
    --partition day \
    --account "pi_${NETID}" \
    --cpus-per-task 1 \
    --mem-per-cpu "$MEM" \
    --time "$TIME" \
    --max-jobs "$MAXJOBS" \
    --output "logs/dsq-recovery-%A_%3a.out"

echo
echo "Created $BATCH.  Review it, then submit with:"
echo "    sbatch $BATCH"
echo "Monitor:   squeue --me    |   dSQAutopsy $BATCH $JOB"
echo
echo "When it lands, bring $OUTDIR/ back and run:"
echo "    python experiments/collect_recovery.py"
echo "It will not print a single number until task 0 reproduces disp = 0.364054."
