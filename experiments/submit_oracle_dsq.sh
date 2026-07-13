#!/bin/bash
# Build a joblist + dSQ batch for the ORACLE-WEIGHTS array. Run FROM THE REPO ROOT. Does NOT submit.
#
#   bash experiments/submit_oracle_dsq.sh <conda_env> <pi_netid> [mem]
#
# WHAT IT ASKS. On a planted graph we can cross the two choices in a repair -- WHICH edges, and WHAT weights --
# because we know both the corrupted set and the true weights. On real data neither exists. So we define the
# true weight of an edge as d*(u,v), its distance in the external ground truth, and ask: given the TRUE
# weights for the |S| edges an algorithm selected, how much of the recoverable geometry does it capture?
#
# WHAT IT WILL FIND (a prediction, made before the run). Two cheap diagnostics say the answer is "almost
# nothing", and this array is here to turn that into a measurement:
#   * the error is NOT in the heavy set. err(e) = |w(e) - d*(e)|, and H carries its PROPORTIONAL share and no
#     more: concentration 1.0x on nmr_residue, 0.7x on nmr_atom, 0.0x on pbmc3k.
#   * the error is worth fixing. Set EVERY edge to d* and 58% of the disparity comes back on nmr_residue, 70%
#     on dimacs_ny_t.
#   Smoke-tested: gmr_ilp's 14-edge cover on nmr_residue, handed the true weights, captures 1.9% of the gain
#   -- roughly its 4.5% share of the edges. DOMR's cover captures ZERO. The heavy set is not the wrong set.
#
# 80 tasks = 5 real graphs x 16 algorithms. One task loads its graph and truth ONCE and scores every saved
# cover of that algorithm (1 for the deterministic methods, 30 seeds for the randomized ones), under two
# corrections: `restore` (the canonical rule -- it reproduces the existing pipeline, and that reproduction is
# the gate) and `oracle` (the new arm). Per graph it also writes `observed` and `all_oracle` (the ceiling).
#
# NON-DESTRUCTIVE, BY CONSTRUCTION. Reads data/processed/, data/processed/gt/ and results_real/. Writes ONLY
# results_oracle/. No existing CSV, figure, or paper number is touched.
#
# COST. ripe_atlas is the long pole: n=999 but m=442,707, so its APSP is the expensive one. dimacs_ny_t pays
# for a 5000x5000 classical MDS of the truth, once per task. 2h walltime is generous; 16g covers the 5000^2
# float64 matrices.
set -euo pipefail

ENV="${1:-metricrepair}"
NETID="${2:-CHANGE_ME}"
MEM="${3:-16g}"
MAXJOBS=80
TIME=02:00:00

OUTDIR=results_oracle
JOB="oracle_joblist.txt"
BATCH="dsq_oracle_submit.sh"

module load miniconda dSQ 2>/dev/null || echo "note: 'module load miniconda dSQ' failed -- load them yourself"
conda activate "$ENV"

python - <<'PY' || { echo "PREREQ CHECK FAILED"; exit 1; }
import os, sys, glob
try:
    import pandas, scipy, networkx  # noqa: F401
except ImportError as e:
    sys.exit(f"  missing package: {e}")
need = ["data/processed/gt", "results_real/results_real_covers"]
for d in need:
    if not os.path.isdir(d):
        sys.exit(f"  {d}/ missing -- scp it up. The gt/ dir IS the ground truth; the covers are the input.")
n = sum(len(glob.glob(f"results_real/results_real_covers/{g}/*.txt"))
        for g in ("nmr_1d3z_residue","nmr_1d3z_atom","dimacs_ny_t","pbmc3k_cosine_knn","ripe_atlas"))
if n < 600:
    sys.exit(f"  only {n} saved covers found (expected 619) -- the covers directory is incomplete")
print(f"  prereqs ok: ground truth present, {n} saved covers")
PY

mkdir -p "$OUTDIR" logs
NTASKS=$(python experiments/oracle_weights.py --count)
echo "oracle array: tasks=$NTASKS mem=$MEM time=$TIME -> $OUTDIR"

CONDA_SETUP='module load miniconda && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate '"$ENV"
python - "$NTASKS" <<'PY' > "$JOB"
import sys
n = int(sys.argv[1])
setup = 'module load miniconda && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ENVNAME'
for i in range(n):
    print(f"{setup} && python experiments/oracle_weights.py --task-index {i} --outdir results_oracle")
PY
sed -i.bak "s/ENVNAME/$ENV/g" "$JOB" && rm -f "$JOB.bak"
echo "  wrote $JOB ($NTASKS lines)"

dsq --job-file "$JOB" --batch-file "$BATCH" --partition day --account "pi_${NETID}" \
    --cpus-per-task 1 --mem-per-cpu "$MEM" --time "$TIME" --max-jobs "$MAXJOBS" \
    --output "logs/dsq-oracle-%A_%3a.out"

echo
echo "Created $BATCH.  Review it, then:  sbatch $BATCH"
echo "When it lands, bring $OUTDIR/ back and run:"
echo "    sage -python experiments/collect_oracle.py"
echo "It prints nothing until DOMR reproduces `observed` and every graph has its ceiling row."
