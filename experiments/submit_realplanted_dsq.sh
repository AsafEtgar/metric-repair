#!/bin/bash
# PLANTED corruptions in REAL metric bases, re-run against the fixed inflate. A SEPARATE array from the RGG.
# Run FROM THE REPO ROOT. Does NOT submit.
#
#   bash experiments/submit_realplanted_dsq.sh <conda_env> <pi_netid> [mem]
#
# THE GRID is rgg_harness's existing "realrec": 5 real bases x {inflate, deflate, mixed} x 4 fractions x 15
# seeds = 900 tasks, magnitude 5.0 throughout. Nothing new is invented; realplanted_harness.py adds only the
# preflight (rgg_harness is import-path code and must not be edited).
#
# HOW MUCH OF THIS ACTUALLY NEEDED RE-RUNNING -- MEASURED, NOT ASSUMED. The old inflate floor was an ABSOLUTE
# 1, so whether it bound depends on the scale of the DETOURS, and the preflight counts the edges it moved:
#
#     dimacs_ny_d        0.0% of inflated edges move   -- bit-identical under the fix
#     dimacs_ny_t        0.0%                          -- bit-identical
#     fish1_ten_lin      0.0%                          -- bit-identical
#     fish1_ten_log      0.0%                          -- bit-identical
#     pbmc3k_cosine_knn  8.8%, worst edge off by 3.2x  -- WRONG, and its inflate + mixed arms must be re-run
#
# Only pbmc3k was damaged. It is also the one place a MEDIAN would have lied: pbmc3k's median inflated edge is
# untouched while 8.8% of its edges are not. An absolute floor meets a DISTRIBUTION of detours -- it bites the
# short tail and leaves the middle alone, which is exactly the shape a median cannot see. The gate counts
# edges. (The RGG was hit so much harder because its detours are ~0.09: 5*0.09 = 0.45, well under the ~1.07
# floor. pbmc3k's cosine detours are large enough that 5*detour clears it -- usually.)
#
# The whole grid is submitted anyway. Re-running the four unaffected bases costs compute and changes nothing,
# but splicing two CSV vintages into one analysis is the provenance rot the gates exist to stop, and 900 tasks
# is cheap next to that risk. If you would rather not pay it, run pbmc3k alone -- the other four CSVs are still
# valid and the collector will say so.
#
# RESOURCES. dimacs_ny_d/t are n=5000 and the APSP dominates: 24 g, as the published realrec array used.
# Budget is rgg_harness's own task_budget("realrec") = 6 h; walltime must clear 6 h + TIMEOUT_S (30 min).
set -euo pipefail

ENV="${1:-metricrepair}"
NETID="${2:-CHANGE_ME}"
MEM="${3:-24g}"            # dimacs_ny n=5000: the APSP dict is the memory, not the repair
MAXJOBS=900                # embarrassingly parallel, and only 900 wide
TIME=08:00:00              # > the 6 h budget + the 30 min per-algorithm cap. Never shrink one alone.

OUTDIR=results_realplanted
JOB="realplanted_joblist.txt"
BATCH="dsq_realplanted_submit.sh"

module load miniconda dSQ 2>/dev/null || echo "note: 'module load miniconda dSQ' failed -- load them yourself"
conda activate "$ENV"

echo "PREFLIGHT: does every base load, is the magnitude knob live, and WHICH bases were actually damaged?"
python experiments/realplanted_harness.py --preflight || {
    echo
    echo "PREFLIGHT FAILED -- NOT submitting."
    exit 1
}

mkdir -p "$OUTDIR" logs
NTASKS=$(python experiments/realplanted_harness.py --count)
echo
echo "realplanted array: tasks=$NTASKS mem=$MEM time=$TIME maxjobs=$MAXJOBS -> $OUTDIR"

python - "$NTASKS" "$ENV" "$OUTDIR" <<'PY' > "$JOB"
import sys
n, env, outdir = int(sys.argv[1]), sys.argv[2], sys.argv[3]
setup = ('module load miniconda && source "$(conda info --base)/etc/profile.d/conda.sh" '
         f'&& conda activate {env}')
for i in range(n):
    print(f"{setup} && python experiments/realplanted_harness.py --task-index {i} --outdir {outdir}")
PY
echo "  wrote $JOB ($NTASKS lines)"

dsq --job-file "$JOB" --batch-file "$BATCH" --partition day --account "pi_${NETID}" \
    --cpus-per-task 1 --mem-per-cpu "$MEM" --time "$TIME" --max-jobs "$MAXJOBS" \
    --output "logs/dsq-realplanted-%A_%3a.out"

echo
echo "Created $BATCH.  Review it, then:  sbatch $BATCH"
echo
echo "When it lands:"
echo "    sage -python experiments/collect_realplanted.py"
