"""make_ilp_rerun.py -- find the real (graph, variant) exact ILPs that lack a TRUSTWORTHY optimum
(missing CSV, timed out, or unverified) and emit a dSQ joblist to rerun exactly those with ample
resources. Run on the cluster, where results_real/ holds the ILP CSVs.

    python experiments/make_ilp_rerun.py --results results_real > ilp_rerun.txt
    # inspect the per-task verdicts printed to stderr, then submit ilp_rerun.txt (see below)

A pair is TRUSTWORTHY exactly when its `<graph>__{gmr,iomr}_ilp.csv` has the ILP row with
status == "ok", converged == True, and valid == 1 -- the same test real_analyze.py uses to accept an
ILP as the exact optimum. Everything else (no file, a timed-out/partial run, or an unverified cover)
gets a long rerun. Each emitted line runs one ILP task with ILP_TIMEOUT_S overridden to 47h; the
submit script (below) gives it the 2-day walltime and the memory. If the rerun ALSO fails to converge
in 47h at high memory, its CSV records that -- which is the evidence for "infeasible even with ample
resources", not a gap.

    # submit (adjust partition to one that allows a 2-day walltime + big memory on your cluster):
    SETUP='module load miniconda && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate metricrepair'
    module load dSQ
    dsq --job-file ilp_rerun.txt --batch-file dsq_ilp_long.sh --partition week --account pi_ag245 \
        --cpus-per-task 1 --mem 240g --time 2-00:00:00 --output logs/dsq-ilp-%A_%3a.out
    sbatch dsq_ilp_long.sh
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from real_harness import ilp_tasks                                                # noqa: E402

SETUP = ('module load miniconda && source "$(conda info --base)/etc/profile.d/conda.sh" '
         '&& conda activate metricrepair')
LONG_TIMEOUT_S = 47 * 3600                # per-component ILP cap, just under the 2-day walltime


def _is_true(x):
    return str(x).strip() in ("True", "true", "1", "1.0")


def verdict(results_dir, graph, algo):
    """(trustworthy_exact, reason) for one (graph, algo) ILP, matching real_analyze.exact_ok()."""
    path = os.path.join(results_dir, f"{graph}__{algo}.csv")
    if not os.path.exists(path):
        return False, "missing csv"
    try:
        d = pd.read_csv(path)
    except Exception as e:                                            # noqa: BLE001
        return False, f"unreadable ({type(e).__name__})"
    r = d[d.get("algo") == algo] if "algo" in d.columns else d
    if r.empty:
        return False, "no ILP row"
    r = r.iloc[0]
    status, conv, valid = r.get("status"), r.get("converged"), r.get("valid")
    if status == "ok" and _is_true(conv) and (pd.notna(valid) and float(valid) == 1):
        return True, f"exact (size={r.get('size')})"
    return False, f"status={status} converged={conv} valid={valid}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_real", help="dir with the <graph>__{gmr,iomr}_ilp.csv")
    ap.add_argument("--outdir", default="results_real", help="where the reruns write (default: overwrite)")
    ap.add_argument("--covers", default="results_real_covers")
    a = ap.parse_args()

    tasks = ilp_tasks()
    need = []
    print(f"# checking {len(tasks)} ILP tasks in {a.results}/ ...", file=sys.stderr)
    for idx, (graph, algo, _seed) in enumerate(tasks):
        ok, why = verdict(a.results, graph, algo)
        print(f"#   {'OK   ' if ok else 'RERUN'} idx={idx:2d}  {graph:26s} {algo:9s}  [{why}]", file=sys.stderr)
        if not ok:
            need.append((idx, graph, algo))

    for idx, graph, algo in need:
        print(f"{SETUP} && ILP_TIMEOUT_S={LONG_TIMEOUT_S} python experiments/run_real_task.py "
              f"--array ilp --task-index {idx} --outdir {a.outdir} --covers {a.covers}")

    print(f"\n# {len(need)} of {len(tasks)} ILP tasks need a long rerun "
          f"({len(need)} lines written to stdout).", file=sys.stderr)
    if not need:
        print("# all ILP optima are already trustworthy -- nothing to submit.", file=sys.stderr)


if __name__ == "__main__":
    main()
