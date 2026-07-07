"""Write a joblist for one real-data dSQ array (heur | ilp). See REAL_EXPERIMENTS.md.

    python experiments/make_real_joblist.py --array heur --outdir results_real --covers results_real_covers
        [--python python] [--joblist real_heur_joblist.txt]
        [--setup 'module load miniconda && conda activate metricrepair']

Each line is `python experiments/run_real_task.py --array A --task-index K --outdir ... --covers ...`,
optionally prefixed by --setup so every task self-activates its conda env on the compute node.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_harness import all_tasks   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", choices=["heur", "ilp"], required=True)
    ap.add_argument("--python", default="python")
    ap.add_argument("--outdir", default="results_real")
    ap.add_argument("--covers", default="results_real_covers")
    ap.add_argument("--joblist", default=None)
    ap.add_argument("--setup", default="", help="shell prefix prepended (with &&) to every task command")
    a = ap.parse_args()
    joblist = a.joblist or f"real_{a.array}_joblist.txt"
    n = len(all_tasks(a.array))
    prefix = f"{a.setup} && " if a.setup else ""
    with open(joblist, "w") as f:
        for i in range(n):
            f.write(f"{prefix}{a.python} experiments/run_real_task.py --array {a.array} "
                    f"--task-index {i} --outdir {a.outdir} --covers {a.covers}\n")
    print(f"{n} tasks ({a.array}) -> {joblist}  (outdir={a.outdir}, covers={a.covers})")


if __name__ == "__main__":
    main()
