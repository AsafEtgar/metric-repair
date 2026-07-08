"""Write rgg_joblist.txt: one command per RGG task, for Dead-Simple-Queue (dSQ) on the cluster.

    python experiments/make_rgg_joblist.py [--python python] [--outdir results_rgg]
        [--joblist rgg_joblist.txt] [--setup 'module load miniconda && conda activate metricrepair']

Mirrors make_joblist.py; each line self-activates the conda env on the compute node (dSQ runs each line in
a bare shell that does NOT inherit the login-shell env). Feed rgg_joblist.txt to dSQ (see RGG_EXPERIMENTS.md).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rgg_harness import all_tasks   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="python", help="interpreter on the cluster (conda env python)")
    ap.add_argument("--outdir", default="results_rgg")
    ap.add_argument("--joblist", default="rgg_joblist.txt")
    ap.add_argument("--grid", default="full", choices=["full", "poc", "large"], help="which task grid")
    ap.add_argument("--setup", default="", help="shell prefix prepended (with &&) to every task command")
    a = ap.parse_args()
    n = len(all_tasks(a.grid))
    prefix = f"{a.setup} && " if a.setup else ""
    with open(a.joblist, "w") as f:
        for i in range(n):
            f.write(f"{prefix}{a.python} experiments/run_rgg_task.py --task-index {i} "
                    f"--outdir {a.outdir} --grid {a.grid}\n")
    print(f"{n} rgg tasks ({a.grid}) -> {a.joblist}  (outdir={a.outdir})")


if __name__ == "__main__":
    main()
