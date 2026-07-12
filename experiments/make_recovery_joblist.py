"""Build the dSQ joblist for the recovery array (matched topology + geometry on the planted road network).

Sibling of make_real_joblist.py. One line per task; dSQ turns each into an array element.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recovery_harness import all_tasks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="python")
    ap.add_argument("--outdir", default="results_recovery")
    ap.add_argument("--joblist", default="recovery_joblist.txt")
    ap.add_argument("--setup", default="")
    a = ap.parse_args()

    tasks = all_tasks()
    with open(a.joblist, "w") as f:
        for i, (spec, algo) in enumerate(tasks):
            cmd = f"{a.python} experiments/recovery_harness.py --task-index {i} --outdir {a.outdir}"
            if a.setup:
                cmd = f"{a.setup} && {cmd}"
            f.write(cmd + "\n")
    print(f"  wrote {a.joblist}: {len(tasks)} tasks "
          f"({len({s[0] for s, _ in tasks})} planted specs x (1 observed + 15 algorithms))")
    print(f"  results -> {os.path.join(a.outdir, 'recovery_<idx>_<graph>_<algo>.csv')}")


if __name__ == "__main__":
    main()
