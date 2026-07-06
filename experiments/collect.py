"""Concatenate all per-task CSVs into one tidy table.

    python experiments/collect.py [--indir results] [--out results_all.csv]
"""
import argparse
import csv
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results")
    ap.add_argument("--out", default="results_all.csv")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.indir, "task_*.csv")))
    if not files:
        print(f"no task_*.csv in {a.indir}")
        return
    header, n_rows = None, 0
    with open(a.out, "w", newline="") as out:
        writer = None
        for fp in files:
            with open(fp, newline="") as f:
                r = csv.DictReader(f)
                if writer is None:
                    header = r.fieldnames
                    writer = csv.DictWriter(out, fieldnames=header)
                    writer.writeheader()
                for row in r:
                    writer.writerow(row)
                    n_rows += 1
    print(f"{len(files)} files, {n_rows} rows -> {a.out}")


if __name__ == "__main__":
    main()
