#!/usr/bin/env python
"""Merge all per-task result CSVs in results/ into one tidy table.

Run with any Python that has pandas:  python process_results.py [output.csv]

Scatter-gather pattern: each experiment task writes its own file (no parallel-write races),
then this script concatenates them for analysis/plotting. Picks up both the CLI's plain `.csv`
and save_results()'s gzipped `.csv.gz`.
"""
import glob, os, sys
import pandas as pd

RESULTS_DIR = "results"
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS_DIR, "all_results.csv")

files = sorted(f for f in glob.glob(os.path.join(RESULTS_DIR, "*.csv")) +
               glob.glob(os.path.join(RESULTS_DIR, "*.csv.gz"))
               if os.path.basename(f) != os.path.basename(out))
if not files:
    sys.exit(f"no result CSVs found in {RESULTS_DIR}/")

df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
df.to_csv(out, index=False)
print(f"merged {len(files)} files, {len(df)} rows -> {out}")
