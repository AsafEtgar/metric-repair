"""Concatenate all per-task CSVs into one tidy table.

    python experiments/collect.py [--indir results] [--out results_all.csv]
                                  [--grid GRID --harness {geometric,rgg}] [--max-age-days N]

Two hazards this guards against (AUDIT_REPORT.md A9):

1. STALE FILES. Output directories are REUSED across re-runs. `results_rgg/` held 3160 CSVs from the old
   `full` grid before that grid grew to 5000 tasks. A re-run overwrites `task_%06d.csv` only for the tasks
   that SUCCEED -- any task that dies leaves the PREVIOUS run's CSV sitting there, produced by older code
   (e.g. before the deflate fix, when deflate was a silent no-op). A blind glob cannot tell the two apart.
   `--grid` restricts ingestion to the indices the current grid actually defines; `--max-age-days` refuses
   files older than the current run.

2. RAGGED HEADERS. The old code took the header from the FIRST file and fed a DictWriter whose default
   `extrasaction='raise'`. A later file with an extra column crashed collect mid-write, leaving a truncated
   output; a file with fewer columns was silently blank-padded. We now verify every file shares one schema.
"""
import argparse
import csv
import glob
import os
import sys
import time


def _expected_indices(harness, grid):
    """The task indices the CURRENT grid defines. Anything else in the directory is stale or orphaned."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if harness == "rgg":
        from rgg_harness import all_tasks
    else:
        from harness import all_tasks
    return set(range(len(all_tasks(grid))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results")
    ap.add_argument("--out", default="results_all.csv")
    ap.add_argument("--grid", default=None, help="restrict to this grid's task indices (needs --harness)")
    ap.add_argument("--harness", default="geometric", choices=["geometric", "rgg"])
    ap.add_argument("--max-age-days", type=float, default=None,
                    help="refuse task CSVs older than this (catches pre-fix survivors from a reused dir)")
    ap.add_argument("--allow-stale", action="store_true", help="warn instead of aborting on stale/orphan files")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.indir, "task_*.csv")))
    if not files:
        print(f"no task_*.csv in {a.indir}")
        return

    problems = []

    # --- orphan check: a file whose index the current grid does not define ----------------------------
    if a.grid:
        want = _expected_indices(a.harness, a.grid)
        orphans = [f for f in files
                   if int(os.path.basename(f)[5:-4]) not in want]
        if orphans:
            problems.append(f"{len(orphans)} orphan CSVs (index not in grid '{a.grid}'): "
                            f"{[os.path.basename(f) for f in orphans[:5]]}")
            files = [f for f in files if f not in set(orphans)]

    # --- staleness check: survivors of an older run, left behind where a task failed ------------------
    if a.max_age_days is not None:
        cutoff = time.time() - a.max_age_days * 86400
        stale = [f for f in files if os.path.getmtime(f) < cutoff]
        if stale:
            problems.append(f"{len(stale)} CSVs older than {a.max_age_days}d -- likely pre-fix survivors: "
                            f"{[os.path.basename(f) for f in stale[:5]]}")
            files = [f for f in files if f not in set(stale)]

    if problems:
        for p in problems:
            print(f"  !! {p}")
        if not a.allow_stale:
            sys.exit("ABORTING: refusing to build a combined CSV from a mixed-provenance directory.\n"
                     "  Inspect them, delete or quarantine them, then re-run. --allow-stale to skip anyway.")
        print("  (--allow-stale: the files above were EXCLUDED from the output)")

    # --- schema check: every file must agree on the header --------------------------------------------
    headers = {}
    for fp in files:
        with open(fp, newline="") as f:
            headers[fp] = tuple(csv.DictReader(f).fieldnames or ())
    distinct = set(headers.values())
    if len(distinct) > 1:
        ref = headers[files[0]]
        odd = [os.path.basename(f) for f, h in headers.items() if h != ref][:5]
        sys.exit(f"ABORTING: {len(distinct)} distinct CSV schemas in {a.indir} (e.g. {odd}).\n"
                 "  Mixing harnesses or code versions. The old code would have silently blank-padded or crashed.")

    header = list(distinct.pop())

    # --- suite check: every task must have run the SAME set of algorithms -------------------------------
    # Matching columns are not enough. dSQ reads the working copy when each task LAUNCHES, so editing the
    # suite mid-array silently splits the campaign: results_small has 1068 task CSVs with 18 algorithms and
    # 1392 with 21 (the three GMR covering analogues were added partway through), all on the same date, same
    # header. The affected algorithms' medians then rest on a non-randomly chosen subset -- whichever tasks
    # happened to run after the edit. Row sets, not column sets, are the thing to compare.
    n_rows = 0
    suites, rows_by_file = {}, {}
    for fp in files:
        with open(fp, newline="") as f:
            rows = list(csv.DictReader(f))
        rows_by_file[fp] = rows
        if rows and "algo" in rows[0]:
            suites[fp] = frozenset(r["algo"] for r in rows)
    if len(set(suites.values())) > 1:
        from collections import Counter
        counts = Counter(suites.values())
        biggest = max(counts, key=len)
        print(f"  !! {len(counts)} different algorithm suites across {len(files)} task CSVs:")
        for s, n in counts.most_common():
            miss = sorted(biggest - s)
            print(f"       {n:5d} files with {len(s):2d} algos" + (f"  missing: {miss}" if miss else "  (full)"))
        if not a.allow_stale:
            sys.exit("ABORTING: the suite changed mid-campaign, so these tasks are not comparable.\n"
                     "  Re-run the grid, or pass --allow-stale to concatenate anyway.")
        print("  (--allow-stale: concatenating a mixed-suite campaign)")

    with open(a.out, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=header)
        writer.writeheader()
        for fp in files:
            for row in rows_by_file[fp]:
                writer.writerow(row)
                n_rows += 1
    print(f"{len(files)} files, {n_rows} rows -> {a.out}")


if __name__ == "__main__":
    main()
