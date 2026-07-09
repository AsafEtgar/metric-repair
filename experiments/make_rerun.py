"""make_rerun.py -- compute EXACTLY which tasks must be re-run, and emit a dSQ joblist for them.

Run from the REPO ROOT, in the conda env. See RERUN.md for the full runbook.

A task earns a re-run for one of three reasons:

  MISSING  its CSV is absent. Either the task died (OOM / SLURM walltime / the pre-fix OSError bug that
           destroyed a whole task, AUDIT_REPORT.md A5) or it never ran.

  STALE    its CSV predates --stale-before. Output directories are REUSED: a re-run overwrites
           `task_%06d.csv` only for the tasks that SUCCEED, so a failed task leaves the PREVIOUS run's
           file sitting there, written by older code. `results_rgg/` is the live example -- it held 3160
           CSVs from the old 3160-task `full` grid before that grid grew to 5000.

  AFFECTED its graph is touched by a correctness fix. Today that is AUDIT_REPORT.md A1: the separation
           oracle used to delete every edge of weight <= 1e-8, so any graph carrying such an edge got an
           invalid cover from the ILP and the whole covering-LP rounding family. Select those by graph
           name (--graphs, real arrays) or by RGG base (--bases, the realrec grid).

Examples
--------
  # RGG 'full': missing + anything older than the re-run's start date
  python experiments/make_rerun.py --harness rgg --grid full --outdir results_rgg \\
      --stale-before 2026-07-08 --joblist rerun_rgg_full.txt

  # realrec: force the two fish1 bases (sub-1e-8 edges), plus any missing task
  python experiments/make_rerun.py --harness rgg --grid realrec --outdir results_rgg_realrec \\
      --bases fish1_ten_lin,fish1_ten_log --joblist rerun_realrec.txt

  # real data: the four A1-affected graphs, both arrays
  python experiments/make_rerun.py --real-array heur --graphs A1 --joblist rerun_real_heur.txt
  python experiments/make_rerun.py --real-array ilp  --graphs A1 --joblist rerun_real_ilp.txt
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The graphs whose covers were corrupted by A1 (they carry an edge of weight <= 1e-8, the EPS=1e-9 floor
# that build_inversions.py adds to the _lin/_log similarity conversions). fish1_ten_log's covers happened to
# stay valid (|H|=4, no cut ever needed that edge) but its oracle output still changed, so re-run it too.
A1_GRAPHS = ["bct_coactivation_lin", "bct_coactivation_log", "flycns_male_log", "fish1_ten_log"]


def _mtime_before(path, cutoff):
    return cutoff is not None and os.path.getmtime(path) < cutoff


def _w_min(path, _cache={}):
    """Smallest edge weight of the CORRUPTED graph this task ran on. A meta field, identical on every row,
    so read only the first. Cached: --w-min-le would otherwise re-read 5000 files."""
    if path not in _cache:
        import csv
        try:
            with open(path, newline="") as f:
                row = next(csv.DictReader(f), None)
            _cache[path] = float(row["w_min"]) if row and row.get("w_min") else None
        except (OSError, ValueError, KeyError):
            _cache[path] = None
    return _cache[path]


def _parse_date(s):
    if not s:
        return None
    return time.mktime(time.strptime(s, "%Y-%m-%d"))


def synthetic(a):
    if a.harness == "rgg":
        from rgg_harness import all_tasks
        runner = "experiments/run_rgg_task.py"
    else:
        from harness import all_tasks
        runner = "experiments/run_task.py"
    tasks = all_tasks(a.grid)
    cutoff = _parse_date(a.stale_before)
    bases = set(a.bases.split(",")) if a.bases else set()

    # The reasons are independent -- a task can be both stale AND affected -- so classify with plain `if`s
    # and take the union. (An `elif` chain would under-count and, worse, hide an affected task behind a
    # stale one.)
    #
    # --w-min-le is the EXACT test for A1 exposure. Every task CSV records `w_min`, the smallest edge weight
    # of the corrupted graph the algorithms actually ran on. A1 deleted every edge of weight <= 1e-8 from the
    # separation oracle's view, so `w_min <= 1e-8` <=> this task's oracle was blind. That beats guessing by
    # base name: deflate sets w = gap/magnitude and only rejects gap <= 1e-9, so it can MINT a sub-1e-8 edge
    # on any float base, not just on the ones that ship the EPS=1e-9 floor from the _lin/_log inversions.
    missing, stale, affected = [], [], []
    for i, (cfg, _s) in enumerate(tasks):
        fp = os.path.join(a.outdir, "task_%06d.csv" % i)
        if not os.path.exists(fp):
            missing.append(i)
            continue
        if _mtime_before(fp, cutoff):
            stale.append(i)
        if bases and cfg.get("base") in bases:
            affected.append(i)
        elif a.w_min_le is not None and _w_min(fp) is not None and _w_min(fp) <= a.w_min_le:
            affected.append(i)

    idxs = sorted(set(missing) | set(stale) | set(affected))
    print("grid=%s  outdir=%s  total tasks=%d" % (a.grid, a.outdir, len(tasks)))
    why = a.bases or ("w_min <= %g" % a.w_min_le if a.w_min_le is not None else "none")
    print("  MISSING  %5d" % len(missing))
    print("  STALE    %5d  (mtime < %s)" % (len(stale), a.stale_before or "n/a"))
    print("  AFFECTED %5d  (%s)" % (len(affected), why))
    print("  -> RE-RUN %4d tasks" % len(idxs))
    if stale:
        print("\n  NOTE: stale files are PRE-FIX survivors. Delete them so a failed re-run cannot leave them"
              "\n        behind a second time:  for i in $(cat %s.stale); do rm -f %s/task_$i.csv; done"
              % (a.joblist, a.outdir))
        with open(a.joblist + ".stale", "w") as f:
            for i in stale:
                f.write("%06d\n" % i)

    prefix = (a.setup + " && ") if a.setup else ""
    with open(a.joblist, "w") as f:
        for i in idxs:
            f.write("%s%s %s --task-index %d --outdir %s --grid %s\n"
                    % (prefix, a.python, runner, i, a.outdir, a.grid))
    return len(idxs)


def real(a):
    from real_harness import all_tasks
    graphs = A1_GRAPHS if a.graphs == "A1" else (a.graphs.split(",") if a.graphs else [])
    excl = set(a.exclude_graphs.split(",")) if a.exclude_graphs else set()
    tasks = all_tasks(a.real_array)

    def fname(t):
        g, mode, s = t
        return g + "__" + mode + (("_s%02d" % s) if mode == "rand" else "") + ".csv"

    missing, affected = [], []
    for i, t in enumerate(tasks):
        if t[0] in excl:
            continue
        fp = os.path.join(a.outdir, fname(t))
        if not os.path.exists(fp):
            missing.append(i)
        elif t[0] in graphs:
            affected.append(i)

    idxs = sorted(set(missing) | set(affected))
    print("array=%s  outdir=%s  total tasks=%d" % (a.real_array, a.outdir, len(tasks)))
    print("  MISSING  %5d  %s" % (len(missing), [fname(tasks[i]) for i in missing[:4]]))
    print("  AFFECTED %5d  (graphs: %s)" % (len(affected), ", ".join(graphs) or "none"))
    if excl:
        print("  EXCLUDED       (graphs: %s)" % ", ".join(sorted(excl)))
    print("  -> RE-RUN %4d tasks" % len(idxs))
    if a.real_array == "ilp" and any(tasks[i][0] == "ripe_atlas" for i in missing):
        print("\n  NOTE: ripe_atlas__gmr_ilp is missing because it was OOM-killed at 8GB, and it will not"
              "\n        converge at 95.3% break density. Re-running it buys a 17h timeout row, not an"
              "\n        optimum. Drop it with:  --exclude-graphs ripe_atlas")

    prefix = (a.setup + " && ") if a.setup else ""
    with open(a.joblist, "w") as f:
        for i in idxs:
            f.write("%s%s experiments/run_real_task.py --array %s --task-index %d --outdir %s --covers %s\n"
                    % (prefix, a.python, a.real_array, i, a.outdir, a.covers))
    return len(idxs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", choices=["geometric", "rgg"], default=None)
    ap.add_argument("--grid", default=None)
    ap.add_argument("--real-array", choices=["heur", "ilp"], default=None)
    ap.add_argument("--graphs", default=None, help="comma list, or the literal 'A1' for the A1-affected set")
    ap.add_argument("--bases", default=None, help="comma list of RGG realrec bases to force re-run")
    ap.add_argument("--w-min-le", type=float, default=None,
                    help="force re-run of tasks whose recorded w_min <= this. Use 1e-8 to select exactly "
                         "the tasks whose separation oracle was blinded by A1.")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--covers", default="results_real_covers")
    ap.add_argument("--stale-before", default=None, help="YYYY-MM-DD; CSVs older than this are pre-fix")
    ap.add_argument("--exclude-graphs", default=None, help="comma list to skip entirely (e.g. ripe_atlas)")
    ap.add_argument("--joblist", required=True)
    ap.add_argument("--python", default="python")
    ap.add_argument("--setup", default="", help="shell prefix prepended to every line (conda activate ...)")
    a = ap.parse_args()

    if a.real_array:
        a.outdir = a.outdir or "results_real"
    # Without this guard, a mistyped or not-yet-created outdir makes EVERY task look MISSING and the tool
    # cheerfully proposes re-running the entire campaign.
    if not os.path.isdir(a.outdir):
        sys.exit("outdir %r does not exist. Run this from the repo ROOT on the cluster, where the results "
                 "directories live." % a.outdir)

    if a.real_array:
        n = real(a)
    else:
        if not (a.harness and a.grid and a.outdir):
            ap.error("--harness, --grid and --outdir are required for synthetic grids")
        n = synthetic(a)

    if n:
        print("\nwrote %s (%d lines). Build the array with:" % (a.joblist, n))
        print("  dsq --job-file %s --batch-file dsq_$(basename %s .txt).sh --partition day \\\n"
              "      --account pi_<netid> --cpus-per-task 1 --mem-per-cpu <MEM> --time <HH:MM:SS> \\\n"
              "      --max-jobs 64 --output logs/dsq-rerun-%%A_%%3a.out" % (a.joblist, a.joblist))
    else:
        print("\nnothing to re-run.")


if __name__ == "__main__":
    main()
