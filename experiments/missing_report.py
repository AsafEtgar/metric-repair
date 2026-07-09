#!/usr/bin/env python
"""Which submitted tasks produced no CSV, and what do the failures have in common?

Run from the REPO ROOT on the cluster, inside the conda env:

    python experiments/missing_report.py

A task index maps 1:1 to a dSQ array line and to `<outdir>/task_%06d.csv` (real data uses
`<graph>__<mode>[_s<seed>].csv`). Grouping the missing indices BY CONFIG turns "214 missing"
into either a block (a cost cliff at the heavy end -> bump mem/time) or scatter (a real bug).
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def report(title, tasks, missing, keyfn):
    print("\n" + "=" * 78)
    print("%s: %d / %d missing" % (title, len(missing), len(tasks)))
    print("=" * 78)
    if not missing:
        print("  (complete)")
        return
    miss = collections.Counter(keyfn(tasks[i]) for i in missing)
    tot = collections.Counter(keyfn(t) for t in tasks)
    for k in sorted(miss, key=lambda x: (-miss[x], str(x))):
        flag = "   <-- ALL" if miss[k] == tot[k] else ""
        print("  %-56s %4d/%-4d%s" % (str(k), miss[k], tot[k], flag))


def orphans(outdir, expected_names):
    """CSVs present that no CURRENT task would write -- stale output from a renamed graph or an older
    grid definition. collect.py globs *.csv, so these silently enter the combined analysis CSV."""
    have = set(f for f in os.listdir(outdir) if f.endswith(".csv"))
    extra = sorted(have - set(expected_names))
    print("\n  files present=%d  expected=%d  ORPHANS=%d" % (len(have), len(expected_names), len(extra)))
    for f in extra:
        print("    ORPHAN  %s" % f)
    return extra


def synth_missing(mod, grid, outdir):
    tasks = mod.all_tasks(grid)
    miss = [i for i in range(len(tasks))
            if not os.path.exists(os.path.join(outdir, "task_%06d.csv" % i))]
    orphans(outdir, ["task_%06d.csv" % i for i in range(len(tasks))])
    return tasks, miss


if os.path.isdir("results_large"):
    import harness
    tasks, miss = synth_missing(harness, "large", "results_large")
    report("results_large (geometric)", tasks, miss,
           lambda t: "%-6s n=%-5d p=%.4f alpha=%s" % (
               t[0]["exp"], t[0]["n"], t[0]["p"],
               ("%.3f" % t[0]["alpha"]) if t[0].get("alpha") is not None else "  -  "))
    report("results_large  [by exp]", tasks, miss, lambda t: t[0]["exp"])

if os.path.isdir("results_rgg_realrec"):
    import rgg_harness
    tasks, miss = synth_missing(rgg_harness, "realrec", "results_rgg_realrec")
    report("results_rgg_realrec", tasks, miss,
           lambda t: "%-20s n=%-6d %-8s frac=%.2f" % (
               t[0]["base"], t[0]["n"], t[0]["direction"], t[0]["frac_q"]))
    report("results_rgg_realrec  [by base]", tasks, miss, lambda t: t[0]["base"])
    report("results_rgg_realrec  [by frac]", tasks, miss, lambda t: "frac=%.2f" % t[0]["frac_q"])

if os.path.isdir("results_real"):
    import real_harness

    def _fname(t):
        g, mode, s = t
        return g + "__" + mode + (("_s%02d" % s) if mode == "rand" else "") + ".csv"

    # BOTH arrays write into results_real/ (submit_real_dsq.sh hardcodes OUTDIR=results_real):
    #   heur = 19 graphs x (1 det + 30 rand) = 589      ilp = 16 dist-sensible x 2 variants = 32
    # -> a complete results_real/ holds 621 CSVs. Checking only `heur` would flag the 32 ILP files as orphans.
    heur = real_harness.all_tasks("heur")
    ilp = real_harness.all_tasks("ilp")

    for label, tasks in (("heur", heur), ("ilp", ilp)):
        miss = [i for i, t in enumerate(tasks)
                if not os.path.exists(os.path.join("results_real", _fname(t)))]
        report("results_real [%s, by graph]" % label, tasks, miss, lambda t: t[0])
        if miss:
            print("\n  missing files:")
            for i in miss:
                print("    %s" % _fname(tasks[i]))

    extra = orphans("results_real", [_fname(t) for t in heur] + [_fname(t) for t in ilp])

    # An orphan filename is "<graph>__<mode>[_sNN].csv" -- collapse on <graph> so a renamed/removed
    # graph shows up as a single block of 31 rather than 31 unrelated-looking files.
    if extra:
        by_graph = collections.Counter(f.split("__")[0] for f in extra)
        print("\n" + "=" * 78)
        print("results_real ORPHANS by graph (these are NOT in real_graphs() any more)")
        print("=" * 78)
        for g, c in by_graph.most_common():
            print("  %-40s %4d file(s)%s" % (g, c, "   <-- a full graph (1 det + 30 rand)" if c == 31 else ""))
        print("\n  quarantine them before collecting:")
        print("    mkdir -p results_real_orphans && mv \\")
        for g in by_graph:
            print("      results_real/%s__*.csv \\" % g)
        print("      results_real_orphans/")
