#!/bin/bash
# bundle_analysis.sh -- collect exactly what the paper needs into one tarball you can download by hand.
#
# Run FROM THE REPO ROOT on the cluster, AFTER reanalyze_all.sh has completed:
#
#   bash experiments/bundle_analysis.sh          # -> analysis_bundle.tar.gz
#
# What goes in:
#   summary*.csv          the aggregated medians + the n_ok / n_usable denominators
#   rows_with_ratio.csv   geometric per-task rows (needed to check the GMR covering polytope is integral)
#   real_rows_with_ratio.csv  real per-task rows (~3k) -- the algorithm ranking is computed from these
#   *_fit.csv             the runtime/iteration scaling fits -- the three cost laws
#   *.out                 the health-check outputs (missing_report, real_check, rgg_check)
#   reanalyze2.log        the run itself
#   *.pdf / *.png         every figure
#
# What stays behind: rgg_rows_with_ratio.csv (175,218 rows). Its medians are already in summary_edit.csv.
set -euo pipefail

OUT="${1:-analysis_bundle}"
LOG="analysis/reanalyze2.log"

# Refuse to bundle a half-written analysis. reanalyze_all.sh prints "=== done ===" as its final banner; if
# that string is absent the run aborted (probably on its own preflight) or is still going, and every CSV in
# analysis/ is a mix of this run and the last one. We have been bitten by exactly this three times.
if [ ! -f "$LOG" ] || ! grep -q "=== done ===" "$LOG"; then
    echo "ABORT: $LOG is missing or does not contain '=== done ==='."
    echo "       The analysis did not finish, so anything under analysis/ is a mix of runs."
    echo "       Run:  STALE_DAYS=365 bash experiments/reanalyze_all.sh python 2>&1 | tee $LOG"
    exit 1
fi

rm -rf "$OUT" "$OUT.tar.gz"
mkdir -p "$OUT"

find analysis -type f \( \
        -name 'summary*.csv' \
     -o -name 'rows_with_ratio.csv' \
     -o -name 'real_rows_with_ratio.csv' \
     -o -name '*_fit.csv' \
     -o -name '*.out' \
     -o -name 'reanalyze2.log' \
     -o -name '*.pdf' \
     -o -name '*.png' \) ! -name 'rgg_rows_with_ratio.csv' -print0 \
| while IFS= read -r -d '' f; do
    mkdir -p "$OUT/$(dirname "$f")"
    cp "$f" "$OUT/$(dirname "$f")/"
done

tar czf "$OUT.tar.gz" "$OUT"
rm -rf "$OUT"

echo
echo "wrote $(pwd)/$OUT.tar.gz   ($(du -h "$OUT.tar.gz" | cut -f1))"
echo
echo "summaries included:"
tar tzf "$OUT.tar.gz" | grep -E 'summary.*\.csv$' | sed 's/^/  /' | sort
echo
echo "figures: $(tar tzf "$OUT.tar.gz" | grep -cE '\.(pdf|png)$')   csvs: $(tar tzf "$OUT.tar.gz" | grep -cE '\.csv$')"
echo
echo "Download $OUT.tar.gz and drop it in the repo root on your Mac."
