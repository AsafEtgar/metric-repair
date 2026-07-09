# RERUN runbook — after the current campaign drains

Everything below is prepared. The only judgement calls left are the `<netid>` and the memory/time requests.

> ## ⚠️ Do not `git pull` on the cluster until `squeue --me` is empty.
> dSQ launches a fresh `python` per array task, so a task reads the working copy **at the moment it starts**.
> The commit you are about to pull changes `_apsp_positions`, which changes algorithm output on graphs with
> sub-`1e-8` edges. Pulling mid-array would run half a campaign on old code and half on new. Wait.

---

## 0. Confirm the campaign is finished

```bash
squeue --me -h -r | wc -l          # must be 0
```

## 1. Pull the fixes

```bash
cd /path/to/metric-repair && git pull
```

What you are pulling, and why each matters:

| fix | effect on re-runs |
|---|---|
| **A1** `_apsp_positions` no longer deletes edges of weight `<= 1e-8` | the ILP and the covering-LP rounding family now produce **valid** covers on `bct_coactivation_lin/_log`, `flycns_male_log`, `fish1_ten_log` |
| **B21** oracle break-detection tolerance now matches the verifier (`1e-9`) | closes a 1000× blind spot; provably a no-op on RGG/geometric |
| **A5** `except (EOFError, OSError)` | an OOM-killed child now records `status=killed` instead of destroying the whole task's CSV |
| **_aggregate** nulls `size/valid/converged/exact_opt` when `status != "ok"` | a partially timed-out graph can no longer report a partial optimum as `converged=True` |
| **A2/A3** analyzers gate the exact reference on `status=="ok" and valid==1`, and drop invalid covers | the published ratios stop being computed against unverifiable ILP covers |

**Verified before commit:** on RGG and geometric graphs the fixed `_apsp_positions` returns distances
**bit-identical** to the old one (max abs diff `0.000e+00`) and there are **zero** breaks in the old blind
band — so A1/B21 change nothing outside the four sub-`1e-8` real graphs. On those four, the ILP optimum grows
by exactly one edge (87→88, 609→610, 227→228) and every cover now verifies.

## 2. Work out exactly what to re-run

`make_rerun.py` selects tasks that are **MISSING** (died or never ran), **STALE** (a pre-fix CSV left behind
in a reused directory), or **AFFECTED** (touched by the A1 fix). Run it from the repo root.

```bash
module load miniconda && conda activate metricrepair
SETUP='module load miniconda && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate metricrepair'
DAY=2026-07-08          # <-- the date the current campaign STARTED; anything older is pre-fix

# --- synthetic: missing + stale ------------------------------------------------------------------
# results_rgg is the one REUSED directory (it held the old 3160-task grid). Always pass --stale-before.
python experiments/make_rerun.py --harness rgg --grid full      --outdir results_rgg          --stale-before $DAY --setup "$SETUP" --joblist rerun_rgg_full.txt
python experiments/make_rerun.py --harness rgg --grid large     --outdir results_rgg_large    --setup "$SETUP" --joblist rerun_rgg_large.txt
python experiments/make_rerun.py --harness rgg --grid mixed     --outdir results_rgg_mixed    --setup "$SETUP" --joblist rerun_rgg_mixed.txt
python experiments/make_rerun.py --harness rgg --grid largemix  --outdir results_rgg_largemix --setup "$SETUP" --joblist rerun_rgg_largemix.txt
python experiments/make_rerun.py --harness geometric --grid large --outdir results_large      --setup "$SETUP" --joblist rerun_geo_large.txt

# --- realrec: missing + FORCE the two fish1 bases (they carry sub-1e-8 edges) ---------------------
python experiments/make_rerun.py --harness rgg --grid realrec --outdir results_rgg_realrec \
    --bases fish1_ten_lin,fish1_ten_log --setup "$SETUP" --joblist rerun_realrec.txt      # ~360 + missing

# --- real: the four A1-affected graphs, both arrays. ripe_atlas__gmr_ilp stays excluded. ----------
python experiments/make_rerun.py --real-array heur --graphs A1 --exclude-graphs ripe_atlas --setup "$SETUP" --joblist rerun_real_heur.txt   # 4 x 31 = 124
python experiments/make_rerun.py --real-array ilp  --graphs A1 --exclude-graphs ripe_atlas --setup "$SETUP" --joblist rerun_real_ilp.txt    # 4 x 2  = 8

wc -l rerun_*.txt
```

**Delete the stale files first.** `make_rerun.py` writes `<joblist>.stale` listing them. If a re-run task
fails, an undeleted stale CSV survives a second time and looks fresh:

```bash
[ -f rerun_rgg_full.txt.stale ] && while read i; do rm -f "results_rgg/task_$i.csv"; done < rerun_rgg_full.txt.stale
```

## 3. Submit

Memory/time mirror the original arrays. Re-runs are small, so `--max-jobs 64` is plenty.

```bash
mkdir -p logs
sub() {  # sub <joblist> <mem> <time>
  dsq --job-file "$1" --batch-file "dsq_${1%.txt}.sh" --partition day --account "pi_<netid>" \
      --cpus-per-task 1 --mem-per-cpu "$2" --time "$3" --max-jobs 64 \
      --output "logs/dsq-${1%.txt}-%A_%3a.out"
  sbatch "dsq_${1%.txt}.sh"
}
sub rerun_rgg_full.txt       8g  02:30:00
sub rerun_rgg_large.txt      16g 04:00:00
sub rerun_rgg_mixed.txt      8g  02:30:00
sub rerun_rgg_largemix.txt   16g 04:00:00
sub rerun_geo_large.txt      32g 08:00:00   # exp2b dense tail: bumped from 16g/4h, it OOM'd/timed out there
sub rerun_realrec.txt        32g 06:00:00   # dimacs_ny_big n=10000 APSP dict; bumped from 24g/4h
sub rerun_real_heur.txt      16g 04:00:00   # bct/flycns are big-H; 8g was the stress point
sub rerun_real_ilp.txt       16g 17:30:00
```

Skip any joblist that came back empty.

## 4. Re-analyze everything

Not just the new results — **the analyzers changed**, so every summary and figure built before the reference
gate is suspect.

```bash
bash experiments/reanalyze_all.sh python
```

`collect.py` now **aborts** on orphan indices, stale mtimes, or ragged schemas. If it aborts, look at what it
names; do not reach for `--allow-stale` reflexively.

## 5. What "clean" looks like

* `missing_report.py` → 0 missing, 0 orphans in every directory.
* `real_check.py` → **HARD failures = 0**, **invalid covers = 0** (was 198 — A1 fixed the cause), metric
  control `dimacs_ny_d` shows 0 rows with `size > 0`, and 1 missing file (`ripe_atlas__gmr_ilp`, deliberate).
* `analyze.py` / `rgg_analyze.py` → any `DROPPING N invalid-cover rows` line is now **`l1_separation` exiting
  at `max_rounds` without converging** (AUDIT_REPORT.md A4), not the oracle. Expect it to grow with `n`.
* `real_analyze.py` → `ref_kind` should read `exact` for **11** graphs, not 8: the three A1 graphs rejoin the
  exact-reference set once their ILP covers verify.

## 6. Then the paper

With `results_real` re-run, the §7 caveat *"three real graphs excluded … a real limitation of the float-weight
oracle"* is **obsolete** — it was our bug, it is fixed, and the graphs come back. Rewrite it as a fixed defect,
and drop the unsupported "up to 37%" figure. See AUDIT_REPORT.md §5.
