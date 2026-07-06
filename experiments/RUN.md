# Running the POC experiments on the Bouchet cluster — order of operations

Full experimental design: `../EXPERIMENTS.md`. Everything runs from the **repo root**.

## 0. One-time setup (on the cluster)
```bash
# clone (or pull) the repo
git clone https://github.com/AsafEtgar/metric-repair.git
cd metric-repair/"Average Metric Repair Sage"      # (the code lives in this subdir)

# a conda env with the deps -- the pure-Python port needs NO Sage on the cluster
module load miniconda
conda create -y -n metric-repair python=3.11 numpy scipy networkx
```

## 1. Smoke-test ONE task first (always do this)
```bash
module load miniconda && conda activate metric-repair
python experiments/run_task.py --count                 # -> 3200
python experiments/run_task.py --task-index 0 --outdir results   # runs one graph (n=100, p=0.3)
head -3 results/task_000000.csv                        # sanity-check the columns/values
```
If that produces a CSV with 18 rows and `valid=1` on the cover rows, the pipeline works.

## 2. Build the joblist + dSQ batch script (does NOT submit)
```bash
bash experiments/submit_dsq.sh metric-repair <your_netid>
# -> writes joblist.txt (3200 lines) and dsq_submit.sh
```
`submit_dsq.sh` sets: `day` partition, `-A pi_<netid>`, 1 core + 1 GB per task, `--time 02:30:00`,
**`--max-jobs 64`** (64 concurrent cores). Edit `MAXJOBS` at the top of the script to change it.

## 3. Submit and monitor
```bash
sbatch dsq_submit.sh
squeue --me                                            # queue / running
dSQAutopsy dsq_submit.sh joblist.txt                   # which tasks succeeded / failed / are pending
cat logs/dsq-*_*.out | tail                            # task stdout (one file per task)
```
Estimated wall-clock ≈ **3–6 h** on 64 cores. Each task writes `results/task_<index>.csv`; failed/rerun
tasks just overwrite their own file.

## 4. Collect into one table
```bash
python experiments/collect.py --indir results --out results_all.csv
# -> results_all.csv : one row per (graph, algorithm), all metadata denormalised in.
```

## 5. If a task failed / to reproduce exactly
```bash
# find failed task indices from the autopsy, then rerun just those:
python experiments/run_task.py --task-index <K> --outdir results
```
Seeds are deterministic (`seed` column = `crc32(config)`), so a rerun of task K is bit-for-bit identical.

---
### Notes
- **Local testing on the Mac** uses `sage -python experiments/run_task.py ...` (system Python here lacks
  numpy); on the cluster it's plain `python` inside the conda env.
- To change the **concurrency**, edit `MAXJOBS` in `experiments/submit_dsq.sh` (it becomes `--max-jobs`).
- To change **what runs** (grids, samples, algorithms, caps), edit the constants/`build_suite` at the top
  of `experiments/harness.py`, then regenerate the joblist (step 2).
