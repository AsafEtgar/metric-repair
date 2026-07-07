# Real-dataset metric-repair experiments — spec

Goal: take the algorithm suite validated on synthetic G(n,p) and RGG data and run it on **10 real
weighted graphs** spanning transport, networking, neuroscience, structural biology, and genomics — to
answer (R1) does it work on *structured* (non-i.i.d.) non-metricity, (R2) does repair *recover a true
metric* where we have ground truth, (R3) which repair *direction* is physically right per domain, (R4) does
repair help a *downstream task*, and (R5) where does exact break and which heuristic tracks the bound.

This is the design spec (mirrors `RGG_EXPERIMENTS.md`). No code is written until it's signed off.

---

## 1. Datasets & sources

All graphs are already built to `data/processed/<name>.csv` (columns `u,v,weight`, load with
`datasets.load_edgelist`). Raw sources are in `data/raw/<dataset>/` (immutable); the builder is
`build_real_graphs.py`; full provenance is in `data/DATASETS.md`. Fetched **2026-07-07**. Every weight is a
**distance-like** quantity (non-negative, undirected, incomplete).

| graph (`data/processed/…`) | domain | source (where from) | raw files | build recipe | n / m | non-metric frac | weight = |
|---|---|---|---|---|---|---|---|
| **`dimacs_ny_d`** | road network | 9th DIMACS Impl. Challenge — `www.diag.uniroma1.it/challenge9/data/` (the `users.diag…` host 403s non-browsers) | `dimacs/USA-road-d.NY.gr.gz` (264,346 nodes / 733,846 arcs) | BFS ball n=1200 from a seed | 1200 / 1455 | **0.0%** (metric) | road distance (meters) |
| **`dimacs_ny_t`** | road network | same challenge, travel-time file | `dimacs/USA-road-t.NY.gr.gz` | BFS ball n=1200 | 1200 / 1455 | 0.14% | travel time |
| **`ripe_atlas`** | internet latency | RIPE Atlas REST API `atlas.ripe.net/api/v2` (open, no key) | `ripe_atlas/{rtt_matrix.csv, rtt_edgelist.csv, anchors_meta.csv}` | 1000 IPv4 anchors, per-pair **min RTT** from the latest anchoring-ping run, symmetrized (avg of both dirs) | 999 / 442,707 | **95.3%** | min RTT (ms) |
| **`bct_coactivation`** | brain (fMRI) | Brain Connectivity Toolbox `sites.google.com/site/bctnet` (2019-03-03 dist) | `bct/Coactivation_matrix.mat` (+ `Coord` 638×3) | invert similarity → distance | 638 / 18,625 | **43.5%** | d = 1 / similarity |
| **`nmr_1d3z_atom`** | NMR distance geometry | RCSB PDB entry **1D3Z** (Cornilescu & Bax 1998, ubiquitin) | `nmr/1D3Z.mr` (2,727 NOE restraints), `nmr/1D3Z.pdb` (10-model ensemble) | atom-group graph from NOE **upper bounds** | 430 / 1,357 (3 comps, giant 426) | 1.1% | NOE upper bound (Å) |
| **`nmr_1d3z_residue`** | NMR distance geometry | same PDB entry 1D3Z | same | residue graph (min over atom pairs) | 75 / 308 | 5.2% | NOE upper bound (Å) |
| **`pbmc3k_cosine_knn`** | scRNA-seq | 10x Genomics `cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k` (2,700 PBMCs) | `scrna_pbmc3k/…/{matrix.mtx,genes.tsv,barcodes.tsv}` | log-CPM → PCA-50 → **cosine kNN** (k=15) | 2700 / 31,639 | 0.21% | cosine distance |
| **`cassiopeia_barcode_knn`** | single-cell lineage | GitHub `YosefLab/Cassiopeia` `notebooks/data/` | `cassiopeia_spatial/{spatial_allele_table.txt, spatial_adata.h5ad}` | **barcode-dissimilarity kNN** (k=15) on 1000 cells | 1000 / 12,760 | 30.7% | allele dissimilarity |
| **`fish1_ten`** | zebrafish connectome | GCS `fish1-release/paper_data/` (no token) | `fish1/TEN_analysis.zip` (adjacency lists + `TEN_cats.csv`) | TEN-core BFS ball n=1000 | 1000 / 1,175 | 1.8% | d = 1 / synapse count |
| **`flycns_male`** | fly connectome | GCS `flyem-male-cns/v1.0/…/flat-connectome` (no token) | `flycns/{connectome-weights…feather, body-annotations…feather}` | Traced-neuron BFS ball n=1200, symmetrize directed | 1200 / 14,025 | **83.1%** | d = 1 / (Σ synapse counts, both dirs) |

**Non-metricity spectrum** (this is the experimental axis): metric control `dimacs_ny_d` (0%) → near-metric
`dimacs_ny_t`, `pbmc3k`, `nmr_atom`, `fish1_ten` (0.1–1.8%) → mild `nmr_residue` (5%) → moderate
`cassiopeia` (31%), `bct` (44%) → heavy `flycns` (83%), `ripe` (95%).

### 1.1 Similarity-inversion variants (`bct`, `flycns`, `fish1`)

Three of the graphs derive a distance from a **similarity/strength** `s` (higher = closer): `bct` (fMRI
coactivation) and the two connectomes (`flycns`, `fish1`; `s` = synapse count). Non-metricity is **not
invariant** to how `s` is turned into a distance, so each of these three becomes a 4-way conversion study —
turning "how much non-metricity is intrinsic vs an artifact of the transform" into an experiment. **12
processed graphs** (3 bases × 4 conversions); with the 7 non-inverted graphs above → **19 total**.

| suffix | conversion `d(s)` | strong `s` → | distance-sensible? |
|---|---|---|---|
| *(none)* — current | `1 / s` | small `d` | yes |
| `_lin` | `max(s) − s` | small `d` | yes |
| `_log` | `log(max(s) / s)` (≥ 0) | small `d` | yes |
| `_raw` | `s` (identity) | **large `d`** | **no — inverted** (contrast only) |

Names: `bct_coactivation{,_lin,_log,_raw}`, `flycns_male{,_lin,_log,_raw}`, `fish1_ten{,_lin,_log,_raw}`.
The directed→undirected symmetrization (connectomes) is unchanged; only the inversion varies. `_raw` treats
the similarity *as* a distance (similar → far), so it is semantically inverted: fine for **characterization**
(R1/R5) but excluded from **ground-truth recovery** (R2/R3), where it would be anti-correlated with the
coordinates by construction. (Aside: `GroupAverage_rsfMRI_matrix.mat` is a second, un-built BCT graph — a
possible separate dataset later, orthogonal to this inversion axis.)

**Built result — non-metricity is dominated by the inversion, not the data** (`build_inversions.py`, which
recovers `s = 1/d` from the `d=1/s` graph so all four conversions share *identical topology*):

| base | `1/s` (current) | `_lin` (max−s) | `_log` | `_raw` (s) |
|---|---|---|---|---|
| `bct_coactivation` | **43.5%** | **1.1%** | 17.6% | 18.1% |
| `flycns_male` | **83.1%** | **0.6%** | 12.6% | 49.1% |
| `fish1_ten` | 1.8% | **0.0%** (metric) | 0.3% | 1.9% |

The headline non-metricity of `bct` (44%) and `flycns` (83%) **collapses to ~1% under the linear
conversion**, and `fish1_lin` is *exactly metric*. So most of those triangle violations are an **artifact of
the `1/s` transform** (which explodes weak similarities into large distances), not intrinsic structure. This
makes **R3b a substantive finding**: for similarity-derived graphs, "how non-metric is this data?" is
largely a modeling choice (`s → d`), so a headline claim of "brain/connectome graphs are highly non-metric"
would really be a claim about `1/s`. The four conversions let us separate the two. (`_log` and `_raw` sit in
between; `_raw`'s value is high precisely because large similarities become large distances.)

---

## 2. Algorithm suite (18 algorithms)

Same suite as synthetic, **minus the 3 `rsp` methods** (`gmr_lp_rsp`, `iomr_lp_rsp`, `iomr_thr_rsp`): real
weights are float (rsp falls back to naive) or integer with huge `w_max` (e.g. DIMACS meters → the
O(w_max·n²) weight-budget DP is intractable). We also **added the GMR analogues** of the covering-LP
rounding family (`gmr_thr_naive`, `gmr_bestofk`, `gmr_rand`) — on float data `gmr_lp_rsp`'s exact integral
cover is gone, so GMR otherwise lacked a fast valid-cover rounding heuristic. (No `gmr_thr_rsp`: redundant
with the integral rsp LP. No `gmr_regiongrow`: region growing is an IOMR light-edge construction.)

| class | algorithms | runs per graph |
|---|---|---|
| **Randomized** (use an RNG) | `pivot`, `iomr_rand`, `iomr_bestofk`, `gmr_rand`, `gmr_bestofk` | **×30 seeds** → mean/median/IQR |
| **Deterministic** | `domr`, `gmr_lp_naive`, `gmr_thr_naive`, `iomr_lp_naive`, `iomr_thr_naive`, `iomr_regiongrow`, `l1sep_gmr`, `l1sep_iomr`, `spc_gmr`, `spc_iomr`, `left_edge` | ×1 |
| **Exact ILP** | `gmr_ilp`, `iomr_ilp` | ×1, own long jobs (§3) |

`gmr_lp_naive` / `iomr_lp_naive` are LP **lower bounds** (not covers); they + the ILPs supply the reference
optimum. MR variants: **GMR** = general (any edit; DOMR covers count as GMR), **IOMR** = increase-only,
**DOMR** = decrease-only (exact, `domr_alg`).

---

## 3. Job architecture (two independent SLURM arrays)

The 17 h exact-ILP tail is **decoupled** from everything fast so the heuristics land today and the ILPs land
whenever they finish.

**Array H — heuristics** (short walltime, 8 GB/task): per graph, one deterministic-bundle task + the 5
randomized algorithms ×30 seeds → `19 × (1 + 30) = 589 tasks` (all 19 graphs, incl. the `_raw` variants).
Each task runs per connected component and aggregates (as in the synthetic harness).

**Array I — exact ILP** (`--time=17:00:00`, `--mem-per-cpu=8g`, independent): one task per
`(graph, {gmr_ilp, iomr_ilp})` over the **16 distance-sensible graphs** (the 3 `_raw` variants are
contrast-only and skip the ILP) → `16 × 2 = 32 tasks`. The real-data runner **overrides the shared 45 s
`ALGO_TIMEOUT`** with a 17 h cap. Small graphs (`nmr_*`, sparse `dimacs`/`fish1`) finish in seconds;
`ripe`/`flycns` and their conversions will burn the full 17 h and OOM/time-out → LP-bound fallback. The 32
ILP tasks run concurrently (subject to the array throttle), so wall-clock ≈ 17 h regardless of count. Run
**once, ever**.

**Reference merge (post-hoc, per graph):** `ratio = cover_size / ref`, where
`ref = ILP OPT if converged, else the gmr/iomr_lp_naive lower bound` — same fallback as RGG, so a timed-out
ILP never leaves a graph without a reference. Randomized algos → distribution over 30 seeds; deterministic →
single value; all keyed on **graph name** (not a synthetic config).

CSVs: one per task → `collect.py` → `results_real_all.csv`. Memory: 8 GB across both arrays (the dense
`ripe` at 442k edges is the stress case; bump to 16 GB if Array H OOMs on it).

---

## 4. Experiments

| graph(s) | R1 bench | R2 GT-recover | R3 dir/inversion | R4 downstream | R5 scale |
|---|:-:|:-:|:-:|:-:|:-:|
| `dimacs_ny_d` (control) | ✓ | – | – | – | ✓ |
| `dimacs_ny_t` | ✓ | ~ | ~ | – | ✓ |
| `ripe_atlas` | ✓ | **✓** | **✓** (dir) | ✓ | ✓ (stress) |
| `bct_coactivation_*` (×4) | ✓ | ~ (3 sensible) | **✓** (inv) | ? | ✓ |
| `nmr_1d3z_atom/residue` | ✓ | **✓** | **✓** (dir) | – | ✓ |
| `pbmc3k` | ✓ | – | – | – | ✓ |
| `cassiopeia` | ✓ | ~ | ~ | ✓ | ✓ |
| `fish1_ten_*` (×4) | ✓ | ~ (3 sensible) | **✓** (inv) | ~ | ✓ |
| `flycns_male_*` (×4) | ✓ | ~ (3 sensible) | **✓** (inv) | ✓ | ✓ |

**✓ = headline, ~ = supporting/exploratory, ? = pending label check.** `_raw` variants are R1/R5 (+ R3b
contrast) only; the 3 distance-sensible conversions carry R2/R3 GT-recovery.

### R1 — Benchmark on real non-metricity (all 10)
*Does the suite generalize off i.i.d. structure?* Reuse `analyze.py` (keyed on graph): per graph × algo —
cover size, `ratio` (size/OPT-or-bound), `ratio_domr` (size/|H|), runtime, separation `rounds`; randomized
averaged over 30 seeds. Compare exponents/ratios to the synthetic runs. `dimacs_ny_d` (metric) is the
**control**: repair must return an empty cover. Deliverable: per-graph bars of cover size by variant/algo +
a ratio/runtime table across the non-metricity spectrum.

### R2 — Ground-truth recovery (headline: `nmr`, `ripe`)
*Does repair denoise a corrupted metric toward its true one?* GT = a true metric **T** (§5); observed graph =
**C**; repaired graph **F** = `D_{G∖S}` (cover edges → detour distances). The RGG kNN machinery, with a real T:
- **weight fidelity** — `|repaired − T|` vs `|observed − T|` on the edge set (does repair move edges toward truth?),
- **kNN lift** — `jaccard(kNN_F, kNN_T) − jaccard(kNN_C, kNN_T)` (>0 = repair helps),
- **triplet accuracy** vs T's distance ordering,
- **edit-vs-outlier** — precision/recall of the cover against the *true outliers* (edges with the largest `|observed − T|`); the real-data analogue of edit-precision (no planted set, so GT deviation defines "wrong").

### R3 — Modeling choices: repair direction & similarity inversion

**R3a — Domain-appropriate direction (GMR vs IOMR vs DOMR).** *Which direction is physically right, and does
it recover GT best?* Physically-motivated priors, tested empirically:
- **`ripe`**: RTT ≥ true propagation (routing detours inflate direct edges) → weights **overestimate** → **DOMR** prior (decrease violators to the lower/geodesic envelope). Test: does DOMR recover lat/lon distances better than IOMR/GMR?
- **`nmr`**: NOE weights are distance **upper bounds** ≥ true → also **overestimate** → **DOMR** prior (tighten loose bounds to the path). This is the canonical distance-geometry recovery; test vs the PDB 3D distances. DOMR is exact (`domr_alg`), so the comparison is clean.

**R3b — Similarity-inversion sensitivity (`bct`, `flycns`, `fish1` × 4 conversions each; §1.1).** *How much
of the non-metricity is intrinsic vs an artifact of the `s → d` transform?* Across `1/s`, `max−s`, `log`,
and `raw`: compare `|H|` / non-metric fraction, DOMR/GMR/IOMR cover sizes and ratios (R1 machinery), and —
on the three distance-sensible conversions only — GT-recovery vs the coordinates (R2 machinery). `_raw` is a
**contrast/control** (inverted → expected anti-correlation with GT is the sanity check). `bct` is the pilot;
`flycns`/`fish1` extend it to the connectome/synapse-count setting.

### R4 — Downstream utility (`ripe`, `cassiopeia`, `flycns`; `fish1`/`bct` maybe)
*Does repairing the graph improve a real task vs the raw graph?* kNN label-classification accuracy /
clustering ARI / embedding trustworthiness, **repaired vs raw**, against known labels (§6). `pbmc3k` is
**out** of R4 (no labels in raw).

### R5 — Scalability & iteration behavior (all 10)
*Where does exact break, and which heuristic tracks the bound?* Feasibility map over (n, m, |H|); the dense
`ripe` (442k edges) as a memory/runtime stress test; separation `rounds` on hub-heavy graphs (`flycns`,
`ripe`) vs the synthetic n^0.7 scaling. Mostly free from R1's columns + the ILP timeout pattern.

---

## 5. Ground-truth loaders (prerequisite for R2/R3)

Coordinates/true metrics live in `data/raw/`, not the processed edge lists, and node IDs must be **aligned**
to the processed graph. New step (extend `build_real_graphs.py` or a `ground_truth.py`) emitting a
node-aligned true-distance source per graph:

| graph | GT source | true metric | tier | notes |
|---|---|---|---|---|
| `nmr_1d3z_*` | `1D3Z.pdb` model 1 | Euclidean 3D interatomic distance | **strong** | NOE *are* noisy measurements of these distances — gold standard |
| `ripe_atlas` | `anchors_meta.csv` `lat,lon` | haversine geographic distance | **strong** | latency ≈ propagation ∝ distance; violations = routing detours |
| `bct_coactivation{,_lin,_log}` | `Coactivation_matrix.mat` `Coord` (638×3) | Euclidean | moderate | functional ≠ spatial; `_raw` excluded (inverted) |
| `cassiopeia` | lineage in `spatial_allele_table.txt`; spatial in `.h5ad` | tree metric / spatial | moderate | **spatial needs `h5py`** (absent from Sage env) |
| `dimacs_ny_t` | (see prereq) | — | moderate | **coords not fetched** (`.co` missing); or use `-d` as a reference for `-t` |
| `flycns_male{,_lin,_log}`, `fish1_ten{,_lin,_log}` | soma positions (feather, `pyarrow` ✓) | Euclidean | weak | synapse-distance ≠ soma-distance; `_raw` excluded |
| `pbmc3k` | — | none | — | R1/R5 only |

R2/R3 conclusions **lean on the strong tier** (`nmr`, `ripe`); the rest are supporting.

---

## 6. Labels for R4

| graph | labels | source | usable |
|---|---|---|---|
| `ripe_atlas` | country / city | `anchors_meta.csv` | ✅ strong (geographic classification) |
| `cassiopeia` | lineage groups (3) | `spatial_allele_table.txt` | ✅ |
| `flycns_male` | cell types | `body-annotations…feather` (`pyarrow` ✓) | ✅ |
| `fish1_ten` | TEN classI–IV | `TEN_cats.csv` | ⚠️ only ~43 of 1000 nodes labeled |
| `bct_coactivation` | brain regions? | `.mat` vars (TBD) | ❓ check `.mat` for a region/label var |
| `pbmc3k` | — | (needs clustering) | ❌ out of R4 |

---

## 7. Prerequisites & open items (before code)

1. **GT loaders** (§5) — the main new build. Start by validating `nmr` + `ripe` (strong tier) end-to-end,
   then wire the rest. Node-ID alignment is the fiddly part (PDB atom/residue IDs ↔ graph nodes; anchor IDs
   ↔ graph nodes).
2. **Env**: `h5py`/`anndata` for cassiopeia spatial coords (`sage -pip install h5py anndata`); `pyarrow`
   already installed (flycns/fish1 positions & labels). If we want `dimacs` Euclidean GT, fetch
   `USA-road-d.NY.co.gz` — otherwise `dimacs` GT is dropped or `-d` serves as the `-t` reference.
3. **Real-data runner**: overrides the 45 s ILP cap to 17 h for Array I; keeps the heuristics' normal caps.
   Emits the same CSV schema as the synthetic harness (+ a `graph` column, − synthetic config columns).
4. **`bct` label var** — confirm whether the `.mat` carries region labels (decides `bct`'s R4 status).
5. **Inversion-variant builds** — extend `build_real_graphs.py` to emit the 9 new conversion graphs
   (`bct`/`flycns`/`fish1` × `{_lin, _log, _raw}`; §1.1) and regenerate `REAL_GRAPHS_REPORT.csv`. Handle
   edge cases (`s = 0`, normalization for the log/lin forms).

## 8. Outputs & analysis

- `results_real/task_*.csv` → `results_real_all.csv` (`collect.py`).
- **R1/R5**: `analyze.py`-style summary keyed on graph (ratio / ratio_domr / runtime / rounds), randomized
  averaged over 30 seeds; figures split by MR axis (`gmr`/`iomr`) as in `plots.py`, into
  `analysis/figs/real/{gmr,iomr}/`.
- **R2/R3**: a real-data recovery module (analogue of the RGG kNN pass) → weight-fidelity, kNN-lift,
  triplet-accuracy, edit-vs-outlier per (graph, variant, direction); figures per GT dataset.
- **R4**: a downstream module → classification/clustering/embedding metrics, repaired vs raw.

## 9. Cluster runbook (sketch)

```
# Array H (heuristics): 310 tasks, 8 GB, short walltime
bash experiments/submit_real_dsq.sh metricrepair <PI_netid> heur
sbatch dsq_real_heur_submit.sh
# Array I (exact ILP): 20 tasks, 17 h, 8 GB, independent
bash experiments/submit_real_dsq.sh metricrepair <PI_netid> ilp
sbatch dsq_real_ilp_submit.sh
# collect + analyze (locally, sage -python)
python experiments/collect.py --indir results_real --out results_real_all.csv
sage -python experiments/real_analyze.py --results results_real_all.csv --outdir analysis
```
