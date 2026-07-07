# Real datasets — fetch log & provenance

Datasets pulled for metric repair (incomplete, weighted, non-negative, undirected; edge weight =
distance-like). Raw sources land in `data/raw/<dataset>/` (gitignored, immutable); cleaned
metric-repair-ready graphs go to `data/processed/`. Fetched **2026-07-07**.

Selection follows the spec table the user provided; the four "quick-start" rows were fetched first.
Each still needs a **build step** (below) to become a weighted graph — none ships as a ready distance
matrix except DIMACS (which is far too large and needs subsampling).

---

## Fetched (raw sources on disk)

| dataset | picked variant | source | files (`data/raw/…`) | size | key stats |
|---|---|---|---|---|---|
| **DIMACS road (NY)** | distance `-d` **and** travel-time `-t` | 9th DIMACS Challenge — `www.diag.uniroma1.it/challenge9/data/` | `dimacs/USA-road-{d,t}.NY.gr.gz` | 3.5M + 3.6M | 264,346 nodes / 733,846 arcs each |
| **RIPE Atlas latency** | IPv4 ping mesh, ~1000 anchors | REST `atlas.ripe.net/api/v2` (no key) | `ripe_atlas/rtt_matrix.csv`, `rtt_edgelist.csv`, `anchors_meta.csv` | 7.4M + 8.3M | **1000 anchors, 442,707 edges (88.6% fill)**; min RTT/pair ms, symmetrized; median 133 ms, max ~998 (trim outliers) |
| **scRNA pbmc3k** | cosine kNN | 10x Genomics `cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k` | `scrna_pbmc3k/filtered_gene_bc_matrices/hg19/{matrix.mtx,genes.tsv,barcodes.tsv}` | 7.3M tar | 32,738 genes × 2,700 cells |
| **Cassiopeia spatial** | spatial dataset | GitHub `YosefLab/Cassiopeia` `notebooks/data/` | `cassiopeia_spatial/spatial_{adata.h5ad,allele_table.txt}` | 42M + 3.9M | allele table 27,272 rows, 10,066 cells × 18 intBC sites, 3 lineage groups |
| **BCT brain (largest)** | largest on the BCT site | Brain Connectivity Toolbox `sites.google.com/site/bctnet` (2019-03-03 dist, user-downloaded) | `bct/{Coactivation_matrix,GroupAverage_rsfMRI_matrix}.mat` | 342K + 353K | **638 nodes, 18,625 edges (9.2% dense)**, symmetric, + 638×3 coords |
| **Fly CNS connectome** | male-CNS v1.0 flat connectome | GCS `flyem-male-cns/v1.0/…/flat-connectome` (no token) | `flycns/connectome-weights-…feather` (1.0G), `body-annotations-…feather` (14M) | 1.0G + 14M | full table **151.9M rows**; neuron↔neuron subset = **26.0M edges / ~185k neurons**; weight=synapse count |
| **NMR ubiquitin** | canonical entry, both granularities | RCSB PDB entry **1D3Z** (Cornilescu/Bax 1998) | `nmr/1D3Z.pdb` (992K, 10-model ensemble), `1D3Z.mr` (1.1M, XPLOR restraints) | 2.1M | **2,727 NOE restraints** → residue graph n≈75, atom graph n≈512; + reference model |
| **Fish1 zebrafish circuits** | 2 circuit zips (not whole-brain) | GCS `fish1-release/paper_data/` (no token) | `fish1/{TEN,HMI}_analysis/…` | 1.4M + 3.3M | **TEN**: synapse adjacency lists (24,286 in / 7,477 out neurons); **HMI**: 30,346 soma positions + build scripts |

**Access notes**
- **DIMACS:** `users.diag.uniroma1.it` (the address on the spec sheet) returns **HTTP 403** to non-browser
  clients; the `www.diag.uniroma1.it/challenge9/data/…` mirror serves the same files and works. `.gr` is a
  DIMACS shortest-path graph (`p sp <n> <m>`, then `a u v w` arcs); undirected in practice (each edge listed
  both ways). Weight `w` is integer (meters for `-d`, ~travel-time units for `-t`).
- **RIPE Atlas:** open read API, no key. 1,764 anchors total but only **1,064 are active + IPv4**; subset =
  smallest 1,000 anchor ids. Weight = per-pair **min RTT (ms)** from the *latest run* of each target's
  anchoring ping measurement, symmetrized by averaging both directions. Snapshot only — a temporal median
  over a window is a possible refinement. `anchors_meta.csv` carries lat/lon (ground-truth coordinates).
- **Fly CNS connectome:** `.feather` needs `pyarrow` (now installed in the Sage env: `sage -pip install pyarrow`).
  The 1 GB `connectome-weights` file is the *full* pre→post table over all 1.8M+ segmented bodies — the real
  neuron connectome is the subset where **both endpoints are in `body-annotations`** (211,577 bodies): 26.0M
  weighted **directed** edges among ~185k neurons. `status=='Traced'` + `somaLocation` gives a clean neuron
  set with approximate positions. Directed + synapse-count weight → symmetrize + invert for repair.
- **Fish1 zebrafish circuits:** whole-brain connectivity is **CAVE-only** (no flat connectome like Fly CNS);
  only two circuit packages are static-downloadable. **TEN** = pre-computed adjacency lists — parse
  `incoming/outgoing_synapses.csv` (partner-id list per `query_neuron`, repeats = synapse weight) → weighted
  **directed** graph; resolve ids to somas via `agglomerated_segments_and_soma_ids.csv`; 43 core TEN neurons
  in `TEN_cats.csv`. **HMI** = 30,346 soma positions (`cave_somas_in_big_box.csv`: `pt_position`, cell_type)
  + `em_zfish1_dataframe.xlsx` + `scripts/connectivity_matrix.py` to *generate* the adjacency (ships empty
  `results/`). Same directed synapse-count caveat as Fly CNS; positions = ground truth.
- **NMR ubiquitin (1D3Z):** `.mr` is XPLOR/CNS format — `assign (sel) (sel) d d- d+`, upper bound = `d+d+`,
  ambiguous atoms joined by `or` (methyls, `HE#`). NOE section = lines 17–4783 (2,727 restraints); later
  sections are H-bond / dihedral / RDC / scalar (ignore for the distance graph). `1D3Z.pdb` model 1 =
  ground-truth coordinates. Build both residue-level (n≈75) and atom-level (n≈512) graphs.
- **BCT brain:** `.mat` with two vars — the connectivity matrix (`Coactivation_matrix` / `GroupAverage_rsfMRI`) and `Coord` (638×3 node positions). Weights are **similarities** (higher = more connected = *closer*), so the build must invert them to distances (`1/w`, `max−w`, or `−log w`) before repair; `Coord` gives a Euclidean ground-truth metric. The largest of the BCT collection; others there are ≤95 nodes.
- **pbmc3k:** raw 10x counts (Matrix Market). Cosine kNN is **not** built yet (see below).
- **Cassiopeia:** `.h5ad` needs `h5py`/`anndata` (absent from the Sage env); the `.txt` allele table is
  plain TSV and is the input for barcode-dissimilarity. Spatial coords live in the `.h5ad`.

---

## Built graphs

`build_real_graphs.py` (repo root) turns every raw source into the weighted-graph representation, saves
edge lists to `data/processed/<name>.csv`, and writes `data/processed/REAL_GRAPHS_REPORT.csv` (n, m,
weight range, cycle bound, is_metric, non-metric fraction). Modeling choices are documented inline
(BFS-ball subsampling to n≈1000–1200, similarity→distance inversion, directed-connectome symmetrization,
kNN construction). Re-run with `sage -python build_real_graphs.py`.

## Build step (raw → metric-repair graph) — reference

- **DIMACS NY:** extract a connected **subgraph of n≈1000** (e.g. BFS ball from a seed node, or a
  bounding-box crop of the coordinates) for the exact-ILP path; keep the full graph for heuristics.
  `-d` ≈ metric baseline (few repairs), `-t` is the non-metric-interesting one.
- **RIPE Atlas:** already a distance matrix once the fetch finishes → `load_distance_matrix` after
  reshaping `rtt_matrix.csv`. Likely still large (n≈1000, dense-ish) → subsample for exact.
- **pbmc3k:** normalize → PCA → **cosine** kNN graph (choose k). No scanpy in the Sage env, so build with
  numpy/scipy directly (or install scanpy). Cosine → genuinely non-metric = the interesting repair target.
- **BCT brain (638 nodes):** invert similarity → distance, load via `load_mat_matrix` (var
  `Coactivation_matrix` or `GroupAverage_rsfMRI`); already the right size (n=638). Coords → ground-truth
  Euclidean metric for comparison.
- **Cassiopeia spatial:** cell × intBC allele table → **barcode dissimilarity** between cells
  (e.g. fraction of shared intBC sites with disagreeing alleles) → sparsify to a kNN graph;
  spatial coords / lineage tree = ground truth.

---

## Not yet fetched (deferred / optional)

- **CAIDA** latency — **deliberately skipped**: same modality as RIPE (internet RTT, non-metric via routing),
  so redundant now that RIPE is in hand.
- **Phylogenetic distances** (TreeBASE / OpenTree, or Pfam→FastME) — build a tree metric, sparsify. Option for later.
- **Google neural-mapping** (H01 / fly CNS / Fish1 / ZAPBench) — connectomics. Fly CNS + Fish1 **circuits**
  fetched (above); Fish1 **whole-brain** is CAVE-only (deferred), H01 / ZAPBench left as later options
  (directed, synapse-count weight, 100k+ nodes → weaker "distance-like" fit).
- **Single-cell lineage — C. elegans ground truth** — not in the Cassiopeia repo; external source needed.
