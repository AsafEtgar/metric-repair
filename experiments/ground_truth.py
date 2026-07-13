"""ground_truth.py -- build & load node-aligned TRUE metrics for the ground-truth-recovery experiments
(R2/R3, see REAL_EXPERIMENTS.md). The observed processed graph is a noisy/incomplete proxy of these.

Strong tier (built here):
  ripe_atlas          -- anchor lat/lon (anchors_meta.csv) -> haversine geographic distance.
  nmr_1d3z_residue    -- 1D3Z.pdb model 1 -> min proton-proton 3D distance between residues.
  nmr_1d3z_atom       -- same, between NOE atom-groups (resid:token, wildcards *,# over proton names).
  dimacs_ny_d/_t      -- DIMACS .co node coordinates -> haversine geography. The road net's true metric is
                         the map it is drawn on; _d (metres) and _t (travel time) share one geography.
  pbmc3k_cosine_knn   -- the ambient PCA-50 expression space (d* = 1 - Pn @ Pn.T) the 15-NN graph is drawn
                         from: the graph keeps 15 neighbours per cell, d* holds all 2,700.

Artifacts land in data/processed/gt/:  <graph>__coords.csv (node,lat,lon)  for coord-based (haversine),
or  <graph>__truedist.npz (nodes, D)  for set-based min distances (nmr). `load_gt(graph)` returns
(nodes, D_true) either way; the R2 module aligns D_true to a component's node order.

    sage -python experiments/ground_truth.py --build ripe_atlas nmr_1d3z_residue nmr_1d3z_atom
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_edgelist                                                # noqa: E402

RAW = os.path.join("data", "raw")
GT_DIR = os.path.join("data", "processed", "gt")


# ----------------------------------------------------------------------------
# RIPE Atlas: anchor lat/lon -> haversine
# ----------------------------------------------------------------------------

def _haversine(lat, lon):
    """Full pairwise great-circle distance (km) for arrays of lat/lon (degrees)."""
    la = np.radians(lat)[:, None]; lo = np.radians(lon)[:, None]
    dla = la - la.T; dlo = lo - lo.T
    a = np.sin(dla / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlo / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_ripe_gt():
    import csv
    G = load_edgelist(os.path.join("data", "processed", "ripe_atlas.csv"))
    meta = {}
    with open(os.path.join(RAW, "ripe_atlas", "anchors_meta.csv")) as f:
        for row in csv.DictReader(f):
            try:
                meta[int(row["id"])] = (float(row["lat"]), float(row["lon"]))
            except (ValueError, KeyError):
                pass
    nodes = [u for u in G.nodes() if int(u) in meta]
    miss = [u for u in G.nodes() if int(u) not in meta]
    os.makedirs(GT_DIR, exist_ok=True)
    path = os.path.join(GT_DIR, "ripe_atlas__coords.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["node", "lat", "lon"])
        for u in nodes:
            la, lo = meta[int(u)]; w.writerow([u, la, lo])
    print(f"ripe_atlas GT: {len(nodes)}/{G.number_of_nodes()} anchors mapped "
          f"({len(miss)} missing) -> {path}")
    return path


# ----------------------------------------------------------------------------
# DIMACS road network: the challenge's .co coordinate file -> haversine geography
# ----------------------------------------------------------------------------

def build_dimacs_gt(graph):
    """USA-road-d.NY.co.gz -> gt/<graph>__coords.csv. The .co lines are `v <id> <lon> <lat>` in
    MICRODEGREES, LONGITUDE FIRST (the DIMACS convention -- reversing them silently rotates the map).
    build_real_graphs.py writes ORIGINAL DIMACS node ids into the processed CSV, so the join is direct.
    Both dimacs_ny_d (metres) and dimacs_ny_t (travel time) share the same node set and the same geography."""
    import csv
    import gzip
    G = load_edgelist(os.path.join("data", "processed", f"{graph}.csv"))
    want = {str(u) for u in G.nodes()}
    meta = {}
    with gzip.open(os.path.join(RAW, "dimacs", "USA-road-d.NY.co.gz"), "rt") as f:
        for ln in f:
            if not ln.startswith("v "):
                continue
            _, nid, lon, lat = ln.split()
            if nid in want:
                meta[nid] = (int(lat) / 1e6, int(lon) / 1e6)          # microdegrees -> degrees
    nodes = [u for u in G.nodes() if str(u) in meta]
    lats = [meta[str(u)][0] for u in nodes]
    lons = [meta[str(u)][1] for u in nodes]
    # NY city/state box -- a lat/lon swap or a sign error lands far outside it, and would otherwise pass
    # silently as a plausible-looking but meaningless "geography".
    assert 40.0 < min(lats) and max(lats) < 45.5, f"{graph}: latitudes out of NY range {min(lats)}..{max(lats)}"
    assert -80.0 < min(lons) and max(lons) < -71.0, f"{graph}: longitudes out of NY range {min(lons)}..{max(lons)}"
    os.makedirs(GT_DIR, exist_ok=True)
    path = os.path.join(GT_DIR, f"{graph}__coords.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["node", "lat", "lon"])
        for u in nodes:
            la, lo = meta[str(u)]; w.writerow([u, la, lo])
    print(f"{graph} GT: {len(nodes)}/{G.number_of_nodes()} nodes mapped -> {path}")
    return path


# ----------------------------------------------------------------------------
# pbmc3k single-cell RNA: the ambient expression space the k-NN graph is drawn from
# ----------------------------------------------------------------------------

def build_pbmc3k_gt():
    """pbmc3k_cosine_knn -> gt/pbmc3k_cosine_knn__truedist.npz (nodes, D), D = ambient cosine distance.

    THE TRUTH. `build_pbmc3k` (build_real_graphs.py) log-CPM normalizes the 10x counts, keeps the top-2000
    highly-variable genes, projects to 50 PCs, row-normalizes to `Pn`, and sets each k-NN edge weight to
    `1 - (Pn @ Pn.T)[i, j]`. The true distance is that same ambient matrix, over ALL pairs:

            d*  =  1 - Pn @ Pn.T          (2700 x 2700, cosine distance in PCA-50 expression space)

    EXTERNAL, NOT CIRCULAR. d* is not read off the graph: the graph keeps only 15 neighbours per cell and
    throws the other 2,684 columns of every row away; its shortest paths GUESS them. The 31,639 surviving
    edge weights literally equal d* on their pairs (asserted below to 0 error), which is exactly what makes
    the comparison honest -- the graph is a sparse, path-completed proxy of a space we can compute in full.
    Asking whether repair moves its shortest-path metric closer to d* is the same question `ripe_atlas` asks
    of geography and `nmr_1d3z` of the PDB structure.

    ORDER. Rows are CELL order (expression-matrix row index), and `nodes` are those indices as STRINGS,
    "0".."2699". downstream_recovery.load_graph sorts labels as strings ('10' < '2'), so the two orders
    DIFFER; true_distances() does the permutation from this npz's `nodes` array. Writing D in the loader's
    order instead would double-permute it and yield plausible, stable, meaningless numbers.

    Reuses mds_rna.ambient_truth verbatim (imported lazily -- it drags in scipy/sklearn-free SVD only when
    this builder actually runs), so the truth here and the truth the MDS figure uses cannot drift apart.
    """
    from mds_rna import ambient_truth                     # noqa: E402  (lazy: heavy, and only needed here)
    graph = "pbmc3k_cosine_knn"
    G = load_edgelist(os.path.join("data", "processed", f"{graph}.csv"))
    labels = sorted(int(u) for u in G.nodes())
    n = len(labels)
    assert labels == list(range(n)), \
        f"{graph}: node labels are not exactly 0..{n - 1}; they cannot index the expression matrix"

    D = ambient_truth(n)                                  # cell order: row i == cell i
    assert D.shape == (n, n), f"{graph}: truth is {D.shape}, expected ({n}, {n})"

    # The graph's edge weights ARE 1 - sim on those pairs. If the space or the order is wrong this is the
    # one check that catches it -- everything downstream would otherwise be plausible and meaningless.
    err = max(abs(float(w) - D[int(u), int(v)]) for u, v, w in G.edges(data="weight"))
    assert err < 1e-9, f"{graph}: max |edge weight - d*| = {err:.2e} -- truth is in the wrong space or order"

    os.makedirs(GT_DIR, exist_ok=True)
    path = os.path.join(GT_DIR, f"{graph}__truedist.npz")
    np.savez(path, nodes=np.array([str(i) for i in range(n)]), D=D)
    print(f"{graph} GT: {n} cells, ambient cosine (PCA-50); max |edge weight - d*| over "
          f"{G.number_of_edges()} edges = {err:.2e} -> {path}")
    return path


# ----------------------------------------------------------------------------
# NMR 1D3Z: PDB model-1 proton coordinates -> min proton-proton distance
# ----------------------------------------------------------------------------

def _pdb_model1_atoms(path):
    """{resSeq: {atomName: np.array([x,y,z])}} for MODEL 1 (all atoms; proton filter applied later)."""
    atoms, in_model = {}, False
    with open(path) as f:
        for ln in f:
            if ln.startswith("MODEL"):
                in_model = True
            elif ln.startswith("ENDMDL"):
                break
            elif in_model and ln.startswith(("ATOM", "HETATM")):
                name = ln[12:16].strip()
                res = int(ln[22:26])
                xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
                atoms.setdefault(res, {})[name] = xyz
    return atoms


def _protons(res_atoms):
    return {n: c for n, c in res_atoms.items() if n.startswith("H")}


def _group_coords(resid, token, atoms):
    """3D coords of the proton(s) a NOE atom-group token names, in residue `resid`. Wildcards *,# =
    prefix match; HN = amide H (H / HN, else N-terminal H1/H2/H3); else exact then prefix fallback."""
    prot = _protons(atoms.get(resid, {}))
    if token == "HN":
        names = [n for n in prot if n in ("H", "HN")] or [n for n in prot if n in ("H1", "H2", "H3")]
    elif token[-1] in "*#":
        pre = token[:-1]
        names = [n for n in prot if n.startswith(pre)]
    else:
        names = [n for n in prot if n == token] or [n for n in prot if n.startswith(token)]
    return [prot[n] for n in names]


def _min_dist(A, B):
    """Min pairwise Euclidean distance between two lists of 3D points (np.inf if either empty)."""
    if not A or not B:
        return np.inf
    A = np.asarray(A); B = np.asarray(B)
    return float(np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min())


def _node_points(graph, atoms):
    """For each processed-graph node, the list of 3D proton coords it maps to (empty = unmappable)."""
    G = load_edgelist(os.path.join("data", "processed", f"{graph}.csv"))
    nodes = list(G.nodes())
    pts = {}
    if graph == "nmr_1d3z_residue":
        for u in nodes:
            pts[u] = list(_protons(atoms.get(int(u), {})).values())
    else:                                                     # nmr_1d3z_atom: "resid:token"
        for u in nodes:
            resid, token = str(u).split(":", 1)
            pts[u] = _group_coords(int(resid), token, atoms)
    return nodes, pts


def build_nmr_gt(graph):
    atoms = _pdb_model1_atoms(os.path.join(RAW, "nmr", "1D3Z.pdb"))
    nodes, pts = _node_points(graph, atoms)
    mapped = [u for u in nodes if pts[u]]
    D = np.full((len(mapped), len(mapped)), np.inf)
    for i, u in enumerate(mapped):
        D[i, i] = 0.0
        for j in range(i + 1, len(mapped)):
            d = _min_dist(pts[u], pts[mapped[j]]); D[i, j] = D[j, i] = d
    os.makedirs(GT_DIR, exist_ok=True)
    path = os.path.join(GT_DIR, f"{graph}__truedist.npz")
    np.savez(path, nodes=np.array([str(u) for u in mapped]), D=D)
    print(f"{graph} GT: {len(mapped)}/{len(nodes)} nodes mapped -> {path}")
    return path


# ----------------------------------------------------------------------------
# Unified loader
# ----------------------------------------------------------------------------

def load_gt(graph):
    """Return (nodes, D_true): node list + aligned true-distance matrix (km for ripe, Å for nmr)."""
    coords = os.path.join(GT_DIR, f"{graph}__coords.csv")
    dist = os.path.join(GT_DIR, f"{graph}__truedist.npz")
    if os.path.exists(dist):
        z = np.load(dist, allow_pickle=True)
        return list(z["nodes"]), z["D"]
    if os.path.exists(coords):
        import csv
        nodes, lat, lon = [], [], []
        with open(coords) as f:
            for row in csv.DictReader(f):
                nodes.append(row["node"]); lat.append(float(row["lat"])); lon.append(float(row["lon"]))
        return nodes, _haversine(np.array(lat), np.array(lon))
    raise SystemExit(f"no GT artifact for {graph} (run --build {graph})")


def subdist(nodes_all, D_all, nodes_sub):
    """Reorder/subselect D_all to the ordering `nodes_sub` (str-compared); returns (kept_nodes, D_sub)."""
    idx = {str(u): i for i, u in enumerate(nodes_all)}
    keep = [u for u in nodes_sub if str(u) in idx]
    ii = [idx[str(u)] for u in keep]
    return keep, D_all[np.ix_(ii, ii)]


BUILDERS = {"ripe_atlas": build_ripe_gt,
            "nmr_1d3z_residue": lambda: build_nmr_gt("nmr_1d3z_residue"),
            "nmr_1d3z_atom": lambda: build_nmr_gt("nmr_1d3z_atom"),
            "dimacs_ny_d": lambda: build_dimacs_gt("dimacs_ny_d"),
            "dimacs_ny_t": lambda: build_dimacs_gt("dimacs_ny_t"),
            "pbmc3k_cosine_knn": build_pbmc3k_gt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", nargs="+", default=list(BUILDERS), choices=list(BUILDERS))
    a = ap.parse_args()
    for g in a.build:
        BUILDERS[g]()


if __name__ == "__main__":
    main()
