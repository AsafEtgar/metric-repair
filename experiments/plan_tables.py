import pandas as pd, numpy as np
K = 20                      # pbmc3k is a 15-NN graph: k<=15 is a construction ceiling. k=20 is the honest cell.
TOL = 0.01

# ---------- Table: real datasets and their properties ----------
R = pd.read_csv("data/processed/REAL_GRAPHS_REPORT.csv").set_index("dataset")
M = pd.read_csv("analysis/summary_mds_sweep.csv")
rna = pd.read_csv("analysis/summary_mds_rna.csv")
H = {g: int(v) for g, v in M[M.algo == "domr"].set_index("graph").cover_size.items()}
H["pbmc3k_cosine_knn"] = int(rna[rna.algo == "domr"].cover_size.iloc[0])

ROWS = [
    ("dimacs_ny_d",            "road network (distance)",   r"geography (lat/lon)"),
    ("dimacs_ny_t",            "road network (travel time)",r"geography (lat/lon)"),
    ("ripe_atlas",             "internet latency",          r"geography (lat/lon)"),
    ("pbmc3k_cosine_knn",      "single-cell RNA-seq",       r"expression space (PCA-50)"),
    ("nmr_1d3z_atom",          "NMR contact map (atom)",    r"3-D fold"),
    ("nmr_1d3z_residue",       "NMR contact map (residue)", r"3-D fold"),
    ("bct_coactivation",       "fMRI co-activation",        r"\code{--}"),
    ("cassiopeia_barcode_knn", "lineage barcodes",          r"\code{--}"),
    ("flycns_male",            "fly connectome",            r"\code{--}"),
    ("fish1_ten",              "fish connectome",           r"\code{--}"),
]
def tex(g): return r"\code{%s}" % g.replace("_", r"\_")
def th(x): return f"{x:,}".replace(",", r"{,}")

L = [r"\begin{table}[!t]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
     r"\caption{The real graphs, and the one number that governs everything downstream: $|H|/m$, the "
     r"fraction of edges that are non-metric. It spans four orders of magnitude. \code{dimacs\_ny\_d} is "
     r"\emph{exactly} metric ($|H| = 0$) and so serves as the base for planted corruption; \code{ripe\_atlas} "
     r"is $95.3\%$ non-metric, so every cover must rewrite most of the graph. Only the top six carry an "
     r"external ground truth, and only those can answer whether repair moves a graph \emph{toward} anything. "
     r"\textbf{One truth is partial and we say so:} \code{nmr\_1d3z\_atom} has $430$ nodes but only $343$ of "
     r"them ($79.8\%$) map to a proton in the PDB structure, so every recovery number on that graph is "
     r"computed over a $343$-node core (and its MDS core is $340$, being the intersection with the graph's "
     r"connected component). Every other truth covers its graph completely.}",
     r"\label{tab:realdata}",
     r"\begin{tabular}{@{}lllrrrrl@{}}", r"\toprule",
     r"graph & domain & $n$ & $m$ & $|H|$ & $|H|/m$ & external truth \\", r"\midrule"]
for g, dom, truth in ROWS:
    r = R.loc[g]
    frac = float(r.nonmetric_frac)
    h = H.get(g)
    if h is None:                      # no domr row: |H| = frac * m, which is how the report defines frac
        h = int(round(frac * float(r.m)))
    hs = th(h)
    L.append(f"{tex(g)} & {dom} & ${th(int(r.n))}$ & ${th(int(r.m))}$ & ${hs}$ & ${frac:.4f}$ & {truth} \\\\")
    if g == "nmr_1d3z_residue":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
print("\n".join(L)); print()

# ---------- Table: at best a tie, at worst a disaster ----------
P = pd.read_csv("analysis/summary_pure_real.csv"); P = P[(P.k == K) & (P.algo != "domr")]
rna["graph"] = "pbmc3k_cosine_knn"
MM = pd.concat([M, rna], ignore_index=True)
MM = MM[MM.status.fillna("ok").eq("ok") & MM.disp_smacof.notna()]
NAME = {"dimacs_ny_t": r"\code{dimacs\_ny\_t}", "pbmc3k_cosine_knn": r"\code{pbmc3k}",
        "nmr_1d3z_atom": r"\code{nmr\_atom}", "nmr_1d3z_residue": r"\code{nmr\_residue}",
        "ripe_atlas": r"\code{ripe\_atlas}"}
def verdict(g):
    return r"\textbf{WIN}" if g > TOL else (r"\textbf{LOSE}" if g < -TOL else "tie")
L = [r"\begin{table}[!t]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
     r"\caption{\textbf{On real graphs, repair at best ties --- and at worst is a disaster.} Topology is "
     r"$k$-NN Jaccard against the external truth ($k = 20$; the sign of every entry is unchanged at "
     r"$k = 5, 10$), geometry the Procrustes disparity against the same truth. \emph{best} is the best "
     r"algorithm in the suite on that axis; \emph{med.} is the median over every algorithm that returned a "
     r"cover. \textbf{Read \emph{med.}}: choosing the best algorithm requires the ground truth one is trying "
     r"to recover, so \emph{best} is an oracle and \emph{med.} is what a practitioner actually gets. "
     r"\textbf{No repair improves topology on any real graph} --- the best lift anywhere is $-0.0000$. "
     r"Geometry improves on exactly one graph. On \code{ripe\_atlas} the median algorithm destroys $91\%$ of "
     r"the recoverable neighbourhood structure. \DOMR{} is the control and is exactly $0$ on both axes "
     r"(Lemma~\ref{lem:domr}).}",
     r"\label{tab:verdict}",
     r"\begin{tabular}{@{}lrrrlrrrl@{}}", r"\toprule",
     r"& \multicolumn{4}{c}{topology: $k$-NN Jaccard ($\uparrow$)} "
     r"& \multicolumn{4}{c}{geometry: Procrustes disparity ($\downarrow$)} \\",
     r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
     r"graph & obs. & med. & best & verdict & obs. & med. & best & verdict \\", r"\midrule"]
for g in ["dimacs_ny_t", "pbmc3k_cosine_knn", "nmr_1d3z_atom", "nmr_1d3z_residue", "ripe_atlas"]:
    T = P[P.graph == g]
    ko = float(T.recovery_obs.iloc[0])
    kb = ko + float(T.lift_med.max()); km = ko + float(T.lift_med.median())
    G = MM[MM.graph == g]
    go = float(G[G.algo == "observed"].disp_smacof.iloc[0])
    Gr = G[~G.algo.isin(["observed", "domr"])]
    gb = float(Gr.disp_smacof.min()); gm = float(Gr.disp_smacof.median())
    vk = verdict((km - ko) / ko); vg = verdict((go - gm) / go)
    L.append(f"{NAME[g]} & ${ko:.4f}$ & ${km:.4f}$ & ${kb:.4f}$ & {vk} & "
             f"${go:.4f}$ & ${gm:.4f}$ & ${gb:.4f}$ & {vg} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
print("\n".join(L))
