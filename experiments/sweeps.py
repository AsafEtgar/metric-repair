"""Canonical RGG sweep -> x-axis mapping. Imported by BOTH rgg_analyze.py and rgg_plots.py.

These lived in two places and drifted: rgg_plots knew about the deflate/mixed/real-recovery sweeps while
rgg_analyze did not, so `add_x` left their `x` as NaN and every one of them was silently dropped from the
figures -- the deflate-kNN sweeps (P2df/P2dm), the deflate size ladder (S1d), the whole mixed campaign, and
all of realrec. Those are the "does repair help kNN" and "can we recover a road network" results.

Adding a sweep? Add it HERE. rgg_analyze prints a warning for any sweep not in SWEEP_X, so a missing entry
is loud rather than silent -- but only if this map is the single source of truth.

No matplotlib import: rgg_analyze must stay importable without a display.
"""

# sweep -> the config column that varies along it (its natural x-axis)
SWEEP_X = {
    # size ladders
    "S1": "n", "S1d": "n", "S1m": "n", "P2size": "n",
    "POCsize_inflate": "n", "POCsize_jitter": "n",
    # density
    "S2": "deg", "S2k": "k",
    # corruption magnitude
    "S3": "magnitude", "S3d": "magnitude", "S3m": "magnitude", "S6": "magnitude",
    "P2dm": "magnitude", "P2mm": "magnitude",
    # corruption fraction
    "S4i": "frac_q", "S4d": "frac_q", "S4m": "frac_q",
    "P2df": "frac_q", "P2mf": "frac_q",
    # jitter
    "S5a": "n_jitter", "P2n": "n_jitter",
    "S5b": "jitter", "P2j": "jitter",
    "S5c": "subset_s", "P2s": "subset_s",
    # real-base recovery: one curve per corruption direction, swept over the corrupted fraction.
    # NB the five base graphs are separated by the `base` column, not by the sweep id -- see
    # rgg_analyze.attach_base(). Pooling them would average a road network with a scRNA kNN graph.
    "RR_inflate": "frac_q", "RR_deflate": "frac_q", "RR_mixed": "frac_q",
}

# 2-D sweeps: the secondary knob becomes a line series within one panel
SWEEP_SERIES = {"S6": "frac_q"}

SWEEP_TITLE = {
    "S1": "S1 size (inflate)", "S1d": "S1d size (deflate)", "S1m": "S1m size (mixed)",
    "S2": "S2 density (radius)", "S2k": "S2' density (knn)",
    "S3": "S3 inflate mag.", "S3d": "S3' deflate mag.", "S3m": "S3m mixed mag.",
    "S4i": "S4 inflate frac", "S4d": "S4' deflate frac", "S4m": "S4m mixed frac",
    "S5a": "S5a jitter count", "S5b": "S5b jitter mag.", "S5c": "S5c jitter subset",
    "S6": "S6 mag×frac (q averaged)",
    "P2size": "P2 size (jitter)", "P2n": "P2 jitter count", "P2j": "P2 jitter mag.", "P2s": "P2 subset",
    "P2df": "P2 kNN vs deflate frac", "P2dm": "P2 kNN vs deflate mag.",
    "P2mf": "P2 kNN vs mixed frac", "P2mm": "P2 kNN vs mixed mag.",
    "RR_inflate": "real recovery: inflate", "RR_deflate": "real recovery: deflate",
    "RR_mixed": "real recovery: mixed",
    "POCsize_inflate": "POC size (inflate)", "POCsize_jitter": "POC size (jitter)",
}
