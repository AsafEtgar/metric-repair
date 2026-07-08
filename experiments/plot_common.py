"""plot_common.py -- shared styling for the experiment figures (plots.py, rgg_plots.py).

With ~8-18 algorithms in a panel, COLOUR ALONE cannot separate the lines. So every algorithm gets a stable
(colour, marker, linestyle) TRIPLE -- composite encoding -- drawn from a colour-blind-safe order (Okabe-Ito).
Reused hues are then still told apart by marker + dash pattern, and the same algorithm looks identical in
every figure. One shared legend per figure, placed OUTSIDE the axes, replaces the per-panel legends that used
to cover the data. Median line + a light IQR band. Lines that genuinely sit on the same values (e.g. all the
exact algorithms at ratio 1.0) legitimately overlap -- the caller annotates those rather than fanning them out.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito colour-blind-safe categorical order (the light yellow #F0E442 is dropped -- poor contrast on white).
_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_LINESTYLES = ["-", "--", "-.", ":"]

FAMILY = {"gmr": {"GMR", "DOMR"}, "iomr": {"IOMR"}}      # MR axis -> variants drawn in that family's folder
FAMILIES = ["gmr", "iomr"]
FAM_TITLE = {"gmr": "GMR (general MR)", "iomr": "IOMR (increase-only MR)"}


def style_map(algos):
    """Stable algo -> {'color','marker','ls'} via composite encoding: hue cycles fastest, then marker+dash,
    so the first 7 algos are distinct hues, the next 7 the same hues with a new marker+dash, etc. Sorted for
    determinism so an algorithm keeps one look across every figure ('colour follows the entity')."""
    sty = {}
    for i, a in enumerate(sorted(set(algos))):
        j = i // len(_COLORS)
        sty[a] = {"color": _COLORS[i % len(_COLORS)],
                  "marker": _MARKERS[j % len(_MARKERS)],
                  "ls": _LINESTYLES[j % len(_LINESTYLES)]}
    return sty


def family_of(variant):
    return "gmr" if variant in FAMILY["gmr"] else "iomr"


def ylab(base, better=None):
    """y-axis label with an explicit direction-of-goodness cue ('up'/'down'/None)."""
    tag = {"up": "↑ higher is better", "down": "↓ lower is better"}.get(better)
    return f"{base}\n({tag})" if tag else base


def band(ax, sub, xcol, ycol, lo, hi, label, st, logx=False, logy=False, iqr=True, dim=False):
    """One algo's median line + (optional) IQR band with its composite style. `dim` de-emphasises a line that
    is being drawn only for completeness (e.g. the exact algorithms pinned at ratio 1.0). Returns True if drawn."""
    sub = sub.sort_values(xcol)
    x = sub[xcol].to_numpy(dtype=float)
    y = sub[ycol].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if logx:
        m &= (x > 0)
    if logy:
        m &= (y > 0)
    if not m.any():
        return False
    ax.plot(x[m], y[m], marker=st["marker"], ms=(4 if dim else 5.5), lw=(1.0 if dim else 1.7),
            ls=st["ls"], color=st["color"], label=label, alpha=(0.55 if dim else 1.0),
            markeredgecolor="white", markeredgewidth=0.4, zorder=(2 if dim else 3))
    if iqr and not dim and lo in sub and hi in sub:
        ylo, yhi = sub[lo].to_numpy(dtype=float), sub[hi].to_numpy(dtype=float)
        b = m & np.isfinite(ylo) & np.isfinite(yhi)
        if b.any():
            ax.fill_between(x[b], ylo[b], yhi[b], color=st["color"], alpha=0.12, lw=0)
    return True


def figure_legend(fig, title="algorithm", ncol=1):
    """ONE de-duplicated legend for the whole figure, to the RIGHT of the axes (never over the data). Collects
    handles from every sub-axis. Return it so save() can keep it inside the tight bbox."""
    seen = {}
    for ax in fig.axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    if not seen:
        return None
    return fig.legend(seen.values(), seen.keys(), loc="center left", bbox_to_anchor=(1.005, 0.5),
                      fontsize=8, frameon=False, ncol=ncol, title=title, title_fontsize=9)


def note(ax, text, loc="upper left"):
    """A small boxed annotation inside an axis (e.g. 'these algorithms coincide at 1.0')."""
    va, ha = ("top", "left") if "left" in loc else ("top", "right")
    x = 0.02 if "left" in loc else 0.98
    y = 0.97 if "upper" in loc else 0.03
    va = "top" if "upper" in loc else "bottom"
    ax.text(x, y, text, transform=ax.transAxes, fontsize=7, va=va, ha=ha,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))


def save(fig, outdir, name, legend=None):
    """Write PDF + PNG, keeping an outside legend inside the tight bounding box."""
    os.makedirs(outdir, exist_ok=True)
    extra = [legend] if legend is not None else None
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight",
                    bbox_extra_artists=extra, dpi=150)
    plt.close(fig)
    print(f"    wrote {name}.pdf / .png")
