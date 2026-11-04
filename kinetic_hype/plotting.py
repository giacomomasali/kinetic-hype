"""Figure style for the notebooks.

Each quantity has a fixed colour and line style; the mean-field solution is a
solid line and the DSMC a dotted black line, so no plot relies on colour alone.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

QUANTITY = {
    "m1": dict(color="#2a78d6", label=r"Expectations $m_1(t)$"),
    "rho": dict(color="#eb6834", label=r"Adoption $\rho_2(t)$"),
    "r": dict(color="#1baf7a", label=r"Realized performance $r(t)$"),
}


def use_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "font.size": 10, "axes.titlesize": 11,
        "legend.frameon": False, "figure.dpi": 110,
    })


def plot_trajectory(ax, df, cols=("m1", "rho", "r"), dsmc=None, label_dsmc="DSMC"):
    """Mean-field curves (solid, coloured) and optional DSMC overlay (dotted black)."""
    for c in cols:
        ax.plot(df["t"], df[c], **QUANTITY[c])
    if dsmc is not None:
        for k, c in enumerate(cols):
            ax.plot(dsmc["t"], dsmc[c], color=INK, ls=":", lw=1.6,
                    label=label_dsmc if k == 0 else None)
    ax.set_xlabel("time")
