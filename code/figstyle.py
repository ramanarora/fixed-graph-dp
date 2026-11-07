"""Shared matplotlib styling for publication-quality figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#56B4E9"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300,
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "legend.fontsize": 8.2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "lines.linewidth": 1.7, "lines.markersize": 5,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})


def save(fig, name, figdir):
    fig.savefig(f"{figdir}/{name}.pdf")
    fig.savefig(f"{figdir}/{name}.png")
    print(f"saved {name}.pdf/.png")
