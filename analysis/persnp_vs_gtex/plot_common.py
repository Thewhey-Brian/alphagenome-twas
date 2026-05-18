"""Shared plotting config."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
RCPARAMS = {
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif", "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
}
def apply_style():
    plt.rcParams.update(RCPARAMS)
