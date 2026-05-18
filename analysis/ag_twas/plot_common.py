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

C_AG = "#e74c3c"
C_TWAS = "#3498db"
C_BOTH = "#8e44ad"
C_GREY = "#7f8c8d"

def apply_style():
    plt.rcParams.update(RCPARAMS)
