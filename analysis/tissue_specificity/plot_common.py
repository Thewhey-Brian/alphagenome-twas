"""Shared plotting config for tissue specificity analysis."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RCPARAMS = {
    "font.size": 7, "font.family": "sans-serif",
    "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.linewidth": 0.4,
    "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.spines.top": False, "axes.spines.right": False,
}

C_AG = "#4C72B0"
C_GTEX = "#55A868"
C_RED = "#C44E52"
C_GREY = "#BDBDBD"
C_ORANGE = "#E8A838"
C_PURPLE = "#8C6BB1"

OUT_DIR = "results/insights/tissue_specificity"

def apply_style():
    plt.rcParams.update(RCPARAMS)

def clean_tissue(t):
    return (t.replace("_", " ")
             .replace("EBV-transformed", "EBV")
             .replace("cervical c-1", "cervical c1"))
