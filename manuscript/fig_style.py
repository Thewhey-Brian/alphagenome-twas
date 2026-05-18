"""Unified publication-quality style for all manuscript figures.

Targets Cell Press / AJHG formatting:
  - Arial (Helvetica fallback), 7pt body / 6pt ticks
  - Single column: 85mm (3.35in), 1.5 col: 114mm (4.49in), double: 174mm (6.85in)
  - 300 dpi raster, editable PDF fonts (Type 42)
  - Clean spines, minimal gridlines, colorblind-friendly Okabe-Ito palette
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Color palette (Okabe-Ito colorblind-safe) ────────────────
# From: Okabe & Ito (2008), widely recommended by Nature Methods
C_AG     = "#E69F00"   # orange — AlphaGenome
C_TWAS   = "#0072B2"   # blue — TWAS
C_BOTH   = "#CC79A7"   # reddish purple — overlap/both
C_GTEX   = "#009E73"   # bluish green — GTEx
C_GREY   = "#999999"   # neutral
C_ORANGE = "#E69F00"   # orange (alias)
C_BLUE   = "#56B4E9"   # sky blue — accent
C_RED    = "#D55E00"   # vermillion — emphasis/threshold
C_YELLOW = "#F0E442"   # yellow — accent (rarely used)
C_BLACK  = "#000000"   # black

# ── Figure widths (inches) ───────────────────────────────────
SINGLE_COL = 3.35   # 85 mm  (Cell Press single column)
ONE_HALF   = 4.49   # 114 mm (Cell Press 1.5 column)
DOUBLE_COL = 6.85   # 174 mm (Cell Press full/double column)

# ── RC params ────────────────────────────────────────────────
RCPARAMS = {
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.title_fontsize": 7,
    # Lines & spines
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    # Grid
    "axes.grid": False,
    # Export
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,        # editable text in PDF
    "ps.fonttype": 42,
}


def apply_style():
    """Apply publication rcParams globally."""
    plt.rcParams.update(RCPARAMS)


def panel_label(ax, label, x=-0.12, y=1.08, **kw):
    """Add bold lowercase panel label (Nature style)."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left", **kw)


def despine(ax, keep_left=True, keep_bottom=True):
    """Remove specified spines."""
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
    if not keep_bottom:
        ax.spines["bottom"].set_visible(False)


def stat_box(ax, text, x=0.97, y=0.95, **kw):
    """Small statistics annotation box."""
    defaults = dict(transform=ax.transAxes, ha="right", va="top",
                    fontsize=5.5, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#cccccc", alpha=0.9, lw=0.4))
    defaults.update(kw)
    ax.text(x, y, text, **defaults)
