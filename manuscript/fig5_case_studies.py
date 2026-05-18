#!/usr/bin/env python3
"""Figure 5: AG-prioritised non-coding case studies (3 panels).

  a. CDKN2B-AS1 (AG-exclusive; no TWAS model)
  b. PTENP1     (shared; TWAS model exists but not significant)
  c. HOTAIR     (shared; TWAS model exists but not significant)

Each panel is a bipartite trait × tissue diagram of (trait, tissue) entries at
which AlphaGenome calls the gene significant (|max_lfc| > 0.01).

Encoding:
  Link width   ∝ |max_lfc|   (AG effect magnitude)
  Link colour  = TWAS −log10(P)   (panels b, c only; panel a uses AG orange)
"""
import sys
from pathlib import Path
import os as _os
ROOT = _os.environ.get("AG_LD_ROOT", _os.getcwd())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_style, panel_label, C_AG, C_TWAS, C_GREY, DOUBLE_COL
apply_style()

DATA_DIR = Path(ROOT) / "insights/ag_twas"
TRAITS_CSV = Path(ROOT) / "insights/manuscript/tables/tableS1_gwas_traits.csv"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

THRESH = 0.01
GENES = ["CDKN2B-AS1", "PTENP1", "HOTAIR"]

# ── Load data ────────────────────────────────────────────────
gs = pd.read_parquet(DATA_DIR / "merged" / "gene_scores.parquet",
                     columns=["gene_name", "tissue", "accession",
                              "max_lfc", "max_abs_lfc"])
gs = gs[gs["gene_name"].isin(GENES)]

m = pd.read_parquet(DATA_DIR / "merged" / "ag_twas_merged.parquet",
                    columns=["gene_name", "tissue", "accession",
                             "TWAS.P", "TWAS.Z", "twas_sig_bonf", "_merge"])
m = m[m["gene_name"].isin(GENES) & (m["_merge"] == "both")]

traits_tbl = pd.read_csv(TRAITS_CSV).set_index("Accession")["Trait"].to_dict()


def pretty_tissue(t: str) -> str:
    t = t.replace("_", " ")
    # shorten a few common long names for legibility
    t = (t.replace("Cells EBV-transformed lymphocytes", "Cells EBV-lymphocytes")
           .replace("Cells Cultured fibroblasts", "Cells fibroblasts")
           .replace("Brain Anterior cingulate cortex BA24", "Brain Ant. cing. cortex")
           .replace("Skin Not Sun Exposed Suprapubic", "Skin (unexposed)")
           .replace("Skin Sun Exposed Lower leg", "Skin (sun-exp. leg)")
           .replace("Esophagus Gastroesophageal Junction", "Esophagus GJ")
           .replace("Adipose Visceral Omentum", "Adipose (visceral)")
           .replace("Adipose Subcutaneous", "Adipose (subQ)")
           .replace("Small Intestine Terminal Ileum", "Small Int. terminal ileum")
           .replace("Heart Left Ventricle", "Heart LV")
           .replace("Heart Atrial Appendage", "Heart atrial app.")
           .replace("Brain Nucleus accumbens basal ganglia", "Brain nucleus acc.")
           .replace("Brain Putamen basal ganglia", "Brain putamen")
           .replace("Brain Caudate basal ganglia", "Brain caudate")
           .replace("Brain Spinal cord cervical c-1", "Brain spinal cord"))
    return t


def collect(gene: str) -> pd.DataFrame:
    sig = gs[(gs["gene_name"] == gene) & (gs["max_abs_lfc"] > THRESH)].copy()
    sig = (sig.sort_values("max_abs_lfc", ascending=False)
              .drop_duplicates(subset=["accession", "tissue"], keep="first"))
    sig["trait"] = sig["accession"].map(traits_tbl).fillna(sig["accession"])
    sig["tissue_label"] = sig["tissue"].map(pretty_tissue)
    tw = m[m["gene_name"] == gene].copy()
    tw = (tw.sort_values("TWAS.P")
            .drop_duplicates(subset=["accession", "tissue"], keep="first"))
    sig = sig.merge(
        tw[["accession", "tissue", "TWAS.P", "twas_sig_bonf"]],
        on=["accession", "tissue"], how="left")
    return sig


gene_data = {g: collect(g) for g in GENES}

# ── Figure ───────────────────────────────────────────────────
fig = plt.figure(figsize=(DOUBLE_COL, 8.4))
gs_outer = GridSpec(1, 3, figure=fig,
                    left=0.07, right=0.97, top=0.97, bottom=0.13,
                    wspace=0.50)

# Discrete TWAS encoding — three classes:
#   - TWAS model unavailable for this tissue            → AG orange (link carries AG info only)
#   - TWAS model exists, not Bonferroni significant     → neutral grey
#   - TWAS model exists, Bonferroni significant         → TWAS blue
C_TWAS_NS   = "#9aa6b2"   # neutral slate
C_TWAS_SIG  = C_TWAS


def _link_color(row):
    """Pick a discrete link colour based on TWAS availability / significance."""
    if pd.isna(row.get("TWAS.P", np.nan)):
        return C_AG          # no TWAS model in this tissue
    if bool(row.get("twas_sig_bonf", False)):
        return C_TWAS_SIG    # TWAS Bonferroni-sig
    return C_TWAS_NS         # TWAS available, not sig


def _draw_panel(ax, data, gene, letter,
                show_twas_color=False, lw_scale=None):
    panel_label(ax, letter, x=-0.04, y=1.02)

    if len(data) == 0:
        ax.text(0.5, 0.5, f"No significant entries\nfor {gene}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=6, color="#888")
        ax.axis("off")
        return

    traits  = list(pd.Series(data["trait"]).drop_duplicates())
    tissues = sorted(pd.Series(data["tissue_label"]).drop_duplicates().tolist())
    n_t, n_ti = len(traits), len(tissues)

    # Vertical extent shared across panels so spacing is consistent.
    y_top, y_bot = 0.97, 0.03
    y_traits  = np.linspace(y_top, y_bot, n_t)  if n_t  > 1 else np.array([0.5])
    y_tissues = np.linspace(y_top, y_bot, n_ti) if n_ti > 1 else np.array([0.5])

    x_left, x_right = 0.08, 0.78

    # Sort links so that thin ones are drawn first (thick on top)
    data_sorted = data.sort_values("max_abs_lfc", ascending=True)

    for _, row in data_sorted.iterrows():
        i_t  = traits.index(row["trait"])
        i_ti = tissues.index(row["tissue_label"])
        y0, y1 = y_traits[i_t], y_tissues[i_ti]
        xs = np.linspace(x_left, x_right, 80)
        t  = (xs - x_left) / (x_right - x_left)
        s  = 0.5 - 0.5 * np.cos(np.pi * t)     # smooth S-curve
        ys = y0 + (y1 - y0) * s

        lw = max(row["max_abs_lfc"] * lw_scale, 0.3)
        col = _link_color(row) if show_twas_color else C_AG
        ax.plot(xs, ys, color=col, lw=lw, alpha=0.82,
                solid_capstyle="round", zorder=2)

    # Trait nodes & labels (left)
    for i, tr in enumerate(traits):
        ax.plot(x_left, y_traits[i], "o", color=C_AG,
                markersize=5.5, mec="white", mew=0.7, zorder=5)
        ax.text(x_left - 0.02, y_traits[i], tr,
                fontsize=6.0, ha="right", va="center",
                color="#222", fontweight="bold")

    # Tissue nodes & labels (right)
    tissue_fs = 5.0 if n_ti <= 30 else 4.3
    for i, ti in enumerate(tissues):
        ax.plot(x_right, y_tissues[i], "s", color="#3a3a3a",
                markersize=3.2, mec="white", mew=0.4, zorder=5)
        ax.text(x_right + 0.02, y_tissues[i], ti,
                fontsize=tissue_fs, ha="left", va="center", color="#333")

    # Title: gene name
    ax.text(0.5, 1.015, gene,
            transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", ha="center", va="bottom", color="#111")

    ax.set_xlim(-0.02, 1.16)
    ax.set_ylim(-0.02, 1.05)
    ax.axis("off")


# Global width scale so widths are comparable across panels
global_max = max(d["max_abs_lfc"].max() for d in gene_data.values() if len(d))
LW_SCALE = 3.5 / max(global_max, 0.02)

# Panels
ax_a = fig.add_subplot(gs_outer[0, 0])
_draw_panel(ax_a, gene_data["CDKN2B-AS1"], "CDKN2B-AS1", "a",
            show_twas_color=False, lw_scale=LW_SCALE)

ax_b = fig.add_subplot(gs_outer[0, 1])
_draw_panel(ax_b, gene_data["PTENP1"], "PTENP1", "b",
            show_twas_color=True, lw_scale=LW_SCALE)

ax_c = fig.add_subplot(gs_outer[0, 2])
_draw_panel(ax_c, gene_data["HOTAIR"], "HOTAIR", "c",
            show_twas_color=True, lw_scale=LW_SCALE)

# ── Legend strip at the bottom (two rows, clean fixed columns) ───
leg_ax = fig.add_axes([0.04, 0.005, 0.94, 0.088])
leg_ax.axis("off"); leg_ax.set_xlim(0, 1); leg_ax.set_ylim(0, 1)

# Row 1 (top): link-width reference
y_row1 = 0.72
leg_ax.text(0.02, y_row1, "Link width \u221d |max_lfc|",
            fontsize=6.2, va="center", fontweight="bold", color="#333",
            transform=leg_ax.transAxes)
for i, v in enumerate([0.02, 0.04, 0.06]):
    x0 = 0.22 + i * 0.12
    lw = max(v * LW_SCALE, 0.3)
    leg_ax.plot([x0, x0 + 0.04], [y_row1, y_row1], color="#555",
                lw=lw, solid_capstyle="round", transform=leg_ax.transAxes)
    leg_ax.text(x0 + 0.045, y_row1, f"{v:.2f}",
                fontsize=6, va="center", transform=leg_ax.transAxes)

# Row 2 (bottom): link-colour legend — 3 discrete classes at fixed columns
y_row2 = 0.22
leg_ax.text(0.02, y_row2, "Link colour",
            fontsize=6.2, va="center", fontweight="bold", color="#333",
            transform=leg_ax.transAxes)
items = [
    (C_AG,       "no TWAS model"),
    (C_TWAS_NS,  "TWAS model, not sig"),
    (C_TWAS_SIG, "TWAS model, Bonferroni sig"),
]
col_starts = [0.22, 0.45, 0.70]
for (col, label), x0 in zip(items, col_starts):
    leg_ax.plot([x0, x0 + 0.04], [y_row2, y_row2], color=col,
                lw=2.6, solid_capstyle="round",
                transform=leg_ax.transAxes)
    leg_ax.text(x0 + 0.045, y_row2, label,
                fontsize=6, va="center", transform=leg_ax.transAxes)

fig.savefig(OUT / "fig5_case_studies.png", dpi=300)
fig.savefig(OUT / "fig5_case_studies.pdf")
plt.close()
print(f"Saved fig5_case_studies  ({sum(len(d) for d in gene_data.values())} total links)")
for g, d in gene_data.items():
    print(f"  {g}: {len(d)} trait\u00d7tissue links, "
          f"|max_lfc| range [{d['max_abs_lfc'].min():.3f}, "
          f"{d['max_abs_lfc'].max():.3f}]")
