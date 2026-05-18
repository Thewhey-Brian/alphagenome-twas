#!/usr/bin/env python3
"""Figure 3: AG captures tissue-specific regulation through a different lens than GTEx.

Panels (1x3):
  a. Tissue specificity (tau) distributions: AG vs GTEx (FULLY INDEPENDENT filtering)
  b. Tau by number of tissues (each method on its own tissue count)
  c. AG tau vs GTEx tau hexbin scatter (paired — needs matched pairs)
"""
import sys
from pathlib import Path
import os as _os
ROOT = _os.environ.get("AG_LD_ROOT", _os.getcwd())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import (apply_style, panel_label, stat_box,
                       C_AG, C_GTEX, C_GREY, DOUBLE_COL)
apply_style()

DATA = Path(__file__).resolve().parent.parent / "tissue_specificity"
OUT  = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────
# Independent tau sets (no mutual filtering between AG and GTEx)
ag_ind = pd.read_parquet(DATA / "tau_ag_independent.parquet").dropna(subset=["ag_tau"])
gt_ind = pd.read_parquet(DATA / "tau_gtex_independent.parquet").dropna(subset=["gtex_tau"])

# Restrict GTEx-independent to credible-set SNPs (same ~42,936 SNPs used throughout)
matched = pd.read_parquet(
    Path(ROOT) / "figures/_archive_v1/fig1_validation/ag_gtex_matched.parquet"
)
cs_keys = set(zip(matched["chr"], matched["pos"]))
gt_ind = gt_ind[[(c, p) in cs_keys for c, p in zip(gt_ind["chr"], gt_ind["pos"])]].copy()

# Paired set (for panel c scatter only)
paired = pd.read_parquet(DATA / "tau_metrics.parquet").dropna(subset=["ag_tau", "gtex_tau"])

# ── Layout ───────────────────────────────────────────────────
fig = plt.figure(figsize=(DOUBLE_COL, 2.6))
gs  = GridSpec(1, 3, figure=fig,
               left=0.08, right=0.97, top=0.82, bottom=0.18,
               wspace=0.42)

# ══ a. Tau distributions (independent) ═══════════════════════
ax = fig.add_subplot(gs[0, 0])
panel_label(ax, "a")

bins = np.linspace(0, 1, 41)
ax.hist(ag_ind["ag_tau"],   bins=bins, density=True, alpha=0.6, color=C_AG,
        edgecolor="white", lw=0.2, label=f"AG (n = {len(ag_ind):,})", zorder=3)
ax.hist(gt_ind["gtex_tau"], bins=bins, density=True, alpha=0.6, color=C_GTEX,
        edgecolor="white", lw=0.2, label=f"GTEx eQTLs (n = {len(gt_ind):,})", zorder=3)

med_ag   = ag_ind["ag_tau"].median()
med_gtex = gt_ind["gtex_tau"].median()
ax.axvline(med_ag,   color=C_AG,   ls="--", lw=1.1, zorder=4)
ax.axvline(med_gtex, color=C_GTEX, ls="--", lw=1.1, zorder=4)

ax.annotate(f"AG\nmedian = {med_ag:.2f}",
            xy=(med_ag, 1.0), xycoords=("data", "axes fraction"),
            xytext=(med_ag - 0.14, 1.12), textcoords=("data", "axes fraction"),
            fontsize=5, color=C_AG, fontweight="bold", ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-|>", color=C_AG, lw=0.7, mutation_scale=5))
ax.annotate(f"GTEx\nmedian = {med_gtex:.2f}",
            xy=(med_gtex, 1.0), xycoords=("data", "axes fraction"),
            xytext=(med_gtex + 0.14, 1.12), textcoords=("data", "axes fraction"),
            fontsize=5, color=C_GTEX, fontweight="bold", ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-|>", color=C_GTEX, lw=0.7, mutation_scale=5))

ks_stat, ks_p = sp_stats.ks_2samp(ag_ind["ag_tau"], gt_ind["gtex_tau"])
p_str = "p < 1e-10" if ks_p < 1e-10 else f"p = {ks_p:.1e}"
stat_box(ax, f"KS D = {ks_stat:.3f}\n{p_str}", x=0.97, y=0.68, ha="right")

ax.set_xlabel("Tissue specificity (\u03c4)", labelpad=3)
ax.set_ylabel("Density", labelpad=3)
ax.set_xlim(0, 1)
leg = ax.legend(frameon=False, fontsize=5.5, loc="upper right",
                bbox_to_anchor=(1.0, 1.0), alignment="right",
                markerfirst=False, handlelength=1.2, handletextpad=0.4)

# ══ b. Tau by n_tissues (each method's own tissue count) ═════
ax = fig.add_subplot(gs[0, 1])
panel_label(ax, "b")

edges    = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50]
labels_b = [f"{a}\u2013{b-1}" if b - a > 1 else str(a)
            for a, b in zip(edges[:-1], edges[1:])]

def bin_medians(df, val_col):
    d = df.copy()
    d["tb"] = pd.cut(d["n_tissues"], bins=edges, right=False, labels=labels_b)
    return d.groupby("tb", observed=True).agg(
        n=(val_col, "size"), m=(val_col, "median")
    ).reset_index()

ag_b = bin_medians(ag_ind, "ag_tau")
gt_b = bin_medians(gt_ind, "gtex_tau")
x = np.arange(len(labels_b))
ax.plot(x, ag_b["m"], "o-", color=C_AG,   ms=4, lw=1.3, label="AG",   zorder=3)
ax.plot(x, gt_b["m"], "s-", color=C_GTEX, ms=3.5, lw=1.3, label="GTEx", zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(labels_b, rotation=45, ha="right", fontsize=5)
ax.set_xlabel("Tissues per pair", labelpad=3)
ax.set_ylabel("Median \u03c4", labelpad=3)
ymin = min(ag_b["m"].min(), gt_b["m"].min())
ymax = max(ag_b["m"].max(), gt_b["m"].max())
pad  = (ymax - ymin) * 0.15
ax.set_ylim(max(0, ymin - pad), ymax + pad * 1.5)
ax.legend(frameon=False, fontsize=5.5, loc="upper left")

# sample-size annotations are omitted to keep the panel clean; see supplementary table

# ══ c. Tau scatter (paired) ══════════════════════════════════
ax = fig.add_subplot(gs[0, 2])
panel_label(ax, "c", x=-0.12, y=1.15)

hb = ax.hexbin(paired["ag_tau"], paired["gtex_tau"], gridsize=38, cmap="OrRd",
               mincnt=1, linewidths=0, alpha=0.9, zorder=2)
ax.plot([0, 1], [0, 1], ":", color=C_GREY, lw=0.6)

rho, _ = sp_stats.spearmanr(paired["ag_tau"], paired["gtex_tau"])
r,   _ = sp_stats.pearsonr(paired["ag_tau"],  paired["gtex_tau"])
stat_box(ax, f"n = {len(paired):,}\nSpearman \u03c1 = {rho:.3f}\nPearson r = {r:.3f}",
         x=0.04, y=0.97, ha="left")

ax.set_xlabel("AG \u03c4", labelpad=3)
ax.set_ylabel("GTEx \u03c4", labelpad=3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_aspect("equal")

cb = plt.colorbar(hb, ax=ax, shrink=0.75, pad=0.03)
cb.set_label("Count", fontsize=5.5)
cb.ax.tick_params(labelsize=5)

# ── Save ─────────────────────────────────────────────────────
fig.savefig(OUT / "fig3_tissue_specificity.png", dpi=300)
fig.savefig(OUT / "fig3_tissue_specificity.pdf")
plt.close()
print("Saved fig3_tissue_specificity")
print(f"  AG:   n={len(ag_ind):,}  mean tau={ag_ind['ag_tau'].mean():.3f}  median={med_ag:.3f}")
print(f"  GTEx: n={len(gt_ind):,}  mean tau={gt_ind['gtex_tau'].mean():.3f}  median={med_gtex:.3f}")
print(f"  KS D={ks_stat:.3f}, p={ks_p:.2e}")
print(f"  Paired scatter: n={len(paired):,}  rho={rho:.3f}  r={r:.3f}")
