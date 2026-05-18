#!/usr/bin/env python3
"""Panel a: Density scatter of AG raw_score vs GTEx beta, color-coded by AG score threshold."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from plot_common import apply_style
apply_style()

OUT_DIR = Path("results/insights/persnp_vs_gtex")
samp = pd.read_parquet(OUT_DIR / "scatter_sample.parquet")

fig, ax = plt.subplots(figsize=(4.5, 3.5))

# Bin by AG score threshold for color coding
thresholds = [0, 0.005, 0.01, 0.05, 0.1]
colors = ["#d4d4d4", "#93c4e8", "#4C72B0", "#E8A838", "#C44E52"]
labels = ["|AG|<0.005", "0.005–0.01", "0.01–0.05", "0.05–0.1", "|AG|≥0.1"]

# Plot from low to high so high-score points are on top
for i in range(len(thresholds)):
    lo = thresholds[i]
    hi = thresholds[i + 1] if i + 1 < len(thresholds) else np.inf
    mask = (samp["abs_ag"] >= lo) & (samp["abs_ag"] < hi)
    sub = samp[mask]
    alpha = 0.15 if i == 0 else 0.4 if i == 1 else 0.6
    size = 0.3 if i < 2 else 1.5 if i < 4 else 4
    ax.scatter(sub["raw_score"], sub["beta"], s=size, c=colors[i], alpha=alpha,
               edgecolors="none", rasterized=True, label=f"{labels[i]} (n={len(sub):,})")

ax.axhline(0, color="gray", ls=":", lw=0.5)
ax.axvline(0, color="gray", ls=":", lw=0.5)

lim_x, lim_y = 0.15, 1.5
ax.set_xlim(-lim_x, lim_x)
ax.set_ylim(-lim_y, lim_y)
ax.set_xlabel("AG per-SNP raw score (LFC)")
ax.set_ylabel("GTEx eQTL β")
ax.set_title("a", fontweight="bold", loc="left", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

# Stats
from scipy import stats as sp_stats
r_val = np.corrcoef(samp["raw_score"], samp["beta"])[0, 1]
rho_val, _ = sp_stats.spearmanr(samp["raw_score"], samp["beta"])
ax.text(0.04, 0.96, f"n = {len(samp):,}\nr = {r_val:.3f}\nρ = {rho_val:.3f}",
        transform=ax.transAxes, va="top", fontsize=6, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9, lw=0.4))

ax.legend(frameon=False, fontsize=5, loc="lower right", markerscale=3)

plt.tight_layout()
plt.savefig(OUT_DIR / "panel_a.png"); plt.savefig(OUT_DIR / "panel_a.pdf"); plt.close()
print("Saved panel_a")
