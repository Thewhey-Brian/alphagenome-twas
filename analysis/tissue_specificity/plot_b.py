#!/usr/bin/env python3
"""Panel b: Tissue coverage per SNP × gene pair (|AG| ≥ 0.01)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_common import apply_style, C_AG, C_RED, C_ORANGE
apply_style()

OUT_DIR = Path("results/insights/tissue_specificity")
pair = pd.read_parquet(OUT_DIR / "pair_tissue_counts.parquet")
nt = pair["n_tissues"]

fig, ax = plt.subplots(figsize=(4.5, 3.2))

bins = np.arange(0.5, nt.max() + 1.5, 1)
counts, edges, patches = ax.hist(nt, bins=bins, color=C_AG, alpha=0.8,
                                  edgecolor="white", lw=0.3, zorder=3)
patches[0].set_facecolor(C_RED)
patches[0].set_alpha(0.7)

ax2 = ax.twinx()
sorted_vals = np.sort(nt.values)
cumfrac = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
ax2.plot(sorted_vals, cumfrac, color=C_ORANGE, lw=1.5, zorder=4)
ax2.set_ylabel("Cumulative fraction", fontsize=7, color=C_ORANGE)
ax2.tick_params(axis="y", colors=C_ORANGE)
ax2.spines["right"].set_color(C_ORANGE)
ax2.set_ylim(0, 1.05)

ax.axvline(1.5, color=C_RED, ls=":", lw=0.6, alpha=0.6)
for t in [3, 10]:
    ax.axvline(t - 0.5, color="#888", ls=":", lw=0.5, alpha=0.5)

ax.set_xlabel("Number of tissues per SNP × gene pair")
ax.set_ylabel("Number of pairs")
ax.set_xlim(0, 50)
ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49])

n_total = len(nt)
n1 = (nt == 1).sum()
n3 = (nt >= 3).sum()
n10 = (nt >= 10).sum()
ax.text(0.55, 0.55,
        f"n = {n_total:,} pairs\n"
        f"median = {nt.median():.0f}, mean = {nt.mean():.1f}\n\n"
        f"1 tissue: {n1:,} ({100*n1/n_total:.0f}%)\n"
        f"≥ 3 tissues: {n3:,} ({100*n3/n_total:.0f}%)\n"
        f"≥ 10 tissues: {n10:,} ({100*n10/n_total:.0f}%)",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=5.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.95, lw=0.3),
        zorder=10)

ax.set_title("Tissue coverage per SNP × gene pair (|AG score| ≥ 0.01)",
             fontsize=8, fontweight="bold", loc="left")

fig.tight_layout()
fig.savefig(OUT_DIR / "panel_b.png"); fig.savefig(OUT_DIR / "panel_b.pdf")
plt.close()
print("Saved panel_b")
