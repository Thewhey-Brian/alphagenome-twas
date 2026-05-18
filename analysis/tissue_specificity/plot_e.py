#!/usr/bin/env python3
"""Panel e: AG tau vs GTEx tau scatter — all pairs and |AG| ≥ 0.01."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from plot_common import apply_style, C_AG
apply_style()

OUT_DIR = Path("results/insights/tissue_specificity")
df = pd.read_parquet(OUT_DIR / "tau_metrics.parquet").dropna(subset=["ag_tau", "gtex_tau"])
df_filt = df[df["max_abs_ag"] >= 0.01]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.2))

for ax, sub, title in [(ax1, df, "All matched pairs"), (ax2, df_filt, "|AG score| ≥ 0.01")]:
    ax.hexbin(sub["ag_tau"], sub["gtex_tau"], gridsize=45, cmap="Blues",
              mincnt=1, linewidths=0, alpha=0.85, zorder=2)
    ax.plot([0, 1], [0, 1], ":", color="#888", lw=0.6, alpha=0.5)
    rho, _ = sp_stats.spearmanr(sub["ag_tau"], sub["gtex_tau"])
    r, _ = sp_stats.pearsonr(sub["ag_tau"], sub["gtex_tau"])
    ax.text(0.04, 0.96, f"n = {len(sub):,}\nSpearman ρ = {rho:.3f}\nPearson r = {r:.3f}",
            transform=ax.transAxes, va="top", fontsize=6, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.95, lw=0.3))
    ax.set_xlabel("AG τ (tissue specificity)")
    ax.set_ylabel("GTEx τ (tissue specificity)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=8, fontweight="bold", loc="left")

fig.suptitle("AG vs GTEx tissue specificity (τ index)",
             fontsize=8.5, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "panel_e.png"); fig.savefig(OUT_DIR / "panel_e.pdf")
plt.close()
print("Saved panel_e")
