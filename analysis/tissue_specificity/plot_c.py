#!/usr/bin/env python3
"""Panel c: Tau distribution — AG vs GTEx, all pairs and |AG| ≥ 0.01."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from plot_common import apply_style, C_AG, C_GTEX
apply_style()

OUT_DIR = Path("results/insights/tissue_specificity")
df = pd.read_parquet(OUT_DIR / "tau_metrics.parquet").dropna(subset=["ag_tau", "gtex_tau"])
df_filt = df[df["max_abs_ag"] >= 0.01]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True)
bins = np.linspace(0, 1, 41)

for ax, sub, title in [(ax1, df, "All matched pairs"), (ax2, df_filt, "|AG score| ≥ 0.01")]:
    ax.hist(sub["ag_tau"], bins=bins, density=True, alpha=0.55, color=C_AG,
            edgecolor="white", lw=0.2, label="AG", zorder=3)
    ax.hist(sub["gtex_tau"], bins=bins, density=True, alpha=0.55, color=C_GTEX,
            edgecolor="white", lw=0.2, label="GTEx", zorder=3)
    ax.axvline(sub["ag_tau"].median(), color=C_AG, ls="--", lw=1.2, zorder=4)
    ax.axvline(sub["gtex_tau"].median(), color=C_GTEX, ls="--", lw=1.2, zorder=4)
    ax.set_xlabel("Tissue specificity index (τ)")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    ks_stat, _ = sp_stats.ks_2samp(sub["ag_tau"], sub["gtex_tau"])
    ax.text(0.97, 0.95,
            f"n = {len(sub):,} pairs\nAG median τ = {sub['ag_tau'].median():.3f}\n"
            f"GTEx median τ = {sub['gtex_tau'].median():.3f}\nKS D = {ks_stat:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.95, lw=0.3))
    ax.set_title(title, fontsize=8, fontweight="bold", loc="left")

ax1.set_ylabel("Density")
fig.suptitle("Distribution of tissue specificity (τ): AG predictions vs GTEx eQTLs",
             fontsize=8.5, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "panel_c.png"); fig.savefig(OUT_DIR / "panel_c.pdf")
plt.close()
print("Saved panel_c")
