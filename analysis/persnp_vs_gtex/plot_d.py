#!/usr/bin/env python3
"""Panel d: |AG score| vs GTEx eQTL significance — mean, median, and p95."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from plot_common import apply_style
apply_style()

OUT_DIR = Path("results/insights/persnp_vs_gtex")
C_BLUE, C_GREEN, C_RED, C_ORANGE = "#4C72B0", "#55A868", "#C44E52", "#E8A838"

pv = pd.read_csv(OUT_DIR / "pval_vs_ag.csv").dropna(subset=["log10p_mean"])

fig, ax = plt.subplots(figsize=(4.5, 3.2))

ax.plot(pv["log10p_mean"], pv["mean_abs_ag"], "o-", color=C_BLUE,
        markersize=4, lw=1.3, label="Mean |AG|")
ax.plot(pv["log10p_mean"], pv["median_abs_ag"], "s--", color=C_GREEN,
        markersize=3.5, lw=1.0, label="Median |AG|", alpha=0.8)
ax.plot(pv["log10p_mean"], pv["p95_abs_ag"], "^:", color=C_ORANGE,
        markersize=3.5, lw=1.0, label="95th pctl |AG|", alpha=0.8)

# Trend on mean
slope, intercept, r_trend, p_trend, _ = stats.linregress(
    pv["log10p_mean"], pv["mean_abs_ag"])
xfit = np.linspace(pv["log10p_mean"].min(), pv["log10p_mean"].max(), 50)
ax.plot(xfit, slope * xfit + intercept, color=C_RED, ls="--", lw=0.8, alpha=0.5)
ax.text(0.95, 0.08, f"mean trend: r = {r_trend:.2f}",
        transform=ax.transAxes, ha="right", fontsize=6, color=C_RED, fontstyle="italic")

ax.set_xlabel("−log₁₀(GTEx eQTL p-value)")
ax.set_ylabel("|AG per-SNP score|")
ax.set_title("d", fontweight="bold", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=6)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_DIR / "panel_d.png"); plt.savefig(OUT_DIR / "panel_d.pdf"); plt.close()
print("Saved panel_d")
