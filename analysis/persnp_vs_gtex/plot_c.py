#!/usr/bin/env python3
"""Panel c: Sign concordance + Spearman ρ by |AG score| threshold, with colored threshold zones."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_common import apply_style
apply_style()

OUT_DIR = Path("results/insights/persnp_vs_gtex")
C_BLUE, C_GREEN, C_RED, C_ORANGE = "#4C72B0", "#55A868", "#C44E52", "#E8A838"

ts = pd.read_csv(OUT_DIR / "threshold_stats.csv")

fig, ax = plt.subplots(figsize=(4.5, 3.2))

# Two y-axis quantities on same plot
ax.plot(ts["threshold"], ts["sign_concordance"], "o-", color=C_GREEN,
        markersize=5, lw=1.5, label="Sign concordance", zorder=3)
ax.plot(ts["threshold"], ts["spearman_rho"], "s-", color=C_BLUE,
        markersize=5, lw=1.5, label="Spearman ρ", zorder=3)

# Chance line
ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
ax.text(0.21, 0.505, "chance", fontsize=5, color="gray", fontstyle="italic")

# Threshold zone coloring
zone_colors = ["#d4d4d4", "#d4d4d4", "#93c4e8", "#93c4e8", "#4C72B0", "#E8A838", "#C44E52", "#C44E52"]
for i, (_, row) in enumerate(ts.iterrows()):
    ax.axvspan(row["threshold"] - 0.003, row["threshold"] + 0.003,
               alpha=0.15, color=zone_colors[i], zorder=0)

# Annotate values
for _, row in ts.iterrows():
    ax.text(row["threshold"], row["sign_concordance"] + 0.025,
            f'{row["sign_concordance"]:.0%}', ha="center", fontsize=5,
            color=C_GREEN, fontweight="bold")
    if row["threshold"] >= 0.01:
        ax.text(row["threshold"], row["spearman_rho"] - 0.04,
                f'{row["spearman_rho"]:.2f}', ha="center", fontsize=5,
                color=C_BLUE, fontweight="bold")

# n count on secondary axis
ax2 = ax.twinx()
ax2.bar(ts["threshold"], ts["n"], width=0.006, color="#cccccc", alpha=0.3, zorder=0)
ax2.set_ylabel("n triplets", fontsize=6, color="#aaaaaa")
ax2.tick_params(labelsize=5, colors="#aaaaaa")
ax2.set_yscale("log")
ax2.spines[["top"]].set_visible(False)

ax.set_xlabel("|AG score| threshold")
ax.set_ylabel("Concordance / Correlation")
ax.set_ylim(0, 1.05)
ax.set_title("c", fontweight="bold", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=6, loc="center left")
ax.spines[["top"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_DIR / "panel_c.png"); plt.savefig(OUT_DIR / "panel_c.pdf"); plt.close()
print("Saved panel_c")
