#!/usr/bin/env python3
"""Panel d: Tau by number of matched tissues — all pairs and |AG| ≥ 0.01 side by side."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_common import apply_style, C_AG, C_GTEX
apply_style()

OUT_DIR = Path("results/insights/tissue_specificity")
df = pd.read_parquet(OUT_DIR / "tau_metrics.parquet").dropna(subset=["ag_tau", "gtex_tau"])
df_filt = df[df["max_abs_ag"] >= 0.01]

edges = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50]
labels = [f"{a}–{b-1}" if b - a > 1 else str(a) for a, b in zip(edges[:-1], edges[1:])]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.2), sharey=True)

for ax, sub, title in [(ax1, df, "All matched pairs"), (ax2, df_filt, "|AG score| ≥ 0.01")]:
    sub = sub.copy()
    sub["tissue_bin"] = pd.cut(sub["n_tissues"], bins=edges, right=False, labels=labels)
    agg = sub.groupby("tissue_bin", observed=True).agg(
        n=("ag_tau", "size"),
        ag_med=("ag_tau", "median"), ag_q25=("ag_tau", lambda x: x.quantile(0.25)),
        ag_q75=("ag_tau", lambda x: x.quantile(0.75)),
        gtex_med=("gtex_tau", "median"), gtex_q25=("gtex_tau", lambda x: x.quantile(0.25)),
        gtex_q75=("gtex_tau", lambda x: x.quantile(0.75)),
    ).reset_index()

    x = np.arange(len(agg))
    ax.fill_between(x, agg["ag_q25"], agg["ag_q75"], alpha=0.15, color=C_AG, zorder=1)
    ax.fill_between(x, agg["gtex_q25"], agg["gtex_q75"], alpha=0.15, color=C_GTEX, zorder=1)
    ax.plot(x, agg["ag_med"], "o-", color=C_AG, ms=4, lw=1.2, label="AG", zorder=3)
    ax.plot(x, agg["gtex_med"], "s-", color=C_GTEX, ms=3, lw=1.2, label="GTEx", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(agg["tissue_bin"], rotation=45, ha="right", fontsize=5)
    ax.set_xlabel("Tissues per pair")
    ax.set_ylim(0, 0.85)
    ax.legend(frameon=False, fontsize=5.5, loc="upper left")
    ax.set_title(title, fontsize=8, fontweight="bold", loc="left")

    for i, row in agg.iterrows():
        ax.text(i, 0.02, f"n={row['n']:,}", fontsize=3.5, ha="center", va="bottom", color="#888")

ax1.set_ylabel("Tissue specificity τ (median ± IQR)")

fig.suptitle("Tissue specificity (τ) by number of matched tissues",
             fontsize=8.5, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "panel_d.png"); fig.savefig(OUT_DIR / "panel_d.pdf")
plt.close()
print("Saved panel_d")
