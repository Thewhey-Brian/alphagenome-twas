#!/usr/bin/env python3
"""Panel a: Tissue distribution of matched pairs alongside GTEx sample sizes."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from plot_common import apply_style, clean_tissue, C_AG, C_GREY
apply_style()

OUT_DIR = Path("results/insights/tissue_specificity")

tc = pd.read_csv(OUT_DIR / "tissue_counts.csv").sort_values("n_triplets", ascending=True)
n_tissues = len(tc)
y = np.arange(n_tissues)
labels = [clean_tissue(t) for t in tc["gtex_tissue"]]

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(5.5, 7), sharey=True,
    gridspec_kw={"width_ratios": [1, 1], "wspace": 0.05})

ax_l.barh(y, tc["gtex_n"], height=0.72, color=C_GREY, edgecolor="white", lw=0.2)
ax_l.set_xlim(ax_l.get_xlim()[::-1])
ax_l.set_xlabel("GTEx RNA-seq\nsample size", fontsize=7)
ax_l.set_yticks(y)
ax_l.set_yticklabels(labels, fontsize=4.8)
ax_l.tick_params(axis="y", length=0, pad=2)
ax_l.spines["left"].set_visible(False)
ax_l.xaxis.set_major_locator(ticker.MaxNLocator(4, integer=True))

ax_r.barh(y, tc["n_triplets"], height=0.72, color=C_AG, edgecolor="white", lw=0.2)
ax_r.set_xlabel("Matched AG×GTEx triplets\n(|AG score| ≥ 0.01)", fontsize=7)
ax_r.xaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
for i, (_, row) in enumerate(tc.iterrows()):
    ax_r.text(row["n_triplets"] + 8, i, f'{int(row["n_triplets"])}',
              fontsize=3.8, va="center", color="#555")

n_total = tc["n_triplets"].sum()
pt = pd.read_parquet(OUT_DIR / "pair_tissue_counts.parquet")
ax_r.text(0.95, 0.04,
          f"{n_total:,} triplets\n{pt['rsid'].nunique():,} SNPs\n{n_tissues} tissues",
          transform=ax_r.transAxes, ha="right", va="bottom", fontsize=5.5,
          family="monospace",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.9, lw=0.3))

fig.suptitle("Tissue representation in high-confidence AG×GTEx matched pairs",
             fontsize=8.5, fontweight="bold", y=0.995)

fig.tight_layout()
fig.savefig(OUT_DIR / "panel_a.png"); fig.savefig(OUT_DIR / "panel_a.pdf")
plt.close()
print("Saved panel_a")
