#!/usr/bin/env python3
"""Combined figure: Tissue specificity of AG predictions (panels a–d).

Layout: a (left, tall) | b,c,d stacked (right)
  a. Tissue distribution with GTEx sample sizes (tall)
  b. Tissue coverage per pair histogram
  c. Tau distribution (all matched pairs)
  d. Tau by number of tissues (all matched pairs)

Usage:
    python results/insights/tissue_specificity/plot_combined.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from scipy import stats as sp_stats
from plot_common import apply_style, clean_tissue, C_AG, C_GTEX, C_RED, C_ORANGE, C_GREY
apply_style()

OUT_DIR = Path("results/insights/tissue_specificity")

tc = pd.read_csv(OUT_DIR / "tissue_counts.csv").sort_values("n_triplets", ascending=True)
pair = pd.read_parquet(OUT_DIR / "pair_tissue_counts.parquet")
df = pd.read_parquet(OUT_DIR / "tau_metrics.parquet").dropna(subset=["ag_tau", "gtex_tau"])

fig = plt.figure(figsize=(10, 10))
# Left column (a): spans full height. Right column (b,c,d): 3 rows.
gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30,
             left=0.06, right=0.96, top=0.95, bottom=0.04,
             width_ratios=[1, 1.2])

# ══════════════════════════════════════════════════════════════
# a. Tissue distribution with GTEx sizes (left, full height)
# ══════════════════════════════════════════════════════════════
gs_a = gs[:, 0].subgridspec(1, 2, wspace=0.04, width_ratios=[0.75, 1])
ax_l = fig.add_subplot(gs_a[0, 0])
ax_r = fig.add_subplot(gs_a[0, 1], sharey=ax_l)
ax_l.set_title("a", fontweight="bold", loc="left", fontsize=11)

n_tis = len(tc)
y = np.arange(n_tis)
labels_a = [clean_tissue(t) for t in tc["gtex_tissue"]]

ax_l.barh(y, tc["gtex_n"], height=0.72, color=C_GREY, edgecolor="white", lw=0.15)
ax_l.set_xlim(ax_l.get_xlim()[::-1])
ax_l.set_xlabel("GTEx RNA-seq\nsample size", fontsize=7)
ax_l.set_yticks(y)
ax_l.set_yticklabels(labels_a, fontsize=4.5)
ax_l.tick_params(axis="y", length=0, pad=2)
ax_l.spines["left"].set_visible(False)
ax_l.xaxis.set_major_locator(ticker.MaxNLocator(4, integer=True))

ax_r.barh(y, tc["n_triplets"], height=0.72, color=C_AG, edgecolor="white", lw=0.15)
ax_r.set_xlabel("Matched AG×GTEx triplets\n(|AG score| ≥ 0.01)", fontsize=7)
ax_r.xaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
plt.setp(ax_r.get_yticklabels(), visible=False)

for i, (_, row) in enumerate(tc.iterrows()):
    ax_r.text(row["n_triplets"] + 8, i, f'{int(row["n_triplets"])}',
              fontsize=3.2, va="center", color="#555")

n_total = tc["n_triplets"].sum()
ax_r.text(0.95, 0.04,
          f"{n_total:,} triplets\n{pair['rsid'].nunique():,} SNPs\n{n_tis} tissues",
          transform=ax_r.transAxes, ha="right", va="bottom", fontsize=5.5,
          family="monospace",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.9, lw=0.3))

# ══════════════════════════════════════════════════════════════
# b. Pair tissue coverage (right, top)
# ══════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0, 1])
ax.set_title("b", fontweight="bold", loc="left", fontsize=11)

nt = pair["n_tissues"]
bins_b = np.arange(0.5, nt.max() + 1.5, 1)
_, _, patches = ax.hist(nt, bins=bins_b, color=C_AG, alpha=0.8,
                         edgecolor="white", lw=0.2, zorder=3)
patches[0].set_facecolor(C_RED); patches[0].set_alpha(0.7)

ax2 = ax.twinx()
ax2.plot(np.sort(nt.values), np.arange(1, len(nt)+1)/len(nt),
         color=C_ORANGE, lw=1.2, zorder=4)
ax2.set_ylabel("Cumul. frac.", fontsize=6, color=C_ORANGE)
ax2.tick_params(axis="y", colors=C_ORANGE, labelsize=5)
ax2.spines["right"].set_color(C_ORANGE); ax2.set_ylim(0, 1.05)

ax.set_xlabel("Tissues per SNP × gene pair")
ax.set_ylabel("Number of pairs")
ax.set_xlim(0, 50); ax.set_xticks([1, 10, 20, 30, 40, 49])

n_p = len(nt); n1 = (nt == 1).sum()
n3 = (nt >= 3).sum(); n10 = (nt >= 10).sum()
ax.text(0.55, 0.60,
        f"n = {n_p:,} pairs\nmedian = {nt.median():.0f}\n\n"
        f"1 tissue: {n1:,} ({100*n1/n_p:.0f}%)\n"
        f"≥3: {n3:,} ({100*n3/n_p:.0f}%)\n"
        f"≥10: {n10:,} ({100*n10/n_p:.0f}%)",
        transform=ax.transAxes, ha="left", va="top", fontsize=5.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbb", alpha=0.95, lw=0.3),
        zorder=10)

# ══════════════════════════════════════════════════════════════
# c. Tau distribution (right, middle)
# ══════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[1, 1])
ax.set_title("c", fontweight="bold", loc="left", fontsize=11)

bins_t = np.linspace(0, 1, 41)
ax.hist(df["ag_tau"], bins=bins_t, density=True, alpha=0.55, color=C_AG,
        edgecolor="white", lw=0.15, label="AG", zorder=3)
ax.hist(df["gtex_tau"], bins=bins_t, density=True, alpha=0.55, color=C_GTEX,
        edgecolor="white", lw=0.15, label="GTEx", zorder=3)
ax.axvline(df["ag_tau"].median(), color=C_AG, ls="--", lw=1, zorder=4)
ax.axvline(df["gtex_tau"].median(), color=C_GTEX, ls="--", lw=1, zorder=4)
ax.set_xlabel("Tissue specificity (τ)"); ax.set_ylabel("Density"); ax.set_xlim(0, 1)
ax.legend(frameon=False, loc="upper left", fontsize=6)

ks_stat, _ = sp_stats.ks_2samp(df["ag_tau"], df["gtex_tau"])
ax.text(0.97, 0.95,
        f"n = {len(df):,}\nAG med τ = {df['ag_tau'].median():.3f}\n"
        f"GTEx med τ = {df['gtex_tau'].median():.3f}\nKS D = {ks_stat:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=5.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbb", alpha=0.95, lw=0.3))

# ══════════════════════════════════════════════════════════════
# d. Tau by n_tissues (right, bottom)
# ══════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[2, 1])
ax.set_title("d", fontweight="bold", loc="left", fontsize=11)

edges = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50]
labels_d = [f"{a}–{b-1}" if b-a > 1 else str(a) for a, b in zip(edges[:-1], edges[1:])]
df_c = df.copy()
df_c["tb"] = pd.cut(df_c["n_tissues"], bins=edges, right=False, labels=labels_d)
agg = df_c.groupby("tb", observed=True).agg(
    n=("ag_tau", "size"),
    ag_m=("ag_tau", "median"), ag25=("ag_tau", lambda x: x.quantile(.25)),
    ag75=("ag_tau", lambda x: x.quantile(.75)),
    gt_m=("gtex_tau", "median"), gt25=("gtex_tau", lambda x: x.quantile(.25)),
    gt75=("gtex_tau", lambda x: x.quantile(.75)),
).reset_index()
x = np.arange(len(agg))
ax.fill_between(x, agg["ag25"], agg["ag75"], alpha=0.15, color=C_AG)
ax.fill_between(x, agg["gt25"], agg["gt75"], alpha=0.15, color=C_GTEX)
ax.plot(x, agg["ag_m"], "o-", color=C_AG, ms=5, lw=1.5, label="AG predictions", zorder=3)
ax.plot(x, agg["gt_m"], "s-", color=C_GTEX, ms=4, lw=1.5, label="GTEx eQTLs", zorder=3)
ax.set_xticks(x); ax.set_xticklabels(labels_d, rotation=45, ha="right")
ax.set_xlabel("Tissues per pair"); ax.set_ylabel("τ (median ± IQR)"); ax.set_ylim(0, 0.85)
ax.legend(frameon=False, fontsize=6, loc="upper left")

for i, row in agg.iterrows():
    ax.text(i, 0.02, f"n={row['n']:,}", fontsize=3.5, ha="center", va="bottom", color="#888")

mid = len(agg) // 2
gap = agg.iloc[mid]["ag_m"] - agg.iloc[mid]["gt_m"]
ax.annotate("", xy=(mid + 0.15, agg.iloc[mid]["ag_m"]),
            xytext=(mid + 0.15, agg.iloc[mid]["gt_m"]),
            arrowprops=dict(arrowstyle="<->", color="#888", lw=0.8))
ax.text(mid + 0.3, (agg.iloc[mid]["ag_m"] + agg.iloc[mid]["gt_m"]) / 2,
        f"Δτ ≈ {gap:.2f}", fontsize=6, color="#555", va="center")

fig.suptitle("Tissue specificity of AG predictions vs GTEx eQTLs",
             fontsize=10, fontweight="bold")

fig.savefig(OUT_DIR / "tissue_specificity.png")
fig.savefig(OUT_DIR / "tissue_specificity.pdf")
plt.close()
print("Saved tissue_specificity.png/pdf")
