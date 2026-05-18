#!/usr/bin/env python3
"""Per-SNP AG score validation against GTEx eQTLs — combined figure.

Layout (2×2):
  a. Scatter: AG raw_score vs GTEx beta, color-coded by |AG| threshold
  b. Venn diagram: AG triplets ∩ GTEx eQTL triplets
  c. Sign concordance + Spearman ρ by |AG| threshold
  d. |AG score| vs GTEx eQTL significance (mean, median, p95)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from scipy import stats as sp_stats
from plot_common import apply_style
apply_style()

OUT_DIR = Path("results/insights/persnp_vs_gtex")

C_BLUE, C_GREEN, C_RED, C_ORANGE = "#4C72B0", "#55A868", "#C44E52", "#E8A838"

ts = pd.read_csv(OUT_DIR / "threshold_stats.csv")
pv = pd.read_csv(OUT_DIR / "pval_vs_ag.csv").dropna(subset=["log10p_mean"])
ov = pd.read_csv(OUT_DIR / "overlap_stats.csv", index_col=0); ov.columns = ["value"]
samp = pd.read_parquet(OUT_DIR / "scatter_sample.parquet")

fig = plt.figure(figsize=(7.2, 6.8))
gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38,
             left=0.09, right=0.97, top=0.96, bottom=0.07)

# ══ a. Scatter color-coded by |AG| threshold ══
ax_a = fig.add_subplot(gs[0, 0])
thresholds = [0, 0.005, 0.01, 0.05, 0.1]
colors_a = ["#d4d4d4", "#93c4e8", "#4C72B0", "#E8A838", "#C44E52"]
labels_a = ["|AG|<0.005", "0.005–0.01", "0.01–0.05", "0.05–0.1", "|AG|≥0.1"]
for i in range(len(thresholds)):
    lo = thresholds[i]
    hi = thresholds[i + 1] if i + 1 < len(thresholds) else np.inf
    mask = (samp["abs_ag"] >= lo) & (samp["abs_ag"] < hi)
    sub = samp[mask]
    alpha = 0.15 if i == 0 else 0.4 if i == 1 else 0.6
    size = 0.3 if i < 2 else 1.5 if i < 4 else 4
    ax_a.scatter(sub["raw_score"], sub["beta"], s=size, c=colors_a[i], alpha=alpha,
                 edgecolors="none", rasterized=True, label=f"{labels_a[i]} (n={len(sub):,})")
ax_a.axhline(0, color="gray", ls=":", lw=0.5); ax_a.axvline(0, color="gray", ls=":", lw=0.5)
ax_a.set_xlim(-0.15, 0.15); ax_a.set_ylim(-1.5, 1.5)
ax_a.set_xlabel("AG per-SNP raw score (LFC)"); ax_a.set_ylabel("GTEx eQTL β")
ax_a.set_title("a", fontweight="bold", loc="left", fontsize=10)
ax_a.spines[["top", "right"]].set_visible(False)
r_val = np.corrcoef(samp["raw_score"], samp["beta"])[0, 1]
rho_val, _ = sp_stats.spearmanr(samp["raw_score"], samp["beta"])
ax_a.text(0.04, 0.96, f"n={len(samp):,}\nr={r_val:.3f}, ρ={rho_val:.3f}",
          transform=ax_a.transAxes, va="top", fontsize=5.5, family="monospace",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9, lw=0.4))
ax_a.legend(frameon=False, fontsize=4.5, loc="lower right", markerscale=3)

# ══ b. Venn diagram ══
ax_b = fig.add_subplot(gs[0, 1])
c_ag = Circle((-0.3, 0), 0.55, fc=C_BLUE, alpha=0.2, ec=C_BLUE, lw=1.2)
c_gtex = Circle((0.3, 0), 0.55, fc=C_GREEN, alpha=0.2, ec=C_GREEN, lw=1.2)
ax_b.add_patch(c_ag); ax_b.add_patch(c_gtex)
ag_only = int(ov.loc["ag_snps", "value"]) - int(ov.loc["matched_snps", "value"])
gtex_only = int(ov.loc["gtex_snps", "value"]) - int(ov.loc["matched_snps", "value"])
matched = int(ov.loc["matched_snps", "value"])
ax_b.text(-0.65, 0.12, f'{ag_only:,}\nSNPs', ha="center", fontsize=7, fontweight="bold", color=C_BLUE)
ax_b.text(-0.65, -0.18, f'{int(ov.loc["ag_triplets","value"]):,}\ntriplets', ha="center", fontsize=5, color=C_BLUE)
ax_b.text(0, 0.12, f'{matched:,}\nSNPs', ha="center", fontsize=7, fontweight="bold", color="#333")
ax_b.text(0, -0.18, f'{int(ov.loc["matched_triplets","value"]):,}\ntriplets', ha="center", fontsize=5, color="#333")
ax_b.text(0.65, 0.12, f'{gtex_only:,}\nSNPs', ha="center", fontsize=6, fontweight="bold", color=C_GREEN)
ax_b.text(0.65, -0.18, f'{int(ov.loc["gtex_triplets","value"]):,}\ntriplets', ha="center", fontsize=4.5, color=C_GREEN)
ax_b.text(-0.5, 0.52, "AG per-SNP", ha="center", fontsize=6, color=C_BLUE, fontstyle="italic")
ax_b.text(0.5, 0.52, "GTEx sig eQTL", ha="center", fontsize=6, color=C_GREEN, fontstyle="italic")
ax_b.text(0, -0.62, f"SNP overlap: {matched:,}/{int(ov.loc['ag_snps','value']):,} ({matched/int(ov.loc['ag_snps','value']):.1%})",
          ha="center", fontsize=6, color="#555")
ax_b.set_xlim(-1.15, 1.15); ax_b.set_ylim(-0.78, 0.68)
ax_b.set_aspect("equal"); ax_b.axis("off")
ax_b.set_title("b", fontweight="bold", loc="left", fontsize=10)

# ══ c. Sign concordance + Spearman ρ by threshold ══
ax_c = fig.add_subplot(gs[1, 0])
ax_c.plot(ts["threshold"], ts["sign_concordance"], "o-", color=C_GREEN,
          markersize=4.5, lw=1.5, label="Sign concordance", zorder=3)
ax_c.plot(ts["threshold"], ts["spearman_rho"], "s-", color=C_BLUE,
          markersize=4.5, lw=1.5, label="Spearman ρ", zorder=3)
ax_c.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
zone_colors = ["#d4d4d4", "#d4d4d4", "#93c4e8", "#93c4e8", "#4C72B0", "#E8A838", "#C44E52", "#C44E52"]
for i, (_, row) in enumerate(ts.iterrows()):
    ax_c.axvspan(row["threshold"] - 0.003, row["threshold"] + 0.003,
                 alpha=0.12, color=zone_colors[i], zorder=0)
for _, row in ts.iterrows():
    ax_c.text(row["threshold"], row["sign_concordance"] + 0.025,
              f'{row["sign_concordance"]:.0%}', ha="center", fontsize=4.5, color=C_GREEN, fontweight="bold")
    if row["threshold"] >= 0.01:
        ax_c.text(row["threshold"], row["spearman_rho"] - 0.04,
                  f'{row["spearman_rho"]:.2f}', ha="center", fontsize=4.5, color=C_BLUE, fontweight="bold")
ax_c2 = ax_c.twinx()
ax_c2.bar(ts["threshold"], ts["n"], width=0.005, color="#cccccc", alpha=0.3, zorder=0)
ax_c2.set_ylabel("n triplets", fontsize=5, color="#aaa"); ax_c2.tick_params(labelsize=4.5, colors="#aaa")
ax_c2.set_yscale("log"); ax_c2.spines[["top"]].set_visible(False)
ax_c.set_xlabel("|AG score| threshold"); ax_c.set_ylabel("Concordance / Correlation")
ax_c.set_ylim(0, 1.05); ax_c.set_title("c", fontweight="bold", loc="left", fontsize=10)
ax_c.legend(frameon=False, fontsize=5.5, loc="center left"); ax_c.spines[["top"]].set_visible(False)

# ══ d. |AG| vs GTEx significance ══
ax_d = fig.add_subplot(gs[1, 1])
ax_d.plot(pv["log10p_mean"], pv["mean_abs_ag"], "o-", color=C_BLUE,
          markersize=3.5, lw=1.3, label="Mean |AG|")
ax_d.plot(pv["log10p_mean"], pv["median_abs_ag"], "s--", color=C_GREEN,
          markersize=3, lw=1.0, label="Median |AG|", alpha=0.8)
ax_d.plot(pv["log10p_mean"], pv["p95_abs_ag"], "^:", color=C_ORANGE,
          markersize=3, lw=1.0, label="95th pctl |AG|", alpha=0.8)
slope, intercept, r_trend, p_trend, _ = sp_stats.linregress(pv["log10p_mean"], pv["mean_abs_ag"])
xfit = np.linspace(pv["log10p_mean"].min(), pv["log10p_mean"].max(), 50)
ax_d.plot(xfit, slope * xfit + intercept, color=C_RED, ls="--", lw=0.8, alpha=0.5)
ax_d.text(0.95, 0.08, f"mean trend: r = {r_trend:.2f}", transform=ax_d.transAxes,
          ha="right", fontsize=5.5, color=C_RED, fontstyle="italic")
ax_d.set_xlabel("−log₁₀(GTEx eQTL p-value)"); ax_d.set_ylabel("|AG per-SNP score|")
ax_d.set_title("d", fontweight="bold", loc="left", fontsize=10)
ax_d.legend(frameon=False, fontsize=5.5); ax_d.spines[["top", "right"]].set_visible(False)

plt.savefig(OUT_DIR / "persnp_vs_gtex.png"); plt.savefig(OUT_DIR / "persnp_vs_gtex.pdf"); plt.close()
print("Saved combined figure")
