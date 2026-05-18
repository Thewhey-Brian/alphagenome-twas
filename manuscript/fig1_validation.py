#!/usr/bin/env python3
"""Figure 1: AG predictions validated against GTEx eQTLs with self-calibrated threshold.

Panels (2×2, top-left span):
  a. Density scatter: AG LFC vs GTEx beta, color by confidence (wide, top row)
  b. Sign concordance + Spearman rho by |AG| threshold (key result)
  c. |AG score| vs GTEx eQTL significance
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import (apply_style, panel_label, stat_box,
                       C_AG, C_GTEX, C_ORANGE, C_RED, C_GREY, C_BLUE,
                       DOUBLE_COL)
apply_style()

DATA = Path(__file__).resolve().parent.parent / "persnp_vs_gtex"
OUT  = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────
full = pd.read_parquet(DATA / "scatter_full.parquet")
ts   = pd.read_csv(DATA / "threshold_stats.csv")
pv   = pd.read_csv(DATA / "pval_vs_ag.csv").dropna(subset=["log10p_mean"])

# ── Figure ───────────────────────────────────────────────────
fig = plt.figure(figsize=(DOUBLE_COL, 5.8))
gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38,
             left=0.08, right=0.97, top=0.96, bottom=0.07)

# ══ a. Scatter (full dataset) ════════════════════════════════
ax = fig.add_subplot(gs[0, :])
panel_label(ax, "a")

thresholds = [0, 0.005, 0.01, 0.05, 0.1]
colors_a   = [C_GREY, C_BLUE, C_AG, C_ORANGE, C_RED]
labels_a   = ["|LFC|<0.005", "0.005-0.01", "0.01-0.05", "0.05-0.1", "|LFC|>=0.1"]
alphas     = [0.15, 0.25, 0.45, 0.55, 0.7]
sizes      = [0.5, 1.0, 1.5, 3, 6]

for i in range(len(thresholds)):
    lo = thresholds[i]
    hi = thresholds[i + 1] if i + 1 < len(thresholds) else np.inf
    mask = (full["abs_ag"] >= lo) & (full["abs_ag"] < hi)
    sub = full[mask]
    ax.scatter(sub["raw_score"], sub["beta"], s=sizes[i], c=colors_a[i],
               alpha=alphas[i], edgecolors="none", rasterized=True,
               label=f"{labels_a[i]} (n={len(sub):,})")

ax.axhline(0, color=C_GREY, ls=":", lw=0.4); ax.axvline(0, color=C_GREY, ls=":", lw=0.4)
ax.set_xlim(-0.15, 0.15); ax.set_ylim(-1.5, 1.5)
ax.set_xlabel("AG independent score (LFC)"); ax.set_ylabel("GTEx eQTL effect size (beta)")

r_val = np.corrcoef(full["raw_score"], full["beta"])[0, 1]
rho_val, _ = sp_stats.spearmanr(full["raw_score"], full["beta"])
stat_box(ax, f"n = {len(full):,}\nr = {r_val:.3f}\n\u03c1 = {rho_val:.3f}", x=0.04, y=0.96, ha="left")
leg = ax.legend(frameon=False, fontsize=4.5, loc="lower right", markerscale=4,
                handletextpad=0.3, borderpad=0.3)
# Ensure all legend markers are fully opaque and visible
for lh in leg.legend_handles:
    lh.set_alpha(1.0)

# ══ b. Concordance by threshold (KEY PANEL) ══════════════════
ax = fig.add_subplot(gs[1, 0])
panel_label(ax, "b")

ax.plot(ts["threshold"], ts["sign_concordance"], "o-", color=C_GTEX,
        markersize=4, lw=1.5, label="Sign concordance", zorder=3)
ax.plot(ts["threshold"], ts["spearman_rho"], "s-", color=C_AG,
        markersize=4, lw=1.5, label="Spearman \u03c1", zorder=3)
ax.axhline(0.5, color=C_GREY, ls="--", lw=0.6, alpha=0.5)

# Highlight the 0.01 threshold
ax.axvline(0.01, color=C_RED, ls="--", lw=0.8, alpha=0.6, zorder=1)
ax.text(0.012, 0.98, "|LFC| = 0.01\nthreshold", fontsize=5, color=C_RED,
        fontstyle="italic", va="top")

# Annotate key values
for _, row in ts.iterrows():
    if row["threshold"] >= 0.01:
        ax.text(row["threshold"], row["sign_concordance"] + 0.03,
                f'{row["sign_concordance"]:.0%}', ha="center", fontsize=5,
                color=C_GTEX, fontweight="bold")
        ax.text(row["threshold"], row["spearman_rho"] - 0.045,
                f'{row["spearman_rho"]:.2f}', ha="center", fontsize=5,
                color=C_AG, fontweight="bold")

# n count bars (background)
ax2 = ax.twinx()
ax2.bar(ts["threshold"], ts["n"], width=0.005, color=C_GREY, alpha=0.15, zorder=0)
ax2.set_ylabel("Number of triplets", fontsize=5.5, color="#aaa")
ax2.tick_params(labelsize=5, colors="#aaa")
ax2.set_yscale("log")
ax2.spines["top"].set_visible(False)

ax.set_xlabel("|AG score| threshold")
ax.set_ylabel("Concordance / Correlation")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=6, loc="lower right")

# ══ c. AG score vs GTEx significance ═════════════════════════
ax = fig.add_subplot(gs[1, 1])
panel_label(ax, "c")

ax.plot(pv["log10p_mean"], pv["p95_abs_ag"], "^-", color=C_RED,
        markersize=3, lw=1.0, label="95th percentile", alpha=0.85, zorder=3)
ax.plot(pv["log10p_mean"], pv["mean_abs_ag"], "o-", color=C_AG,
        markersize=3.5, lw=1.2, label="Mean", zorder=4)
ax.plot(pv["log10p_mean"], pv["median_abs_ag"], "s-", color=C_GTEX,
        markersize=3, lw=1.0, label="Median", alpha=0.85, zorder=3)

# Shaded region between median and 95th percentile
ax.fill_between(pv["log10p_mean"], pv["median_abs_ag"], pv["p95_abs_ag"],
                color=C_AG, alpha=0.08)

# Trend line on mean
slope, intercept, r_trend, _, _ = sp_stats.linregress(pv["log10p_mean"], pv["mean_abs_ag"])
xfit = np.linspace(pv["log10p_mean"].min(), pv["log10p_mean"].max(), 50)
ax.plot(xfit, slope * xfit + intercept, color=C_GREY, ls="--", lw=0.7, alpha=0.6)

ax.set_xlabel(u"\u2212log\u2081\u2080(GTEx eQTL p-value)")
ax.set_ylabel("|AG score| magnitude")
ax.legend(frameon=False, fontsize=5.5, loc="upper left", title="AG score summary",
          title_fontsize=5.5)
stat_box(ax, f"r = {r_trend:.2f} (mean trend)", x=0.97, y=0.25)

# ── Save ─────────────────────────────────────────────────────
fig.savefig(OUT / "fig1_validation.png")
fig.savefig(OUT / "fig1_validation.pdf")
plt.close()
print("Saved fig1_validation")
