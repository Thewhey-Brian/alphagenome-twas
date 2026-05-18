#!/usr/bin/env python3
"""Figure 2: Joint haplotype scoring reveals broader effects but converges for strong signals.

Panels (2x2):
  a. Spearman rho & sign agreement vs |score| threshold (joint vs independent)
  b. Genes per locus distribution
  c. Tissues per locus distribution
  d. Tissue specificity (tau) distributions
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import (apply_style, panel_label, stat_box,
                       C_AG, C_RED, C_BLUE, C_GREY, DOUBLE_COL)
apply_style()

DATA   = Path(__file__).resolve().parent.parent / "persnp_vs_joint"
THRESH = 0.01   # |LFC| significance threshold — consistent across all panels
OUT    = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

C_INDEP = C_RED       # orange/vermillion — AG uses independent (per-SNP) scoring
C_JOINT  = C_BLUE     # blue — joint (multi-variant) scoring

# ── Load & filter ────────────────────────────────────────────
pleio  = pd.read_parquet(DATA / "pleiotropy_comparison.parquet")
tau_df = pd.read_parquet(DATA / "tissue_specificity.parquet")
thresh = pd.read_csv(DATA / "correlation_by_threshold.csv")

joint_p  = pleio[pleio["method"] == "joint"]
persnp_p = pleio[pleio["method"] == "persnp_max"]

# Fig 2d: separate thresholds — each method filtered on its own max |score|.
persnp_tau_vals = tau_df.loc[
    tau_df["persnp_tau"].notna() & (tau_df["max_abs_persnp"] > THRESH),
    "persnp_tau"
]
joint_tau_vals = tau_df.loc[
    tau_df["joint_tau"].notna() & (tau_df["max_abs_joint"] > THRESH),
    "joint_tau"
]

# ── Layout ───────────────────────────────────────────────────
fig = plt.figure(figsize=(DOUBLE_COL, 4.6))
gs  = GridSpec(2, 2, figure=fig,
               left=0.09, right=0.95, top=0.90, bottom=0.11,
               hspace=0.62, wspace=0.38)

# ══ a. Convergence by threshold ══════════════════════════════
ax  = fig.add_subplot(gs[0, 0])
ax2 = ax.twinx()
panel_label(ax, "a")

ax.plot(thresh["threshold"], thresh["spearman_rho"],
        "o-", color=C_JOINT, ms=4, lw=1.4, label="Spearman \u03c1", zorder=3)
ax2.plot(thresh["threshold"], thresh["sign_agreement"] * 100,
         "s--", color=C_INDEP, ms=4, lw=1.2, label="Sign agreement", zorder=3)

# Threshold reference line
ax.axvline(THRESH, color=C_RED, ls="--", lw=0.8, alpha=0.7, zorder=1)
ax.text(THRESH * 1.18, 0.44, f"|LFC| = {THRESH}", fontsize=5.5,
        color=C_RED, fontstyle="italic", va="bottom")

# Axes formatting
ax.set_xscale("log")
ax.set_xlabel("|Score| threshold (LFC)", labelpad=3)
ax.set_ylabel("Spearman \u03c1", color=C_JOINT, labelpad=3)
ax2.set_ylabel("Sign agreement (%)", color=C_INDEP, labelpad=3)
ax.set_ylim(0.38, 0.88)
ax2.set_ylim(58, 102)
ax2.axhline(50, color=C_GREY, ls=":", lw=0.5, alpha=0.6)

# Tick colours
ax.tick_params(axis="y", colors=C_JOINT)
ax2.tick_params(axis="y", colors=C_INDEP)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color(C_INDEP)
ax2.spines["right"].set_linewidth(0.5)
ax2.spines["top"].set_visible(False)

# Highlight the two key thresholds on both curves.
row01 = thresh[thresh["threshold"] == 0.01].iloc[0]
row05 = thresh[thresh["threshold"] == 0.05].iloc[0]
for row in (row01, row05):
    ax.plot(row["threshold"], row["spearman_rho"], "o",
            color=C_JOINT, ms=5.5, mec="white", mew=0.8, zorder=5)
    ax2.plot(row["threshold"], row["sign_agreement"] * 100, "s",
             color=C_INDEP, ms=5, mec="white", mew=0.8, zorder=5)

# Short leader lines from each label to its marker, with the labels placed
# in curve-free regions so they don't overlap the plotted lines.
_arrow = lambda c: dict(arrowstyle="-", color=c, lw=0.4, alpha=0.8,
                        shrinkA=1, shrinkB=3)

# |LFC|=0.01: stack ρ label above, sign label below (both to lower-right of markers).
ax.annotate(f'{row01["spearman_rho"]:.2f}',
            xy=(row01["threshold"], row01["spearman_rho"]),
            xytext=(18, -8), textcoords="offset points",
            fontsize=5.5, color=C_JOINT, fontweight="bold",
            ha="left", va="center", arrowprops=_arrow(C_JOINT))
ax2.annotate(f'{row01["sign_agreement"]*100:.0f}%',
             xy=(row01["threshold"], row01["sign_agreement"] * 100),
             xytext=(18, -18), textcoords="offset points",
             fontsize=5.5, color=C_INDEP, fontweight="bold",
             ha="left", va="center", arrowprops=_arrow(C_INDEP))

# |LFC|=0.05: ρ above, sign below (switched from prior arrangement).
ax.annotate(f'{row05["spearman_rho"]:.2f}',
            xy=(row05["threshold"], row05["spearman_rho"]),
            xytext=(14, 10), textcoords="offset points",
            fontsize=5.5, color=C_JOINT, fontweight="bold",
            ha="left", va="center", arrowprops=_arrow(C_JOINT))
ax2.annotate(f'{row05["sign_agreement"]*100:.0f}%',
             xy=(row05["threshold"], row05["sign_agreement"] * 100),
             xytext=(14, -14), textcoords="offset points",
             fontsize=5.5, color=C_INDEP, fontweight="bold",
             ha="left", va="center", arrowprops=_arrow(C_INDEP))

lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, frameon=False, fontsize=5.5,
          loc="lower right", bbox_to_anchor=(1.0, -0.02),
          handlelength=1.5, handletextpad=0.4, borderpad=0.2)

# ══ b. Genes per locus ═══════════════════════════════════════
ax = fig.add_subplot(gs[0, 1])
panel_label(ax, "b")

xlim_b = 22
bins_b = np.arange(0, xlim_b + 2) - 0.5
ax.hist(persnp_p["n_genes"].clip(upper=xlim_b), bins=bins_b,
        alpha=0.6, color=C_INDEP, density=True, edgecolor="white", lw=0.3,
        label=f"Independent  (n={len(persnp_p):,})")
ax.hist(joint_p["n_genes"].clip(upper=xlim_b),  bins=bins_b,
        alpha=0.6, color=C_JOINT,  density=True, edgecolor="white", lw=0.3,
        label=f"Joint  (n={len(joint_p):,})")

med_persnp_g = persnp_p["n_genes"].median()
med_joint_g  = joint_p["n_genes"].median()
ax.axvline(med_persnp_g, color=C_INDEP, ls="--", lw=1.1, alpha=0.9)
ax.axvline(med_joint_g,  color=C_JOINT,  ls="--", lw=1.1, alpha=0.9)

ymax_b = ax.get_ylim()[1]
ax.text(med_persnp_g + 0.4, ymax_b * 0.94,
        f"median = {med_persnp_g:.0f}", fontsize=5, color=C_INDEP,
        fontweight="bold", va="top")
ax.text(med_joint_g  + 0.4, ymax_b * 0.78,
        f"median = {med_joint_g:.0f}",  fontsize=5, color=C_JOINT,
        fontweight="bold", va="top")

ax.set_xlim(-0.5, xlim_b + 0.5)
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
ax.set_xlabel("Genes per locus", labelpad=3)
ax.set_ylabel("Density", labelpad=3)
ax.legend(frameon=False, fontsize=5.5, loc="upper right",
          handlelength=1.2, handletextpad=0.4)

# ══ c. Tissues per locus ═════════════════════════════════════
ax = fig.add_subplot(gs[1, 0])
panel_label(ax, "c")

bins_c = np.arange(0, 56) - 0.5
ax.hist(persnp_p["n_tissues"], bins=bins_c,
        alpha=0.6, color=C_INDEP, density=True, edgecolor="white", lw=0.3,
        label="Independent")
ax.hist(joint_p["n_tissues"],  bins=bins_c,
        alpha=0.6, color=C_JOINT,  density=True, edgecolor="white", lw=0.3,
        label="Joint")

med_persnp_t = persnp_p["n_tissues"].median()
med_joint_t  = joint_p["n_tissues"].median()
ax.axvline(med_persnp_t, color=C_INDEP, ls="--", lw=1.1, alpha=0.9)
ax.axvline(med_joint_t,  color=C_JOINT,  ls="--", lw=1.1, alpha=0.9)

ymax_c = ax.get_ylim()[1]
ax.text(med_persnp_t - 1.5, ymax_c * 0.94,
        f"median = {med_persnp_t:.0f}", fontsize=5, color=C_INDEP,
        fontweight="bold", va="top", ha="right")
ax.text(med_joint_t  - 1.5, ymax_c * 0.94,
        f"median = {med_joint_t:.0f}",  fontsize=5, color=C_JOINT,
        fontweight="bold", va="top", ha="right")

ax.set_xlim(0, 55)
ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
ax.set_xlabel("Tissues per locus", labelpad=3)
ax.set_ylabel("Density", labelpad=3)
ax.legend(frameon=False, fontsize=5.5, loc="upper left",
          handlelength=1.2, handletextpad=0.4)

# ══ d. Tau distributions ═════════════════════════════════════
ax = fig.add_subplot(gs[1, 1])
panel_label(ax, "d")

bins_d = np.linspace(0, 1, 45)
ax.hist(persnp_tau_vals, bins=bins_d,
        alpha=0.6, color=C_INDEP, density=True, edgecolor="white", lw=0.3,
        label=f"Independent  (n = {len(persnp_tau_vals):,})")
ax.hist(joint_tau_vals,  bins=bins_d,
        alpha=0.6, color=C_JOINT,  density=True, edgecolor="white", lw=0.3,
        label=f"Joint  (n = {len(joint_tau_vals):,})")

med_persnp_tau = persnp_tau_vals.median()
med_joint_tau  = joint_tau_vals.median()
ax.axvline(med_persnp_tau, color=C_INDEP, ls="--", lw=1.1, alpha=0.9)
ax.axvline(med_joint_tau,  color=C_JOINT,  ls="--", lw=1.1, alpha=0.9)

# Median labels above the histogram, staggered, with leader arrows
ax.annotate(f"Independent\nmedian = {med_persnp_tau:.2f}",
            xy=(med_persnp_tau, 1.0), xycoords=("data", "axes fraction"),
            xytext=(med_persnp_tau - 0.14, 1.16), textcoords=("data", "axes fraction"),
            fontsize=5, color=C_INDEP, fontweight="bold", ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-|>", color=C_INDEP, lw=0.7,
                            mutation_scale=5))
ax.annotate(f"Joint\nmedian = {med_joint_tau:.2f}",
            xy=(med_joint_tau, 1.0), xycoords=("data", "axes fraction"),
            xytext=(med_joint_tau + 0.14, 1.16), textcoords=("data", "axes fraction"),
            fontsize=5, color=C_JOINT, fontweight="bold", ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-|>", color=C_JOINT, lw=0.7,
                            mutation_scale=5))

mwu_stat, mwu_p = mannwhitneyu(joint_tau_vals, persnp_tau_vals,
                                alternative="greater")
p_str = "p < 1e-10" if mwu_p < 1e-10 else f"p = {mwu_p:.1e}"
stat_box(ax, f"Mann\u2013Whitney U\n{p_str}", x=0.97, y=0.97)

ax.set_xlim(0, 1.0)
ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax.set_xlabel("Tissue specificity (\u03c4)", labelpad=3)
ax.set_ylabel("Density", labelpad=3)
ax.legend(frameon=False, fontsize=5.5, loc="upper left",
          handlelength=1.2, handletextpad=0.4)

# ── Save ─────────────────────────────────────────────────────
fig.savefig(OUT / "fig2_joint_scoring.png", dpi=300)
fig.savefig(OUT / "fig2_joint_scoring.pdf")
plt.close()
print("Saved fig2_joint_scoring")
