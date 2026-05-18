#!/usr/bin/env python3
"""Figure 4: AG and TWAS are complementary for gene prioritization.

Panels (3 rows):
  Row 1:  a. Coverage overlap bars  |  b. Gene type composition
  Row 2:  c. 2x2 significance      |  d. Per-trait×tissue ranking correlation
  Row 3:  e. Validation enrichment (drug targets + OpenTargets) — full width
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import (apply_style, panel_label, stat_box,
                       C_AG, C_TWAS, C_BOTH, C_GREY, C_RED, DOUBLE_COL)
apply_style()

DATA = Path(__file__).resolve().parent.parent / "ag_twas"
OUT  = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

def fmt(n):
    if n >= 1_000_000: return f"{n/1e6:.1f}M"
    elif n >= 1_000: return f"{n/1e3:.0f}K"
    return f"{n:,}"

# ── Load data ────────────────────────────────────────────────
cov      = pd.read_csv(DATA / "coverage.csv")
cov_sig  = pd.read_csv(DATA / "coverage_sig.csv")
sig_bd   = pd.read_csv(DATA / "sig_breakdown.csv")
# 5-way decomposition of significant genes (from sig_breakdown.csv):
ag_only_row   = sig_bd[sig_bd["direction"].str.startswith("AG-sig")].iloc[0]
twas_only_row = sig_bd[sig_bd["direction"].str.startswith("TWAS-sig")].iloc[0]
N_AG_EXCL_SIG   = int(ag_only_row["no_model"])            # AG-sig, no TWAS model
N_SHARED_AG_SIG = int(ag_only_row["has_model_not_sig"])   # shared, AG-sig only
N_BOTH_SIG      = int(cov_sig[cov_sig["level"].str.contains("Gene")].iloc[0]["Shared"])
N_SHARED_TW_SIG = int(twas_only_row["has_score_not_sig"]) # shared, TWAS-sig only
N_TW_EXCL_SIG   = int(twas_only_row["no_score"])          # TWAS-sig, no AG score
gc       = pd.read_csv(DATA / "gene_classification.csv", index_col=0)
cont     = pd.read_csv(DATA / "concordance_2x2.csv", index_col=0)
summary  = pd.read_csv(DATA / "concordance_summary.csv", index_col=0, header=0)["value"]
rho_df   = pd.read_csv(DATA / "concordance_rho_per_tt.csv")
val      = pd.read_csv(DATA / "validation_enrichment.csv")

# ── Figure ───────────────────────────────────────────────────
fig = plt.figure(figsize=(DOUBLE_COL, 7.5))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.2, 1.0],
             hspace=0.50, wspace=0.40,
             left=0.09, right=0.97, top=0.96, bottom=0.06)

# ══ a. Coverage + significance breakdown ═════════════════════
# Two sub-axes with INDEPENDENT x-axes:
#   top   → COVERAGE (outlined bars, 0–54K gene scale)
#   bottom → SIGNIFICANCE (filled bars, 0–100% of each universe)
# Absolute counts annotated on each segment.
row_all = cov[cov["level"] == "Gene"].iloc[0]
N_AG_EXCL = int(row_all["AG_only"])
N_SHARED  = int(row_all["Shared"])
N_TW_EXCL = int(row_all["TWAS_only"])
total_all = N_AG_EXCL + N_SHARED + N_TW_EXCL

N_SHARED_BOTH    = N_BOTH_SIG
N_SHARED_AGSIG   = N_SHARED_AG_SIG
N_SHARED_TWSIG   = N_SHARED_TW_SIG
N_SHARED_NEITHER = N_SHARED - N_SHARED_BOTH - N_SHARED_AGSIG - N_SHARED_TWSIG
N_AGEX_SIG   = N_AG_EXCL_SIG
N_AGEX_NOSIG = N_AG_EXCL - N_AGEX_SIG
N_TWEX_SIG   = N_TW_EXCL_SIG
N_TWEX_NOSIG = N_TW_EXCL - N_TWEX_SIG

C_NOSIG = "#e6e6e6"

inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0],
                                 height_ratios=[1.0, 2.4], hspace=0.5)
ax_cov = fig.add_subplot(inner[0])
ax_sig = fig.add_subplot(inner[1])
panel_label(ax_cov, "a")

# ── Coverage (absolute counts) ──────────────────────────────
cov_segs = [(N_AG_EXCL, C_AG, "AG-exclusive"),
            (N_SHARED,  C_BOTH, "Shared"),
            (N_TW_EXCL, C_TWAS, "TWAS-exclusive")]
left = 0
for v, c, _ in cov_segs:
    ax_cov.barh(0, v, height=0.55, left=left,
                facecolor="white", edgecolor=c, lw=1.4, zorder=3)
    if v / total_all > 0.04:
        ax_cov.text(left + v / 2, 0, fmt(int(v)),
                    ha="center", va="center", fontsize=5.5,
                    color=c, fontweight="bold")
    left += v
ax_cov.text(total_all * 1.01, 0, f"n = {fmt(total_all)}",
            fontsize=5, color="#444", va="center", ha="left")
ax_cov.set_yticks([0])
ax_cov.set_yticklabels(["All genes"], fontsize=5.8)
ax_cov.set_xlabel("Coverage — number of genes", fontsize=6, labelpad=2)
ax_cov.set_xlim(0, total_all * 1.12)
ax_cov.set_ylim(-0.5, 0.5)
ax_cov.tick_params(axis="x", labelsize=5)
# Coverage legend — 3 outlined patches, placed above the coverage bar
ax_cov.legend(handles=[
    Patch(facecolor="white", edgecolor=C_AG,   lw=1.3, label="AG-exclusive"),
    Patch(facecolor="white", edgecolor=C_BOTH, lw=1.3, label="Shared"),
    Patch(facecolor="white", edgecolor=C_TWAS, lw=1.3, label="TWAS-exclusive"),
], frameon=False, fontsize=5, loc="lower center",
   bbox_to_anchor=(0.5, 1.05), ncol=3,
   handlelength=1.3, handletextpad=0.3, columnspacing=1.0)

# ── Significance (% within each universe) ───────────────────
# Bar order within Shared row: AG-only sig | Both sig | TWAS-only sig | Neither
sig_rows = [
    (0, "AG-exclusive",   N_AG_EXCL, [
        (N_AGEX_SIG,   C_AG,    None, "AG prio"),
        (N_AGEX_NOSIG, C_NOSIG, None, None),
    ]),
    (1, "Shared",         N_SHARED, [
        (N_SHARED_AGSIG,   C_AG,    None, "AG-only prio"),
        (N_SHARED_BOTH,    C_BOTH,  None, "Both prio"),
        (N_SHARED_TWSIG,   C_TWAS,  None, "TWAS-only prio"),
        (N_SHARED_NEITHER, C_NOSIG, None, None),
    ]),
    (2, "TWAS-exclusive", N_TW_EXCL, [
        (N_TWEX_SIG,   C_TWAS,  None, "TWAS prio"),
        (N_TWEX_NOSIG, C_NOSIG, None, None),
    ]),
]
for y, label, uni_total, segs in sig_rows:
    left = 0.0
    for v, c, h, _ in segs:
        pct = v / uni_total * 100
        ax_sig.barh(y, pct, height=0.55, left=left, color=c, hatch=h,
                    edgecolor="white", lw=0.5, zorder=3)
        if pct > 4:
            ax_sig.text(left + pct / 2, y, fmt(int(v)),
                        ha="center", va="center", fontsize=5,
                        color="white" if c != C_NOSIG else "#333",
                        fontweight="bold")
        left += pct
    ax_sig.text(101, y, f"n = {fmt(uni_total)}",
                fontsize=5, color="#444", va="center", ha="left")

ax_sig.set_yticks([0, 1, 2])
ax_sig.set_yticklabels([""] * 3)  # suppress defaults; use colored bbox labels instead
_label_colors = [C_AG, C_BOTH, C_TWAS]
_label_text   = ["AG-exclusive", "Shared", "TWAS-exclusive"]
for yi, (txt, col) in enumerate(zip(_label_text, _label_colors)):
    ax_sig.text(-0.02, yi, txt, transform=ax_sig.get_yaxis_transform(),
                fontsize=5.8, ha="right", va="center", color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor=col, lw=1.0))
ax_sig.invert_yaxis()
ax_sig.set_xlabel("Prioritization — % of genes in universe", fontsize=6, labelpad=2)
ax_sig.set_xlim(0, 115)
ax_sig.set_xticks([0, 25, 50, 75, 100])
ax_sig.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=5)
# Significance legend — four entries
leg_sig = [
    Patch(facecolor=C_AG,    label="AG prio"),
    Patch(facecolor=C_BOTH,  label="Both prio"),
    Patch(facecolor=C_TWAS,  label="TWAS prio"),
    Patch(facecolor=C_NOSIG, label="Not prio"),
]
ax_sig.legend(handles=leg_sig, frameon=False, fontsize=5,
              loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=4,
              handlelength=1.3, handletextpad=0.3, columnspacing=1.0)

# ══ b. Gene type composition ═════════════════════════════════
ax = fig.add_subplot(gs[0, 1])
panel_label(ax, "b")

cats = ["protein_coding", "lncRNA", "pseudogene", "miRNA", "snRNA", "snoRNA", "other_ncRNA"]
cats = [c for c in cats if c in gc.index]
# Panel b shows coverage-universe gene-type composition.
# Use OUTLINED bars to match the coverage style in panel a.
sets_info = [("AG_only", C_AG,   "AG-exclusive"),
             ("shared",  C_BOTH, "Shared"),
             ("TWAS_only", C_TWAS, "TWAS-exclusive")]

y = np.arange(len(cats))
w = 0.25
pct_data = {}
for i, (s, color, label) in enumerate(sets_info):
    vals = [gc.loc[c, s] if s in gc.columns and c in gc.index else 0 for c in cats]
    total = sum(vals)
    pct = [v / total * 100 if total > 0 else 0 for v in vals]
    pct_data[s] = pct
    ax.barh(y + (i - 1) * w, pct, height=w,
            facecolor="white", edgecolor=color, lw=1.1,
            label=f"{label} (n={total:,})")
    for j, p in enumerate(pct):
        if p > 0.5:
            ax.text(p + 0.8, y[j] + (i - 1) * w, f"{p:.0f}%",
                    fontsize=4.5, color=color, fontweight="bold", va="center")

ax.set_yticks(y); ax.set_yticklabels(cats, fontsize=5.5)
ax.set_xlabel("% of genes in set")
ax.set_xlim(0, max(max(v) for v in pct_data.values()) * 1.35)
ax.legend(frameon=False, fontsize=4.5, loc="lower right")
ax.invert_yaxis()

# ══ c. 2×2 significance table ════════════════════════════════
ax = fig.add_subplot(gs[1, 0])
panel_label(ax, "c")

# Re-ordered 2×2 so the canonical cell labels (a,b,c,d) are at the
# top-left → top-right → bottom-left → bottom-right positions:
#   a (top-left)  = AG sig  & TWAS sig      (Both sig)          → pink tint
#   b (top-right) = AG sig  & TWAS not sig  (AG-only sig)       → orange tint
#   c (bot-left)  = AG not sig & TWAS sig   (TWAS-only sig)     → blue tint
#   d (bot-right) = AG not sig & TWAS not sig (Neither)         → grey
vals_raw = cont.values  # rows: [AG-, AG+], cols: [TWAS-, TWAS+]
N_BOTH    = int(vals_raw[1, 1])   # AG+ & TWAS+
N_AG_ONLY = int(vals_raw[1, 0])   # AG+ & TWAS-
N_TW_ONLY = int(vals_raw[0, 1])   # AG- & TWAS+
N_NEITHER = int(vals_raw[0, 0])   # AG- & TWAS-
total = N_BOTH + N_AG_ONLY + N_TW_ONLY + N_NEITHER

# cells[(col, row)] = (value, color, alpha)  — col/row follow matplotlib axes
# with top row at y=1, left col at x=0.
cells = {
    (0, 1): (N_BOTH,    C_BOTH,  0.38),   # a: top-left
    (1, 1): (N_AG_ONLY, C_AG,    0.30),   # b: top-right
    (0, 0): (N_TW_ONLY, C_TWAS,  0.28),   # c: bottom-left
    (1, 0): (N_NEITHER, C_NOSIG, 1.00),   # d: bottom-right
}
cell_letters = {(0, 1): "a", (1, 1): "b", (0, 0): "c", (1, 0): "d"}

for (j, row_y), (v, color, alpha) in cells.items():
    ax.add_patch(plt.Rectangle((j, row_y), 1, 1,
                 facecolor=color, alpha=alpha, ec="white", lw=2))
    ax.text(j + 0.5, row_y + 0.5,
            f"{v:,}\n({v / total * 100:.0f}%)",
            ha="center", va="center", fontsize=7)
    # corner letter annotation
    ax.text(j + 0.06, row_y + 0.94, cell_letters[(j, row_y)],
            fontsize=6, fontweight="bold", color="#444",
            ha="left", va="top")

ax.set_xlim(0, 2); ax.set_ylim(-0.35, 2)
ax.set_xticks([]); ax.set_yticks([])
ax.set_aspect("equal")

# Column headers — TWAS sig on LEFT so "a" sits at top-left
ax.text(0.5, -0.08, "TWAS\nprio",     ha="center", va="top",
        fontsize=6, fontweight="bold", color="#444")
ax.text(1.5, -0.08, "TWAS\nnot prio", ha="center", va="top",
        fontsize=6, fontweight="bold", color="#444")
# Row headers — AG sig on TOP
ax.text(-0.08, 1.5, "AG\nprio",     ha="right", va="center",
        fontsize=6, fontweight="bold", color="#444")
ax.text(-0.08, 0.5, "AG\nnot prio", ha="right", va="center",
        fontsize=6, fontweight="bold", color="#444")

or_val = float(summary["OR"])
from scipy.stats import fisher_exact as _fe
_, p_c = _fe([[N_BOTH, N_AG_ONLY], [N_TW_ONLY, N_NEITHER]], alternative="greater")
p_str_c = "p < 10\u207b\u00b3\u2070\u2070" if p_c < 1e-300 else f"p = {p_c:.1e}"
ax.set_title(f"OR = {or_val:.1f},  {p_str_c}  (Fisher's exact)",
             fontsize=6.5, pad=4)
ax.text(0.5, -0.06, f"Shared universe, gene\u00d7trait\u00d7tissue triplets (n = {int(total):,})",
        ha="center", va="top", fontsize=5, color="#666",
        transform=ax.transAxes)

# ══ d. Ranking correlation histogram ═════════════════════════
ax = fig.add_subplot(gs[1, 1])
panel_label(ax, "d")

ax.hist(rho_df["rho"], bins=50, color=C_GREY, alpha=0.7, edgecolor="none")
med = rho_df["rho"].median()
ax.axvline(med, color=C_BOTH, ls="--", lw=1.5, label=f"Median \u03c1 = {med:.3f}")
ax.axvline(0, color="k", ls=":", lw=0.4)
pct_pos = (rho_df["rho"] > 0).mean() * 100
pct_sig = (rho_df["p"] < 0.05).mean() * 100
stat_box(ax, f"{pct_pos:.0f}% positive\n{pct_sig:.0f}% sig (p<0.05)")
ax.set_xlabel("Spearman \u03c1 (|max_lfc| vs |TWAS.Z|),  shared universe")
ax.set_ylabel("Trait \u00d7 tissue pairs")
ax.legend(frameon=False, fontsize=6)

# ══ e. Validation enrichment ═════════════════════════════════
ax_l = fig.add_subplot(gs[2, 0])
ax_r = fig.add_subplot(gs[2, 1])
panel_label(ax_l, "e")

# Shared-universe categories (genes with both AG score and TWAS model).
# Labels aligned with panel-a terminology ("AG-only prio", "Both prio", "TWAS-only prio").
cats_shared = ["ag_only_sig", "both_sig", "twas_only_sig"]
labs_shared  = ["AG-only prio\n(shared)",
                "Both prio\n(shared)",
                "TWAS-only prio\n(shared)"]
cols_shared  = [C_AG, C_BOTH, C_TWAS]

# Exclusive universes (genes present in only one method's coverage).
cats_agonly  = ["ag_exclusive_sig", "twas_exclusive_sig"]
labs_agonly  = ["AG prio\n(AG-exclusive)", "TWAS prio\n(TWAS-exclusive)"]
cols_agonly  = [C_AG, C_TWAS]

for ax, (gt, title) in [(ax_l, ("drug_targets", "Drug target enrichment")),
                         (ax_r, ("nongwas_ot", "Non-GWAS evidence enrichment"))]:
    sub = val[val["ground_truth"] == gt]

    # Collect shared-set rows
    ors_s, labs_s, cols_s, sigs_s = [], [], [], []
    for cat, label, color in zip(cats_shared, labs_shared, cols_shared):
        row = sub[sub["category"] == cat]
        if len(row) == 0: continue
        row = row.iloc[0]
        ors_s.append(row["OR"]); labs_s.append(label)
        cols_s.append(color); sigs_s.append(row["sig_label"])

    # Collect AG-unique rows
    ors_a, labs_a, cols_a, sigs_a = [], [], [], []
    for cat, label, color in zip(cats_agonly, labs_agonly, cols_agonly):
        row = sub[sub["category"] == cat]
        if len(row) == 0: continue
        row = row.iloc[0]
        ors_a.append(row["OR"]); labs_a.append(label)
        cols_a.append(color); sigs_a.append(row["sig_label"])

    # Combined: shared first, then each exclusive universe as its own row with a gap
    n_s = len(ors_s)
    n_a = len(ors_a)
    gap_group = 0.6   # gap between shared group and first exclusive row
    gap_excl  = 0.35  # gap between successive exclusive rows (different universes)
    y_s = np.arange(n_s, dtype=float)
    start = n_s + gap_group
    y_a = np.array([start + i * (1.0 + gap_excl) for i in range(n_a)])

    all_ors  = ors_s  + ors_a
    all_labs = labs_s + labs_a
    all_cols = cols_s + cols_a
    all_sigs = sigs_s + sigs_a
    all_y    = list(y_s) + list(y_a)

    bars = ax.barh(all_y, all_ors, color=all_cols, height=0.55,
                   edgecolor="white", lw=0.5)
    ax.axvline(1.0, color="k", ls="--", lw=0.5)

    ax.set_yticks(all_y)
    ax.set_yticklabels(all_labs, fontsize=5.5)
    ax.set_xlabel("Odds ratio (Fisher's exact, one-sided)", fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(0, max(all_ors) * 1.5)
    ax.set_title(title, fontsize=7, pad=3)

    # OR + significance labels
    for y_i, or_v, sig in zip(all_y, all_ors, all_sigs):
        ax.text(or_v + 0.05, y_i, f"{or_v:.1f}\u00d7 {sig}", va="center", fontsize=5.5)

    # Separator between the two universes
    y_div = (y_s[-1] + y_a[0]) / 2
    ax.axhline(y_div, color=C_GREY, ls="--", lw=0.8, alpha=0.8)

    # Shaded background bands per universe.
    # Top band: shared universe; bottom band(s): one per exclusive category.
    ax.axhspan(y_s[0] - 0.4, y_s[-1] + 0.4, color="#f7f7f7", zorder=0)
    for y_i, label in zip(y_a, all_labs[n_s:]):
        if "AG-exclusive" in label:
            band_color = "#fff8ee"
            uni_label  = "Universe: AG-exclusive genes (no TWAS model)"
        else:
            band_color = "#eaf4fb"
            uni_label  = "Universe: TWAS-exclusive genes (no AG score)"
        ax.axhspan(y_i - 0.4, y_i + 0.4, color=band_color, zorder=0)
        ax.text(0.99, y_i - 0.35, uni_label,
                fontsize=4, color="#888", ha="right", va="bottom",
                style="italic", transform=ax.get_yaxis_transform())

    # Universe label for shared band
    ax.text(0.99, y_s[0] - 0.35,
            "Universe: shared genes (with TWAS model)",
            fontsize=4, color="#888", ha="right", va="bottom",
            style="italic", transform=ax.get_yaxis_transform())

# ── Save ─────────────────────────────────────────────────────
fig.savefig(OUT / "fig4_ag_vs_twas.png")
fig.savefig(OUT / "fig4_ag_vs_twas.pdf")
plt.close()
print("Saved fig4_ag_vs_twas")
