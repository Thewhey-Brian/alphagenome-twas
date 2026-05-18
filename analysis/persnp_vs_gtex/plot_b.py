#!/usr/bin/env python3
"""Panel b: Venn-style overlap diagram — AG triplets ∩ GTEx eQTL triplets."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from plot_common import apply_style
apply_style()

OUT_DIR = Path("results/insights/persnp_vs_gtex")
ov = pd.read_csv(OUT_DIR / "overlap_stats.csv", index_col=0)
ov.columns = ["value"]

C_BLUE, C_GREEN = "#4C72B0", "#55A868"

fig, ax = plt.subplots(figsize=(4.5, 3.2))

# Draw proportional-ish circles
c_ag = Circle((-0.3, 0), 0.55, fc=C_BLUE, alpha=0.25, ec=C_BLUE, lw=1.5, label="AG per-SNP")
c_gtex = Circle((0.3, 0), 0.55, fc=C_GREEN, alpha=0.25, ec=C_GREEN, lw=1.5, label="GTEx sig eQTL")
ax.add_patch(c_ag)
ax.add_patch(c_gtex)

# Labels
ag_only_snps = int(ov.loc["ag_snps", "value"]) - int(ov.loc["matched_snps", "value"])
gtex_only_snps = int(ov.loc["gtex_snps", "value"]) - int(ov.loc["matched_snps", "value"])
matched_snps = int(ov.loc["matched_snps", "value"])

# AG only
ax.text(-0.65, 0.15, f'{ag_only_snps:,}\nSNPs', ha="center", fontsize=8, fontweight="bold", color=C_BLUE)
ax.text(-0.65, -0.15, f'{int(ov.loc["ag_triplets","value"]):,}\ntriplets', ha="center", fontsize=6, color=C_BLUE)

# Overlap
ax.text(0, 0.15, f'{matched_snps:,}\nSNPs', ha="center", fontsize=8, fontweight="bold", color="#333333")
ax.text(0, -0.15, f'{int(ov.loc["matched_triplets","value"]):,}\ntriplets', ha="center", fontsize=6, color="#333333")

# GTEx only
ax.text(0.65, 0.15, f'{gtex_only_snps:,}\nSNPs', ha="center", fontsize=7, fontweight="bold", color=C_GREEN)
ax.text(0.65, -0.15, f'{int(ov.loc["gtex_triplets","value"]):,}\ntriplets', ha="center", fontsize=5.5, color=C_GREEN)

# Title labels
ax.text(-0.55, 0.55, "AG per-SNP\n(CS SNPs × genes × tissues)", ha="center", fontsize=6.5, color=C_BLUE, fontstyle="italic")
ax.text(0.55, 0.55, "GTEx v8 sig eQTLs\n(variant × gene × tissue)", ha="center", fontsize=6.5, color=C_GREEN, fontstyle="italic")

# Summary
ax.text(0, -0.65, f"SNP overlap: {matched_snps:,} / {int(ov.loc['ag_snps','value']):,} CS SNPs ({matched_snps/int(ov.loc['ag_snps','value']):.1%})",
        ha="center", fontsize=6.5, color="#555555")

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-0.85, 0.75)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("b", fontweight="bold", loc="left", fontsize=10)

plt.tight_layout()
plt.savefig(OUT_DIR / "panel_b.png"); plt.savefig(OUT_DIR / "panel_b.pdf"); plt.close()
print("Saved panel_b")
