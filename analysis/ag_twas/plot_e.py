#!/usr/bin/env python3
"""Panel E: Validation — drug target and OpenTargets enrichment by sig group."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_common import apply_style, C_AG, C_TWAS, C_BOTH, C_GREY

apply_style()
OUT_DIR = Path("results/insights/ag_twas")


def main():
    v = pd.read_csv(OUT_DIR / "validation_enrichment.csv")

    # Show only the sig groups
    cats = ['both_sig', 'ag_only_sig', 'twas_only_sig', 'ag_exclusive_sig']
    cat_labels = ['Both sig', 'AG-only sig', 'TWAS-only sig', 'AG exclusive\n(no TWAS model)']
    cat_colors = [C_BOTH, C_AG, C_TWAS, C_AG]
    cat_hatches = ['', '', '', '//']

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    for ax_idx, (gt, title) in enumerate([('drug_targets', 'Drug target enrichment'),
                                           ('nongwas_ot', 'Non-GWAS evidence enrichment')]):
        ax = axes[ax_idx]
        sub = v[v['ground_truth'] == gt]

        ors = []
        labels = []
        colors = []
        hatches = []
        sig_labels = []
        for cat, label, color, hatch in zip(cats, cat_labels, cat_colors, cat_hatches):
            row = sub[sub['category'] == cat]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            ors.append(row['OR'])
            labels.append(label)
            colors.append(color)
            hatches.append(hatch)
            sig_labels.append(row['sig_label'])

        y = np.arange(len(ors))
        bars = ax.barh(y, ors, color=colors, height=0.6, edgecolor='white', linewidth=0.5)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        ax.axvline(1.0, color='k', ls='--', lw=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Odds ratio")
        ax.set_title(title, fontweight='bold')

        # Add OR values and significance
        for i, (or_val, sig) in enumerate(zip(ors, sig_labels)):
            x_pos = or_val + 0.05
            ax.text(x_pos, i, f"{or_val:.1f}× {sig}", va='center', fontsize=6.5)

        ax.invert_yaxis()
        ax.set_xlim(0, max(ors) * 1.4)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "panel_e.png")
    fig.savefig(OUT_DIR / "panel_e.pdf")
    plt.close()
    print("Saved panel_e")


if __name__ == "__main__":
    main()
