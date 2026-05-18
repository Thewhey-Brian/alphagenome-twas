#!/usr/bin/env python3
"""Panel A: Coverage overlap — gene and triplet levels, unfiltered and filtered."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plot_common import apply_style, C_AG, C_TWAS, C_BOTH

apply_style()
OUT_DIR = Path("results/insights/ag_twas")


def fmt(n):
    if n >= 1_000_000:
        return f"{n/1e6:.1f}M"
    elif n >= 1_000:
        return f"{n/1e3:.0f}K"
    return f"{n:,}"


def draw_bar(ax, ag_only, shared, twas_only, label):
    """Single stacked horizontal bar."""
    total = ag_only + shared + twas_only
    b1 = ax.barh(0, ag_only, height=0.5, color=C_AG)
    b2 = ax.barh(0, shared, height=0.5, left=ag_only, color=C_BOTH)
    b3 = ax.barh(0, twas_only, height=0.5, left=ag_only + shared, color=C_TWAS)

    # Labels
    for val, left in [(ag_only, 0), (shared, ag_only), (twas_only, ag_only + shared)]:
        if val / total > 0.06:
            ax.text(left + val / 2, 0, fmt(int(val)),
                    ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    ax.set_xlim(0, total * 1.02)
    ax.set_yticks([])
    ax.set_title(label, fontsize=8, fontweight='bold', loc='left')


def main():
    cov = pd.read_csv(OUT_DIR / "coverage.csv")
    cov_sig = pd.read_csv(OUT_DIR / "coverage_sig.csv")

    # Pick gene and triplet rows
    gene_row = cov[cov['level'] == 'Gene'].iloc[0]
    trip_row = cov[cov['level'] == 'Gene × Trait × Tissue'].iloc[0]
    gene_sig_row = cov_sig[cov_sig['level'].str.contains('Gene')].iloc[0]
    trip_sig_row = cov_sig[cov_sig['level'].str.contains('Triplet')].iloc[0]

    fig, axes = plt.subplots(4, 1, figsize=(6, 3.5), gridspec_kw={'hspace': 0.8})

    draw_bar(axes[0], gene_row['AG_only'], gene_row['Shared'], gene_row['TWAS_only'],
             "Gene (all)")
    draw_bar(axes[1], trip_row['AG_only'], trip_row['Shared'], trip_row['TWAS_only'],
             "Gene × Trait × Tissue (all)")
    draw_bar(axes[2], gene_sig_row['AG_only'], gene_sig_row['Shared'], gene_sig_row['TWAS_only'],
             "Gene (significant)")
    draw_bar(axes[3], trip_sig_row['AG_only'], trip_sig_row['Shared'], trip_sig_row['TWAS_only'],
             "Gene × Trait × Tissue (significant)")

    # Shared legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(color=C_AG, label='AG only'),
                       Patch(color=C_BOTH, label='Shared'),
                       Patch(color=C_TWAS, label='TWAS only')]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "panel_a.png", bbox_inches='tight')
    fig.savefig(OUT_DIR / "panel_a.pdf", bbox_inches='tight')
    plt.close()
    print("Saved panel_a")


if __name__ == "__main__":
    main()
