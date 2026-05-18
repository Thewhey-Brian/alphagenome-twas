#!/usr/bin/env python3
"""Panel B: Gene classification — proportion of each category within each gene set."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plot_common import apply_style, C_AG, C_TWAS, C_BOTH

apply_style()
OUT_DIR = Path("results/insights/ag_twas")


def main():
    gc = pd.read_csv(OUT_DIR / "gene_classification.csv", index_col=0)

    cats = ['protein_coding', 'lncRNA', 'pseudogene', 'miRNA', 'snRNA',
            'snoRNA', 'other_ncRNA', 'IG/TR', 'other']
    cats = [c for c in cats if c in gc.index]

    sets = [('AG_only', C_AG, 'AG only'), ('shared', C_BOTH, 'Shared'), ('TWAS_only', C_TWAS, 'TWAS only')]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    y = np.arange(len(cats))
    w = 0.25

    for i, (s, color, label) in enumerate(sets):
        vals = [gc.loc[c, s] if s in gc.columns and c in gc.index else 0 for c in cats]
        total = sum(vals)
        pct = [v / total * 100 if total > 0 else 0 for v in vals]
        bars = ax.barh(y + (i - 1) * w, pct, height=w, color=color, label=f"{label} (n={total:,})")

    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.set_xlabel("% of genes in set")
    ax.legend(frameon=False, fontsize=6, loc='lower right')
    ax.set_title("Gene type composition by set", fontweight='bold')
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(OUT_DIR / "panel_b.png")
    fig.savefig(OUT_DIR / "panel_b.pdf")
    plt.close()
    print("Saved panel_b")


if __name__ == "__main__":
    main()
