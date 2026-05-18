#!/usr/bin/env python3
"""Panel D: Concordance — 2×2 table, jointly sig scatter, ranking histogram, per-trait agreement."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_common import apply_style, C_AG, C_TWAS, C_BOTH, C_GREY

apply_style()
OUT_DIR = Path("results/insights/ag_twas")
MERGED_DIR = OUT_DIR / "merged"


def main():
    contingency = pd.read_csv(OUT_DIR / "concordance_2x2.csv", index_col=0)
    summary = pd.read_csv(OUT_DIR / "concordance_summary.csv", index_col=0, header=0)['value']
    rho_df = pd.read_csv(OUT_DIR / "concordance_rho_per_tt.csv")
    trait_stats = pd.read_csv(OUT_DIR / "concordance_per_trait.csv")

    # Load matched data for scatter
    m = pd.read_parquet(MERGED_DIR / "ag_twas_merged.parquet")
    both = m[m['_merge'] == 'both']
    js = both[(both['max_lfc'].abs() > 0.01) & (both['twas_sig_bonf'])]

    # Trait name mapping
    trait_names = pd.read_parquet("/media/brian/TOSHIBA EXT/AG/trait_map.parquet")
    if 'trait_name' in trait_names.columns:
        tmap = trait_names.drop_duplicates('accession').set_index('accession')['trait_name'].to_dict()
    else:
        tmap = {}

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    # --- (a) 2×2 significance table ---
    ax = axes[0, 0]
    vals = contingency.values  # [[d, c], [b, a]]
    total = vals.sum()
    # Colors: neither=light grey, TWAS-only=light blue, AG-only=light red, both=light purple
    cell_colors = [['#f0f0f0', '#d4e6f1'], ['#fadbd8', '#d7bde2']]
    cell_labels = [
        [f"{int(vals[0,0]):,}\n({vals[0,0]/total*100:.0f}%)",
         f"{int(vals[0,1]):,}\n({vals[0,1]/total*100:.0f}%)"],
        [f"{int(vals[1,0]):,}\n({vals[1,0]/total*100:.0f}%)",
         f"{int(vals[1,1]):,}\n({vals[1,1]/total*100:.0f}%)"],
    ]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1-i), 1, 1, color=cell_colors[i][j], ec='white', lw=2))
            ax.text(j + 0.5, 1.5 - i, cell_labels[i][j], ha='center', va='center', fontsize=8)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(['TWAS −', 'TWAS +'], fontweight='bold')
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(['AG +', 'AG −'], fontweight='bold')
    ax.set_title(f"a  Significance overlap (OR = {float(summary['OR']):.2f})",
                 loc='left', fontweight='bold')

    # --- (b) Jointly significant scatter ---
    ax = axes[0, 1]
    sample = js.sample(min(10000, len(js)), random_state=42)
    ax.scatter(sample['max_lfc'].abs(), sample['TWAS.Z'].abs(),
               s=3, alpha=0.3, color=C_BOTH, edgecolors='none')
    from scipy.stats import spearmanr
    rho_js, _ = spearmanr(js['max_lfc'].abs(), js['TWAS.Z'].abs())
    ax.text(0.05, 0.95, f"ρ = {rho_js:.3f}\nN = {len(js):,}",
            transform=ax.transAxes, va='top', fontsize=7)
    ax.set_xlabel("AG |max_lfc|")
    ax.set_ylabel("|TWAS.Z|")
    ax.set_title("b  Jointly significant genes", loc='left', fontweight='bold')

    # --- (c) Per trait×tissue ranking correlation ---
    ax = axes[1, 0]
    ax.hist(rho_df['rho'], bins=50, color=C_GREY, alpha=0.7, edgecolor='none')
    med = rho_df['rho'].median()
    ax.axvline(med, color=C_BOTH, ls='--', lw=1.5, label=f"Median ρ = {med:.3f}")
    ax.axvline(0, color='k', ls=':', lw=0.5)
    pct_pos = (rho_df['rho'] > 0).mean() * 100
    pct_sig = (rho_df['p'] < 0.05).mean() * 100
    ax.text(0.95, 0.92, f"{pct_pos:.0f}% positive\n{pct_sig:.0f}% sig (p<0.05)",
            transform=ax.transAxes, ha='right', fontsize=7)
    ax.set_xlabel("Spearman ρ (|max_lfc| vs |TWAS.Z|)")
    ax.set_ylabel("Trait × tissue pairs")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("c  Per trait×tissue correlation", loc='left', fontweight='bold')

    # --- (d) Per-trait gene-level agreement ---
    ax = axes[1, 1]
    # Compute median rho per trait (across tissues)
    trait_rho = rho_df.groupby('accession').agg(
        median_rho=('rho', 'median'),
        n_tissues=('rho', 'size'),
        n_sig=('p', lambda x: (x < 0.05).sum()),
    ).reset_index()
    trait_rho['trait_name'] = trait_rho['accession'].map(tmap).fillna(trait_rho['accession'])
    trait_rho = trait_rho.sort_values('median_rho', ascending=True)

    y = np.arange(len(trait_rho))
    colors = [C_AG if r < 0 else '#2ecc71' for r in trait_rho['median_rho']]
    ax.barh(y, trait_rho['median_rho'], color=colors, height=0.7)
    ax.axvline(0, color='k', lw=0.5)

    # Significance stars
    for i, (_, row) in enumerate(trait_rho.iterrows()):
        frac_sig = row['n_sig'] / row['n_tissues'] if row['n_tissues'] > 0 else 0
        star = ''
        if frac_sig > 0.5:
            star = '***'
        elif frac_sig > 0.3:
            star = '**'
        elif frac_sig > 0.1:
            star = '*'
        if star:
            x_pos = row['median_rho'] + (0.003 if row['median_rho'] >= 0 else -0.003)
            ax.text(x_pos, i, star, va='center', fontsize=5,
                    ha='left' if row['median_rho'] >= 0 else 'right',
                    color=C_BOTH)

    ax.set_yticks(y)
    ax.set_yticklabels(trait_rho['trait_name'], fontsize=5)
    ax.set_xlabel("Median Spearman ρ across tissues")
    ax.set_title("d  Per-trait gene-level agreement", loc='left', fontweight='bold')

    # Star legend
    ax.text(0.98, 0.05, "* >10% sig pairs\n** >30%\n*** >50%",
            transform=ax.transAxes, fontsize=5, ha='right', va='bottom', color=C_BOTH)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "panel_d.png")
    fig.savefig(OUT_DIR / "panel_d.pdf")
    plt.close()
    print("Saved panel_d")


if __name__ == "__main__":
    main()
