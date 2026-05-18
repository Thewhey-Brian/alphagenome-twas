#!/usr/bin/env python3
"""Panel C: Significance breakdown and sig rate by gene type."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_common import apply_style, C_AG, C_TWAS, C_BOTH, C_GREY

apply_style()
OUT_DIR = Path("results/insights/ag_twas")
MERGED_DIR = OUT_DIR / "merged"


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    # --- Panel c1: Why are method-specific sig genes not sig in the other? ---
    ax = axes[0]
    bd = pd.read_csv(OUT_DIR / "sig_breakdown.csv")

    # AG-sig, not TWAS-sig
    ag_row = bd[bd['direction'].str.startswith('AG')]
    no_model = int(ag_row['no_model'].values[0])
    has_model = int(ag_row['has_model_not_sig'].values[0])
    total_ag = no_model + has_model

    # TWAS-sig, not AG-sig
    tw_row = bd[bd['direction'].str.startswith('TWAS')]
    no_score = int(tw_row['no_score'].values[0])
    has_score = int(tw_row['has_score_not_sig'].values[0])
    total_tw = no_score + has_score

    y = [1.5, 0.5]
    # AG-sig not TWAS-sig
    ax.barh(y[0], no_model, height=0.6, color=C_GREY, label='No model/score (coverage gap)')
    ax.barh(y[0], has_model, height=0.6, left=no_model, color=C_AG, label='Has model, not sig (disagreement)')
    ax.text(no_model / 2, y[0], f"{no_model:,}\n({no_model/total_ag*100:.0f}%)",
            ha='center', va='center', fontsize=6, color='white', fontweight='bold')
    ax.text(no_model + has_model / 2, y[0], f"{has_model:,}\n({has_model/total_ag*100:.0f}%)",
            ha='center', va='center', fontsize=6, color='white', fontweight='bold')

    # TWAS-sig not AG-sig
    ax.barh(y[1], no_score, height=0.6, color=C_GREY)
    ax.barh(y[1], has_score, height=0.6, left=no_score, color=C_TWAS, label='Has score, not sig (disagreement)')
    ax.text(no_score + has_score / 2, y[1], f"{has_score:,}\n({has_score/total_tw*100:.0f}%)",
            ha='center', va='center', fontsize=6, color='white', fontweight='bold')
    # no_score is small, label outside
    ax.text(no_score / 2, y[1], f"{no_score:,}", ha='center', va='center', fontsize=5, color='white')

    ax.set_yticks(y)
    ax.set_yticklabels([f"AG-sig,\nnot TWAS\n(n={total_ag:,})", f"TWAS-sig,\nnot AG\n(n={total_tw:,})"])
    ax.set_xlabel("Number of genes")
    ax.legend(frameon=False, fontsize=5.5, bbox_to_anchor=(0.5, -0.25), loc='upper center', ncol=1)
    ax.set_title("Why method-specific genes\nare not sig in the other", fontweight='bold')

    # --- Panel c2: Sig rate by gene type ---
    ax = axes[1]
    ag = pd.read_parquet(MERGED_DIR / "gene_scores.parquet")
    twas = pd.read_parquet(MERGED_DIR / "twas_all.parquet")
    shared_traits = set(ag['accession'].unique()) & set(twas['accession'].unique())
    shared_tissues = set(ag['tissue'].unique()) & set(twas['tissue'].unique())
    ag = ag[ag['accession'].isin(shared_traits) & ag['tissue'].isin(shared_tissues)]
    twas = twas[twas['accession'].isin(shared_traits) & twas['tissue'].isin(shared_tissues)]

    gene_info = pd.read_parquet(OUT_DIR / "gene_info.parquet")

    ag_gene_sig = ag.groupby('gene_id')['max_lfc'].agg(lambda x: (x.abs() > 0.01).any()).reset_index()
    ag_gene_sig.columns = ['gene_id', 'ag_sig']
    twas_gene_sig = twas.groupby('gene_id')['twas_sig_bonf'].any().reset_index()
    twas_gene_sig.columns = ['gene_id', 'twas_sig']

    gene_info = gene_info.merge(ag_gene_sig, on='gene_id', how='left')
    gene_info = gene_info.merge(twas_gene_sig, on='gene_id', how='left')
    gene_info['ag_sig'] = gene_info['ag_sig'].fillna(False).astype(bool)
    gene_info['twas_sig'] = gene_info['twas_sig'].fillna(False).astype(bool)

    cats = ['protein_coding', 'lncRNA', 'pseudogene', 'miRNA', 'snRNA', 'snoRNA']
    cats = [c for c in cats if c in gene_info['category'].values]

    y_pos = np.arange(len(cats))
    w = 0.25
    for i, (s, sig_col, color, label) in enumerate([
        ('AG_only', 'ag_sig', C_AG, 'AG-only (AG sig)'),
        ('shared', 'ag_sig', C_BOTH, 'Shared (AG sig)'),
        ('TWAS_only', 'twas_sig', C_TWAS, 'TWAS-only (TWAS sig)'),
    ]):
        vals = []
        for c in cats:
            sub = gene_info[(gene_info['category'] == c) & (gene_info['set'] == s)]
            vals.append(sub[sig_col].mean() * 100 if len(sub) > 0 else 0)
        ax.barh(y_pos + (i - 1) * w, vals, height=w, color=color, label=label)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cats)
    ax.set_xlabel("% genes ever significant")
    ax.legend(frameon=False, fontsize=5.5, loc='lower right')
    ax.set_title("Significance rate by gene type", fontweight='bold')
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(OUT_DIR / "panel_c.png")
    fig.savefig(OUT_DIR / "panel_c.pdf")
    plt.close()
    print("Saved panel_c")


if __name__ == "__main__":
    main()
