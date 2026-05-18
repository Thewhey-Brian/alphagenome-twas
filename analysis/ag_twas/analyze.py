#!/usr/bin/env python3
"""AG vs TWAS: Coverage analysis.

Compares gene and triplet coverage between AG and TWAS, characterizes
AG-only genes, and saves summary tables for plotting.

Usage:
    python results/insights/ag_twas/analyze.py
"""
import logging
from pathlib import Path
import os as _os
ROOT = _os.environ.get("AG_LD_ROOT", _os.getcwd())


import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MERGED_DIR = Path("results/insights/ag_twas/merged")
OUT_DIR = Path("results/insights/ag_twas")
OUT_DIR.mkdir(parents=True, exist_ok=True)


GENCODE_GTF = Path(ROOT) / "reference/gencode.v44.annotation.gtf.gz"

# Simplify GENCODE biotypes into broader categories
BIOTYPE_MAP = {
    'protein_coding': 'protein_coding',
    'lncRNA': 'lncRNA',
    'miRNA': 'miRNA',
    'snRNA': 'snRNA',
    'snoRNA': 'snoRNA',
    'misc_RNA': 'other_ncRNA',
    'rRNA': 'other_ncRNA',
    'scaRNA': 'snoRNA',
    'TEC': 'other',
}
# All pseudogene types → pseudogene
PSEUDO_TYPES = {'processed_pseudogene', 'unprocessed_pseudogene', 'transcribed_unprocessed_pseudogene',
                'transcribed_processed_pseudogene', 'rRNA_pseudogene', 'transcribed_unitary_pseudogene',
                'unitary_pseudogene', 'polymorphic_pseudogene', 'pseudogene'}
# IG/TR types → IG/TR
IGTR_TYPES = {'IG_V_gene', 'IG_V_pseudogene', 'IG_D_gene', 'IG_J_gene', 'IG_C_gene',
              'IG_C_pseudogene', 'IG_J_pseudogene', 'TR_V_gene', 'TR_V_pseudogene',
              'TR_J_gene', 'TR_J_pseudogene', 'TR_C_gene', 'TR_D_gene'}


def load_gencode_biotypes():
    """Load gene_id → simplified biotype from GENCODE GTF."""
    import gzip
    biotypes = {}
    with gzip.open(str(GENCODE_GTF), 'rt') as f:
        for line in f:
            if line.startswith('#') or '\tgene\t' not in line:
                continue
            attrs = line.strip().split('\t')[8]
            gid = gtype = ''
            for attr in attrs.split(';'):
                attr = attr.strip()
                if attr.startswith('gene_id'):
                    gid = attr.split('"')[1].split('.')[0]
                elif attr.startswith('gene_type'):
                    gtype = attr.split('"')[1]
            if gid and gtype:
                if gtype in BIOTYPE_MAP:
                    biotypes[gid] = BIOTYPE_MAP[gtype]
                elif gtype in PSEUDO_TYPES:
                    biotypes[gid] = 'pseudogene'
                elif gtype in IGTR_TYPES:
                    biotypes[gid] = 'IG/TR'
                else:
                    biotypes[gid] = 'other'
    return biotypes


def main():
    ag = pd.read_parquet(MERGED_DIR / "gene_scores.parquet")
    twas = pd.read_parquet(MERGED_DIR / "twas_all.parquet")

    # Filter to shared traits and tissues
    shared_traits = set(ag['accession'].unique()) & set(twas['accession'].unique())
    shared_tissues = set(ag['tissue'].unique()) & set(twas['tissue'].unique())
    ag = ag[ag['accession'].isin(shared_traits) & ag['tissue'].isin(shared_tissues)]
    twas = twas[twas['accession'].isin(shared_traits) & twas['tissue'].isin(shared_tissues)]
    log.info("Shared scope: %d traits, %d tissues", len(shared_traits), len(shared_tissues))
    log.info("AG: %d rows, %d genes", len(ag), ag['gene_id'].nunique())
    log.info("TWAS: %d rows, %d genes", len(twas), twas['gene_id'].nunique())

    ag_genes = set(ag['gene_id'].unique())
    twas_genes = set(twas['gene_id'].unique())
    shared_genes = ag_genes & twas_genes
    ag_only_genes = ag_genes - twas_genes
    twas_only_genes = twas_genes - ag_genes

    # === Coverage at multiple levels ===
    ag_gt = set(zip(ag['gene_id'], ag['accession']))
    twas_gt = set(zip(twas['gene_id'], twas['accession']))
    ag_gtis = set(zip(ag['gene_id'], ag['tissue']))
    twas_gtis = set(zip(twas['gene_id'], twas['tissue']))
    ag_trip = set(zip(ag['gene_id'], ag['accession'], ag['tissue']))
    twas_trip = set(zip(twas['gene_id'], twas['accession'], twas['tissue']))

    coverage = pd.DataFrame([
        {"level": "Gene", "AG": len(ag_genes), "TWAS": len(twas_genes),
         "Shared": len(shared_genes), "AG_only": len(ag_only_genes), "TWAS_only": len(twas_only_genes)},
        {"level": "Gene × Trait", "AG": len(ag_gt), "TWAS": len(twas_gt),
         "Shared": len(ag_gt & twas_gt), "AG_only": len(ag_gt - twas_gt), "TWAS_only": len(twas_gt - ag_gt)},
        {"level": "Gene × Tissue", "AG": len(ag_gtis), "TWAS": len(twas_gtis),
         "Shared": len(ag_gtis & twas_gtis), "AG_only": len(ag_gtis - twas_gtis), "TWAS_only": len(twas_gtis - ag_gtis)},
        {"level": "Gene × Trait × Tissue", "AG": len(ag_trip), "TWAS": len(twas_trip),
         "Shared": len(ag_trip & twas_trip), "AG_only": len(ag_trip - twas_trip), "TWAS_only": len(twas_trip - ag_trip)},
    ])
    coverage.to_csv(OUT_DIR / "coverage.csv", index=False)
    log.info("Coverage:\n%s", coverage.to_string(index=False))

    # === Filtered coverage (significant only) ===
    ag_sig = ag[ag['max_lfc'].abs() > 0.01]
    twas_sig = twas[twas['twas_sig_bonf'] == True]
    ag_genes_sig = set(ag_sig['gene_id'].unique())
    twas_genes_sig = set(twas_sig['gene_id'].unique())
    ag_trip_sig = set(zip(ag_sig['gene_id'], ag_sig['accession'], ag_sig['tissue']))
    twas_trip_sig = set(zip(twas_sig['gene_id'], twas_sig['accession'], twas_sig['tissue']))

    coverage_sig = pd.DataFrame([
        {"level": "Gene (significant)", "AG": len(ag_genes_sig), "TWAS": len(twas_genes_sig),
         "Shared": len(ag_genes_sig & twas_genes_sig),
         "AG_only": len(ag_genes_sig - twas_genes_sig),
         "TWAS_only": len(twas_genes_sig - ag_genes_sig)},
        {"level": "Triplet (significant)", "AG": len(ag_trip_sig), "TWAS": len(twas_trip_sig),
         "Shared": len(ag_trip_sig & twas_trip_sig),
         "AG_only": len(ag_trip_sig - twas_trip_sig),
         "TWAS_only": len(twas_trip_sig - ag_trip_sig)},
    ])
    coverage_sig.to_csv(OUT_DIR / "coverage_sig.csv", index=False)
    log.info("Filtered coverage:\n%s", coverage_sig.to_string(index=False))

    # Breakdown: why AG-sig not TWAS-sig and vice versa
    ag_only_sig = ag_genes_sig - twas_genes_sig
    twas_only_sig = twas_genes_sig - ag_genes_sig
    breakdown = pd.DataFrame([
        {"direction": "AG-sig, not TWAS-sig", "total": len(ag_only_sig),
         "no_model": len(ag_only_sig - twas_genes), "has_model_not_sig": len(ag_only_sig & twas_genes)},
        {"direction": "TWAS-sig, not AG-sig", "total": len(twas_only_sig),
         "no_score": len(twas_only_sig - ag_genes), "has_score_not_sig": len(twas_only_sig & ag_genes)},
    ])
    breakdown.to_csv(OUT_DIR / "sig_breakdown.csv", index=False)
    log.info("Sig breakdown:\n%s", breakdown.to_string(index=False))

    # === Gene classification using GENCODE biotypes ===
    biotypes = load_gencode_biotypes()
    log.info("GENCODE biotypes loaded: %d genes", len(biotypes))

    ag_gene_info = ag.drop_duplicates('gene_id')[['gene_id', 'gene_name']].copy()
    ag_gene_info['category'] = ag_gene_info['gene_id'].map(biotypes).fillna('other')
    ag_gene_info['set'] = ag_gene_info['gene_id'].map(
        lambda g: 'shared' if g in shared_genes else 'AG_only')

    twas_only_info = twas[twas['gene_id'].isin(twas_only_genes)].drop_duplicates('gene_id')[['gene_id']].copy()
    twas_only_info['gene_name'] = None
    twas_only_info['category'] = twas_only_info['gene_id'].map(biotypes).fillna('other')
    twas_only_info['set'] = 'TWAS_only'

    gene_info = pd.concat([ag_gene_info, twas_only_info], ignore_index=True)
    gene_info.to_parquet(OUT_DIR / "gene_info.parquet", index=False)

    # Classification summary
    gene_class = gene_info.groupby(['set', 'category']).size().reset_index(name='count')
    gene_class_pivot = gene_class.pivot(index='category', columns='set', values='count').fillna(0).astype(int)
    for col in gene_class_pivot.columns:
        total = gene_class_pivot[col].sum()
        gene_class_pivot[f'{col}_pct'] = (gene_class_pivot[col] / total * 100).round(1)
    gene_class_pivot.to_csv(OUT_DIR / "gene_classification.csv")
    log.info("Gene classification:\n%s", gene_class_pivot.to_string())

    # === AG score comparison: AG-only vs shared genes ===
    ag_ao = ag[ag['gene_id'].isin(ag_only_genes)]
    ag_sh = ag[ag['gene_id'].isin(shared_genes)]

    score_comp = pd.DataFrame([
        {"set": "AG_only", "n_genes": len(ag_only_genes),
         "mean_abs_lfc": ag_ao['max_lfc'].abs().mean(),
         "median_abs_lfc": ag_ao['max_lfc'].abs().median(),
         "frac_sig": (ag_ao['max_lfc'].abs() > 0.01).mean(),
         "n_sig_genes": ag_ao[ag_ao['max_lfc'].abs() > 0.01]['gene_id'].nunique()},
        {"set": "Shared", "n_genes": len(shared_genes),
         "mean_abs_lfc": ag_sh['max_lfc'].abs().mean(),
         "median_abs_lfc": ag_sh['max_lfc'].abs().median(),
         "frac_sig": (ag_sh['max_lfc'].abs() > 0.01).mean(),
         "n_sig_genes": ag_sh[ag_sh['max_lfc'].abs() > 0.01]['gene_id'].nunique()},
    ])
    score_comp.to_csv(OUT_DIR / "score_comparison.csv", index=False)
    log.info("Score comparison:\n%s", score_comp.to_string(index=False))

    # === Top AG-only discoveries ===
    ao_sig = ag_ao[ag_ao['max_lfc'].abs() > 0.01]
    top_genes = ao_sig.groupby(['gene_id', 'gene_name']).agg(
        max_lfc=('max_lfc', lambda x: x.iloc[x.abs().argmax()]),
        n_traits=('accession', 'nunique'),
        n_tissues=('tissue', 'nunique'),
    ).reset_index()
    top_genes['abs_lfc'] = top_genes['max_lfc'].abs()
    top_genes['category'] = top_genes['gene_id'].map(biotypes).fillna('other')
    top_genes = top_genes.sort_values('abs_lfc', ascending=False)
    top_genes.to_csv(OUT_DIR / "ag_only_top_genes.csv", index=False)
    log.info("AG-only sig genes: %d, top: %s (|LFC|=%.3f)",
             len(top_genes), top_genes.iloc[0]['gene_name'], top_genes.iloc[0]['abs_lfc'])

    log.info("Done — outputs in %s", OUT_DIR)


if __name__ == "__main__":
    main()
