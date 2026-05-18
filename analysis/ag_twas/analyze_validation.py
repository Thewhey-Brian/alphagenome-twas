#!/usr/bin/env python3
"""Validate AG and TWAS discoveries against external databases.

Two independent ground truth sources (no GWAS overlap):
  1. Drug targets (phase >= 3 clinical trials, ChEMBL via OpenTargets)
  2. Non-GWAS OpenTargets evidence (literature, animal models, pathways, etc.)

Computes odds ratios of enrichment for AG-only, TWAS-only, both-sig, and
AG-only genes (those not in TWAS at all).

Produces:
  - validation_enrichment.csv   (OR / p / counts per category × ground truth)
  - validation_details.csv      (per-trait drug target ORs)

Usage:
    cd {ROOT}/Desktop/Research/ag_ld
    {ROOT}/miniconda3/envs/ag_twas/bin/python3 \
        results/insights/ag_twas/analyze_validation.py
"""

import logging
import sys
from pathlib import Path
import os as _os
ROOT = _os.environ.get("AG_LD_ROOT", _os.getcwd())


import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

OUT_DIR = Path('results/insights/ag_twas')
VAL_DIR = Path('results/validation_data')
AG_DIR = Path('/media/brian/TOSHIBA EXT/AG')
LFC_THRESHOLD = 0.01


# ── helpers ──────────────────────────────────────────────────────────────────

def load_trait_names():
    tm = pd.read_parquet(AG_DIR / 'trait_map.parquet', columns=['accession', 'trait'])
    return dict(tm.drop_duplicates('accession')[['accession', 'trait']].values)


def compute_or(sig_mask, validated_mask):
    """Odds ratio + Fisher exact (one-sided, greater)."""
    a = int((sig_mask & validated_mask).sum())
    b = int((sig_mask & ~validated_mask).sum())
    c = int((~sig_mask & validated_mask).sum())
    d = int((~sig_mask & ~validated_mask).sum())
    odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
    precision = a / (a + b) * 100 if (a + b) > 0 else 0
    return {
        'OR': odds_ratio, 'p': p_value,
        'TP': a, 'FP': b, 'FN': c, 'TN': d,
        'n_sig': a + b, 'n_validated': a + c,
        'precision': precision,
    }


def sig_label(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    trait_names = load_trait_names()

    # ── 1. Load merged AG×TWAS (gene×tissue×trait rows) ──────────────────
    log.info('Loading merged AG×TWAS data...')
    m = pd.read_parquet(OUT_DIR / 'merged' / 'ag_twas_merged.parquet')

    # Rows present in both AG and TWAS
    both = m[(m['_merge'] == 'both') & m['TWAS.Z'].notna()].copy()
    both['ag_sig'] = both['max_abs_lfc'] > LFC_THRESHOLD
    both['twas_sig'] = both['twas_sig_bonf'].fillna(False)

    # Collapse to gene×trait level (any tissue significant → sig)
    gene_trait = both.groupby(['gene_id', 'accession']).agg(
        ag_sig=('ag_sig', 'any'),
        twas_sig=('twas_sig', 'any'),
    ).reset_index()
    gene_trait['trait'] = gene_trait['accession'].map(trait_names)
    log.info('Gene×trait pairs (AG∩TWAS overlap): %s', f'{len(gene_trait):,}')

    # ── 2. AG-only genes (in AG but not TWAS) ───────────────────────────
    log.info('Loading AG-only gene scores...')
    gs = pd.read_parquet(OUT_DIR / 'merged' / 'gene_scores.parquet')
    # Genes that appear in gene_scores but NOT in the merged overlap
    overlap_keys = set(zip(both['gene_id'], both['accession']))
    gs['in_twas'] = gs.apply(lambda r: (r['gene_id'], r['accession']) in overlap_keys, axis=1)
    ag_only_genes = gs[~gs['in_twas']].copy()
    ag_only_genes['ag_sig'] = ag_only_genes['max_abs_lfc'] > LFC_THRESHOLD

    ag_only_gt = ag_only_genes.groupby(['gene_id', 'accession']).agg(
        ag_sig=('ag_sig', 'any'),
    ).reset_index()
    ag_only_gt['twas_sig'] = False  # not in TWAS by definition
    ag_only_gt['trait'] = ag_only_gt['accession'].map(trait_names)
    log.info('Gene×trait pairs (AG-only, no TWAS): %s', f'{len(ag_only_gt):,}')

    # ── 3. Load ground truth ─────────────────────────────────────────────
    drug_gt = pd.read_parquet(VAL_DIR / 'drug_targets_ground_truth.parquet')
    drug_set = set(zip(drug_gt['gene_id'], drug_gt['accession']))
    drug_traits = set(drug_gt['accession'])

    ot_ng = pd.read_parquet(VAL_DIR / 'opentargets_nongwas_validation.parquet')
    ot_set = set(zip(ot_ng['gene_id'], ot_ng['accession']))
    ot_traits = set(ot_ng['accession'])

    log.info('Drug target GT: %d pairs across %d traits', len(drug_set), len(drug_traits))
    log.info('Non-GWAS OT GT: %d pairs across %d traits', len(ot_set), len(ot_traits))

    # ── 4. Attach validation labels ──────────────────────────────────────
    def attach_val(df, val_set, val_traits):
        sub = df[df['accession'].isin(val_traits)].copy()
        sub['validated'] = sub.apply(
            lambda r: (r['gene_id'], r['accession']) in val_set, axis=1)
        return sub

    gt_drug_overlap = attach_val(gene_trait, drug_set, drug_traits)
    gt_ot_overlap = attach_val(gene_trait, ot_set, ot_traits)
    gt_drug_agonly = attach_val(ag_only_gt, drug_set, drug_traits)
    gt_ot_agonly = attach_val(ag_only_gt, ot_set, ot_traits)

    log.info('Drug — overlap pairs: %s (validated %d)',
             f'{len(gt_drug_overlap):,}', gt_drug_overlap['validated'].sum())
    log.info('Drug — AG-only pairs: %s (validated %d)',
             f'{len(gt_drug_agonly):,}', gt_drug_agonly['validated'].sum())
    log.info('OT   — overlap pairs: %s (validated %d)',
             f'{len(gt_ot_overlap):,}', gt_ot_overlap['validated'].sum())
    log.info('OT   — AG-only pairs: %s (validated %d)',
             f'{len(gt_ot_agonly):,}', gt_ot_agonly['validated'].sum())

    # ── 5. Compute enrichment ORs ────────────────────────────────────────
    categories_overlap = [
        ('both_sig',      lambda d: d['ag_sig'] & d['twas_sig']),
        ('ag_only_sig',   lambda d: d['ag_sig'] & ~d['twas_sig']),
        ('twas_only_sig', lambda d: ~d['ag_sig'] & d['twas_sig']),
        ('ag_all',        lambda d: d['ag_sig']),
        ('twas_all',      lambda d: d['twas_sig']),
    ]

    categories_agonly = [
        ('ag_exclusive_sig', lambda d: d['ag_sig']),
    ]

    results = []

    # Overlap categories
    for label, mask_fn in categories_overlap:
        for gt_name, gt_df in [('drug_targets', gt_drug_overlap),
                                ('nongwas_ot', gt_ot_overlap)]:
            if len(gt_df) == 0:
                continue
            r = compute_or(mask_fn(gt_df), gt_df['validated'])
            r['category'] = label
            r['ground_truth'] = gt_name
            r['universe'] = 'ag_twas_overlap'
            results.append(r)
            log.info('  %s × %s: OR=%.2f (%s), TP=%d, precision=%.1f%%',
                     gt_name, label, r['OR'], sig_label(r['p']),
                     r['TP'], r['precision'])

    # AG-exclusive (genes not in TWAS at all)
    for label, mask_fn in categories_agonly:
        for gt_name, gt_df in [('drug_targets', gt_drug_agonly),
                                ('nongwas_ot', gt_ot_agonly)]:
            if len(gt_df) == 0:
                continue
            r = compute_or(mask_fn(gt_df), gt_df['validated'])
            r['category'] = label
            r['ground_truth'] = gt_name
            r['universe'] = 'ag_only_genes'
            results.append(r)
            log.info('  %s × %s: OR=%.2f (%s), TP=%d, precision=%.1f%%',
                     gt_name, label, r['OR'], sig_label(r['p']),
                     r['TP'], r['precision'])

    enrich_df = pd.DataFrame(results)
    enrich_df['sig_label'] = enrich_df['p'].apply(sig_label)
    enrich_path = OUT_DIR / 'validation_enrichment.csv'
    enrich_df.to_csv(enrich_path, index=False)
    log.info('Saved %s (%d rows)', enrich_path, len(enrich_df))

    # ── 6. Per-trait drug target ORs (detail table) ──────────────────────
    trait_rows = []
    for acc in sorted(drug_traits):
        sub = gt_drug_overlap[gt_drug_overlap['accession'] == acc]
        if sub['validated'].sum() < 2:
            continue
        for label, mask_fn in [('ag_all', lambda d: d['ag_sig']),
                                ('twas_all', lambda d: d['twas_sig']),
                                ('both_sig', lambda d: d['ag_sig'] & d['twas_sig'])]:
            r = compute_or(mask_fn(sub), sub['validated'])
            trait_rows.append({
                'accession': acc,
                'trait': trait_names.get(acc, acc),
                'category': label,
                'OR': r['OR'],
                'p': r['p'],
                'sig_label': sig_label(r['p']),
                'TP': r['TP'],
                'n_sig': r['n_sig'],
                'n_validated': r['n_validated'],
                'n_total': len(sub),
                'precision': r['precision'],
            })

    detail_df = pd.DataFrame(trait_rows)
    detail_path = OUT_DIR / 'validation_details.csv'
    detail_df.to_csv(detail_path, index=False)
    log.info('Saved %s (%d rows)', detail_path, len(detail_df))

    log.info('Done.')


if __name__ == '__main__':
    main()
