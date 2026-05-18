#!/usr/bin/env python3
"""AG vs TWAS: Concordance analysis on shared triplets.

Compares AG and TWAS significance, ranking, and effect direction
on the 2.2M matched (gene, trait, tissue) triplets.

Usage:
    python results/insights/ag_twas/analyze_concordance.py
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MERGED_DIR = Path("results/insights/ag_twas/merged")
OUT_DIR = Path("results/insights/ag_twas")


def main():
    m = pd.read_parquet(MERGED_DIR / "ag_twas_merged.parquet")
    both = m[m['_merge'] == 'both'].copy()
    log.info("Matched triplets: %d (%d genes, %d traits, %d tissues)",
             len(both), both['gene_id'].nunique(),
             both['accession'].nunique(), both['tissue'].nunique())

    ag_sig = both['max_lfc'].abs() > 0.01
    twas_sig = both['twas_sig_bonf']

    # === 2×2 significance table ===
    a = (ag_sig & twas_sig).sum()
    b = (ag_sig & ~twas_sig).sum()
    c = (~ag_sig & twas_sig).sum()
    d = (~ag_sig & ~twas_sig).sum()
    or_val = (a * d) / (b * c) if b * c > 0 else float('inf')
    # Fisher's exact test (one-sided, greater) for enrichment of co-prioritization
    fisher_or, fisher_p = stats.fisher_exact([[a, b], [c, d]], alternative='greater')

    contingency = pd.DataFrame({
        'TWAS-': [d, b],
        'TWAS+': [c, a],
    }, index=['AG-', 'AG+'])
    contingency.to_csv(OUT_DIR / "concordance_2x2.csv")

    log.info("=== 2×2 table (AG |LFC|>0.01, TWAS Bonferroni) ===")
    log.info("\n%s", contingency.to_string())
    log.info("OR = %.2f (Fisher OR = %.2f, p = %.3g, one-sided)", or_val, fisher_or, fisher_p)
    log.info("AG sig: %d (%.1f%%), TWAS sig: %d (%.1f%%)",
             ag_sig.sum(), ag_sig.mean() * 100, twas_sig.sum(), twas_sig.mean() * 100)
    log.info("AG→TWAS confirmation: %.1f%%, TWAS→AG confirmation: %.1f%%",
             a / (a + b) * 100, a / (a + c) * 100)

    summary = {
        'n_matched': len(both),
        'ag_sig': int(ag_sig.sum()), 'ag_sig_pct': ag_sig.mean() * 100,
        'twas_sig': int(twas_sig.sum()), 'twas_sig_pct': twas_sig.mean() * 100,
        'both_sig': int(a), 'ag_only_sig': int(b), 'twas_only_sig': int(c), 'neither': int(d),
        'OR': or_val,
        'fisher_OR': fisher_or,
        'fisher_p_one_sided': fisher_p,
        'ag_to_twas_confirm': a / (a + b) * 100,
        'twas_to_ag_confirm': a / (a + c) * 100,
    }

    # === Effect direction agreement ===
    # Among jointly significant
    js = both[ag_sig & twas_sig].copy()
    if len(js) > 0:
        sign_agree = (np.sign(js['max_lfc']) == np.sign(js['TWAS.Z'])).mean()
        log.info("=== Effect direction (jointly significant, n=%d) ===", len(js))
        log.info("  Sign agreement: %.1f%%", sign_agree * 100)
        summary['sign_agree_both_sig'] = sign_agree * 100

    # Among all with either sig
    either = both[ag_sig | twas_sig].copy()
    sign_agree_either = (np.sign(either['max_lfc']) == np.sign(either['TWAS.Z'])).mean()
    log.info("  Sign agreement (either sig, n=%d): %.1f%%", len(either), sign_agree_either * 100)
    summary['sign_agree_either_sig'] = sign_agree_either * 100

    # === Ranking correlation ===
    log.info("=== Ranking correlation ===")

    # Overall
    rho_all, p_all = stats.spearmanr(both['max_lfc'].abs(), both['TWAS.Z'].abs())
    log.info("  All triplets (n=%d): rho=%.4f", len(both), rho_all)
    summary['rho_all'] = rho_all

    # Per trait × tissue
    rho_per_tt = []
    MIN_SHARED = 50  # minimum shared genes per trait × tissue for rank correlation
    for (acc, tis), grp in both.groupby(['accession', 'tissue']):
        if len(grp) < MIN_SHARED:
            continue
        r, p = stats.spearmanr(grp['max_lfc'].abs(), grp['TWAS.Z'].abs())
        rho_per_tt.append({'accession': acc, 'tissue': tis, 'rho': r, 'p': p, 'n': len(grp)})
    rho_df = pd.DataFrame(rho_per_tt)
    rho_df.to_csv(OUT_DIR / "concordance_rho_per_tt.csv", index=False)
    log.info("  Per trait×tissue (n=%d pairs): median rho=%.4f, %% positive=%.1f%%",
             len(rho_df), rho_df['rho'].median(), (rho_df['rho'] > 0).mean() * 100)
    summary['rho_per_tt_median'] = rho_df['rho'].median()
    summary['rho_per_tt_pct_positive'] = (rho_df['rho'] > 0).mean() * 100

    # === Effect size correlation (jointly significant) ===
    if len(js) > 0:
        rho_js, _ = stats.spearmanr(js['max_lfc'].abs(), js['TWAS.Z'].abs())
        log.info("  Jointly significant (n=%d): rho=%.4f", len(js), rho_js)
        summary['rho_jointly_sig'] = rho_js

    # === Characterize the three sig groups ===
    log.info("=== Sig group characteristics ===")
    for label, mask in [('both_sig', ag_sig & twas_sig),
                        ('ag_only_sig', ag_sig & ~twas_sig),
                        ('twas_only_sig', ~ag_sig & twas_sig)]:
        sub = both[mask]
        log.info("  %s (n=%d):", label, len(sub))
        log.info("    |max_lfc| median=%.4f, |TWAS.Z| median=%.2f",
                 sub['max_lfc'].abs().median(), sub['TWAS.Z'].abs().median())
        log.info("    |GWAS.Z| median=%.2f, eQTL R2 median=%.3f",
                 sub['BEST.GWAS.Z'].abs().median(), sub['MODELCV.R2'].median())
        log.info("    n_snps median=%.0f", sub['n_snps'].median())

    # Save group stats
    group_stats = []
    for label, mask in [('both_sig', ag_sig & twas_sig),
                        ('ag_only_sig', ag_sig & ~twas_sig),
                        ('twas_only_sig', ~ag_sig & twas_sig),
                        ('neither', ~ag_sig & ~twas_sig)]:
        sub = both[mask]
        group_stats.append({
            'group': label, 'n': len(sub),
            'abs_max_lfc_median': sub['max_lfc'].abs().median(),
            'abs_twas_z_median': sub['TWAS.Z'].abs().median(),
            'abs_gwas_z_median': sub['BEST.GWAS.Z'].abs().median(),
            'eqtl_r2_median': sub['MODELCV.R2'].median(),
            'n_snps_median': sub['n_snps'].median(),
        })
    pd.DataFrame(group_stats).to_csv(OUT_DIR / "concordance_group_stats.csv", index=False)

    # === Per-trait concordance ===
    trait_stats = []
    for acc, grp in both.groupby('accession'):
        ag_s = grp['max_lfc'].abs() > 0.01
        tw_s = grp['twas_sig_bonf']
        a_t = (ag_s & tw_s).sum()
        b_t = (ag_s & ~tw_s).sum()
        c_t = (~ag_s & tw_s).sum()
        d_t = (~ag_s & ~tw_s).sum()
        or_t = (a_t * d_t) / (b_t * c_t) if b_t * c_t > 0 else np.nan
        trait_stats.append({
            'accession': acc, 'n': len(grp),
            'both_sig': a_t, 'ag_only': b_t, 'twas_only': c_t, 'neither': d_t,
            'OR': or_t,
            'ag_sig_pct': ag_s.mean() * 100,
            'twas_sig_pct': tw_s.mean() * 100,
        })
    pd.DataFrame(trait_stats).to_csv(OUT_DIR / "concordance_per_trait.csv", index=False)

    # Save summary
    pd.Series(summary).to_csv(OUT_DIR / "concordance_summary.csv", header=['value'])

    log.info("Done — concordance outputs in %s", OUT_DIR)


if __name__ == "__main__":
    main()
