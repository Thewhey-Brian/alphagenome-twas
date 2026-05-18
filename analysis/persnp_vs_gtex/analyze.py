#!/usr/bin/env python3
"""Per-SNP AG score validation against GTEx eQTLs.

Compares AG raw_score to GTEx effect size (beta) at matched
SNP × gene × tissue triplets.

Usage:
    python results/insights/persnp_vs_gtex/analyze.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/insights/persnp_vs_gtex")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use archived matched cache
MATCH_CACHE = Path("results/figures/_archive_v1/fig1_validation/ag_gtex_matched.parquet")


def main():
    m = pd.read_parquet(MATCH_CACHE)
    log.info("Loaded %d matched AG×GTEx triplets", len(m))
    log.info("  Unique SNPs: %d, genes: %d, tissues: %d",
             m["rsid"].nunique(), m["gene_id_stable"].nunique(), m["gtex_tissue"].nunique())

    m["abs_ag"] = m["raw_score"].abs()
    m["abs_gtex"] = m["beta"].abs()
    m["sign_agree"] = np.sign(m["raw_score"]) == np.sign(m["beta"])
    m["log10_pval"] = -np.log10(m["pval"].clip(lower=1e-300))

    # === Overall stats ===
    r_all = np.corrcoef(m["raw_score"], m["beta"])[0, 1]
    rho_all, _ = stats.spearmanr(m["raw_score"], m["beta"])
    sign_all = m["sign_agree"].mean()
    log.info("Overall: r=%.4f, ρ=%.4f, sign=%.1f%%", r_all, rho_all, sign_all * 100)

    # === Sign concordance and correlation by |AG score| threshold ===
    thresholds = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    thresh_results = []
    for t in thresholds:
        sub = m[m["abs_ag"] >= t]
        if len(sub) < 100:
            continue
        rho, pval_rho = stats.spearmanr(sub["raw_score"], sub["beta"])
        thresh_results.append({
            "threshold": t, "n": len(sub),
            "sign_concordance": sub["sign_agree"].mean(),
            "spearman_rho": rho, "spearman_pval": pval_rho,
            "pearson_r": np.corrcoef(sub["raw_score"], sub["beta"])[0, 1],
        })
        log.info("  |AG|≥%.3f: n=%d, sign=%.1f%%, ρ=%.3f",
                 t, len(sub), sub["sign_agree"].mean() * 100, rho)
    pd.DataFrame(thresh_results).to_csv(OUT_DIR / "threshold_stats.csv", index=False)

    # === By AG score decile ===
    m["ag_decile"] = pd.qcut(m["abs_ag"], 10, labels=False, duplicates="drop")
    decile_results = []
    for dec in sorted(m["ag_decile"].unique()):
        sub = m[m["ag_decile"] == dec]
        rho, _ = stats.spearmanr(sub["raw_score"], sub["beta"])
        decile_results.append({
            "decile": dec,
            "abs_ag_mean": sub["abs_ag"].mean(),
            "n": len(sub),
            "sign_concordance": sub["sign_agree"].mean(),
            "spearman_rho": rho,
            "pearson_r": np.corrcoef(sub["raw_score"], sub["beta"])[0, 1],
        })
    pd.DataFrame(decile_results).to_csv(OUT_DIR / "decile_stats.csv", index=False)

    # === AG score vs GTEx significance (quantile bins for even n) ===
    m["pval_bin"] = pd.qcut(m["log10_pval"], 15, labels=False, duplicates="drop")
    pval_results = []
    for binlabel, sub in m.groupby("pval_bin", observed=True):
        pval_results.append({
            "pval_bin": binlabel,
            "log10p_mean": sub["log10_pval"].mean(),
            "log10p_median": sub["log10_pval"].median(),
            "mean_abs_ag": sub["abs_ag"].mean(),
            "median_abs_ag": sub["abs_ag"].median(),
            "max_abs_ag": sub["abs_ag"].max(),
            "p75_abs_ag": sub["abs_ag"].quantile(0.75),
            "p95_abs_ag": sub["abs_ag"].quantile(0.95),
            "n": len(sub),
        })
    pd.DataFrame(pval_results).to_csv(OUT_DIR / "pval_vs_ag.csv", index=False)

    # === Per-tissue stats ===
    tissue_results = []
    for tissue, sub in m.groupby("gtex_tissue"):
        if len(sub) < 500:
            continue
        rho, _ = stats.spearmanr(sub["raw_score"], sub["beta"])
        tissue_results.append({
            "tissue": tissue,
            "n": len(sub),
            "sign_concordance": sub["sign_agree"].mean(),
            "spearman_rho": rho,
            "pearson_r": np.corrcoef(sub["raw_score"], sub["beta"])[0, 1],
        })
    tissue_df = pd.DataFrame(tissue_results).sort_values("spearman_rho", ascending=False)
    tissue_df.to_csv(OUT_DIR / "tissue_stats.csv", index=False)
    log.info("Tissue stats: %d tissues, ρ range [%.3f, %.3f]",
             len(tissue_df), tissue_df["spearman_rho"].min(), tissue_df["spearman_rho"].max())

    # === Overlap stats for Venn ===
    overlap = {
        "ag_triplets": 96_084_252, "ag_snps": 60_511, "ag_genes": 51_335, "ag_tissues": 54,
        "gtex_triplets": 71_478_479, "gtex_snps": 4_631_659, "gtex_genes": 34_548, "gtex_tissues": 49,
        "matched_triplets": len(m), "matched_snps": m["rsid"].nunique(),
        "matched_genes": m["gene_id_stable"].nunique(), "matched_tissues": m["gtex_tissue"].nunique(),
    }
    pd.Series(overlap).to_csv(OUT_DIR / "overlap_stats.csv", header=["value"])

    # === Save sample for scatter plot ===
    sample = m.sample(min(500_000, len(m)), random_state=42)
    sample[["raw_score", "beta", "abs_ag", "log10_pval", "sign_agree"]].to_parquet(
        OUT_DIR / "scatter_sample.parquet", index=False)

    log.info("Done — outputs in %s", OUT_DIR)


if __name__ == "__main__":
    main()
