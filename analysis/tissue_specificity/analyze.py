#!/usr/bin/env python3
"""Tissue specificity analysis for AG×GTEx matched pairs.

Computes:
  1. Per-tissue triplet counts (|AG| ≥ 0.01) with GTEx sample sizes
  2. Per SNP×gene pair tissue counts (|AG| ≥ 0.01)
  3. Tau index (Yanai et al. 2005) for AG and GTEx across all matched pairs (≥2 tissues)
  4. AG vs GTEx agreement metrics (profile ρ, top-tissue, cosine similarity)

Usage:
    python results/insights/tissue_specificity/analyze.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/insights/tissue_specificity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MATCH_CACHE = Path("results/figures/_archive_v1/fig1_validation/ag_gtex_matched.parquet")
AG_THRESH = 0.01
MIN_TISSUES = 2


def tau_index(values: np.ndarray) -> float:
    """Tissue specificity index (Yanai et al. 2005). 0=ubiquitous, 1=specific."""
    x = np.abs(values)
    if x.max() == 0:
        return np.nan
    return float(np.sum(1 - x / x.max()) / (len(x) - 1))


def main():
    m = pd.read_parquet(MATCH_CACHE)
    log.info("Loaded %d matched AG×GTEx triplets (%d SNPs, %d genes, %d tissues)",
             len(m), m["rsid"].nunique(), m["gene_id_stable"].nunique(),
             m["gtex_tissue"].nunique())

    # ================================================================
    # Part 1: Tissue distribution (|AG| ≥ threshold)
    # ================================================================
    m01 = m[m["raw_score"].abs() >= AG_THRESH].copy()
    log.info("|AG| >= %.3f: %d triplets (%.1f%%), %d SNPs, %d genes",
             AG_THRESH, len(m01), 100 * len(m01) / len(m),
             m01["rsid"].nunique(), m01["gene_id_stable"].nunique())

    # Per-tissue counts
    tc = m01.groupby("gtex_tissue").size().reset_index(name="n_triplets")
    tc = tc.sort_values("n_triplets", ascending=False)
    gtex = pd.read_csv(OUT_DIR / "gtex_sample_sizes.csv")
    tc = tc.merge(gtex, left_on="gtex_tissue", right_on="tissue", how="left")
    tc = tc.drop(columns=["tissue"])
    tc.to_csv(OUT_DIR / "tissue_counts.csv", index=False)
    log.info("Saved tissue_counts.csv (%d tissues)", len(tc))

    # Tissue count per SNP×gene pair
    pair_tissues = m01.groupby(["rsid", "gene_id_stable"])["gtex_tissue"].nunique()
    pair_tissues = pair_tissues.reset_index(name="n_tissues")
    pair_tissues.to_parquet(OUT_DIR / "pair_tissue_counts.parquet", index=False)
    log.info("Pairs (|AG|≥%.3f): %d total, median %d tissues, mean %.1f",
             AG_THRESH, len(pair_tissues), pair_tissues["n_tissues"].median(),
             pair_tissues["n_tissues"].mean())

    # ================================================================
    # Part 2: Tau analysis (pairs with ≥2 tissues AND max|AG| > threshold)
    # ================================================================
    tissue_counts_all = m.groupby(["rsid", "gene_id_stable"])["gtex_tissue"].nunique()
    pairs_2t = set(tissue_counts_all[tissue_counts_all >= MIN_TISSUES].index)

    # Require at least one tissue with |AG| > threshold (avoid computing tau on pure noise)
    max_abs_ag = m.groupby(["rsid", "gene_id_stable"])["raw_score"].apply(lambda x: x.abs().max())
    pairs_sig = set(max_abs_ag[max_abs_ag > AG_THRESH].index)
    valid_pairs = pairs_2t & pairs_sig
    log.info("Pairs with ≥%d tissues AND max|AG|>%.3f: %d / %d",
             MIN_TISSUES, AG_THRESH, len(valid_pairs), len(tissue_counts_all))

    m_valid = m.set_index(["rsid", "gene_id_stable"])
    m_valid = m_valid.loc[m_valid.index.isin(valid_pairs)].reset_index()

    results = []
    for (rsid, gene_id), grp in m_valid.groupby(["rsid", "gene_id_stable"]):
        n_tissues = len(grp)
        ag = grp["raw_score"].values
        gtex_vals = grp["beta"].values

        ag_tau = tau_index(ag)
        gtex_tau = tau_index(gtex_vals)

        # Top-tissue agreement
        ag_top = grp.iloc[np.abs(ag).argmax()]["gtex_tissue"]
        gtex_top = grp.iloc[np.abs(gtex_vals).argmax()]["gtex_tissue"]
        top_agree = ag_top == gtex_top

        # Profile correlation
        if n_tissues >= 3:
            profile_rho, profile_p = stats.spearmanr(ag, gtex_vals)
        else:
            profile_rho, profile_p = np.nan, np.nan

        # Cosine similarity
        ag_norm = np.linalg.norm(ag)
        gtex_norm = np.linalg.norm(gtex_vals)
        if ag_norm > 0 and gtex_norm > 0:
            cosine_sim = 1 - cosine_dist(ag, gtex_vals)
        else:
            cosine_sim = np.nan

        # Sign concordance
        sign_agree = (np.sign(ag) == np.sign(gtex_vals)).mean()

        results.append({
            "rsid": rsid, "gene_id_stable": gene_id,
            "gene_name": grp["gene_name"].iloc[0],
            "n_tissues": n_tissues,
            "ag_tau": ag_tau, "gtex_tau": gtex_tau,
            "top_tissue_agree": top_agree,
            "profile_rho": profile_rho, "profile_p": profile_p,
            "cosine_sim": cosine_sim, "sign_concordance": sign_agree,
            "max_abs_ag": np.abs(ag).max(),
            "max_abs_gtex": np.abs(gtex_vals).max(),
        })

    df = pd.DataFrame(results)
    df.to_parquet(OUT_DIR / "tau_metrics.parquet", index=False)
    log.info("Computed tau metrics for %d pairs", len(df))

    # ================================================================
    # Summary
    # ================================================================
    log.info("--- AG tau: mean=%.3f, median=%.3f", df["ag_tau"].mean(), df["ag_tau"].median())
    log.info("--- GTEx tau: mean=%.3f, median=%.3f", df["gtex_tau"].mean(), df["gtex_tau"].median())

    valid_both = df.dropna(subset=["ag_tau", "gtex_tau"])
    rho, p = stats.spearmanr(valid_both["ag_tau"], valid_both["gtex_tau"])
    log.info("--- AG tau vs GTEx tau: ρ=%.4f, p=%.2e (n=%d)", rho, p, len(valid_both))

    df3 = df[df["n_tissues"] >= 3]
    log.info("--- Profile ρ (≥3 tissues, n=%d): median=%.4f, frac>0=%.3f",
             len(df3), df3["profile_rho"].median(), (df3["profile_rho"] > 0).mean())
    log.info("--- Top-tissue agreement: %.3f", df["top_tissue_agree"].mean())

    # Binned summary
    bins = [2, 3, 5, 10, 20, 50]
    df["tissue_bin"] = pd.cut(df["n_tissues"], bins=bins, right=False,
                               labels=[f"{a}-{b-1}" for a, b in zip(bins[:-1], bins[1:])])
    binned = df.groupby("tissue_bin", observed=True).agg(
        n_pairs=("ag_tau", "size"),
        ag_tau_median=("ag_tau", "median"),
        gtex_tau_median=("gtex_tau", "median"),
        top_agree=("top_tissue_agree", "mean"),
        profile_rho_median=("profile_rho", "median"),
        sign_concord=("sign_concordance", "mean"),
    ).reset_index()
    binned.to_csv(OUT_DIR / "tau_by_tissue_bin.csv", index=False)
    log.info("Binned summary:\n%s", binned.to_string(index=False))

    log.info("Done — outputs in %s", OUT_DIR)


if __name__ == "__main__":
    main()
