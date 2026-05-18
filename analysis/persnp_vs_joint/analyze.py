#!/usr/bin/env python3
"""Per-SNP (max) vs Joint scoring: head-to-head comparison.

Compares per-SNP max_lfc against joint (multi-variant) scoring at the
locus × gene × tissue level. Two dimensions:
  A. Pleiotropy — number of genes/tissues affected per locus
  B. Tissue specificity — tau index per (locus, gene) pair

Usage:
    python results/insights/persnp_vs_joint/analyze.py
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

AG_DIR = Path("/media/brian/TOSHIBA EXT/AG")
OUT_DIR = Path("results/insights/persnp_vs_joint")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Colliding gene names (20 names mapping to multiple ENSG IDs)
COLLISION_FILE = Path("results/insights/_archive_v1/determinism/gene_name_collisions.csv")
THRESH = 0.01  # |LFC| threshold for significance


def tau_index(values: np.ndarray) -> float:
    """Tissue specificity index (Yanai et al. 2005). 0=ubiquitous, 1=specific."""
    x = np.abs(values)
    if x.max() == 0:
        return np.nan
    return float(np.sum(1 - x / x.max()) / (len(x) - 1))


def load_data():
    """Load locus-level data with both joint and per-SNP max scores."""
    # Locus-level (has joint_lfc aka haplotype_lfc)
    locus = pd.read_parquet(AG_DIR / "hap_vs_persnp_locus.parquet")

    # Filter gene name collisions
    if COLLISION_FILE.exists():
        collisions = pd.read_csv(COLLISION_FILE)
        colliding = set(collisions["gene_name"].unique())
        locus = locus[~locus["gene_name"].isin(colliding)].copy()
        log.info("Filtered %d colliding gene names", len(colliding))

    # Load cached max per-SNP (deduplicate — cache has duplicates from chunked computation)
    max_cache = Path("results/insights/persnp_vs_haplotype/locus_max_cache.parquet")
    max_df = pd.read_parquet(max_cache)
    key = ["locus_id", "gene_name", "gtex_tissue"]
    max_df["_abs"] = max_df["max_persnp"].abs()
    max_df = max_df.sort_values("_abs", ascending=False).drop_duplicates(subset=key, keep="first").drop(columns=["_abs"])
    log.info("Max cache deduplicated: %d rows", len(max_df))

    locus = locus.merge(max_df, on=key, how="left")

    # Rename for clarity
    locus = locus.rename(columns={"haplotype_lfc": "joint_lfc"})

    log.info("Loaded %d locus × gene × tissue triples (%d loci)",
             len(locus), locus["locus_id"].nunique())
    return locus


def analyze_pleiotropy(locus):
    """Compare pleiotropy: how many genes/tissues each locus affects."""
    log.info("=== Pleiotropy comparison ===")

    results = []
    for method, col in [("joint", "joint_lfc"), ("persnp_max", "max_persnp")]:
        sig = locus[locus[col].abs() > THRESH]
        locus_stats = sig.groupby("locus_id").agg(
            n_genes=("gene_name", "nunique"),
            n_tissues=("gtex_tissue", "nunique"),
            n_triples=("gene_name", "size"),
            max_score=(col, lambda x: x.abs().max()),
        ).reset_index()
        locus_stats["method"] = method

        log.info("  %s: %d loci with signal, median genes=%.0f, median tissues=%.0f",
                 method, len(locus_stats),
                 locus_stats["n_genes"].median(),
                 locus_stats["n_tissues"].median())
        results.append(locus_stats)

    pleio = pd.concat(results, ignore_index=True)
    pleio.to_parquet(OUT_DIR / "pleiotropy_comparison.parquet", index=False)

    # Summary table
    summary = []
    for method in ["joint", "persnp_max"]:
        sub = pleio[pleio["method"] == method]
        summary.append({
            "method": method,
            "n_loci": len(sub),
            "n_genes_median": sub["n_genes"].median(),
            "n_genes_mean": sub["n_genes"].mean(),
            "n_tissues_median": sub["n_tissues"].median(),
            "n_tissues_mean": sub["n_tissues"].mean(),
            "n_triples_median": sub["n_triples"].median(),
            "n_triples_mean": sub["n_triples"].mean(),
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_DIR / "pleiotropy_summary.csv", index=False)
    log.info("Pleiotropy summary:\n%s", summary_df.to_string(index=False))

    # Per-locus paired comparison (loci with signal in both methods)
    joint_loci = set(pleio[pleio["method"] == "joint"]["locus_id"])
    persnp_loci = set(pleio[pleio["method"] == "persnp_max"]["locus_id"])
    both_loci = joint_loci & persnp_loci
    only_joint = joint_loci - persnp_loci
    only_persnp = persnp_loci - joint_loci

    log.info("  Loci with signal in both: %d, joint-only: %d, persnp-only: %d",
             len(both_loci), len(only_joint), len(only_persnp))

    # Paired: same locus, compare n_genes
    if both_loci:
        jp = pleio[(pleio["method"] == "joint") & (pleio["locus_id"].isin(both_loci))].set_index("locus_id")
        pp = pleio[(pleio["method"] == "persnp_max") & (pleio["locus_id"].isin(both_loci))].set_index("locus_id")
        paired = jp[["n_genes", "n_tissues"]].join(pp[["n_genes", "n_tissues"]], lsuffix="_joint", rsuffix="_persnp")
        paired["gene_ratio"] = paired["n_genes_joint"] / paired["n_genes_persnp"].clip(lower=1)
        paired["tissue_ratio"] = paired["n_tissues_joint"] / paired["n_tissues_persnp"].clip(lower=1)
        paired.to_csv(OUT_DIR / "pleiotropy_paired.csv")

        log.info("  Paired gene ratio: median=%.2f (joint/persnp)", paired["gene_ratio"].median())
        log.info("  Paired tissue ratio: median=%.2f", paired["tissue_ratio"].median())
        log.info("  Joint has MORE genes: %.1f%%", (paired["n_genes_joint"] > paired["n_genes_persnp"]).mean() * 100)

    return pleio


def analyze_tissue_specificity(locus):
    """Compare tissue specificity (tau) between joint and per-SNP max."""
    log.info("=== Tissue specificity comparison ===")

    # Need (locus, gene) pairs observed in >= 2 tissues
    pair_tissues = locus.groupby(["locus_id", "gene_name"])["gtex_tissue"].nunique()
    valid_pairs = pair_tissues[pair_tissues >= 2].index
    log.info("Pairs with >=2 tissues: %d / %d", len(valid_pairs), len(pair_tissues))

    lv = locus.set_index(["locus_id", "gene_name"])
    lv = lv.loc[lv.index.isin(valid_pairs)].reset_index()

    results = []
    for (locus_id, gene_name), grp in lv.groupby(["locus_id", "gene_name"]):
        n_tissues = len(grp)
        joint_vals = grp["joint_lfc"].values
        persnp_vals = grp["max_persnp"].values

        joint_tau = tau_index(joint_vals)
        persnp_tau = tau_index(persnp_vals)

        # Top-tissue agreement between methods
        joint_top = grp.iloc[np.abs(joint_vals).argmax()]["gtex_tissue"]
        persnp_top = grp.iloc[np.abs(persnp_vals).argmax()]["gtex_tissue"]
        top_agree = joint_top == persnp_top

        # Profile correlation between methods
        if n_tissues >= 3:
            profile_rho, profile_p = stats.spearmanr(joint_vals, persnp_vals)
        else:
            profile_rho, profile_p = np.nan, np.nan

        # Sign concordance across tissues
        sign_agree = (np.sign(joint_vals) == np.sign(persnp_vals)).mean()

        results.append({
            "locus_id": locus_id, "gene_name": gene_name,
            "n_tissues": n_tissues,
            "joint_tau": joint_tau, "persnp_tau": persnp_tau,
            "top_tissue_agree": top_agree,
            "profile_rho": profile_rho,
            "sign_concordance": sign_agree,
            "max_abs_joint": np.abs(joint_vals).max(),
            "max_abs_persnp": np.abs(persnp_vals).max(),
        })

    df = pd.DataFrame(results)
    df.to_parquet(OUT_DIR / "tissue_specificity.parquet", index=False)
    log.info("Computed tissue metrics for %d pairs", len(df))

    # Summary
    valid = df.dropna(subset=["joint_tau", "persnp_tau"])
    log.info("--- Joint tau: mean=%.3f, median=%.3f", valid["joint_tau"].mean(), valid["joint_tau"].median())
    log.info("--- Per-SNP tau: mean=%.3f, median=%.3f", valid["persnp_tau"].mean(), valid["persnp_tau"].median())

    ks_stat, ks_p = stats.ks_2samp(valid["joint_tau"], valid["persnp_tau"])
    log.info("--- KS test: D=%.3f, p=%.2e", ks_stat, ks_p)

    log.info("--- Top-tissue agreement (joint vs persnp): %.1f%%", df["top_tissue_agree"].mean() * 100)

    df3 = df[df["n_tissues"] >= 3]
    log.info("--- Profile rho (>=3 tissues, n=%d): median=%.3f, frac>0=%.1f%%",
             len(df3), df3["profile_rho"].median(), (df3["profile_rho"] > 0).mean() * 100)
    log.info("--- Sign concordance: mean=%.3f", df["sign_concordance"].mean())

    # Tissue agreement by score threshold
    thresholds = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    tissue_thresh_results = []
    for t in thresholds:
        if t == 0:
            sub = valid
        else:
            sub = valid[(valid["max_abs_joint"] >= t) | (valid["max_abs_persnp"] >= t)]
        if len(sub) < 10:
            continue
        sub3 = sub[sub["n_tissues"] >= 3]
        tissue_thresh_results.append({
            "threshold": t, "n": len(sub),
            "top_tissue_agree": sub["top_tissue_agree"].mean(),
            "profile_rho_median": sub3["profile_rho"].median() if len(sub3) > 0 else np.nan,
            "sign_concordance": sub["sign_concordance"].mean(),
            "joint_tau_median": sub["joint_tau"].median(),
            "persnp_tau_median": sub["persnp_tau"].median(),
        })
        log.info("  |score|>=%.3f (n=%d): top_agree=%.1f%%, profile_rho=%.3f, sign=%.1f%%",
                 t, len(sub), sub["top_tissue_agree"].mean() * 100,
                 sub3["profile_rho"].median() if len(sub3) > 0 else float("nan"),
                 sub["sign_concordance"].mean() * 100)
    pd.DataFrame(tissue_thresh_results).to_csv(OUT_DIR / "tissue_agreement_by_threshold.csv", index=False)

    return df


def analyze_correlation(locus):
    """Overall correlation between joint and per-SNP max scores."""
    log.info("=== Overall correlation ===")
    sub = locus.dropna(subset=["joint_lfc", "max_persnp"])
    v1, v2 = sub["joint_lfc"].values, sub["max_persnp"].values

    pr = np.corrcoef(v1, v2)[0, 1]
    sr, _ = stats.spearmanr(v1, v2)
    sa = (np.sign(v1) == np.sign(v2)).mean()

    log.info("  All triples (n=%d): r=%.3f, rho=%.3f, sign=%.1f%%", len(sub), pr, sr, sa * 100)

    # By |joint| threshold
    thresholds = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    thresh_results = []
    for t in thresholds:
        mask = (sub["joint_lfc"].abs() >= t) | (sub["max_persnp"].abs() >= t)
        s = sub[mask]
        if len(s) < 10:
            continue
        pr_t = np.corrcoef(s["joint_lfc"].values, s["max_persnp"].values)[0, 1]
        sr_t, _ = stats.spearmanr(s["joint_lfc"].values, s["max_persnp"].values)
        sa_t = (np.sign(s["joint_lfc"].values) == np.sign(s["max_persnp"].values)).mean()
        thresh_results.append({
            "threshold": t, "n": len(s),
            "pearson_r": pr_t, "spearman_rho": sr_t, "sign_agreement": sa_t,
        })
        log.info("  |score|>=%.3f (n=%d): r=%.3f, rho=%.3f, sign=%.1f%%",
                 t, len(s), pr_t, sr_t, sa_t * 100)

    pd.DataFrame(thresh_results).to_csv(OUT_DIR / "correlation_by_threshold.csv", index=False)

    # Overall stats
    pd.Series({
        "n": len(sub), "pearson_r": pr, "spearman_rho": sr, "sign_agreement": sa,
    }).to_csv(OUT_DIR / "overall_correlation.csv", header=["value"])


def main():
    locus = load_data()
    analyze_correlation(locus)
    analyze_pleiotropy(locus)
    analyze_tissue_specificity(locus)
    log.info("All done — outputs in %s", OUT_DIR)


if __name__ == "__main__":
    main()
