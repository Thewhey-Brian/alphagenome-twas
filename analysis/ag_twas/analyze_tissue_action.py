#!/usr/bin/env python3
"""Tissue-of-action agreement between AG and TWAS for jointly significant genes."""

import logging
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/insights/ag_twas")
MERGED = OUT_DIR / "merged" / "ag_twas_merged.parquet"


def main():
    log.info("Loading merged data")
    df = pd.read_parquet(MERGED)
    df = df[df["_merge"] == "both"].copy()
    log.info(f"Both-method rows: {len(df):,}")

    # Step 1: jointly significant triplets
    df["abs_max_lfc"] = df["max_lfc"].abs()
    df["abs_twas_z"] = df["TWAS.Z"].abs()
    sig = df[(df["abs_max_lfc"] > 0.01) & (df["twas_sig_bonf"] == True)].copy()
    log.info(f"Jointly significant triplets: {len(sig):,}")

    # Step 2: pairs with >=2 tissues
    pair_cols = ["gene_id", "accession"]
    tissue_counts = sig.groupby(pair_cols)["tissue"].nunique().rename("n_tissues")
    multi = tissue_counts[tissue_counts >= 2].reset_index()
    sig_multi = sig.merge(multi[pair_cols], on=pair_cols)
    log.info(f"Pairs with >=2 tissues: {len(multi):,}")

    # Per-pair: top tissue for AG and TWAS, agreement, profile correlation
    records = []
    for (gene_id, accession), grp in sig_multi.groupby(pair_cols):
        grp_dedup = grp.drop_duplicates(subset="tissue")
        n_tissues = grp_dedup["tissue"].nunique()
        ag_top = grp_dedup.loc[grp_dedup["abs_max_lfc"].idxmax(), "tissue"]
        twas_top = grp_dedup.loc[grp_dedup["abs_twas_z"].idxmax(), "tissue"]
        agree = ag_top == twas_top

        rho = np.nan
        if n_tissues >= 3:
            r, _ = spearmanr(grp_dedup["abs_max_lfc"].values, grp_dedup["abs_twas_z"].values)
            rho = r

        records.append({
            "gene_id": gene_id,
            "accession": accession,
            "n_tissues": n_tissues,
            "ag_top_tissue": ag_top,
            "twas_top_tissue": twas_top,
            "agree": agree,
            "spearman_rho": rho,
        })

    pairs = pd.DataFrame(records)
    log.info(f"Computed metrics for {len(pairs):,} pairs")

    # Step 3-4: overall agreement and chance
    overall_agree = pairs["agree"].mean()
    overall_chance = (1.0 / pairs["n_tissues"]).mean()
    rho_subset = pairs["spearman_rho"].dropna()
    overall_rho = rho_subset.mean() if len(rho_subset) > 0 else np.nan
    median_rho = rho_subset.median() if len(rho_subset) > 0 else np.nan

    summary = pd.DataFrame([{
        "n_pairs": len(pairs),
        "agreement_rate": overall_agree,
        "chance_rate": overall_chance,
        "enrichment": overall_agree / overall_chance if overall_chance > 0 else np.nan,
        "mean_spearman_rho": overall_rho,
        "median_spearman_rho": median_rho,
        "n_pairs_rho": len(rho_subset),
    }])
    log.info(f"Overall agreement: {overall_agree:.3f}, chance: {overall_chance:.3f}, "
             f"enrichment: {overall_agree/overall_chance:.2f}x")

    # Step 5: stratify by n_tissues
    bins = pd.cut(pairs["n_tissues"], bins=[1, 2, 5, 10, 100], labels=["2", "3-5", "6-10", "11+"])
    strat = pairs.groupby(bins, observed=False).agg(
        n_pairs=("agree", "size"),
        agreement_rate=("agree", "mean"),
        chance_rate=("n_tissues", lambda x: (1.0 / x).mean()),
        mean_spearman_rho=("spearman_rho", lambda x: x.dropna().mean() if x.dropna().size > 0 else np.nan),
    ).reset_index().rename(columns={"n_tissues": "n_tissues_bin"})
    strat["enrichment"] = strat["agreement_rate"] / strat["chance_rate"]

    log.info("Stratified results:\n" + strat.to_string(index=False))

    # Save
    summary.to_csv(OUT_DIR / "tissue_action_summary.csv", index=False)
    strat.to_csv(OUT_DIR / "tissue_action_by_ntissues.csv", index=False)
    pairs.to_parquet(OUT_DIR / "tissue_action_pairs.parquet", index=False)
    log.info("Saved outputs")


if __name__ == "__main__":
    main()
