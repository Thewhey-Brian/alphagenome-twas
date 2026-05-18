#!/usr/bin/env python3
"""Characterize significance groups from AG vs TWAS concordance analysis.

Groups:
  both_sig:       AG |max_lfc|>0.01 AND TWAS Bonferroni sig
  ag_only_sig:    AG sig, TWAS not sig
  twas_only_sig:  TWAS sig, AG not sig

Outputs CSVs with trait enrichment, tissue enrichment, biotype composition,
and summary statistics per group.
"""

import logging
from pathlib import Path
import os as _os
ROOT = _os.environ.get("AG_LD_ROOT", _os.getcwd())


import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(ROOT) / "insights/ag_twas"
MERGED_PATH = BASE / "merged" / "ag_twas_merged.parquet"
GENE_INFO_PATH = BASE / "gene_info.parquet"

AG_THRESHOLD = 0.01


def main():
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    log.info("Loading merged data from %s", MERGED_PATH)
    df = pd.read_parquet(MERGED_PATH)
    df = df[df["_merge"] == "both"].copy()
    log.info("Rows with _merge=='both': %d", len(df))

    log.info("Loading gene info from %s", GENE_INFO_PATH)
    gene_info = pd.read_parquet(GENE_INFO_PATH)[["gene_id", "category"]]
    df = df.merge(gene_info, on="gene_id", how="left")

    # ------------------------------------------------------------------
    # Define significance groups
    # ------------------------------------------------------------------
    ag_sig = df["max_abs_lfc"] > AG_THRESHOLD
    twas_sig = df["twas_sig_bonf"]

    df["sig_group"] = "neither"
    df.loc[ag_sig & twas_sig, "sig_group"] = "both_sig"
    df.loc[ag_sig & ~twas_sig, "sig_group"] = "ag_only_sig"
    df.loc[~ag_sig & twas_sig, "sig_group"] = "twas_only_sig"

    groups = ["both_sig", "ag_only_sig", "twas_only_sig"]
    counts = df["sig_group"].value_counts()
    for g in groups:
        log.info("  %s: %d", g, counts.get(g, 0))
    log.info("  neither: %d", counts.get("neither", 0))

    dfg = df[df["sig_group"].isin(groups)]

    # ------------------------------------------------------------------
    # 1. Trait enrichment
    # ------------------------------------------------------------------
    log.info("Computing trait enrichment")
    trait_counts = (
        dfg.groupby(["sig_group", "accession"])
        .size()
        .rename("n")
        .reset_index()
    )
    trait_totals = dfg.groupby("sig_group").size().rename("group_total")
    trait_counts = trait_counts.merge(
        trait_totals, left_on="sig_group", right_index=True
    )
    trait_counts["frac"] = trait_counts["n"] / trait_counts["group_total"]

    # Also compute overall fraction per trait for comparison
    overall_frac = (
        dfg.groupby("accession").size() / len(dfg)
    ).rename("overall_frac")
    trait_counts = trait_counts.merge(
        overall_frac, left_on="accession", right_index=True
    )
    trait_counts["enrichment"] = trait_counts["frac"] / trait_counts["overall_frac"]

    trait_out = BASE / "sig_group_trait_enrichment.csv"
    trait_counts.sort_values(["sig_group", "enrichment"], ascending=[True, False]).to_csv(
        trait_out, index=False
    )
    log.info("Saved %s", trait_out)

    # ------------------------------------------------------------------
    # 2. Tissue enrichment
    # ------------------------------------------------------------------
    log.info("Computing tissue enrichment")
    tissue_counts = (
        dfg.groupby(["sig_group", "tissue"])
        .size()
        .rename("n")
        .reset_index()
    )
    tissue_counts = tissue_counts.merge(
        trait_totals, left_on="sig_group", right_index=True
    )
    tissue_counts["frac"] = tissue_counts["n"] / tissue_counts["group_total"]

    overall_tissue_frac = (
        dfg.groupby("tissue").size() / len(dfg)
    ).rename("overall_frac")
    tissue_counts = tissue_counts.merge(
        overall_tissue_frac, left_on="tissue", right_index=True
    )
    tissue_counts["enrichment"] = tissue_counts["frac"] / tissue_counts["overall_frac"]

    tissue_out = BASE / "sig_group_tissue_enrichment.csv"
    tissue_counts.sort_values(["sig_group", "enrichment"], ascending=[True, False]).to_csv(
        tissue_out, index=False
    )
    log.info("Saved %s", tissue_out)

    # ------------------------------------------------------------------
    # 3. Gene biotype composition
    # ------------------------------------------------------------------
    log.info("Computing biotype composition")
    biotype_counts = (
        dfg.groupby(["sig_group", "category"])
        .size()
        .rename("n")
        .reset_index()
    )
    biotype_totals = dfg.groupby("sig_group").size().rename("group_total")
    biotype_counts = biotype_counts.merge(
        biotype_totals, left_on="sig_group", right_index=True
    )
    biotype_counts["frac"] = biotype_counts["n"] / biotype_counts["group_total"]

    biotype_out = BASE / "sig_group_biotype.csv"
    biotype_counts.sort_values(["sig_group", "frac"], ascending=[True, False]).to_csv(
        biotype_out, index=False
    )
    log.info("Saved %s", biotype_out)

    # ------------------------------------------------------------------
    # 4-7. Summary statistics per group
    # ------------------------------------------------------------------
    log.info("Computing summary statistics per group")
    records = []
    for g in groups:
        sub = df[df["sig_group"] == g]
        rec = {"sig_group": g, "n": len(sub)}

        # GWAS Z
        gz = sub["BEST.GWAS.Z"].abs()
        rec["gwas_z_abs_median"] = gz.median()
        rec["gwas_z_abs_mean"] = gz.mean()
        rec["gwas_z_abs_q25"] = gz.quantile(0.25)
        rec["gwas_z_abs_q75"] = gz.quantile(0.75)

        # MODELCV.R2
        r2 = sub["MODELCV.R2"]
        rec["modelcv_r2_median"] = r2.median()
        rec["modelcv_r2_mean"] = r2.mean()
        rec["modelcv_r2_q25"] = r2.quantile(0.25)
        rec["modelcv_r2_q75"] = r2.quantile(0.75)

        # AG |max_lfc|
        alfc = sub["max_abs_lfc"]
        rec["max_abs_lfc_median"] = float(alfc.median())
        rec["max_abs_lfc_mean"] = float(alfc.mean())
        rec["max_abs_lfc_q25"] = float(alfc.quantile(0.25))
        rec["max_abs_lfc_q75"] = float(alfc.quantile(0.75))

        # TWAS |Z|
        tz = sub["TWAS.Z"].abs()
        rec["twas_z_abs_median"] = tz.median()
        rec["twas_z_abs_mean"] = tz.mean()
        rec["twas_z_abs_q25"] = tz.quantile(0.25)
        rec["twas_z_abs_q75"] = tz.quantile(0.75)

        records.append(rec)

    props = pd.DataFrame(records)
    props_out = BASE / "sig_group_properties.csv"
    props.to_csv(props_out, index=False)
    log.info("Saved %s", props_out)

    log.info("Done.")


if __name__ == "__main__":
    main()
