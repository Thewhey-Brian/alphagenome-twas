#!/usr/bin/env python3
"""Compute validation enrichment for TWAS-exclusive genes (no AG score).

Mirrors the AG-exclusive analysis in analyze_validation.py, but for genes
that have a TWAS model but were never scored by AG.

Appends a single new category (`twas_exclusive_sig`, universe `twas_only_genes`)
to validation_enrichment.csv.
"""
from pathlib import Path
import os as _os
ROOT = _os.environ.get("AG_LD_ROOT", _os.getcwd())

import pandas as pd
from scipy.stats import fisher_exact

OUT  = Path(ROOT) / "insights/ag_twas"
VAL  = Path(ROOT) / "validation_data"

# 1. AG-scored gene set (from the AG-only gene list + the shared set)
ag_only_gids    = set(pd.read_parquet(OUT / "merged" / "ag_only_gene_ids.parquet")["gene_id"])
# Shared genes = have both AG score and TWAS model; derive from merged 'both'
merged = pd.read_parquet(OUT / "merged" / "ag_twas_merged.parquet",
                          columns=["gene_id", "accession", "tissue", "TWAS.Z",
                                   "twas_sig_bonf", "_merge"])
ag_shared_gids  = set(merged.loc[merged["_merge"] == "both", "gene_id"].unique())
ag_all_gids     = ag_only_gids | ag_shared_gids

# 2. TWAS-exclusive = TWAS-universe \ AG-universe
twas_all_gids = set(merged.loc[merged["TWAS.Z"].notna(), "gene_id"].unique())
twas_excl_gids = twas_all_gids - ag_all_gids
print(f"TWAS-exclusive gene set: {len(twas_excl_gids):,}  (expected ≈ 2,911)")

# 3. Collapse to gene × trait: twas_sig = any tissue Bonferroni sig
tx = merged[merged["gene_id"].isin(twas_excl_gids) & merged["TWAS.Z"].notna()].copy()
tx["twas_sig"] = tx["twas_sig_bonf"].fillna(False)
gt_tx = tx.groupby(["gene_id", "accession"]).agg(
    twas_sig=("twas_sig", "any")
).reset_index()
print(f"TWAS-excl gene × trait pairs: {len(gt_tx):,}")

# 4. Load ground truth
drug_gt = pd.read_parquet(VAL / "drug_targets_ground_truth.parquet")
ot_gt   = pd.read_parquet(VAL / "opentargets_nongwas_validation.parquet")
drug_set    = set(zip(drug_gt["gene_id"],  drug_gt["accession"]))
drug_traits = set(drug_gt["accession"])
ot_set      = set(zip(ot_gt["gene_id"],    ot_gt["accession"]))
ot_traits   = set(ot_gt["accession"])

def attach(df, vset, vtraits):
    sub = df[df["accession"].isin(vtraits)].copy()
    sub["validated"] = [(g, a) in vset for g, a in zip(sub["gene_id"], sub["accession"])]
    return sub

gt_drug = attach(gt_tx, drug_set, drug_traits)
gt_ot   = attach(gt_tx, ot_set,   ot_traits)

def compute_or(sig_mask, valid_mask):
    TP = int((sig_mask & valid_mask).sum())
    FP = int((sig_mask & ~valid_mask).sum())
    FN = int((~sig_mask & valid_mask).sum())
    TN = int((~sig_mask & ~valid_mask).sum())
    try:
        OR, p = fisher_exact([[TP, FP], [FN, TN]], alternative="greater")
    except Exception:
        OR, p = float("nan"), 1.0
    n_sig = TP + FP
    n_validated = TP + FN
    precision = TP / n_sig * 100 if n_sig else 0.0
    return dict(OR=OR, p=p, TP=TP, FP=FP, FN=FN, TN=TN,
                n_sig=n_sig, n_validated=n_validated, precision=precision)

def sig_label(p):
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 0.05: return "*"
    return "ns"

rows = []
for gt_name, gt_df in [("drug_targets", gt_drug), ("nongwas_ot", gt_ot)]:
    r = compute_or(gt_df["twas_sig"], gt_df["validated"])
    r.update(category="twas_exclusive_sig", ground_truth=gt_name,
             universe="twas_only_genes", sig_label=sig_label(r["p"]))
    rows.append(r)
    print(f"  {gt_name}: OR={r['OR']:.2f} ({r['sig_label']}), "
          f"TP={r['TP']}, FP={r['FP']}, n_sig={r['n_sig']}, precision={r['precision']:.1f}%")

# 5. Append to validation_enrichment.csv
enrich_path = OUT / "validation_enrichment.csv"
existing = pd.read_csv(enrich_path)
existing = existing[existing["category"] != "twas_exclusive_sig"]   # replace if re-run
new = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
new.to_csv(enrich_path, index=False)
print(f"Appended twas_exclusive_sig rows → {enrich_path}")
