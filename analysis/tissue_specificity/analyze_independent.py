#!/usr/bin/env python3
"""Fully independent tau distributions for AG and GTEx.

AG tau:   per (rsid, gene_id), over all tissues where |raw_score| > AG_THRESH (>=2 tissues).
GTEx tau: per (chr,pos,gene_id), over all tissues where eQTL is in significant-set (>=2 tissues).

No mutual filtering between the two sets.
"""
from pathlib import Path
import os as _os
ROOT = _os.environ.get("AG_LD_ROOT", _os.getcwd())

import numpy as np
import pandas as pd

OUT = Path(ROOT) / "insights/tissue_specificity"
AG_CONS  = Path("/media/brian/TOSHIBA EXT/AG/ag_scores_consolidated.parquet")
GTEX_SIG = Path(ROOT) / "validation_data/gtex_eqtls.parquet"
AG_THRESH   = 0.01
MIN_TISSUES = 2


def tau_vectorized(df: pd.DataFrame, key_cols, val_col: str, name: str) -> pd.DataFrame:
    """Vectorized tau per group: sum(1 - |x|/max|x|) / (n-1)."""
    df = df.copy()
    df["abs_v"] = df[val_col].abs()
    g = df.groupby(key_cols, sort=False, observed=True)
    agg = g["abs_v"].agg(["max", "sum", "count"])
    agg = agg[agg["count"] >= MIN_TISSUES]
    agg = agg[agg["max"] > 0]
    # sum(1 - |x|/max) = n - sum/max  →  tau = (n - sum/max) / (n-1)
    agg[name] = (agg["count"] - agg["sum"] / agg["max"]) / (agg["count"] - 1)
    agg = agg.rename(columns={"count": "n_tissues"})
    return agg[["n_tissues", name]].reset_index()


def build_ag():
    cols = ["rsid", "gene_id", "gtex_tissue", "raw_score"]
    df = pd.read_parquet(AG_CONS, columns=cols)
    df = df[df["gtex_tissue"].astype(str) != ""]
    df = df[df["raw_score"].abs() > AG_THRESH]
    return tau_vectorized(df, ["rsid", "gene_id"], "raw_score", "ag_tau") \
             .rename(columns={"gene_id": "gene_id_stable"})


def build_gtex():
    df = pd.read_parquet(GTEX_SIG, columns=["chr", "pos", "gene_id_stable", "tissue", "beta"])
    return tau_vectorized(df, ["chr", "pos", "gene_id_stable"], "beta", "gtex_tau")


def main():
    print("Building AG-independent tau...", flush=True)
    ag = build_ag()
    ag.to_parquet(OUT / "tau_ag_independent.parquet", index=False)
    print(f"  AG pairs: {len(ag):,}  median tau={ag['ag_tau'].median():.3f}")

    print("Building GTEx-independent tau...", flush=True)
    gt = build_gtex()
    gt.to_parquet(OUT / "tau_gtex_independent.parquet", index=False)
    print(f"  GTEx pairs: {len(gt):,}  median tau={gt['gtex_tau'].median():.3f}")


if __name__ == "__main__":
    main()
