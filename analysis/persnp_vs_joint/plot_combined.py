#!/usr/bin/env python3
"""Combined figure: Per-SNP max vs Joint scoring comparison."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from plot_common import apply_style, C_JOINT, C_PERSNP

apply_style()
OUT_DIR = Path("results/insights/persnp_vs_joint")


def main():
    pleio = pd.read_parquet(OUT_DIR / "pleiotropy_comparison.parquet")
    tau_df = pd.read_parquet(OUT_DIR / "tissue_specificity.parquet")
    thresh = pd.read_csv(OUT_DIR / "correlation_by_threshold.csv")

    joint_p = pleio[pleio["method"] == "joint"]
    persnp_p = pleio[pleio["method"] == "persnp_max"]
    valid = tau_df.dropna(subset=["joint_tau", "persnp_tau"])

    fig, axes = plt.subplots(2, 3, figsize=(8, 5))

    # --- Row 1: Pleiotropy ---
    # A: n_genes
    ax = axes[0, 0]
    bins = np.arange(0, joint_p["n_genes"].quantile(0.99) + 2) - 0.5
    ax.hist(persnp_p["n_genes"], bins=bins, alpha=0.6, color=C_PERSNP,
            label=f"Per-SNP max (n={len(persnp_p)})", density=True)
    ax.hist(joint_p["n_genes"], bins=bins, alpha=0.6, color=C_JOINT,
            label=f"Joint (n={len(joint_p)})", density=True)
    ax.set_xlabel("Genes per locus")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("a  Gene pleiotropy", loc="left", fontweight="bold")

    # B: n_tissues
    ax = axes[0, 1]
    bins = np.arange(0, 56) - 0.5
    ax.hist(persnp_p["n_tissues"], bins=bins, alpha=0.6, color=C_PERSNP, density=True)
    ax.hist(joint_p["n_tissues"], bins=bins, alpha=0.6, color=C_JOINT, density=True)
    ax.set_xlabel("Tissues per locus")
    ax.set_ylabel("Density")
    ax.set_title("b  Tissue pleiotropy", loc="left", fontweight="bold")

    # C: Correlation by threshold
    ax = axes[0, 2]
    ax.plot(thresh["threshold"], thresh["spearman_rho"], "o-", color=C_JOINT, ms=3, label="Spearman ρ")
    ax2 = ax.twinx()
    ax2.plot(thresh["threshold"], thresh["sign_agreement"] * 100, "s--", color=C_PERSNP, ms=3, label="Sign %")
    ax.set_xlabel("|Score| threshold")
    ax.set_ylabel("Spearman ρ", color=C_JOINT)
    ax2.set_ylabel("Sign agreement (%)", color=C_PERSNP)
    ax.set_xscale("log")
    ax2.axhline(50, color="grey", ls=":", lw=0.5)
    ax.set_title("c  Joint vs per-SNP correlation", loc="left", fontweight="bold")

    # --- Row 2: Tissue specificity ---
    # D: Tau distributions
    ax = axes[1, 0]
    bins = np.linspace(0, 1, 50)
    ax.hist(valid["persnp_tau"], bins=bins, alpha=0.6, color=C_PERSNP,
            label="Per-SNP max", density=True)
    ax.hist(valid["joint_tau"], bins=bins, alpha=0.6, color=C_JOINT,
            label="Joint", density=True)
    ax.set_xlabel("Tau index")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("d  Tissue specificity", loc="left", fontweight="bold")

    # E: Tau scatter
    ax = axes[1, 1]
    sample = valid.sample(min(20000, len(valid)), random_state=42)
    ax.scatter(sample["persnp_tau"], sample["joint_tau"], s=1, alpha=0.1, c="grey")
    ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.5)
    ax.set_xlabel("Per-SNP max tau")
    ax.set_ylabel("Joint tau")
    rho, _ = spearmanr(valid["joint_tau"], valid["persnp_tau"])
    ax.text(0.05, 0.92, f"ρ = {rho:.3f}", transform=ax.transAxes, fontsize=7)
    ax.set_title("e  Tau agreement", loc="left", fontweight="bold")

    # F: Tissue agreement by threshold
    ax = axes[1, 2]
    tissue_thresh = pd.read_csv(OUT_DIR / "tissue_agreement_by_threshold.csv")
    tt = tissue_thresh[tissue_thresh["threshold"] > 0]  # skip 0 threshold
    ax.plot(tt["threshold"], tt["top_tissue_agree"], "o-", color=C_JOINT, ms=3, label="Top-tissue match")
    ax.plot(tt["threshold"], tt["profile_rho_median"], "s-", color=C_PERSNP, ms=3, label="Tissue profile ρ")
    ax.set_xlabel("|Score| threshold")
    ax.set_xscale("log")
    ax.set_ylabel("Agreement (proportion)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.set_title("f  Tissue agreement by confidence", loc="left", fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "persnp_vs_joint.png")
    fig.savefig(OUT_DIR / "persnp_vs_joint.pdf")
    plt.close()
    print("Saved persnp_vs_joint combined figure")


if __name__ == "__main__":
    main()
