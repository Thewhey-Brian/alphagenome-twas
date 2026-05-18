#!/usr/bin/env python3
"""Panel B: Tissue specificity — tau distributions and profile agreement."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plot_common import apply_style, C_JOINT, C_PERSNP

apply_style()
OUT_DIR = Path("results/insights/persnp_vs_joint")


def main():
    df = pd.read_parquet(OUT_DIR / "tissue_specificity.parquet")
    valid = df.dropna(subset=["joint_tau", "persnp_tau"])

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))

    # Tau distributions
    ax = axes[0]
    bins = np.linspace(0, 1, 50)
    ax.hist(valid["persnp_tau"], bins=bins, alpha=0.6, color=C_PERSNP,
            label="Per-SNP max", density=True)
    ax.hist(valid["joint_tau"], bins=bins, alpha=0.6, color=C_JOINT,
            label="Joint", density=True)
    ax.set_xlabel("Tau (tissue specificity)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.set_title("Tissue specificity index")

    # Tau scatter: joint vs persnp
    ax = axes[1]
    sample = valid.sample(min(20000, len(valid)), random_state=42)
    ax.scatter(sample["persnp_tau"], sample["joint_tau"], s=1, alpha=0.1, c="grey")
    ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.5)
    ax.set_xlabel("Per-SNP max tau")
    ax.set_ylabel("Joint tau")
    ax.set_title("Tau agreement")
    from scipy.stats import spearmanr
    rho, _ = spearmanr(valid["joint_tau"], valid["persnp_tau"])
    ax.text(0.05, 0.92, f"ρ = {rho:.3f}", transform=ax.transAxes, fontsize=7)

    # Profile correlation distribution
    ax = axes[2]
    df3 = df[df["n_tissues"] >= 3].dropna(subset=["profile_rho"])
    ax.hist(df3["profile_rho"], bins=50, color="grey", alpha=0.7, density=True)
    ax.axvline(df3["profile_rho"].median(), color=C_JOINT, ls="--", lw=1,
               label=f"Median = {df3['profile_rho'].median():.3f}")
    ax.set_xlabel("Profile ρ (joint vs per-SNP max)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.set_title("Tissue profile correlation")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "panel_b.png")
    fig.savefig(OUT_DIR / "panel_b.pdf")
    plt.close()
    print("Saved panel_b")


if __name__ == "__main__":
    main()
