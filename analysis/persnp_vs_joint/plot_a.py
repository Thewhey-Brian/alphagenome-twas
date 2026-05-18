#!/usr/bin/env python3
"""Panel A: Pleiotropy comparison — n_genes and n_tissues distributions."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plot_common import apply_style, C_JOINT, C_PERSNP

apply_style()
OUT_DIR = Path("results/insights/persnp_vs_joint")


def main():
    pleio = pd.read_parquet(OUT_DIR / "pleiotropy_comparison.parquet")
    joint = pleio[pleio["method"] == "joint"]
    persnp = pleio[pleio["method"] == "persnp_max"]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    # n_genes
    ax = axes[0]
    bins = np.arange(0, joint["n_genes"].quantile(0.99) + 2) - 0.5
    ax.hist(persnp["n_genes"], bins=bins, alpha=0.6, color=C_PERSNP,
            label=f"Per-SNP max (n={len(persnp)})", density=True)
    ax.hist(joint["n_genes"], bins=bins, alpha=0.6, color=C_JOINT,
            label=f"Joint (n={len(joint)})", density=True)
    ax.set_xlabel("Genes per locus (|LFC| > 0.01)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.set_title("Gene pleiotropy")

    # n_tissues
    ax = axes[1]
    bins = np.arange(0, 56) - 0.5
    ax.hist(persnp["n_tissues"], bins=bins, alpha=0.6, color=C_PERSNP,
            label="Per-SNP max", density=True)
    ax.hist(joint["n_tissues"], bins=bins, alpha=0.6, color=C_JOINT,
            label="Joint", density=True)
    ax.set_xlabel("Tissues per locus (|LFC| > 0.01)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.set_title("Tissue pleiotropy")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "panel_a.png")
    fig.savefig(OUT_DIR / "panel_a.pdf")
    plt.close()
    print("Saved panel_a")


if __name__ == "__main__":
    main()
