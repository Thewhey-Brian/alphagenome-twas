#!/usr/bin/env python3
"""Panel C: Overall correlation — joint vs per-SNP max by threshold."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from plot_common import apply_style, C_JOINT

apply_style()
OUT_DIR = Path("results/insights/persnp_vs_joint")


def main():
    thresh = pd.read_csv(OUT_DIR / "correlation_by_threshold.csv")

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    # Spearman rho by threshold
    ax = axes[0]
    ax.plot(thresh["threshold"], thresh["spearman_rho"], "o-", color=C_JOINT, ms=4)
    ax.set_xlabel("|Score| threshold")
    ax.set_ylabel("Spearman ρ")
    ax.set_xscale("log")
    ax.set_title("Correlation by threshold")

    # Sign agreement by threshold
    ax = axes[1]
    ax.plot(thresh["threshold"], thresh["sign_agreement"] * 100, "o-", color=C_JOINT, ms=4)
    ax.set_xlabel("|Score| threshold")
    ax.set_ylabel("Sign agreement (%)")
    ax.set_xscale("log")
    ax.axhline(50, color="grey", ls=":", lw=0.5)
    ax.set_title("Sign concordance by threshold")

    # Add sample sizes
    for _, row in thresh.iterrows():
        axes[0].annotate(f"n={int(row['n']):,}", (row["threshold"], row["spearman_rho"]),
                         fontsize=5, ha="center", va="bottom", xytext=(0, 3),
                         textcoords="offset points")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "panel_c.png")
    fig.savefig(OUT_DIR / "panel_c.pdf")
    plt.close()
    print("Saved panel_c")


if __name__ == "__main__":
    main()
