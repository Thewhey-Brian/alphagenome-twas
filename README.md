# alphagenome-twas

Analysis code for a systematic comparison of **AlphaGenome** (sequence-based
regulatory variant prediction) and **TWAS** for GWAS gene prioritization across
43 complex traits and 11,456 fine-mapped loci.

## Layout

```
analysis/
  persnp_vs_gtex/      AG calibration vs GTEx eQTLs              → Fig 1
  persnp_vs_joint/     Per-SNP vs joint AG scoring               → Fig 2
  tissue_specificity/  τ tissue specificity, AG vs GTEx          → Fig 3
  ag_twas/             AG vs TWAS: concordance, validation,
                       sig-groups, tissue-of-action              → Fig 4
manuscript/
  fig{1..5}_*.py       Figure-generating scripts
  fig_style.py         Shared matplotlib style
```

## Usage

```bash
export AG_LD_ROOT=/path/to/data
python analysis/<module>/analyze.py        # produces intermediate CSVs / parquets
python manuscript/fig<N>_*.py              # renders the published figure
```

Intermediate data tables and merged AG–TWAS parquet are released separately
(Zenodo DOI pending).
