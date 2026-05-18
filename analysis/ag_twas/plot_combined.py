#!/usr/bin/env python3
"""Combined figure: AG vs TWAS complementarity analysis."""
from pathlib import Path
import subprocess
import sys

OUT_DIR = Path("results/insights/ag_twas")


def main():
    # Regenerate all panels first
    scripts = ['plot_a.py', 'plot_b.py', 'plot_c.py', 'plot_d.py', 'plot_e.py']
    for s in scripts:
        subprocess.run([sys.executable, str(OUT_DIR / s)], check=True)

    # Combine using matplotlib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from plot_common import apply_style
    apply_style()

    panels = {
        'a': OUT_DIR / 'panel_a.png',
        'b': OUT_DIR / 'panel_b.png',
        'c': OUT_DIR / 'panel_c.png',
        'd': OUT_DIR / 'panel_d.png',
        'e': OUT_DIR / 'panel_e.png',
    }

    fig = plt.figure(figsize=(14, 16))

    # Layout: 5 rows
    # Row 1: a (coverage) — full width
    # Row 2: b (gene types) + c (sig breakdown) side by side
    # Row 3-4: d (concordance 4-panel) — full width, tall
    # Row 5: e (validation) — full width

    gs = fig.add_gridspec(5, 2, height_ratios=[1.0, 1.2, 1.2, 1.2, 0.9],
                          hspace=0.3, wspace=0.05)

    def add_panel(img_path, gs_spec, label):
        ax = fig.add_subplot(gs_spec)
        img = mpimg.imread(str(img_path))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'  {label}', loc='left', fontsize=11, fontweight='bold',
                     pad=5)

    add_panel(panels['a'], gs[0, :], 'A  Coverage overlap')
    add_panel(panels['b'], gs[1, 0], 'B  Gene type composition')
    add_panel(panels['c'], gs[1, 1], 'C  Significance breakdown')
    add_panel(panels['d'], gs[2:4, :], 'D  Concordance')
    add_panel(panels['e'], gs[4, :], 'E  Validation')

    fig.savefig(OUT_DIR / 'ag_twas_combined.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'ag_twas_combined.pdf', bbox_inches='tight')
    plt.close()
    print("Saved ag_twas_combined")


if __name__ == "__main__":
    main()
