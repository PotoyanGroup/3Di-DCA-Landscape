"""Supplementary Figure S4: Fisher-Rao geodesic distance on the 2D latents
recovers pairwise structural similarity (3Di-Hamming) better than Euclidean
distance.

For each protein family (MDH, TRPM, Globin, E1, E2) the script reads
precomputed Pearson correlations between pairwise latent distances and the
pairwise 3Di-Hamming structural distance, then draws a 1x2 grouped-bar
figure. Panel A uses the 3Di latent (built directly from structural tokens)
and Panel B uses the AA latent (which captures structure only indirectly via
coevolution); within each panel, Euclidean and Fisher-Rao distances are shown
side by side. Families are color-tagged by regime (consistent vs.
complementary) beneath the x-axis, highlighting that the AA latent only
recovers structural similarity in the consistent regime.

Inputs (paths resolved through analyses/scripts/_paths.py unless noted):
  - OUT_DATA / "55_structural_similarity_recovery.csv" : per-family table of
    Pearson r values (columns r_3DiLat_Eucl_vs_3DiHam, r_3DiLat_FR_vs_3DiHam,
    r_AALat_Eucl_vs_3DiHam, r_AALat_FR_vs_3DiHam) indexed by family.

Outputs:
  - OUT.parent / "figures_for_manuscript" / "SI_fisher_geodesic" /
    "SI_fisher_geodesic.pdf" : the 1x2 grouped-bar figure (PDF).
  - OUT.parent / "figures_for_manuscript" / "SI_fisher_geodesic" /
    "SI_fisher_geodesic.png" : the same figure as PNG (dpi=200).

Usage:
  python analyses/scripts/S4_fisher_geodesic.py
  No command-line arguments.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from _paths import OUT, OUT_DATA

OUTDIR = OUT.parent / "figures_for_manuscript" / "SI_fisher_geodesic"
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(OUT_DATA / "55_structural_similarity_recovery.csv")
FAM_ORDER = ["MDH", "TRPM", "Globin", "E1", "E2"]
df = df.set_index("family").loc[FAM_ORDER].reset_index()

REGIME = {"MDH": "consistent", "TRPM": "consistent",
          "Globin": "complementary", "E1": "complementary",
          "E2": "complementary"}
COL_CONS = "#2ca02c"
COL_COMP = "#9467bd"

COL_EUCL = "#6baed6"   # light blue
COL_FR   = "#fd8d3c"   # orange

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4),
                          gridspec_kw=dict(wspace=0.30))

x = np.arange(len(FAM_ORDER))
bw = 0.36


def add_panel_label(ax, letter, descriptor):
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", color="black",
            va="bottom", ha="left")
    ax.set_title(descriptor, fontsize=11, color="black", pad=6)


def plot_panel(ax, eucl_col, fr_col, letter, descriptor):
    ax.bar(x - bw/2, df[eucl_col], width=bw, color=COL_EUCL,
           edgecolor="black", linewidth=0.4, label="Euclidean")
    ax.bar(x + bw/2, df[fr_col], width=bw, color=COL_FR,
           edgecolor="black", linewidth=0.4, label="Fisher-Rao")
    for i in range(len(FAM_ORDER)):
        e = df[eucl_col].iloc[i]; f = df[fr_col].iloc[i]
        ax.text(i - bw/2, e + 0.02, f"{e:+.2f}", ha="center", va="bottom",
                fontsize=8.5, color="0.25")
        ax.text(i + bw/2, f + 0.02, f"{f:+.2f}", ha="center", va="bottom",
                fontsize=8.5, color="0.25")
        # Regime tag below x-axis label
        col = COL_CONS if REGIME[FAM_ORDER[i]] == "consistent" else COL_COMP
        ax.add_patch(plt.Rectangle((i - 0.42, -0.10), 0.84, 0.018,
                                      facecolor=col, clip_on=False,
                                      transform=ax.transData))
    ax.set_xticks(x); ax.set_xticklabels(FAM_ORDER, fontsize=10)
    ax.set_ylim(-0.12, 0.85)
    ax.set_ylabel("Pearson $r$ with pairwise 3Di-Hamming",
                  fontsize=10)
    ax.axhline(0, ls=":", color="0.3", lw=0.5)
    ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.9)
    add_panel_label(ax, letter, descriptor)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


plot_panel(axes[0], "r_3DiLat_Eucl_vs_3DiHam", "r_3DiLat_FR_vs_3DiHam",
           "A", "3Di latent  $\\to$  structural similarity")
plot_panel(axes[1], "r_AALat_Eucl_vs_3DiHam",  "r_AALat_FR_vs_3DiHam",
           "B", "AA latent  $\\to$  structural similarity")

out_pdf = OUTDIR / "SI_fisher_geodesic.pdf"
out_png = OUTDIR / "SI_fisher_geodesic.png"
fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")
