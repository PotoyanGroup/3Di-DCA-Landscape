"""Supplementary Figure S8: per-family AA / 3Di Fisher determinant ratio.

For each protein family (MDH, TRPM, Globin, E1, E2) the script draws three
grouped, log-scaled bars summarizing the ratio of the amino-acid (AA) to the
3Di Fisher-information determinant across the latent landscape: the ratio at
the median, at the 90th percentile (P90), and the total integrated "volume".
A dotted reference line marks ratio = 1. AA exceeds 3Di in latent volume for
most families, but the ratio is family-dependent and inverts in the
high-Fisher tail for Globin. (This panel was previously Fig 3A in the main
text and was moved to the SI during the revision.)

Inputs (paths resolved through analyses/scripts/_paths.py unless noted):
  - OUT_DATA / "10_fisher_universality.csv" : precomputed per-family Fisher
    determinant ratio table with columns "family", "AA/3Di median",
    "AA/3Di P90", and "AA/3Di total".

Outputs:
  - OUT.parent / "figures_for_manuscript" / "SI_fisher_universality" /
    "SI_fisher_universality.pdf" : the grouped-bar figure (PDF).
  - OUT.parent / "figures_for_manuscript" / "SI_fisher_universality" /
    "SI_fisher_universality.png" : the same figure (PNG).

Usage:
  python analyses/scripts/S8_fisher_universality.py
  No command-line arguments.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _paths import OUT, OUT_DATA

OUTDIR = OUT.parent / "figures_for_manuscript" / "SI_fisher_universality"
OUTDIR.mkdir(parents=True, exist_ok=True)

univ = pd.read_csv(OUT_DATA / "10_fisher_universality.csv")
FAM = ["MDH", "TRPM", "Globin", "E1", "E2"]
univ = univ.set_index("family").loc[FAM].reset_index()


fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.6))

xpos = np.arange(len(FAM)); bw = 0.28
ax.bar(xpos - bw, univ["AA/3Di median"], width=bw, color="#2ca02c",
       alpha=0.85, edgecolor="black", linewidth=0.4, label="median")
ax.bar(xpos,       univ["AA/3Di P90"],   width=bw, color="#9467bd",
       alpha=0.85, edgecolor="black", linewidth=0.4, label="P90")
ax.bar(xpos + bw,  univ["AA/3Di total"], width=bw, color="#d62728",
       alpha=0.85, edgecolor="black", linewidth=0.4, label="total volume")
ax.axhline(1.0, ls=":", color="0.4", lw=0.8)
ax.set_xticks(xpos); ax.set_xticklabels(FAM, fontsize=12)
ax.set_ylabel("AA / 3Di Fisher determinant ratio", fontsize=12)
ax.tick_params(axis="y", labelsize=11)
ax.set_yscale("log")
ax.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


out_pdf = OUTDIR / "SI_fisher_universality.pdf"
out_png = OUTDIR / "SI_fisher_universality.png"
fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")
