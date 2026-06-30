"""Supplementary Figure S5: joint (AA + 3Di) Fisher-Rao distance correlates
better with the joint pairwise Hamming distance than either single
representation alone.

Reads a precomputed CSV of Pearson correlations between Fisher-Rao distances
and the joint pairwise Hamming distance for five protein families
(MDH, TRPM, Globin, E1, E2), and renders a single 1x1 panel of grouped bars.
Each family shows three bars (AA Fisher-Rao only, 3Di Fisher-Rao only, joint
Fisher-Rao), with the bar values annotated above each bar. A small colored
rectangle under each family label marks its regime (consistent vs
complementary), and the legend keys both the three Fisher-Rao series and the
two regime tags.

Inputs (paths resolved through analyses/scripts/_paths.py unless noted):
  - OUT_DATA / "53_joint_fisher_vs_hamming.csv" : per-family table (indexed by
    `family`) with columns r_FR_AA_vs_jointHam, r_FR_3Di_vs_jointHam, and
    r_FR_joint_vs_jointHam (Pearson r of each Fisher-Rao distance against the
    joint pairwise Hamming distance).

Outputs:
  - OUT.parent / "figures_for_manuscript" / "SI_joint_fisher" / "SI_joint_fisher.pdf"
    : the grouped-bar figure (PDF, dpi 200).
  - OUT.parent / "figures_for_manuscript" / "SI_joint_fisher" / "SI_joint_fisher.png"
    : the same figure (PNG, dpi 200).

Usage:
  python analyses/scripts/S5_joint_fisher.py
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

OUTDIR = OUT.parent / "figures_for_manuscript" / "SI_joint_fisher"
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(OUT_DATA / "53_joint_fisher_vs_hamming.csv")
FAM_ORDER = ["MDH", "TRPM", "Globin", "E1", "E2"]
df = df.set_index("family").loc[FAM_ORDER].reset_index()

REGIME = {"MDH": "consistent", "TRPM": "consistent",
          "Globin": "complementary", "E1": "complementary", "E2": "complementary"}

COL_AA   = "#1f77b4"
COL_3DI  = "#ff7f0e"
COL_JNT  = "#d62728"
COL_CONS = "#2ca02c"
COL_COMP = "#9467bd"

fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.6))
x = np.arange(len(FAM_ORDER))
bw = 0.27

ax.bar(x - bw, df["r_FR_AA_vs_jointHam"],   width=bw,
       color=COL_AA, edgecolor="black", linewidth=0.4,
       label="AA Fisher-Rao")
ax.bar(x,       df["r_FR_3Di_vs_jointHam"],  width=bw,
       color=COL_3DI, edgecolor="black", linewidth=0.4,
       label="3Di Fisher-Rao")
ax.bar(x + bw,  df["r_FR_joint_vs_jointHam"], width=bw,
       color=COL_JNT, edgecolor="black", linewidth=0.4,
       label="Joint Fisher-Rao")
for i in range(len(FAM_ORDER)):
    for j, col in enumerate(["r_FR_AA_vs_jointHam",
                              "r_FR_3Di_vs_jointHam",
                              "r_FR_joint_vs_jointHam"]):
        v = df[col].iloc[i]
        ax.text(i + (j - 1) * bw, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.5, color="0.25")

# Regime annotation bars above x-axis labels
for i, fam in enumerate(FAM_ORDER):
    col = COL_CONS if REGIME[fam] == "consistent" else COL_COMP
    ax.add_patch(plt.Rectangle((i - 0.42, -0.07), 0.84, 0.018,
                                facecolor=col, clip_on=False,
                                transform=ax.transData))

ax.set_xticks(x); ax.set_xticklabels(FAM_ORDER, fontsize=10)
ax.set_ylabel("Pearson $r$ with joint pairwise Hamming",
              fontsize=10)
ax.set_ylim(-0.08, 0.85)
ax.axhline(0, ls=":", color="0.3", lw=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend with regime tags
handles = [
    Patch(facecolor=COL_AA,  edgecolor="black", label="AA Fisher-Rao only"),
    Patch(facecolor=COL_3DI, edgecolor="black", label="3Di Fisher-Rao only"),
    Patch(facecolor=COL_JNT, edgecolor="black", label="Joint Fisher-Rao"),
    Patch(facecolor=COL_CONS, edgecolor="none", label="consistent regime"),
    Patch(facecolor=COL_COMP, edgecolor="none", label="complementary regime"),
]
ax.legend(handles=handles, loc="upper right", fontsize=9, ncol=1,
          frameon=True, framealpha=0.9)

out_pdf = OUTDIR / "SI_joint_fisher.pdf"
out_png = OUTDIR / "SI_joint_fisher.png"
fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")
