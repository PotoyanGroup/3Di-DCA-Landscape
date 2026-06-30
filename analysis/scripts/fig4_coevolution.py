"""Main-text Figure 4: DCA contact prediction comparing amino-acid (AA) and 3Di MSAs for the globin and peptidase families.

The script builds a 2x3 figure (top row = globin, bottom row = peptidase) from
pre-computed direct-coupling-analysis (DCA) results. For each family it draws a
true-positive-rate (TPR) curve as a function of the number of top-ranked
predictions N/L (panels A, D), a contact map of the top-L AA-DCA predictions
overlaid on the reference contacts (panels B, E), and the corresponding top-L
3Di-DCA predictions (panels C, F). In the contact panels, reference contacts are
shown in grey, true positives as green circles, and false positives as colored
crosses, with the per-panel counts of each annotated in the legend.

Inputs (paths resolved through analyses/scripts/_paths.py unless noted):
  - OUT_DATA / "43_globin_dca.npz"    : cached globin DCA results (DI_aa, DI_3di,
                                        contact_ref, L_aln, x_curve, tpr_aa, tpr_3di)
  - OUT_DATA / "56_peptidase_dca.npz" : cached peptidase DCA results (same keys)
  - OUT                               : output-directory alias from _paths.py, used
                                        to locate the figures_for_manuscript folder

Outputs (written under OUT.parent / "figures_for_manuscript" / "Fig4_coevolution"):
  - "Fig4_coevolution.pdf" : the assembled 2x3 figure (PDF, dpi=200)
  - "Fig4_coevolution.png" : the same figure as PNG (dpi=200)

Usage:
  python analyses/scripts/fig4_coevolution.py
  No command-line arguments.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

from _paths import OUT, OUT_DATA

OUTDIR = OUT.parent / "figures_for_manuscript" / "Fig4_coevolution"
OUTDIR.mkdir(parents=True, exist_ok=True)

COL_AA = "#1f77b4"
COL_3DI = "#ff7f0e"
COL_REF = "#bdbdbd"

FAMILIES = [
    dict(name="globin",   npz="43_globin_dca.npz",
         label="Globin", n_msa=2437),
    dict(name="peptidase", npz="56_peptidase_dca.npz",
         label="Peptidase", n_msa=10506),
]


def add_panel_label(ax, letter, descriptor):
    ax.text(-0.13, 1.05, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", color="black",
            va="bottom", ha="left")
    ax.set_title(descriptor, fontsize=10.5, color="black", pad=6)


def top_pairs(DI, L, K, sep=4):
    pairs = []
    for i in range(L):
        for j in range(i + sep + 1, L):
            pairs.append((DI[i, j], i, j))
    pairs.sort(reverse=True)
    return [(i, j) for _, i, j in pairs[:K]]


def plot_tpr(ax, x, tpr_aa, tpr_3di, label_letter, fam_label):
    ax.plot(x, tpr_aa,  "-o", ms=4, color=COL_AA,  lw=1.8, label="AA-DCA")
    ax.plot(x, tpr_3di, "-o", ms=4, color=COL_3DI, lw=1.8, label="3Di-DCA")
    ax.set_xlabel("top-$N$ predictions, $N/L$", fontsize=10)
    ax.set_ylabel("true-positive rate", fontsize=10)
    ax.set_xlim(0, x.max())
    ax.set_ylim(0, max(tpr_aa.max(), tpr_3di.max()) * 1.20)
    ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.9)
    add_panel_label(ax, label_letter, f"{fam_label}: TPR")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_contact_panel(ax, DI, contact_ref, L, label_letter, descriptor, model_col):
    ii, jj = np.where(np.triu(contact_ref, k=1))
    ax.scatter(jj, ii, s=8, c=COL_REF, marker="s", alpha=0.55,
                edgecolor="none", label="reference contacts")
    ax.scatter(ii, jj, s=8, c=COL_REF, marker="s", alpha=0.55,
                edgecolor="none")
    top = top_pairs(DI, L, L)
    tp = [(i, j) for (i, j) in top if contact_ref[i, j]]
    fp = [(i, j) for (i, j) in top if not contact_ref[i, j]]
    if tp:
        ti, tj = zip(*tp)
        ax.scatter(tj, ti, s=18, c="green", marker="o",
                    edgecolor="black", linewidth=0.4,
                    label=f"true positive ($n={len(tp)}$)", zorder=4)
        ax.scatter(ti, tj, s=18, c="green", marker="o",
                    edgecolor="black", linewidth=0.4, zorder=4)
    if fp:
        fi, fj = zip(*fp)
        ax.scatter(fj, fi, s=18, c=model_col, marker="x", linewidth=1.3,
                    label=f"false positive ($n={len(fp)}$)", zorder=3)
        ax.scatter(fi, fj, s=18, c=model_col, marker="x", linewidth=1.3,
                    zorder=3)
    ax.plot([0, L], [0, L], color="0.55", lw=0.7, ls="--")
    ax.set_xlim(0, L); ax.set_ylim(L, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("residue $j$", fontsize=9.5)
    ax.set_ylabel("residue $i$", fontsize=9.5)
    add_panel_label(ax, label_letter, descriptor)
    ax.legend(loc="lower left", fontsize=7.5, frameon=True, framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =====================================================================
# 2x3 figure
# =====================================================================
fig = plt.figure(figsize=(14.0, 9.0))
gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0],
                       hspace=0.42, wspace=0.32,
                       top=0.95, bottom=0.07, left=0.06, right=0.985)

letters = [("A", "B", "C"), ("D", "E", "F")]

for row, fam in enumerate(FAMILIES):
    d = np.load(OUT_DATA / fam["npz"], allow_pickle=True)
    DI_aa  = d["DI_aa"]; DI_3di = d["DI_3di"]
    contact_ref = d["contact_ref"].astype(bool)
    L = int(d["L_aln"])
    x = d["x_curve"]
    tpr_aa  = d["tpr_aa"]; tpr_3di = d["tpr_3di"]

    plot_tpr(fig.add_subplot(gs[row, 0]),
              x, tpr_aa, tpr_3di,
              letters[row][0], fam["label"])
    plot_contact_panel(fig.add_subplot(gs[row, 1]),
                        DI_aa, contact_ref, L,
                        letters[row][1],
                        f"{fam['label']} AA-DCA top-$L$", COL_AA)
    plot_contact_panel(fig.add_subplot(gs[row, 2]),
                        DI_3di, contact_ref, L,
                        letters[row][2],
                        f"{fam['label']} 3Di-DCA top-$L$", COL_3DI)


out_pdf = OUTDIR / "Fig4_coevolution.pdf"
out_png = OUTDIR / "Fig4_coevolution.png"
fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")
