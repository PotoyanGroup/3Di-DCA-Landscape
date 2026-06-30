"""Main-text Figure 1: framework overview and the central AA-vs-3Di question.

Assembles a four-panel overview figure from precomputed summary tables. Panel A
is a hand-drawn pipeline schematic (Family MSA -> ProstT5 translator -> paired
AA / 3Di MSAs -> AA-VAE / 3Di-VAE -> paired latents with DCA Hamiltonians ->
compare representations). Panel B is a "systems studied" table listing the five
protein families (MDH, TRPM, Globin, E1, E2) with a family-color marker, a
subfamily color strip, and an n_seq / L annotation (these per-family metadata
are hard-coded in the SYSTEMS list, not read from disk). Panel C draws
per-family half-violins of the pairwise AA-Hamming (left) and 3Di-Hamming
(right) distance distributions. Panel D overlays, per family, the model-free
per-pair Pearson r between AA- and 3Di-Hamming distances (left bars, with a 0.4
threshold splitting "consistent" vs "complementary" latents) and the
model-based cross-latent mutual information in nats (right twin-axis bars).

Inputs (paths resolved through analyses/scripts/_paths.py unless noted):
  - OUT_DATA / "15_pairwise_hamming.npz" : per-family arrays "{fam}_AA" and
        "{fam}_3Di" of pairwise Hamming distances (panel C half-violins).
  - OUT_DATA / "15_diversity_summary.csv" : per-family diversity summary
        (loaded and reindexed to FAM_ORDER).
  - OUT_DATA / "33_per_pair_correlations.csv" : per-family model-free per-pair
        Pearson r (column "per_pair_Pearson_r"; panel D left bars).
  - OUT_DATA / "24_mutual_information.csv" : per-family cross-latent mutual
        information (column "MI_nats"; panel D right twin-axis bars).
  - SYSTEMS, FAM_ORDER : in-script constants for family/subfamily labels,
        colors, n_seq, and L (panel B); not read from disk.

Outputs (written to OUT.parent / "figures_for_manuscript" / "Fig1_overview"):
  - Fig1_overview_v3.pdf : the assembled four-panel overview figure.
  - Fig1_overview_v3.png : PNG copy of the same figure (dpi=200).

Usage:
  python analyses/scripts/fig1_overview.py
  No command-line arguments.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle as MPLRect

from _paths import OUT, OUT_DATA

OUTDIR = OUT.parent / "figures_for_manuscript" / "Fig1_overview"
OUTDIR.mkdir(parents=True, exist_ok=True)

COL_AA = "#1f77b4"
COL_3DI = "#ff7f0e"
COL_NEUTRAL = "#666666"
COL_HIGHLIGHT = "#2ca02c"

# Per-family pairwise Hamming distributions (for panel C half-violins)
FAM_ORDER = ["MDH", "TRPM", "Globin", "E1", "E2"]
pair_npz = np.load(OUT_DATA / "15_pairwise_hamming.npz", allow_pickle=True)
pair_data = {f: {"AA": pair_npz[f"{f}_AA"], "3Di": pair_npz[f"{f}_3Di"]}
             for f in FAM_ORDER}
div = pd.read_csv(OUT_DATA / "15_diversity_summary.csv")
div_sorted = div.set_index("family").loc[FAM_ORDER].reset_index()

# Per-family per-pair Pearson r (model-free regime classification, for panel D)
pp = pd.read_csv(OUT_DATA / "33_per_pair_correlations.csv")
pp_sorted = pp.set_index("family").loc[FAM_ORDER].reset_index()

# Model-based cross-latent mutual information per family (folded in for panel D)
mi = pd.read_csv(OUT_DATA / "24_mutual_information.csv")
mi_sorted = mi.set_index("family").loc[FAM_ORDER].reset_index()

SYSTEMS = [
    dict(name="MDH",    fold="Rossmann fold",   n_seq=2100, L=332,
         color="#2ca02c",
         subfams=["psychro", "meso", "thermo"],
         sub_colors=["#1f77b4", "#7f7f7f", "#d62728"]),
    dict(name="TRPM",   fold="6-TM channel",    n_seq=3002, L=289,
         color="#9467bd",
         subfams=["TRPM1/3", "TRPM6/7", "TRPM4/5", "TRPM2/8"],
         sub_colors=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]),
    dict(name="Globin", fold="8-helix bundle",  n_seq=10570, L=118,
         color="#e377c2",
         subfams=["hemo", "myo", "flavo", "cyto", "neuro"],
         sub_colors=["#d62728", "#ff7f0e", "#1f77b4", "#9467bd", "#8c564b"]),
    dict(name="E1",     fold="class-II fusion", n_seq=191,  L=205,
         color="#8c564b",
         subfams=["hepaci", "pegi", "pesti"],
         sub_colors=["#d62728", "#2ca02c", "#1f77b4"]),
    dict(name="E2",     fold="class-II fusion", n_seq=168,  L=321,
         color="#17becf",
         subfams=["hepaci", "pegi", "pesti"],
         sub_colors=["#d62728", "#2ca02c", "#1f77b4"]),
]


# ============================================================
# Build figure
# ============================================================
fig = plt.figure(figsize=(15.5, 8.5), constrained_layout=False)
gs = fig.add_gridspec(
    2, 1, height_ratios=[0.85, 1.0], hspace=0.50,
    top=0.94, bottom=0.08, left=0.04, right=0.99,
)

# ============================================================
# Panel A - Pipeline schematic
# ============================================================
axA = fig.add_subplot(gs[0])
axA.set_xlim(0, 10)
axA.set_ylim(0, 3.2)
axA.axis("off")
fig.text(0.06, 0.945, "A",
         fontsize=13, fontweight="bold")


def draw_box(x, y, w, h, label, color, fontsize=9, fontweight="normal",
             alpha=0.20):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.10",
        linewidth=1.5, edgecolor=color, facecolor=color, alpha=alpha,
    )
    axA.add_patch(box)
    axA.text(x + w / 2, y + h / 2, label, ha="center", va="center",
             fontsize=fontsize, fontweight=fontweight)


def draw_arrow(x1, y1, x2, y2, color="0.35", lw=1.5):
    axA.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>",
        mutation_scale=14, linewidth=lw, color=color,
    ))


y_top = 1.85
y_bot = 0.55
h_box = 0.85
y_mid = (y_top + y_bot + h_box) / 2 - h_box / 2

draw_box(0.10, y_mid, 1.30, h_box, "Family\nMSA", COL_NEUTRAL,
         fontweight="bold", alpha=0.25)
draw_box(1.85, y_mid, 1.60, h_box, "ProstT5\ntranslator",
         "#9467bd", alpha=0.22)
draw_box(3.90, y_top, 1.20, h_box, "AA MSA", COL_AA)
draw_box(3.90, y_bot, 1.20, h_box, "3Di MSA", COL_3DI)
draw_box(5.55, y_top, 1.20, h_box, "AA-VAE", COL_AA)
draw_box(5.55, y_bot, 1.20, h_box, "3Di-VAE", COL_3DI)
draw_box(7.20, y_top, 1.55, h_box,
         "AA latent\n+ DCA $H_{AA}(z)$", COL_AA)
draw_box(7.20, y_bot, 1.55, h_box,
         "3Di latent\n+ DCA $H_{3Di}(z)$", COL_3DI)
draw_box(8.95, y_mid, 0.95, h_box,
         "compare\nrepresentations", COL_HIGHLIGHT,
         fontweight="bold", alpha=0.30)

draw_arrow(1.40, y_mid + h_box / 2, 1.85, y_mid + h_box / 2)
draw_arrow(3.45, y_mid + h_box / 2, 3.90, y_top + h_box / 2)
draw_arrow(3.45, y_mid + h_box / 2, 3.90, y_bot + h_box / 2)
draw_arrow(5.10, y_top + h_box / 2, 5.55, y_top + h_box / 2)
draw_arrow(5.10, y_bot + h_box / 2, 5.55, y_bot + h_box / 2)
draw_arrow(6.75, y_top + h_box / 2, 7.20, y_top + h_box / 2)
draw_arrow(6.75, y_bot + h_box / 2, 7.20, y_bot + h_box / 2)
draw_arrow(8.75, y_top + h_box / 2, 8.95, y_mid + h_box / 2)
draw_arrow(8.75, y_bot + h_box / 2, 8.95, y_mid + h_box / 2)


# ============================================================
# Bottom row: B (families table) | C (Hamming half-violins) | D (per-pair r)
# ============================================================
gs_mid = gs[1].subgridspec(
    1, 3, width_ratios=[1.1, 1.0, 0.85], wspace=0.30,
)

# --- Panel B: systems-studied table ---
axB_sys = fig.add_subplot(gs_mid[0])
axB_sys.set_xlim(0, 1); axB_sys.set_ylim(0, 1)
axB_sys.axis("off")
fig.text(0.05, 0.50, "B",
         fontsize=13, fontweight="bold")

X_NAME, X_SUB, X_NL = 0.085, 0.30, 0.80
SUB_W, SUB_H = 0.42, 0.045
axB_sys.text(X_NAME, 0.95, "family",      fontsize=11, fontweight="bold",
             ha="left", va="center", color="0.20")
axB_sys.text(X_SUB,  0.95, "subfamilies", fontsize=11, fontweight="bold",
             ha="left", va="center", color="0.20")
axB_sys.text(X_NL,   0.95, "$n_\\mathrm{seq}$ / $L$", fontsize=11,
             fontweight="bold", ha="left", va="center", color="0.20")
axB_sys.plot([0.02, 0.99], [0.89, 0.89], color="0.6", lw=0.7)

row_y = np.linspace(0.78, 0.08, len(SYSTEMS))
for y, fam in zip(row_y, SYSTEMS):
    axB_sys.scatter([0.03], [y], s=120, c=fam["color"],
                    edgecolor="white", linewidth=0.7, zorder=3)
    axB_sys.text(X_NAME, y, fam["name"], fontsize=11, fontweight="bold",
                 ha="left", va="center", color=fam["color"])
    n_sub = len(fam["subfams"])
    block_w = SUB_W / n_sub
    for j, sc in enumerate(fam["sub_colors"]):
        rect = MPLRect((X_SUB + j * block_w, y - SUB_H / 2),
                       block_w * 0.92, SUB_H,
                       facecolor=sc, edgecolor="white", linewidth=0.4)
        axB_sys.add_patch(rect)
    axB_sys.text(X_SUB, y - SUB_H * 1.6,
                 " | ".join(fam["subfams"]),
                 fontsize=8, ha="left", va="top", color="black",
                 style="italic")
    axB_sys.text(X_NL, y, f"{fam['n_seq']:,} / {fam['L']}",
                 fontsize=10, ha="left", va="center", color="0.20")


# --- Panel C: per-family half-violins of pairwise AA / 3Di Hamming ---
axC = fig.add_subplot(gs_mid[1])
fig.text(0.405, 0.50,
         "C",
         fontsize=13, fontweight="bold")


def half_violin(ax, data, position, side, color, width=0.42):
    """Draw a one-sided (half) violin at x = position."""
    parts = ax.violinplot([data], positions=[position],
                          widths=width * 2, showmeans=False,
                          showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        # Clip one side to make half-violin
        verts = body.get_paths()[0].vertices
        if side == "left":
            verts[:, 0] = np.clip(verts[:, 0], -np.inf, position)
        else:
            verts[:, 0] = np.clip(verts[:, 0], position, np.inf)
        body.set_facecolor(color); body.set_edgecolor(color)
        body.set_alpha(0.6); body.set_linewidth(0.7)
    # Median bar
    med = float(np.median(data))
    offset = 0.04
    if side == "left":
        ax.plot([position - offset, position], [med, med],
                color="black", lw=1.3, zorder=4)
    else:
        ax.plot([position, position + offset], [med, med],
                color="black", lw=1.3, zorder=4)


xpos = np.arange(len(FAM_ORDER))
for i, fam in enumerate(FAM_ORDER):
    half_violin(axC, pair_data[fam]["AA"],  i, side="left",  color=COL_AA)
    half_violin(axC, pair_data[fam]["3Di"], i, side="right", color=COL_3DI)

axC.set_xticks(xpos)
axC.set_xticklabels(FAM_ORDER, fontsize=10)
axC.set_ylabel("pairwise Hamming distance", fontsize=11)
axC.set_ylim(0, 1.05)
axC.set_xlim(-0.6, len(FAM_ORDER) - 0.4)
# Legend with two color swatches
from matplotlib.patches import Patch
axC.legend(handles=[
    Patch(facecolor=COL_AA,  alpha=0.6, edgecolor=COL_AA,  label="AA-Hamming (sequence)"),
    Patch(facecolor=COL_3DI, alpha=0.6, edgecolor=COL_3DI, label="3Di-Hamming (structure)"),
], loc="upper left", fontsize=8.5, frameon=True, framealpha=0.9)
axC.spines["top"].set_visible(False)
axC.spines["right"].set_visible(False)
axC.tick_params(labelsize=10)


# --- Panel D: per-family model-free per-pair Pearson r ---
axD = fig.add_subplot(gs_mid[2])
fig.text(0.71, 0.50,
         "D",
         fontsize=13, fontweight="bold")

COL_CONS = "#2ca02c"   # consistent latents (high per-pair r)
COL_COMP = "#9467bd"   # complementary latents
xpos_d = np.arange(len(FAM_ORDER))
r_vals = pp_sorted["per_pair_Pearson_r"].values
mi_vals = mi_sorted["MI_nats"].values
bar_colors_d = np.where(r_vals > 0.4, COL_CONS, COL_COMP)
bw = 0.38

axD.bar(xpos_d - bw / 2, r_vals, width=bw, color=bar_colors_d,
        alpha=0.85, edgecolor="black", linewidth=0.5,
        label="per-pair $r$ (left)")
axD2 = axD.twinx()
axD2.bar(xpos_d + bw / 2, mi_vals, width=bw, color=bar_colors_d,
         alpha=0.35, edgecolor="black", linewidth=0.5, hatch="//",
         label="MI (right)")

axD.axhline(0.4, ls=":", color="0.4", lw=0.8)
axD.set_xticks(xpos_d); axD.set_xticklabels(FAM_ORDER, fontsize=10)
axD.set_ylim(0, 1.05)
axD2.set_ylim(0, max(mi_vals) * 1.15)
axD.set_ylabel("per-pair Pearson $r$\n(AA-Hamming, 3Di-Hamming)",
               fontsize=10)
axD2.set_ylabel(r"cross-latent MI  $I(\mu_\mathrm{AA};\mu_\mathrm{3Di})$  (nats)",
                fontsize=10)
axD2.tick_params(axis="y", labelsize=10)
axD2.spines["top"].set_visible(False)
# Regime annotations
axD.text(0.97, 0.97, "consistent\nlatents",
         transform=axD.transAxes, ha="right", va="top",
         fontsize=9, color=COL_CONS, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                   edgecolor=COL_CONS, alpha=0.85))
axD.text(0.97, 0.32, "complementary\nlatents",
         transform=axD.transAxes, ha="right", va="top",
         fontsize=9, color=COL_COMP, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                   edgecolor=COL_COMP, alpha=0.85))
axD.spines["top"].set_visible(False)
axD.spines["right"].set_visible(False)
axD.tick_params(labelsize=10)


# ============================================================
# Save
# ============================================================
out_pdf = OUTDIR / "Fig1_overview_v3.pdf"
out_png = OUTDIR / "Fig1_overview_v3.png"
fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")
