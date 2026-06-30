"""Main-text Figure 2: MDH case study - latent spaces and information geometry.

Builds a 2x2 figure dedicated to the MDH-specific latent geometry of two
thermal-adaptation cohorts (psychrophilic vs. thermo/meso). Panel A shows the
AA-VAE latent embedding of both cohorts and Panel B the 3Di-VAE embedding on a
shared axis, each annotated with the Gaussian Wasserstein-2 distance between
the cohort point clouds. Panel C is a pullback-Fisher AA/3Di ratio heatmap over
the MDH latent grid (where the 3Di metric compresses relative to AA). Panel D
is a per-sequence comparison plane of H_AA - H_3Di (sequence- vs.
structure-level DCA typicality) against log10 F_AA / F_3Di (which decoder is
locally sharper), with the natural MDH sequence cloud drawn as a KDE-density
landscape behind the two cohorts. The script also writes a JSON summary of
cohort sizes, W2 distances, and within/between TM-score and sequence-identity
means.

Inputs (paths resolved through analyses/scripts/_paths.py unless noted):
  - MDH_CLUSTER_DATA (imported as BASE) : cohort directory tree holding, for
    each cohort, cluster_coordinates_AA.txt / cluster_coordinates_3DI.txt
    (latent coords), inter_pairwise_tm_scores.csv, and the per-cohort MSAs
    filtered_psycro.fasta / thermo_meso_filtered.fasta; plus the top-level
    intra_cluster_pairwise_tm_scores.csv and
    intra_cluster_pairwise_sequence_identity.csv (between-cohort stats).
  - OUT_DATA / "08_pullback_metric_aa_mdh.npz" : AA pullback-Fisher volume
    field + z0/z1 axes on the MDH latent grid.
  - OUT_DATA / "08_pullback_metric_3di_mdh.npz" : 3Di pullback-Fisher volume
    field + axes (used for the ratio map and Panel D).
  - OUT_DATA / "16_fisher_dca_joint_aa_mdh.npz" : AA Fisher (F) and DCA
    Hamiltonian (H) fields, latent axes (z0/z1), and quadrant labels.
  - OUT_DATA / "17_H_3di_grid.npy" / "17_H_3di_coords.npy" : 3Di-DCA
    Hamiltonian field and its (z0, z1) coordinate axes.
  - OUT_DATA / "cache_paired_mus.npz" : cached natural MDH latent means
    (MDH_mu_aa, MDH_mu_3di) for the background sequence cloud.

Outputs:
  - OUTDIR / "Fig2_MDH_cohort.pdf" : the assembled 2x2 figure (PDF), where
    OUTDIR = OUT.parent / "figures_for_manuscript" / "Fig2_MDH_cohort".
  - OUTDIR / "Fig2_MDH_cohort.png" : PNG copy of the same figure.
  - OUT_DATA / "35_mdh_cohort_summary.json" : cohort sizes, W2_AA / W2_3Di,
    and within/between TM-score and sequence-identity means.

Usage:
  python analyses/scripts/fig2_mdh_cohort.py
  No command-line arguments.
"""
from pathlib import Path
import sys
import ast
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from Bio import SeqIO

from _paths import OUT

from _paths import MDH_CLUSTER_DATA as BASE  # noqa: E402
OUT_DATA = OUT / "data"
OUTDIR = OUT.parent / "figures_for_manuscript" / "Fig2_MDH_cohort"
OUTDIR.mkdir(parents=True, exist_ok=True)

COL_AA = "#1f77b4"
COL_3DI = "#ff7f0e"
COL_PSYC = "#1f77b4"
COL_THER = "#d62728"
COL_WITHIN_P = "#1f77b4"
COL_WITHIN_T = "#d62728"
COL_BETWEEN = "#7f7f7f"


# =====================================================================
# Load cohort data (latent coords + pairwise stats)
# =====================================================================
def load_coords(path):
    return ast.literal_eval(path.read_text())


psyc_aa  = np.array(list(load_coords(BASE / "psycrophilic_cluster" /
                                       "cluster_coordinates_AA.txt").values()))
psyc_di  = np.array(list(load_coords(BASE / "psycrophilic_cluster" /
                                       "cluster_coordinates_3DI.txt").values()))
ther_aa  = np.array(list(load_coords(BASE / "thermo_meso_cluster" /
                                       "cluster_coordinates_AA.txt").values()))
ther_di  = np.array(list(load_coords(BASE / "thermo_meso_cluster" /
                                       "cluster_coordinates_3DI.txt").values()))
n_psyc, n_ther = len(psyc_aa), len(ther_aa)


def gaussian_w2(mu1, mu2):
    m1, m2 = mu1.mean(0), mu2.mean(0)
    c1, c2 = np.cov(mu1.T), np.cov(mu2.T)
    s1, U1 = np.linalg.eigh(c1); s1 = np.clip(s1, 0, None)
    sq1 = U1 @ np.diag(np.sqrt(s1)) @ U1.T
    inner = sq1 @ c2 @ sq1
    s_in, U_in = np.linalg.eigh(inner); s_in = np.clip(s_in, 0, None)
    trace = np.trace(c1 + c2 - 2 * U_in @ np.diag(np.sqrt(s_in)) @ U_in.T)
    return float(np.sqrt(np.sum((m1 - m2)**2) + max(trace, 0)))


w2_aa = gaussian_w2(psyc_aa, ther_aa)
w2_di = gaussian_w2(psyc_di, ther_di)


# Pairwise TM / identity
tm_within_p = pd.read_csv(BASE / "psycrophilic_cluster" /
                          "inter_pairwise_tm_scores.csv")["TM_avg"].values
tm_within_t = pd.read_csv(BASE / "thermo_meso_cluster" /
                          "inter_pairwise_tm_scores.csv")["TM_avg"].values
tm_between  = pd.read_csv(BASE /
                          "intra_cluster_pairwise_tm_scores.csv")["TM_avg"].values
id_between  = pd.read_csv(BASE / "intra_cluster_pairwise_sequence_identity.csv")[
                          "sequence_identity_percent"].values


def within_identity(fasta_path):
    recs = list(SeqIO.parse(str(fasta_path), "fasta"))
    out = []
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            a = str(recs[i].seq).upper(); b = str(recs[j].seq).upper()
            valid = [(x, y) for x, y in zip(a, b) if x != "-" and y != "-"]
            if not valid: continue
            match = sum(1 for x, y in valid if x == y)
            out.append(100.0 * match / len(valid))
    return np.array(out)


id_within_p = within_identity(BASE / "psycrophilic_cluster" / "filtered_psycro.fasta")
id_within_t = within_identity(BASE / "thermo_meso_cluster" / "thermo_meso_filtered.fasta")


# Fisher fields (for the ratio map and joint plot)
mdh_aa  = np.load(OUT_DATA / "08_pullback_metric_aa_mdh.npz",  allow_pickle=True)
mdh_3di = np.load(OUT_DATA / "08_pullback_metric_3di_mdh.npz", allow_pickle=True)
log_ratio = np.log10(mdh_aa["vol"] + 1e-6) - np.log10(mdh_3di["vol"] + 1e-6)
v_abs = float(np.max(np.abs(log_ratio)))

d16 = np.load(OUT_DATA / "16_fisher_dca_joint_aa_mdh.npz", allow_pickle=True)
F_grid = d16["F"].flatten(); H_grid = d16["H"].flatten()
quad_grid = d16["quadrant"].flatten()
F_field = d16["F"]; H_field = d16["H"]
z0_grid = d16["z0"]; z1_grid = d16["z1"]


def interp_FH(mu_arr):
    """Bilinear interp of F and H fields at array of (N, 2) latent points."""
    px = np.clip(mu_arr[:, 0], z0_grid[0], z0_grid[-1] - 1e-9)
    py = np.clip(mu_arr[:, 1], z1_grid[0], z1_grid[-1] - 1e-9)
    Nz0, Nz1 = len(z0_grid), len(z1_grid)
    fx = (px - z0_grid[0]) / (z0_grid[-1] - z0_grid[0]) * (Nz0 - 1)
    fy = (py - z1_grid[0]) / (z1_grid[-1] - z1_grid[0]) * (Nz1 - 1)
    ix = np.clip(np.floor(fx).astype(int), 0, Nz0 - 2)
    iy = np.clip(np.floor(fy).astype(int), 0, Nz1 - 2)
    tx = fx - ix; ty = fy - iy
    def bilerp(field):
        return ((1 - tx) * (1 - ty) * field[ix, iy] +
                tx * (1 - ty) * field[ix + 1, iy] +
                (1 - tx) * ty * field[ix, iy + 1] +
                tx * ty * field[ix + 1, iy + 1])
    return bilerp(F_field), bilerp(H_field)


F_psyc, H_psyc = interp_FH(psyc_aa)
F_ther, H_ther = interp_FH(ther_aa)


# =====================================================================
# Figure
# =====================================================================
fig = plt.figure(figsize=(11.5, 9.0))
gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.32,
                      top=0.96, bottom=0.07, left=0.07, right=0.97)

# Shared axis range for A and B
all_x = np.concatenate([psyc_aa[:, 0], ther_aa[:, 0]])
all_y = np.concatenate([psyc_aa[:, 1], ther_aa[:, 1]])
xlim = (all_x.min() - 1.0, all_x.max() + 1.0)
ylim = (all_y.min() - 1.0, all_y.max() + 1.0)


def add_panel_label(ax, letter, descriptor):
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes,
            fontsize=16, fontweight="bold", color="black",
            va="bottom", ha="left")
    ax.set_title(descriptor, fontsize=13, color="black", pad=6)


def plot_latent(ax, psyc, ther, label_p, label_t, letter, descriptor,
                show_legend, w2):
    ax.scatter(psyc[:, 0], psyc[:, 1], s=58, c=COL_PSYC,
               edgecolor="white", linewidth=0.7, alpha=0.85,
               label=label_p, zorder=3)
    ax.scatter(ther[:, 0], ther[:, 1], s=58, c=COL_THER,
               edgecolor="white", linewidth=0.7, alpha=0.85,
               label=label_t, zorder=3)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("$z_0$", fontsize=13); ax.set_ylabel("$z_1$", fontsize=13)
    ax.tick_params(labelsize=11)
    add_panel_label(ax, letter, descriptor)
    if show_legend:
        ax.legend(loc="lower left", fontsize=10, frameon=True, framealpha=0.9)
    ax.text(0.02, 0.97,
            f"$W_2 = {w2:.2f}$",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      alpha=0.9, edgecolor="0.7"),
            va="top", ha="left")
    ax.axhline(0, ls=":", color="0.85", lw=0.5)
    ax.axvline(0, ls=":", color="0.85", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =====================================================================
# Panels A and B: latent embeddings (AA, 3Di)
# =====================================================================
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
plot_latent(axA, psyc_aa, ther_aa,
            f"psychrophilic  (n = {n_psyc})",
            f"thermo/meso  (n = {n_ther})",
            "A", "AA latent", True, w2_aa)
plot_latent(axB, psyc_di, ther_di, None, None,
            "B", "3Di latent", False, w2_di)


# Panel C: Fisher AA/3Di ratio map
axC = fig.add_subplot(gs[1, 0])
Z0, Z1 = np.meshgrid(mdh_aa["z0_axis"], mdh_aa["z1_axis"], indexing="ij")
im_r = axC.pcolormesh(Z0, Z1, log_ratio, shading="auto",
                      cmap="RdBu_r",
                      norm=TwoSlopeNorm(vmin=-v_abs, vcenter=0, vmax=v_abs))
axC.set_xlabel("$z_0$", fontsize=13); axC.set_ylabel("$z_1$", fontsize=13)
axC.tick_params(labelsize=11)
add_panel_label(axC, "C", "Where 3Di compresses")
axC.set_aspect("auto")
cb = fig.colorbar(im_r, ax=axC, fraction=0.04, pad=0.02, shrink=0.85)
cb.ax.tick_params(labelsize=11)
# Label above the colorbar, left-aligned so the text grows into the
# empty gutter to the right of the colorbar rather than back into the
# Panel C heatmap.
cb.ax.set_title(r"$\log_{10}\, F_\mathrm{AA}/F_\mathrm{3Di}$",
                fontsize=10, pad=6, loc="left")


# Panel D: AA-vs-3Di per-sequence comparison plane
# x = H_AA - H_3Di  (sequence-level vs structure-level typicality)
# y = log10(F_AA / F_3Di)  (which decoder is locally sharper)
axD = fig.add_subplot(gs[1, 1])

# Load 3Di Fisher and 3Di-DCA Hamiltonian fields
d3di = np.load(OUT_DATA / "08_pullback_metric_3di_mdh.npz", allow_pickle=True)
F_3di_field = d3di["vol"]; z0_3di = d3di["z0_axis"]; z1_3di = d3di["z1_axis"]
H_3di_grid_arr = np.load(OUT_DATA / "17_H_3di_grid.npy")
H_3di_coords = np.load(OUT_DATA / "17_H_3di_coords.npy")
z0_3di_H = H_3di_coords[0]; z1_3di_H = H_3di_coords[1]


def bilerp(field, ax0, ax1, points):
    px = np.clip(points[:, 0], ax0[0], ax0[-1] - 1e-9)
    py = np.clip(points[:, 1], ax1[0], ax1[-1] - 1e-9)
    Nz0, Nz1 = len(ax0), len(ax1)
    fx = (px - ax0[0]) / (ax0[-1] - ax0[0]) * (Nz0 - 1)
    fy = (py - ax1[0]) / (ax1[-1] - ax1[0]) * (Nz1 - 1)
    ix = np.clip(np.floor(fx).astype(int), 0, Nz0 - 2)
    iy = np.clip(np.floor(fy).astype(int), 0, Nz1 - 2)
    tx = fx - ix; ty = fy - iy
    return ((1 - tx) * (1 - ty) * field[ix, iy]
            + tx * (1 - ty) * field[ix + 1, iy]
            + (1 - tx) * ty * field[ix, iy + 1]
            + tx * ty * field[ix + 1, iy + 1])


# Per-cohort per-sequence quantities
F_3di_psyc = bilerp(F_3di_field, z0_3di, z1_3di, psyc_di)
F_3di_ther = bilerp(F_3di_field, z0_3di, z1_3di, ther_di)
H_3di_psyc = bilerp(H_3di_grid_arr, z0_3di_H, z1_3di_H, psyc_di)
H_3di_ther = bilerp(H_3di_grid_arr, z0_3di_H, z1_3di_H, ther_di)
dH_psyc = H_psyc - H_3di_psyc
dH_ther = H_ther - H_3di_ther
ratio_psyc = np.log10((F_psyc + 1e-12) / (F_3di_psyc + 1e-12))
ratio_ther = np.log10((F_ther + 1e-12) / (F_3di_ther + 1e-12))

# Natural-cloud quantities
cache_mdh = np.load(OUT_DATA / "cache_paired_mus.npz", allow_pickle=True)
mu_aa_cache  = cache_mdh["MDH_mu_aa"]
mu_3di_cache = cache_mdh["MDH_mu_3di"]
F_nat_aa  = bilerp(F_field, z0_grid, z1_grid, mu_aa_cache)
F_nat_3di = bilerp(F_3di_field, z0_3di, z1_3di, mu_3di_cache)
H_nat_aa  = bilerp(H_field, z0_grid, z1_grid, mu_aa_cache)
H_nat_3di = bilerp(H_3di_grid_arr, z0_3di_H, z1_3di_H, mu_3di_cache)
dH_nat = H_nat_aa - H_nat_3di
ratio_nat = np.log10((F_nat_aa + 1e-12) / (F_nat_3di + 1e-12))

# Density contour of the natural-sequence cloud as a "landscape" background
# (filled only where natural cloud has support; outermost level is opaque-edge)
from scipy.stats import gaussian_kde
nat_xy = np.vstack([dH_nat, ratio_nat])
nat_kde = gaussian_kde(nat_xy)
x_range = (dH_nat.min() - 200, dH_nat.max() + 200)
y_range = (ratio_nat.min() - 0.5, ratio_nat.max() + 0.5)
xg, yg = np.meshgrid(np.linspace(*x_range, 80),
                      np.linspace(*y_range, 80))
zg = nat_kde(np.vstack([xg.ravel(), yg.ravel()])).reshape(xg.shape)
# Start levels above ~5% of max density so contours hug the cloud
levels = np.linspace(zg.max() * 0.07, zg.max(), 8)
axD.contourf(xg, yg, zg, levels=levels, cmap="Greys", alpha=0.55,
              zorder=0, extend="neither")
axD.scatter(dH_nat, ratio_nat, s=10, c="0.40", alpha=0.55,
            edgecolor="none", label="natural MDH sequences", zorder=2)
axD.scatter(dH_psyc, ratio_psyc, s=22, c=COL_PSYC,
            edgecolor="black", linewidth=0.4, alpha=0.95,
            label="psychrophilic cohort", zorder=5)
axD.scatter(dH_ther, ratio_ther, s=22, c=COL_THER,
            edgecolor="black", linewidth=0.4, alpha=0.95,
            label="thermo/meso cohort", zorder=5)
axD.axhline(0, ls=":", color="0.4", lw=0.7)
axD.axvline(0, ls=":", color="0.4", lw=0.7)
axD.set_xlabel(r"$H_\mathrm{AA} - H_\mathrm{3Di}$", fontsize=12)
axD.set_ylabel(r"$\log_{10}\, F_\mathrm{AA}/F_\mathrm{3Di}$", fontsize=12)
axD.tick_params(labelsize=11)
add_panel_label(axD, "D", r"Interplay of AA and 3Di coevolution")
# Legend outside Panel D as a horizontal row below the x-axis so it
# doesn't collide with the in-plot directional arrows.
axD.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
           ncol=3, fontsize=9, frameon=True, framealpha=0.9,
           markerscale=1.2, handletextpad=0.5, columnspacing=1.0)
axD.spines["top"].set_visible(False); axD.spines["right"].set_visible(False)

# Directional arrows + biological captions, all four placed inside
# Panel D. Arrows are shorter and equal-length per axis; text labels
# hug the corresponding spine on the other side of each arrow.
ARROW_COL = "0.30"
TEXT_COL = "black"
BBOX_KW = dict(boxstyle="round,pad=0.15", facecolor="white",
               edgecolor="none", alpha=0.85)

# x-axis: arrow at y = 0.08, label at y = 0.025 (closer to bottom spine)
y_x_arrow = 0.08
y_x_text = 0.025
# Equal length 0.24 on each side
axD.annotate("", xy=(0.06, y_x_arrow), xytext=(0.30, y_x_arrow),
             xycoords="axes fraction",
             arrowprops=dict(arrowstyle="->", color=ARROW_COL, lw=1.3))
axD.text(0.18, y_x_text, "structure-unusual",
         transform=axD.transAxes, ha="center", va="bottom",
         fontsize=9.5, style="italic", color=TEXT_COL, bbox=BBOX_KW)
axD.annotate("", xy=(0.94, y_x_arrow), xytext=(0.70, y_x_arrow),
             xycoords="axes fraction",
             arrowprops=dict(arrowstyle="->", color=ARROW_COL, lw=1.3))
axD.text(0.82, y_x_text, "sequence-unusual",
         transform=axD.transAxes, ha="center", va="bottom",
         fontsize=9.5, style="italic", color=TEXT_COL, bbox=BBOX_KW)

# y-axis: arrow brought close to the rotated text (tight gap on each side)
# Equal length 0.24 on each side
x_y_arrow = 0.062
x_y_text = 0.025
axD.annotate("", xy=(x_y_arrow, 0.94), xytext=(x_y_arrow, 0.70),
             xycoords="axes fraction",
             arrowprops=dict(arrowstyle="->", color=ARROW_COL, lw=1.3))
axD.text(x_y_text, 0.82, "high seq variation",
         transform=axD.transAxes, ha="center", va="center",
         rotation=90, fontsize=9.5, style="italic", color=TEXT_COL,
         bbox=BBOX_KW)
axD.annotate("", xy=(x_y_arrow, 0.20), xytext=(x_y_arrow, 0.44),
             xycoords="axes fraction",
             arrowprops=dict(arrowstyle="->", color=ARROW_COL, lw=1.3))
axD.text(x_y_text, 0.32, "high struc variation",
         transform=axD.transAxes, ha="center", va="center",
         rotation=90, fontsize=9.5, style="italic", color=TEXT_COL,
         bbox=BBOX_KW)


out_pdf = OUTDIR / "Fig2_MDH_cohort.pdf"
out_png = OUTDIR / "Fig2_MDH_cohort.png"
fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")


summary = dict(
    n_psyc=int(n_psyc), n_ther=int(n_ther),
    W2_AA=float(w2_aa), W2_3Di=float(w2_di),
    tm_within_psyc_mean=float(tm_within_p.mean()),
    tm_within_thermo_mean=float(tm_within_t.mean()),
    tm_between_mean=float(tm_between.mean()),
    id_within_psyc_mean=float(id_within_p.mean()),
    id_within_thermo_mean=float(id_within_t.mean()),
    id_between_mean=float(id_between.mean()),
)
(OUT_DATA / "35_mdh_cohort_summary.json").write_text(json.dumps(summary, indent=2))
print("Saved cohort summary JSON.")
