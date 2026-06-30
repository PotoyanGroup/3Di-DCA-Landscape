"""Supplementary Figure S6: per-latent Fisher determinant vs DCA Hamiltonian planes (F vs H) for the AA and 3Di latents of MDH.

Builds a 1x2 figure of F-vs-H scatter planes that complements main-text
Fig 2D (which uses AA-vs-3Di difference and ratio axes); each SI panel here
gives the single-latent view that Fig 2D summarizes. Panel A plots the
AA-VAE pullback Fisher volume F_AA against the AA DCA Hamiltonian H_AA on
the AA latent; Panel B plots the 3Di-VAE Fisher volume F_3Di against the
3Di DCA Hamiltonian H_3Di on the 3Di latent (both fields on a matching
60x60 grid). Each panel layers the full latent grid (light gray), the
natural MDH MSA cloud (gray), and the psychrophilic and thermo/meso
cohorts (blue/red), with cohort and natural-sequence values bilinearly
interpolated onto the precomputed fields; the F axis is log-scaled.

Inputs (paths resolved through analyses/scripts/_paths.py unless noted):
  - OUT : output root; OUTDIR is OUT.parent / "figures_for_manuscript" / "SI_fh_planes"
  - OUT_DATA : cached-data directory holding the npz field/latent caches below
  - MDH_CLUSTER_DATA (aliased BASE) : root for MDH cohort cluster-coordinate files
  - BASE / "psycrophilic_cluster" / "cluster_coordinates_AA.txt" : psychrophilic cohort AA latent coords
  - BASE / "psycrophilic_cluster" / "cluster_coordinates_3DI.txt" : psychrophilic cohort 3Di latent coords
  - BASE / "thermo_meso_cluster" / "cluster_coordinates_AA.txt" : thermo/meso cohort AA latent coords
  - BASE / "thermo_meso_cluster" / "cluster_coordinates_3DI.txt" : thermo/meso cohort 3Di latent coords
  - OUT_DATA / "16_fisher_dca_joint_aa_mdh.npz" : AA Hamiltonian field H, Fisher field F, and z0/z1 axes
  - OUT_DATA / "08_pullback_metric_3di_mdh.npz" : 3Di pullback Fisher volume "vol" and z0_axis/z1_axis (60x60)
  - OUT_DATA / "17_fisher_dca_joint_3di_plus_bootstrap.npz" : 3Di Hamiltonian field H_3di interpolated to the 60x60 grid
  - OUT_DATA / "cache_paired_mus.npz" : cached paired natural-MDH latents (MDH_mu_aa, MDH_mu_3di)

Outputs:
  - OUTDIR / "SI_fh_planes_aa_3di.pdf" : the 1x2 F-vs-H figure (PDF, dpi 200)
  - OUTDIR / "SI_fh_planes_aa_3di.png" : the 1x2 F-vs-H figure (PNG, dpi 200)

Usage:
  python analyses/scripts/S6_fh_planes.py
  No command-line arguments.
"""
import ast, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

from _paths import OUT, OUT_DATA

OUTDIR = OUT.parent / "figures_for_manuscript" / "SI_fh_planes"
OUTDIR.mkdir(parents=True, exist_ok=True)

from _paths import MDH_CLUSTER_DATA as BASE  # noqa: E402

COL_PSYC = "#1f77b4"
COL_THER = "#d62728"


def load_coords(p):
    return np.array(list(ast.literal_eval(p.read_text()).values()))


psyc_aa = load_coords(BASE / "psycrophilic_cluster" / "cluster_coordinates_AA.txt")
psyc_di = load_coords(BASE / "psycrophilic_cluster" / "cluster_coordinates_3DI.txt")
ther_aa = load_coords(BASE / "thermo_meso_cluster"  / "cluster_coordinates_AA.txt")
ther_di = load_coords(BASE / "thermo_meso_cluster"  / "cluster_coordinates_3DI.txt")


# AA fields
d_aa = np.load(OUT_DATA / "16_fisher_dca_joint_aa_mdh.npz", allow_pickle=True)
H_aa_field = d_aa["H"]; F_aa_field = d_aa["F"]
z0_aa = d_aa["z0"]; z1_aa = d_aa["z1"]
Z0a, Z1a = np.meshgrid(z0_aa, z1_aa, indexing="ij")

# 3Di fields — both at the 60x60 resolution that matches the AA panel
d_3di_F = np.load(OUT_DATA / "08_pullback_metric_3di_mdh.npz", allow_pickle=True)
F_3di_field = d_3di_F["vol"]
z0_3di = d_3di_F["z0_axis"]; z1_3di = d_3di_F["z1_axis"]
# H_3di on the same 60x60 grid as F_3di (already interpolated by script 17)
d_17 = np.load(OUT_DATA / "17_fisher_dca_joint_3di_plus_bootstrap.npz",
                allow_pickle=True)
H_3di_field = d_17["H_3di"]
z0_3di_H = z0_3di; z1_3di_H = z1_3di


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


# Natural cloud (cached paired latents)
cache = np.load(OUT_DATA / "cache_paired_mus.npz", allow_pickle=True)
mu_aa_nat  = cache["MDH_mu_aa"]
mu_3di_nat = cache["MDH_mu_3di"]


# =====================================================================
# Figure
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0),
                          gridspec_kw=dict(wspace=0.30))


def add_panel_label(ax, letter, descriptor):
    ax.text(-0.13, 1.05, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", color="black",
            va="bottom", ha="left")
    ax.set_title(descriptor, fontsize=11, color="black", pad=6)


def panel(ax, H_field, F_field, ax_z0, ax_z1, mu_nat, mu_psyc, mu_ther,
          letter, descriptor):
    # Background grid (downsample for visual)
    Z0m, Z1m = np.meshgrid(ax_z0, ax_z1, indexing="ij")
    ax.scatter(H_field.flatten(), F_field.flatten(),
               s=4, c="0.82", alpha=0.45, edgecolor="none",
               label="latent grid", zorder=0)
    # Natural cloud
    H_n = bilerp(H_field, ax_z0, ax_z1, mu_nat)
    F_n = bilerp(F_field, ax_z0, ax_z1, mu_nat)
    ax.scatter(H_n, F_n, s=12, c="0.45", alpha=0.55,
               edgecolor="none", label="natural MDH sequences", zorder=2)
    # Cohorts
    H_p = bilerp(H_field, ax_z0, ax_z1, mu_psyc)
    F_p = bilerp(F_field, ax_z0, ax_z1, mu_psyc)
    H_t = bilerp(H_field, ax_z0, ax_z1, mu_ther)
    F_t = bilerp(F_field, ax_z0, ax_z1, mu_ther)
    ax.scatter(H_p, F_p, s=22, c=COL_PSYC,
                edgecolor="black", linewidth=0.4, alpha=0.95,
                label="psychrophilic cohort", zorder=5)
    ax.scatter(H_t, F_t, s=22, c=COL_THER,
                edgecolor="black", linewidth=0.4, alpha=0.95,
                label="thermo/meso cohort", zorder=5)
    ax.set_yscale("log")
    add_panel_label(ax, letter, descriptor)
    ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.9,
                markerscale=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# A: AA latent F×H
panel(axes[0], H_aa_field, F_aa_field, z0_aa, z1_aa,
      mu_aa_nat, psyc_aa, ther_aa,
      "A", "AA latent")
axes[0].set_xlabel(r"$H_\mathrm{AA}$", fontsize=10)
axes[0].set_ylabel(r"$F_\mathrm{AA}$", fontsize=10)

# B: 3Di latent F×H — both fields on the same 60x60 grid now
panel(axes[1], H_3di_field, F_3di_field, z0_3di, z1_3di,
      mu_3di_nat, psyc_di, ther_di,
      "B", "3Di latent")
axes[1].set_xlabel(r"$H_\mathrm{3Di}$", fontsize=10)
axes[1].set_ylabel(r"$F_\mathrm{3Di}$", fontsize=10)


out_pdf = OUTDIR / "SI_fh_planes_aa_3di.pdf"
out_png = OUTDIR / "SI_fh_planes_aa_3di.png"
fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")
