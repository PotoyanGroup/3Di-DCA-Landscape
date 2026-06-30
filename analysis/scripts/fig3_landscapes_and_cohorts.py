"""Main-text Figure 3: family-level 3Di landscapes plus within-family AA-vs-3Di cohort studies.

Builds a 2x2 figure. The top row draws the Globin and TRPM 3Di-VAE
latent-entropy landscapes and overlays each family's natural
subfamilies/subtypes (z1,z2 scatter colored by class), showing the 3Di
latent registers natural taxonomy. The bottom row places targeted
within-family cohorts on the AA-vs-3Di plane (dH = H_AA - H_3Di on x,
log10(F_AA/F_3Di) on y, with a natural-family background cloud and
median-marker stars): Globin contrasts Subunit beta vs Cytoglobin,
and TRPM8 contrasts homeotherm vs ectotherm sequences (ectotherms
identified by OS= genus from the FASTA description; invertebrate
genera excluded). Per-cohort entropy and pullback-volume features are
sampled from cached grids by bilinear interpolation, and separation
z-scores (Cohen's-d-style, in AA, 3Di, and plane spaces) are annotated
on each cohort panel.

Inputs (paths resolved through analyses/scripts/_paths.py unless noted):
  - GLOBIN_3DI_DIR / "zed_coordinates_with_classes.csv" : Globin 3Di latent
      coordinates with class labels (subfamily scatter + cohort accessions)
  - GLOBIN_3DI_DIR / "all_grid.pkl" : Globin 3Di latent-entropy grid points
  - GLOBIN_AA_DIR / "all_grid.pkl" : Globin AA latent-entropy grid points
  - OUT_DATA / "08_pullback_metric_aa_globin.npz" : Globin AA pullback-metric
      volume field (vol, z0_axis, z1_axis)
  - OUT_DATA / "08_pullback_metric_3di_globin.npz" : Globin 3Di pullback-metric
      volume field (vol, z0_axis, z1_axis)
  - OUT_DATA / "04a_globin_aa_per_accession.npz" : cached Globin AA latent means
      keyed by accession (accessions, mu)
  - TRPM_3DI_DIR / "zed_coordinates_with_classes.csv" : TRPM 3Di latent
      coordinates with class labels + descriptions
  - TRPM_3DI_DIR / "all_grid.pkl" : TRPM 3Di latent-entropy grid points
  - TRPM_AA_DIR / "all_grid.pkl" : TRPM AA latent-entropy grid points
  - OUT_DATA / "08_pullback_metric_aa_trpm.npz" : TRPM AA pullback-metric
      volume field (vol, z0_axis, z1_axis)
  - OUT_DATA / "08_pullback_metric_3di_trpm.npz" : TRPM 3Di pullback-metric
      volume field (vol, z0_axis, z1_axis)
  - OUT_DATA / "04a_trpm_aa_per_accession.npz" : cached TRPM AA latent means
      keyed by accession (accessions, mu)

Outputs:
  - OUT_FIG / "04f_fig3_main_4panel.pdf" : the 2x2 Figure 3 (vector)
  - OUT_FIG / "04f_fig3_main_4panel.png" : the 2x2 Figure 3 (160 dpi raster)

Usage:
  python analyses/scripts/fig3_landscapes_and_cohorts.py
  No command-line arguments.
"""
from pathlib import Path
import sys
import os
import re
import csv
import pickle
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt

from _paths import OUT_FIG, OUT_DATA  # noqa
from _paths import GLOBIN_AA_DIR, GLOBIN_3DI_DIR, TRPM_AA_DIR, TRPM_3DI_DIR  # noqa: E402


GLOBIN_ZED = GLOBIN_3DI_DIR / "zed_coordinates_with_classes.csv"
GLOBIN_3DI_GRID = GLOBIN_3DI_DIR / "all_grid.pkl"
GLOBIN_AA_GRID = GLOBIN_AA_DIR / "all_grid.pkl"
GLOBIN_F_AA = OUT_DATA / "08_pullback_metric_aa_globin.npz"
GLOBIN_F_3DI = OUT_DATA / "08_pullback_metric_3di_globin.npz"
GLOBIN_AA_CACHE = OUT_DATA / "04a_globin_aa_per_accession.npz"

TRPM_ZED = TRPM_3DI_DIR / "zed_coordinates_with_classes.csv"
TRPM_3DI_GRID = TRPM_3DI_DIR / "all_grid.pkl"
TRPM_AA_GRID = TRPM_AA_DIR / "all_grid.pkl"
TRPM_F_AA = OUT_DATA / "08_pullback_metric_aa_trpm.npz"
TRPM_F_3DI = OUT_DATA / "08_pullback_metric_3di_trpm.npz"
TRPM_AA_CACHE = OUT_DATA / "04a_trpm_aa_per_accession.npz"

ACC_RX = re.compile(r"(?:^|[_|])([OPQ][0-9][A-Z0-9]{3}[0-9]|"
                    r"[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})")

GLOBIN_CLASSES = ["Flavohemoglobin", "Subunit Alpha", "Subunit Beta",
                  "Cytoglobin", "Neuroglobin", "Myoglobin",
                  "Bacterial hemoglobin", "Leghemoglobin"]
GLOBIN_SHORT = {"Flavohemoglobin": "Flavo", "Subunit Alpha": r"$\alpha$",
                "Subunit Beta": r"$\beta$", "Cytoglobin": "Cyto",
                "Neuroglobin": "Neuro", "Myoglobin": "Myo",
                "Bacterial hemoglobin": "Bact",
                "Leghemoglobin": "Lego"}

TRPM_CLASSES = [f"member {i}" for i in range(1, 9)]
TRPM_SHORT = {f"member {i}": f"TRPM{i}" for i in range(1, 9)}

ECTOTHERM_GENERA = {
    "Gopherus", "Anolis", "Alligator", "Platysternon", "Pelodiscus",
    "Xenopus", "Latimeria",
}
INVERT_GENERA = {"Crassostrea", "Mizuhopecten"}


def extract_accession(s):
    m = ACC_RX.search(str(s))
    return m.group(1) if m else None


def load_grid_H(pkl_path):
    pts = pickle.load(open(pkl_path, "rb"))
    z0 = np.unique(pts[:, 0])
    z1 = np.unique(pts[:, 1])
    H = pts[:, 2].reshape(z0.size, z1.size)
    return H, z0, z1


def bilerp(field, ax0, ax1, points):
    px = np.clip(points[:, 0], ax0[0] + 1e-9, ax0[-1] - 1e-9)
    py = np.clip(points[:, 1], ax1[0] + 1e-9, ax1[-1] - 1e-9)
    Nz0, Nz1 = len(ax0), len(ax1)
    fx = (px - ax0[0]) / (ax0[-1] - ax0[0]) * (Nz0 - 1)
    fy = (py - ax1[0]) / (ax1[-1] - ax1[0]) * (Nz1 - 1)
    ix = np.clip(np.floor(fx).astype(int), 0, Nz0 - 2)
    iy = np.clip(np.floor(fy).astype(int), 0, Nz1 - 2)
    tx = fx - ix
    ty = fy - iy
    return ((1 - tx) * (1 - ty) * field[ix, iy]
            + tx * (1 - ty) * field[ix + 1, iy]
            + (1 - tx) * ty * field[ix, iy + 1]
            + tx * ty * field[ix + 1, iy + 1])


def load_subfamily_3di(zed_csv, classes):
    """Returns {class: array (N,2)}."""
    out = {c: [] for c in classes}
    with open(zed_csv) as fh:
        for r in csv.DictReader(fh):
            cls = r["class"].strip()
            if cls not in classes:
                continue
            try:
                z = (float(r["z1"]), float(r["z2"]))
            except (ValueError, KeyError):
                continue
            out[cls].append(z)
    return {k: np.array(v) for k, v in out.items()}


def panel_3di_landscape(ax, H, z0, z1, subfam_dict, classes,
                         class_short, title, scatter_alpha=0.80):
    extent = (z0.min(), z0.max(), z1.min(), z1.max())
    im = ax.imshow(H.T, origin="lower", extent=extent,
                    cmap="viridis", aspect="auto",
                    interpolation="bilinear")
    cmap = plt.cm.tab10
    for i, c in enumerate(classes):
        pts = subfam_dict.get(c, np.zeros((0, 2)))
        if len(pts) == 0:
            continue
        color = cmap(i / max(len(classes) - 1, 1))
        ax.scatter(pts[:, 0], pts[:, 1], s=22, color=color,
                    edgecolor="black", linewidth=0.4,
                    alpha=scatter_alpha, zorder=4,
                    label=class_short.get(c, c))
    ax.set_xlabel(r"$z_1$", fontsize=11)
    ax.set_ylabel(r"$z_2$", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=8, frameon=True,
               handletextpad=0.3, columnspacing=0.4, ncol=2)
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
    cb.set_label(r"$H_\mathrm{3Di}$", fontsize=9)
    cb.ax.tick_params(labelsize=7)


def w2_cohort(z1, z2):
    mu1, mu2 = z1.mean(0), z2.mean(0)
    sd1, sd2 = z1.std(0), z2.std(0)
    d = float(np.linalg.norm(mu1 - mu2))
    s = float(np.sqrt(0.5 * (sd1**2 + sd2**2)).mean())
    return d, s, d / max(s, 1e-9)


def panel_plane(ax, dH_a, logF_a, dH_b, logF_b,
                nat_dH, nat_logF,
                col_a, col_b, label_a, label_b, title,
                stats_text=None, xlim=None, ylim=None,
                legend_loc="upper left"):
    ax.scatter(nat_dH, nat_logF, s=3, color="0.7", alpha=0.18,
                zorder=1, label="natural family cloud")
    ax.scatter(dH_a, logF_a, s=16, color=col_a,
                edgecolor="black", linewidth=0.2, alpha=0.55,
                zorder=4, label=f"{label_a} (n={len(dH_a)})")
    ax.scatter(dH_b, logF_b, s=16, color=col_b,
                edgecolor="black", linewidth=0.2, alpha=0.55,
                zorder=4, label=f"{label_b} (n={len(dH_b)})")
    ca = (float(np.median(dH_a)), float(np.median(logF_a)))
    cb = (float(np.median(dH_b)), float(np.median(logF_b)))
    ax.scatter(*ca, s=400, marker="*", facecolor=col_a,
                edgecolor="black", linewidth=1.5, zorder=10)
    ax.scatter(*cb, s=400, marker="*", facecolor=col_b,
                edgecolor="black", linewidth=1.5, zorder=10)
    ax.axhline(0, color="0.55", ls="--", lw=0.7, zorder=1)
    ax.axvline(0, color="0.55", ls="--", lw=0.7, zorder=1)
    ax.set_xlabel(r"$\Delta H = H_\mathrm{AA} - H_\mathrm{3Di}$",
                   fontsize=11)
    ax.set_ylabel(r"$\log_{10}(F_\mathrm{AA} / F_\mathrm{3Di})$",
                   fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc=legend_loc, fontsize=8, frameon=True)
    ax.grid(alpha=0.3)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    # Axis-end labels matching Fig 2D convention (italic, arrows)
    ax.text(0.02, 0.04, r"$\leftarrow$ structure-unusual",
             transform=ax.transAxes, ha="left", va="bottom",
             fontsize=7.5, fontstyle="italic", color="0.30")
    ax.text(0.98, 0.04, r"sequence-unusual $\rightarrow$",
             transform=ax.transAxes, ha="right", va="bottom",
             fontsize=7.5, fontstyle="italic", color="0.30")
    ax.text(0.02, 0.96, r"$\uparrow$ high seq variation",
             transform=ax.transAxes, ha="left", va="top",
             fontsize=7.5, fontstyle="italic", color="0.30")
    ax.text(0.02, 0.40, r"$\downarrow$ high struc variation",
             transform=ax.transAxes, ha="left", va="top",
             fontsize=7.5, fontstyle="italic", color="0.30")
    if stats_text:
        ax.text(0.98, 0.96, stats_text,
                 transform=ax.transAxes,
                 ha="right", va="top",
                 fontsize=7.5, color="0.30",
                 bbox=dict(boxstyle="round,pad=0.25",
                           fc="white", ec="0.55", lw=0.5, alpha=0.92))


def build_globin_cohorts(class_a="Subunit Beta", class_b="Cytoglobin"):
    aa_cache = np.load(GLOBIN_AA_CACHE, allow_pickle=True)
    acc_to_aa = dict(zip(aa_cache["accessions"].tolist(), aa_cache["mu"]))
    rows_a, rows_b, all_rows = [], [], []
    with open(GLOBIN_ZED) as fh:
        for r in csv.DictReader(fh):
            cls = r["class"].strip()
            acc = extract_accession(r["id"])
            if not acc or acc not in acc_to_aa:
                continue
            try:
                z3 = (float(r["z1"]), float(r["z2"]))
            except (ValueError, KeyError):
                continue
            entry = (acc, np.asarray(acc_to_aa[acc]), np.asarray(z3))
            all_rows.append(entry)
            if cls == class_a:
                rows_a.append(entry)
            elif cls == class_b:
                rows_b.append(entry)
    return rows_a, rows_b, all_rows


def build_trpm_cohorts():
    aa_cache = np.load(TRPM_AA_CACHE, allow_pickle=True)
    acc_to_aa = dict(zip(aa_cache["accessions"].tolist(), aa_cache["mu"]))
    rows_h, rows_e, all_rows = [], [], []
    with open(TRPM_ZED) as fh:
        for r in csv.DictReader(fh):
            if r["class"].strip() != "member 8":
                continue
            acc = extract_accession(r["id"])
            if not acc or acc not in acc_to_aa:
                continue
            try:
                z3 = (float(r["z1"]), float(r["z2"]))
            except (ValueError, KeyError):
                continue
            m = re.search(r"OS=(\S+)", r["description"])
            if not m:
                continue
            genus = m.group(1)
            if genus in INVERT_GENERA:
                continue
            entry = (acc, np.asarray(acc_to_aa[acc]),
                     np.asarray(z3), genus)
            all_rows.append(entry)
            if genus in ECTOTHERM_GENERA:
                rows_e.append(entry)
            else:
                rows_h.append(entry)
    return rows_h, rows_e, all_rows


def compute_features(rows, H_aa, z0_aa, z1_aa, H_di, z0_di, z1_di,
                     F_aa, z0_F_aa, z1_F_aa, F_di, z0_F_di, z1_F_di,
                     dh_offset):
    z_aa = np.array([r[1] for r in rows])
    z_di = np.array([r[2] for r in rows])
    H_aa_v = bilerp(H_aa, z0_aa, z1_aa, z_aa)
    H_di_v = bilerp(H_di, z0_di, z1_di, z_di)
    F_aa_v = bilerp(F_aa, z0_F_aa, z1_F_aa, z_aa)
    F_di_v = bilerp(F_di, z0_F_di, z1_F_di, z_di)
    dH = H_aa_v - H_di_v - dh_offset
    logF = (np.log10(np.maximum(F_aa_v, 1e-12))
            - np.log10(np.maximum(F_di_v, 1e-12)))
    return dH, logF


def main():
    # ==== Top row: Family 3Di landscapes ====
    H_di_g, z0_di_g, z1_di_g = load_grid_H(GLOBIN_3DI_GRID)
    H_di_t, z0_di_t, z1_di_t = load_grid_H(TRPM_3DI_GRID)
    subfam_g = load_subfamily_3di(GLOBIN_ZED, GLOBIN_CLASSES)
    subfam_t = load_subfamily_3di(TRPM_ZED, TRPM_CLASSES)

    # ==== Bottom row: cohort study features ====
    # Globin
    g_a, g_b, g_all = build_globin_cohorts("Subunit Beta", "Cytoglobin")
    H_aa_g, z0_aa_g, z1_aa_g = load_grid_H(GLOBIN_AA_GRID)
    F_aa_g_d = np.load(GLOBIN_F_AA, allow_pickle=True)
    F_di_g_d = np.load(GLOBIN_F_3DI, allow_pickle=True)
    g_F_aa = F_aa_g_d["vol"]; g_z0F_aa = F_aa_g_d["z0_axis"]
    g_z1F_aa = F_aa_g_d["z1_axis"]
    g_F_di = F_di_g_d["vol"]; g_z0F_di = F_di_g_d["z0_axis"]
    g_z1F_di = F_di_g_d["z1_axis"]
    nat_zaa_g = np.array([r[1] for r in g_all])
    nat_zdi_g = np.array([r[2] for r in g_all])
    nat_H_aa_g = bilerp(H_aa_g, z0_aa_g, z1_aa_g, nat_zaa_g)
    nat_H_di_g = bilerp(H_di_g, z0_di_g, z1_di_g, nat_zdi_g)
    g_offset = float(np.median(nat_H_aa_g - nat_H_di_g))
    nat_F_aa_g = bilerp(g_F_aa, g_z0F_aa, g_z1F_aa, nat_zaa_g)
    nat_F_di_g = bilerp(g_F_di, g_z0F_di, g_z1F_di, nat_zdi_g)
    nat_dH_g = nat_H_aa_g - nat_H_di_g - g_offset
    nat_logF_g = (np.log10(np.maximum(nat_F_aa_g, 1e-12))
                  - np.log10(np.maximum(nat_F_di_g, 1e-12)))
    dH_ga, logF_ga = compute_features(g_a, H_aa_g, z0_aa_g, z1_aa_g,
        H_di_g, z0_di_g, z1_di_g, g_F_aa, g_z0F_aa, g_z1F_aa,
        g_F_di, g_z0F_di, g_z1F_di, g_offset)
    dH_gb, logF_gb = compute_features(g_b, H_aa_g, z0_aa_g, z1_aa_g,
        H_di_g, z0_di_g, z1_di_g, g_F_aa, g_z0F_aa, g_z1F_aa,
        g_F_di, g_z0F_di, g_z1F_di, g_offset)

    # TRPM8
    t_h, t_e, t_all = build_trpm_cohorts()
    H_aa_t, z0_aa_t, z1_aa_t = load_grid_H(TRPM_AA_GRID)
    F_aa_t_d = np.load(TRPM_F_AA, allow_pickle=True)
    F_di_t_d = np.load(TRPM_F_3DI, allow_pickle=True)
    t_F_aa = F_aa_t_d["vol"]; t_z0F_aa = F_aa_t_d["z0_axis"]
    t_z1F_aa = F_aa_t_d["z1_axis"]
    t_F_di = F_di_t_d["vol"]; t_z0F_di = F_di_t_d["z0_axis"]
    t_z1F_di = F_di_t_d["z1_axis"]
    nat_zaa_t = np.array([r[1] for r in t_all])
    nat_zdi_t = np.array([r[2] for r in t_all])
    nat_H_aa_t = bilerp(H_aa_t, z0_aa_t, z1_aa_t, nat_zaa_t)
    nat_H_di_t = bilerp(H_di_t, z0_di_t, z1_di_t, nat_zdi_t)
    t_offset = float(np.median(nat_H_aa_t - nat_H_di_t))
    nat_F_aa_t = bilerp(t_F_aa, t_z0F_aa, t_z1F_aa, nat_zaa_t)
    nat_F_di_t = bilerp(t_F_di, t_z0F_di, t_z1F_di, nat_zdi_t)
    nat_dH_t = nat_H_aa_t - nat_H_di_t - t_offset
    nat_logF_t = (np.log10(np.maximum(nat_F_aa_t, 1e-12))
                  - np.log10(np.maximum(nat_F_di_t, 1e-12)))
    dH_th, logF_th = compute_features([r[:3] for r in t_h],
        H_aa_t, z0_aa_t, z1_aa_t, H_di_t, z0_di_t, z1_di_t,
        t_F_aa, t_z0F_aa, t_z1F_aa, t_F_di, t_z0F_di, t_z1F_di,
        t_offset)
    dH_te, logF_te = compute_features([r[:3] for r in t_e],
        H_aa_t, z0_aa_t, z1_aa_t, H_di_t, z0_di_t, z1_di_t,
        t_F_aa, t_z0F_aa, t_z1F_aa, t_F_di, t_z0F_di, t_z1F_di,
        t_offset)

    # Stats text
    def stat_string(z_aa, z_di, z_pl):
        return (rf"$z_\mathrm{{AA}}={z_aa:.2f},\ "
                rf"z_\mathrm{{3Di}}={z_di:.2f},\ "
                rf"z_\mathrm{{plane}}={z_pl:.2f}$")

    z_aa_g_arr_a = np.array([r[1] for r in g_a])
    z_aa_g_arr_b = np.array([r[1] for r in g_b])
    z_di_g_arr_a = np.array([r[2] for r in g_a])
    z_di_g_arr_b = np.array([r[2] for r in g_b])
    _, _, sz_aa_g = w2_cohort(z_aa_g_arr_a, z_aa_g_arr_b)
    _, _, sz_di_g = w2_cohort(z_di_g_arr_a, z_di_g_arr_b)
    _, _, sz_pl_g = w2_cohort(np.column_stack([dH_ga, logF_ga]),
                               np.column_stack([dH_gb, logF_gb]))

    z_aa_t_arr_h = np.array([r[1] for r in t_h])
    z_aa_t_arr_e = np.array([r[1] for r in t_e])
    z_di_t_arr_h = np.array([r[2] for r in t_h])
    z_di_t_arr_e = np.array([r[2] for r in t_e])
    _, _, sz_aa_t = w2_cohort(z_aa_t_arr_h, z_aa_t_arr_e)
    _, _, sz_di_t = w2_cohort(z_di_t_arr_h, z_di_t_arr_e)
    _, _, sz_pl_t = w2_cohort(np.column_stack([dH_th, logF_th]),
                               np.column_stack([dH_te, logF_te]))

    # ==== Plot ====
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.0),
                              gridspec_kw={"wspace": 0.32,
                                           "hspace": 0.50})

    panel_3di_landscape(axes[0, 0], H_di_g, z0_di_g, z1_di_g,
                         subfam_g, GLOBIN_CLASSES, GLOBIN_SHORT,
                         "(A) Globin: 3Di-VAE")
    panel_3di_landscape(axes[0, 1], H_di_t, z0_di_t, z1_di_t,
                         subfam_t, TRPM_CLASSES, TRPM_SHORT,
                         "(B) TRPM: 3Di-VAE")
    # Cohort colors matched to A/B legend colors:
    # Panel A: Subunit beta -> tab10[2] (green), Cytoglobin -> tab10[4] (purple)
    # Panel B: TRPM8 -> tab10[9] (cyan); ectotherm gets a contrast color
    col_beta = plt.cm.tab10(2 / 7)   # green, matches Subunit beta in (A)
    col_cyto = plt.cm.tab10(3 / 7)   # purple, matches Cytoglobin in (A)
    col_homo = plt.cm.tab10(7 / 7)   # cyan, matches TRPM8 in (B)
    col_ecto = plt.cm.tab10(3.0/10.0)  # red, contrast within TRPM8

    panel_plane(axes[1, 0], dH_ga, logF_ga, dH_gb, logF_gb,
                 nat_dH_g, nat_logF_g,
                 col_beta, col_cyto,
                 r"Subunit $\beta$", "Cytoglobin",
                 r"(C) Globin: Subunit $\beta$ vs Cytoglobin",
                 stats_text=stat_string(sz_aa_g, sz_di_g, sz_pl_g))
    panel_plane(axes[1, 1], dH_th, logF_th, dH_te, logF_te,
                 nat_dH_t, nat_logF_t,
                 col_homo, col_ecto,
                 "homeotherm", "ectotherm",
                 r"(D) TRPM8: homeotherm vs ectotherm",
                 stats_text=stat_string(sz_aa_t, sz_di_t, sz_pl_t),
                 xlim=(-500, 700))

    fig.tight_layout()

    out_pdf = OUT_FIG / "04f_fig3_main_4panel.pdf"
    out_png = OUT_FIG / "04f_fig3_main_4panel.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_pdf}")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
