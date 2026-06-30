"""Main-text Figure 2D (Globin/TRPM generalization): subfamily placement on the AA-vs-3Di plane.

Multi-cohort generalization of main-text Fig 2D (originally MDH Psy / T_M
cohorts). For each natural sequence that has a paired (z_AA, z_3Di)
embedding, the script evaluates the per-family DCA Hamiltonian H and the
pullback-metric volume element F = sqrt(det g) at the sequence's AA and
3Di latent coordinates (via bilinear interpolation of the landscape
grids), then plots each sequence on the (ΔH, log10(F_AA / F_3Di)) plane,
colored by annotated subfamily. Per family the ΔH cloud is recentered by
its median (an arbitrary Potts additive constant) so the natural cloud
sits at the origin. Two panels are produced side by side: Globin (8
classes) and TRPM (8 subtypes), each showing a light per-class scatter
plus prominent per-class median centroids with MAD error bars, and a
per-class summary is printed to stdout. 3Di latent coordinates come from
the zed CSVs; AA latent coordinates are obtained by re-encoding the AA
training MSA through the AA-VAE (cached per accession for re-runs).

Inputs (paths resolved through analyses/scripts/_paths.py unless noted):
  - GLOBIN_3DI_DIR / "zed_coordinates_with_classes.csv" : Globin 3Di latent coords + subfamily class labels
  - GLOBIN_AA_DIR / "globin_pfam_training_set.fasta" : Globin AA training MSA (re-encoded through AA-VAE)
  - GLOBIN_AA_DIR (SavedModel / tf_keras model) : Globin AA-VAE used to encode sequences to z_AA
  - GLOBIN_AA_DIR / "all_grid.pkl" : Globin AA latent-grid DCA Hamiltonian points
  - GLOBIN_3DI_DIR / "all_grid.pkl" : Globin 3Di latent-grid DCA Hamiltonian points
  - OUT_DATA / "08_pullback_metric_aa_globin.npz" : Globin AA pullback-metric field (vol, z0_axis, z1_axis)
  - OUT_DATA / "08_pullback_metric_3di_globin.npz" : Globin 3Di pullback-metric field (vol, z0_axis, z1_axis)
  - TRPM_3DI_DIR / "zed_coordinates_with_classes.csv" : TRPM 3Di latent coords + subtype class labels
  - TRPM_AA_DIR / "trmp8_training_set.fasta" : TRPM AA training MSA (re-encoded through AA-VAE)
  - TRPM_AA_DIR (SavedModel / tf_keras model) : TRPM AA-VAE used to encode sequences to z_AA
  - TRPM_AA_DIR / "all_grid.pkl" : TRPM AA latent-grid DCA Hamiltonian points
  - TRPM_3DI_DIR / "all_grid.pkl" : TRPM 3Di latent-grid DCA Hamiltonian points
  - OUT_DATA / "08_pullback_metric_aa_trpm.npz" : TRPM AA pullback-metric field (vol, z0_axis, z1_axis)
  - OUT_DATA / "08_pullback_metric_3di_trpm.npz" : TRPM 3Di pullback-metric field (vol, z0_axis, z1_axis)
  - OUT_DATA / "04a_globin_aa_per_accession.npz" : per-accession Globin z_AA cache (read if present, else written)
  - OUT_DATA / "04a_trpm_aa_per_accession.npz" : per-accession TRPM z_AA cache (read if present, else written)
  - model.generator.seq_code : one-hot residue alphabet used when encoding sequences (imported only on cache miss)

Outputs:
  - OUT_FIG / "04a_fig2d_globin_trpm.pdf" : two-panel Globin/TRPM (ΔH, log10 F ratio) figure
  - OUT_FIG / "04a_fig2d_globin_trpm.png" : PNG version of the same figure (dpi 160)
  - OUT_DATA / "04a_globin_aa_per_accession.npz" : Globin per-accession z_AA cache (written on first run)
  - OUT_DATA / "04a_trpm_aa_per_accession.npz" : TRPM per-accession z_AA cache (written on first run)

Usage:
  python analyses/scripts/fig2d_globin_trpm_subfamilies.py
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
from Bio import SeqIO

from _paths import OUT_FIG, OUT_DATA  # noqa
from _paths import GLOBIN_AA_DIR, GLOBIN_3DI_DIR, TRPM_AA_DIR, TRPM_3DI_DIR  # noqa: E402


GLOBIN_ZED = GLOBIN_3DI_DIR / "zed_coordinates_with_classes.csv"
GLOBIN_AA_FASTA = GLOBIN_AA_DIR / "globin_pfam_training_set.fasta"
GLOBIN_AA_VAE = GLOBIN_AA_DIR
GLOBIN_AA_GRID = GLOBIN_AA_DIR / "all_grid.pkl"
GLOBIN_3DI_GRID = GLOBIN_3DI_DIR / "all_grid.pkl"
GLOBIN_F_AA = OUT_DATA / "08_pullback_metric_aa_globin.npz"
GLOBIN_F_3DI = OUT_DATA / "08_pullback_metric_3di_globin.npz"

TRPM_ZED = TRPM_3DI_DIR / "zed_coordinates_with_classes.csv"
TRPM_AA_FASTA = TRPM_AA_DIR / "trmp8_training_set.fasta"
TRPM_AA_VAE = TRPM_AA_DIR
TRPM_AA_GRID = TRPM_AA_DIR / "all_grid.pkl"
TRPM_3DI_GRID = TRPM_3DI_DIR / "all_grid.pkl"
TRPM_F_AA = OUT_DATA / "08_pullback_metric_aa_trpm.npz"
TRPM_F_3DI = OUT_DATA / "08_pullback_metric_3di_trpm.npz"

CACHE = OUT_DATA / "cache_paired_mus.npz"
GLOBIN_AA_CACHE = OUT_DATA / "04a_globin_aa_per_accession.npz"
TRPM_AA_CACHE = OUT_DATA / "04a_trpm_aa_per_accession.npz"

ACC_RX = re.compile(r"(?:^|[_|])([OPQ][0-9][A-Z0-9]{3}[0-9]|"
                    r"[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})")

GLOBIN_CLASSES = ["Flavohemoglobin", "Subunit Alpha", "Subunit Beta",
                  "Cytoglobin", "Neuroglobin", "Myoglobin",
                  "Bacterial hemoglobin", "Leghemoglobin"]

TRPM_RENAME = {f"member {i}": f"TRPM{i}" for i in range(1, 9)}
TRPM_ORDER = [f"TRPM{i}" for i in range(1, 9)]


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


def load_zed_csv(csv_path):
    out = {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            acc = extract_accession(r["id"])
            if not acc:
                continue
            try:
                out[acc] = (r["class"].strip(),
                            (float(r["z1"]), float(r["z2"])))
            except (ValueError, KeyError):
                continue
    return out


def encode_family_aa(zed_data, fasta_path, vae_dir, cache_path,
                      L, q=23):
    """Generic AA encoder for any family. Caches by accession to disk."""
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        return dict(zip(d["accessions"].tolist(), d["mu"]))

    import tensorflow as tf
    from model.generator import seq_code

    aa_seqs = {}
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        acc = extract_accession(rec.id)
        if acc and acc in zed_data:
            s = str(rec.seq).upper()
            if len(s) == L:
                aa_seqs[acc] = s
    print(f"  matched {len(aa_seqs)} sequences for AA encoding")

    try:
        m = tf.saved_model.load(str(vae_dir))
    except (AttributeError, OSError):
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        import tf_keras
        m = tf_keras.models.load_model(str(vae_dir), compile=False)
    enc = m.encoder

    X = []
    kept_accs = []
    for acc, seq in aa_seqs.items():
        ohe = np.zeros((q, L), dtype=np.float32)
        ok = True
        for i, c in enumerate(seq):
            idx = seq_code.get(c)
            if idx is None:
                ok = False
                break
            if isinstance(idx, (range, list)):
                v = list(idx)
                for j in v:
                    ohe[j, i] = 1.0 / len(v)
            else:
                ohe[idx, i] = 1.0
        if ok:
            X.append(ohe.flatten())
            kept_accs.append(acc)

    X = np.asarray(X, dtype=np.float32)
    print(f"  encoding {len(X)} through AA-VAE...")
    mu = enc(tf.constant(X))[0].numpy()

    np.savez(cache_path,
             accessions=np.array(kept_accs),
             mu=mu.astype(np.float32))
    print(f"  saved {cache_path}")
    return dict(zip(kept_accs, mu))


def compute_per_seq(rows, H_aa, z0_aa, z1_aa, H_di, z0_di, z1_di,
                    F_aa, z0_F_aa, z1_F_aa, F_di, z0_F_di, z1_F_di):
    clss = [r[1] for r in rows]
    z_aa_arr = np.array([r[2] for r in rows], dtype=np.float64)
    z_di_arr = np.array([r[3] for r in rows], dtype=np.float64)
    H_aa_v = bilerp(H_aa, z0_aa, z1_aa, z_aa_arr)
    H_di_v = bilerp(H_di, z0_di, z1_di, z_di_arr)
    F_aa_v = bilerp(F_aa, z0_F_aa, z1_F_aa, z_aa_arr)
    F_di_v = bilerp(F_di, z0_F_di, z1_F_di, z_di_arr)
    dH = H_aa_v - H_di_v
    logF = (np.log10(np.maximum(F_aa_v, 1e-12))
            - np.log10(np.maximum(F_di_v, 1e-12)))
    return clss, dH, logF


def main():
    print("=== Globin ===")
    globin_zed = load_zed_csv(GLOBIN_ZED)
    print(f"  zed CSV: {len(globin_zed)} accessions")

    globin_aa = encode_family_aa(globin_zed, GLOBIN_AA_FASTA,
                                  GLOBIN_AA_VAE, GLOBIN_AA_CACHE, L=118)
    print(f"  AA latent (cached): {len(globin_aa)} accessions")

    globin_rows = []
    for acc, (cls, z_3di) in globin_zed.items():
        if acc in globin_aa:
            globin_rows.append((acc, cls,
                                np.asarray(globin_aa[acc]),
                                np.asarray(z_3di)))
    print(f"  final paired set: {len(globin_rows)} sequences")

    print("\n=== TRPM ===")
    trpm_zed = load_zed_csv(TRPM_ZED)
    print(f"  zed CSV: {len(trpm_zed)} accessions")
    trpm_aa = encode_family_aa(trpm_zed, TRPM_AA_FASTA,
                                TRPM_AA_VAE, TRPM_AA_CACHE, L=289)
    print(f"  AA latent (cached): {len(trpm_aa)} accessions")

    trpm_rows = []
    for acc, (cls, z_3di) in trpm_zed.items():
        if acc in trpm_aa:
            cls_r = TRPM_RENAME.get(cls, cls)
            trpm_rows.append((acc, cls_r,
                              np.asarray(trpm_aa[acc]),
                              np.asarray(z_3di)))
    print(f"  final paired set: {len(trpm_rows)} sequences")

    # H and F fields
    H_aa_g, z0_aa_g, z1_aa_g = load_grid_H(GLOBIN_AA_GRID)
    H_di_g, z0_di_g, z1_di_g = load_grid_H(GLOBIN_3DI_GRID)
    F_aa_g_d = np.load(GLOBIN_F_AA, allow_pickle=True)
    F_di_g_d = np.load(GLOBIN_F_3DI, allow_pickle=True)
    H_aa_t, z0_aa_t, z1_aa_t = load_grid_H(TRPM_AA_GRID)
    H_di_t, z0_di_t, z1_di_t = load_grid_H(TRPM_3DI_GRID)
    F_aa_t_d = np.load(TRPM_F_AA, allow_pickle=True)
    F_di_t_d = np.load(TRPM_F_3DI, allow_pickle=True)

    clss_g, dH_g, logF_g = compute_per_seq(
        globin_rows, H_aa_g, z0_aa_g, z1_aa_g, H_di_g, z0_di_g, z1_di_g,
        F_aa_g_d["vol"], F_aa_g_d["z0_axis"], F_aa_g_d["z1_axis"],
        F_di_g_d["vol"], F_di_g_d["z0_axis"], F_di_g_d["z1_axis"])

    clss_t, dH_t, logF_t = compute_per_seq(
        trpm_rows, H_aa_t, z0_aa_t, z1_aa_t, H_di_t, z0_di_t, z1_di_t,
        F_aa_t_d["vol"], F_aa_t_d["z0_axis"], F_aa_t_d["z1_axis"],
        F_di_t_d["vol"], F_di_t_d["z0_axis"], F_di_t_d["z1_axis"])

    # Shift ΔH by the per-family natural-cloud mean so the natural
    # cloud sits at the origin (cf. main-text Fig.~2D convention).
    # H is defined only up to an additive constant per Potts model;
    # subtracting <ΔH>_nat per family removes that arbitrary offset
    # without rescaling. log F ratio is dimensionless and not shifted.
    dH_g_offset = float(np.median(dH_g))
    dH_t_offset = float(np.median(dH_t))
    dH_g = dH_g - dH_g_offset
    dH_t = dH_t - dH_t_offset
    print(f"\nNatural-cloud ΔH offsets removed: "
          f"Globin {dH_g_offset:+.0f}, TRPM {dH_t_offset:+.0f}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2),
                              gridspec_kw={"wspace": 0.28})

    panels = [
        (axes[0], r"Globin (complementary regime)", clss_g, dH_g, logF_g,
         GLOBIN_CLASSES),
        (axes[1], r"TRPM (intermediate regime)", clss_t, dH_t, logF_t,
         TRPM_ORDER),
    ]

    for ax, title, clss, dH, logF, all_classes in panels:
        clss_arr = np.array(clss)
        cmap = plt.cm.tab10
        # Light cloud per class
        for i, c in enumerate(all_classes):
            mask = clss_arr == c
            if not mask.any():
                continue
            color = cmap(i / max(len(all_classes) - 1, 1))
            ax.scatter(dH[mask], logF[mask], s=10, alpha=0.18,
                       facecolor=color, edgecolor="none",
                       zorder=2)
        # Prominent per-class centroid with error bars (1 sigma)
        for i, c in enumerate(all_classes):
            mask = clss_arr == c
            n = int(mask.sum())
            if n < 2:
                continue
            color = cmap(i / max(len(all_classes) - 1, 1))
            dH_med = float(np.median(dH[mask]))
            logF_med = float(np.median(logF[mask]))
            dH_mad = float(np.median(np.abs(dH[mask] - dH_med)))
            logF_mad = float(np.median(np.abs(logF[mask] - logF_med)))
            ax.errorbar(dH_med, logF_med,
                        xerr=dH_mad, yerr=logF_mad,
                        fmt="o", markersize=12,
                        markerfacecolor=color, markeredgecolor="black",
                        markeredgewidth=1.2,
                        ecolor="black", elinewidth=0.9, capsize=3,
                        alpha=0.95, zorder=6,
                        label=f"{c} (n={n})")
        ax.axhline(0, color="0.55", lw=0.6, ls="--", zorder=1)
        ax.axvline(0, color="0.55", lw=0.6, ls="--", zorder=1)
        ax.set_xlabel(r"$\Delta H = H_\mathrm{AA} - H_\mathrm{3Di}$",
                      fontsize=11)
        ax.set_ylabel(r"$\log_{10}(F_\mathrm{AA} / F_\mathrm{3Di})$",
                      fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(-500, 500)
        ax.set_ylim(-2.5, 0.5)
        ax.legend(fontsize=7, loc="upper left", frameon=True,
                   markerscale=0.9)
        ax.grid(alpha=0.3, zorder=0)

    fig.suptitle("Subfamily placement on the AA-vs-3Di plane "
                 "(generalization of main-text Fig.~2D)",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()

    out_pdf = OUT_FIG / "04a_fig2d_globin_trpm.pdf"
    out_png = OUT_FIG / "04a_fig2d_globin_trpm.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_pdf}")
    print(f"saved {out_png}")

    # Per-class summary
    print("\n--- Globin per-class summary ---")
    for c in GLOBIN_CLASSES:
        mask = np.array(clss_g) == c
        if mask.any():
            print(f"  {c:>22}  n={mask.sum():3d}  "
                  f"dH med={np.median(dH_g[mask]):+.0f}  "
                  f"logF med={np.median(logF_g[mask]):+.2f}")
    print("--- TRPM per-class summary ---")
    for c in TRPM_ORDER:
        mask = np.array(clss_t) == c
        if mask.any():
            print(f"  {c:>10}  n={mask.sum():3d}  "
                  f"dH med={np.median(dH_t[mask]):+.0f}  "
                  f"logF med={np.median(logF_t[mask]):+.2f}")


if __name__ == "__main__":
    main()
