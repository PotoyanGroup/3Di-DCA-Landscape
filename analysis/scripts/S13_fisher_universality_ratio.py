"""Supplementary Figure S13: spatial AA/3Di Fisher-metric ratio maps across five protein families.

Spatial counterpart of the per-family AA/3Di Fisher-ratio bar summary
(median/P90/total). For each of the five families (MDH, Globin, TRPM, E1,
E2) this script loads the precomputed pullback (Fisher) metric for the
amino-acid and 3Di VAE decoders and renders a 2 x 5 grid showing *where* in
latent space the AA-vs-3Di compression and metric collapse happen. Row 1
maps log10(F_AA(z) / F_3Di(z)) (the local metric volume ratio, a seismic
red/blue map of which decoder is locally sharper). Row 2 maps the log10 of
the per-point maximum intrinsic anisotropy ratio (lambda_max/lambda_min)
across the two representations (a magma map of where the local metric
collapses toward a near-1D corridor in either latent).

Inputs (paths resolved through analyses/scripts/_paths.py unless noted):
  - OUT_DATA / "08_pullback_metric_aa_mdh.npz",
    OUT_DATA / "08_pullback_metric_3di_mdh.npz",
    OUT_DATA / "08_pullback_metric_aa_globin.npz",
    OUT_DATA / "08_pullback_metric_3di_globin.npz",
    OUT_DATA / "08_pullback_metric_aa_trpm.npz",
    OUT_DATA / "08_pullback_metric_3di_trpm.npz",
    OUT_DATA / "08_pullback_metric_aa_e1.npz",
    OUT_DATA / "08_pullback_metric_3di_e1.npz",
    OUT_DATA / "08_pullback_metric_aa_e2.npz",
    OUT_DATA / "08_pullback_metric_3di_e2.npz" :
      cached pullback-metric grids (one AA + one 3Di file per family),
      each providing the arrays "vol" (local metric volume),
      "intrinsic_ratio" (local anisotropy ratio), and the latent grid
      axes "z0_axis" and "z1_axis".

Outputs:
  - OUT_FIG / "10b_fisher_universality_ratio_si.pdf" : the 2 x 5 figure (vector)
  - OUT_FIG / "10b_fisher_universality_ratio_si.png" : the same figure (raster, 160 dpi)

Usage:
  python analyses/scripts/S13_fisher_universality_ratio.py
  No command-line arguments.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from _paths import OUT_DATA, OUT_FIG

FAMILIES = [
    ("MDH",    "aa_mdh",    "3di_mdh"),
    ("Globin", "aa_globin", "3di_globin"),
    ("TRPM",   "aa_trpm",   "3di_trpm"),
    ("E1",     "aa_e1",     "3di_e1"),
    ("E2",     "aa_e2",     "3di_e2"),
]


def load(tag):
    return np.load(OUT_DATA / f"08_pullback_metric_{tag}.npz")


def main():
    maps = []
    for label, aa, td in FAMILIES:
        A = load(aa)
        D = load(td)
        maps.append((label, A, D))

    fig, axes = plt.subplots(
        2, len(FAMILIES),
        figsize=(3.2 * len(FAMILIES) + 1.4, 6.8),
        gridspec_kw={"hspace": 0.45, "wspace": 0.55},
    )

    # =================================================================
    # Row 1: log10(F_AA / F_3Di) ratio maps
    # =================================================================
    for col, (label, A, D) in enumerate(maps):
        ax = axes[0, col]
        aa_vol = A["vol"]; td_vol = D["vol"]
        z0 = A["z0_axis"]; z1 = A["z1_axis"]
        extent = [z1.min(), z1.max(), z0.min(), z0.max()]
        ratio = (aa_vol + 1e-6) / (td_vol + 1e-6)
        log_r = np.log10(ratio)
        abs_max = float(max(abs(np.percentile(log_r, 1)),
                            abs(np.percentile(log_r, 99))))
        im = ax.imshow(log_r, origin="lower", extent=extent, cmap="seismic",
                       aspect="auto",
                       norm=mcolors.TwoSlopeNorm(vmin=-abs_max,
                                                  vcenter=0.0,
                                                  vmax=abs_max))
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel(r"$z_1$", fontsize=10)
        if col == 0:
            ax.set_ylabel(r"$z_0$", fontsize=10)
        cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
        cb.ax.tick_params(labelsize=9)
        cb.set_label(r"$\log_{10}\, F_\mathrm{AA}/F_\mathrm{3Di}$",
                     fontsize=10, rotation=90, labelpad=8)

    # =================================================================
    # Row 2: where the local metric is most anisotropic (max of AA,3Di)
    # =================================================================
    for col, (label, A, D) in enumerate(maps):
        ax = axes[1, col]
        ar_aa = A["intrinsic_ratio"]
        ar_td = D["intrinsic_ratio"]
        z0 = A["z0_axis"]; z1 = A["z1_axis"]
        extent = [z1.min(), z1.max(), z0.min(), z0.max()]
        # Per-point pick: the latent that is more anisotropic at z.
        max_anis = np.maximum(ar_aa, ar_td)
        log_a = np.log10(np.maximum(max_anis, 1.0))
        vmax = float(np.percentile(log_a, 99))
        im = ax.imshow(log_a, origin="lower", extent=extent, cmap="magma",
                       aspect="auto", vmin=0.0, vmax=vmax)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel(r"$z_1$", fontsize=10)
        if col == 0:
            ax.set_ylabel(r"$z_0$", fontsize=10)
        cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
        cb.ax.tick_params(labelsize=9)
        cb.set_label(r"$\log_{10}\,\lambda_{\max}/\lambda_{\min}$",
                     fontsize=10, rotation=90, labelpad=8)

    # Row labels on the far left, outside the panels (bumped to clear
    # the y-axis ticks).
    fig.text(-0.005, 0.74, "AA-vs-3Di compression",
             rotation=90, fontsize=13, fontweight="bold",
             va="center", ha="center")
    fig.text(-0.005, 0.28, "metric anisotropy",
             rotation=90, fontsize=13, fontweight="bold",
             va="center", ha="center")

    fig.suptitle("Fisher metric across protein families: "
                 "where compression happens and where the metric collapses",
                 fontsize=13, fontweight="bold", y=1.005)
    fig.tight_layout(rect=[0.04, 0.0, 1, 0.99])

    out_pdf = OUT_FIG / "10b_fisher_universality_ratio_si.pdf"
    out_png = OUT_FIG / "10b_fisher_universality_ratio_si.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_pdf}")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
