"""Supplementary Figure S9: persistent homology of the MDH amino-acid Hamiltonian landscape.

Tests whether the basins ("wells") of the MDH AA latent Hamiltonian landscape are
robust topological features or finite-sampling artifacts. The script builds a cubical
complex from the 2D Hamiltonian grid H(z), computes its sublevel-set persistence with
GUDHI, and separates the diagram into H_0 (connected components / basins) and H_1
(loops). The resulting figure shows the persistence diagram of H(z) and a barcode of the
top-40 H_0 features sorted by persistence: long bars (far from the diagonal) are robust
basins, while near-diagonal points are noise that vanishes under small threshold/smoothing
changes. A summary reports the top-K most persistent basins and the top-1/median
persistence ratio.

Inputs (paths resolved through analyses/scripts/_paths.py unless noted):
  - MDH_GRID_PKL (mdh_aa/all_grid.pkl) : latent-grid points (z0, z1, H), loaded via _helpers.load_landscape
  - MDH_HAMIL (mdh_aa/hamil_mat.npy)   : Hamiltonian values on the grid, reshaped to H[i,j], loaded via _helpers.load_landscape

Outputs:
  - OUT_FIG / "01_persistent_homology.pdf"     : persistence diagram + top-40 H_0 barcode of H(z)
  - OUT_DATA / "01_persistence.npz"            : arrays dim0 (finite H_0 births/deaths), dim1 (H_1), dim0_inf (essential H_0 births)
  - OUT_DATA / "01_persistence_summary.txt"    : text summary of feature counts, top-10 H_0 persistences, top-6 threshold, top-1/median ratio

Usage:
  python analyses/scripts/topography/S9_persistent_homology.py
  No command-line arguments.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import gudhi

from _paths import OUT_FIG, OUT_DATA
from _helpers import load_landscape

H, z0, z1, pts = load_landscape()
print(f"Landscape: {H.shape}, H in [{H.min():.1f}, {H.max():.1f}]")

# Cubical complex on the negated Hamiltonian — sublevel filtration of H
# corresponds to basins forming as energy threshold rises from the minimum.
cc = gudhi.CubicalComplex(top_dimensional_cells=H)
cc.compute_persistence()
diag = cc.persistence()
# diag is list of (dim, (birth, death))

# Separate by dimension
dim0 = [(b, d) for (k, (b, d)) in diag if k == 0 and not np.isinf(d)]
dim0_inf = [(b, d) for (k, (b, d)) in diag if k == 0 and np.isinf(d)]
dim1 = [(b, d) for (k, (b, d)) in diag if k == 1]

dim0 = np.array(dim0) if dim0 else np.zeros((0, 2))
dim1 = np.array(dim1) if dim1 else np.zeros((0, 2))

print(f"H_0 finite features: {len(dim0)} (essential class(es): {len(dim0_inf)})")
print(f"H_1 features:        {len(dim1)}")

if len(dim0):
    persistence0 = dim0[:, 1] - dim0[:, 0]
    print(f"H_0 persistence summary: max={persistence0.max():.1f} "
          f"median={np.median(persistence0):.2f} "
          f"top-10 = {np.sort(persistence0)[-10:][::-1]}")

# --- persistence diagram + barcode ---
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
lo, hi = H.min(), H.max()
ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, lw=1)
if len(dim0):
    ax.scatter(dim0[:, 0], dim0[:, 1], s=10, c="C0", label=r"$H_0$ (basins)")
if len(dim1):
    ax.scatter(dim1[:, 0], dim1[:, 1], s=20, c="C3", marker="^", label=r"$H_1$ (loops)")
# essential class: ray
for b, d in dim0_inf:
    ax.axvline(b, color="C0", alpha=0.4, lw=0.5)
ax.set_xlabel("Birth (Hamiltonian)")
ax.set_ylabel("Death (Hamiltonian)")
ax.set_title("Persistence diagram of $H(z)$ — MDH AA landscape")
ax.legend(frameon=False)

# Barcode of H_0, sorted by persistence (most persistent on top)
ax = axes[1]
if len(dim0):
    pers = dim0[:, 1] - dim0[:, 0]
    order = np.argsort(pers)[::-1]
    # only plot top 40 + essential
    top = order[:40]
    for k, idx in enumerate(top):
        ax.plot([dim0[idx, 0], dim0[idx, 1]], [k, k], "C0-", lw=1.5)
    # add essential ray at top
    for b, d in dim0_inf:
        ax.plot([b, H.max()], [-1, -1], "k-", lw=2)
        ax.plot([H.max()], [-1], ">k", ms=6)
ax.set_xlabel("Hamiltonian")
ax.set_ylabel("$H_0$ bars (sorted by persistence)")
ax.set_title("Top-40 $H_0$ barcodes  (long = robust basin)")
ax.invert_yaxis()

fig.tight_layout()
out = OUT_FIG / "01_persistent_homology.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")

# --- save numeric data ---
np.savez(OUT_DATA / "01_persistence.npz",
         dim0=dim0,
         dim1=dim1,
         dim0_inf=np.array([[b, np.nan] for b, _ in dim0_inf]))
print(f"Saved {OUT_DATA/'01_persistence.npz'}")

# --- threshold by persistence and overlay basins on landscape ---
# Use the top-K most persistent H_0 generators as "robust basins"
K_TOP = 6
if len(dim0):
    pers = dim0[:, 1] - dim0[:, 0]
    p_thresh = np.sort(pers)[-K_TOP] if len(pers) >= K_TOP else pers.min()
    print(f"Robust-basin persistence threshold (top {K_TOP}): {p_thresh:.2f}")
    # store as text
    with open(OUT_DATA / "01_persistence_summary.txt", "w") as fh:
        fh.write(f"H0 finite features: {len(dim0)}\n")
        fh.write(f"H0 essential classes: {len(dim0_inf)}\n")
        fh.write(f"H1 features: {len(dim1)}\n")
        fh.write(f"H0 persistence top-10: "
                 f"{np.sort(pers)[-10:][::-1].tolist()}\n")
        fh.write(f"persistence threshold for top-{K_TOP}: {p_thresh:.2f}\n")
        # report ratio of top-1 to top-K and to median:
        fh.write(f"top1/median ratio: {pers.max()/max(np.median(pers),1e-12):.1f}\n")
print("Done.")
