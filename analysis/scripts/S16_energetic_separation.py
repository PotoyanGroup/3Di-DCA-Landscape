"""Supplementary Figure S16: robust basins are separated by unambiguous Potts-energy barriers (distance-free).

For each protein family's 3Di-VAE Hamiltonian landscape, the script identifies the K deepest
robust basins (well-separated local minima of the smoothed Hamiltonian), then for every basin
pair computes the energy barrier as the minimax (lowest saddle) along the minimum-energy path
via Dijkstra on the latent grid. Barriers are reported in units of sigma_H, the standard
deviation of the Potts Hamiltonian in the low-energy region of the landscape. The figure is a
per-family strip/scatter plot of every basin-pair barrier (with the per-family median marked),
showing that every pair is separated by far more than sigma_H.

IMPORTANT: sigma_H is a model energy scale, NOT a physical temperature. The DCA Potts
Hamiltonian is in dimensionless coupling units (log-probabilities, P proportional to exp(-H)).
Do not call this k_B T_eff / thermal energy.

Inputs (paths resolved through analyses/scripts/_paths.py unless noted):
  - MDH_3DI_GRID_PKL : MDH 3Di-VAE latent-grid pickle (rows of z0, z1, H).
  - TRPM_3DI_GRID_PKL : TRPM8 3Di-VAE latent-grid pickle (rows of z0, z1, H).
  - GLOBIN_3DI_GRID_PKL : Globin 3Di-VAE latent-grid pickle (rows of z0, z1, H).
  - E1_DIR / "all_grid.pkl" : E1 glycoprotein 3Di-VAE latent-grid pickle (rows of z0, z1, H).
  - E2_DIR / "all_grid.pkl" : E2 glycoprotein 3Di-VAE latent-grid pickle (rows of z0, z1, H).

Outputs:
  - OUT_DATA / "barrier_separation.csv" : per basin-pair table (family, latent distance dist,
    barrier in units of sigma_H).
  - OUT_FIG / "S16_energetic_separation.pdf" : the figure (vector).
  - OUT_FIG / "S16_energetic_separation.png" : the figure (raster).

Usage:
  python analyses/scripts/S16_energetic_separation.py
  No command-line arguments.
"""
from pathlib import Path
import sys, heapq, pickle, csv, itertools
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, minimum_filter
from _paths import (OUT_FIG, OUT_DATA, MDH_3DI_GRID_PKL, GLOBIN_3DI_GRID_PKL,
                    TRPM_3DI_GRID_PKL, E1_DIR, E2_DIR)

K_TOP, MINSEP, SIGMA = 4, 6, 1.0
FAMILIES = [
    ("MDH", MDH_3DI_GRID_PKL), ("TRPM8", TRPM_3DI_GRID_PKL), ("Globin", GLOBIN_3DI_GRID_PKL),
    ("E1", E1_DIR / "all_grid.pkl"), ("E2", E2_DIR / "all_grid.pkl"),
]
FCOL = {"MDH": "#d62728", "TRPM8": "#9467bd", "Globin": "#2ca02c", "E1": "#1f77b4", "E2": "#ff7f0e"}


def load_grid(pkl):
    pts = pickle.load(open(pkl, "rb"))
    z0, z1 = np.unique(pts[:, 0]), np.unique(pts[:, 1])
    return pts[:, 2].reshape(z0.size, z1.size), z0, z1


def top_basins(Hs, k=K_TOP, minsep=MINSEP):
    locmin = Hs == minimum_filter(Hs, size=5)
    cand = sorted(zip(*np.where(locmin)), key=lambda ij: Hs[ij])
    picked = []
    for ij in cand:
        if all((ij[0] - p[0]) ** 2 + (ij[1] - p[1]) ** 2 > minsep ** 2 for p in picked):
            picked.append(ij)
        if len(picked) >= k:
            break
    return picked


def minimax(H, src, dst):
    """Dijkstra with edge weight max(H along path); returns the path's bottleneck H."""
    h, w = H.shape; dist = np.full((h, w), np.inf); dist[src] = H[src]
    pq = [(H[src], src[0], src[1])]
    nbr = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while pq:
        d, i, j = heapq.heappop(pq)
        if (i, j) == dst:
            return d
        if d > dist[i, j]:
            continue
        for di, dj in nbr:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w:
                nd = max(d, H[ni, nj])
                if nd < dist[ni, nj]:
                    dist[ni, nj] = nd; heapq.heappush(pq, (nd, ni, nj))
    return float(dist[dst])


rows = []
for name, pkl in FAMILIES:
    print(f"=== {name} ===")
    H, z0, z1 = load_grid(pkl)
    Hs = gaussian_filter(H, SIGMA)
    sigma_H = float(np.std(Hs[Hs < np.percentile(Hs, 25)]))
    basins = top_basins(Hs)
    for (ia, ja), (ib, jb) in itertools.combinations(basins, 2):
        bar = minimax(Hs, (ia, ja), (ib, jb)) - min(Hs[ia, ja], Hs[ib, jb])
        d = float(np.hypot(z0[ia] - z0[ib], z1[ja] - z1[jb]))
        rows.append(dict(family=name, dist=d, barrier_sigmaH=bar / sigma_H))
        print(f"  dist={d:.2f}  barrier={bar/sigma_H:.1f} sigma_H")

with open(OUT_DATA / "barrier_separation.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["family", "dist", "barrier_sigmaH"]); w.writeheader(); w.writerows(rows)

# ---- figure: barrier magnitudes per family (sigma_H) ----
FAMS = [f for f, _ in FAMILIES]
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(8.4, 4.9))
allb = []
for i, f in enumerate(FAMS):
    vals = np.array([r["barrier_sigmaH"] for r in rows if r["family"] == f]); allb += list(vals)
    ax.scatter(i + rng.uniform(-0.13, 0.13, len(vals)), vals, s=80, c=FCOL[f], edgecolor="k", lw=0.6, zorder=3)
    ax.hlines(np.median(vals), i - 0.25, i + 0.25, color="k", lw=2.5, zorder=4)
ax.axhline(1, color="0.45", lw=1.0)
ax.text(len(FAMS) - 0.55, 1.18, r"$\Delta H^\ddagger=\sigma_H$  (barrier = energy spread)", ha="right", fontsize=8, color="0.35")
ax.set_xticks(range(len(FAMS))); ax.set_xticklabels(FAMS, fontsize=11)
ax.set_ylabel(r"barrier between basins  $\Delta H^\ddagger / \sigma_H$", fontsize=11); ax.set_ylim(0, None)
allb = np.array(allb)
ax.set_title("Robust basins are separated by unambiguous Potts-energy barriers\n"
             rf"all {len(allb)} basin pairs: $\Delta H^\ddagger$ = {allb.min():.0f}-{allb.max():.0f}$\,\sigma_H$ "
             rf"(median {np.median(allb):.0f}), every pair $\gg\sigma_H$", fontsize=10.5)
ax.text(0.5, -0.155, r"$\sigma_H$ = std of the DCA Potts Hamiltonian in the low-energy region "
        r"(a model energy scale, not a physical temperature)", transform=ax.transAxes, ha="center", va="top",
        fontsize=7.5, color="0.4")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"S16_energetic_separation.{ext}", dpi=150, bbox_inches="tight")
print(f"\nbarriers: {allb.min():.1f}-{allb.max():.1f} sigma_H (median {np.median(allb):.1f}); n={len(allb)}")
print("wrote", OUT_FIG / "S16_energetic_separation.pdf")
