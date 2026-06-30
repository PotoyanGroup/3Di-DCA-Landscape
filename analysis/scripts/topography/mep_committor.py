"""Minimum-energy paths and committor probabilities between basins.

Reviewer concern #2: "I do not really see [the barriers], they look small."

We compute:
  1) Locations of the top-K robust basins (via watershed on smoothed H).
  2) For each pair, the minimum-energy path (MEP) using a Dijkstra-on-grid
     shortest-path where edge weight = max(H(u), H(v)) — this returns the
     minimax-energy path, whose maximum is the barrier height ΔH^‡.
  3) The committor q(z) between two chosen basins solved as the steady-state
     of an overdamped Langevin process on H(z): the boundary-value problem
     ∇·(e^{-βH} ∇q) = 0 with q=0 on basin A, q=1 on basin B.
  4) Reports ΔH^‡ in units of σ_H (std of H across the landscape) and as
     an effective k_B T_eff so the reviewer can judge separability.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, minimum_filter, label as cc_label
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import heapq

from _paths import OUT_FIG, OUT_DATA
from _helpers import load_landscape, smooth

H, z0, z1, pts = load_landscape()
print("Landscape:", H.shape)

# ============================================================
# 1) Identify robust basins  (smoothed H, large watershed regions)
# ============================================================
SIGMA = 4.0
H_s = smooth(H, SIGMA)

# Watershed: each pixel flows to its steepest-descent local minimum.
# We use cell-by-cell descent until reaching a fixed point.
def watershed_descent(H):
    h, w = H.shape
    parent = -np.ones((h, w, 2), dtype=np.int32)
    # 8-conn neighbours
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    next_step = np.empty((h, w, 2), dtype=np.int32)
    for i in range(h):
        for j in range(w):
            best = H[i, j]; best_ij = (i, j)
            for di, dj in nbrs:
                ni, nj = i+di, j+dj
                if 0 <= ni < h and 0 <= nj < w and H[ni, nj] < best:
                    best = H[ni, nj]; best_ij = (ni, nj)
            next_step[i, j] = best_ij
    # path compression
    label = -np.ones((h, w), dtype=np.int32)
    minima = []
    def root(i, j):
        # iterative pointer chasing
        cur = (i, j)
        path = []
        while True:
            ni, nj = next_step[cur]
            if (ni, nj) == cur:
                break
            path.append(cur)
            cur = (int(ni), int(nj))
        if label[cur] < 0:
            label[cur] = len(minima)
            minima.append(cur)
        for p in path:
            label[p] = label[cur]
        return label[cur]
    for i in range(h):
        for j in range(w):
            root(i, j)
    return label, minima

labels, minima = watershed_descent(H_s)
n_basins = len(minima)
print(f"Watershed found {n_basins} basins.")

# Rank by basin area × depth (volume-like)
areas = np.bincount(labels.ravel())
depths = np.array([H_s.max() - H_s[m] for m in minima])
volumes = areas * depths
order = np.argsort(volumes)[::-1]
# choose top-K spatially distinct basins
sep_pix = 60
chosen = []
for k in order:
    i0, j0 = minima[k]
    if all((i0 - ci)**2 + (j0 - cj)**2 > sep_pix**2 for ci, cj in chosen):
        chosen.append((int(i0), int(j0)))
    if len(chosen) == 4:
        break
print(f"Top {len(chosen)} watershed basins (i,j; z; H; area):")
for k, (i, j) in enumerate(chosen):
    a = areas[labels[i, j]]
    print(f"  basin {k}: ({i},{j})  z=({z0[i]:.2f}, {z1[j]:.2f})  "
          f"H={H[i,j]:.1f}  area={a}")

# ============================================================
# 2) Minimax-energy paths (Dijkstra with max-edge weighting)
# ============================================================
def minimax_path(H, src, dst):
    """Path minimizing the maximum H encountered along the way (8-conn)."""
    h, w = H.shape
    dist = np.full((h, w), np.inf)
    parent = np.full((h, w, 2), -1, dtype=np.int32)
    dist[src] = H[src]
    pq = [(H[src], src)]
    nbr = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while pq:
        d, (i, j) = heapq.heappop(pq)
        if (i, j) == dst:
            break
        if d > dist[i, j]:
            continue
        for di, dj in nbr:
            ni, nj = i+di, j+dj
            if not (0 <= ni < h and 0 <= nj < w):
                continue
            nd = max(d, H[ni, nj])
            if nd < dist[ni, nj]:
                dist[ni, nj] = nd
                parent[ni, nj] = (i, j)
                heapq.heappush(pq, (nd, (ni, nj)))
    # reconstruct
    path = []
    cur = dst
    while cur != (-1, -1) and cur != src:
        path.append(cur)
        ci, cj = parent[cur]
        if ci < 0:
            break
        cur = (int(ci), int(cj))
    path.append(src)
    path.reverse()
    return path, dist[dst]


# Compute MEPs for all unique pairs of basins
results = {}
for a in range(len(chosen)):
    for b in range(a+1, len(chosen)):
        src, dst = chosen[a], chosen[b]
        path, barrier = minimax_path(H, src, dst)
        Hs_along = np.array([H[i, j] for (i, j) in path])
        delta_h = Hs_along.max() - max(H[src], H[dst])
        results[(a, b)] = dict(path=path, barrier=barrier,
                               profile=Hs_along, delta_h=delta_h)
        print(f"  basin {a} → basin {b}: max-H on path = {barrier:.1f}  "
              f"ΔH‡ from deeper endpoint = {delta_h:.1f}")

# Global H std for unit conversion
H_std = float(np.std(H))
H_med_pers = 1500.0   # from script 01 (top H_0 persistence scale)
# Effective k_B T: use std of H in the well region (lowest 25%)
well_mask = H < np.percentile(H, 25)
k_B_T_eff = float(np.std(H[well_mask]))
print(f"H_std={H_std:.1f}  kB*T_eff={k_B_T_eff:.1f}")

# ============================================================
# 3) Committor between basin 0 and basin 1
# ============================================================
def solve_committor(H, srcA, srcB, beta=1.0/k_B_T_eff, radius=10):
    """Solve ∇·(D ∇q)=0 with D=exp(-βH), Dirichlet q=0 on A, q=1 on B."""
    h, w = H.shape
    Aregion = np.zeros((h, w), dtype=bool)
    Bregion = np.zeros((h, w), dtype=bool)
    yy, xx = np.indices((h, w))
    Aregion[(yy-srcA[0])**2 + (xx-srcA[1])**2 <= radius**2] = True
    Bregion[(yy-srcB[0])**2 + (xx-srcB[1])**2 <= radius**2] = True

    # Diffusion coefficient on faces (exp(-βH)). Use β chosen so β·H_std~1.
    D = np.exp(-beta * (H - H.min()))   # rescale to keep numerics finite

    # Build sparse linear system: for free pixels, (D_e*(q_e - q_c) summed) = 0
    fixed = Aregion | Bregion
    free_mask = ~fixed
    idx = -np.ones((h, w), dtype=np.int32)
    idx[free_mask] = np.arange(free_mask.sum())
    N = int(free_mask.sum())
    print(f"committor system size: {N} unknowns")

    rows, cols, vals = [], [], []
    rhs = np.zeros(N)
    for ci in range(h):
        for cj in range(w):
            if not free_mask[ci, cj]:
                continue
            row = idx[ci, cj]
            diag = 0.0
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = ci+di, cj+dj
                if not (0 <= ni < h and 0 <= nj < w):
                    continue
                # face conductance = harmonic mean of D values
                Dc = D[ci, cj]; Dn = D[ni, nj]
                Df = 2*Dc*Dn / (Dc + Dn + 1e-300)
                diag += Df
                if free_mask[ni, nj]:
                    rows.append(row); cols.append(idx[ni, nj]); vals.append(-Df)
                else:
                    # boundary contribution to RHS
                    rhs[row] += Df * (1.0 if Bregion[ni, nj] else 0.0)
            rows.append(row); cols.append(row); vals.append(diag)
    A_mat = csr_matrix((vals, (rows, cols)), shape=(N, N))
    q_free = spsolve(A_mat, rhs)
    q = np.zeros((h, w))
    q[Aregion] = 0.0
    q[Bregion] = 1.0
    q[free_mask] = q_free
    return q

print("\nSolving committor (basin 0 ↔ basin 1) on a coarsened grid...")
# coarsen for speed: 500 -> 200
factor = 3
H_c = H[::factor, ::factor]
# nearest indices for chosen basins on coarse grid
b0 = (chosen[0][0]//factor, chosen[0][1]//factor)
b1 = (chosen[1][0]//factor, chosen[1][1]//factor)

# pick beta so that beta*H_std ~ a few (we want the committor sharp at the barrier)
beta = 4.0 / H_std
q = solve_committor(H_c, b0, b1, beta=beta, radius=5)

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

extent = [z1.min(), z1.max(), z0.min(), z0.max()]
ax = axes[0, 0]
im = ax.imshow(H, origin="lower", extent=extent, cmap="viridis", aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.85, label="H")
for k, (i, j) in enumerate(chosen):
    ax.plot(z1[j], z0[i], "o", ms=14, mfc="white", mec="black",
            label=f"basin {k}")
    ax.annotate(str(k), (z1[j], z0[i]), color="black", weight="bold",
                ha="center", va="center", fontsize=10)
# overlay the MEPs
for (a, b), r in results.items():
    pp = np.array(r["path"])
    ax.plot(z1[pp[:, 1]], z0[pp[:, 0]], "-", lw=1.5,
            alpha=0.85, label=f"MEP {a}→{b}")
ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_0$")
ax.set_title("MDH AA landscape with MEPs between top basins")
ax.legend(fontsize=7, loc="lower left", framealpha=0.85)

# 1D barrier profiles
ax = axes[0, 1]
for (a, b), r in results.items():
    s = np.linspace(0, 1, len(r["profile"]))
    ax.plot(s, r["profile"], lw=1.8, label=f"{a}→{b}  Δ={r['delta_h']:.0f}")
ax.set_xlabel("normalized arc length")
ax.set_ylabel("$H$ along MEP")
ax.set_title("Minimax-energy profiles  (ΔH$^\\ddagger$ in legend)")
ax.legend(fontsize=8)

# committor map
ax = axes[1, 0]
extent_c = extent
im = ax.imshow(q, origin="lower", extent=extent_c, cmap="coolwarm",
               aspect="auto", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, shrink=0.85, label="$q(z)$")
# overlay basins (on coarse coords)
ax.plot(z1[chosen[0][1]], z0[chosen[0][0]], "o", ms=12,
        mfc="white", mec="black"); ax.text(z1[chosen[0][1]], z0[chosen[0][0]],
                                            "A", ha="center", va="center", weight="bold")
ax.plot(z1[chosen[1][1]], z0[chosen[1][0]], "o", ms=12,
        mfc="white", mec="black"); ax.text(z1[chosen[1][1]], z0[chosen[1][0]],
                                            "B", ha="center", va="center", weight="bold")
# iso-committor contour at 0.5 = transition state
cs = ax.contour(z1[::factor], z0[::factor], q,
                levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                colors="k", linewidths=0.8)
ax.clabel(cs, fontsize=7)
ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_0$")
ax.set_title(f"Committor $q(z)$ between basins 0 (A) and 1 (B)\n"
             f"$β H_\\sigma = {beta*H_std:.1f}$")

# summary table panel
ax = axes[1, 1]
ax.axis("off")
txt = [f"Landscape statistics", "—" * 24,
       f"H_min          {H.min():.1f}", f"H_max          {H.max():.1f}",
       f"H_std          {H_std:.1f}",
       f"k_B T_eff      {k_B_T_eff:.1f}",
       "", "Pairwise barrier heights ΔH‡ (from deeper endpoint)",
       "—" * 56]
for (a, b), r in results.items():
    txt.append(f"basin {a} → basin {b}:  ΔH‡ = {r['delta_h']:.0f}   "
               f"= {r['delta_h']/H_std:.1f}·σ_H   = {r['delta_h']/k_B_T_eff:.1f}·k_BT_eff")
txt += ["",
        "(For separability, ΔH‡ ≳ 3·k_BT_eff is the conventional",
        " threshold above which thermally-driven crossing is rare.)"]
ax.text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=8)

fig.tight_layout()
out = OUT_FIG / "03_mep_committor.pdf"
fig.savefig(out, bbox_inches="tight", dpi=150)
print(f"Saved {out}")

# Save numeric data
np.savez(OUT_DATA / "03_mep_committor.npz",
         chosen_basins=np.array(chosen),
         committor=q,
         beta=beta,
         H_std=H_std, k_B_T_eff=k_B_T_eff,
         paths={f"{a}_{b}": np.array(r["path"]) for (a, b), r in results.items()},
         profiles={f"{a}_{b}": r["profile"] for (a, b), r in results.items()},
         deltaH={f"{a}_{b}": r["delta_h"] for (a, b), r in results.items()})
print("Done.")
