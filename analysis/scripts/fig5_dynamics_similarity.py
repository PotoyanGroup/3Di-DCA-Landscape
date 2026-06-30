"""Main-text Figure 5: dynamical similarity of flavohemoglobins tracks 3Di-DCA landscape clustering.

Builds a three-panel figure linking the globin 3Di-DCA latent landscape to all-atom MD
dynamics for eight representative species drawn from four landscape basins (clusters). The
script reads the per-representative latent coordinates and RMSF profiles, computes all
pairwise RMSF-profile Pearson correlations and Euclidean latent distances, and renders:
panel A, a zoomed globin 3Di-DCA energy landscape with the eight MD representatives colored by
basin and basin outlines from a steepest-descent watershed of the smoothed landscape; panel B,
within- vs between-cluster latent distance (with means and the between/within separation
ratio); and panel C, RMSF-profile correlation vs latent distance, showing that dynamical
similarity decays with latent distance (annotated with the overall Pearson r and p-value).

Inputs (paths resolved through analyses/scripts/_paths.py unless noted):
  - GLOBIN_3DI_DIR / "coord_rmsf.txt" : Python-literal dict {species: [[z0, z1], rmsf_profile]}
      giving the latent coordinates and per-residue RMSF profile for the 8 representatives
      (parsed via eval with a restricted namespace).
  - GLOBIN_3DI_GRID_PKL : pickled (N, 3) array of latent-grid points (z0, z1, H_3Di) used to
      reconstruct the energy-landscape grid.

Outputs:
  - OUT_DATA / "globin_rmsf_correlations.csv" : per-pair table (sp_i, sp_j, within flag,
      pearson_r of RMSF profiles, d_latent Euclidean latent distance).
  - OUT_FIG / "Fig5_dynamics.pdf" : the three-panel figure (vector).
  - OUT_FIG / "Fig5_dynamics.png" : the three-panel figure (raster, dpi=185).

Usage:
  python analyses/scripts/fig5_dynamics_similarity.py
  No command-line arguments.
"""
from pathlib import Path
import sys, itertools, pickle
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter
from _paths import GLOBIN_3DI_DIR, GLOBIN_3DI_GRID_PKL, OUT_FIG, OUT_DATA

CLUSTER = {
    "Podospora anserina": "olive", "Arcticibacter pallidicorallinus": "olive",
    "Planctopirus limnophila": "orange", "Kroppenstedtia pulmonis": "orange",
    "Microbacterium kyungheense": "red", "Nonomuraea phyllanthi": "red",
    "Virgibacillus salinus": "magenta", "Streptomyces nitrosporeus": "magenta",
}
clu_order = ["olive", "orange", "red", "magenta"]

# --- representatives: latent coords + RMSF, from coord_rmsf.txt ---
raw = eval((GLOBIN_3DI_DIR / "coord_rmsf.txt").read_text(),
           {"array": np.array, "float32": np.float32, "__builtins__": {}})
coords, rmsf = {}, {}
for name, val in raw.items():
    name = name.strip()
    (z0, z1), prof = val if len(val) == 2 else ((val[0], val[1]), val[2])
    coords[name] = np.array([float(z0), float(z1)]); rmsf[name] = np.asarray(prof, float).squeeze()
order = [s for s in CLUSTER if s in rmsf]
L = min(len(rmsf[s]) for s in order)

# --- pairwise RMSF correlation + Euclidean latent distance ---
rows = []
for a, b in itertools.combinations(order, 2):
    rows.append(dict(sp_i=a, sp_j=b, within=CLUSTER[a] == CLUSTER[b],
                     pearson_r=np.corrcoef(rmsf[a][:L], rmsf[b][:L])[0, 1],
                     d_latent=float(np.hypot(*(coords[a] - coords[b])))))
df = pd.DataFrame(rows)
wm, bm = df.loc[df.within, "pearson_r"].mean(), df.loc[~df.within, "pearson_r"].mean()
df.to_csv(OUT_DATA / "globin_rmsf_correlations.csv", index=False)

# --- landscape grid ---
g = pickle.load(open(GLOBIN_3DI_GRID_PKL, "rb"))
z0u, z1u = np.unique(g[:, 0]), np.unique(g[:, 1])
H = np.full((len(z1u), len(z0u)), np.nan)
H[np.searchsorted(z1u, g[:, 1]), np.searchsorted(z0u, g[:, 0])] = g[:, 2]


def basin_labels(F):
    """Steepest-descent watershed (numpy only): each cell -> index of the local min it drains to."""
    ny, nx = F.shape
    idx = np.arange(ny * nx).reshape(ny, nx); nxt = idx.copy(); best = F.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            sh = np.full_like(F, np.inf); nb = np.full_like(idx, -1)
            ys = slice(max(0, di), ny + min(0, di)); xs = slice(max(0, dj), nx + min(0, dj))
            yt = slice(max(0, -di), ny + min(0, -di)); xt = slice(max(0, -dj), nx + min(0, -dj))
            sh[yt, xt] = F[ys, xs]; nb[yt, xt] = idx[ys, xs]
            better = sh < best; best[better] = sh[better]; nxt[better] = nb[better]
    root = nxt.ravel()
    for _ in range(60):
        nr = root[root]
        if np.array_equal(nr, root):
            break
        root = nr
    return root.reshape(ny, nx)


# zoom window around the representatives
pz0 = np.array([coords[s][0] for s in order]); pz1 = np.array([coords[s][1] for s in order])
pad = 1.1
x0, x1, y0, y1 = pz0.min() - pad, pz0.max() + pad, pz1.min() - pad, pz1.max() + pad
xm, ym = (z0u >= x0) & (z0u <= x1), (z1u >= y0) & (z1u <= y1)
Hc = H[np.ix_(ym, xm)]
ext = [z0u[xm].min(), z0u[xm].max(), z1u[ym].min(), z1u[ym].max()]

fig = plt.figure(figsize=(14.0, 4.6), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.72, 1.0])
axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))

# A: zoomed landscape + basin outlines + cluster points
im = axA.imshow(Hc, origin="lower", extent=ext, cmap="viridis", aspect="auto")
Hsm = gaussian_filter(Hc, sigma=2.5)
axA.contour(Hsm, levels=10, extent=ext, origin="lower", colors="white", linewidths=0.4, alpha=0.4, zorder=2)
labels = basin_labels(Hsm)
gz0, gz1 = z0u[xm], z1u[ym]
def cell(z0, z1):
    return (int(np.clip(np.searchsorted(gz1, z1), 0, len(gz1) - 1)),
            int(np.clip(np.searchsorted(gz0, z0), 0, len(gz0) - 1)))
pt_lab = {s: labels[cell(*coords[s])] for s in order}
for lab in set(pt_lab.values()):
    here = [CLUSTER[s] for s in order if pt_lab[s] == lab]
    axA.contour((labels == lab).astype(float), levels=[0.5], extent=ext, origin="lower",
                colors=[max(set(here), key=here.count)], linewidths=2.4, zorder=3)
for s in order:
    axA.scatter(*coords[s], c=CLUSTER[s], s=150, edgecolor="white", linewidth=1.8, zorder=5)
axA.set_xlim(*ext[:2]); axA.set_ylim(*ext[2:]); axA.set_xlabel("Latent $Z_0$"); axA.set_ylabel("Latent $Z_1$")
axA.set_title("Globin 3Di-DCA landscape (zoom)", fontsize=11)
fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04).set_label("$H_{3Di}$", fontsize=9)
axA.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="k",
                           markersize=8, label=f"cluster {k+1}") for k, c in enumerate(clu_order)],
           loc="upper left", fontsize=8, framealpha=0.85)
axA.text(-0.16, 1.02, "A", transform=axA.transAxes, fontsize=15, fontweight="bold")

# B: within vs between Euclidean latent distance
rng = np.random.default_rng(1)
wd, bd = df.loc[df.within], df.loc[~df.within]
axB.scatter(rng.uniform(-0.12, 0.12, len(wd)), wd.d_latent,
            c=[CLUSTER[r.sp_i] for r in wd.itertuples()], s=85, edgecolor="k", lw=0.6, zorder=3)
axB.scatter(1 + rng.uniform(-0.13, 0.13, len(bd)), bd.d_latent, c="0.62", s=42, edgecolor="k", lw=0.4, zorder=3)
for x, m in [(0, wd.d_latent.mean()), (1, bd.d_latent.mean())]:
    axB.hlines(m, x - 0.3, x + 0.3, color="k", lw=3, zorder=4)
    axB.text(x + 0.34, m, f"{m:.2f}", va="center", fontsize=12, fontweight="bold")
axB.set_xticks([0, 1]); axB.set_xticklabels(["within\ncluster", "between\ncluster"], fontsize=11)
axB.set_ylabel("latent distance  $\\|z_i-z_j\\|$", fontsize=11)
axB.set_xlim(-0.55, 1.7); axB.set_ylim(0, None)
axB.set_title(f"cluster separation  ($\\approx${bd.d_latent.mean()/wd.d_latent.mean():.1f}$\\times$)", fontsize=10)
for sp in ("top", "right"): axB.spines[sp].set_visible(False)
axB.text(-0.3, 1.02, "B", transform=axB.transAxes, fontsize=15, fontweight="bold")

# C: RMSF correlation vs latent distance
rr, pp = pearsonr(df.d_latent, df.pearson_r)
axC.scatter(bd.d_latent, bd.pearson_r, c="0.62", s=46, edgecolor="k", lw=0.4, zorder=3, label="between cluster")
axC.scatter(wd.d_latent, wd.pearson_r, c=[CLUSTER[r.sp_i] for r in wd.itertuples()],
            s=95, edgecolor="k", lw=0.7, zorder=4, label="within cluster")
xs = np.linspace(df.d_latent.min(), df.d_latent.max(), 50); m, b0 = np.polyfit(df.d_latent, df.pearson_r, 1)
axC.plot(xs, m * xs + b0, "k--", lw=1.6, zorder=2)
axC.set_xlabel("latent distance  $\\|z_i - z_j\\|$", fontsize=11)
axC.set_ylabel("RMSF profile correlation  $r$", fontsize=11); axC.set_ylim(-0.05, 1.0)
axC.set_title(f"dynamic similarity with distance\nPearson $r$={rr:.2f}, $p$={pp:.3f}", fontsize=10)
axC.legend(fontsize=8, loc="upper right", framealpha=0.85)
for sp in ("top", "right"): axC.spines[sp].set_visible(False)
axC.text(-0.24, 1.02, "C", transform=axC.transAxes, fontsize=15, fontweight="bold")

for ext_ in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"Fig5_dynamics.{ext_}", dpi=185, bbox_inches="tight")
print(f"within r={wm:.3f}  between r={bm:.3f}  decay r={rr:.3f} (p={pp:.3f})")
print("wrote", OUT_FIG / "Fig5_dynamics.pdf")
