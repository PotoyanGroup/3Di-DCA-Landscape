"""Cross-family Langevin barrier comparison (MDH and Globin, AA and 3Di).

For each of the four (family, representation) combinations, runs both
uniform-D and Riemannian (Fisher-metric) Langevin between the family's
cohort centroids and collects per-trajectory barrier heights. Produces:

  1. A single-panel strip plot of all 8 barrier distributions side by
     side (4 family/rep groups, 2 methods per group). No MEP overlay.
  2. A CSV of all per-trajectory barriers.

Cohort centroids:
  - MDH AA / 3Di:    psychrophilic vs thermophilic/mesophilic
                     (from MDH_cluster_data/{psy,tm}/cluster_coordinates_*)
  - Globin AA:       cytoglobin vs flavohemoglobin
                     (from outputs/data/00d_globin_aa_subfamily_mu.npz)
  - Globin 3Di:      cytoglobin vs flavohemoglobin
                     (from zed_coordinates_with_classes.csv)
"""
from pathlib import Path
import pickle
import sys
import heapq
import ast
import csv
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "main_script",
    Path(__file__).resolve().parent / "watershed_disconnectivity.py",
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

from _paths import (OUT_FIG, OUT_DATA, MDH_CLUSTER_DATA, GLOBIN_AA_DIR,  # noqa: E402
                    GLOBIN_3DI_DIR, MDH_GRID_PKL, MDH_HAMIL, MDH_3DI_GRID_PKL)


MDH_CLUSTER_ROOT = MDH_CLUSTER_DATA
GLOBIN_DIR = GLOBIN_3DI_DIR
GLOBIN_AA_GRID_PKL = GLOBIN_AA_DIR / "all_grid.pkl"
GLOBIN_3DI_GRID_PKL = GLOBIN_DIR / "all_grid.pkl"
CACHE_MU = OUT_DATA / "cache_paired_mus.npz"
GLOBIN_AA_NPZ = OUT_DATA / "00d_globin_aa_subfamily_mu.npz"
ZED_CSV = GLOBIN_DIR / "zed_coordinates_with_classes.csv"

T_DIM = 1.0
ALPHA = 0.01
MAX_STEPS = 120_000
MAX_STEPS_RIEMANNIAN = 480_000
N_TRAJ = 400
CAPTURE_RADIUS = 0.4
EXIT_RADIUS = 0.5
APPROACH_SNAPSHOTS = 60
SEED = 0


# ====================== helpers ======================
def load_grid(pkl_path, hamil_npy=None):
    pts = pickle.load(open(pkl_path, "rb"))
    z0 = np.unique(pts[:, 0]); z1 = np.unique(pts[:, 1])
    if hamil_npy is not None and Path(hamil_npy).exists():
        H = np.load(hamil_npy)
        if H.shape != (z0.size, z1.size):
            H = pts[:, 2].reshape(z0.size, z1.size)
    else:
        H = pts[:, 2].reshape(z0.size, z1.size)
    return H, z0, z1


def load_coord_dict(path):
    return ast.literal_eval(Path(path).read_text())


def natural_H_spread(H, z0, z1, mu_train):
    Hs = [H[int(np.argmin(np.abs(z0 - z0v))),
            int(np.argmin(np.abs(z1 - z1v)))]
          for z0v, z1v in mu_train]
    return float(np.std(Hs))


def load_and_resample_fisher(fisher_npz, z0_fine, z1_fine):
    d = np.load(fisher_npz)
    g_coarse = d["g"].astype(np.float64)
    z0_c = d["z0_axis"]; z1_c = d["z1_axis"]
    g_fine = np.zeros((len(z0_fine), len(z1_fine), 2, 2))
    Z0, Z1 = np.meshgrid(z0_fine, z1_fine, indexing="ij")
    pts = np.column_stack([Z0.ravel(), Z1.ravel()])
    for i in range(2):
        for j in range(2):
            interp = RegularGridInterpolator(
                (z0_c, z1_c), g_coarse[:, :, i, j],
                bounds_error=False, fill_value=None, method="linear")
            g_fine[:, :, i, j] = interp(pts).reshape(len(z0_fine),
                                                     len(z1_fine))
    g_fine = 0.5 * (g_fine + np.swapaxes(g_fine, -1, -2))
    a = g_fine[..., 0, 0]; b = g_fine[..., 0, 1]; c = g_fine[..., 1, 1]
    det = np.maximum(a * c - b * b, 1e-3)
    a = np.maximum(a, 1e-3); c = np.maximum(c, 1e-3)
    g_inv = np.zeros_like(g_fine)
    g_inv[..., 0, 0] = c / det
    g_inv[..., 0, 1] = -b / det
    g_inv[..., 1, 0] = -b / det
    g_inv[..., 1, 1] = a / det
    L = np.zeros_like(g_fine)
    L[..., 0, 0] = np.sqrt(c / det)
    L[..., 1, 0] = -b / np.sqrt(c * det)
    L[..., 1, 1] = 1.0 / np.sqrt(c)
    det_g_inv = np.linalg.det(g_inv)
    scale = float(np.exp(np.mean(np.log(np.maximum(det_g_inv, 1e-12)))))
    s = scale ** 0.5
    g_inv /= scale
    L /= s
    return g_inv, L


def run_langevin(Hs, z0, z1, sigma_nat, src_z, dst_z,
                 n_traj, max_steps, alpha, T_dim, capture_radius,
                 rng, g_inv=None, L=None):
    H_norm = Hs / sigma_nat
    g0, g1 = np.gradient(H_norm, z0, z1)
    z0_min, z0_max = float(z0.min()), float(z0.max())
    z1_min, z1_max = float(z1.min()), float(z1.max())
    dz0 = (z0_max - z0_min) / (len(z0) - 1)
    dz1 = (z1_max - z1_min) / (len(z1) - 1)

    def idx0(x):
        return np.clip(((x - z0_min) / dz0).astype(int), 0, len(z0) - 1)

    def idx1(x):
        return np.clip(((x - z1_min) / dz1).astype(int), 0, len(z1) - 1)

    pos0 = rng.normal(src_z[0], 0.10, size=n_traj)
    pos1 = rng.normal(src_z[1], 0.10, size=n_traj)
    alive = np.ones(n_traj, dtype=bool)
    in_basin_a = np.ones(n_traj, dtype=bool)
    seg_max_h = np.full(n_traj, -np.inf)
    crossed = np.zeros(n_traj, dtype=bool)
    saved_max_h = np.full(n_traj, np.nan)
    floor_A = float(Hs[int(np.argmin(np.abs(z0 - src_z[0]))),
                       int(np.argmin(np.abs(z1 - src_z[1])))] / sigma_nat)

    sigma_step = np.sqrt(2 * alpha * T_dim)
    for step in range(max_steps):
        if not alive.any():
            break
        idx = np.where(alive)[0]
        ia = idx0(pos0[idx]); ja = idx1(pos1[idx])
        gH0 = g0[ia, ja]; gH1 = g1[ia, ja]
        if g_inv is None:
            drift0 = -alpha * gH0
            drift1 = -alpha * gH1
            kick0 = sigma_step * rng.standard_normal(idx.size)
            kick1 = sigma_step * rng.standard_normal(idx.size)
        else:
            gi = g_inv[ia, ja]
            drift0 = -alpha * (gi[:, 0, 0] * gH0 + gi[:, 0, 1] * gH1)
            drift1 = -alpha * (gi[:, 1, 0] * gH0 + gi[:, 1, 1] * gH1)
            Lm = L[ia, ja]
            xi0 = rng.standard_normal(idx.size)
            xi1 = rng.standard_normal(idx.size)
            kick0 = sigma_step * (Lm[:, 0, 0] * xi0 + Lm[:, 0, 1] * xi1)
            kick1 = sigma_step * (Lm[:, 1, 0] * xi0 + Lm[:, 1, 1] * xi1)
        pos0[idx] = np.clip(pos0[idx] + drift0 + kick0, z0_min, z0_max)
        pos1[idx] = np.clip(pos1[idx] + drift1 + kick1, z1_min, z1_max)

        ia2 = idx0(pos0[idx]); ja2 = idx1(pos1[idx])
        h_now = H_norm[ia2, ja2]
        d_a0 = pos0 - src_z[0]; d_a1 = pos1 - src_z[1]
        d_b0 = pos0 - dst_z[0]; d_b1 = pos1 - dst_z[1]
        near_a = (d_a0**2 + d_a1**2 < EXIT_RADIUS**2)
        near_b = (d_b0**2 + d_b1**2 < capture_radius**2)
        re_entered = near_a & (~in_basin_a) & alive
        for k in np.where(re_entered)[0]:
            seg_max_h[k] = -np.inf
        in_basin_a |= near_a
        in_basin_a &= near_a
        recording = (~in_basin_a) & alive
        for kk, k in enumerate(idx):
            if recording[k] and h_now[kk] > seg_max_h[k]:
                seg_max_h[k] = float(h_now[kk])
        reached = near_b & alive
        for k in np.where(reached)[0]:
            if not np.isfinite(seg_max_h[k]):
                ii = int(np.where(idx == k)[0][0]) if k in idx else 0
                seg_max_h[k] = float(h_now[ii]) if k in idx else 0.0
            saved_max_h[k] = seg_max_h[k]
        crossed |= reached
        alive &= ~reached

    barriers = []
    for k in range(n_traj):
        if not crossed[k]:
            continue
        b = saved_max_h[k] - floor_A
        if np.isfinite(b):
            barriers.append(b)
    return np.asarray(barriers), int(crossed.sum())


# ====================== cohort centroids ======================
def mdh_centroids(alphabet):
    psy_path = MDH_CLUSTER_ROOT / "psycrophilic_cluster" / (
        "cluster_coordinates_AA.txt" if alphabet == "AA"
        else "cluster_coordinates_3DI.txt")
    tm_path = MDH_CLUSTER_ROOT / "thermo_meso_cluster" / (
        "cluster_coordinates_AA.txt" if alphabet == "AA"
        else "cluster_coordinates_3DI.txt")
    psy = np.array(list(load_coord_dict(psy_path).values()))
    tm = np.array(list(load_coord_dict(tm_path).values()))
    return psy.mean(axis=0), tm.mean(axis=0)


def globin_3di_centroids():
    cyto_xy = []; flavo_xy = []
    with open(ZED_CSV) as fh:
        for r in csv.DictReader(fh):
            cls = r["class"].strip()
            try:
                z = (float(r["z1"]), float(r["z2"]))
            except (ValueError, KeyError):
                continue
            if cls == "Cytoglobin":
                cyto_xy.append(z)
            elif cls == "Flavohemoglobin":
                flavo_xy.append(z)
    return np.array(cyto_xy).mean(axis=0), np.array(flavo_xy).mean(axis=0)


def globin_aa_centroids():
    d = np.load(GLOBIN_AA_NPZ)
    return d["Cytoglobin_centroid"], d["Flavohemoglobin_centroid"]


# ====================== conditions ======================
def main():
    rng = np.random.default_rng(SEED)
    cache = np.load(CACHE_MU, allow_pickle=True)

    conditions = [
        ("MDH",    "AA",  MDH_GRID_PKL, MDH_HAMIL,
         OUT_DATA / "08_pullback_metric_aa_mdh.npz",
         "MDH_mu_aa",  mdh_centroids("AA")),
        ("MDH",    "3Di", MDH_3DI_GRID_PKL, None,
         OUT_DATA / "08_pullback_metric_3di_mdh.npz",
         "MDH_mu_3di", mdh_centroids("3Di")),
        ("Globin", "AA",  GLOBIN_AA_GRID_PKL, None,
         OUT_DATA / "08_pullback_metric_aa_globin.npz",
         "Globin_mu_aa",  globin_aa_centroids()),
        ("Globin", "3Di", GLOBIN_3DI_GRID_PKL, None,
         OUT_DATA / "08_pullback_metric_3di_globin.npz",
         "Globin_mu_3di", globin_3di_centroids()),
    ]

    results = []
    for fam, rep, pkl, hamil, fisher_npz, mu_key, (cen_a, cen_b) in conditions:
        tag = f"{fam}-{rep}"
        print(f"\n=========== {tag} ===========")
        H, z0, z1 = load_grid(pkl, hamil)
        R = m.compute_landscape(H, z0, z1, 8, sigma=1.0)
        Hs = R["Hs"]
        mu_train = np.asarray(cache[mu_key])
        sigma_nat = natural_H_spread(Hs, z0, z1, mu_train)
        print(f"  sigma_nat={sigma_nat:.2f}, "
              f"src=({cen_a[0]:.2f},{cen_a[1]:.2f}), "
              f"dst=({cen_b[0]:.2f},{cen_b[1]:.2f})")
        src_z = (float(cen_a[0]), float(cen_a[1]))
        dst_z = (float(cen_b[0]), float(cen_b[1]))

        # Uniform-D
        bars_u, n_u = run_langevin(
            Hs, z0, z1, sigma_nat, src_z, dst_z, N_TRAJ, MAX_STEPS,
            ALPHA, T_DIM, CAPTURE_RADIUS, rng)
        print(f"  uniform-D : {n_u}/{N_TRAJ}, "
              f"median={np.median(bars_u):.2f} sigma_nat")

        # Riemannian
        print(f"  loading Fisher from {fisher_npz.name}")
        g_inv, L_chol = load_and_resample_fisher(fisher_npz, z0, z1)
        bars_r, n_r = run_langevin(
            Hs, z0, z1, sigma_nat, src_z, dst_z, N_TRAJ,
            MAX_STEPS_RIEMANNIAN, ALPHA, T_DIM, CAPTURE_RADIUS, rng,
            g_inv=g_inv, L=L_chol)
        print(f"  Riemannian: {n_r}/{N_TRAJ}, "
              f"median={np.median(bars_r) if len(bars_r) else float('nan'):.2f} sigma_nat")

        results.append(dict(
            family=fam, alphabet=rep, sigma_nat=sigma_nat,
            bars_u=bars_u, bars_r=bars_r, n_u=n_u, n_r=n_r,
        ))

    # ====================== figure: strip plot ======================
    fig, ax = plt.subplots(figsize=(11, 6))
    method_color = {"uniform-D": "#1f77b4", "Riemannian": "#ff7f0e"}
    width = 0.32
    xticks = []; xticklabels = []
    for ci, res in enumerate(results):
        x_center = ci
        xticks.append(x_center)
        xticklabels.append(f"{res['family']}\n{res['alphabet']}")
        for off, key, label, bars, n_cross in [
                (-width / 2, "uniform-D",
                 "uniform-$D$", res["bars_u"], res["n_u"]),
                (+width / 2, "Riemannian",
                 "Riemannian", res["bars_r"], res["n_r"]),
        ]:
            x_jit = (x_center + off
                     + rng.uniform(-width * 0.18, width * 0.18,
                                   size=len(bars)))
            ax.scatter(x_jit, bars, s=14, color=method_color[key],
                       alpha=0.55, edgecolor="none",
                       label=label if ci == 0 else None)
            if len(bars):
                med = float(np.median(bars))
                ax.hlines(med, x_center + off - width * 0.45,
                          x_center + off + width * 0.45,
                          colors="black", linewidth=2.0, zorder=5)
                ax.text(x_center + off,
                        max(bars.max() if len(bars) else 0,
                            med * 1.05) + 0.3,
                        f"med {med:.1f}\nn={n_cross}",
                        ha="center", va="bottom", fontsize=8)
            else:
                # No crossings: mark with empty annotation
                ax.text(x_center + off, 0.3, "n=0",
                        ha="center", va="bottom", fontsize=8,
                        color="0.4", style="italic")

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=11)
    ax.set_ylabel(r"$\Delta H^\ddagger\ /\ \sigma_{H,\mathrm{nat}}$",
                  fontsize=12)
    ax.set_title("Per-trajectory Langevin barrier heights across families "
                 "and representations: uniform-$D$ vs Riemannian (Fisher)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, ls="-", alpha=0.12, axis="y")
    fig.tight_layout()

    out_pdf = OUT_FIG / "03f_langevin_comparison.pdf"
    out_png = OUT_FIG / "03f_langevin_comparison.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_pdf}")

    # ====================== save barriers CSV ======================
    out_csv = OUT_DATA / "03f_langevin_comparison_barriers.csv"
    with open(out_csv, "w") as fh:
        fh.write("family,alphabet,method,deltaH_in_sigma_nat\n")
        for res in results:
            for b in res["bars_u"]:
                fh.write(f"{res['family']},{res['alphabet']},"
                         f"uniform,{b:.4f}\n")
            for b in res["bars_r"]:
                fh.write(f"{res['family']},{res['alphabet']},"
                         f"riemannian,{b:.4f}\n")
    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
