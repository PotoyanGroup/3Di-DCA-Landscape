"""Supplementary Figure S10: uniform-D Langevin trajectories over the
learned latent free-energy landscapes for MDH, TRPM, and Globin in both
amino-acid (AA) and 3Di representations.

The figure is a 3 (family) x 2 (representation) grid of panels. For each
family/representation the script loads the latent grid, rebuilds a smoothed
free-energy surface H via watershed_disconnectivity.compute_landscape,
normalizes it by the natural spread of H over the training latent means,
and integrates an ensemble of N_TRAJ=80 overdamped uniform-diffusivity
Langevin trajectories between a chosen source and target subfamily
centroid (with absorbing capture at the target). Each panel renders the H
landscape (filled + line contours), marks cohort centroids (source = blue
star, target = red filled diamond, others = open circles), overlays a
subsample of the reactive trajectory segments as thin red lines, and
annotates the reactive fraction. It is the companion geometry view to the
barrier-distribution analysis; Riemannian (position-dependent diffusivity)
dynamics are intentionally omitted as they add no insight beyond uniform-D
for cross-family comparison.

Inputs (paths resolved through analyses/scripts/_paths.py unless noted):
  - watershed_disconnectivity.py : sibling script imported dynamically to
    reuse compute_landscape() for building the smoothed H surface
  - OUT_DATA / "cache_paired_mus.npz" (CACHE_MU) : cached per-family
    training latent means, keyed by MDH/TRPM/Globin_mu_aa and _mu_3di;
    used to compute the natural H spread for normalization
  - MDH_GRID_PKL, MDH_HAMIL, MDH_3DI_GRID_PKL : MDH AA grid pickle,
    AA Hamiltonian matrix, and 3Di grid pickle
  - MDH_CLUSTER_DATA (MDH_CL) : MDH cohort cluster directory; reads
    psycrophilic_cluster / thermo_meso_cluster cluster_coordinates_AA.txt
    and cluster_coordinates_3DI.txt for MDH source/target centroids
  - TRPM_AA_DIR / "all_grid.pkl" (TRPM_AA_PKL), TRPM_3DI_DIR /
    "all_grid.pkl" (TRPM_3DI_PKL) : TRPM AA and 3Di latent grids
  - OUT_DATA / "00e_trpm_aa_subfamily_mu.npz" (TRPM_AA_NPZ) : TRPM AA
    subfamily centroids (per-subfamily *_centroid arrays)
  - TRPM_3DI_DIR / "zed_coordinates_with_classes.csv" (TRPM_ZED_CSV) :
    TRPM 3Di latent coordinates with class labels, for 3Di centroids
  - GLOBIN_AA_DIR / "all_grid.pkl" (GLOBIN_AA_PKL), GLOBIN_3DI_DIR /
    "all_grid.pkl" (GLOBIN_3DI_PKL) : Globin AA and 3Di latent grids
  - OUT_DATA / "00d_globin_aa_subfamily_mu.npz" (GLOBIN_AA_NPZ) : Globin
    AA subfamily centroids (per-subfamily *_centroid arrays)
  - GLOBIN_3DI_DIR / "zed_coordinates_with_classes.csv" (GLOBIN_ZED_CSV) :
    Globin 3Di latent coordinates with class labels, for 3Di centroids

Outputs:
  - OUT_FIG / "03p_langevin_trajectories_si.pdf" : the 3x2 figure (PDF)
  - OUT_FIG / "03p_langevin_trajectories_si.png" : the 3x2 figure (PNG,
    160 dpi)

Usage:
  python analyses/scripts/topography/S10_langevin_trajectories.py
  No command-line arguments. Sampling/integration are fixed module-level
  constants (e.g. SEED=0, N_TRAJ=80, MAX_STEPS=60000, ALPHA=0.01,
  T_DIM=1.0, CAPTURE_RADIUS=0.4, N_TRAJ_DRAW=10).
"""
from pathlib import Path
import pickle
import sys
import csv
import ast
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

import importlib.util
_spec_w = importlib.util.spec_from_file_location(
    "main_script",
    Path(__file__).resolve().parent / "watershed_disconnectivity.py",
)
m = importlib.util.module_from_spec(_spec_w)
_spec_w.loader.exec_module(m)

from _paths import (OUT_FIG, OUT_DATA, MDH_CLUSTER_DATA, GLOBIN_AA_DIR,  # noqa: E402
                    GLOBIN_3DI_DIR, TRPM_AA_DIR, TRPM_3DI_DIR,
                    MDH_GRID_PKL, MDH_HAMIL, MDH_3DI_GRID_PKL)

GLOBIN_3DI_PKL = GLOBIN_3DI_DIR / "all_grid.pkl"
GLOBIN_AA_PKL  = GLOBIN_AA_DIR / "all_grid.pkl"
GLOBIN_ZED_CSV = GLOBIN_3DI_DIR / "zed_coordinates_with_classes.csv"
GLOBIN_AA_NPZ  = OUT_DATA / "00d_globin_aa_subfamily_mu.npz"

TRPM_3DI_PKL   = TRPM_3DI_DIR / "all_grid.pkl"
TRPM_AA_PKL    = TRPM_AA_DIR / "all_grid.pkl"
TRPM_ZED_CSV   = TRPM_3DI_DIR / "zed_coordinates_with_classes.csv"
TRPM_AA_NPZ    = OUT_DATA / "00e_trpm_aa_subfamily_mu.npz"

MDH_CL = MDH_CLUSTER_DATA
CACHE_MU = OUT_DATA / "cache_paired_mus.npz"

T_DIM = 1.0
ALPHA = 0.01
MAX_STEPS = 60_000
N_TRAJ = 80
N_TRAJ_DRAW = 10    # subsample for visualization
CAPTURE_RADIUS = 0.4
EXIT_RADIUS = 0.5
SEED = 0

SHORT = {
    "Cytoglobin": "Cyto", "Flavohemoglobin": "Flavo",
    "Myoglobin": "Myo", "Neuroglobin": "Neuro",
    "Subunit Alpha": r"$\alpha$-sub",
    "Psy": "Psy", "T_M": "T/M",
    "TRPM1": "TRPM1", "TRPM3": "TRPM3",
    "TRPM7": "TRPM7", "TRPM8": "TRPM8",
}


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


def natural_H_spread(H, z0, z1, mu_train):
    Hs = [H[int(np.argmin(np.abs(z0 - z0v))),
            int(np.argmin(np.abs(z1 - z1v)))]
          for z0v, z1v in mu_train]
    return float(np.std(Hs))


def zed_centroids(csv_path, class_label_map):
    pts = {k: [] for k in class_label_map}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            cls = r["class"].strip()
            for our_name, zed_val in class_label_map.items():
                if cls == zed_val:
                    try:
                        pts[our_name].append((float(r["z1"]),
                                              float(r["z2"])))
                    except (ValueError, KeyError):
                        pass
    return {name: np.array(p).mean(axis=0) for name, p in pts.items()}


def load_mdh_centroids():
    aa, di = {}, {}
    for label, sub in [("Psy", "psycrophilic_cluster"),
                        ("T_M", "thermo_meso_cluster")]:
        aa_pts = ast.literal_eval(
            (MDH_CL / sub / "cluster_coordinates_AA.txt").read_text())
        di_pts = ast.literal_eval(
            (MDH_CL / sub / "cluster_coordinates_3DI.txt").read_text())
        aa[label] = np.array(list(aa_pts.values())).mean(axis=0)
        di[label] = np.array(list(di_pts.values())).mean(axis=0)
    return aa, di


def run_langevin_capture_paths(Hs, z0, z1, sigma_nat, src_z, dst_z,
                                n_traj, max_steps, alpha, T_dim,
                                capture_radius, rng,
                                record_every=15):
    """Same uniform-D Langevin as 03j; additionally records the path
    of each trajectory at decimated time-steps. Returns list of
    (T_i, 2) arrays per trajectory and a `crossed` boolean array.
    """
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
    crossed = np.zeros(n_traj, dtype=bool)
    paths = [[(pos0[k], pos1[k])] for k in range(n_traj)]
    sigma_step = np.sqrt(2 * alpha * T_dim)
    for step in range(max_steps):
        if not alive.any():
            break
        idx = np.where(alive)[0]
        ia = idx0(pos0[idx]); ja = idx1(pos1[idx])
        drift0 = -alpha * g0[ia, ja]
        drift1 = -alpha * g1[ia, ja]
        kick0 = sigma_step * rng.standard_normal(idx.size)
        kick1 = sigma_step * rng.standard_normal(idx.size)
        pos0[idx] = np.clip(pos0[idx] + drift0 + kick0, z0_min, z0_max)
        pos1[idx] = np.clip(pos1[idx] + drift1 + kick1, z1_min, z1_max)
        if step % record_every == 0:
            for k in idx:
                paths[k].append((pos0[k], pos1[k]))
        d_b = (pos0 - dst_z[0])**2 + (pos1 - dst_z[1])**2
        near_b = d_b < capture_radius**2
        reached = near_b & alive
        for k in np.where(reached)[0]:
            paths[k].append((pos0[k], pos1[k]))
        crossed |= reached
        alive &= ~reached

    paths_arr = [np.asarray(p) for p in paths]
    return paths_arr, crossed


GLOBIN_SUBFAMS = ["Cytoglobin", "Flavohemoglobin", "Myoglobin",
                  "Neuroglobin", "Subunit Alpha"]
TRPM_SUBFAMS = ["TRPM1", "TRPM3", "TRPM7", "TRPM8"]
GLOBIN_ZED_MAP = {s: s for s in GLOBIN_SUBFAMS}
TRPM_ZED_MAP = {"TRPM1": "member 1", "TRPM3": "member 3",
                "TRPM7": "member 7", "TRPM8": "member 8"}

FAMILIES = [
    {"name": "MDH", "is_mdh": True,
     "subfams": ["Psy", "T_M"],
     "pair": ("Psy", "T_M"),
     "aa_pkl": MDH_GRID_PKL, "aa_hamil": MDH_HAMIL,
     "3di_pkl": MDH_3DI_GRID_PKL,
     "cache_aa_key": "MDH_mu_aa", "cache_3di_key": "MDH_mu_3di",
     "regime": "consistent"},
    {"name": "TRPM", "is_mdh": False,
     "subfams": TRPM_SUBFAMS,
     "pair": ("TRPM3", "TRPM8"),
     "aa_pkl": TRPM_AA_PKL, "3di_pkl": TRPM_3DI_PKL,
     "aa_npz": TRPM_AA_NPZ, "zed_csv": TRPM_ZED_CSV,
     "zed_map": TRPM_ZED_MAP,
     "cache_aa_key": "TRPM_mu_aa", "cache_3di_key": "TRPM_mu_3di",
     "regime": "intermediate"},
    {"name": "Globin", "is_mdh": False,
     "subfams": GLOBIN_SUBFAMS,
     "pair": ("Cytoglobin", "Flavohemoglobin"),
     "aa_pkl": GLOBIN_AA_PKL, "3di_pkl": GLOBIN_3DI_PKL,
     "aa_npz": GLOBIN_AA_NPZ, "zed_csv": GLOBIN_ZED_CSV,
     "zed_map": GLOBIN_ZED_MAP,
     "cache_aa_key": "Globin_mu_aa", "cache_3di_key": "Globin_mu_3di",
     "regime": "complementary"},
]


def render_panel(ax, H, z0, z1, cent_dict, pair, paths, crossed,
                 title):
    Z0, Z1 = np.meshgrid(z0, z1, indexing="ij")
    # Light grayscale fill for context (low alpha)
    ax.contourf(Z0, Z1, H, levels=18, cmap="Greys", alpha=0.30)
    # Dark contour lines for structure
    ax.contour(Z0, Z1, H, levels=14, colors="0.35",
               alpha=0.55, linewidths=0.5)
    src, dst = pair
    src_xy = cent_dict[src]
    crossed_idx = np.where(crossed)[0]
    rng_local = np.random.default_rng(1)
    if len(crossed_idx) > N_TRAJ_DRAW:
        chosen = rng_local.choice(crossed_idx, size=N_TRAJ_DRAW,
                                  replace=False)
    else:
        chosen = crossed_idx
    for k in chosen:
        p = paths[k]
        if len(p) < 2:
            continue
        # Trim to reactive segment: from LAST exit from source basin
        # to capture at target
        d = np.hypot(p[:, 0] - src_xy[0], p[:, 1] - src_xy[1])
        in_src = d < 0.55
        if in_src.any():
            last_in_src = int(np.where(in_src)[0][-1])
            p = p[last_in_src:]
        if len(p) < 2:
            continue
        ax.plot(p[:, 0], p[:, 1], color="#c8344c", lw=0.7,
                alpha=0.55, zorder=2)
    # Other cohort markers (background)
    for n, c in cent_dict.items():
        if n in (src, dst):
            continue
        ax.scatter(c[0], c[1], s=80, marker="o",
                   facecolor="white", edgecolor="black",
                   linewidth=1.0, zorder=8)
        ax.annotate(SHORT.get(n, n), xy=c, xytext=(7, 5),
                    textcoords="offset points",
                    fontsize=8, fontweight="bold",
                    color="black", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="white", ec="0.6", lw=0.4,
                              alpha=0.85))
    # Source: star
    cs = cent_dict[src]
    ax.scatter(cs[0], cs[1], s=240, marker="*",
               facecolor="#3b8bd2", edgecolor="black",
               linewidth=1.0, zorder=10)
    ax.annotate(SHORT.get(src, src) + " (source)",
                xy=cs, xytext=(10, 10),
                textcoords="offset points",
                fontsize=9, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.25",
                          fc="#3b8bd2", ec="black", lw=0.5,
                          alpha=0.95),
                zorder=11)
    # Target: filled diamond
    cd = cent_dict[dst]
    ax.scatter(cd[0], cd[1], s=180, marker="D",
               facecolor="#ff3554", edgecolor="black",
               linewidth=1.0, zorder=10)
    ax.annotate(SHORT.get(dst, dst) + " (target)",
                xy=cd, xytext=(10, 10),
                textcoords="offset points",
                fontsize=9, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.25",
                          fc="#ff3554", ec="black", lw=0.5,
                          alpha=0.95),
                zorder=11)
    n_cross = int(crossed.sum())
    ax.text(0.98, 0.03, f"{n_cross}/{len(paths)} reactive",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="0.20",
            bbox=dict(boxstyle="round,pad=0.25",
                      fc="white", ec="0.5", lw=0.4, alpha=0.92))
    ax.set_xlabel(r"$z_1$", fontsize=9)
    ax.set_ylabel(r"$z_2$", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8)


def main():
    rng = np.random.default_rng(SEED)
    cache = np.load(CACHE_MU, allow_pickle=True)

    fig, axes = plt.subplots(3, 2, figsize=(12, 14),
                              gridspec_kw={"hspace": 0.32,
                                           "wspace": 0.22})

    for row, fam in enumerate(FAMILIES):
        is_mdh = fam.get("is_mdh", False)
        if is_mdh:
            mdh_aa_cent, mdh_3di_cent = load_mdh_centroids()
            aa_cent = mdh_aa_cent
            di_cent = mdh_3di_cent
        else:
            aa_data = np.load(fam["aa_npz"])
            aa_cent = {s: aa_data[f"{s}_centroid"] for s in fam["subfams"]}
            di_cent = zed_centroids(fam["zed_csv"], fam["zed_map"])

        for col, (rep, pkl, cache_key, cent) in enumerate([
            ("AA",  fam["aa_pkl"],  fam["cache_aa_key"],  aa_cent),
            ("3Di", fam["3di_pkl"], fam["cache_3di_key"], di_cent),
        ]):
            print(f"\n--- {fam['name']} {rep} ---")
            hamil = fam.get("aa_hamil") if rep == "AA" else None
            H, z0, z1 = load_grid(pkl, hamil)
            R = m.compute_landscape(H, z0, z1, 8, sigma=1.0)
            Hs = R["Hs"]
            mu_train = np.asarray(cache[cache_key])
            sigma_nat = natural_H_spread(Hs, z0, z1, mu_train)
            src, dst = fam["pair"]
            src_z = (float(cent[src][0]), float(cent[src][1]))
            dst_z = (float(cent[dst][0]), float(cent[dst][1]))
            paths, crossed = run_langevin_capture_paths(
                Hs, z0, z1, sigma_nat, src_z, dst_z,
                N_TRAJ, MAX_STEPS, ALPHA, T_DIM, CAPTURE_RADIUS, rng,
                record_every=40,
            )
            print(f"  {src} -> {dst}: {crossed.sum()}/{N_TRAJ} crossings")
            title = (rf"{fam['name']}  {rep}    "
                     rf"({fam['regime']} regime)")
            render_panel(axes[row, col], Hs, z0, z1, cent,
                         fam["pair"], paths, crossed, title)

    out_pdf = OUT_FIG / "03p_langevin_trajectories_si.pdf"
    out_png = OUT_FIG / "03p_langevin_trajectories_si.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_pdf}")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
