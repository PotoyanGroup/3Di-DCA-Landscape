"""All-pairs uniform-D Langevin between subfamily centroids per family,
in both AA and 3Di representations. Produces the per-family barrier
matrix that serves as input to the Markov-model analysis (03k).

Per family:
  - read centroids (AA from encoded *_subfamily_mu.npz,
    3Di from zed_coordinates_with_classes.csv)
  - for every pair (src, dst) and each rep:
      * run uniform-D Langevin
      * record (n_crossed, median, mean, std) per pair-rep

Outputs:
  - outputs/data/03j_all_pairs_barriers_summary.csv  (one row / pair-rep)
  - outputs/data/03j_all_pairs_barriers_raw.csv      (per-trajectory)
"""
from pathlib import Path
import pickle
import sys
import csv
import itertools
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "main_script",
    Path(__file__).resolve().parent / "watershed_disconnectivity.py",
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

from _paths import OUT_DATA
from _paths import GLOBIN_AA_DIR, GLOBIN_3DI_DIR, TRPM_AA_DIR, TRPM_3DI_DIR  # noqa: E402

GLOBIN_3DI_PKL = GLOBIN_3DI_DIR / "all_grid.pkl"
GLOBIN_AA_PKL  = GLOBIN_AA_DIR / "all_grid.pkl"
GLOBIN_ZED_CSV = GLOBIN_3DI_DIR / "zed_coordinates_with_classes.csv"
GLOBIN_AA_NPZ  = OUT_DATA / "00d_globin_aa_subfamily_mu.npz"

TRPM_3DI_PKL   = TRPM_3DI_DIR / "all_grid.pkl"
TRPM_AA_PKL    = TRPM_AA_DIR / "all_grid.pkl"
TRPM_ZED_CSV   = TRPM_3DI_DIR / "zed_coordinates_with_classes.csv"
TRPM_AA_NPZ    = OUT_DATA / "00e_trpm_aa_subfamily_mu.npz"

CACHE_MU = OUT_DATA / "cache_paired_mus.npz"

T_DIM = 1.0
ALPHA = 0.01
MAX_STEPS = 120_000
N_TRAJ = 200            # reduced from 400 for total wall-time
CAPTURE_RADIUS = 0.4
EXIT_RADIUS = 0.5
SEED = 0


def load_grid(pkl_path):
    pts = pickle.load(open(pkl_path, "rb"))
    z0 = np.unique(pts[:, 0]); z1 = np.unique(pts[:, 1])
    H = pts[:, 2].reshape(z0.size, z1.size)
    return H, z0, z1


def natural_H_spread(H, z0, z1, mu_train):
    Hs = [H[int(np.argmin(np.abs(z0 - z0v))),
            int(np.argmin(np.abs(z1 - z1v)))]
          for z0v, z1v in mu_train]
    return float(np.std(Hs))


def zed_centroids(csv_path, class_label_map):
    """class_label_map: {our_name: zed_class_string}.
    Returns {our_name: (z1, z2)}."""
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


def run_langevin(Hs, z0, z1, sigma_nat, src_z, dst_z,
                 n_traj, max_steps, alpha, T_dim, capture_radius, rng):
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
        drift0 = -alpha * g0[ia, ja]
        drift1 = -alpha * g1[ia, ja]
        kick0 = sigma_step * rng.standard_normal(idx.size)
        kick1 = sigma_step * rng.standard_normal(idx.size)
        pos0[idx] = np.clip(pos0[idx] + drift0 + kick0, z0_min, z0_max)
        pos1[idx] = np.clip(pos1[idx] + drift1 + kick1, z1_min, z1_max)
        ia2 = idx0(pos0[idx]); ja2 = idx1(pos1[idx])
        h_now = H_norm[ia2, ja2]
        d_a = (pos0 - src_z[0])**2 + (pos1 - src_z[1])**2
        d_b = (pos0 - dst_z[0])**2 + (pos1 - dst_z[1])**2
        near_a = d_a < EXIT_RADIUS**2
        near_b = d_b < capture_radius**2
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


# ============================================================
# Per-family configuration
# ============================================================
GLOBIN_SUBFAMS = ["Cytoglobin", "Flavohemoglobin", "Myoglobin",
                  "Neuroglobin", "Subunit Alpha"]
TRPM_SUBFAMS = ["TRPM1", "TRPM3", "TRPM7", "TRPM8"]

GLOBIN_ZED_MAP = {s: s for s in GLOBIN_SUBFAMS}
TRPM_ZED_MAP = {"TRPM1": "member 1", "TRPM3": "member 3",
                "TRPM7": "member 7", "TRPM8": "member 8"}

FAMILIES = [
    {
        "name": "Globin",
        "subfams": GLOBIN_SUBFAMS,
        "aa_npz": GLOBIN_AA_NPZ,
        "zed_csv": GLOBIN_ZED_CSV,
        "zed_map": GLOBIN_ZED_MAP,
        "aa_pkl": GLOBIN_AA_PKL,
        "3di_pkl": GLOBIN_3DI_PKL,
        "cache_aa_key": "Globin_mu_aa",
        "cache_3di_key": "Globin_mu_3di",
    },
    {
        "name": "TRPM",
        "subfams": TRPM_SUBFAMS,
        "aa_npz": TRPM_AA_NPZ,
        "zed_csv": TRPM_ZED_CSV,
        "zed_map": TRPM_ZED_MAP,
        "aa_pkl": TRPM_AA_PKL,
        "3di_pkl": TRPM_3DI_PKL,
        "cache_aa_key": "TRPM_mu_aa",
        "cache_3di_key": "TRPM_mu_3di",
    },
]


def main():
    rng = np.random.default_rng(SEED)
    cache = np.load(CACHE_MU, allow_pickle=True)

    summary_rows = []
    raw_rows = []

    for fam in FAMILIES:
        print(f"\n############### {fam['name']} ###############")
        # Centroids in both reps
        aa_data = np.load(fam["aa_npz"])
        aa_cent = {s: aa_data[f"{s}_centroid"] for s in fam["subfams"]}
        di_cent = zed_centroids(fam["zed_csv"], fam["zed_map"])

        for rep, pkl, cache_key, centroids in [
            ("AA",  fam["aa_pkl"],  fam["cache_aa_key"],  aa_cent),
            ("3Di", fam["3di_pkl"], fam["cache_3di_key"], di_cent),
        ]:
            print(f"\n--- {fam['name']} {rep} ---")
            H, z0, z1 = load_grid(pkl)
            R = m.compute_landscape(H, z0, z1, 8, sigma=1.0)
            Hs = R["Hs"]
            mu_train = np.asarray(cache[cache_key])
            sigma_nat = natural_H_spread(Hs, z0, z1, mu_train)
            print(f"  sigma_nat = {sigma_nat:.2f}")

            for a, b in itertools.combinations(fam["subfams"], 2):
                ca, cb = centroids[a], centroids[b]
                src_z = (float(ca[0]), float(ca[1]))
                dst_z = (float(cb[0]), float(cb[1]))
                bars, n_cross = run_langevin(
                    Hs, z0, z1, sigma_nat, src_z, dst_z,
                    N_TRAJ, MAX_STEPS, ALPHA, T_DIM, CAPTURE_RADIUS,
                    rng)
                med = float(np.median(bars)) if len(bars) else float("nan")
                mean = float(bars.mean()) if len(bars) else float("nan")
                std = float(bars.std()) if len(bars) else float("nan")
                summary_rows.append({
                    "family": fam["name"], "rep": rep,
                    "src": a, "dst": b,
                    "n_crossed": n_cross, "n_traj": N_TRAJ,
                    "median": med, "mean": mean, "std": std,
                    "sigma_nat": sigma_nat,
                })
                for v in bars:
                    raw_rows.append({
                        "family": fam["name"], "rep": rep,
                        "src": a, "dst": b,
                        "deltaH_in_sigma_nat": float(v),
                    })
                print(f"  {a:>17}  ->  {b:<17}  "
                      f"n={n_cross}/{N_TRAJ}  med={med:.2f}  "
                      f"mean={mean:.2f}  std={std:.2f}")

    out_summary = OUT_DATA / "03j_all_pairs_barriers_summary.csv"
    with open(out_summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nsaved {out_summary}  ({len(summary_rows)} pair-rep rows)")

    out_raw = OUT_DATA / "03j_all_pairs_barriers_raw.csv"
    with open(out_raw, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(raw_rows[0].keys()))
        w.writeheader()
        w.writerows(raw_rows)
    print(f"saved {out_raw}  ({len(raw_rows)} barrier values)")


if __name__ == "__main__":
    main()
