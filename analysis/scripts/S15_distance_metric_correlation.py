"""Supplementary Figure S15: the three latent-distance metrics capture the same trends.

For each protein family (MDH, TRPM, Globin, E1, E2) in both the amino-acid (AA) and 3Di
latent spaces, the script computes the inter-basin / intra-basin separation ratio of the
biologically labeled subgroups under three distance metrics: Euclidean latent distance,
an energy-minimized Fisher-Rao geodesic on the pullback metric, and a Wasserstein-2 optimal-
transport distance. It then makes a three-panel figure of pairwise metric scatter plots across
the 10 family x representation cases (with Pearson and Spearman correlations and a linear fit),
showing the metrics are strongly correlated so distance-based conclusions do not depend on the
metric choice. NOTE: barriers/energies reported elsewhere are in sigma_H (Potts-energy spread),
a model energy scale, NOT a physical temperature; this script uses distances only.

Inputs (paths resolved through analyses/scripts/_paths.py unless noted):
  - OUT_DATA / "pullback_metric_{rep}_{fam}.npz" : per-family, per-representation pullback
      metric field (g, z0_axis, z1_axis), produced by pullback_metric.py; loaded for
      rep in {aa, 3di} and fam in {mdh, trpm, globin, e1, e2}.
  - OUT_DATA / "trpm_aa_per_accession.npz", "globin_aa_per_accession.npz" : per-accession AA
      latent means (accessions, mu) for TRPM and Globin subfamilies.
  - TRPM_ZED_CSV, GLOBIN_ZED_CSV : zed_coordinates_with_classes.csv for TRPM and Globin
      (per-accession 3Di latent z1,z2 and biological class labels).
  - OUT_DATA / "cache_paired_mus.npz" : paired AA/3Di latent means and headers for the E1 and
      E2 Flaviviridae glycoproteins (E1_headers, E1_mu_aa, E1_mu_3di, and E2_* equivalents),
      produced by cache_paired_mus.py.
  - MDH_CLUSTER_DATA / "{psycrophilic_cluster,thermo_meso_cluster}" /
      "cluster_coordinates_{AA,3DI}.txt" : MDH psychrophilic vs thermo/meso cohort latent
      coordinates.

Outputs:
  - OUT_DATA / "distance_metrics_all.csv" : separation ratios per (family, metric, rep) for the
      5 families x 3 metrics x 2 representations.
  - OUT_FIG / "S15_distance_metric_correlation.pdf" and ".png" : the three-panel metric-
      correlation figure.

Usage:
  python analyses/scripts/S15_distance_metric_correlation.py
  No command-line arguments.
"""
from pathlib import Path
import sys, csv, re, itertools
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from _paths import (OUT_FIG, OUT_DATA, GLOBIN_ZED_CSV, TRPM_ZED_CSV, MDH_CLUSTER_DATA)

rng = np.random.default_rng(0)
N_INTER, N_INTRA, MMIN = 30, 25, 15
METRICS = [("euclid", "Euclidean"), ("geo", "Fisher-Rao geodesic"), ("w2", "Wasserstein-2")]
acc_re = re.compile(r"\|([A-Za-z0-9]+)\|")

# ---------------- geometry ----------------
def load_field(fam, rep):
    z = np.load(OUT_DATA / f"pullback_metric_{rep}_{fam}.npz", allow_pickle=True)
    return dict(g=z["g"], z0=z["z0_axis"], z1=z["z1_axis"])

def interp_g(field, pts):
    z0, z1, gg = field["z0"], field["z1"], field["g"]; N0, N1 = len(z0), len(z1)
    px = np.clip(pts[:, 0], z0[0], z0[-1] - 1e-9); py = np.clip(pts[:, 1], z1[0], z1[-1] - 1e-9)
    fx = (px - z0[0]) / (z0[-1] - z0[0]) * (N0 - 1); fy = (py - z1[0]) / (z1[-1] - z1[0]) * (N1 - 1)
    ix = np.clip(np.floor(fx).astype(int), 0, N0 - 2); iy = np.clip(np.floor(fy).astype(int), 0, N1 - 2)
    tx = (fx - ix)[:, None, None]; ty = (fy - iy)[:, None, None]
    return ((1 - tx) * (1 - ty) * gg[ix, iy] + tx * (1 - ty) * gg[ix + 1, iy]
            + (1 - tx) * ty * gg[ix, iy + 1] + tx * ty * gg[ix + 1, iy + 1])

def geo(field, za, zb, M=10):
    t = np.linspace(0, 1, M + 1)[1:-1, None]; init = (za * (1 - t) + zb * t).ravel()
    def energy(flat):
        p = np.vstack([za, flat.reshape(M - 1, 2), zb]); mids = 0.5 * (p[:-1] + p[1:]); dz = np.diff(p, axis=0)
        return np.einsum("ki,kij,kj->k", dz, interp_g(field, mids), dz).sum()
    r = minimize(energy, init, method="L-BFGS-B", options={"maxiter": 120})
    p = np.vstack([za, r.x.reshape(M - 1, 2), zb]); mids = 0.5 * (p[:-1] + p[1:]); dz = np.diff(p, axis=0)
    return float(np.sqrt(np.clip(np.einsum("ki,kij,kj->k", dz, interp_g(field, mids), dz), 0, None)).sum())

def w2(A, B, m=90):
    A = A[rng.choice(len(A), min(len(A), m), replace=False)]; B = B[rng.choice(len(B), min(len(B), m), replace=False)]
    n = min(len(A), len(B)); C = cdist(A[:n], B[:n], "sqeuclidean"); ri, ci = linear_sum_assignment(C)
    return float(np.sqrt(C[ri, ci].mean()))

def prs(A, B, npairs, same):
    out = []
    for _ in range(npairs):
        i = rng.integers(len(A)); j = rng.integers(len(B))
        if same and len(A) > 1:
            while j == i: j = rng.integers(len(B))
        out.append((i, j))
    return out

def matrix(coords, subs, field, kind):
    n = len(subs); M = np.zeros((n, n))
    for i, j in itertools.combinations_with_replacement(range(n), 2):
        A, B = coords[subs[i]], coords[subs[j]]
        if kind == "w2":
            if i == j:
                Ar = A.copy(); rng.shuffle(Ar); h = len(Ar) // 2; d = w2(Ar[:h], Ar[h:])
            else: d = w2(A, B)
        else:
            pr = prs(A, B, N_INTRA if i == j else N_INTER, same=(i == j))
            d = (np.mean([np.linalg.norm(A[a] - B[b]) for a, b in pr]) if kind == "euclid"
                 else np.mean([geo(field, A[a], B[b]) for a, b in pr]))
        M[i, j] = M[j, i] = d
    return M

def sep_ratio(M):
    n = M.shape[0]; return float(M[~np.eye(n, dtype=bool)].mean() / np.diag(M).mean())

# ---------------- per-family loaders -> (subs, AA dict, 3Di dict) ----------------
def _zed(path):
    di, sub = {}, {}
    for r in csv.DictReader(open(path)):
        mm = acc_re.search(r["id"])
        if mm:
            di[mm.group(1)] = np.array([float(r["z1"]), float(r["z2"])]); sub[mm.group(1)] = r["class"]
    return di, sub

def load_subfam(fam, per_acc_npz, zed_csv, members):
    z = np.load(OUT_DATA / per_acc_npz, allow_pickle=True)
    aa = {str(a): m for a, m in zip(z["accessions"], z["mu"])}
    di, sub = _zed(zed_csv)
    keep = [a for a in aa if a in di and sub.get(a) in members]
    A = {s: np.array([aa[a] for a in keep if sub[a] == s]) for s in members}
    D = {s: np.array([di[a] for a in keep if sub[a] == s]) for s in members}
    return members, A, D

def load_flavi(fam):
    d = np.load(OUT_DATA / "cache_paired_mus.npz", allow_pickle=True)
    hdr = [str(x).lower() for x in d[f"{fam}_headers"]]; aa = d[f"{fam}_mu_aa"]; di = d[f"{fam}_mu_3di"]
    gen = lambda s: next((g for g in ["hepaci", "pegi", "pesti"] if g in s), None)
    gl = [gen(h) for h in hdr]; gens = ["hepaci", "pegi", "pesti"]
    return gens, {g: aa[[i for i, x in enumerate(gl) if x == g]] for g in gens}, \
           {g: di[[i for i, x in enumerate(gl) if x == g]] for g in gens}

def load_mdh():
    def dct(p): return np.array(list(eval(Path(p).read_text(), {"__builtins__": {}}).values()), float)
    subs = ["psychro", "thermo/meso"]
    aa = {"psychro": dct(MDH_CLUSTER_DATA / "psycrophilic_cluster" / "cluster_coordinates_AA.txt"),
          "thermo/meso": dct(MDH_CLUSTER_DATA / "thermo_meso_cluster" / "cluster_coordinates_AA.txt")}
    di = {"psychro": dct(MDH_CLUSTER_DATA / "psycrophilic_cluster" / "cluster_coordinates_3DI.txt"),
          "thermo/meso": dct(MDH_CLUSTER_DATA / "thermo_meso_cluster" / "cluster_coordinates_3DI.txt")}
    return subs, aa, di

GLOBIN_SUB = ["Cytoglobin", "Myoglobin", "Flavohemoglobin", "Neuroglobin",
              "Leghemoglobin", "Bacterial hemoglobin", "Subunit Alpha", "Subunit Beta"]
TRPM_SUB = [f"member {k}" for k in [1, 2, 3, 4, 5, 7, 8]]
FAMS = {
    "mdh": ("MDH", load_mdh),
    "trpm": ("TRPM", lambda: load_subfam("trpm", "trpm_aa_per_accession.npz", TRPM_ZED_CSV, TRPM_SUB)),
    "globin": ("Globin", lambda: load_subfam("globin", "globin_aa_per_accession.npz", GLOBIN_ZED_CSV, GLOBIN_SUB)),
    "e1": ("E1", lambda: load_flavi("E1")),
    "e2": ("E2", lambda: load_flavi("E2")),
}

# ---------------- compute ----------------
rows = []
for fkey, (fname, loader) in FAMS.items():
    print(f"=== {fname} ===")
    subs, AA, DI = loader()
    subs = [s for s in subs if len(AA[s]) >= MMIN and len(DI[s]) >= MMIN]
    F = {"aa": load_field(fkey, "aa"), "3di": load_field(fkey, "3di")}
    CO = {"aa": AA, "3di": DI}
    for mkey, _ in METRICS:
        for rep in ("aa", "3di"):
            rows.append(dict(family=fname, metric=mkey, rep=rep,
                             ratio=sep_ratio(matrix(CO[rep], subs, F[rep], mkey))))
with open(OUT_DATA / "distance_metrics_all.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["family", "metric", "rep", "ratio"]); w.writeheader(); w.writerows(rows)

# ---------------- figure: pairwise metric correlation across 10 family x rep cases ----------------
FAM_ORDER = ["MDH", "TRPM", "Globin", "E1", "E2"]
FCOL = {"MDH": "#d62728", "TRPM": "#9467bd", "Globin": "#2ca02c", "E1": "#1f77b4", "E2": "#ff7f0e"}
val = lambda fam, mk, rep: next(r["ratio"] for r in rows if r["family"] == fam and r["metric"] == mk and r["rep"] == rep)
cases = [(f, rep) for f in FAM_ORDER for rep in ("aa", "3di")]
E = np.array([val(f, "euclid", r) for f, r in cases]); G = np.array([val(f, "geo", r) for f, r in cases])
W = np.array([val(f, "w2", r) for f, r in cases])
pairs = [("Euclidean", E, "Fisher-Rao geodesic", G), ("Euclidean", E, "Wasserstein-2", W),
         ("Fisher-Rao geodesic", G, "Wasserstein-2", W)]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, (xn, xv, yn, yv) in zip(axes, pairs):
    for (f, rep), xi, yi in zip(cases, xv, yv):
        ax.scatter(xi, yi, s=85, c=FCOL[f], marker="o" if rep == "aa" else "s", edgecolor="k", lw=0.6, zorder=3)
    m, b = np.polyfit(xv, yv, 1); xs = np.array([xv.min(), xv.max()]); ax.plot(xs, m * xs + b, "k--", lw=1.3)
    ax.set_xlabel(xn + "  (inter/intra)", fontsize=10); ax.set_ylabel(yn + "  (inter/intra)", fontsize=10)
    ax.set_title(f"Pearson $r$={pearsonr(xv, yv)[0]:.2f},  Spearman={spearmanr(xv, yv)[0]:.2f}", fontsize=10)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
from matplotlib.lines import Line2D
axes[0].legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=FCOL[f], markeredgecolor="k", markersize=8, label=f) for f in FAM_ORDER]
               + [Line2D([0], [0], marker="o", color="w", markerfacecolor="0.5", markeredgecolor="k", markersize=8, label="AA"),
                  Line2D([0], [0], marker="s", color="w", markerfacecolor="0.5", markeredgecolor="k", markersize=8, label="3Di")],
               fontsize=8, frameon=False, loc="upper left", ncol=2)
fig.suptitle("The three distance metrics capture the same trends (separation ratio, 10 family x representation cases)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"S15_distance_metric_correlation.{ext}", dpi=185, bbox_inches="tight")
print("wrote", OUT_FIG / "S15_distance_metric_correlation.pdf")
