"""Watershed-colored latent + matching disconnectivity (merge) tree.

Coordinated 2-panel view of an MDH Hamiltonian landscape:
  Left  - H(z) heatmap, top-K robust basins colored by watershed.
  Right - Wales-style disconnectivity graph; leaf colors match watershed.

The K robust basins are the top-K by persistence (basin minimum to saddle).
Union-find on watershed adjacencies gives both persistence and the merge
tree directly, keeping basin labels consistent between the two panels.

Outputs:
  01b_watershed_disconnectivity_aa_kN.pdf  + .png + .json   (AA, K)
  01b_watershed_disconnectivity_3di_kN.pdf + .png + .json   (3Di, K)
  01b_compare_aa_vs_3di_kN.pdf             + .png           (combined)
"""
from pathlib import Path
import json
import pickle
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.ndimage import gaussian_filter, minimum_filter
from skimage.segmentation import watershed

from _paths import OUT_FIG, OUT_DATA, MDH_GRID_PKL, MDH_HAMIL, MDH_3DI_GRID_PKL


SIGMA = 1.0
KS = [4, 6]


# ---------- landscape loaders ----------
def load_aa_mdh():
    with open(MDH_GRID_PKL, "rb") as f:
        pts = pickle.load(f)
    H = np.load(MDH_HAMIL)
    z0 = np.unique(pts[:, 0]); z1 = np.unique(pts[:, 1])
    if H.shape != (z0.size, z1.size):
        H = pts[:, 2].reshape(z0.size, z1.size)
    return H, z0, z1


def load_3di_mdh():
    with open(MDH_3DI_GRID_PKL, "rb") as f:
        pts = pickle.load(f)
    z0 = np.unique(pts[:, 0]); z1 = np.unique(pts[:, 1])
    H = pts[:, 2].reshape(z0.size, z1.size)
    return H, z0, z1


# ---------- core: basins + merge tree ----------
def adjacent_saddles(labels, H):
    saddles = {}
    rows, cols = labels.shape
    for di, dj in [(0, 1), (1, 0)]:
        a = labels[:rows - di, :cols - dj]
        b = labels[di:, dj:]
        ha = H[:rows - di, :cols - dj]
        hb = H[di:, dj:]
        mask = a != b
        if not mask.any():
            continue
        la = a[mask]; lb = b[mask]
        sad = np.maximum(ha[mask], hb[mask])
        lo = np.minimum(la, lb); hi = np.maximum(la, lb)
        for x, y, s in zip(lo, hi, sad):
            key = (int(x), int(y))
            if key not in saddles or s < saddles[key]:
                saddles[key] = float(s)
    return saddles


class Node:
    __slots__ = ("basin_id", "H", "children", "x")
    def __init__(self, basin_id=None, H=None):
        self.basin_id = basin_id
        self.H = H
        self.children = []
        self.x = None


def compute_landscape(H, z0, z1, K, sigma=SIGMA):
    """Returns dict with everything plotting needs."""
    Hs = gaussian_filter(H, sigma=sigma)

    # all local minima
    local_min = (Hs == minimum_filter(Hs, size=3))
    local_min[0, :] = local_min[-1, :] = False
    local_min[:, 0] = local_min[:, -1] = False
    minima_ij = np.argwhere(local_min)

    seeds = np.zeros_like(Hs, dtype=np.int32)
    for k, (i, j) in enumerate(minima_ij, start=1):
        seeds[i, j] = k
    full_basins = watershed(Hs, markers=seeds)

    saddles = adjacent_saddles(full_basins, Hs)
    H_min = {k: float(Hs[i, j]) for k, (i, j) in enumerate(minima_ij, start=1)}

    # persistence via union-find
    parent = {k: k for k in H_min}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    persistence = {k: np.inf for k in H_min}
    edges_sorted = sorted(saddles.items(), key=lambda kv: kv[1])
    for (a, b), s in edges_sorted:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        dying = ra if H_min[ra] > H_min[rb] else rb
        surviving = rb if dying == ra else ra
        persistence[dying] = s - H_min[dying]
        parent[dying] = surviving

    ranked = sorted(H_min, key=lambda k: persistence[k], reverse=True)
    robust_ids = ranked[:K]
    robust = set(robust_ids)

    # absorb small basins into nearest robust basin
    parent2 = {k: k for k in H_min}
    def find2(x):
        while parent2[x] != x:
            parent2[x] = parent2[parent2[x]]
            x = parent2[x]
        return x
    for (a, b), s in edges_sorted:
        ra, rb = find2(a), find2(b)
        if ra == rb:
            continue
        ra_r = ra in robust; rb_r = rb in robust
        if ra_r and rb_r:
            continue
        if ra_r:
            parent2[rb] = ra
        elif rb_r:
            parent2[ra] = rb
        else:
            if H_min[ra] > H_min[rb]:
                parent2[ra] = rb
            else:
                parent2[rb] = ra
    final = np.zeros_like(full_basins)
    for k in H_min:
        root = find2(k)
        if root in robust:
            final[full_basins == k] = root

    # merge tree on the K robust basins
    robust_saddles = {kv: v for kv, v in adjacent_saddles(final, Hs).items()
                      if 0 not in kv}
    leaves = {k: Node(basin_id=k, H=H_min[k]) for k in robust_ids}
    forest = {k: k for k in robust_ids}
    nodes = dict(leaves)
    def froot(x):
        while forest[x] != x:
            forest[x] = forest[forest[x]]
            x = forest[x]
        return x
    merge_root = None
    tree_saddles = []
    for (a, b), s in sorted(robust_saddles.items(), key=lambda kv: kv[1]):
        ra, rb = froot(a), froot(b)
        if ra == rb:
            continue
        internal = Node(basin_id=None, H=s)
        internal.children = [nodes[ra], nodes[rb]]
        key = ("m", ra, rb)
        nodes[key] = internal
        forest[ra] = key; forest[rb] = key; forest[key] = key
        merge_root = internal
        tree_saddles.append(s)
    if merge_root is None:
        merge_root = Node(basin_id=None, H=float(Hs.max()))
        merge_root.children = list(leaves.values())

    def assign_x(node, counter):
        if node.basin_id is not None:
            node.x = counter[0]; counter[0] += 1; return node.x
        xs = [assign_x(c, counter) for c in node.children]
        node.x = float(np.mean(xs)); return node.x
    assign_x(merge_root, [0])

    return {
        "H": H, "Hs": Hs, "z0": z0, "z1": z1,
        "minima_ij": minima_ij,
        "robust_ids": robust_ids,
        "H_min": H_min,
        "persistence": persistence,
        "final": final,
        "leaves": leaves,
        "merge_root": merge_root,
        "robust_saddles": robust_saddles,
        "tree_saddles": tree_saddles,
    }


# ---------- plotting ----------
def plot_watershed(ax, R, color, title=""):
    H = R["H"]; z0 = R["z0"]; z1 = R["z1"]; final = R["final"]
    minima_ij = R["minima_ij"]; robust_ids = R["robust_ids"]
    H_mask = float(np.median(R["tree_saddles"])) if R["tree_saddles"] \
             else float(np.percentile(H, 25))
    extent = [z1.min(), z1.max(), z0.min(), z0.max()]

    overlay = np.zeros((*final.shape, 4))
    for k in robust_ids:
        rgba = list(to_rgba(color[k])); rgba[3] = 0.55
        overlay[final == k] = rgba
    ax.imshow(overlay, extent=extent, origin="lower",
              aspect="auto", interpolation="nearest", zorder=1)

    levels = np.linspace(H.min(), float(np.percentile(H, 60)), 8)
    Z1, Z0 = np.meshgrid(z1, z0)
    ax.contour(Z1, Z0, H, levels=levels, colors="k",
               linewidths=0.5, alpha=0.45, zorder=2)
    ax.contour(Z1, Z0, H, levels=[H_mask], colors="black",
               linewidths=1.2, alpha=0.85, linestyles="--", zorder=2)

    for r, k in enumerate(robust_ids):
        i, j = minima_ij[k - 1]
        ax.scatter([z1[j]], [z0[i]], s=200, marker="*",
                   edgecolor="black", linewidth=1.5,
                   facecolor=color[k], zorder=5)
        ax.annotate(f"{r + 1}", (z1[j], z0[i]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=12, fontweight="bold", color="black")
    ax.set_xlabel(r"$z_1$"); ax.set_ylabel(r"$z_0$")
    if title:
        ax.set_title(title)

    # zoom: bounding box of robust-basin minima
    star_ij = np.array([minima_ij[k - 1] for k in robust_ids])
    pad = 60
    i_lo = max(0, star_ij[:, 0].min() - pad)
    i_hi = min(H.shape[0] - 1, star_ij[:, 0].max() + pad)
    j_lo = max(0, star_ij[:, 1].min() - pad)
    j_hi = min(H.shape[1] - 1, star_ij[:, 1].max() + pad)
    ax.set_xlim(z1[j_lo], z1[j_hi])
    ax.set_ylim(z0[i_lo], z0[i_hi])


def plot_tree(ax, R, color, title=""):
    merge_root = R["merge_root"]; leaves = R["leaves"]; robust_ids = R["robust_ids"]

    def draw(node):
        if node.basin_id is not None:
            return
        xs = [c.x for c in node.children]
        ax.plot([min(xs), max(xs)], [node.H, node.H], "k-", lw=1.3, zorder=2)
        for c in node.children:
            col = color[c.basin_id] if c.basin_id is not None else "0.4"
            ax.plot([c.x, c.x], [c.H, node.H], "-", lw=2.2,
                    color=col, zorder=3)
            draw(c)
    draw(merge_root)

    for r, k in enumerate(robust_ids):
        leaf = leaves[k]
        ax.scatter([leaf.x], [leaf.H], s=200, marker="*",
                   edgecolor="black", linewidth=1.5,
                   facecolor=color[k], zorder=5)
        ax.annotate(f"{r + 1}", (leaf.x, leaf.H),
                    xytext=(0, -16), textcoords="offset points",
                    ha="center", fontsize=12, fontweight="bold")

    ax.set_xticks([])
    ax.set_xlabel("basins")
    ax.set_ylabel("Hamiltonian")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    if title:
        ax.set_title(title)


def make_color(K):
    palette = plt.cm.tab10.colors[:K]
    return palette


def single_figure(R, K, label, tag):
    color = {k: make_color(K)[r] for r, k in enumerate(R["robust_ids"])}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    plot_watershed(axes[0], R, color, title=f"Top-{K} robust basins on $H(z)$  ({label})")
    plot_tree(axes[1], R, color, title="Disconnectivity graph (Wales-style)")
    fig.tight_layout()
    out_pdf = OUT_FIG / f"01b_watershed_disconnectivity_{tag}_k{K}.pdf"
    out_png = OUT_FIG / f"01b_watershed_disconnectivity_{tag}_k{K}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_pdf}")


def full_F_vib(R, T):
    """F_vib at every detected local minimum (not just the K robust ones).

    Same harmonic-superposition formula as basin_curvatures, but evaluated
    over all 1851 minima.  Returns (F_vib_dict, F_vib_rel_dict, det_H_dict),
    where F_vib_rel has the minimum F_vib subtracted off so the widest basin
    has F_vib_rel = 0.
    """
    Hs = R["Hs"]
    z0 = R["z0"]; z1 = R["z1"]
    dz0 = float(z0[1] - z0[0]); dz1 = float(z1[1] - z1[0])

    H_00 = np.zeros_like(Hs); H_11 = np.zeros_like(Hs); H_01 = np.zeros_like(Hs)
    H_00[1:-1, :] = (Hs[2:, :] - 2 * Hs[1:-1, :] + Hs[:-2, :]) / (dz0 * dz0)
    H_11[:, 1:-1] = (Hs[:, 2:] - 2 * Hs[:, 1:-1] + Hs[:, :-2]) / (dz1 * dz1)
    H_01[1:-1, 1:-1] = ((Hs[2:, 2:] - Hs[2:, :-2]
                        - Hs[:-2, 2:] + Hs[:-2, :-2])
                       / (4.0 * dz0 * dz1))
    H_00 = gaussian_filter(H_00, sigma=2.0)
    H_11 = gaussian_filter(H_11, sigma=2.0)
    H_01 = gaussian_filter(H_01, sigma=2.0)
    det_H_field = H_00 * H_11 - H_01 * H_01

    det_H = {}; F_vib = {}
    DET_FLOOR = 1.0  # below this the Hessian is effectively degenerate; flag
    valid_keys = []
    for k, (i, j) in enumerate(R["minima_ij"], start=1):
        d = float(det_H_field[i, j])
        det_H[k] = d
        if d > DET_FLOOR:
            F_vib[k] = 0.5 * T * np.log(d)
            valid_keys.append(k)
        else:
            F_vib[k] = np.nan  # mark as invalid

    # reference: median F_vib over basins with a meaningful Hessian
    valid_F = np.array([F_vib[k] for k in valid_keys])
    F_vib_ref = float(np.median(valid_F))
    F_vib_rel = {k: (v - F_vib_ref if np.isfinite(v) else np.nan)
                 for k, v in F_vib.items()}
    return F_vib, F_vib_rel, det_H


def F_persistence(R, T):
    """Free-energy persistence for every basin: F_pers = H_pers - F_vib_rel.
    Essential class (H_pers = inf) stays inf.  Basins with NaN F_vib_rel
    (degenerate Hessian) are returned as NaN.
    """
    _, F_vib_rel, _ = full_F_vib(R, T)
    H_pers = R["persistence"]
    out = {}
    for k, p in H_pers.items():
        if not np.isfinite(p):
            out[k] = np.inf
        elif not np.isfinite(F_vib_rel[k]):
            out[k] = np.nan
        else:
            out[k] = p - F_vib_rel[k]
    return out


def basin_curvatures(R):
    """Hessian determinant at each robust basin minimum.

    Uses central differences on the smoothed H grid, with additional
    Gaussian smoothing of the second-derivative fields for stability.
    Returns dict {basin_id: det_Hessian}.
    """
    H = R["Hs"]
    z0 = R["z0"]; z1 = R["z1"]
    dz0 = float(z0[1] - z0[0])
    dz1 = float(z1[1] - z1[0])

    H_00 = np.zeros_like(H); H_11 = np.zeros_like(H); H_01 = np.zeros_like(H)
    H_00[1:-1, :] = (H[2:, :] - 2 * H[1:-1, :] + H[:-2, :]) / (dz0 * dz0)
    H_11[:, 1:-1] = (H[:, 2:] - 2 * H[:, 1:-1] + H[:, :-2]) / (dz1 * dz1)
    H_01[1:-1, 1:-1] = ((H[2:, 2:] - H[2:, :-2]
                        - H[:-2, 2:] + H[:-2, :-2])
                       / (4.0 * dz0 * dz1))

    H_00 = gaussian_filter(H_00, sigma=2.0)
    H_11 = gaussian_filter(H_11, sigma=2.0)
    H_01 = gaussian_filter(H_01, sigma=2.0)
    det_H = H_00 * H_11 - H_01 * H_01

    out = {}
    for k in R["robust_ids"]:
        i, j = R["minima_ij"][k - 1]
        det_local = float(det_H[i, j])
        out[k] = max(det_local, 1e-12)
    return out


def harmonic_F_vib(R, T):
    """Harmonic vibrational free energy correction at each basin.
    F_vib(k, T) = (T/2) log(det Hessian).  Drops the basin-independent
    -2T log(T) constant; only relative values are meaningful.
    """
    curvs = basin_curvatures(R)
    return {k: 0.5 * T * np.log(d) for k, d in curvs.items()}, curvs


def plot_free_energy_bars(ax, R, color, T, title=""):
    """Paired bars: potential-energy depth H_min vs free-energy depth F_min,
    both expressed as depth below the shallowest robust basin.
    """
    robust = R["robust_ids"]
    K = len(robust)
    F_vib, _ = harmonic_F_vib(R, T)
    H_mins = np.array([R["H_min"][k] for k in robust])
    F_mins = np.array([R["H_min"][k] + F_vib[k] for k in robust])

    # depth below the shallowest (most positive) value, separately in each
    # quantity so the bars are positive and comparable within the panel
    H_ref = float(H_mins.max())
    F_ref = float(F_mins.max())
    H_depth = H_ref - H_mins
    F_depth = F_ref - F_mins
    cols = [color[k] for k in robust]

    xs = np.arange(K)
    w = 0.38
    ax.bar(xs - w / 2, H_depth, w, color=cols, edgecolor="black",
           linewidth=1.0, alpha=0.65, label=r"$H_\mathrm{min}$ depth")
    ax.bar(xs + w / 2, F_depth, w, color=cols, edgecolor="black",
           linewidth=1.0, hatch="///", alpha=0.95,
           label=r"$F_\mathrm{min}$ depth ($T$ = " + f"{T:.0f})")
    for x, h, f in zip(xs, H_depth, F_depth):
        ax.text(x - w / 2, h + max(H_depth) * 0.02, f"{h:.0f}",
                ha="center", va="bottom", fontsize=9)
        ax.text(x + w / 2, f + max(F_depth) * 0.02, f"{f:.0f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r + 1}" for r in range(K)],
                       fontsize=11, fontweight="bold")
    ax.set_xlabel("basin rank")
    ax.set_ylabel("depth below shallowest")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_ylim(0, max(H_depth.max(), F_depth.max()) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title)


def plot_persistence_bars(ax, R, color, title=""):
    """Vertical bars of persistence (descending), one per robust basin.
    Essential basin (persistence=inf) is shown as the tallest bar with a label.
    """
    robust = R["robust_ids"]
    K = len(robust)
    # use the tallest tree saddle as a reference for the "essential" basin's
    # effective persistence so the bar has a finite height we can draw
    finite_pers = [R["persistence"][k] for k in robust
                   if np.isfinite(R["persistence"][k])]
    ref_top = float(max(R["tree_saddles"])) if R["tree_saddles"] \
              else float(np.percentile(R["H"], 50))
    H_min_global = float(R["Hs"].min())
    inf_height = ref_top - H_min_global  # effective persistence of essential
    heights = []
    labels = []
    cols = []
    for r, k in enumerate(robust):
        p = R["persistence"][k]
        if not np.isfinite(p):
            heights.append(inf_height); labels.append(r"$\infty$")
        else:
            heights.append(p); labels.append(f"{p:.0f}")
        cols.append(color[k])
    xs = np.arange(K)
    bars = ax.bar(xs, heights, color=cols, edgecolor="black", linewidth=1.0)
    for x, h, lab in zip(xs, heights, labels):
        ax.text(x, h * 1.02, lab, ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r+1}" for r in range(K)], fontsize=11,
                       fontweight="bold")
    ax.set_xlabel("basin rank")
    ax.set_ylabel("persistence  (saddle $-$ min)")
    ax.set_ylim(0, max(heights) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title)


def boltzmann_weights(R, T):
    """Fraction of Z owned by each robust basin at temperature T.
    Pixels NOT assigned to a robust basin are excluded (peripheral noise)."""
    H = R["H"]; final = R["final"]; robust = R["robust_ids"]
    H_ref = float(H.min())  # shift for numerical stability
    weights = {}
    Z_total = 0.0
    for k in robust:
        mask = final == k
        if not mask.any():
            weights[k] = 0.0; continue
        w = float(np.exp(-(H[mask] - H_ref) / T).sum())
        weights[k] = w
        Z_total += w
    if Z_total > 0:
        for k in weights:
            weights[k] /= Z_total
    return weights


def plot_boltzmann_bars(ax, R, color, T, title=""):
    robust = R["robust_ids"]
    K = len(robust)
    w = boltzmann_weights(R, T)
    xs = np.arange(K)
    heights = [w[k] for k in robust]
    cols = [color[k] for k in robust]
    ax.bar(xs, heights, color=cols, edgecolor="black", linewidth=1.0)
    for x, h in zip(xs, heights):
        ax.text(x, h + 0.01, f"{h * 100:.0f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r+1}" for r in range(K)], fontsize=11,
                       fontweight="bold")
    ax.set_xlabel("basin rank")
    ax.set_ylabel(f"Boltzmann weight  (T = {T:.0f})")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title)


def compare_figure(R_aa, R_3di, K, *, third_panel="persistence", T=1000.0):
    """third_panel: 'persistence' or 'boltzmann'."""
    color_aa = {k: make_color(K)[r] for r, k in enumerate(R_aa["robust_ids"])}
    color_3di = {k: make_color(K)[r] for r, k in enumerate(R_3di["robust_ids"])}
    fig, axes = plt.subplots(2, 3, figsize=(17, 10),
                             gridspec_kw={"hspace": 0.32, "wspace": 0.28,
                                          "width_ratios": [1.2, 1.0, 0.8]})
    plot_watershed(axes[0, 0], R_aa, color_aa,
                   title=f"AA  —  top-{K} basins on $H(z)$")
    plot_tree(axes[0, 1], R_aa, color_aa,
              title="AA  —  disconnectivity")
    plot_watershed(axes[1, 0], R_3di, color_3di,
                   title=f"3Di  —  top-{K} basins on $H(z)$")
    plot_tree(axes[1, 1], R_3di, color_3di,
              title="3Di  —  disconnectivity")

    if third_panel == "persistence":
        plot_persistence_bars(axes[0, 2], R_aa, color_aa,
                              title="AA  —  persistence")
        plot_persistence_bars(axes[1, 2], R_3di, color_3di,
                              title="3Di  —  persistence")
        suffix = "persistence"
    elif third_panel == "boltzmann":
        plot_boltzmann_bars(axes[0, 2], R_aa, color_aa, T,
                            title="AA  —  basin Boltzmann fraction")
        plot_boltzmann_bars(axes[1, 2], R_3di, color_3di, T,
                            title="3Di  —  basin Boltzmann fraction")
        suffix = "boltzmann"
    elif third_panel == "freeenergy":
        plot_free_energy_bars(axes[0, 2], R_aa, color_aa, T,
                              title="AA  —  $H_\\mathrm{min}$ vs $F_\\mathrm{min}$")
        plot_free_energy_bars(axes[1, 2], R_3di, color_3di, T,
                              title="3Di  —  $H_\\mathrm{min}$ vs $F_\\mathrm{min}$")
        suffix = "freeenergy"
    else:
        raise ValueError(third_panel)

    fig.suptitle(f"MDH:  AA (top) vs 3Di (bottom)  —  K={K}",
                 fontsize=14, fontweight="bold")
    out_pdf = OUT_FIG / f"01b_compare_aa_vs_3di_k{K}_{suffix}.pdf"
    out_png = OUT_FIG / f"01b_compare_aa_vs_3di_k{K}_{suffix}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_pdf}")


def plot_persistence_ladder(ax, R, T, label, accent_color):
    """Sorted persistence vs rank for all basins, H view and F view overlaid."""
    H_pers = R["persistence"]
    F_pers = F_persistence(R, T)
    h_vals = sorted([v for v in H_pers.values() if np.isfinite(v)],
                    reverse=True)
    f_vals = sorted([v for v in F_pers.values() if np.isfinite(v) and v > 0],
                    reverse=True)
    ranks_h = np.arange(1, len(h_vals) + 1)
    ranks_f = np.arange(1, len(f_vals) + 1)

    ax.plot(ranks_h, h_vals, "-", color="0.45", lw=2,
            label=f"{label}  H persistence  ({len(h_vals)} basins)")
    ax.plot(ranks_f, f_vals, "-", color=accent_color, lw=2,
            label=f"{label}  F persistence  ({len(f_vals)} basins, T={T:.0f})")

    # Annotate counts above two thresholds
    for tau, ls in [(800, ":"), (300, "--")]:
        ax.axhline(tau, ls=ls, color="black", alpha=0.35, lw=0.8)
        n_h = sum(1 for v in h_vals if v > tau)
        n_f = sum(1 for v in f_vals if v > tau)
        ax.text(1.05, tau * 1.05, f"$\\tau$={tau}:  H={n_h}, F={n_f}",
                fontsize=9, va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, max(len(h_vals), 200))
    ax.set_ylim(10, max(h_vals) * 1.2 if h_vals else 2000)
    ax.set_xlabel("basin rank")
    ax.set_ylabel("persistence")
    ax.set_title(f"{label}:  basins surviving H- vs F-persistence")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def persistence_ladder_figure(R_aa, R_3di, T=1000.0):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    plot_persistence_ladder(axes[0], R_aa, T, "AA", "C0")
    plot_persistence_ladder(axes[1], R_3di, T, "3Di", "C1")
    fig.suptitle("Free-energy persistence: how many basins are real?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_pdf = OUT_FIG / f"01c_persistence_ladder_T{int(T)}.pdf"
    out_png = OUT_FIG / f"01c_persistence_ladder_T{int(T)}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_pdf}")


def dump_json(R, K, tag):
    z0 = R["z0"]; z1 = R["z1"]; minima_ij = R["minima_ij"]
    data = {
        "tag": tag, "K": K,
        "robust_basins": [
            {
                "rank": r + 1,
                "basin_id": int(k),
                "z0": float(z0[minima_ij[k - 1][0]]),
                "z1": float(z1[minima_ij[k - 1][1]]),
                "H_min": float(R["H_min"][k]),
                "persistence": (float(R["persistence"][k])
                                if np.isfinite(R["persistence"][k]) else None),
            }
            for r, k in enumerate(R["robust_ids"])
        ],
        "tree_saddles": [float(s) for s in R["tree_saddles"]],
    }
    with open(OUT_DATA / f"01b_watershed_disconnectivity_{tag}_k{K}.json", "w") as fh:
        json.dump(data, fh, indent=2)


# ---------- main ----------
def main():
    print("loading AA MDH...")
    H_aa, z0a, z1a = load_aa_mdh()
    print(f"  H in [{H_aa.min():.1f}, {H_aa.max():.1f}]  shape {H_aa.shape}")
    print("loading 3Di MDH...")
    H_3di, z0b, z1b = load_3di_mdh()
    print(f"  H in [{H_3di.min():.1f}, {H_3di.max():.1f}]  shape {H_3di.shape}")

    # Free-energy persistence ladder uses the FULL set of basins, so build
    # it once at K=4 (the K doesn't affect the ladder — it only sets which
    # basins are colored as "robust" in the watershed/tree).
    R_aa_full = compute_landscape(H_aa, z0a, z1a, 4)
    R_3di_full = compute_landscape(H_3di, z0b, z1b, 4)
    persistence_ladder_figure(R_aa_full, R_3di_full, T=1000.0)

    for K in KS:
        print(f"\n=== K = {K} ===")
        R_aa = compute_landscape(H_aa, z0a, z1a, K)
        R_3di = compute_landscape(H_3di, z0b, z1b, K)

        print(f"AA  top-{K} persistences: " +
              ", ".join(f"{R_aa['persistence'][k]:.0f}"
                        for k in R_aa["robust_ids"]))
        print(f"3Di top-{K} persistences: " +
              ", ".join(f"{R_3di['persistence'][k]:.0f}"
                        for k in R_3di["robust_ids"]))

        single_figure(R_aa, K, "AA", "aa")
        single_figure(R_3di, K, "3Di", "3di")
        compare_figure(R_aa, R_3di, K, third_panel="persistence")
        compare_figure(R_aa, R_3di, K, third_panel="boltzmann", T=1000.0)
        compare_figure(R_aa, R_3di, K, third_panel="freeenergy", T=1000.0)

        # report free-energy diagnostics for the K = 4 figure
        if K == 4:
            for label, R in [("AA", R_aa), ("3Di", R_3di)]:
                F_vib, curvs = harmonic_F_vib(R, 1000.0)
                print(f"  {label} det(Hessian) per basin: " +
                      ", ".join(f"{curvs[k]:.2e}" for k in R['robust_ids']))
                print(f"  {label} F_vib (T=1000) per basin: " +
                      ", ".join(f"{F_vib[k]:.0f}" for k in R['robust_ids']))

        dump_json(R_aa, K, "aa")
        dump_json(R_3di, K, "3di")

    print("done.")


if __name__ == "__main__":
    main()
