"""DCA bootstrap stability of landscape basins.

Reviewer concern #3: "Are the basins/wells in Fig 7 finite-sampling artifacts?"

We resample the MDH MSA with replacement B times, refit a mean-field DCA
model on each bootstrap, score the decoded grid sequences with the new
Hamiltonian, and ask: do the basins persist under bootstrap?

Approach:
  - Take all_grid.fasta (the pre-decoded grid of 250000 3Di sequences).
  - For B bootstraps of the input MSA, fit mfDCA, get (e_ij, h_i).
  - Compute H_b(z) on the same grid.
  - Stability score per pixel = fraction of bootstraps in which H is below
    the median Hamiltonian (i.e. pixel is in a "low-energy region").
  - Compare basin centroids: do top-K minima land in the same place?

NOTE: This script depends on the parent repo's `dca/` module and on the
all_grid.fasta file. The all_grid.fasta lives at MDH_DIR/all_grid.fasta
(~93 MB). Sequence-level Hamiltonian scoring runs in seconds with mfDCA.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter

# add parent repo to sys.path for the dca module
from dca.dca_class import dca   # noqa: E402

from _paths import MDH_DIR, MDH_MSA_ALIGNED, OUT_FIG, OUT_DATA   # noqa: E402

# DCA needs an aligned MSA with `-` gaps preserved; clean_MDH_MSA.fasta has
# gaps stripped per-sequence and is therefore not suitable. Use the aligned
# alternative in the same folder.
MDH_MSA = MDH_MSA_ALIGNED
GRID_FA = MDH_DIR / "all_grid.fasta"
B_BOOTSTRAPS = 5                # number of bootstrap resamples
RNG = np.random.default_rng(7)


def write_subset_fasta(in_fa, out_fa, idx):
    """Write a FASTA containing only sequences at the given (multiset) indices."""
    seqs = []
    with open(in_fa) as fh:
        cur_h = None; cur_s = []
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_h is not None:
                    seqs.append((cur_h, "".join(cur_s)))
                cur_h = line; cur_s = []
            else:
                cur_s.append(line)
        if cur_h is not None:
            seqs.append((cur_h, "".join(cur_s)))
    with open(out_fa, "w") as oh:
        for k, i in enumerate(idx):
            h, s = seqs[i]
            # rename to keep unique headers under bootstrap
            oh.write(f">b{k}_{h[1:]}\n{s}\n")


def main():
    if not GRID_FA.exists():
        print(f"all_grid.fasta not found at {GRID_FA}; cannot run bootstrap.")
        sys.exit(1)
    # how many sequences in the MSA?
    with open(MDH_MSA) as fh:
        n_seq = sum(1 for line in fh if line.startswith(">"))
    print(f"MSA size N = {n_seq}, will resample N draws each bootstrap")

    tmp_dir = OUT_DATA / "_bootstrap_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Cache reference DCA on full MSA
    print("Fitting reference DCA on full MSA...")
    ref = dca(str(MDH_MSA), stype="protein")
    ref.mean_field(pseudocount_weight=0.5, theta=0.2, beta=1)
    print("Scoring reference grid...")
    H_ref, _ = ref.compute_Hamiltonian(str(GRID_FA))
    H_ref = np.array(H_ref).reshape(500, 500)
    print(f"H_ref shape {H_ref.shape}, range [{H_ref.min():.1f}, {H_ref.max():.1f}]")

    # Bootstraps
    Hs = [H_ref]
    for b in range(B_BOOTSTRAPS):
        boot_fa = tmp_dir / f"boot_{b}.fasta"
        idx = RNG.integers(0, n_seq, size=n_seq)
        write_subset_fasta(MDH_MSA, boot_fa, idx)
        print(f"\nBootstrap {b+1}/{B_BOOTSTRAPS}  fasta written")
        try:
            m = dca(str(boot_fa), stype="protein")
            m.mean_field(pseudocount_weight=0.5, theta=0.2, beta=1)
            Hb, _ = m.compute_Hamiltonian(str(GRID_FA))
            Hb = np.array(Hb).reshape(500, 500)
            Hs.append(Hb)
            print(f"  H range [{Hb.min():.1f}, {Hb.max():.1f}]")
        except Exception as e:
            print(f"  ERROR: {e}")
        # cleanup the temp fasta
        try: boot_fa.unlink()
        except: pass

    Hs = np.array(Hs)   # (B+1, 500, 500)
    print(f"\nAll bootstraps done: {Hs.shape}")
    np.save(OUT_DATA / "06_bootstrap_landscapes.npy", Hs)

    # === stability metrics ===
    # 1) low-energy mask per bootstrap (below 25th percentile)
    masks = []
    for i in range(Hs.shape[0]):
        thr = np.percentile(Hs[i], 25)
        masks.append(Hs[i] < thr)
    masks = np.array(masks)
    stability = masks.mean(axis=0)  # fraction of bootstraps in which pixel is low-E

    # 2) per-pixel std of H across bootstraps, normalized to ref H range
    H_std_boot = Hs.std(axis=0)
    H_mean_boot = Hs.mean(axis=0)

    # 3) basin minimum location stability
    # For each bootstrap, find top-5 most-spread, deepest basins; compare to ref
    def top_basins(Hh, K=5, sep=60):
        from scipy.ndimage import gaussian_filter
        Hs = gaussian_filter(Hh, 4)
        locmin = (Hs == minimum_filter(Hs, 11))
        ii, jj = np.where(locmin)
        order = np.argsort(Hs[ii, jj])
        chosen = []
        for k in order:
            i0, j0 = int(ii[k]), int(jj[k])
            if all((i0 - ci)**2 + (j0 - cj)**2 > sep**2 for ci, cj in chosen):
                chosen.append((i0, j0))
            if len(chosen) == K:
                break
        return np.array(chosen)

    basins_ref = top_basins(H_ref)
    basin_displacements = []
    for b in range(Hs.shape[0]):
        bb = top_basins(Hs[b])
        # greedy match: for each ref basin, find nearest in bb
        d = []
        for r in basins_ref:
            if len(bb) == 0:
                d.append(np.nan); continue
            dists = np.hypot(bb[:, 0] - r[0], bb[:, 1] - r[1])
            d.append(np.min(dists))
        basin_displacements.append(d)
    basin_displacements = np.array(basin_displacements)  # (B+1, K)
    print(f"\nReference basins (i,j):\n{basins_ref}")
    print(f"\nMean basin displacement under bootstrap (pixels):")
    for k in range(basin_displacements.shape[1]):
        print(f"  basin {k}: mean {np.nanmean(basin_displacements[1:, k]):.1f}  "
              f"max {np.nanmax(basin_displacements[1:, k]):.1f}")

    # === Plot ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    extent = [-10, 10, -10, 10]
    im = axes[0, 0].imshow(H_ref, origin="lower", extent=extent, cmap="viridis",
                            aspect="auto")
    axes[0, 0].set_title("Reference DCA Hamiltonian $H_{ref}(z)$")
    axes[0, 0].set_xlabel("$z_1$"); axes[0, 0].set_ylabel("$z_0$")
    plt.colorbar(im, ax=axes[0, 0], shrink=0.85)

    im = axes[0, 1].imshow(H_mean_boot, origin="lower", extent=extent,
                            cmap="viridis", aspect="auto")
    axes[0, 1].set_title(f"Bootstrap mean $\\langle H \\rangle$ ({Hs.shape[0]-1} resamples + ref)")
    axes[0, 1].set_xlabel("$z_1$")
    plt.colorbar(im, ax=axes[0, 1], shrink=0.85)

    im = axes[1, 0].imshow(H_std_boot, origin="lower", extent=extent,
                            cmap="magma", aspect="auto")
    axes[1, 0].set_title("Bootstrap std $\\sigma_H(z)$  (lower = more reliable)")
    axes[1, 0].set_xlabel("$z_1$"); axes[1, 0].set_ylabel("$z_0$")
    plt.colorbar(im, ax=axes[1, 0], shrink=0.85)

    im = axes[1, 1].imshow(stability, origin="lower", extent=extent,
                            cmap="cividis", aspect="auto", vmin=0, vmax=1)
    axes[1, 1].set_title("Low-energy stability (fraction of bootstraps with $H<Q_{25}$)")
    axes[1, 1].set_xlabel("$z_1$")
    plt.colorbar(im, ax=axes[1, 1], shrink=0.85, label="stability")

    fig.tight_layout()
    out = OUT_FIG / "06_dca_bootstrap.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"\nSaved {out}")

    np.savez(OUT_DATA / "06_dca_bootstrap.npz",
             H_ref=H_ref, H_mean=H_mean_boot, H_std=H_std_boot,
             stability=stability,
             basins_ref=basins_ref,
             basin_displacements=basin_displacements)
    print("Done.")


if __name__ == "__main__":
    main()
