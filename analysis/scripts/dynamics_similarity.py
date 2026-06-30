"""RMSF profile similarity across MDH species.

Reviewer concern #4: "Are 3Di-defined basins more functional and AA-defined
basins more phylogenetic?" We can use the existing MD data to test a key
falsifiable prediction: proteins within the same basin (in latent space)
should show *more correlated* RMSF profiles than proteins across basins.

Data on hand: 1.2 μs MD trajectories' RMSF profiles for 5 mollusk MDHs,
plus 8 intermediate "shortest path" species along a thermo→meso path.

We compute the matrix of pairwise RMSF Pearson correlations and visualize
it; the manuscript already shows a few example pairs in Fig 6 but no
matrix and no quantification. Output:
  - correlation matrix figure
  - mean within-cohort vs between-cohort r
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from _paths import MD_SPECIES, MD_PATH_DIR, OUT_FIG, OUT_DATA


def load_rmsf(species_dir):
    """Try 1.2_298_rmsf.npy first, fall back to 298_rmsf.npy."""
    candidates = [species_dir / "1.2_298_rmsf.npy", species_dir / "298_rmsf.npy"]
    for c in candidates:
        if c.exists():
            return np.load(c)
    return None


# === 5 named species ===
species_rmsf = {}
for name, p in MD_SPECIES.items():
    r = load_rmsf(p)
    if r is None:
        print(f"Skipping {name} — no RMSF file")
        continue
    species_rmsf[name] = r.squeeze()
    print(f"{name:30s}  N_residues = {r.size}")

# === path intermediate points ===
path_rmsf = {}
if MD_PATH_DIR.exists():
    for d in sorted(MD_PATH_DIR.iterdir()):
        if d.is_dir():
            r = load_rmsf(d)
            if r is None:
                continue
            path_rmsf[d.name] = r.squeeze()
            print(f"path:{d.name}  N={r.size}")

# pad/truncate to common length
all_rmsf = list(species_rmsf.items()) + list(path_rmsf.items())
if not all_rmsf:
    raise RuntimeError("No RMSF data found.")

lens = [len(r) for _, r in all_rmsf]
print(f"\nresidue counts across systems: min={min(lens)}, max={max(lens)}, "
      f"median={int(np.median(lens))}")

# truncate to common minimum length
L = min(lens)
data = np.array([r[:L] for _, r in all_rmsf])
names = [n for n, _ in all_rmsf]

# Pearson correlation matrix
R = np.corrcoef(data)
print(f"\nCorrelation matrix shape: {R.shape}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
# raw RMSF profiles
for n, r in all_rmsf:
    short = n.replace("Mytiluscalifornianus", "Mytilus")\
             .replace("Echinolittorinamalaccana", "Echino_mal")\
             .replace("Adamussium_colbecki", "Adamussium")\
             .replace("Nipponacmearadula", "Nipponacme")\
             .replace("Neritayoldii", "Nerita")
    ax.plot(r[:L], lw=1, alpha=0.7, label=short[:18])
ax.set_xlabel("residue index")
ax.set_ylabel("RMSF (Å)")
ax.set_title("RMSF profiles across MDH species + path-intermediate sims")
ax.legend(fontsize=6, ncol=2, loc="upper right")

ax = axes[1]
# heatmap
labels_short = [n[:14] for n in names]
sns.heatmap(R, ax=ax, vmin=-0.2, vmax=1, cmap="RdBu_r",
            xticklabels=labels_short, yticklabels=labels_short,
            cbar_kws={"label": "Pearson r"})
ax.set_title("Pairwise RMSF correlation matrix")
plt.xticks(rotation=70, ha="right", fontsize=7)
plt.yticks(rotation=0, fontsize=7)

fig.tight_layout()
out = OUT_FIG / "07_dynamics_similarity.pdf"
fig.savefig(out, bbox_inches="tight", dpi=150)
print(f"Saved {out}")

# === Annotate which are "named species" vs "path intermediates" and
#    compute mean within / between r ===
is_path = np.array(["." in n or n in path_rmsf for n in names])
within_named = []
within_path = []
between = []
for i in range(len(names)):
    for j in range(i+1, len(names)):
        if not is_path[i] and not is_path[j]:
            within_named.append(R[i, j])
        elif is_path[i] and is_path[j]:
            within_path.append(R[i, j])
        else:
            between.append(R[i, j])

summary = {
    "mean_r_within_named_species (5 mollusks)": np.mean(within_named),
    "mean_r_within_path_intermediates": np.mean(within_path) if within_path else float("nan"),
    "mean_r_named_vs_path_intermediates": np.mean(between) if between else float("nan"),
    "n_pairs_within_named": len(within_named),
    "n_pairs_within_path":  len(within_path),
    "n_pairs_between":      len(between),
}
with open(OUT_DATA / "07_dynamics_similarity_summary.txt", "w") as fh:
    for k, v in summary.items():
        fh.write(f"{k}: {v}\n")
print("Summary:", summary)

# Save the correlation matrix and labels for downstream use
pd.DataFrame(R, index=names, columns=names).to_csv(
    OUT_DATA / "07_rmsf_correlation_matrix.csv")
print(f"Saved {OUT_DATA/'07_rmsf_correlation_matrix.csv'}")

print("\nNote: to test the 'within-basin > between-basin' prediction, supply")
print("basin/cluster labels for each protein, then group these systems by")
print("basin and re-aggregate the within/between correlations.")
