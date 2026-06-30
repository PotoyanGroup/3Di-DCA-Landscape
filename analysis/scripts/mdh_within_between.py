"""Within-cluster vs between-cluster sequence identity & TM-score for MDH.

Reviewer concern #1: "the appropriate measure ... is not absolute inter-group
distance but relative separation, the ratio of between-group to within-group
distances." We need both within-group and between-group values for sequence
identity and TM-score.

Strategy:
  - Sample N AlphaFold-predicted MDH structures (UniProt-ID labelled).
  - All-vs-all TM-score with TMalign (parallel).
  - Pairwise sequence identity from clean_MDH_MSA.fasta (cheap).
  - Discover the bimodal structure of the cohort by clustering the
    TM-score matrix (spectral, K=2).  If known cluster IDs for the
    cohort are available, this clustering can be swapped for the
    supplied labels.
  - Report within(C1), within(C2), between(C1,C2) for both metrics.

The reviewer-style table is the deliverable.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
import re
import json
import random
import subprocess
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from sklearn.cluster import SpectralClustering

from _paths import MDH_PDB_DIR, MDH_MSA, OUT_FIG, OUT_DATA

# Path to the TMalign binary. Override with the TMALIGN environment variable;
# defaults to "TMalign" on PATH. TMalign must be installed separately
# (https://zhanggroup.org/TM-align/).
TMALIGN = os.environ.get("TMALIGN", "TMalign")
N_SAMPLE = 150                 # number of structures to sample
RANDOM_SEED = 11
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def list_pdbs():
    return sorted(MDH_PDB_DIR.glob("*.pdb"))


def uniprot_id(pdb_path):
    """A0A0A0AGK0_unrelaxed_rank_001_..._.pdb → A0A0A0AGK0"""
    return pdb_path.stem.split("_")[0]


def load_msa_seqs(path):
    """Return dict {uniprot_id: ungapped AA sequence}."""
    out = {}
    for rec in SeqIO.parse(str(path), "fasta"):
        out[rec.id.split("/")[0].split(" ")[0]] = str(rec.seq)
    return out


def aligned_pair_identity(s1, s2):
    """Identity computed on positions where both have an AA (no gaps)."""
    s1, s2 = s1.upper(), s2.upper()
    n_match = n_pos = 0
    for a, b in zip(s1, s2):
        if a == "-" or b == "-":
            continue
        n_pos += 1
        if a == b:
            n_match += 1
    return n_match / n_pos if n_pos else float("nan")


def run_tmalign(args):
    """Return (TM-score, sequence identity) parsed from TMalign output."""
    p1, p2 = args
    try:
        r = subprocess.run(
            [TMALIGN, str(p1), str(p2)],
            capture_output=True, text=True, timeout=120)
    except Exception as e:
        return (float("nan"), float("nan"))
    out = r.stdout
    # average of the two TM-scores (different length normalizations)
    tms = re.findall(r"TM-score=\s*([0-9.]+)", out)
    tm = 0.5 * (float(tms[0]) + float(tms[1])) if len(tms) >= 2 else float("nan")
    # Seq_ID=n_identical/n_aligned= 0.479
    sid_m = re.search(r"Seq_ID=n_identical/n_aligned=\s*([0-9.]+)", out)
    sid = float(sid_m.group(1)) if sid_m else float("nan")
    return (tm, sid)


def main():
    pdbs = list_pdbs()
    print(f"Found {len(pdbs)} PDB files")
    # restrict to those with parseable IDs that exist in the MSA
    msa = load_msa_seqs(MDH_MSA)
    print(f"MSA loaded: {len(msa)} sequences")
    valid = [p for p in pdbs if uniprot_id(p) in msa]
    print(f"PDBs with matching MSA entry: {len(valid)}")
    if len(valid) < N_SAMPLE:
        sample = valid
    else:
        sample = random.sample(valid, N_SAMPLE)
    sample.sort()
    ids = [uniprot_id(p) for p in sample]
    print(f"Sampling {len(sample)} structures for all-vs-all TMalign")

    # pairs
    pairs = []
    for i in range(len(sample)):
        for j in range(i+1, len(sample)):
            pairs.append((sample[i], sample[j]))
    print(f"#pairs = {len(pairs)}")

    # parallel TMalign — returns (TM, identity) tuples
    n_workers = min(8, mp.cpu_count())
    print(f"Running TMalign in {n_workers}-way parallel...")
    with mp.Pool(n_workers) as pool:
        results = pool.map(run_tmalign, pairs, chunksize=64)
    TM = np.eye(len(sample))
    ID = np.eye(len(sample))
    k = 0
    for i in range(len(sample)):
        for j in range(i+1, len(sample)):
            tm, sid = results[k]
            TM[i, j] = TM[j, i] = tm
            ID[i, j] = ID[j, i] = sid
            k += 1

    # discover bimodal cluster structure on TM-score
    # Spectral with 2 clusters using TM as affinity (already in [0,1])
    aff = TM.copy()
    np.fill_diagonal(aff, 1.0)
    aff = (aff + aff.T) / 2
    sc = SpectralClustering(n_clusters=2, affinity="precomputed",
                            random_state=RANDOM_SEED, assign_labels="kmeans")
    labels = sc.fit_predict(aff)
    n1 = int(np.sum(labels == 0))
    n2 = int(np.sum(labels == 1))
    print(f"Discovered clusters: C0 n={n1}, C1 n={n2}")

    # within / between summary
    def within_between(M, labels):
        m_w = []; m_b = []; vals_w = {0: [], 1: []}; vals_b = []
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                if labels[i] == labels[j]:
                    vals_w[labels[i]].append(M[i, j])
                else:
                    vals_b.append(M[i, j])
        return vals_w, vals_b

    tm_w, tm_b = within_between(TM, labels)
    id_w, id_b = within_between(ID, labels)

    table = pd.DataFrame({
        "metric": ["TM-score"] * 3 + ["seq identity"] * 3,
        "group": ["within C0", "within C1", "between"] * 2,
        "n_pairs": [len(tm_w[0]), len(tm_w[1]), len(tm_b),
                    len(id_w[0]), len(id_w[1]), len(id_b)],
        "mean": [np.mean(tm_w[0]), np.mean(tm_w[1]), np.mean(tm_b),
                 np.mean(id_w[0]), np.mean(id_w[1]), np.mean(id_b)],
        "std":  [np.std(tm_w[0]),  np.std(tm_w[1]),  np.std(tm_b),
                 np.std(id_w[0]),  np.std(id_w[1]),  np.std(id_b)],
    })
    print("\n=== Within-vs-Between table ===")
    print(table.to_string(index=False))

    # relative separation ratios (the reviewer's request)
    sep = {
        "TM-score   between/within(min)": float(np.mean(tm_b) /
                                                 max(min(np.mean(tm_w[0]),
                                                         np.mean(tm_w[1])), 1e-9)),
        "TM-score   |Δ between−within(avg)|": float(abs(
            np.mean(tm_b) - 0.5*(np.mean(tm_w[0]) + np.mean(tm_w[1])))),
        "seq-ID     between/within(min)": float(np.mean(id_b) /
                                                 max(min(np.mean(id_w[0]),
                                                         np.mean(id_w[1])), 1e-9)),
        "seq-ID     |Δ between−within(avg)|": float(abs(
            np.mean(id_b) - 0.5*(np.mean(id_w[0]) + np.mean(id_w[1])))),
    }
    print("\nRelative separation:")
    for k, v in sep.items():
        print(f"  {k}: {v:.4f}")

    # save
    out_csv = OUT_DATA / "05_within_between.csv"
    table.to_csv(out_csv, index=False)
    np.savez(OUT_DATA / "05_within_between.npz",
             ids=np.array(ids), labels=labels, TM=TM, ID=ID)
    with open(OUT_DATA / "05_within_between_separation.json", "w") as fh:
        json.dump(sep, fh, indent=2)
    print(f"Saved {out_csv}")

    # Plot: TM and ID matrices reordered by cluster
    order = np.argsort(labels)
    TM_o = TM[np.ix_(order, order)]
    ID_o = ID[np.ix_(order, order)]
    labs_o = labels[order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    sns.heatmap(TM_o, ax=axes[0], cmap="viridis", vmin=0.7, vmax=1.0,
                cbar_kws={"label": "TM-score"}, square=True,
                xticklabels=False, yticklabels=False)
    axes[0].set_title("TM-score matrix (rows reordered by spectral cluster)")
    # cluster boundary
    n0 = int(np.sum(labs_o == labs_o[0]))
    axes[0].axhline(n0, color="white"); axes[0].axvline(n0, color="white")

    sns.heatmap(ID_o, ax=axes[1], cmap="viridis", vmin=0.2, vmax=1.0,
                cbar_kws={"label": "sequence identity"}, square=True,
                xticklabels=False, yticklabels=False)
    axes[1].set_title("Pairwise sequence identity matrix")
    axes[1].axhline(n0, color="white"); axes[1].axvline(n0, color="white")
    fig.tight_layout()
    out_fig = OUT_FIG / "05_within_between.pdf"
    fig.savefig(out_fig, bbox_inches="tight", dpi=150)
    print(f"Saved {out_fig}")


if __name__ == "__main__":
    main()
