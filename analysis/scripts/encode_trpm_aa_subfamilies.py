"""Encode TRPM3 and TRPM8 subfamily AA sequences through the TRPM AA-VAE
to obtain AA-latent positions per subfamily.

Mirrors 00d_encode_globin_aa_subfamilies.py but for TRPM. Used by the
cross-family Langevin comparison to add TRPM as a third regime-test
family alongside MDH (consistent) and Globin (complementary).

Pipeline:
  1. Parse zed_coordinates_with_classes.csv for TRPM to get UniProt
     accessions labeled 'member 3' (TRPM3) and 'member 8' (TRPM8).
  2. Parse trmp8_training_set.fasta into {accession: aligned seq},
     length L = 289.
  3. Filter to the two subfamilies.
  4. One-hot encode and pass through TRPM AA-VAE encoder.
  5. Compute centroids and save.
"""
from pathlib import Path
import sys, os, re
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import numpy as np
from Bio import SeqIO

from _paths import OUT_DATA
from _paths import TRPM_AA_DIR, TRPM_3DI_DIR  # noqa: E402

ZED_CSV = TRPM_3DI_DIR / "zed_coordinates_with_classes.csv"
AA_MSA = TRPM_AA_DIR / "trmp8_training_set.fasta"
AA_VAE_DIR = TRPM_AA_DIR

# Class strings as they appear in the TRPM zed file. Encoding all four
# so a second pair test (TRPM1 vs TRPM7) can be run without re-encoding.
SUBFAMS = {"TRPM3": "member 3", "TRPM8": "member 8",
           "TRPM1": "member 1", "TRPM7": "member 7"}
L_EXPECTED = 289

ACC_RX = re.compile(r"(?:^|[_|])([OPQ][0-9][A-Z0-9]{3}[0-9]|"
                    r"[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})")


def extract_accession(s: str):
    m = ACC_RX.search(str(s))
    return m.group(1) if m else None


def load_subfamily_accessions(csv_path):
    out = {label: set() for label in SUBFAMS}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            cls = r["class"].strip()
            for label, zed_val in SUBFAMS.items():
                if cls == zed_val:
                    acc = extract_accession(r["id"])
                    if acc:
                        out[label].add(acc)
    return out


def load_aa_msa_by_accession(fasta_path):
    out = {}
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        acc = extract_accession(rec.id)
        if acc:
            out[acc] = str(rec.seq).upper()
    return out


def main():
    import tensorflow as tf
    from model.generator import seq_code

    print(f"Reading zed coords from {ZED_CSV}")
    subfam_accs = load_subfamily_accessions(ZED_CSV)
    for s, accs in subfam_accs.items():
        print(f"  {s}: {len(accs)} accessions in zed file")

    print(f"\nReading AA MSA from {AA_MSA}")
    aa_by_acc = load_aa_msa_by_accession(AA_MSA)
    print(f"  total accessions in AA MSA: {len(aa_by_acc)}")

    seqs_per_subfam = {}
    for s in SUBFAMS:
        seqs = []
        for acc in subfam_accs[s]:
            if acc in aa_by_acc and len(aa_by_acc[acc]) == L_EXPECTED:
                seqs.append(aa_by_acc[acc])
        seqs_per_subfam[s] = seqs
        print(f"  {s}: {len(seqs)} sequences matched at L={L_EXPECTED}")

    if min(len(v) for v in seqs_per_subfam.values()) == 0:
        raise SystemExit("One subfamily has 0 matched AA sequences.")

    print(f"\nLoading TRPM AA-VAE from {AA_VAE_DIR}")
    try:
        m = tf.saved_model.load(str(AA_VAE_DIR))
    except (AttributeError, OSError):
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        import tf_keras
        m = tf_keras.models.load_model(str(AA_VAE_DIR), compile=False)
    enc = m.encoder

    q = 23
    out = {}
    for s, seqs in seqs_per_subfam.items():
        X = []
        for seq in seqs:
            ohe = np.zeros((q, L_EXPECTED), dtype=np.float32)
            ok = True
            for i, c in enumerate(seq):
                idx = seq_code.get(c)
                if idx is None:
                    ok = False; break
                if isinstance(idx, (range, list)):
                    v = list(idx)
                    for j in v:
                        ohe[j, i] = 1.0 / len(v)
                else:
                    ohe[idx, i] = 1.0
            if ok:
                X.append(ohe.flatten())
        X = np.asarray(X, dtype=np.float32)
        print(f"  {s}: encoding {len(X)} sequences ...")
        mu = enc(tf.constant(X))[0].numpy()
        centroid = mu.mean(axis=0)
        out[f"{s}_mu"] = mu.astype(np.float32)
        out[f"{s}_centroid"] = centroid.astype(np.float32)
        print(f"    centroid = ({centroid[0]:.2f}, {centroid[1]:.2f}),  "
              f"spread (std) = ({mu.std(axis=0)[0]:.2f}, "
              f"{mu.std(axis=0)[1]:.2f})")

    out_npz = OUT_DATA / "00e_trpm_aa_subfamily_mu.npz"
    np.savez(out_npz, **out)
    print(f"\nsaved {out_npz}")


if __name__ == "__main__":
    main()
