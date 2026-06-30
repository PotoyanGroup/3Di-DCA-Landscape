"""Cache properly-paired (μ_AA, μ_3Di) arrays per family to a local
npz so subsequent analyses don't need the full data tree mounted.

Run once with the data tree available; produces
`outputs/data/cache_paired_mus.npz` with keys:
    {family}_mu_aa   : (N, 2)
    {family}_mu_3di  : (N, 2)
    {family}_headers : list of paired protein IDs

Pairing strategy:
  - First try header intersection (set(AA_msa) ∩ set(3Di_msa)).
  - If that gives < 30 sequences, fall back to positional pairing
    (assumes the two FASTAs have matching protein order).
"""
from pathlib import Path
import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from Bio import SeqIO

from _paths import OUT_DATA
from _paths import MDH_AA_DIR, MDH_3DI_DIR, GLOBIN_AA_DIR, GLOBIN_3DI_DIR, TRPM_AA_DIR, TRPM_3DI_DIR, E1_DIR, E1_AA_DIR, E2_DIR, E2_AA_DIR  # noqa: E402

N_SAMPLE = 300
RNG = np.random.default_rng(42)

FAMILIES = [
    ("MDH",
        MDH_AA_DIR   / "remove_repetitive.fasta",
        MDH_3DI_DIR   / "MDH_3Di.fasta",
        MDH_AA_DIR,
        MDH_3DI_DIR),
    ("Globin",
        GLOBIN_AA_DIR / "DCA_coorelation_new_align.fasta",
        GLOBIN_3DI_DIR / "DCA_coorelation_3Di.fasta",
        GLOBIN_AA_DIR,
        GLOBIN_3DI_DIR),
    ("TRPM",
        TRPM_AA_DIR  / "trmp8_training_set.fasta",
        TRPM_3DI_DIR  / "trmp8_3di.fasta",
        TRPM_AA_DIR,
        TRPM_3DI_DIR),
    ("E1",
        E1_AA_DIR /
            "BVDV1_WSV_E1_reference_foldseek_aligned_20231128.fasta",
        E1_DIR / "E1_glycoprotein_3Di.fasta",
        E1_AA_DIR / "saved_model.keras",
        E1_DIR / "saved_model.keras"),
    ("E2",
        E2_AA_DIR /
            "BVDV1_TDAV_E2_reference_foldseek_aligned_20231128.fasta",
        E2_DIR / "E2_glycoprotein_3Di.fasta",
        E2_AA_DIR / "saved_model.keras",
        E2_DIR / "saved_model.keras"),
]


def load_dict(p):
    out = {}
    for rec in SeqIO.parse(str(p), "fasta"):
        out[rec.id] = str(rec.seq).upper()
    return out


def encode_through(model_path, seqs, L_expected, q=23):
    import tensorflow as tf
    from model.generator import seq_code
    p = Path(model_path)
    if p.is_dir() and (p / "saved_model.pb").exists():
        try:
            m = tf.saved_model.load(str(p))
        except AttributeError:
            os.environ["TF_USE_LEGACY_KERAS"] = "1"
            import tf_keras
            m = tf_keras.models.load_model(str(p), compile=False)
    elif str(p).endswith(".keras"):
        os.environ.pop("TF_USE_LEGACY_KERAS", None)
        from model.model import VAE
        m = tf.keras.models.load_model(str(p), compile=False,
                                        custom_objects={"VAE": VAE})
    enc = m.encoder
    X = []; kept = []
    for kk, s in enumerate(seqs):
        if len(s) != L_expected: continue
        ohe = np.zeros((q, L_expected), dtype=np.float32)
        ok = True
        for i, c in enumerate(s.upper()):
            idx = seq_code.get(c)
            if idx is None: ok=False; break
            if isinstance(idx, (range, list)):
                v = list(idx)
                for j in v: ohe[j, i] = 1.0/len(v)
            else:
                ohe[idx, i] = 1.0
        if not ok: continue
        X.append(ohe.flatten()); kept.append(kk)
    if not X: return np.zeros((0, 2)), []
    X = np.asarray(X, dtype=np.float32)
    mu = enc(tf.constant(X))[0].numpy()
    return mu, kept


cache = {}
for fam, aa_msa, di_msa, aa_vae, di_vae in FAMILIES:
    print(f"\n=== {fam} ===")
    try:
        aa_d = load_dict(aa_msa); di_d = load_dict(di_msa)
    except FileNotFoundError as e:
        print(f"  share unavailable: {e}")
        continue

    common = sorted(set(aa_d) & set(di_d))
    if len(common) < 30:
        hdrs_aa = list(aa_d.keys()); hdrs_di = list(di_d.keys())
        n_min = min(len(hdrs_aa), len(hdrs_di))
        common_h = [(hdrs_aa[i], hdrs_di[i]) for i in range(n_min)]
        aa_seqs = [aa_d[h] for h, _ in common_h]
        di_seqs = [di_d[h] for _, h in common_h]
        # Use the AA header as the canonical ID
        canon_headers = [h for h, _ in common_h]
        print(f"  positional pairing, n={n_min}")
    else:
        if len(common) > N_SAMPLE:
            idx = list(RNG.choice(len(common), size=N_SAMPLE, replace=False))
            common = [common[i] for i in sorted(idx)]
        aa_seqs = [aa_d[h] for h in common]
        di_seqs = [di_d[h] for h in common]
        canon_headers = common
        print(f"  header pairing, n={len(common)}")

    L = len(aa_seqs[0])
    mu_aa, kept_aa = encode_through(aa_vae, aa_seqs, L)
    mu_di, kept_di = encode_through(di_vae, di_seqs, L)
    kept = sorted(set(kept_aa) & set(kept_di))
    if not kept:
        print(f"  no surviving pairs"); continue
    idx_a = {k: i for i, k in enumerate(kept_aa)}
    idx_d = {k: i for i, k in enumerate(kept_di)}
    mu_aa = mu_aa[[idx_a[k] for k in kept]]
    mu_di = mu_di[[idx_d[k] for k in kept]]
    final_headers = [canon_headers[k] for k in kept]
    print(f"  cached {len(kept)} paired (μ_AA, μ_3Di)")
    cache[f"{fam}_mu_aa"] = mu_aa
    cache[f"{fam}_mu_3di"] = mu_di
    cache[f"{fam}_headers"] = np.array(final_headers)


out = OUT_DATA / "cache_paired_mus.npz"
np.savez(out, **cache)
print(f"\nSaved cache to {out}")
print("Contents:")
for k in cache:
    v = cache[k]
    if isinstance(v, np.ndarray):
        print(f"  {k}: shape {v.shape}, dtype {v.dtype}")
