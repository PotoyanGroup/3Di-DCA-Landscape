"""Pullback Fisher (Fisher-Rao) metric on the VAE latent space.

The decoder p_θ(a | z) defines an exponential family on each grid pixel z.
The Fisher information

    g_ij(z) = E_{a ~ p(·|z)} [ ∂_i log p(a|z) · ∂_j log p(a|z) ]

gives a Riemannian metric on Z. Distances under g are local KL divergences
between decoded sequence distributions — they measure how *different* the
sequences generated at two latent points actually are.

For a categorical decoder p(a_l = k | z) = softmax(ψ_l(z))_k (independent
across positions l = 1..L), the Fisher information per position is the
multinomial Fisher:

    g_ij(z) = Σ_l Σ_k p_l(k|z) · ∂_i logit_l(k) · ∂_j logit_l(k)
            = Σ_l ( ∇_z μ_l )^T diag(p_l) ( ∇_z μ_l )
              − Σ_l ( p_l^T ∇_z μ_l )^T ( p_l^T ∇_z μ_l )

where μ_l(z) are the centered logits. We compute g via the decoder's
Jacobian. The volume element √det g(z) is the area magnification factor
of the latent-to-data map.

Predictions for the revision (information-geometry framing):
  - For the AA-MDH VAE, √det g should be LARGE in the corridor between
    the n=50 / n=40 cohorts → AA latent stretches the cluster separation.
  - For the 3Di-MDH VAE, √det g should be SMALL in the same region →
    3Di latent compresses statistically similar sequences.

THIS SCRIPT IS A SKELETON — it requires TensorFlow installed in the env
and the trained VAE SavedModel. The grid evaluation is set up so it runs
in batches of latent coordinates.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from _paths import MDH_DIR, OUT_FIG, OUT_DATA

# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--vae", default=str(MDH_DIR),
                help="Path to TF SavedModel dir containing saved_model.pb")
ap.add_argument("--latent_dim", type=int, default=2)
ap.add_argument("--alphabet", type=int, default=23,
                help="number of categories per position")
ap.add_argument("--length", type=int, default=332)
ap.add_argument("--grid_n", type=int, default=80,
                help="grid resolution per axis (default 80 = 6400 points)")
ap.add_argument("--z_lim", type=float, default=10.0)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--tag", default="aa_mdh",
                help="label for output file naming, e.g. aa_mdh or 3di_mdh")
args = ap.parse_args()

# ---------------------------------------------------------------
# TF import deferred so the rest of the suite still runs without TF
# ---------------------------------------------------------------
import os
try:
    import tensorflow as tf
except ImportError as e:
    sys.exit(f"TensorFlow not available in this env — install it before "
             f"running 08_pullback_metric.py. {e}")
print(f"TensorFlow {tf.__version__}")

# ---------------------------------------------------------------
# Load the saved decoder. The LGL-VAE checkpoints come in two flavours:
#   1) TF 2.x SavedModel (directory containing saved_model.pb + variables/)
#   2) New-format Keras zip (.keras file) — needs the VAE class registered.
# ---------------------------------------------------------------
def load_decoder(vae_path: str):
    """Return a callable z (batch, latent_dim) -> p (batch, q, L).

    Handles three formats:
      1. TF SavedModel directory (saved_model.pb + variables/), loadable
         by `tf.saved_model.load` — most of the 3Di checkpoints.
      2. SavedModel with attached Keras-2 optimizer state — fails the above
         with `_UserObject has no attribute 'add_slot'`; falls back to
         `tf_keras.models.load_model(compile=False)`.
      3. New-format `.keras` zip — load via `tf.keras.models.load_model`
         with the VAE class registered.
    """
    p = Path(vae_path)
    if p.is_dir() and (p / "saved_model.pb").exists():
        try:
            m = tf.saved_model.load(str(p))
            return m.decoder if hasattr(m, "decoder") else m
        except AttributeError as e:
            if "add_slot" not in str(e):
                raise
            # Fall back to tf_keras (legacy Keras 2)
            os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
            import tf_keras
            m = tf_keras.models.load_model(str(p), compile=False)
            return m.decoder if hasattr(m, "decoder") else m
    if str(p).endswith(".keras"):
        from _paths import REPO_ROOT
        sys.path.insert(0, str(REPO_ROOT))
        from model.model import VAE
        m = tf.keras.models.load_model(str(p), compile=False,
                                       custom_objects={"VAE": VAE})
        return m.decoder if hasattr(m, "decoder") else m
    raise ValueError(f"Don't know how to load: {p}")


decoder = load_decoder(args.vae)
print(f"Loaded decoder from {args.vae}")

# Probe output shape so we can auto-detect alphabet/length
_probe = decoder(tf.constant(np.zeros((1, args.latent_dim), dtype=np.float32)))
if _probe.shape.rank == 3:
    args.alphabet, args.length = int(_probe.shape[1]), int(_probe.shape[2])
else:
    flat = int(_probe.shape[1])
    if flat % args.alphabet != 0:
        # fall back to inspecting common alphabets
        for qq in (20, 21, 22, 23, 24):
            if flat % qq == 0:
                args.alphabet = qq
                break
    args.length = flat // args.alphabet
print(f"Auto-detected decoder output: q={args.alphabet}, L={args.length}")

# ---------------------------------------------------------------
# Build grid
# ---------------------------------------------------------------
z0_axis = np.linspace(-args.z_lim, args.z_lim, args.grid_n)
z1_axis = np.linspace(-args.z_lim, args.z_lim, args.grid_n)
Z0, Z1 = np.meshgrid(z0_axis, z1_axis, indexing="ij")
zgrid = np.stack([Z0.ravel(), Z1.ravel()], axis=1).astype(np.float32)
print(f"Grid: {len(zgrid)} latent coordinates")

# ---------------------------------------------------------------
# Per-batch Fisher metric via decoder Jacobian
# ---------------------------------------------------------------
def fisher_per_batch(z_np):
    """Pullback Fisher metric at each z. Returns g: (B, d_z, d_z).

    Uses forward-mode autodiff (JVP) — for d_z=2 we only need 2 JVPs per
    batch instead of (q*L) VJPs that batch_jacobian would do under
    reverse-mode, which is what makes this ~1000x faster than
    `tape.batch_jacobian`.

    Form:
        g_ij = Σ_l Σ_k (∂_i p_lk)(∂_j p_lk) / p_lk

    This is the categorical Fisher pulled back via ∂p/∂z. We
    differentiate p (not log p) so low-probability states still
    contribute properly; then divide by p with a small additive
    regulariser.
    """
    z = tf.constant(z_np, dtype=tf.float32)
    B = z.shape[0]
    # Stack the two columns of J = ∂p/∂z
    cols = []
    for d in range(args.latent_dim):
        tangent = np.zeros((B, args.latent_dim), dtype=np.float32)
        tangent[:, d] = 1.0
        with tf.autodiff.ForwardAccumulator(z, tf.constant(tangent)) as acc:
            out = decoder(z, training=False)
        jvp = acc.jvp(out)           # (B, q, L) or (B, q*L)
        cols.append(jvp)
    # stack columns → (B, q, L, d_z) or (B, q*L, d_z)
    J = tf.stack(cols, axis=-1)
    # Ensure (B, q, L, d_z)
    if J.shape.rank == 3:
        J = tf.reshape(J, [B, args.alphabet, args.length, args.latent_dim])
    # Also need p at z
    p = decoder(z, training=False)
    if p.shape.rank == 2:
        p = tf.reshape(p, [B, args.alphabet, args.length])
    inv_p = 1.0 / tf.reshape(p + 1e-8,
                              [B, args.alphabet, args.length, 1, 1])
    Ji = tf.expand_dims(J, -1)
    Jj = tf.expand_dims(J, -2)
    g = tf.reduce_sum(inv_p * (Ji * Jj), axis=[1, 2])
    return g.numpy()


import time, sys as _sys
t0 = time.time()
g_list = []
n_batches = (len(zgrid) + args.batch - 1) // args.batch
for i in range(0, len(zgrid), args.batch):
    z_b = zgrid[i:i+args.batch]
    g_b = fisher_per_batch(z_b)
    g_list.append(g_b)
    bi = i // args.batch + 1
    if bi % 5 == 0 or bi == 1:
        dt = time.time() - t0
        eta = dt * (n_batches - bi) / bi
        print(f"  batch {bi}/{n_batches}  elapsed {dt:.1f}s  ETA {eta:.1f}s",
              flush=True)
        _sys.stdout.flush()
g = np.concatenate(g_list, axis=0).reshape(args.grid_n, args.grid_n,
                                            args.latent_dim, args.latent_dim)
print(f"Fisher metric tensor shape: {g.shape}")

# ---------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------
# 1) Volume element √det g(z)  — area-magnification of latent → data map
det_g = np.linalg.det(g)
det_g = np.clip(det_g, 0, None)
vol = np.sqrt(det_g)

# 2) Largest singular value of √g(z)  — "stretch factor" in worst direction
eigs = np.linalg.eigvalsh(g)
lam_max = np.sqrt(np.clip(eigs[..., -1], 0, None))

# 3) Intrinsic-dimension proxy: ratio λ_max / λ_min averaged over grid
intrinsic_ratio = np.sqrt(eigs[..., -1] / np.clip(eigs[..., 0], 1e-12, None))

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
extent = [z1_axis.min(), z1_axis.max(), z0_axis.min(), z0_axis.max()]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

ax = axes[0]
im = ax.imshow(np.log10(vol + 1e-12), origin="lower", extent=extent,
                cmap="magma", aspect="auto")
ax.set_title(r"$\log_{10}\sqrt{\det g(z)}$ — local volume of decoded distribution")
ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_0$")
plt.colorbar(im, ax=ax, shrink=0.85)

ax = axes[1]
im = ax.imshow(np.log10(lam_max + 1e-12), origin="lower", extent=extent,
                cmap="viridis", aspect="auto")
ax.set_title(r"$\log_{10}\,\sqrt{\lambda_{\max}(g)}$ — worst-direction stretch")
ax.set_xlabel("$z_1$")
plt.colorbar(im, ax=ax, shrink=0.85)

ax = axes[2]
im = ax.imshow(np.log10(intrinsic_ratio + 1e-12), origin="lower", extent=extent,
                cmap="cividis", aspect="auto")
ax.set_title(r"$\log_{10}\sqrt{\lambda_{\max}/\lambda_{\min}}$ — anisotropy")
ax.set_xlabel("$z_1$")
plt.colorbar(im, ax=ax, shrink=0.85)

fig.suptitle(f"Pullback Fisher-Rao metric on VAE latent space — {args.tag}")
fig.tight_layout()
out_fig = OUT_FIG / f"08_pullback_metric_{args.tag}.pdf"
fig.savefig(out_fig, bbox_inches="tight", dpi=150)
print(f"Saved {out_fig}")

# ---------------------------------------------------------------
# Save the tensor and derived quantities
# ---------------------------------------------------------------
np.savez(OUT_DATA / f"08_pullback_metric_{args.tag}.npz",
         g=g, vol=vol, lam_max=lam_max, intrinsic_ratio=intrinsic_ratio,
         z0_axis=z0_axis, z1_axis=z1_axis)
print(f"Saved {OUT_DATA / f'08_pullback_metric_{args.tag}.npz'}")

# ---------------------------------------------------------------
# Quick summary stats
# ---------------------------------------------------------------
print("\nSummary statistics across the grid:")
print(f"  median √det g    = {np.median(vol):.3e}")
print(f"  90th-percentile  = {np.percentile(vol, 90):.3e}")
print(f"  ratio P90/median = {np.percentile(vol,90)/max(np.median(vol),1e-12):.2f}")
print(f"  mean anisotropy  = {np.mean(intrinsic_ratio):.2f}")
print("\nWhen the 3Di-VAE checkpoint arrives, re-run with --tag 3di_mdh "
      "--vae /path/to/3di/vae and compare the two volume maps side by side.")
