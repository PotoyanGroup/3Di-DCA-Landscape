"""Common utilities: landscape loading, smoothing, basin detection."""
from pathlib import Path
import pickle

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as cc_label

from _paths import MDH_GRID_PKL, MDH_HAMIL


def load_landscape(grid_pkl: Path = MDH_GRID_PKL, hamil_npy: Path = MDH_HAMIL):
    """Return (H, z0_axis, z1_axis, raw_points).

    H[i,j] is the Hamiltonian at z0=z0_axis[i], z1=z1_axis[j].
    raw_points is the (250000,3) array (z0, z1, H) used to build H.
    """
    with open(grid_pkl, "rb") as f:
        pts = pickle.load(f)
    H = np.load(hamil_npy)
    z0_axis = np.unique(pts[:, 0])
    z1_axis = np.unique(pts[:, 1])
    # all_grid is laid out so first column iterates slowest; reshape verifies that
    if H.shape != (z0_axis.size, z1_axis.size):
        # rebuild explicitly
        H = pts[:, 2].reshape(z0_axis.size, z1_axis.size)
    return H, z0_axis, z1_axis, pts


def smooth(H, sigma):
    """Gaussian smooth on grid (sigma in pixels)."""
    return gaussian_filter(H, sigma=sigma)


def basin_label(H, smoothing_sigma=2.0):
    """Watershed-style basin labels via local-minimum flooding.

    Returns (labels, minima) where labels[i,j] is the basin index of pixel.
    """
    from scipy.ndimage import minimum_filter
    Hs = smooth(H, smoothing_sigma)
    # local minima
    local_min = (Hs == minimum_filter(Hs, size=5))
    labels, n = cc_label(local_min)
    # flood: assign each pixel the label of its closest minimum (by Hs ascent path)
    # cheap proxy: KD-tree nearest minimum
    ii, jj = np.where(local_min)
    if len(ii) == 0:
        return np.zeros_like(H, dtype=int), np.array([])
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([ii, jj]))
    yy, xx = np.indices(H.shape)
    coords = np.column_stack([yy.ravel(), xx.ravel()])
    _, idx = tree.query(coords)
    out = labels[ii[idx], jj[idx]].reshape(H.shape)
    return out, np.column_stack([ii, jj])


def axis_to_pix(z_axis, z_val):
    """Index of grid pixel closest to z_val along z_axis."""
    return int(np.argmin(np.abs(z_axis - z_val)))


def pix_to_coord(z0_axis, z1_axis, i, j):
    return (z0_axis[i], z1_axis[j])
