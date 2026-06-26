#!/usr/bin/env python
"""
criteria.py -- CCV threshold-selection rules + reconstruction helpers.

Operates on a CrossCovarianceSVDResult from SVD_utils.compute_crosscov_svd.

Rules
-----
1. split_half_corr / select_by_corr  (PRIMARY, scale-free)
     r_k = |corr(A_even[:,k], A_odd[:,k])| over pixels.  Signal components
     reproduce (r_k -> 1), noise does not (r_k -> 0).  Scale-free: tau
     transfers across cubes regardless of intensity.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import SVD_utils as su


# ------------------------------------------------------ reconstruction helpers --
def reconstruct_crosscov_idx(cc, cube, keep_idx, smooth=0.0):
    """Reconstruct using the original interleave mode (may show Nyquist comb)."""
    matrix, spatial_shape, spectral_size = su.cube_to_matrix(cube)
    keep_idx = np.asarray(keep_idx, dtype=int)
    if keep_idx.size == 0:
        X_even_hat = np.zeros((matrix.shape[0], cc.even_idx.size)) + cc.even_mean
        X_odd_hat  = np.zeros((matrix.shape[0], cc.odd_idx.size))  + cc.odd_mean
    else:
        Pk, Qk = cc.P[:, keep_idx], cc.Q[:, keep_idx]
        X_even_hat = (cc.A_even[:, keep_idx] @ Pk.T) + cc.even_mean
        X_odd_hat  = (cc.A_odd[:,  keep_idx] @ Qk.T) + cc.odd_mean
    recon = np.asarray(matrix, dtype=float).copy()
    recon[:, cc.even_idx] = X_even_hat
    recon[:, cc.odd_idx]  = X_odd_hat
    recon = su.matrix_to_cube(recon, spatial_shape, spectral_size)
    if smooth > 0:
        recon = su.gaussian_smooth_spectral_axis(recon, sigma_pixels=float(smooth))
    return recon


def _full_signatures(cc, keep):
    """Interleave the even/odd spectral loadings into one full-band signature matrix V,
    sign-aligned so both halves agree. Returns (V, sign, paired_mask)."""
    C  = cc.spectral_size
    Ae, Ao = cc.A_even[:, keep], cc.A_odd[:, keep]
    sign = np.sign(np.sum(Ae * Ao, axis=0)); sign[sign == 0] = 1.0
    V = np.zeros((C, keep.size), dtype=float)
    V[cc.even_idx] = cc.P[:, keep]
    V[cc.odd_idx]  = cc.Q[:, keep] * sign
    paired = np.zeros(C, dtype=bool)
    paired[cc.even_idx] = True; paired[cc.odd_idx] = True
    return V, sign, paired


def reconstruct_crosscov_symmetric(cc, cube, keep_idx, smooth=0.0):
    """Comb-free reconstruction: shared spatial coefficient + single grand mean.

    Gives even and odd bands the same coefficient (sign-aligned average of
    A_even/A_odd) so the two halves no longer disagree at their boundary.
    recon = a @ V^T + grand_mean
    """
    matrix, spatial_shape, C = su.cube_to_matrix(cube)
    keep = np.asarray(keep_idx, dtype=int)
    gm   = matrix.mean(0)
    recon = np.tile(gm, (matrix.shape[0], 1))
    if keep.size:
        V, sign, paired = _full_signatures(cc, keep)
        a     = (cc.A_even[:, keep] + sign * cc.A_odd[:, keep]) / 2.0
        recon = a @ V.T + gm
        recon[:, ~paired] = matrix[:, ~paired]
    recon = su.matrix_to_cube(recon, spatial_shape, C)
    if smooth > 0:
        recon = su.gaussian_smooth_spectral_axis(recon, sigma_pixels=float(smooth))
    return recon


def reconstruct_crosscov_subspace(cc, cube, keep_idx, smooth=0.0):
    """Comb-free variant: orthonormalize kept signatures into subspace B, project once.

    recon = (Xc @ B) @ B^T + grand_mean
    """
    matrix, spatial_shape, C = su.cube_to_matrix(cube)
    keep = np.asarray(keep_idx, dtype=int)
    gm   = matrix.mean(0)
    recon = np.tile(gm, (matrix.shape[0], 1))
    if keep.size:
        V, _, paired = _full_signatures(cc, keep)
        B, _ = np.linalg.qr(V)
        Xc    = matrix - gm
        recon = (Xc @ B) @ B.T + gm
        recon[:, ~paired] = matrix[:, ~paired]
    recon = su.matrix_to_cube(recon, spatial_shape, C)
    if smooth > 0:
        recon = su.gaussian_smooth_spectral_axis(recon, sigma_pixels=float(smooth))
    return recon


RECON = {
    "interleave": reconstruct_crosscov_idx,
    "symmetric":  reconstruct_crosscov_symmetric,
    "subspace":   reconstruct_crosscov_subspace,
}


# ----------------------------------------------- (1) split-half correlation ----
def split_half_corr(cc):
    """Per-component |Pearson corr| between the even and odd score maps."""
    Ae = np.asarray(cc.A_even, dtype=np.float64)
    Ao = np.asarray(cc.A_odd,  dtype=np.float64)
    Ae = Ae - Ae.mean(0, keepdims=True)
    Ao = Ao - Ao.mean(0, keepdims=True)
    num = np.sum(Ae * Ao, axis=0)
    den = np.sqrt(np.sum(Ae ** 2, axis=0) * np.sum(Ao ** 2, axis=0)) + 1e-15
    return np.abs(num / den)


def select_by_corr(cc, tau):
    """Return (keep_idx, r) where r[k] >= tau."""
    r = split_half_corr(cc)
    return np.where(r >= float(tau))[0], r


def corr_knee(r):
    """Knee of sorted-descending correlation curve (Kneedle-style).

    Returns (knee_value, knee_index_in_sorted).
    """
    y = np.sort(np.asarray(r, dtype=float))[::-1]
    n = y.size
    if n < 3:
        return float(y.min() if n else 0.0), 0
    x  = np.linspace(0.0, 1.0, n)
    yn = (y - y.min()) / (np.ptp(y) + 1e-15)
    dist = (1.0 - x) - yn
    i = int(np.argmax(dist))
    return float(y[i]), i

