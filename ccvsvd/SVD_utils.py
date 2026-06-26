"""CCV-SVD pipeline utilities — trimmed to the functions used by this pipeline.

Retained from the original SVD_utils.py:
  CrossCovarianceSVDResult  dataclass
  cube_to_matrix / matrix_to_cube
  compute_crosscov_svd
  gaussian_smooth_spectral_axis

All plotting helpers, classic SVD functions, torch acceleration, and geminate
correlation utilities from the original have been omitted.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CrossCovarianceSVDResult:
    """Container for odd/even cross-covariance SVD workflow."""

    P: np.ndarray
    s: np.ndarray
    Q: np.ndarray
    A_even: np.ndarray
    A_odd: np.ndarray
    even_mean: np.ndarray
    odd_mean: np.ndarray
    even_idx: np.ndarray
    odd_idx: np.ndarray
    spatial_shape: tuple
    spectral_size: int


def cube_to_matrix(cube: np.ndarray):
    """Convert (ny, nx, nspec) cube to (ny*nx, nspec) matrix.

    Returns (matrix, (ny, nx), nspec).
    """
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube (ny, nx, nspec), got shape={cube.shape}")
    ny, nx, nspec = cube.shape
    return cube.reshape(ny * nx, nspec), (ny, nx), nspec


def matrix_to_cube(matrix: np.ndarray, spatial_shape: tuple, spectral_size: int) -> np.ndarray:
    """Convert (ny*nx, nspec) matrix back to (ny, nx, nspec) cube."""
    ny, nx = spatial_shape
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={matrix.shape}")
    if matrix.shape[0] != ny * nx:
        raise ValueError(f"Matrix rows {matrix.shape[0]} != ny*nx={ny*nx}")
    if matrix.shape[1] != spectral_size:
        raise ValueError(f"Matrix cols {matrix.shape[1]} != nspec={spectral_size}")
    return matrix.reshape(ny, nx, spectral_size)


def compute_crosscov_svd(
    cube: np.ndarray,
    center_columns: bool = True,
    normalize_cov: bool = True,
) -> CrossCovarianceSVDResult:
    """Compute SVD of the odd/even interleaved spectral cross-covariance.

    C = X_even^T @ X_odd  (normalized by N-1)

    The resulting factor matrices P, Q and score matrices A_even, A_odd are
    used by criteria.py to select signal components via split-half correlation
    and reconstruct the denoised cube.
    """
    matrix, spatial_shape, spectral_size = cube_to_matrix(cube)

    odd_idx_all  = np.arange(0, spectral_size, 2, dtype=int)
    even_idx_all = np.arange(1, spectral_size, 2, dtype=int)
    n_pair   = min(odd_idx_all.size, even_idx_all.size)
    odd_idx  = odd_idx_all[:n_pair]
    even_idx = even_idx_all[:n_pair]

    X_odd  = matrix[:, odd_idx]
    X_even = matrix[:, even_idx]

    if center_columns:
        odd_mean  = X_odd.mean(axis=0)
        even_mean = X_even.mean(axis=0)
        X_odd_c  = X_odd  - odd_mean
        X_even_c = X_even - even_mean
    else:
        odd_mean  = np.zeros(X_odd.shape[1],  dtype=X_odd.dtype)
        even_mean = np.zeros(X_even.shape[1], dtype=X_even.dtype)
        X_odd_c  = X_odd
        X_even_c = X_even

    C = X_even_c.T @ X_odd_c
    if normalize_cov:
        C = C / max(X_even_c.shape[0] - 1, 1)

    P, s, Qt = np.linalg.svd(C, full_matrices=False)
    Q = Qt.T
    A_even = X_even_c @ P
    A_odd  = X_odd_c  @ Q

    return CrossCovarianceSVDResult(
        P=P, s=s, Q=Q,
        A_even=A_even, A_odd=A_odd,
        even_mean=even_mean, odd_mean=odd_mean,
        even_idx=even_idx, odd_idx=odd_idx,
        spatial_shape=spatial_shape, spectral_size=spectral_size,
    )


def gaussian_smooth_spectral_axis(cube: np.ndarray, sigma_pixels: float) -> np.ndarray:
    """Gaussian smooth along spectral axis (axis=2) using NumPy only."""
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3:
        raise ValueError(f"Expected cube shape (ny, nx, nspec), got {cube.shape}")
    if sigma_pixels <= 0:
        return cube

    radius = max(1, int(np.ceil(3.0 * sigma_pixels)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_pixels) ** 2)
    kernel /= kernel.sum()

    pad = radius
    padded = np.pad(cube, ((0, 0), (0, 0), (pad, pad)), mode="reflect")
    out = np.empty_like(cube, dtype=float)
    for i in range(cube.shape[2]):
        seg = padded[:, :, i: i + kernel.size]
        out[:, :, i] = np.tensordot(seg, kernel, axes=([2], [0]))
    return out
