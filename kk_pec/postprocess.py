"""End-to-end BCARS post-processing: noisy/denoised cube in, Raman spectrum out.

Pipeline:
    cube --(optional SVD denoise)--> (KK | Hilbert phase retrieval)
         --(optional phase-error correction ALS)--> (optional scale-error correction SG)
         --> imag(...) == Raman-like spectrum

Copied from 2025_Celegans/svd-bench/scripts/postprocess.py and adapted for
standalone use in this pipeline (lazy-imports the scoring metrics so the
module loads without the svd-bench scripts/ package).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

# crikit lives at /home/hxu256/CRIKit2 (on sys.path via the crikit3 environment)
from crikit.cri.kk import KramersKronig
from crikit.cri.algorithms.kk import hilbertfft
from crikit.cri.error_correction import PhaseErrCorrectALS, ScaleErrCorrectSG

# metrics are only needed for score() / compare(); lazy import avoids a hard
# dependency on the svd-bench scripts package when running the pipeline.
try:
    from scripts.metrics import all_metrics as _all_metrics
except ImportError:
    _all_metrics = None


__all__ = [
    "PostprocessConfig",
    "PipelineResult",
    "retrieve_phase",
    "run_pipeline",
]


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class PostprocessConfig:
    """Every toggle/parameter of the post-processing chain."""

    # --- phase retrieval: "kk" (BCARS intensity input) or "hilbert" (dispersive chi input) ---
    phase_method: str = "hilbert"

    # KK params (crikit KramersKronig) — used when phase_method="kk"
    cars_amp_offset: float = 1.0
    nrb_amp_offset: float = 1.0
    phase_offset: float = 0.0
    conjugate: bool = False
    norm_to_nrb: bool = True
    pad_factor: int = 1
    n_edge: int = 30
    bad_value: float = 1e-8

    # Hilbert params — used when phase_method="hilbert"
    hilbert_pad_factor: int = 2
    hilbert_n_edge: int = 30

    # --- phase error correction (ALS, optional) ---
    phase_err_correct: bool = False
    pec_kwargs: dict = field(default_factory=lambda: dict(
        smoothness_param=1, asym_param=1e-5, redux=5, order=2,
        max_iter=100, min_diff=1e-6, verbose=False))

    # --- scale error correction (SG, optional) ---
    scale_err_correct: bool = False
    sec_kwargs: dict = field(default_factory=lambda: dict(win_size=601, order=2))


@dataclass
class PipelineResult:
    """Outputs of run_pipeline."""

    denoised: np.ndarray   # input cube (pre phase-retrieval)
    spectrum: np.ndarray   # complex result after retrieval + corrections
    raman: np.ndarray      # np.imag(spectrum) — the Raman-like spectrum


# --------------------------------------------------------------------------------------
# Pipeline stages
# --------------------------------------------------------------------------------------
def retrieve_phase(cube: np.ndarray, cfg: PostprocessConfig,
                   bg: Optional[np.ndarray] = None) -> np.ndarray:
    """Recover the complex spectrum via Kramers-Kronig or direct Hilbert transform.

    cfg.phase_method == "kk"      -> cube is BCARS intensity (bg = NRB required)
    cfg.phase_method == "hilbert" -> cube is already dispersive chi-real
    """
    method = cfg.phase_method.lower()
    if method == "kk":
        if bg is None:
            bg = np.ones_like(cube)
        kk = KramersKronig(
            cars_amp_offset=cfg.cars_amp_offset,
            nrb_amp_offset=cfg.nrb_amp_offset,
            conjugate=cfg.conjugate,
            phase_offset=cfg.phase_offset,
            norm_to_nrb=cfg.norm_to_nrb,
            pad_factor=cfg.pad_factor,
            n_edge=cfg.n_edge,
            axis=-1,
            bad_value=cfg.bad_value,
        )
        return kk.calculate(cube, bg)
    if method == "hilbert":
        # Add C = max(|x|) before Hilbert so the real part stays non-negative.
        # This prevents the phase from winding through 2π, which would cause PEC
        # to overcorrect.  Hilbert(constant) ≈ 0, so C doesn't change imag(result).
        C = float(np.abs(cube).max())
        cube_shifted = (cube + C).astype(np.float64)
        imag = hilbertfft(cube_shifted, pad_factor=cfg.hilbert_pad_factor,
                          n_edge=cfg.hilbert_n_edge, axis=-1)
        return cube_shifted + 1j * imag
    raise ValueError(f"phase_method must be 'kk' or 'hilbert', got {cfg.phase_method!r}")


def run_pipeline(cube: np.ndarray, cfg: PostprocessConfig,
                 bg: Optional[np.ndarray] = None) -> PipelineResult:
    """Run phase retrieval + optional PEC/SEC on a pre-denoised cube."""
    ph = retrieve_phase(cube, cfg, bg)
    if cfg.phase_err_correct:
        ph = PhaseErrCorrectALS(**cfg.pec_kwargs).calculate(ph)
    if cfg.scale_err_correct:
        ph = ScaleErrCorrectSG(**cfg.sec_kwargs).calculate(ph)
    return PipelineResult(denoised=cube, spectrum=ph, raman=np.imag(ph))


# --------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------
def _center_crop_square(arr: np.ndarray) -> np.ndarray:
    ny, nx = arr.shape[0], arr.shape[1]
    side = min(ny, nx)
    y0, x0 = (ny - side) // 2, (nx - side) // 2
    return arr[y0:y0 + side, x0:x0 + side, ...]


def load_mat_inputs(mat_path: Union[str, Path]) -> dict:
    """Load a .mat file produced by h5tomat_detrend.

    Returns dict with y_0_real (normalized cube), wn, norm_min, norm_max,
    and a to_phys() callable that maps normalized -> physical scale.
    """
    from scipy.io import loadmat

    res = loadmat(str(mat_path))
    norm_min = float(np.asarray(res["norm_min"]).flat[0])
    norm_max = float(np.asarray(res["norm_max"]).flat[0])

    def to_phys(cube: np.ndarray) -> np.ndarray:
        return cube * (norm_max - norm_min) + norm_min

    return {
        "y_0_real": np.asarray(res["y_0_real"], dtype=np.float32),
        "img_clean": np.asarray(res.get("img_clean",
                                         np.zeros_like(res["y_0_real"])), dtype=np.float32),
        "wn": np.asarray(res["wn"]).ravel() if "wn" in res else None,
        "norm_min": norm_min,
        "norm_max": norm_max,
        "to_phys": to_phys,
    }


# --------------------------------------------------------------------------------------
# Scoring (optional — only available when scripts.metrics is on the path)
# --------------------------------------------------------------------------------------
def affine_align(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Least-squares fit a*pred + b to ref and apply it."""
    p = np.asarray(pred, dtype=np.float64).ravel()
    r = np.asarray(ref,  dtype=np.float64).ravel()
    mask = np.isfinite(p) & np.isfinite(r)
    if mask.sum() < 2:
        return np.asarray(pred)
    A = np.stack([p[mask], np.ones(mask.sum())], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, r[mask], rcond=None)
    return (a * np.asarray(pred) + b).astype(pred.dtype, copy=False)


def score(pred: np.ndarray, ref: np.ndarray, align: bool = True) -> dict:
    """Affine-align pred to ref then return scalar metrics.

    Requires scripts.metrics (svd-bench) to be importable.
    """
    if _all_metrics is None:
        raise ImportError(
            "scripts.metrics not found. Add svd-bench/scripts to sys.path to use score().")
    if pred.shape != ref.shape:
        pred, ref = _center_crop_square(pred), _center_crop_square(ref)
    aligned = affine_align(pred, ref) if align else pred
    m = _all_metrics(aligned.astype(np.float32), ref.astype(np.float32))
    keys = ("psnr_overall", "psnr_per_band_mean", "ssim_overall",
            "sam_mean_rad", "mse", "pearson_mean")
    return {k: float(m[k]) for k in keys}
