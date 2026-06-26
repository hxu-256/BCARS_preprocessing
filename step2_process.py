"""Step 2: CCV-SVD denoising + phase retrieval + error correction (run in crikit3 env).

Usage:
    conda run -n crikit3 python step2_process.py [--config params.yaml]
    conda run -n crikit3 python step2_process.py --config params.yaml --file path/to/file.mat

Pipeline per .mat file:
  1. Load y_0_real (normalized dispersive VST cube) + wn, norm_min, norm_max
  2. De-normalize to physical scale
  3. CCV-SVD denoising with autotune (split-half correlation, symmetric reconstruction)
  4. Phase retrieval: Hilbert (for dispersive/VST) or KK (for BCARS intensity)
  5. Phase error correction (ALS) if enabled
  6. Scale error correction (SG)  if enabled
  7. Save raman as (nspec, ny, nx) float32 TIFF → <name>_raman.tif
     Write wn.txt (one wavenumber per line) to output_dir
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import tifffile
import yaml

# ── local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "ccvsvd"))
sys.path.insert(0, str(Path(__file__).parent / "kk_pec"))

from SVD_utils import compute_crosscov_svd
from criteria import select_by_corr, RECON
from postprocess import PostprocessConfig, retrieve_phase

from crikit.cri.error_correction import PhaseErrCorrectALS, ScaleErrCorrectSG


# ==============================================================================
# Processing helpers
# ==============================================================================
def run_ccv_svd(cube: np.ndarray, ccv_cfg: dict) -> np.ndarray:
    """CCV-SVD denoising with split-half correlation autotune."""
    tau        = float(ccv_cfg.get('tau', 0.5))
    recon_mode = ccv_cfg.get('recon', 'symmetric')
    smooth     = float(ccv_cfg.get('smooth', 0.0))

    cc = compute_crosscov_svd(cube)
    keep_idx, r = select_by_corr(cc, tau=tau)
    recon_fn    = RECON[recon_mode]
    denoised    = recon_fn(cc, cube, keep_idx, smooth=smooth)

    print(f"     CCV-SVD: tau={tau}, recon={recon_mode}, rank={len(keep_idx)}")
    return denoised


def build_postprocess_config(pp_cfg: dict) -> PostprocessConfig:
    """Build PostprocessConfig from the params.yaml postprocess section."""
    method = pp_cfg.get('phase_method', 'hilbert')

    kk_p  = pp_cfg.get('kk', {})
    hl_p  = pp_cfg.get('hilbert', {})
    pec_p = pp_cfg.get('pec', {})
    sec_p = pp_cfg.get('sec', {})

    pec_enabled = bool(pec_p.get('enabled', False))
    sec_enabled = bool(sec_p.get('enabled', False))

    pec_kwargs = {k: v for k, v in pec_p.items() if k != 'enabled'}
    sec_kwargs = {k: v for k, v in sec_p.items() if k != 'enabled'}

    return PostprocessConfig(
        phase_method=method,
        # KK
        cars_amp_offset=float(kk_p.get('cars_amp_offset', 1.0)),
        nrb_amp_offset=float(kk_p.get('nrb_amp_offset', 1.0)),
        phase_offset=float(kk_p.get('phase_offset', 0.0)),
        conjugate=bool(kk_p.get('conjugate', False)),
        norm_to_nrb=bool(kk_p.get('norm_to_nrb', True)),
        pad_factor=int(kk_p.get('pad_factor', 1)),
        n_edge=int(kk_p.get('n_edge', 30)),
        bad_value=float(kk_p.get('bad_value', 1e-8)),
        # Hilbert
        hilbert_pad_factor=int(hl_p.get('pad_factor', 2)),
        hilbert_n_edge=int(hl_p.get('n_edge', 30)),
        # PEC / SEC
        phase_err_correct=pec_enabled,
        pec_kwargs=pec_kwargs,
        scale_err_correct=sec_enabled,
        sec_kwargs=sec_kwargs,
    )


def process_mat(mat_path: Path, output_dir: Path, ccv_cfg: dict, pp_cfg: dict) -> Path:
    """Run the full step-2 chain on one .mat file. Returns output path."""
    print(f"\n  → {mat_path.name}")
    t0 = time.time()

    # Load
    m       = sio.loadmat(str(mat_path))
    cube_n  = np.asarray(m['y_0_real'], dtype=np.float32)   # normalized [0, 1]
    wn      = np.asarray(m['wn']).ravel()     if 'wn'       in m else None
    norm_min = float(np.asarray(m['norm_min']).flat[0]) if 'norm_min' in m else 0.0
    norm_max = float(np.asarray(m['norm_max']).flat[0]) if 'norm_max' in m else 1.0

    # De-normalize to physical (dispersive VST) scale
    cube_phys = cube_n * (norm_max - norm_min) + norm_min

    # CCV-SVD denoising
    denoised = run_ccv_svd(cube_phys, ccv_cfg)

    # Phase retrieval (Hilbert or KK)
    cfg      = build_postprocess_config(pp_cfg)
    spectrum = retrieve_phase(denoised, cfg)

    # Phase error correction (ALS)
    if cfg.phase_err_correct:
        spectrum = PhaseErrCorrectALS(**cfg.pec_kwargs).calculate(spectrum)
        print(f"     PEC applied")

    # Scale error correction (SG)
    if cfg.scale_err_correct:
        spectrum = ScaleErrCorrectSG(**cfg.sec_kwargs).calculate(spectrum)
        print(f"     SEC applied")

    raman = np.imag(spectrum)

    if bool(pp_cfg.get('clip_negative', False)):
        raman = np.clip(raman, 0, None)
        print(f"     negative values clipped to 0")

    # Save raman as (nspec, ny, nx) float32 TIFF — matches paper_report/fig format
    out_path = output_dir / mat_path.name.replace('.mat', '_raman.tif')
    tifffile.imwrite(str(out_path), raman.transpose(2, 0, 1).astype(np.float32))

    # Write wn.txt (one value per line; same for all files in a batch)
    if wn is not None:
        np.savetxt(str(output_dir / 'wn.txt'), wn)

    print(f"     saved → {out_path.name}  ({(time.time() - t0) / 60:.2f} min)")
    return out_path


# ==============================================================================
# Entry point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="BCARS pipeline Step 2 (crikit3 env)")
    parser.add_argument('--config', default='params.yaml',
                        help='Path to params.yaml (default: params.yaml)')
    parser.add_argument('--file', default=None,
                        help='Process a single .mat file instead of the whole mat_dir')
    args = parser.parse_args()

    with open(args.config) as fh:
        params = yaml.safe_load(fh)

    paths      = params['paths']
    mat_dir    = Path(paths['mat_dir'])
    output_dir = Path(paths['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    ccv_cfg = params.get('ccvsvd', {})
    pp_cfg  = params.get('postprocess', {})

    if args.file:
        mat_files = [Path(args.file)]
    else:
        mat_files = sorted(mat_dir.glob('*.mat'))

    if not mat_files:
        print(f"No .mat files found in {mat_dir}")
        return

    print("=" * 60)
    print(f"Step 2: CCV-SVD + {pp_cfg.get('phase_method','hilbert').upper()} + PEC")
    print(f"  {len(mat_files)} file(s) in {mat_dir}")
    print("=" * 60)

    t_total = time.time()
    for mat_path in mat_files:
        try:
            process_mat(mat_path, output_dir, ccv_cfg, pp_cfg)
        except Exception as e:
            print(f"  SKIP {mat_path.name}: {e}")

    print(f"\nStep 2 complete  ({(time.time() - t_total) / 60:.2f} min total)")


if __name__ == '__main__':
    main()
