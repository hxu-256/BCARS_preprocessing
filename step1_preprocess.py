"""Step 1: Preprocessing pipeline (run in image_proc conda env).

Usage:
    conda run -n image_proc python step1_preprocess.py [--config params.yaml]
    conda run -n image_proc python step1_preprocess.py --config params.yaml --skip_preprocess
    conda run -n image_proc python step1_preprocess.py --config params.yaml --skip_h5tomat

Two sub-steps:
  1a. raw .h5 → preprocessed .h5   (GT_descan_BCARS_tools + lazy5)
  1b. preprocessed .h5 → .mat      (h5tomat_detrend)
"""

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

# ── local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "preprocess"))

from GT_descan_BCARS_tools import (
    intensity_correction, dset_finder_descan, compute_sum, apply_median_filter,
)
from h5tomat_detrend import h5_to_mat


# ==============================================================================
# Step 1a: raw .h5 → preprocessed .h5
# ==============================================================================
def run_preprocessing(raw_dir, out_h5_dir, cfg):
    """Preprocess all raw .h5 files in raw_dir and write to out_h5_dir."""
    import lazy5

    mode      = cfg.get('mode', 'vst')
    apply_med = bool(cfg.get('med_filter', False))
    calib     = cfg.get('calibration', {})

    files = sorted(f for f in os.listdir(raw_dir) if f.endswith('.h5'))
    if not files:
        print(f"[step1a] No .h5 files found in {raw_dir}")
        return

    os.makedirs(out_h5_dir, exist_ok=True)
    print(f"[step1a] {len(files)} file(s) in {raw_dir}")

    for filename in files:
        print(f"\n  → {filename}")
        t0 = time.time()

        bcars, nrb, dark, attrs, _ = dset_finder_descan(
            raw_dir, filename, overwrite_attrs=True, calib_dict=calib)

        # Sum over axis 3 if 4D
        if bcars.ndim == 4:
            with ThreadPoolExecutor() as ex:
                fs = {k: ex.submit(compute_sum, v)
                      for k, v in dict(bcars=bcars, nrb=nrb, dark=dark).items()}
                bcars = fs['bcars'].result()
                nrb   = fs['nrb'].result()
                dark  = fs['dark'].result()

        # Apply median filter to smoothed copies; keep raw copies for nofilter mode
        with ThreadPoolExecutor() as ex:
            fs = {k: ex.submit(apply_median_filter, v)
                  for k, v in dict(bcars=bcars, nrb=nrb, dark=dark).items()}
            dark_med = fs['dark'].result().astype(np.int32)[:, :-1, :]
            data_med = fs['bcars'].result().astype(np.int32)[:, :-1, :]
            nrb_med  = fs['nrb'].result().astype(np.int32)[:, :-1, :]

        if apply_med:
            prefix     = 'medfilter'
            raw_kwargs = {}
        else:
            prefix     = 'nofilter'
            raw_kwargs = dict(
                raw_data=bcars.astype(np.int32)[:, :-1, :],
                raw_nrb=nrb.astype(np.int32)[:, :-1, :],
                raw_dark=dark.astype(np.int32)[:, :-1, :],
            )

        outfile = f'preprocessed_{prefix}_{filename}'

        if mode == 'vst':
            _, _, _, out_dark, vst, vst_nrb_amp = intensity_correction(
                data_med, nrb_med, dark_med, OUTPUT_VST=True, **raw_kwargs)
            nrb_ones = np.ones((10, nrb_med.shape[2]))
            lazy5.create.save(file=outfile, pth=out_h5_dir,
                              dset=f'preprocessed_images/{prefix}_vst',
                              data=np.array(vst, dtype=np.float32), mode='w')
            lazy5.create.save(file=outfile, pth=out_h5_dir,
                              dset=f'preprocessed_images/{prefix}_vst_nrb_amp',
                              data=np.array(vst_nrb_amp, dtype=np.float32), mode='a')
            lazy5.create.save(file=outfile, pth=out_h5_dir,
                              dset=f'preprocessed_images/{prefix}_nrb_for_ratio',
                              data=np.array(nrb_ones, dtype=np.uint16), mode='a')
            lazy5.create.save(file=outfile, pth=out_h5_dir,
                              dset=f'preprocessed_images/{prefix}_dark',
                              data=np.array(out_dark, dtype=np.uint16), mode='a')
            for dset in [f'preprocessed_images/{prefix}_vst',
                         f'preprocessed_images/{prefix}_vst_nrb_amp',
                         f'preprocessed_images/{prefix}_nrb_for_ratio',
                         f'preprocessed_images/{prefix}_dark']:
                lazy5.alter.write_attr_dict(dset=dset, attr_dict=attrs,
                                            fid=os.path.join(out_h5_dir, outfile))

        elif mode == 'ratio':
            _, _, ratio, out_dark, _, _ = intensity_correction(
                data_med, nrb_med, dark_med, OUTPUT_RATIO=True, **raw_kwargs)
            nrb_for_ratio = np.ones((10, nrb_med.shape[2]))
            lazy5.create.save(file=outfile, pth=out_h5_dir,
                              dset=f'preprocessed_images/{prefix}_ratio',
                              data=np.array(ratio), mode='w')
            lazy5.create.save(file=outfile, pth=out_h5_dir,
                              dset=f'preprocessed_images/{prefix}_nrb_for_ratio',
                              data=np.array(nrb_for_ratio, dtype=np.uint16), mode='a')
            lazy5.create.save(file=outfile, pth=out_h5_dir,
                              dset=f'preprocessed_images/{prefix}_dark',
                              data=np.array(out_dark, dtype=np.uint16), mode='a')
            for dset in [f'preprocessed_images/{prefix}_ratio',
                         f'preprocessed_images/{prefix}_nrb_for_ratio',
                         f'preprocessed_images/{prefix}_dark']:
                lazy5.alter.write_attr_dict(dset=dset, attr_dict=attrs,
                                            fid=os.path.join(out_h5_dir, outfile))

        else:  # raw
            data_out, nrb_out, _, out_dark, _, _ = intensity_correction(
                data_med, nrb_med, dark_med, OUTPUT_RATIO=False, **raw_kwargs)
            lazy5.create.save(file=outfile, pth=out_h5_dir,
                              dset=f'preprocessed_images/{prefix}_raw',
                              data=np.array(data_out, dtype=np.uint16), mode='w')
            lazy5.create.save(file=outfile, pth=out_h5_dir,
                              dset=f'preprocessed_images/{prefix}_nrb',
                              data=np.array(nrb_out, dtype=np.uint16), mode='a')
            lazy5.create.save(file=outfile, pth=out_h5_dir,
                              dset=f'preprocessed_images/{prefix}_dark',
                              data=np.array(out_dark, dtype=np.uint16), mode='a')
            for dset in [f'preprocessed_images/{prefix}_raw',
                         f'preprocessed_images/{prefix}_nrb',
                         f'preprocessed_images/{prefix}_dark']:
                lazy5.alter.write_attr_dict(dset=dset, attr_dict=attrs,
                                            fid=os.path.join(out_h5_dir, outfile))

        print(f"     done in {(time.time() - t0) / 60:.2f} min  →  {outfile}")


# ==============================================================================
# Step 1b: preprocessed .h5 → .mat
# ==============================================================================
def run_h5tomat(h5_dir, mat_dir, cfg):
    """Convert preprocessed .h5 files to .mat with detrend + normalization."""
    h5_files = sorted(glob.glob(os.path.join(h5_dir, '*.h5')))
    if not h5_files:
        print(f"[step1b] No .h5 files found in {h5_dir}")
        return

    os.makedirs(mat_dir, exist_ok=True)
    print(f"\n[step1b] {len(h5_files)} file(s) in {h5_dir}")

    t_total = time.time()
    for h5_path in h5_files:
        try:
            h5_to_mat(
                h5_path, mat_dir,
                detrend_mode=cfg.get('detrend_mode', 'whittaker'),
                smoothness=float(cfg.get('smoothness', 1e5)),
                norm_mode=cfg.get('norm_mode', 'robust_symmetric'),
                pct=float(cfg.get('pct', 0.05)),
                wn_min=float(cfg.get('wn_min', 400.0)),
                wn_max=float(cfg.get('wn_max', 1800.0)),
            )
        except Exception as e:
            print(f"  SKIP {os.path.basename(h5_path)}: {e}")

    print(f"\n[step1b] done in {(time.time() - t_total) / 60:.2f} min")


# ==============================================================================
# Entry point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="BCARS pipeline Step 1 (image_proc env)")
    parser.add_argument('--config', default='params.yaml',
                        help='Path to params.yaml (default: params.yaml)')
    parser.add_argument('--skip_preprocess', action='store_true',
                        help='Skip step 1a (raw h5 → preprocessed h5); use existing files')
    parser.add_argument('--skip_h5tomat', action='store_true',
                        help='Skip step 1b (preprocessed h5 → mat); use existing files')
    args = parser.parse_args()

    with open(args.config) as fh:
        params = yaml.safe_load(fh)

    paths   = params['paths']
    raw_dir = paths['raw_h5_dir']
    h5_dir  = paths['preprocessed_h5_dir']
    mat_dir = paths['mat_dir']

    if not args.skip_preprocess:
        print("=" * 60)
        print("Step 1a: Preprocessing  (raw .h5 → preprocessed .h5)")
        print("=" * 60)
        run_preprocessing(raw_dir, h5_dir, params['preprocessing'])

    if not args.skip_h5tomat:
        print("\n" + "=" * 60)
        print("Step 1b: H5 → MAT  (preprocessed .h5 → .mat)")
        print("=" * 60)
        run_h5tomat(h5_dir, mat_dir, params['h5tomat'])

    print("\nStep 1 complete.")


if __name__ == '__main__':
    main()
