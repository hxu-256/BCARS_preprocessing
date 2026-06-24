"""h5tomat — convert preprocessed BCARS VST .h5 cubes to .mat for DDS2M / S2DIP.

Mirrors h5tomat.ipynb but with ROBUST PERCENTILE normalization instead of raw min-max, so a
handful of strong/sparse voxels (e.g. polystyrene bead peaks) can no longer hijack the scale and
squash the real signal into a sliver of [0, 1]. See check_vst.ipynb section 7 for the diagnosis.

Usage (same spirit as GT_descan_BCARS_preprocessing.py):

    python h5tomat.py <input_folder_or_file> <output_folder> [--pct_low L] [--pct_high H] [--dset KEY]

Per-cube pipeline (each *.h5 in the input folder):
    1. read the aligned VST cube  preprocessed_images/<...>_vst   -> y_0_real  (no GT -> img_clean = 0)
    2. spectral clip to the fingerprint window [WN_MIN, WN_MAX] cm^-1 (contiguous, sliced on read)
    3. center-crop spatial dims to the largest U-Net-divisible (2**TT) square that fits
    4. normalize: robust percentile  norm = clip((x - p_lo) / (p_hi - p_lo), 0, 1)
       where p_lo/p_hi = percentile(cube, [PCT_LOW, PCT_HIGH]); saved as norm_min / norm_max
       (de-norm:  phys = norm * (norm_max - norm_min) + norm_min; values outside the percentile
        window are intentionally saturated by the clip)
    5. write .mat with schema: img_clean, y_0_real, wn, norm_min, norm_max, mask_10/20/30

Run in a Python env with numpy / h5py / scipy (e.g. the `image-proc` env).
"""

# ============================== EDIT ME: parameters ==============================
DSET_KEY   = "preprocessed_images/medfilter_vst"   # aligned VST cube inside each .h5
WN_MIN     = 400.0          # fingerprint clip low edge  (cm^-1) -- below this the NRB is absent
WN_MAX     = 1800.0         # fingerprint clip high edge (cm^-1)
TT         = 6              # U-Net depth -> spatial side must be divisible by 2**TT
PCT_LOW    = 0.5            # robust-norm lower percentile (was min)  -> norm_min
PCT_HIGH   = 99.5           # robust-norm upper percentile (was max)  -> norm_max
CLIP_TO_01 = True           # clip normalized cube to [0, 1] (saturate beyond the percentile window)
COMPRESS   = True           # gzip the .mat (smaller on disk)
# ================================================================================

import argparse
import os
import glob
import time
import numpy as np
import h5py
import scipy.io


def build_wavenumber_axis(attrs):
    """Wavenumber axis (cm^-1) from the h5 dataset Calib.* attrs (same formula as extract_nrb.py)."""
    coeffs = np.asarray(attrs["Calib.a_vec"])
    n_pix  = int(np.asarray(attrs["Calib.n_pix"]).flat[0])
    probe  = float(np.asarray(attrs["Calib.probe"]).flat[0]) * 1e-7
    converted_nm = np.polyval(coeffs, np.arange(n_pix)) * 1e-7
    return 1.0 / converted_nm - 1.0 / probe


def resolve_dset_key(h, requested):
    """Return `requested` if present, else the first preprocessed_images/*_vst dataset."""
    if requested in h:
        return requested
    grp = h.get("preprocessed_images")
    if grp is not None:
        for name in grp:
            if name.endswith("_vst"):
                return f"preprocessed_images/{name}"
    raise KeyError(f"{requested!r} not found and no *_vst dataset under preprocessed_images")


def h5_to_mat(h5_path, out_dir, dset_key, pct_low, pct_high,
              wn_min=WN_MIN, wn_max=WN_MAX, tt=TT, clip=CLIP_TO_01, compress=COMPRESS):
    """Convert one preprocessed BCARS VST .h5 -> .mat (DDS2M/S2DIP schema). Returns the output path."""
    with h5py.File(h5_path, "r") as h:
        key = resolve_dset_key(h, dset_key)
        d = h[key]
        ny, nx, nspec = d.shape

        # Spectral clip to [wn_min, wn_max]; wn axis is monotonic so kept bands are a contiguous slice.
        wn_full = build_wavenumber_axis(d.attrs)
        fp = np.where((wn_full >= wn_min) & (wn_full <= wn_max))[0]
        if fp.size == 0:
            raise ValueError(f"no bands in [{wn_min}, {wn_max}] cm^-1 for {os.path.basename(h5_path)}")
        b0, b1 = int(fp[0]), int(fp[-1]) + 1
        wn = wn_full[b0:b1].astype(np.float64)

        s = min(ny, nx)                                  # square side = smaller spatial dim
        s -= s % (2 ** tt)                               # snap down to largest U-Net-divisible side
        r0, c0 = ny // 2 - s // 2, nx // 2 - s // 2      # center-crop offsets
        raw = d[r0:r0 + s, c0:c0 + s, b0:b1].astype(np.float64)   # slice on read -> (s, s, nfp)

    # Robust percentile normalization of the cropped+clipped noisy cube; no GT -> img_clean = zeros.
    p_lo, p_hi = np.percentile(raw, [pct_low, pct_high])
    p_lo, p_hi = float(p_lo), float(p_hi)
    if p_hi <= p_lo:                                     # degenerate (flat cube) -> avoid /0
        p_hi = p_lo + 1.0
    raw_norm = (raw - p_lo) / (p_hi - p_lo)
    if clip:
        raw_norm = np.clip(raw_norm, 0.0, 1.0)
    raw_norm = raw_norm.astype(np.float32)
    gt_norm  = np.zeros_like(raw_norm)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(h5_path))[0] + ".mat")
    scipy.io.savemat(out_path, {
        "img_clean": gt_norm,                            # no ground truth -> zeros
        "y_0_real":  raw_norm,                           # robust-normalized noisy cube
        "wn":        wn,                                 # clipped wavenumber axis (cm^-1)
        "norm_min":  np.array([[p_lo]]),                 # phys = norm*(norm_max-norm_min)+norm_min
        "norm_max":  np.array([[p_hi]]),
        "mask_10":   np.ones_like(raw_norm, dtype=np.float32),
        "mask_20":   np.ones_like(raw_norm, dtype=np.float32),
        "mask_30":   np.ones_like(raw_norm, dtype=np.float32),
    }, do_compression=compress)

    print(f"{os.path.basename(h5_path)}: ({ny},{nx},{nspec}) -> ({s},{s},{b1 - b0})  "
          f"wn=[{wn.min():.0f},{wn.max():.0f}]cm^-1  "
          f"norm[{pct_low},{pct_high}]%=[{p_lo:.3f},{p_hi:.3f}]  "
          f"(raw min/max=[{raw.min():.1f},{raw.max():.1f}])  ->  "
          f"{out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert preprocessed BCARS VST .h5 cubes to .mat "
                                                 "with robust percentile normalization.")
    parser.add_argument("input", help="Input folder of preprocessed *.h5 (or a single .h5 file)")
    parser.add_argument("output", help="Output folder for .mat files")
    parser.add_argument("--dset", default=DSET_KEY, help=f"dataset key inside each .h5 (default {DSET_KEY})")
    parser.add_argument("--pct_low", type=float, default=PCT_LOW,
                        help=f"robust-norm lower percentile (default {PCT_LOW})")
    parser.add_argument("--pct_high", type=float, default=PCT_HIGH,
                        help=f"robust-norm upper percentile (default {PCT_HIGH})")
    parser.add_argument("--wn_min", type=float, default=WN_MIN, help=f"fingerprint low edge cm^-1 (default {WN_MIN})")
    parser.add_argument("--wn_max", type=float, default=WN_MAX, help=f"fingerprint high edge cm^-1 (default {WN_MAX})")
    args = parser.parse_args()

    if args.input.endswith(".h5"):
        h5_files = [args.input]
    else:
        h5_files = sorted(glob.glob(os.path.join(args.input, "*.h5")))
    print(f"found {len(h5_files)} .h5 file(s)\n")

    start = time.time()
    for f in h5_files:
        try:
            h5_to_mat(f, args.output, dset_key=args.dset,
                      pct_low=args.pct_low, pct_high=args.pct_high,
                      wn_min=args.wn_min, wn_max=args.wn_max)
        except Exception as e:
            print(f"{os.path.basename(f)}: SKIP ({e})")
    print(f"\ndone in {(time.time() - start) / 60:.2f} min")


if __name__ == "__main__":
    main()
