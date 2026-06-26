"""h5tomat_detrend - convert preprocessed BCARS VST .h5 cubes to .mat, WITH per-spectrum detrend.

Same spirit as DDS2M/npz2mat.py and h5tomat.py (no ground truth for exp data -> img_clean = zeros;
schema img_clean / y_0_real / norm_min / norm_max / mask_10/20/30; center-crop to a square), plus a
detrend stage before normalization.

Why detrend: the BCARS VST is the *dispersive* Re(chi) (antisymmetric/bipolar lineshapes). On the
worm cubes ~96% of the value spread in 400-1800 cm^-1 is a low-frequency baseline from the NRB not
matching the BCARS LF baseline (amplified by the 1/(2*A_nrb) divide) - NOT Raman (see check_vst.ipynb).

NO median filter: a (1,1,3) median mixes adjacent bands and destroys the i.i.d. white-noise
assumption the denoiser relies on. The detrend instead subtracts a WIDE per-spectrum SYMMETRIC
baseline (Whittaker or running average); that only removes the lowest frequencies, so it leaves the
local noise correlation essentially untouched (variance-safe) while flattening the NRB-mismatch wander.

Because the signal is dispersive we use a SYMMETRIC smoother (NOT ALS/airPLS, which assume one-sided
positive peaks and would distort the negative lobes), and the [0,1] map is symmetric about zero
(0 -> 0.5) so the +/- lobes are treated identically. The robust clip still earns its keep: with no
median filter the cosmic spikes survive, and the percentile clip saturates those rare voxels (without
correlating the bulk) instead of letting them pin the scale.

Usage:

    python h5tomat_detrend.py <input_folder_or_file> <output_folder>
        [--detrend_mode {whittaker,running,none}] [--smoothness S]
        [--norm_mode {robust_symmetric,robust,minmax}] [--pct P] [--dset KEY]

Per-cube pipeline (each *.h5 in the input folder):
    1. read aligned VST cube  preprocessed_images/<...>_vst   -> y_0_real  (no GT -> img_clean = 0)
    2. spectral clip to [WN_MIN, WN_MAX] cm^-1 (contiguous slice on read)
    3. center-crop spatial dims to the largest U-Net-divisible (2**TT) square that fits
    4. detrend: subtract a per-spectrum SYMMETRIC baseline (Whittaker lambda=SMOOTHNESS, or running
       average window=SMOOTHNESS bins). Linear, wide -> variance-safe / ~i.i.d.-preserving.
    5. normalize to [0,1]; robust_symmetric: s = max(|p_lo|,|p_hi|), map [-s,s]->[0,1] so 0->0.5.
       saved as norm_min / norm_max (de-norm: phys = norm*(norm_max-norm_min)+norm_min recovers the
       DETRENDED VST).
    6. write .mat with schema: img_clean, y_0_real, wn, norm_min, norm_max, mask_10/20/30

Run in a Python env with numpy / h5py / scipy (e.g. the `image-proc` env).
"""

# ============================== EDIT ME: parameters ==============================
DSET_KEY      = "preprocessed_images/nofilter_vst"  # aligned VST cube inside each .h5 (auto-resolves *_vst)
WN_MIN        = 400.0          # fingerprint clip low edge  (cm^-1)
WN_MAX        = 1800.0         # fingerprint clip high edge (cm^-1)
TT            = 6              # U-Net depth -> spatial side divisible by 2**TT

DETREND_MODE  = "whittaker"    # "whittaker" | "running" | "none"
SMOOTHNESS    = 1e5            # whittaker: lambda (larger=smoother);  running: window in bins (~250 cm^-1 ≈ 121)

NORM_MODE     = "robust_symmetric"  # "robust_symmetric" | "robust" | "minmax"
PCT           = 0.05            # robust clip percentile (low=PCT, high=100-PCT)
CLIP_TO_01    = False          # clip normalized cube to [0,1]
COMPRESS      = True           # gzip the .mat
# ================================================================================

import argparse
import os
import glob
import time
import numpy as np
import h5py
import scipy.io
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.ndimage import uniform_filter1d


def build_wavenumber_axis(attrs):
    """Wavenumber axis (cm^-1) from the h5 dataset Calib.* attrs (same formula as h5tomat.py)."""
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


def whittaker_baseline(cube, lam, d=2, chunk=20000):
    """Per-spectrum symmetric Whittaker smoother baseline along the last axis.

    Solves (I + lam * D^T D) z = y for every spectrum (rows). Linear filter -> variance-safe.
    Factorizes once and solves in column-chunks to bound memory.
    """
    n = cube.shape[-1]
    D = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    A = (sp.eye(n) + lam * (D.T @ D)).tocsc()
    lu = splu(A)
    Y = cube.reshape(-1, n)                    # (m, n) view
    out = np.empty_like(Y)
    for i in range(0, Y.shape[0], chunk):
        out[i:i + chunk] = lu.solve(Y[i:i + chunk].T.astype(np.float64)).T
    return out.reshape(cube.shape)


def detrend_cube(cube, mode, smoothness):
    """Subtract a per-spectrum SYMMETRIC baseline. Returns the detrended cube."""
    if mode == "none":
        return cube
    if mode == "running":
        base = uniform_filter1d(cube, size=int(round(smoothness)), axis=-1, mode="nearest")
    elif mode == "whittaker":
        base = whittaker_baseline(cube, lam=float(smoothness))
    else:
        raise ValueError(f"unknown detrend_mode {mode!r}")
    return cube - base


def normalize(cube, mode, pct, clip):
    """Affine map to [0,1]. Returns (norm float32, norm_min, norm_max), where
    phys = norm*(norm_max-norm_min)+norm_min."""
    if mode == "minmax":
        lo, hi = float(cube.min()), float(cube.max())
    elif mode == "robust":
        plo, phi = np.percentile(cube, [pct, 100.0 - pct]); lo, hi = float(plo), float(phi)
    elif mode == "robust_symmetric":
        plo, phi = np.percentile(cube, [pct, 100.0 - pct])
        s = float(max(abs(plo), abs(phi))); lo, hi = -s, s   # 0 -> 0.5; +/- lobes symmetric (dispersive-safe)
    else:
        raise ValueError(f"unknown norm_mode {mode!r}")
    if hi <= lo:
        hi = lo + 1.0
    norm = (cube - lo) / (hi - lo)
    if clip:
        norm = np.clip(norm, 0.0, 1.0)
    return norm.astype(np.float32), lo, hi


def h5_to_mat(h5_path, out_dir, dset_key=DSET_KEY, wn_min=WN_MIN, wn_max=WN_MAX, tt=TT,
              detrend_mode=DETREND_MODE, smoothness=SMOOTHNESS,
              norm_mode=NORM_MODE, pct=PCT, clip=CLIP_TO_01, compress=COMPRESS):
    """Convert one preprocessed BCARS VST .h5 -> detrended+normalized .mat. Returns the output path."""
    with h5py.File(h5_path, "r") as h:
        key = resolve_dset_key(h, dset_key)
        d = h[key]
        ny, nx, nspec = d.shape

        wn_full = build_wavenumber_axis(d.attrs)
        fp = np.where((wn_full >= wn_min) & (wn_full <= wn_max))[0]
        if fp.size == 0:
            raise ValueError(f"no bands in [{wn_min}, {wn_max}] cm^-1 for {os.path.basename(h5_path)}")
        b0, b1 = int(fp[0]), int(fp[-1]) + 1
        wn = wn_full[b0:b1].astype(np.float64)

        s = min(ny, nx)
        s -= s % (2 ** tt)                                # largest U-Net-divisible square
        r0, c0 = ny // 2 - s // 2, nx // 2 - s // 2       # center-crop offsets
        raw = d[r0:r0 + s, c0:c0 + s, b0:b1].astype(np.float64)   # (s, s, nfp)

    # 4) per-spectrum symmetric detrend (no median filter -> noise stays i.i.d.)
    cube = detrend_cube(raw, detrend_mode, smoothness)

    # 5) normalize to [0,1]; no GT for exp data -> img_clean = zeros
    raw_norm, p_lo, p_hi = normalize(cube, norm_mode, pct, clip)
    gt_norm = np.zeros_like(raw_norm)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(h5_path))[0] + ".mat")
    scipy.io.savemat(out_path, {
        "img_clean": gt_norm,                            # no ground truth -> zeros
        "y_0_real":  raw_norm,                           # detrended + normalized noisy cube
        "wn":        wn,                                 # clipped wavenumber axis (cm^-1)
        "norm_min":  np.array([[p_lo]]),                 # phys(detrended) = norm*(norm_max-norm_min)+norm_min
        "norm_max":  np.array([[p_hi]]),
        "mask_10":   np.ones_like(raw_norm, dtype=np.float32),
        "mask_20":   np.ones_like(raw_norm, dtype=np.float32),
        "mask_30":   np.ones_like(raw_norm, dtype=np.float32),
    }, do_compression=compress)

    smooth_lbl = (f"lam={smoothness:.0e}" if detrend_mode == "whittaker"
                  else (f"win={int(round(smoothness))}" if detrend_mode == "running" else "-"))
    print(f"{os.path.basename(h5_path)}: ({ny},{nx},{nspec}) -> ({s},{s},{b1 - b0})  "
          f"wn=[{wn.min():.0f},{wn.max():.0f}]  detrend={detrend_mode}({smooth_lbl})  "
          f"norm={norm_mode}[{pct},{100-pct}]%=[{p_lo:.3f},{p_hi:.3f}]  "
          f"(raw det. range=[{cube.min():.1f},{cube.max():.1f}])  ->  "
          f"{out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Convert preprocessed BCARS VST .h5 cubes to .mat with "
                                            "per-spectrum symmetric detrend + robust-symmetric norm.")
    p.add_argument("input", help="Input folder of preprocessed *.h5 (or a single .h5 file)")
    p.add_argument("output", help="Output folder for .mat files")
    p.add_argument("--dset", default=DSET_KEY, help=f"dataset key inside each .h5 (default {DSET_KEY})")
    p.add_argument("--detrend_mode", default=DETREND_MODE, choices=["whittaker", "running", "none"],
                   help=f"per-spectrum baseline (default {DETREND_MODE})")
    p.add_argument("--smoothness", type=float, default=SMOOTHNESS,
                   help=f"whittaker lambda OR running window in bins (default {SMOOTHNESS:g})")
    p.add_argument("--norm_mode", default=NORM_MODE,
                   choices=["robust_symmetric", "robust", "minmax"],
                   help=f"normalization (default {NORM_MODE})")
    p.add_argument("--pct", type=float, default=PCT, help=f"robust clip percentile (default {PCT})")
    p.add_argument("--wn_min", type=float, default=WN_MIN)
    p.add_argument("--wn_max", type=float, default=WN_MAX)
    args = p.parse_args()

    if args.input.endswith(".h5"):
        h5_files = [args.input]
    else:
        h5_files = sorted(glob.glob(os.path.join(args.input, "*.h5")))
    print(f"found {len(h5_files)} .h5 file(s)\n")

    start = time.time()
    for f in h5_files:
        try:
            h5_to_mat(f, args.output, dset_key=args.dset, detrend_mode=args.detrend_mode,
                      smoothness=args.smoothness, norm_mode=args.norm_mode, pct=args.pct,
                      wn_min=args.wn_min, wn_max=args.wn_max)
        except Exception as e:
            print(f"{os.path.basename(f)}: SKIP ({e})")
    print(f"\ndone in {(time.time() - start) / 60:.2f} min")


if __name__ == "__main__":
    main()
