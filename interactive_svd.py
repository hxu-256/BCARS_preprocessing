"""Launch the CRIkit2 interactive SVD dialog on a (detrended) VST cube.

Shows each singular value's reconstructed 2D image + spectrum; you tick the SVs to keep
and click OK. The selected SV indices are printed (and saved to interactive_svd_selection.txt)
so you can plug them into the crikit_pipeline reconstruction.

Usage:
    conda activate image-proc
    python interactive_svd.py [MAT_PATH]

Default MAT_PATH is the Whittaker-detrended glycerol cube. Needs a display (WSLg/X) — same
environment in which the full crikit GUI pops up for you.
"""
import sys
sys.path.insert(0, "/home/hxu256/CRIkit2")
import numpy as np
import scipy.io as sio

from PyQt5.QtWidgets import QApplication
from crikit.preprocess.denoise import SVDDecompose
from crikit.ui.dialog_SVD import DialogSVD

DEFAULT_MAT = "../preprocessed_vst_nomedian/mat/preprocessed_nofilter_111825_male-phe(ctrl)_ABR14(WT)_02.mat"


def load_cube(mat_path):
    r = sio.loadmat(mat_path)
    lo, hi = float(r["norm_min"].flat[0]), float(r["norm_max"].flat[0])
    cube = (np.float32(r["y_0_real"]) * (hi - lo) + lo).astype(np.float32)  # -> physical detrended VST
    wn = np.asarray(r["wn"]).ravel()
    return cube, wn


def main():
    mat_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MAT
    cube, wn = load_cube(mat_path)
    ny, nx, nspec = cube.shape
    print(f"loaded {mat_path}")
    print(f"cube {cube.shape}  wn=[{wn[0]:.0f},{wn[-1]:.0f}]  phys=[{cube.min():.2f},{cube.max():.2f}]")

    # SVD factorization: data = (U, s, Vh), U:(npix,k) s:(k,) Vh:(k,nspec)
    U, s, Vh = SVDDecompose().calculate(cube)
    data = (U, s, Vh)
    print(f"{s.size} singular values. launching interactive dialog (tick SVs to KEEP, then OK)...")

    app = QApplication.instance() or QApplication(sys.argv)
    # img_all / spect_all = the mean image/spectrum of the full reconstruction (shown as reference)
    svs = DialogSVD.dialogSVD(
        data, cube.shape,
        img_all=cube.mean(axis=-1),
        spect_all=cube.reshape(-1, nspec).mean(axis=0),
    )

    if svs is None:
        print("\nno SVs selected (dialog cancelled).")
        return
    svs = np.asarray(svs)
    e = np.sum(s[svs] ** 2) / np.sum(s ** 2) * 100
    print(f"\nselected SVs: {list(svs)}")
    print(f"energy captured: {e:.2f}%")
    print(f"reconstruct with:  recon = (U[:, svs] @ (s[svs,None]*Vh[svs,:])).reshape(cube.shape)")
    with open("interactive_svd_selection.txt", "w") as f:
        f.write(",".join(map(str, svs.tolist())) + "\n")
    print("saved selection -> interactive_svd_selection.txt")


if __name__ == "__main__":
    main()
