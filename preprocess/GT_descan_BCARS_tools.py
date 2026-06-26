import numpy as np
import copy
import h5py
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from statsmodels.tsa.stattools import ccovf
from lazy5.inspect import get_datasets, get_attrs_dset
import os


def dset_finder_descan(DATA_FOLDER, filename, overwrite_attrs=True, calib_dict=None):
    """Load BCARS, NRB, dark arrays and calibration attrs from a raw CRIKit HDF5.

    calib_dict : dict, optional
        Calibration values from params.yaml to override HDF5 attrs. Keys:
          a_vec            (list[float, float]) wavenumber calibration [slope, intercept]
          probe_nm         (float) probe laser wavelength in nm
          ctr_wl0_nm       (float) center wavelength in nm
          fast_axis_step_um (float) pixel size in µm
          fast_axis_steps   (int)  number of fast-axis steps
        Falls back to the 20251111 values when a key is absent.
    """
    HYPERSPECTRAL_IMAGE_EXIST = False
    NRB_LATE_EXIST = False
    DARK_EXIST = False
    ATTRS_EXIST = False

    f = h5py.File(os.path.join(DATA_FOLDER, filename), 'r')
    if '/raw_data/hyperspectral_image_0000' in f['raw_data'].keys():
        data = np.array(f['/raw_data/hyperspectral_image_0000'])
        attrs = get_attrs_dset(os.path.join(DATA_FOLDER, filename),
                               '/raw_data/hyperspectral_image_0000')
        HYPERSPECTRAL_IMAGE_EXIST = True
        ATTRS_EXIST = True

    if '/raw_data/nrb_late_image_post' or '/raw_data/nrb_late_image_pre' in f['raw_data'].keys():
        if '/raw_data/nrb_late_image_pre' in f['raw_data'].keys():
            nrb_late = np.array(f['/raw_data/nrb_late_image_pre'])
            NRB_LATE_EXIST = True
        elif '/raw_data/nrb_late_image_post' in f['raw_data'].keys():
            nrb_late = np.array(f['/raw_data/nrb_late_image_post'])
            NRB_LATE_EXIST = True
        else:
            nrb_late = None

    if '/raw_data/dark_image_post' or '/raw_data/dark_image_pre' in f['raw_data'].keys():
        if '/raw_data/dark_image_pre' in f['raw_data'].keys():
            dark = np.array(f['/raw_data/dark_image_pre'])
            DARK_EXIST = True
        elif '/raw_data/dark_image_post' in f['raw_data'].keys():
            dark = np.array(f['/raw_data/dark_image_post'])
            DARK_EXIST = True
        else:
            dark = None

    if overwrite_attrs:
        c = calib_dict or {}
        attrs['Calib.a_vec']    = np.array(c.get('a_vec', [-1.01604041e-01, 7.93978909e+02]))
        attrs['Calib.probe']    = c.get('probe_nm', 763.5)
        attrs['CalibOrig.probe'] = c.get('probe_nm', 763.5)
        attrs['Calib.ctr_wl0']  = c.get('ctr_wl0_nm', 670.0)
        attrs['CalibOrig.ctr_wl'] = c.get('ctr_wl0_nm', 670.0)
        attrs['RasterScanParams.FastAxisStart']    = 0
        attrs['RasterScanParams.FastAxisStepSize'] = c.get('fast_axis_step_um', 0.26)
        attrs['RasterScanParams.FastAxisSteps']    = c.get('fast_axis_steps', 500)
        attrs['RasterScanParams.FastAxisStop']     = 130
        attrs['RasterScanParams.FastAxisUnits']    = '$\\mu$m'
        attrs['Spectro.CenterWavelength']          = c.get('ctr_wl0_nm', 670.0)

    f.close()
    return data, nrb_late, dark, attrs, [HYPERSPECTRAL_IMAGE_EXIST, NRB_LATE_EXIST, DARK_EXIST, ATTRS_EXIST]


# For non-uniform galvo illumination
def intensity_correction(smoothed_raw, smoothed_nrb, smoothed_dark, OUTPUT_RATIO=True, OUTPUT_VST=False,
                         SPEC_ALIGNING_PIX=546, VST_EPS=1.0,
                         raw_data=None, raw_nrb=None, raw_dark=None):
    # VST and ratio are mutually exclusive; VST takes precedence.
    if OUTPUT_VST:
        OUTPUT_RATIO = False
    smoothed_dark_shifted = copy.copy(smoothed_dark)
    dark_total_avg = np.mean(smoothed_dark, axis=(0, 1))
    nrb_dark_sub = smoothed_nrb - dark_total_avg
    nrb_baseline_difference = np.mean(nrb_dark_sub[:, :, 50:200])
    nrb_dark_sub -= nrb_baseline_difference
    bcars_dark_sub = smoothed_raw - dark_total_avg
    bcars_baseline_difference = np.mean(bcars_dark_sub[:, :, 50:200])
    bcars_dark_sub -= bcars_baseline_difference
    nrb_profile = np.mean(nrb_dark_sub, axis=0)
    nrb_profile_max = np.max(nrb_profile, axis=1)

    norm_factors = nrb_profile_max[np.newaxis, :, np.newaxis] / np.max(nrb_profile_max[:-1])
    bcars_intcorrected = bcars_dark_sub / norm_factors + dark_total_avg
    nrb_intcorrected = nrb_dark_sub / norm_factors + dark_total_avg
    bcars_intcorrected_shifted = copy.deepcopy(bcars_intcorrected)
    nrb_intcorrected_shifted = copy.deepcopy(nrb_intcorrected)

    if OUTPUT_RATIO:
        ratio = np.ones(bcars_intcorrected.shape, dtype=np.float32)
        ratio_shifted = copy.deepcopy(ratio)
    else:
        ratio_shifted = None

    use_raw = raw_data is not None
    if use_raw:
        raw_bcars_dark_sub = raw_data.astype(np.float64) - dark_total_avg
        raw_bcars_dark_sub -= np.mean(raw_bcars_dark_sub[:, :, 50:200])
        raw_bcars_shifted = np.zeros_like(raw_bcars_dark_sub)
        raw_nrb_shifted   = np.zeros(raw_nrb.shape, dtype=np.float64)
        raw_dark_shifted  = np.zeros(raw_dark.shape, dtype=np.float64)

    if OUTPUT_VST:
        vst_src = raw_bcars_dark_sub if use_raw else bcars_dark_sub
        vst_shifted = np.zeros(vst_src.shape, dtype=np.float32)
        vst_nrb_amp = np.zeros((vst_src.shape[1], vst_src.shape[2]), dtype=np.float32)
    else:
        vst_shifted = None
        vst_nrb_amp = None

    for y in np.arange(nrb_intcorrected.shape[1]):
        s_phase = phase_align(nrb_profile[y, :],
                              nrb_profile[int(nrb_intcorrected.shape[1] // 2), :],
                              [1800, 1950])
        difference_pix = int(np.round(s_phase))
        if use_raw:
            raw_bcars_shifted[:, y, :] = np.roll(raw_bcars_dark_sub[:, y, :], difference_pix, axis=1)
            raw_nrb_shifted[:, y, :]   = np.roll(raw_nrb[:, y, :].astype(np.float64), difference_pix, axis=1)
            raw_dark_shifted[:, y, :]  = np.roll(raw_dark[:, y, :].astype(np.float64), difference_pix, axis=1)
            if OUTPUT_RATIO:
                raw_bcars_bl = np.mean(raw_bcars_dark_sub[:, y, 50:250])
                nrb_baseline_difference_ratio = np.mean(nrb_dark_sub[:, y, 50:250])
                ratio[:, y, :] = (raw_bcars_dark_sub[:, y, :] - raw_bcars_bl + 25) / \
                    (np.mean(nrb_dark_sub, axis=0)[y, :] - nrb_baseline_difference_ratio + 25)
                ratio_shifted[:, y, :] = np.roll(ratio[:, y, :], difference_pix, axis=1)
            elif OUTPUT_VST:
                nrb_line_rolled  = np.roll(nrb_profile[y, :], difference_pix)
                nrb_amp_rolled   = np.sqrt(np.clip(nrb_line_rolled, VST_EPS, None))
                bcars_line_rolled = np.roll(raw_bcars_dark_sub[:, y, :], difference_pix, axis=1)
                vst_shifted[:, y, :] = (bcars_line_rolled - nrb_line_rolled) / (2.0 * nrb_amp_rolled)
                vst_nrb_amp[y, :] = nrb_amp_rolled
        else:
            bcars_intcorrected_shifted[:, y, :] = np.roll(bcars_intcorrected[:, y, :], difference_pix, axis=1)
            nrb_intcorrected_shifted[:, y, :]   = np.roll(nrb_intcorrected[:, y, :], difference_pix, axis=1)
            smoothed_dark_shifted[:, y, :]      = np.roll(smoothed_dark[:, y, :], difference_pix, axis=1)
            if OUTPUT_RATIO:
                bcars_baseline_difference_ratio = np.mean(bcars_dark_sub[:, y, 50:250])
                nrb_baseline_difference_ratio   = np.mean(nrb_dark_sub[:, y, 50:250])
                ratio[:, y, :] = (bcars_dark_sub[:, y, :] - bcars_baseline_difference_ratio + 25) / \
                    (np.repeat(np.mean(nrb_dark_sub, axis=0)[np.newaxis, y, :],
                               bcars_dark_sub.shape[0], axis=0)
                     - nrb_baseline_difference_ratio + 25)
                ratio_shifted[:, y, :] = np.roll(ratio[:, y, :], difference_pix, axis=1)
            elif OUTPUT_VST:
                nrb_line_rolled   = np.roll(nrb_profile[y, :], difference_pix)
                nrb_amp_rolled    = np.sqrt(np.clip(nrb_line_rolled, VST_EPS, None))
                bcars_line_rolled = np.roll(bcars_dark_sub[:, y, :], difference_pix, axis=1)
                vst_shifted[:, y, :] = (bcars_line_rolled - nrb_line_rolled) / (2.0 * nrb_amp_rolled)
                vst_nrb_amp[y, :] = nrb_amp_rolled

    if use_raw:
        return raw_bcars_shifted, raw_nrb_shifted, ratio_shifted, raw_dark_shifted, vst_shifted, vst_nrb_amp
    return bcars_intcorrected_shifted, nrb_intcorrected_shifted, ratio_shifted, smoothed_dark_shifted, vst_shifted, vst_nrb_amp


def compute_sum(data):
    return np.sum(data, axis=3)


def apply_median_filter(data):
    return median_filter(data, size=(1, 1, 3), mode='constant', cval=0.0)


def highres(y, kind='cubic', res=100):
    y = np.array(y)
    x = np.arange(0, y.shape[0])
    f = interp1d(x, y, kind='cubic')
    xnew = np.linspace(0, x.shape[0] - 1, x.shape[0] * res)
    ynew = f(xnew)
    return xnew, ynew


def phase_align(reference, target, roi, res=100):
    ROI = slice(int(roi[0]), int(roi[1]), 1)
    x, r1 = highres(reference[ROI], kind='linear', res=res)
    x, r2 = highres(target[ROI], kind='linear', res=res)
    r1 -= r1.mean()
    r2 -= r2.mean()
    cc = ccovf(r1, r2, demean=False, adjusted=False)
    if np.argmax(cc) == 0:
        cc = ccovf(r2, r1, demean=False, adjusted=False)
        mod = -1
    else:
        mod = 1
    return np.argmax(cc) * mod * (1. / res)
