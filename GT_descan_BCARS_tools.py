import numpy as np
import copy
import h5py
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter, shift
from signal_alignment import phase_align
from scipy.ndimage.interpolation import shift
from statsmodels.tsa.stattools import ccovf
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from lazy5.inspect import get_datasets, get_attrs_dset
import os


def dset_finder_descan(DATA_FOLDER, filename, overwrite_attrs = True):
    HYPERSPECTRAL_IMAGE_EXIST = False
    NRB_LATE_EXIST = False
    DARK_EXIST = False
    ATTRS_EXIST = False
    OVERWRITE_ATTRS = overwrite_attrs

    f = h5py.File(os.path.join(DATA_FOLDER,filename), 'r')
    if '/raw_data/hyperspectral_image_0000' in f['raw_data'].keys():
        data = np.array(f['/raw_data/hyperspectral_image_0000'])
        attrs = get_attrs_dset(os.path.join(DATA_FOLDER,filename), '/raw_data/hyperspectral_image_0000') # make sure you get the correct attributes
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
    
    # Over-write the attributes
    if OVERWRITE_ATTRS:
        attrs['Calib.a_vec'] = np.array([-1.01604041e-01,  7.93978909e+02])  #np.array([-1.00147754e-01,  7.92263626e+02])
        attrs['Calib.probe'] = 763.5
        attrs['CalibOrig.probe'] = 763.5
        attrs['Calib.ctr_wl0'] = 670
        attrs['CalibOrig.ctr_wl'] = 670
        attrs['RasterScanParams.FastAxisStart'] = 0
        attrs['RasterScanParams.FastAxisStepSize'] = 0.26
        attrs['RasterScanParams.FastAxisSteps'] = 500
        attrs['RasterScanParams.FastAxisStop'] = 130
        attrs['RasterScanParams.FastAxisUnits'] = '$\\mu$m'
        attrs['Spectro.CenterWavelength'] = 670
    
    f.close()
        
    return data, nrb_late, dark, attrs, [HYPERSPECTRAL_IMAGE_EXIST, NRB_LATE_EXIST, DARK_EXIST, ATTRS_EXIST]
    

# For non-uniform galvo illumination
def intensity_correction(smoothed_raw, smoothed_nrb, smoothed_dark, OUTPUT_RATIO = True, SPEC_ALIGNING_PIX = 546):
    smoothed_dark_shifted = copy.copy(smoothed_dark)
    dark_total_avg = np.mean(smoothed_dark, axis=(0,1))  #average dark on every spatial pixel, getting a 2304 dimensional data
    nrb_dark_sub = smoothed_nrb - dark_total_avg              #nrb - averaged dark on the spectral dimension
    nrb_baseline_difference = np.mean(nrb_dark_sub[:,:,50:200])
    nrb_dark_sub -= nrb_baseline_difference
    bcars_dark_sub = smoothed_raw - dark_total_avg          #bcars - avg dark on the spectral dimension
    bcars_baseline_difference = np.mean(bcars_dark_sub[:,:,50:200])
    bcars_dark_sub -= bcars_baseline_difference
    nrb_profile = np.mean(nrb_dark_sub, axis = 0)
    nrb_profile_max = np.max(nrb_profile,axis = 1)

    # Reshape for broadcasting
    norm_factors = nrb_profile_max[np.newaxis, :, np.newaxis]/np.max(nrb_profile_max[:-1])  # shape: (1, 500, 1)
    # Apply normalization
    bcars_intcorrected = bcars_dark_sub / norm_factors + dark_total_avg    #add dark back
    nrb_intcorrected = nrb_dark_sub / norm_factors + dark_total_avg
    # Creat shifted arrays
    bcars_intcorrected_shifted = copy.deepcopy(bcars_intcorrected)
    nrb_intcorrected_shifted = copy.deepcopy(nrb_intcorrected)

    if OUTPUT_RATIO:
        ratio = np.ones(bcars_intcorrected.shape, dtype=np.float32)
        ratio_shifted = copy.deepcopy(ratio)
    else:
        ratio_shifted = None

    for y in np.arange(nrb_intcorrected.shape[1]):     
        s_phase = phase_align(nrb_profile[y,:], nrb_profile[int(nrb_intcorrected.shape[1]//2),:], [1800,1950])
        difference_pix = int(np.round(s_phase))
        bcars_intcorrected_shifted[:,y,:] = np.roll(bcars_intcorrected[:,y,:], difference_pix, axis=1)
        nrb_intcorrected_shifted[:,y,:] = np.roll(nrb_intcorrected[:,y,:], difference_pix, axis=1)
        smoothed_dark_shifted[:,y,:] = np.roll(smoothed_dark[:,y,:], difference_pix, axis=1)
        if OUTPUT_RATIO:
            bcars_baseline_difference_ratio = np.mean(bcars_dark_sub[:,y,50:250])
            nrb_baseline_difference_ratio = np.mean(nrb_dark_sub[:,y,50:250])
            ratio[:,y,:] = (bcars_dark_sub[:,y,:] - bcars_baseline_difference_ratio + 25) / (np.repeat(np.mean(nrb_dark_sub,axis=0)[np.newaxis, y, :], bcars_dark_sub.shape[0], axis=0) - nrb_baseline_difference_ratio + 25)
            ratio_shifted[:,y,:] = np.roll(ratio[:,y,:], difference_pix, axis=1)

    return bcars_intcorrected_shifted, nrb_intcorrected_shifted, ratio_shifted, smoothed_dark_shifted


# Function to compute the sum along axis=3
def compute_sum(data):
    return np.sum(data, axis=3)

# Function to apply a 3D median along wavenumber axis
def apply_median_filter(data):
    return median_filter(data, size=(1, 1, 3), mode='constant', cval=0.0)



def highres(y,kind='cubic',res=100):
    '''
    Interpolate data onto a higher resolution grid by a factor of *res*

    Args:
        y (1d array/list): signal to be interpolated
        kind (str): order of interpolation (see docs for scipy.interpolate.interp1d)
        res (int): factor to increase resolution of data via linear interpolation
    
    Returns:
        shift (float): offset between target and reference signal 
    '''
    y = np.array(y)
    x = np.arange(0, y.shape[0])
    f = interp1d(x, y,kind='cubic')
    xnew = np.linspace(0, x.shape[0]-1, x.shape[0]*res)
    ynew = f(xnew)
    return xnew,ynew



def phase_align(reference, target, roi, res=100):
    '''
    Cross-correlate data within region of interest at a precision of 1./res
    if data is cross-correlated at native resolution (i.e. res=1) this function
    can only achieve integer precision 

    Args:
        reference (1d array/list): signal that won't be shifted
        target (1d array/list): signal to be shifted to reference
        roi (tuple): region of interest to compute chi-squared
        res (int): factor to increase resolution of data via linear interpolation
    
    Returns:
        shift (float): offset between target and reference signal 
    '''
    # convert to int to avoid indexing issues
    ROI = slice(int(roi[0]), int(roi[1]), 1)

    # interpolate data onto a higher resolution grid 
    x,r1 = highres(reference[ROI],kind='linear',res=res)
    x,r2 = highres(target[ROI],kind='linear',res=res)

    # subtract mean
    r1 -= r1.mean()
    r2 -= r2.mean()

    # compute cross covariance 
    cc = ccovf(r1,r2,demean=False,adjusted=False)

    # determine if shift if positive/negative 
    if np.argmax(cc) == 0:
        cc = ccovf(r2,r1,demean=False,adjusted=False)
        mod = -1
    else:
        mod = 1

    # often found this method to be more accurate then the way below
    return np.argmax(cc)*mod*(1./res)

    # # interpolate data onto a higher resolution grid 
    # x,r1 = highres(reference[ROI],kind='linear',res=res)
    # x,r2 = highres(target[ROI],kind='linear',res=res)

    # # subtract off mean 
    # r1 -= r1.mean()
    # r1 -= r2.mean()

    # # compute the phase-only correlation function
    # product = np.fft.fft(r1) * np.fft.fft(r2).conj()
    # cc = np.fft.fftshift(np.fft.ifft(product))

    # # manipulate the output from np.fft
    # l = reference[ROI].shape[0]
    # shifts = np.linspace(-0.5*l,0.5*l,l*res)

    # # plt.plot(shifts,cc,'k-'); plt.show()
    # return shifts[np.argmax(cc.real)]