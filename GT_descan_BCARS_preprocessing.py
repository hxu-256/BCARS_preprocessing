import argparse
import numpy as np
import lazy5
import time
import os
from concurrent.futures import ThreadPoolExecutor
from GT_descan_BCARS_tools import intensity_correction, dset_finder_descan, compute_sum, apply_median_filter

parser = argparse.ArgumentParser(description='BCARS preprocessing pipeline')
parser.add_argument('input', help='Input folder containing raw HDF5 files')
parser.add_argument('output', help='Output folder for preprocessed HDF5 files')
parser.add_argument('--mode', choices=['ratio', 'raw', 'vst'], default='ratio',
                    help="Output mode (mutually exclusive): 'ratio' = BCARS/NRB intensity ratio, "
                         "'raw' = dark-subtracted/illumination-corrected spectra, "
                         "'vst' = variance-stabilized dispersive-like spectrum (I - A_nrb^2)/(2*A_nrb)")
parser.add_argument('--med_filter', type=int, default=1, help='Apply median filter (1=yes, 0=no)')
args = parser.parse_args()

DATA_FOLDER = args.input
SAVE_FOLDER = args.output
MODE = args.mode
APPLY_MED = bool(args.med_filter)

files = os.listdir(DATA_FOLDER)
data_list = [i for i in files if i.endswith('.h5')]
# DATA_FOLDER
# data_list

start = time.time()

for num, filename in enumerate(data_list):
    print('')
    print(f'Start processing {filename}')
    start2 = time.time()
    bcars, nrb, dark, attrs, parms = dset_finder_descan(DATA_FOLDER, filename, overwrite_attrs = True)

    if bcars.ndim == 4:
        data_dict = {"dark": dark, "bcars": bcars, "nrb": nrb}
        sum_results = {}

        with ThreadPoolExecutor() as executor:
            futures = {key: executor.submit(compute_sum, data) for key, data in data_dict.items()}
            for key, future in futures.items():
                sum_results[key] = future.result()
    else:
        sum_results = {"dark": dark, "bcars": bcars, "nrb": nrb}

    filtered_results = {}
    with ThreadPoolExecutor() as executor:
        futures = {key: executor.submit(apply_median_filter, data) for key, data in sum_results.items()}
        for key, future in futures.items():
            filtered_results[key] = future.result()

    dark_smoothed = filtered_results["dark"].astype(np.int32)[:,:-1,:]  # remove the raw "y=501"
    data_smoothed = filtered_results["bcars"].astype(np.int32)[:,:-1,:] # remove the raw "y=501"
    nrb_smoothed = filtered_results["nrb"].astype(np.int32)[:,:-1,:] # remove the raw "y=501"

    if APPLY_MED:
        prefix = 'medfilter'
        raw_kwargs = {}
    else:
        prefix = 'nofilter'
        dark_raw = sum_results["dark"].astype(np.int32)[:, :-1, :]
        data_raw = sum_results["bcars"].astype(np.int32)[:, :-1, :]
        nrb_raw  = sum_results["nrb"].astype(np.int32)[:, :-1, :]
        raw_kwargs = dict(raw_data=data_raw, raw_nrb=nrb_raw, raw_dark=dark_raw)

    outfile = f'preprocessed_{prefix}_{filename}'

    if MODE == 'ratio':
        _, _, ratio, out_dark, _, _ = intensity_correction(data_smoothed, nrb_smoothed, dark_smoothed, OUTPUT_RATIO=True, **raw_kwargs)
        nrb_for_ratio = np.ones((10, nrb_smoothed.shape[2]))
        lazy5.create.save(file=outfile, pth=SAVE_FOLDER, dset=f'preprocessed_images/{prefix}_ratio', data=np.array(ratio), mode='w')
        lazy5.create.save(file=outfile, pth=SAVE_FOLDER, dset=f'preprocessed_images/{prefix}_nrb_for_ratio', data=np.array(nrb_for_ratio, dtype=np.uint16), mode='a')
        lazy5.create.save(file=outfile, pth=SAVE_FOLDER, dset=f'preprocessed_images/{prefix}_dark', data=np.array(out_dark, dtype=np.uint16), mode='a')
        lazy5.alter.write_attr_dict(dset=f'preprocessed_images/{prefix}_ratio', attr_dict=attrs, fid=os.path.join(SAVE_FOLDER, outfile))
        lazy5.alter.write_attr_dict(dset=f'preprocessed_images/{prefix}_nrb_for_ratio', attr_dict=attrs, fid=os.path.join(SAVE_FOLDER, outfile))
        lazy5.alter.write_attr_dict(dset=f'preprocessed_images/{prefix}_dark', attr_dict=attrs, fid=os.path.join(SAVE_FOLDER, outfile))

    elif MODE == 'vst':
        _, _, _, out_dark, vst, vst_nrb_amp = intensity_correction(data_smoothed, nrb_smoothed, dark_smoothed, OUTPUT_VST=True, **raw_kwargs)
        nrb_ones = np.ones((10, nrb_smoothed.shape[2]))
        # VST output is signed/float -> save as float32 (never uint16)
        lazy5.create.save(file=outfile, pth=SAVE_FOLDER, dset=f'preprocessed_images/{prefix}_vst', data=np.array(vst, dtype=np.float32), mode='w')
        lazy5.create.save(file=outfile, pth=SAVE_FOLDER, dset=f'preprocessed_images/{prefix}_vst_nrb_amp', data=np.array(vst_nrb_amp, dtype=np.float32), mode='a')
        lazy5.create.save(file=outfile, pth=SAVE_FOLDER, dset=f'preprocessed_images/{prefix}_nrb_for_ratio', data=np.array(nrb_ones, dtype=np.uint16), mode='a')
        lazy5.create.save(file=outfile, pth=SAVE_FOLDER, dset=f'preprocessed_images/{prefix}_dark', data=np.array(out_dark, dtype=np.uint16), mode='a')
        lazy5.alter.write_attr_dict(dset=f'preprocessed_images/{prefix}_vst', attr_dict=attrs, fid=os.path.join(SAVE_FOLDER, outfile))
        lazy5.alter.write_attr_dict(dset=f'preprocessed_images/{prefix}_vst_nrb_amp', attr_dict=attrs, fid=os.path.join(SAVE_FOLDER, outfile))
        lazy5.alter.write_attr_dict(dset=f'preprocessed_images/{prefix}_nrb_for_ratio', attr_dict=attrs, fid=os.path.join(SAVE_FOLDER, outfile))
        lazy5.alter.write_attr_dict(dset=f'preprocessed_images/{prefix}_dark', attr_dict=attrs, fid=os.path.join(SAVE_FOLDER, outfile))

    else:  # raw
        data_out, nrb_out, _, out_dark, _, _ = intensity_correction(data_smoothed, nrb_smoothed, dark_smoothed, OUTPUT_RATIO=False, **raw_kwargs)
        lazy5.create.save(file=outfile, pth=SAVE_FOLDER, dset=f'preprocessed_images/{prefix}_raw', data=np.array(data_out, dtype=np.uint16), mode='w')
        lazy5.create.save(file=outfile, pth=SAVE_FOLDER, dset=f'preprocessed_images/{prefix}_nrb', data=np.array(nrb_out, dtype=np.uint16), mode='a')
        lazy5.create.save(file=outfile, pth=SAVE_FOLDER, dset=f'preprocessed_images/{prefix}_dark', data=np.array(out_dark, dtype=np.uint16), mode='a')
        lazy5.alter.write_attr_dict(dset=f'preprocessed_images/{prefix}_raw', attr_dict=attrs, fid=os.path.join(SAVE_FOLDER, outfile))
        lazy5.alter.write_attr_dict(dset=f'preprocessed_images/{prefix}_nrb', attr_dict=attrs, fid=os.path.join(SAVE_FOLDER, outfile))
        lazy5.alter.write_attr_dict(dset=f'preprocessed_images/{prefix}_dark', attr_dict=attrs, fid=os.path.join(SAVE_FOLDER, outfile))
    

    end2 = time.time()
    print(f'spent {round(((end2 - start2)/60),2)} minutes for processing {filename}')

end = time.time()
print('')
print(f'total spent {round(((end - start)/60),2)} minutes')