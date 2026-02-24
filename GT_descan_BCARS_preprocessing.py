import numpy as np
import lazy5
import time
import os
from concurrent.futures import ThreadPoolExecutor
from GT_descan_BCARS_tools import intensity_correction, dset_finder_descan, compute_sum, apply_median_filter

DATA_FOLDER = '20260210'
SAVE_FOLDER = '20260210'
SAVE_RATIO = True

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

    if SAVE_RATIO:
        _, _, ratio, smoothed_dark_shifted = intensity_correction(data_smoothed, nrb_smoothed, dark_smoothed, OUTPUT_RATIO = True)
        nrb_for_ratio = np.ones((10,nrb_smoothed.shape[2]))
        lazy5.create.save(file=f'preprocessed_medfilter_{filename}', pth=SAVE_FOLDER, dset='preprocessed_images/medfilter_ratio', data = np.array(ratio), mode='w')
        lazy5.create.save(file=f'preprocessed_medfilter_{filename}', pth=SAVE_FOLDER, dset='preprocessed_images/medfilter_nrb_for_ratio', data = np.array(nrb_for_ratio, dtype=np.uint16), mode='a')
        lazy5.create.save(file=f'preprocessed_medfilter_{filename}', pth=SAVE_FOLDER, dset='preprocessed_images/medfilter_dark', data = np.array(smoothed_dark_shifted, dtype=np.uint16), mode='a')
        lazy5.alter.write_attr_dict(dset='preprocessed_images/medfilter_ratio', attr_dict=attrs, fid=os.path.join(SAVE_FOLDER,f'preprocessed_medfilter_{filename}'))
        lazy5.alter.write_attr_dict(dset='preprocessed_images/medfilter_nrb_for_ratio', attr_dict=attrs, fid=os.path.join(SAVE_FOLDER,f'preprocessed_medfilter_{filename}'))
        lazy5.alter.write_attr_dict(dset='preprocessed_images/medfilter_dark', attr_dict=attrs, fid=os.path.join(SAVE_FOLDER,f'preprocessed_medfilter_{filename}'))
        
    
    else:
        data_smoothed_intcorrected, nrb_smoothed_intcorrected, _, smoothed_dark_shifted = intensity_correction(data_smoothed, nrb_smoothed, dark_smoothed, OUTPUT_RATIO = False)

        lazy5.create.save(file=f'preprocessed_medfilter_{filename}', pth=SAVE_FOLDER, dset='preprocessed_images/medfilter_raw', data = np.array(data_smoothed_intcorrected, dtype=np.uint16), mode='w')
        lazy5.create.save(file=f'preprocessed_medfilter_{filename}', pth=SAVE_FOLDER, dset='preprocessed_images/medfilter_nrb', data = np.array(nrb_smoothed_intcorrected, dtype=np.uint16), mode='a')
        lazy5.create.save(file=f'preprocessed_medfilter_{filename}', pth=SAVE_FOLDER, dset='preprocessed_images/medfilter_dark', data = np.array(smoothed_dark_shifted, dtype=np.uint16), mode='a')
        lazy5.alter.write_attr_dict(dset='preprocessed_images/medfilter_raw', attr_dict=attrs, fid=os.path.join(SAVE_FOLDER,f'preprocessed_medfilter_{filename}'))
        lazy5.alter.write_attr_dict(dset='preprocessed_images/medfilter_nrb', attr_dict=attrs, fid=os.path.join(SAVE_FOLDER,f'preprocessed_medfilter_{filename}'))
        lazy5.alter.write_attr_dict(dset='preprocessed_images/medfilter_dark', attr_dict=attrs, fid=os.path.join(SAVE_FOLDER,f'preprocessed_medfilter_{filename}'))
    

    end2 = time.time()
    print(f'spent {round(((end2 - start2)/60),2)} minutes for processing {filename}')

end = time.time()
print('')
print(f'total spent {round(((end - start)/60),2)} minutes')