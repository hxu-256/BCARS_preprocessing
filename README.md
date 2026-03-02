# BCARS Preprocessing Pipeline

Preprocessing pipeline for Broadband Coherent Anti-Stokes Raman Scattering (BCARS) microscopy data from C. elegans mitochondrial imaging. Processes hyperspectral HDF5 image stacks through illumination correction and spectral alignment.

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
```

This creates an environment named `image_proc` with Python 3.12 and all required packages.

> **Note:** The `lazy5` HDF5 wrapper may not be fully available via pip. If you encounter import errors, install it manually from the source:
> ```bash
> pip install LazyHdf5
> ```

### 2. Activate the environment

```bash
conda activate image_proc
```

## Usage

```
python GT_descan_BCARS_preprocessing.py <input> <output> [--ratio {0,1}] [--med_filter {0,1}]
```

### Positional arguments

| Argument | Description |
|----------|-------------|
| `input`  | Path to the folder containing raw HDF5 (`.h5`) files |
| `output` | Path to the folder where preprocessed HDF5 files will be saved |

### Optional arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ratio 1` | `1` | Save intensity ratio (BCARS/NRB) output. Set to `0` to save raw corrected spectra instead. |
| `--med_filter 1` | `1` | Apply 3D median filter before intensity correction. Set to `1` or ignore this cmd for crikit pipeline. Set to `0` to skip for N2N pipeline |

## Examples

**Basic usage (Recommended)** — process all `.h5` files in `20260210/`, save results to `20260210_out/`:

```bash
python GT_descan_BCARS_preprocessing.py 20260210 20260210_out
```

**Save raw corrected spectra** instead of the intensity ratio:

```bash
python GT_descan_BCARS_preprocessing.py 20260210 20260210_out --ratio 0
```

**Skip the median filter** (faster, no smoothing):

```bash
python GT_descan_BCARS_preprocessing.py 20260210 20260210_out --med_filter 0
```

**Skip median filter and save raw spectra:**

```bash
python GT_descan_BCARS_preprocessing.py 20260210 20260210_out --ratio 0 --med_filter 0
```

## Output

Each input `.h5` file produces one output file named `preprocessed_<mode>_<filename>.h5` in the output folder, where `<mode>` is either `medfilter` or `nofilter`.

The output HDF5 contains datasets under `preprocessed_images/` with spectroscopic calibration metadata written as HDF5 attributes:

| `--ratio` | `--med_filter` | Datasets saved |
|-----------|----------------|----------------|
| `1` (default) | `1` (default) | `medfilter_ratio`, `medfilter_nrb_for_ratio`, `medfilter_dark` |
| `1` | `0` | `nofilter_ratio`, `nofilter_nrb_for_ratio`, `nofilter_dark` |
| `0` | `1` (default) | `medfilter_raw`, `medfilter_nrb`, `medfilter_dark` |
| `0` | `0` | `nofilter_raw`, `nofilter_nrb`, `nofilter_dark` |

## Pipeline overview

```
signal_alignment.py          — sub-pixel spectral shift correction (cross-covariance + spline interpolation)
       ↓
GT_descan_BCARS_tools.py     — core processing: dark subtraction, illumination normalization, median filtering
       ↓
GT_descan_BCARS_preprocessing.py  — main script: loads HDF5 files, runs pipeline, writes output
```

**Processing stages per file:**
1. Load raw BCARS, NRB reference, and dark frames from HDF5
2. Collapse 4D → 3D (sum over axis 3) if data is 4-dimensional
3. Apply 3×1×1 median filter (parallelized, optional)
4. Dark subtraction, per-row illumination normalization, spectral shift correction
5. Write corrected data to output HDF5 with calibration attributes
