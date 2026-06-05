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
python GT_descan_BCARS_preprocessing.py <input> <output> [--mode {ratio,raw,vst}] [--med_filter {0,1}]
```

### Positional arguments

| Argument | Description |
|----------|-------------|
| `input`  | Path to the folder containing raw HDF5 (`.h5`) files |
| `output` | Path to the folder where preprocessed HDF5 files will be saved |

### Optional arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `ratio` | Output mode (**mutually exclusive**): `ratio` = BCARS/NRB intensity ratio; `raw` = dark-subtracted / illumination-corrected spectra; `vst` = variance-stabilized dispersive-like spectrum `(I − A_nrb²)/(2·A_nrb)` (saved as float32). |
| `--med_filter 1` | `1` | Apply 3D median filter before intensity correction. Set to `1` or ignore this cmd for crikit pipeline. Set to `0` to skip for N2N pipeline |

## Examples

**Basic usage (Recommended)** — process all `.h5` files in `20260210/`, save results to `20260210_out/`:

```bash
python GT_descan_BCARS_preprocessing.py 20260210 20260210_out
```

**Save raw corrected spectra** instead of the intensity ratio:

```bash
python GT_descan_BCARS_preprocessing.py 20260210 20260210_out --mode raw
```

**Save variance-stabilized (VST) spectra** — `(I − A_nrb²)/(2·A_nrb)`, per-line NRB amplitude:

```bash
python GT_descan_BCARS_preprocessing.py 20260210 20260210_out --mode vst
```

**Skip the median filter** (faster, no smoothing):

```bash
python GT_descan_BCARS_preprocessing.py 20260210 20260210_out --med_filter 0
```

**Skip median filter and save raw spectra:**

```bash
python GT_descan_BCARS_preprocessing.py 20260210 20260210_out --mode raw --med_filter 0
```

## Output

Each input `.h5` file produces one output file named `preprocessed_<mode>_<filename>.h5` in the output folder, where `<mode>` is either `medfilter` or `nofilter`.

The output HDF5 contains datasets under `preprocessed_images/` with spectroscopic calibration metadata written as HDF5 attributes. The `<prefix>` is `medfilter` (`--med_filter 1`, default) or `nofilter` (`--med_filter 0`):

| `--mode` | Datasets saved |
|----------|----------------|
| `ratio` (default) | `<prefix>_ratio`, `<prefix>_nrb_for_ratio`, `<prefix>_dark` |
| `raw` | `<prefix>_raw`, `<prefix>_nrb`, `<prefix>_dark` |
| `vst` | `<prefix>_vst` (float32), `<prefix>_vst_nrb_amp` (float32, per-line NRB amplitude `A_nrb`), `<prefix>_nrb_for_ratio` (ones placeholder), `<prefix>_dark` |

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
