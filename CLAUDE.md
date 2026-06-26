# bcars_processing

Self-contained end-to-end BCARS hyperspectral processing pipeline.
All source code is copied here (not referenced from sibling directories).
All tuneable parameters live in `params.yaml`.

## Pipeline

```
Raw HDF5 ──Step 1a──► preprocessed HDF5 ──Step 1b──► .mat ──Step 2──► _raman.tif + wn.txt
          (image-proc)                    (image-proc)       (crikit3)
```

### Step 1 — `step1_preprocess.py` (`image-proc` conda env)

**1a** `raw_h5_dir` → `preprocessed_h5_dir`:
dark subtraction · illumination correction · spectral alignment (CH-stretch) · VST

**1b** `preprocessed_h5_dir` → `mat_dir`:
spectral clip (400–1800 cm⁻¹) · Whittaker detrend · robust-symmetric normalisation → `.mat`

### Step 2 — `step2_process.py` (`crikit3` conda env)

Per `.mat` in `mat_dir`:
1. De-normalise to physical (dispersive VST) scale
2. CCV-SVD denoising — split-half correlation autotune (tau=0.5), symmetric reconstruction
3. Hilbert phase retrieval: add `C = max(|x|)` offset first (keeps real part ≥ 0, prevents phase
   from winding through 2π which would cause PEC to overcorrect). `Hilbert(C) ≈ 0` so Raman unchanged.
4. Phase error correction ALS (optional, default on) — paper params: `smoothness_param=10, asym_param=3e-4`
5. `imag(spectrum)` → Raman output; optionally `clip_negative: true` if PEC is disabled

**Output**: `<name>_raman.tif` — shape `(nspec, ny, nx)` float32 multipage TIFF
            `wn.txt` — one wavenumber (cm⁻¹) per line

## Quick start

```bash
# 1. Edit params.yaml: set paths.* and preprocessing.calibration
# 2. Run both steps:
bash run_pipeline.sh
# or with a custom config:
bash run_pipeline.sh params_0506_test.yaml

# Run steps individually:
conda run -n image-proc python step1_preprocess.py --config params.yaml
conda run -n crikit3    python step2_process.py    --config params.yaml
# Process a single .mat file:
conda run -n crikit3    python step2_process.py    --config params.yaml --file path/to/file.mat
```

## Directory layout

```
bcars_processing/
├── CLAUDE.md              this file
├── params.yaml            ALL tuneable parameters
├── params_0506_test.yaml  example per-dataset override
├── step1_preprocess.py    Step 1 entry point
├── step2_process.py       Step 2 entry point
├── run_pipeline.sh        shell harness (sequences both steps)
├── preprocess/
│   ├── GT_descan_BCARS_tools.py   BCARS loading + VST (calib_dict kwarg for YAML cal)
│   ├── h5tomat_detrend.py         HDF5 → MAT with Whittaker detrend
│   └── signal_alignment.py        cross-covariance spectral shift alignment
├── ccvsvd/
│   ├── SVD_utils.py               CrossCovarianceSVDResult + compute_crosscov_svd (trimmed)
│   └── criteria.py                select_by_corr, RECON dict (interleave/symmetric/subspace)
└── kk_pec/
    └── postprocess.py             PostprocessConfig, retrieve_phase (Hilbert/KK), PEC/SEC
```

## Calibration — must match acquisition date

Edit `preprocessing.calibration` in `params.yaml`:

| Dataset | `a_vec` |
|---------|---------|
| 20251111 | `[-1.01604041e-01, 7.93978909e+02]` |
| 0506 / 05032025 | `[-1.00147754e-01, 7.91061853e+02]` |

Other fields (`probe_nm`, `ctr_wl0_nm`, `fast_axis_step_um`, `fast_axis_steps`) are
typically the same across datasets; check the original `GT_descan_BCARS_tools.py`
comments if in doubt.

## Conda environments

| Env | Step | Key packages |
|-----|------|--------------|
| `image-proc` | 1 | h5py, scipy, statsmodels, lazy5, pyyaml |
| `crikit3` | 2 | crikit, scipy, tifffile, pyyaml |

Note: env name is `image-proc` (hyphen), not `image_proc`.
