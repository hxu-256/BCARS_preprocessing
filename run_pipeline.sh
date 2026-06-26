#!/usr/bin/env bash
# run_pipeline.sh  — end-to-end BCARS processing harness
#
# Usage:
#   bash run_pipeline.sh                       # uses params.yaml
#   bash run_pipeline.sh my_params.yaml        # custom config
#   bash run_pipeline.sh params.yaml --skip_preprocess   # re-run from .h5 only
#
# Environment requirements:
#   Step 1: image_proc   conda env (h5py, scipy, statsmodels, lazy5)
#   Step 2: crikit3      conda env (crikit, scipy, numpy)

set -euo pipefail

CONFIG="${1:-params.yaml}"
EXTRA_ARGS="${*:2}"   # any extra flags forwarded to both steps

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================"
echo "  BCARS pipeline  |  config: $CONFIG"
echo "========================================================"
echo ""

echo "--- Step 1: Preprocessing (image_proc env) ---"
conda run -n image-proc python "$SCRIPT_DIR/step1_preprocess.py" \
    --config "$SCRIPT_DIR/$CONFIG" $EXTRA_ARGS

echo ""
echo "--- Step 2: CCV-SVD + KK/PEC (crikit3 env) ---"
conda run -n crikit3 python "$SCRIPT_DIR/step2_process.py" \
    --config "$SCRIPT_DIR/$CONFIG"

echo ""
echo "Pipeline complete."
