#!/bin/bash
set -e
cd /home/prod-gpu-3/Documents/th/train_segmentation

RESDEPTH_CKPT="tree-heights/benchmarking/repos/ResDepth/logs/pretrained_models/ResDepth-stereo_trained_on_BER_ZUR/checkpoints/Model_best.pth"
CATT_CKPT="checkpoints/tree-height/P1_v1_catt_synoc_dfcval_20260503_023439/epoch=17-val_mae=1.9166.ckpt"
OUTPUT="tree-heights/benchmarking/results_msi"
DEVICE="cuda:1"
CONDA_ENV="pytorch"
IMG_FRAC="0.02"

echo "========================================"
echo "$(date) - Starting full MSI benchmark"
echo "  GPU: $DEVICE"
echo "  Image save fraction: $IMG_FRAC"
echo "========================================"

echo ""
echo "$(date) - [1/2] ResDepth (native 16-bit MSI stereo)"
conda run -n "$CONDA_ENV" python3 tree-heights/benchmarking/eval/corruption_sweep.py \
    --adapter resdepth \
    --checkpoint "$RESDEPTH_CKPT" \
    --dataset dfc_track2_msi \
    --output_dir "$OUTPUT" \
    --device "$DEVICE" \
    --save_images_frac "$IMG_FRAC"

echo ""
echo "$(date) - [2/2] CATT (RGB from MSI bands)"
conda run -n "$CONDA_ENV" python3 tree-heights/benchmarking/eval/corruption_sweep.py \
    --adapter ours \
    --checkpoint "$CATT_CKPT" \
    --dataset dfc_track2_msi \
    --output_dir "$OUTPUT" \
    --device "$DEVICE" \
    --save_images_frac "$IMG_FRAC"

echo ""
echo "========================================"
echo "$(date) - ALL DONE"
echo "========================================"

echo ""
echo "=== RESULTS SUMMARY ==="
for f in "$OUTPUT"/*.json; do
    python3 -c "
import json
d = json.load(open('$f'))
name = d.get('model','?') + ' / ' + d.get('regime','?')
print(f'  {name:<35s} MAE={d[\"mae\"]:6.3f}  RMSE={d[\"rmse\"]:6.3f}  R={d[\"pearson_r\"]:6.3f}')
" 2>/dev/null
done
