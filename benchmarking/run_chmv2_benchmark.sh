#!/bin/bash
set -e
export PYTHONPATH="/home/prod-gpu-3/Documents/th/train_segmentation:/home/prod-gpu-3/Documents/th/train_segmentation/tree-heights/benchmarking"
export PYTHONUNBUFFERED=1

BENCH_DIR="/home/prod-gpu-3/Documents/th/train_segmentation/tree-heights/benchmarking"
SWEEP="$BENCH_DIR/eval/corruption_sweep.py"
GPU="cuda:0"

echo ">>> [1/2] CHMv2 on Track2-RGB (proper RGB satellite + AGL GT)"
conda run -n pytorch --no-capture-output python $SWEEP \
    --adapter chmv2 \
    --checkpoint "" \
    --dataset dfc_track2_rgb \
    --output_dir "$BENCH_DIR/results_chmv2" \
    --device $GPU \
    --regimes regime_clean \
    --save_images_frac 0.02

echo ">>> [2/2] CHMv2 on DFC val (DFC18/19, 11k samples)"
conda run -n pytorch --no-capture-output python $SWEEP \
    --adapter chmv2 \
    --checkpoint "" \
    --dataset dfc_val \
    --output_dir "$BENCH_DIR/results_chmv2" \
    --device $GPU \
    --regimes regime_clean \
    --save_images_frac 0.005

echo "=== CHMv2 benchmarks DONE ==="
