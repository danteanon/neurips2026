#!/bin/bash
set -e

BENCH_DIR="/home/prod-gpu-3/Documents/th/train_segmentation/tree-heights/benchmarking"
VENV_PYTHON="$BENCH_DIR/.venv-opencanopy/bin/python"
SWEEP="$BENCH_DIR/eval/corruption_sweep.py"
OC_CKPT="$BENCH_DIR/repos/Open-Canopy/datasets/pretrained_models/pvtv2.ckpt"
GPU="cuda:0"

export PYTHONPATH="/home/prod-gpu-3/Documents/th/train_segmentation:$BENCH_DIR"

echo ">>> [1/2] Open-Canopy on DFC val"
$VENV_PYTHON $SWEEP \
    --adapter opencanopy \
    --checkpoint "$OC_CKPT" \
    --dataset dfc_val \
    --output_dir "$BENCH_DIR/results_opencanopy" \
    --device $GPU \
    --save_images_frac 0.005

echo ">>> [2/2] Open-Canopy on Track2-RGB"
$VENV_PYTHON $SWEEP \
    --adapter opencanopy \
    --checkpoint "$OC_CKPT" \
    --dataset dfc_track2_rgb \
    --output_dir "$BENCH_DIR/results_opencanopy" \
    --device $GPU \
    --save_images_frac 0.02

echo "=== Open-Canopy benchmark DONE ==="

echo ">>> [3/3] Open-Canopy on Open-Canopy dataset (SPOT 6/7)"
$VENV_PYTHON $SWEEP \
    --adapter opencanopy \
    --checkpoint "$OC_CKPT" \
    --dataset open_canopy \
    --output_dir "$BENCH_DIR/results_opencanopy" \
    --device $GPU \
    --save_images_frac 0.005

echo "=== ALL Open-Canopy benchmarks DONE ==="
