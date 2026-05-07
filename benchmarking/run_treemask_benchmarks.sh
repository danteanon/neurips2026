#!/bin/bash
# Re-run benchmarks with tree-only / ground-only / tree+ground metrics.
# CATT (ours) and Meta CHMv2 on DFC val + Open-Canopy datasets.
set -e
export PYTHONPATH="/home/prod-gpu-3/Documents/th/train_segmentation:/home/prod-gpu-3/Documents/th/train_segmentation/tree-heights/benchmarking"
export PYTHONUNBUFFERED=1

BENCH_DIR="/home/prod-gpu-3/Documents/th/train_segmentation/tree-heights/benchmarking"
SWEEP="$BENCH_DIR/eval/corruption_sweep.py"
GPU="${GPU:-cuda:1}"

CATT_CKPT="/home/prod-gpu-3/Documents/th/train_segmentation/checkpoints/tree-height/P1_v1_catt_synoc_dfcval_20260503_023439/epoch=17-val_mae=1.9166.ckpt"
OUT_CATT="$BENCH_DIR/results_treemask/catt"
OUT_CHMV2="$BENCH_DIR/results_treemask/chmv2"
OC_VENV="$BENCH_DIR/.venv-opencanopy/bin/python"

mkdir -p "$OUT_CATT" "$OUT_CHMV2"

############################################################
# DFC val (DFC19's 11,132 tiles have class masks; DFC18's 85 do not)
# Focused regime set for the headline tree-only comparison:
#   - regime_clean  : best-case CATT (clean prompt)
#   - shift_24      : realistic moderate misalignment
#   - shift_48      : aggressive misalignment
#   - zero_chm      : CATT pretending to be monocular (fairest direct comparison vs CHMv2)
############################################################
echo "==============================================="
echo " [1/4] CATT on DFC val (4 regimes)"
echo "==============================================="
conda run -n pytorch --no-capture-output python $SWEEP \
    --adapter ours \
    --checkpoint "$CATT_CKPT" \
    --dataset dfc_val \
    --output_dir "$OUT_CATT" \
    --device $GPU \
    --regimes regime_clean shift_24 shift_48 zero_chm \
    --save_images_frac 0.005

echo "==============================================="
echo " [2/4] CHMv2 on DFC val (regime_clean only)"
echo "==============================================="
conda run -n pytorch --no-capture-output python $SWEEP \
    --adapter chmv2 \
    --checkpoint "" \
    --dataset dfc_val \
    --output_dir "$OUT_CHMV2" \
    --device $GPU \
    --regimes regime_clean \
    --save_images_frac 0.005

############################################################
# Open-Canopy (40k samples, both models, regime_clean)
# CATT: regime_clean uses the natural stale 2021 LiDAR prompt.
# CHMv2: monocular, ignores prompt.
############################################################
echo "==============================================="
echo " [3/4] CATT on Open-Canopy (10k samples; clean + zero_chm)"
echo "==============================================="
conda run -n pytorch --no-capture-output python $SWEEP \
    --adapter ours \
    --checkpoint "$CATT_CKPT" \
    --dataset open_canopy \
    --output_dir "$OUT_CATT" \
    --device $GPU \
    --max_samples 10000 \
    --regimes regime_clean zero_chm \
    --save_images_frac 0.002

echo "==============================================="
echo " [4/4] CHMv2 on Open-Canopy (10k samples)"
echo "==============================================="
conda run -n pytorch --no-capture-output python $SWEEP \
    --adapter chmv2 \
    --checkpoint "" \
    --dataset open_canopy \
    --output_dir "$OUT_CHMV2" \
    --device $GPU \
    --max_samples 10000 \
    --regimes regime_clean \
    --save_images_frac 0.002

echo "============================================="
echo "  ALL TREE-MASK BENCHMARKS DONE"
echo "============================================="
