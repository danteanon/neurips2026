#!/bin/bash
# Re-run all ResDepth benchmarks after fixing the tile-blending bug
# (adapters/resdepth_adapter.py now uses 50% overlap + linear blend, matching
# the canonical ResDepth.lib.evaluation.predict_linear_blend pipeline).
#
# Two datasets:
#   1. dfc_val      (11,217 tiles, focused regimes, with tree-mask metrics)
#   2. dfc_track2_rgb (1,816 tiles, full 17-regime corruption sweep)
#
# Track2-MSI is intentionally NOT re-run -- it was deemed invalid earlier
# (raw MSI not orthorectified vs orthorectified DFC19 GT).

set -e
export PYTHONPATH="/home/prod-gpu-3/Documents/th/train_segmentation:/home/prod-gpu-3/Documents/th/train_segmentation/tree-heights/benchmarking"
export PYTHONUNBUFFERED=1

BENCH_DIR="/home/prod-gpu-3/Documents/th/train_segmentation/tree-heights/benchmarking"
SWEEP="$BENCH_DIR/eval/corruption_sweep.py"
GPU="${GPU:-cuda:1}"

RESDEPTH_CKPT="$BENCH_DIR/repos/ResDepth/logs/pretrained_models/ResDepth-stereo_trained_on_BER_ZUR/checkpoints/Model_best.pth"

OUT_TREEMASK="$BENCH_DIR/results_treemask/resdepth"
OUT_TRACK2RGB="$BENCH_DIR/results_track2_rgb"

mkdir -p "$OUT_TREEMASK" "$OUT_TRACK2RGB"

############################################################
# 1/2: DFC val (11,217 tiles) -- same focused regime set as CATT/CHMv2
#      so we get a clean head-to-head with tree-only metrics.
############################################################
echo "==============================================="
echo " [1/2] ResDepth on DFC val (4 regimes, tree-mask)"
echo "==============================================="
conda run -n pytorch --no-capture-output python $SWEEP \
    --adapter resdepth \
    --checkpoint "$RESDEPTH_CKPT" \
    --dataset dfc_val \
    --output_dir "$OUT_TREEMASK" \
    --device $GPU \
    --regimes regime_clean shift_24 shift_48 zero_chm \
    --save_images_frac 0.005

############################################################
# 2/2: DFC Track2-RGB (1,816 tiles) -- full 17-regime sweep,
#      mirroring the original benchmark we just deleted.
############################################################
echo "==============================================="
echo " [2/2] ResDepth on Track2-RGB (full sweep)"
echo "==============================================="
conda run -n pytorch --no-capture-output python $SWEEP \
    --adapter resdepth \
    --checkpoint "$RESDEPTH_CKPT" \
    --dataset dfc_track2_rgb \
    --output_dir "$OUT_TRACK2RGB" \
    --device $GPU \
    --save_images_frac 0.01

echo "============================================="
echo "  RESDEPTH RE-RUN COMPLETE"
echo "============================================="
