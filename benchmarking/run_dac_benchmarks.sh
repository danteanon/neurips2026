#!/bin/bash
# Depth Any Canopy (DAC) benchmarks — same data + regime grid as the
# tree-mask CHMv2/CATT runs so we can directly compare in RESULTS.md.
#
# DAC is monocular (DAv2 ViT-B + DPT, fine-tuned on EarthView NEON).
# It ignores the prompt, so we only sweep regime_clean for it (corrupt-
# prompt regimes would yield identical numbers).
set -e
export PYTHONPATH="/home/prod-gpu-3/Documents/th/train_segmentation:/home/prod-gpu-3/Documents/th/train_segmentation/tree-heights/benchmarking"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # use the cached snapshot, do not hit the network

BENCH_DIR="/home/prod-gpu-3/Documents/th/train_segmentation/tree-heights/benchmarking"
SWEEP="$BENCH_DIR/eval/corruption_sweep.py"
GPU="${GPU:-cuda:1}"
OUT_DAC="$BENCH_DIR/results_treemask/dac"

mkdir -p "$OUT_DAC"

############################################################
# 1/2: DFC val (DFC18 Houston + DFC19 Jacksonville/Omaha)
#      11,217 tiles, of which 11,132 have semantic class masks.
############################################################
echo "==============================================="
echo " [1/2] DAC on DFC val (regime_clean only)"
echo "==============================================="
conda run -n pytorch --no-capture-output python "$SWEEP" \
    --adapter dac \
    --checkpoint "" \
    --dataset dfc_val \
    --output_dir "$OUT_DAC" \
    --device "$GPU" \
    --regimes regime_clean \
    --save_images_frac 0.005

############################################################
# 2/2: Open-Canopy (France SPOT + LiDAR HD; in-domain forest)
#      10k samples to mirror the CATT/CHMv2 OC run.
############################################################
echo "==============================================="
echo " [2/2] DAC on Open-Canopy (10k samples; regime_clean)"
echo "==============================================="
conda run -n pytorch --no-capture-output python "$SWEEP" \
    --adapter dac \
    --checkpoint "" \
    --dataset open_canopy \
    --output_dir "$OUT_DAC" \
    --device "$GPU" \
    --max_samples 10000 \
    --regimes regime_clean \
    --save_images_frac 0.002

echo "============================================="
echo "  DAC BENCHMARKS DONE"
echo "============================================="
