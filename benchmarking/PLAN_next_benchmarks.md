# Next Benchmarks Plan

**Date:** 2026-05-04
**Current status:** ResDepth + CATT running on Track2-RGB-2 and Track2-MSI-1 (in tmux `benchmarks`, GPU 1).

---

## Overview

Three new competitor models to benchmark against our CATT:

| # | Model | Type | Input | Code | Weights |
|---|-------|------|-------|------|---------|
| A | **Open-Canopy PVTv2** | Monocular canopy height | RGBNIR (4-ch) | `repos/Open-Canopy/` (cloned) | `pvtv2.ckpt` (present) |
| B | **PromptDA-L** | Prompted metric depth | RGB + LiDAR depth | `repos/PromptDA/` (cloned) | HuggingFace 1.4 GB (verified accessible) |
| C | **Meta CHMv2** | Monocular canopy height | Satellite RGB | HuggingFace `facebook/dinov3-vitl16-chmv2-dpt-head` | 0.3B params, needs `transformers` upgrade |

---

## Benchmark A: Open-Canopy PVTv2 — Monocular vs Prompted

### What it proves
Open-Canopy is image-only (no depth prompt). Comparing it against
CATT across shift regimes shows that **even a degraded/misaligned CHM
prompt adds substantial value** over the best monocular baseline.

### Model details
- **Architecture:** PVTv2-B3 backbone + ConvTranspose2d decoder
- **Training data:** 87,000 km² France, SPOT 6/7 (RGBNIR, 1.5m GSD) + LiDAR-HD
- **Input:** 4-channel RGBNIR (verified: `patch_embed.proj.weight` = `[64, 4, 7, 7]`)
- **Output:** Canopy height map (monocular, no prompt)
- **Checkpoint:** `repos/Open-Canopy/datasets/pretrained_models/pvtv2.ckpt`

### Data available
- **Track2-MSI-1** (188 tiles): Has all 8 WV-3 MSI bands including NIR
  (Band 7, 770-895nm). Can construct proper RGBNIR: R=B5(idx4),
  G=B3(idx2), B=B2(idx1), NIR=B7(idx6). GT from DFC19 nDSM.
- **Track2-RGB-2** (454 tiles): Only RGB. Will duplicate Red as
  synthetic NIR (or zero-fill 4th channel).
- **DFC18/DFC19 val** (11k tiles): RGB only, same NIR workaround.

### What's needed
1. **Fix the Open-Canopy adapter** — currently broken by `timm` version
   issue: `features_only=True` renames `stages.N` → `stages_N` in
   newer timm, breaking checkpoint key matching. Two fix options:
   - (a) Manually reconstruct the PVTv2 forward pass without
     `features_only=True` (partially done in adapter).
   - (b) Pin/downgrade timm to a version where key names match.
2. **Add RGBNIR loader** — new dataset loader variant that constructs
   4-channel input from MSI bands for Open-Canopy.
3. **Run corruption sweep** — Open-Canopy is monocular so it only
   runs once (no shift/cutout variants), but we compare its single
   result against CATT's full sweep to show prompt advantage.

### Comparison table (to fill)

| Config | Model | Dataset |
|--------|-------|---------|
| image only (RGBNIR) | Open-Canopy PVTv2 | Track2-MSI, DFC val |
| image + CHM (clean) | CATT | same |
| image + CHM (shifted 4-48px) | CATT | same |
| image + CHM (degraded) | CATT | same |
| image + CHM (zero) | CATT | same |

### Effort: Medium (~2-3 hours)
### Can start: **NOW** — all data and weights present.

---

## Benchmark B: PromptDA-L — Additive Fusion vs Cross-Attention

### What it proves
PromptDA uses additive prompt fusion (depth features added into decoder
at multiple scales). Their paper **explicitly tested cross-attention
and found it worse** (L1=0.0523 vs 0.0163 for additive) — but only
under pixel-aligned conditions. Our hypothesis: **cross-attention wins
under misalignment**, which is the real-world scenario.

### Model details
- **Architecture:** Depth Anything v2 ViT-L (340M params) + multi-scale
  additive prompt fusion
- **Training data:** ARKitScenes (iPhone LiDAR + Faro GT, indoor) + HyperSim
- **Input:** RGB image + low-res LiDAR depth (192×256)
- **Output:** Metric depth map (up to 4K)
- **Weights:** `depth-anything/prompt-depth-anything-vitl` on HuggingFace (1.4 GB, verified accessible)

### Data available
- **ARKitScenes Training:** 1,693 scenes, 105 GB (downloaded, present)
- **ARKitScenes Validation:** NOT yet downloaded. Need to run the
  download script for the Validation split.
- **Adapter:** `adapters/promptda_adapter.py` exists (needs verification)

### What's needed
1. **Download PromptDA weights** — 1.4 GB from HuggingFace.
2. **Download ARKitScenes Validation split** — extend the existing
   download script to fetch Validation scenes.
3. **Train CATT on ARKitScenes** — config exists at
   `configs/height/production/P2_v1_catt_arkitscenes_depth.yaml`.
   Data loader exists at `data_loaders/arkitscenes_dataset.py`.
   This is the longest lead-time item.
4. **Verify/fix PromptDA adapter** — test that it loads and runs on
   a sample ARKitScenes scene.
5. **Run corruption sweep on ARKitScenes val** — both PromptDA-L
   (pretrained) and CATT (trained on ARKitScenes) with shift sweep
   on the LiDAR prompt.

### Comparison table (to fill)

| Model | shift=0 | shift=4 | shift=8 | shift=16 | shift=24 | shift=48 | zero_prompt | zero_image |
|-------|---------|---------|---------|----------|----------|----------|-------------|------------|
| PromptDA-L (pretrained) | — | — | — | — | — | — | — | — |
| CATT (trained on ARKit) | — | — | — | — | — | — | — | — |

### Effort: High (~1-2 days, mostly CATT training)
### Can start: Download weights + val split NOW, training after GPU frees up.

---

## Benchmark C: Meta CHMv2 — Same Backbone, Monocular vs Prompted

### What it proves
CHMv2 uses **DINOv3 ViT-L** as its backbone — the **same backbone
family as our CATT model**. CHMv2 is monocular (image only), while
CATT adds cross-attention CHM prompt fusion. This is the most
controlled comparison possible: **same backbone, does adding a depth
prompt via cross-attention help?**

### Model details
- **Architecture:** DINOv3 ViT-L (frozen) + DPT head (0.3B params)
- **Training data:** Global ALS canopy height data, SAT-493M satellite
  pre-training
- **Input:** Satellite RGB image
- **Output:** Canopy height at 1m GSD
- **HuggingFace:** `facebook/dinov3-vitl16-chmv2-dpt-head`
- **Status:** Model type `chmv2` not recognized by current `transformers`
  version. Needs upgrade to latest transformers.

### Data available
- **Track2-RGB-2** (454 tiles): Proper satellite RGB + AGL GT.
- **DFC18/DFC19 val** (11k tiles): Satellite RGB + nDSM GT.
- Both datasets ready to go.

### What's needed
1. **Upgrade transformers** — `pip install --upgrade transformers` or
   install from source. CHMv2 model type was added recently.
2. **HuggingFace login** — model may be gated (requires agreeing to
   terms and `hf auth login`).
3. **Write CHMv2 adapter** — load via `CHMv2ForDepthEstimation`, apply
   `CHMv2ImageProcessorFast`, run inference. Monocular so no prompt
   variants, just single run.
4. **Run on Track2-RGB-2 and DFC val** — compare CHMv2 monocular
   against CATT with full corruption sweep.

### Comparison table (to fill)

| Config | Model | Track2-RGB MAE | DFC val MAE |
|--------|-------|----------------|-------------|
| image only (monocular) | CHMv2 (DINOv3 + DPT) | — | — |
| image + CHM (clean) | CATT (DINOv3 + cross-attn) | — | — |
| image + CHM (shifted 48px) | CATT | — | — |
| image + CHM (zero) | CATT | — | — |

### Effort: Medium (~3-4 hours)
### Can start: After `transformers` upgrade + HF login.

---

## Execution Order

```
Priority  Action                                          Blocked by    ETA
────────  ──────────────────────────────────────────────  ──────────    ────
  1       Fix Open-Canopy adapter + run on MSI data       Nothing       2-3h
  2       Upgrade transformers + CHMv2 adapter + run      HF login      3-4h
  3       Download PromptDA weights (background)          Nothing       10min
  4       Download ARKitScenes val split (background)     Nothing       ~2h
  5       Train CATT on ARKitScenes                       GPU slot      ~12h
  6       Run PromptDA benchmark on ARKitScenes val       Steps 4+5     ~2h
```

Items 1 and 2 can run on data we already have.
Items 3 and 4 are background downloads that should start immediately.
Item 5 needs a free GPU (GPU 0 is busy with CATT production training).
Item 6 depends on everything above.

---

## Current Benchmark Status (as of 2026-05-04 17:30 IST)

### Running now (tmux `benchmarks`, GPU 1):
- [x] ResDepth on Track2-RGB-2 (17/17 regimes) — **DONE**
- [~] CATT on Track2-RGB-2 (11/17 regimes) — **running**, ~1h left
- [ ] ResDepth on Track2-MSI-1 — queued
- [ ] CATT on Track2-MSI-1 — queued

### Early results (Track2-RGB-2, 1816 samples):

| Model | shift=0 MAE | Pearson R |
|-------|-------------|-----------|
| **CATT** | **1.47** | **0.834** |
| ResDepth | 3.06 | 0.665 |
