# CATT Reviewer Package

This folder contains everything required to reproduce both benchmarks
and run inference for the paper *"Diagnosing Cross-Attention Collapse
in Asymmetric Multi-Modal Fusion"* (NeurIPS 2026 submission).

Two model checkpoints are released with this package:

* **CATT (P1)** — DINOv3 ViT-L + DPT + cross-attention CHM fusion
  trained on SynRS3D + Open-Canopy stale pairs. Headline aerial
  canopy-height model (DFC corruption sweep).
* **Hp7** — DINOv2 ViT-L + Depth-Anything-V2 DPT head + the same
  cross-attention CHM decoder, trained on HyperSim only. The
  PromptDA-aligned baseline used for the indoor metric-depth
  prompt-corruption sweep against the published PromptDA-ViTL
  (ARKitScenes upsampling Validation, 2 000 frames × 5 regimes).

## Hardware requirements (read first)

A **CUDA-capable GPU is required**. The model loads its DINOv3 ViT-L
backbone weights directly onto GPU and the Lightning checkpoint
likewise stores tensors on CUDA, so a CPU-only run is not supported.

| Workload                         | Minimum         | Recommended            |
|----------------------------------|-----------------|------------------------|
| **Inference** (`scripts/infer.py`) | 1 × CUDA GPU, **24 GB VRAM** | RTX 3090 / 4090 / A5000 / L4 / A6000 |
| **Aerial corruption sweep** (`benchmarking/run_all_benchmarks.sh`) | 1 × CUDA GPU, 24 GB VRAM | same as above; ~30 min on A6000 |
| **Hp7 vs PromptDA on ARKit** (`benchmarking/run_hp7_arkit_benchmark.sh`) | 1 × CUDA GPU, 24 GB VRAM | ~35 min on A6000 (2 000 frames × 5 regimes × 2 models = 20 000 fwd passes) |
| **Training** (`code/scripts/train.py`)                    | 1 × CUDA GPU, 48 GB VRAM | A6000 / L40S / H100; ~7 h on A6000 |

CUDA toolkit **12.x** + driver **≥ 545.x** + PyTorch **2.11** are
expected. 24 GB is sized to comfortably hold the 506 M-parameter CATT
model in `bf16-mixed` plus a single 512 × 512 tile with cross-attention
activations; smaller GPUs will OOM during the corruption sweep.

```
reviewer_package/
├── README.md                   ← you are here
├── DATA.md                     ← per-dataset download recipes (verbatim from the upstream READMEs)
├── requirements.txt            ← pinned pip dependencies
├── environment.yml             ← optional conda alternative
├── setup.sh                    ← one-shot installer (deps + weights + data + competitor repos)
├── LICENSE
│
├── scripts/                    ← reviewer-facing CLIs
│   ├── download_weights.py     ← --variant catt | hp7 → weights/<ckpt>
│   ├── download_data.sh        ← --minimal | --arkitscenes | --full
│   ├── clone_competitor_repos.sh   ← git clones ResDepth, PromptDA, Open-Canopy, ARKitScenes
│   └── infer.py                ← USER-FACING: image + CHM prompt → height GeoTIFF
│
├── code/                       ← minimum subset of the training repo (verbatim copies)
│   ├── configs/
│   │   ├── P1_v1_catt_synoc_dfcval.yaml  ← CATT benchmarked config (aerial)
│   │   ├── B9_v1_catt.yaml               ← synthetic-only ablation parent of P1
│   │   └── Hp7_v1_hypersim_dav2.yaml     ← Hp7 indoor / ARKit baseline
│   ├── model/                  ← Dinov3HeightModelDPT + Dinov2HeightModelDPT + DPT decoder
│   ├── lightning_modules/      ← HeightEstimationModule
│   ├── data_loaders/           ← SynRS3DHeightDataset, HypersimDepthDataset, CHMCorruptor, OpenCanopyStalePairDataset
│   ├── losses/                 ← CATTLoss, GradientMatchingLoss, EdgeAwareSignedGradientLoss, L1HeightLoss, etc.
│   ├── optim/                  ← Muon optimiser
│   ├── utils/                  ← Normalization, schedulers, raster helpers
│   └── scripts/train.py        ← reproducibility entry point
│
├── benchmarking/               ← unified evaluation harness
│   ├── README.md               ← phase-by-phase plan (verbatim from the working repo)
│   ├── RESULTS.md              ← published tables and key takeaways
│   ├── adapters/               ← thin wrappers per competitor
│   ├── eval/                   ← corruption_sweep.py, hp7_vs_promptda_arkit.py, generate_tables.py
│   ├── results/                ← published per-experiment outputs
│   │   └── arkitscenes/        ← Hp7 vs PromptDA summary + per-frame JSON
│   ├── configs/, repos/        ← populated at runtime
│   ├── run_all_benchmarks.sh         ← reproduces the aerial DFC corruption sweep
│   └── run_hp7_arkit_benchmark.sh    ← reproduces Hp7 vs PromptDA on ARKitScenes
│
└── weights/                    ← populated by scripts/download_weights.py
                                  ↳ P1_catt_epoch17.ckpt   (~3.0 GB)
                                  ↳ Hp7_hypersim_last.ckpt (~3.6 GB)
```

## Quickstart (≤ 30 minutes on a single 24 GB GPU)

```bash
# 0. Verify CUDA is visible
nvidia-smi          # must list at least one GPU with ≥ 24 GB
python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'"

# 1. Install dependencies (a fresh venv is recommended)
python -m venv .venv && source .venv/bin/activate
# Install the matching torch wheel for your CUDA toolkit first, e.g.
# CUDA 12.8 (default for recent NVIDIA drivers):
pip install torch==2.11.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. Download the published checkpoints (~6.6 GB combined) into weights/
python scripts/download_weights.py                  # CATT P1 (~3.0 GB)
python scripts/download_weights.py --variant hp7    # Hp7    (~3.6 GB)

# 3. Download the eval data
bash scripts/download_data.sh --minimal       # DFC18 + DFC19 (~22 GB) for the aerial sweep
bash scripts/download_data.sh --arkitscenes   # ARKitScenes Validation (~80 GB) for Hp7 vs PromptDA

# 4. Clone competitor repos (PromptDA, ResDepth, Open-Canopy, ARKitScenes downloader)
bash scripts/clone_competitor_repos.sh

# 5a. Reproduce the aerial corruption sweep on the DFC val split (~30 min on A6000)
bash benchmarking/run_all_benchmarks.sh
#   → results land in benchmarking/results/

# 5b. Reproduce Hp7 vs PromptDA on ARKitScenes (~35 min on A6000, 2000 frames × 5 regimes)
bash benchmarking/run_hp7_arkit_benchmark.sh
#   → results land in benchmarking/results/arkitscenes/

# 6. Single-image inference
python scripts/infer.py \
    --image  path/to/rgb.tif \
    --chm    path/to/chm_prompt.tif \
    --output path/to/predicted_height.tif
```

`setup.sh` runs steps 1, 2, 3, 4 in sequence as a single command.

## What's bundled, what's downloaded

| Asset | Size | Source | Where it lands |
|---|---|---|---|
| Source code (this folder) | 2 MB | this submission | `code/`, `benchmarking/`, `scripts/` |
| **CATT (P1) checkpoint** (`Dinov3HeightModelDPT`, headline aerial model) | 3.0 GB | HF: `mldatauser/model_weights/height_estimation/P1_v1_catt_synoc_dfcval/` | `weights/P1_catt_epoch17.ckpt` |
| **Hp7 checkpoint** (`Dinov2HeightModelDPT`, indoor / ARKitScenes baseline) | 3.6 GB | HF: `mldatauser/model_weights/height_estimation/Hp7_v1_hypersim_dav2/` | `weights/Hp7_hypersim_last.ckpt` |
| **DFC18 + DFC19** (aerial eval, RGB + nDSM, 11 k tiles) | 22 GB | gdown (SynRS3D mirror) | `data/synrs3d/dfc/` |
| **ARKitScenes upsampling Validation** (RGB + lowres-LiDAR + highres-LiDAR triplets, 287 videos) | ~80 GB | Apple ARKitScenes downloader | `data/arkitscenes/upsampling/Validation/` |
| SynRS3D (training, synthetic, 70 k tiles, optional) | 206 GB | HF: `JTRNEO/SynRS3D` | `data/synrs3d/SynRS3D/` |
| Open-Canopy (training, real LiDAR, optional) | 393 GB | HF: `AI4Forest/Open-Canopy` | `data/synrs3d/open_canopy/` |
| HyperSim (Hp7 training, optional) | ~250 GB | follow `data/tg/tree_height/synrs3d/README.md` | `data/hypersim/` |
| ResDepth weights (competitor) | 200 MB | git clone (project page) | `benchmarking/repos/ResDepth/` |
| PromptDA weights (competitor) | 1.4 GB | HF: `depth-anything/prompt-depth-anything-vitl` (cached on-demand) | local HF cache |
| Open-Canopy PVTv2 weights (competitor) | 270 MB | HF: `AI4Forest/Open-Canopy` | downloaded by adapter |

The 22 GB DFC subset is enough to reproduce the aerial corruption
sweep (`benchmarking/RESULTS.md` Table 1). The 80 GB ARKitScenes
Validation split is enough to reproduce the indoor Hp7 vs PromptDA
sweep (`benchmarking/results/arkitscenes/results_hp7_vs_promptda_arkit_summary.md`).
The full ~620 GB SynRS3D + Open-Canopy + HyperSim download is only
needed to retrain from scratch.

## Which checkpoint reproduces which paper numbers?

| Paper section / table | Checkpoint | Local file (after `download_weights.py`) | Eval script |
|---|---|---|---|
| Aerial CHM corruption sweep (DFC18/19) | `P1_v1_catt_synoc_dfcval` epoch 17 (val MAE 1.9166 m) | `weights/P1_catt_epoch17.ckpt` | `benchmarking/run_all_benchmarks.sh` |
| Indoor metric-depth prompt-corruption sweep (ARKitScenes vs PromptDA) | `Hp7_v1_hypersim_dav2` `last.ckpt` (best so far: epoch 15, val MAE 1.7132 m) | `weights/Hp7_hypersim_last.ckpt` | `benchmarking/run_hp7_arkit_benchmark.sh` |

P1 inherits its architecture from the synthetic-only B9 ablation
(`configs/B9_v1_catt.yaml`); the only changes in P1 are training-data
related — adding Open-Canopy real-stale pairs and switching to per-source
stratified batching so every gradient step sees both synthetic and real
distributions. Everything else (σ-reparam stack, decoder LayerScale init,
CATT loss, counterfactual hinge, per-layer probe) is identical to B9.

Hp7 (`configs/Hp7_v1_hypersim_dav2.yaml`) shares the same cross-attention
decoder family as P1 but swaps backbone + head + auxiliary losses to
match PromptDA's recipe one-for-one (frozen DINOv2 ViT-L initialised
from Depth-Anything-V2, DA-V2 DPT head, L1 + edge-aware-signed-gradient
loss only — no CATT / counterfactual / aux-recon). It exists so the
prompt-corruption robustness story can be told in the regime where
PromptDA itself is the published state of the art (indoor metric depth
on ARKitScenes), independent of the aerial canopy-height question.

All three configs are bundled under `code/configs/` so the lineage is
auditable.

## Reproducing the ARKitScenes (Hp7 vs PromptDA) sweep

```bash
# Hp7 weights + ARKit data + PromptDA repo
python scripts/download_weights.py --variant hp7
bash   scripts/download_data.sh --arkitscenes
bash   scripts/clone_competitor_repos.sh

# 2 000 round-robin-sampled frames × {clean, shift_24, shift_48,
# cutout_25pct, cutout_50pct} × {Hp7, PromptDA} = 20 000 forward passes.
# ≈ 35 min on a single A6000.
bash benchmarking/run_hp7_arkit_benchmark.sh
```

Outputs land in `benchmarking/results/arkitscenes/`:

* `results_hp7_vs_promptda_arkit_summary.md` — published summary table.
* `results_hp7_vs_promptda_arkit_summary.json` — same metrics as JSON.
* `results_hp7_vs_promptda_arkit_full.json` — every per-(frame,
  regime, model) row (~3.7 MB).

The bundled summary file in this package is the exact output of the
script run on the blessed Hp7 checkpoint and is what the paper
references.

## Reproducing training (optional, requires the full data download)

```bash
bash scripts/download_data.sh --full       # ~620 GB (aerial + ARKit)

# Reproduce the published P1 (CATT, aerial) run — paper headline numbers
python code/scripts/train.py \
    --config code/configs/P1_v1_catt_synoc_dfcval.yaml \
    --gpu 0

# Or the synthetic-only B9 ablation (faster, no Open-Canopy needed)
python code/scripts/train.py \
    --config code/configs/B9_v1_catt.yaml \
    --gpu 0

# Reproduce the Hp7 (indoor / ARKit) run.  Requires HyperSim — see
# data/tg/tree_height/synrs3d/README.md, mirrored in this package as
# DATA.md, for the download recipe.  Also needs Depth-Anything-V2 ViT-L
# weights at model_weights/depth_anything_v2_vitl.pth (the config will
# fail loudly if either is missing).
python code/scripts/train.py \
    --config code/configs/Hp7_v1_hypersim_dav2.yaml \
    --gpu 0
```

Checkpoints land in `checkpoints/tree-height/<run_id>/`.

## Data licences

See `DATA.md` for a per-dataset summary of licences and citations.
SynRS3D is CC-BY-NC-4.0; DFC18/19 are research-use only via the IEEE
GRSS challenge; Open-Canopy is CC-BY-4.0; ARKitScenes is CC-BY-NC-SA-4.0
(research only); HyperSim is CC-BY-NC-SA-3.0.

## Contact

Anonymised for double-blind review. After acceptance, contact details
will be added here.
