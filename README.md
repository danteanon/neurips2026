# CATT Reviewer Package

This folder contains everything required to reproduce the headline
benchmark and run inference for the paper *"Diagnosing Cross-Attention
Collapse in Asymmetric Multi-Modal Fusion"* (NeurIPS 2026 submission).

## Hardware requirements (read first)

A **CUDA-capable GPU is required**. The model loads its DINOv3 ViT-L
backbone weights directly onto GPU and the Lightning checkpoint
likewise stores tensors on CUDA, so a CPU-only run is not supported.

| Workload                         | Minimum         | Recommended            |
|----------------------------------|-----------------|------------------------|
| **Inference** (`scripts/infer.py`) | 1 × CUDA GPU, **24 GB VRAM** | RTX 3090 / 4090 / A5000 / L4 / A6000 |
| **Benchmark sweep** (`benchmarking/run_all_benchmarks.sh`) | 1 × CUDA GPU, 24 GB VRAM | same as above; ~30 min on A6000 |
| **Training** (`code/scripts/train.py`)                    | 1 × CUDA GPU, 48 GB VRAM | A6000 / L40S / H100; ~7 h on A6000 |

CUDA toolkit **12.x** + driver **≥ 545.x** + PyTorch **2.11** are
expected. 24 GB is sized to comfortably hold the 506 M-parameter B9
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
│   ├── download_weights.py     ← pulls the B9 checkpoint from Hugging Face
│   ├── download_data.sh        ← wraps the DATA.md recipes (gdown + hf download)
│   ├── clone_competitor_repos.sh   ← git clones ResDepth, PromptDA, Open-Canopy
│   └── infer.py                ← USER-FACING: image + CHM prompt → height GeoTIFF
│
├── code/                       ← minimum subset of the training repo (verbatim copies)
│   ├── configs/
│   │   ├── P1_v1_catt_synoc_dfcval.yaml  ← THE benchmarked config (default)
│   │   └── B9_v1_catt.yaml               ← synthetic-only ablation parent of P1
│   ├── model/                  ← Dinov3HeightModelDPT + DPT decoder + DINOv3 backbone
│   ├── lightning_modules/      ← HeightEstimationModule
│   ├── data_loaders/           ← SynRS3DHeightDataset, CHMCorruptor, OpenCanopyStalePairDataset
│   ├── losses/                 ← CATTLoss, GradientMatchingLoss, etc.
│   ├── optim/                  ← Muon optimiser
│   ├── utils/                  ← Normalization, schedulers, raster helpers
│   └── scripts/train.py        ← reproducibility entry point
│
├── benchmarking/               ← unified evaluation harness
│   ├── README.md               ← phase-by-phase plan (verbatim from the working repo)
│   ├── RESULTS.md              ← published tables and key takeaways
│   ├── adapters/               ← thin wrappers per competitor
│   ├── eval/                   ← corruption_sweep.py, benchmark_eval.py, generate_tables.py
│   ├── configs/, repos/        ← populated at runtime
│   └── run_all_benchmarks.sh   ← reproduces our DFC corruption sweep
│
└── weights/                    ← populated by scripts/download_weights.py (~3 GB)
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

# 2. Download the published B9 checkpoint (~3 GB) into weights/
python scripts/download_weights.py

# 3. Download the minimum eval data (DFC18 + DFC19, ~22 GB) into data/
bash scripts/download_data.sh --minimal

# 4. (Optional) Clone competitor repos for cross-model benchmarks
bash scripts/clone_competitor_repos.sh

# 5. Reproduce the corruption sweep on the DFC val split
bash benchmarking/run_all_benchmarks.sh
#   → results land in benchmarking/results/

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
| **P1 checkpoint** (`Dinov3HeightModelDPT`, our published model) | 3.0 GB | HF: `mldatauser/model_weights/height_estimation/P1_v1_catt_synoc_dfcval/` | `weights/P1_catt_epoch17.ckpt` |
| **DFC18 + DFC19** (eval, RGB + nDSM, 11 k tiles) | 22 GB | gdown (SynRS3D mirror) | `data/synrs3d/dfc/` |
| SynRS3D (training, synthetic, 70 k tiles, optional) | 206 GB | HF: `JTRNEO/SynRS3D` | `data/synrs3d/SynRS3D/` |
| Open-Canopy (training, real LiDAR, optional) | 393 GB | HF: `AI4Forest/Open-Canopy` | `data/synrs3d/open_canopy/` |
| ResDepth weights (competitor) | 200 MB | git clone (project page) | `benchmarking/repos/ResDepth/` |
| PromptDA weights (competitor) | 1.4 GB | HF (loaded on-demand by adapter) | local cache |
| Open-Canopy PVTv2 weights (competitor) | 270 MB | HF: `AI4Forest/Open-Canopy` | downloaded by adapter |

The 22 GB DFC subset is enough to reproduce the headline result
(`benchmarking/RESULTS.md` Table 1).  The 600 GB SynRS3D + Open-Canopy
download is only needed to retrain B9 from scratch.

## Which checkpoint reproduces the paper numbers?

Every metric in `benchmarking/RESULTS.md` was produced by a single
checkpoint: `P1_v1_catt_synoc_dfcval` epoch 17, val MAE = 1.9166 m.
This is the file that `scripts/download_weights.py` pulls by default
(`weights/P1_catt_epoch17.ckpt`) and that `benchmarking/run_all_benchmarks.sh`
uses without any extra flags.

P1 inherits its architecture from the synthetic-only B9 ablation
(`configs/B9_v1_catt.yaml`); the only changes in P1 are training-data
related — adding Open-Canopy real-stale pairs and switching to per-source
stratified batching so every gradient step sees both synthetic and real
distributions. Everything else (σ-reparam stack, decoder LayerScale init,
CATT loss, counterfactual hinge, per-layer probe) is identical to B9.

Both configs are bundled under `code/configs/` so the lineage is
auditable.

## Reproducing training (optional, requires the full data download)

```bash
bash scripts/download_data.sh --full       # 600 GB

# Reproduce the published P1 run (this is the one whose numbers go in the paper)
python code/scripts/train.py \
    --config code/configs/P1_v1_catt_synoc_dfcval.yaml \
    --gpu 0

# Or the synthetic-only B9 ablation (faster, no Open-Canopy needed)
python code/scripts/train.py \
    --config code/configs/B9_v1_catt.yaml \
    --gpu 0
```

Checkpoints land in `checkpoints/tree-height/<run_id>/`.

## Data licences

See `DATA.md` for a per-dataset summary of licences and citations.
SynRS3D is CC-BY-NC-4.0; DFC18/19 are research-use only via the IEEE
GRSS challenge; Open-Canopy is CC-BY-4.0.

## Contact

Anonymised for double-blind review. After acceptance, contact details
will be added here.
