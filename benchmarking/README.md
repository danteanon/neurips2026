# CHM Height Model — Benchmarking Plan

**Goal:** Benchmark our CHM-prompted cross-attention height model (CATT)
against competing depth/height models, demonstrating our unique advantage:
robustness to misaligned, degraded, and absent depth prompts.

---

## Models Under Test

| # | Model | Paper | Type | Prompt? | Code |
|---|---|---|---|---|---|
| 1 | **Ours (CATT)** | — | Cross-attn CHM-prompted height | Yes (stale CHM) | This repo |
| 2 | **ResDepth** | ISPRS 2022 | U-Net residual DSM refinement | Yes (noisy DSM) | `repos/ResDepth/` |
| 3 | **Open-Canopy PVTv2** | CVPR 2025 | Monocular image → height | No | `repos/Open-Canopy/` |
| 4 | **Prompt Depth Anything** | CVPR 2025 | LiDAR-prompted depth estimation | Yes (aligned LiDAR) | `repos/PromptDA/` |

### Why these four

- **ResDepth** takes (DSM + image) → refined DSM. Same input modalities as
  ours, but assumes pixel-aligned DSM and learns residuals. Will break under
  spatial shift because residual targets become meaningless when the input
  DSM is shifted relative to the image.
- **Open-Canopy PVTv2** is pure monocular (no depth prompt). It establishes
  the floor: how well can you do without a prompt at all? Our model with any
  non-zero prompt should beat it, and our zero-prompt mode should be
  competitive.
- **Prompt Depth Anything** is the closest architectural analogue — it also
  fuses a depth prompt into a DPT decoder. But it assumes pixel-aligned
  iPhone LiDAR and uses additive feature injection, not cross-attention.
  They explicitly tested cross-attention and it performed worse *under
  alignment*. Our hypothesis: cross-attention is better *under
  misalignment*.

### Dropped

- **ChangeDA** (IEEE TGRS 2025): No code released. Noted in related work.
- **ECHOSAT / PrediTree**: Multi-temporal Sentinel time-series input, not
  (image + CHM). Apples-to-oranges. Referenced in related work only.

---

## Python Environments

| Environment | Type | Location | Purpose |
|---|---|---|---|
| `pytorch` | Conda (global) | `miniconda3/envs/pytorch/` | Main env for CATT, ResDepth, PromptDA, CHMv2. torch=2.11, timm=1.0.15, transformers=5.7.0 |
| `.venv-opencanopy` | uv venv (local) | `tree-heights/benchmarking/.venv-opencanopy/` | Open-Canopy PVTv2 only. timm=0.9.16, torch=2.11. Created because Open-Canopy requires timm 0.9.x which is incompatible with the global timm 1.0.15. |

**Usage:**
```bash
# Global pytorch env (default for most benchmarks)
conda run -n pytorch python corruption_sweep.py ...

# Open-Canopy (needs timm 0.9)
.venv-opencanopy/bin/python corruption_sweep.py --adapter opencanopy ...
```

---

## Evaluation Axes

### Axis 1 — Misalignment Robustness (headline result)

Apply controlled spatial shift to the depth/CHM prompt and measure MAE
degradation. Our `CHMCorruptor` already implements this via `max_shift`.

```
shift_px ∈ {0, 4, 8, 16, 24, 48}
```

**Expected:** Our model's MAE stays nearly flat. Competitors' MAE rises
steeply with shift.

### Axis 2 — Prompt Dependency Probes

Test whether each model actually uses both modalities:

| Probe | What it tests |
|---|---|
| `zero_chm` — replace CHM/depth with all zeros | Is the model just copying the prompt? |
| `zero_image` — replace image with zeros/gray | Is the model just a monocular model ignoring the prompt? |

### Axis 3 — Degradation Regimes

Apply the 6 standard `CHMCorruptor.for_regime()` corruptions to the prompt:

```
clean, shifted, masked, degraded, blurred, zero
```

Plus additional cutout sweep:

```
cutout_area ∈ {10%, 30%, 50%}
```

---

## Directory Layout

```
train_segmentation/
├── tree-heights/                          # gitignored in root repo, tracked locally
│   └── benchmarking/
│       ├── README.md                      # THIS FILE
│       ├── repos/                         # cloned competitor code
│       │   ├── ResDepth/
│       │   ├── PromptDA/
│       │   └── Open-Canopy/
│       ├── adapters/                      # thin wrappers per competitor
│       │   ├── resdepth_adapter.py
│       │   ├── promptda_adapter.py
│       │   └── opencanopy_adapter.py
│       ├── eval/                          # unified evaluation harness
│       │   ├── benchmark_eval.py          # core metrics
│       │   ├── corruption_sweep.py        # regime sweep orchestrator
│       │   └── generate_tables.py         # JSONs → tables + plots
│       ├── configs/                       # per-model benchmark configs
│       ├── results/                       # output JSONs
│       └── figures/                       # generated plots
│
├── data/tg/tree_height/
│   ├── synrs3d/                           # EXISTING
│   │   ├── SynRS3D/                       # synthetic train/val (56k train, 24k val)
│   │   ├── dfc/DFC18, DFC19              # real nDSM (11k samples, no WV-3 stereo)
│   │   └── open_canopy/                   # real stale LiDAR pairs (SPOT + LiDAR-HD)
│   └── benchmarking/                      # NEW — large downloads
│       ├── dfc2019_us3d/                  # DFC2019 full (WV-3 panchromatic + ALS DSM)
│       ├── hypersim/                      # HyperSim subset (Phase 5B only)
│       └── arkitscenes/                   # ARKitScenes depth split (Phase 5B only)
```

---

## Phases

### Phase 1: Scaffold & Downloads

**1a. Directory structure** (done if you're reading this)

```bash
mkdir -p tree-heights/benchmarking/{repos,adapters,eval,configs,results,figures}
mkdir -p data/tg/tree_height/benchmarking/{dfc2019_us3d,hypersim,arkitscenes}
echo "tree-heights/" >> .gitignore
cd tree-heights && git init
```

**1b. Clone repos**

```bash
cd tree-heights/benchmarking/repos
git clone https://github.com/prs-eth/ResDepth.git
git clone https://github.com/DepthAnything/PromptDA.git
git clone https://github.com/fajwel/Open-Canopy.git
```

**1c. Download datasets**

| Dataset | Destination | Source | Size | Needed for |
|---|---|---|---|---|
| DFC2019 US3D | `data/tg/tree_height/benchmarkings/dfc2019_us3d/` | [IEEE DataPort](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019) (free account) | ~13 GB (Track2+3 val) | Phase 3: ResDepth benchmark (WorldView-3 stereo + ALS DSM) |
| ARKitScenes upsampling | `data/tg/tree_height/benchmarkings/arkitscenes/` | [Apple GitHub](https://github.com/apple/ARKitScenes) | ~100 GB | Phase 5: Train our model on PromptDA's data |
| HyperSim subset | `data/tg/tree_height/benchmarkings/hypersim/` | [Apple GitHub](https://github.com/apple/ml-hypersim) | ~50 GB | Phase 5: Additional training data for indoor depth |

**Existing data (no downloads needed):**
- `dfc/DFC18` + `DFC19` — RGB + nDSM (11k samples). Used in Phase 4.
- `open_canopy/` — SPOT + stale LiDAR pairs (57k windows). Used in Phase 4.

Note: existing `dfc/DFC18` + `DFC19` have RGB (`opt/`) + nDSM
(`gt_nDSM/`) but lack the WorldView-3 panchromatic stereo images that
ResDepth requires as input. DFC2019 US3D provides those.

---

### Phase 2: Unified Evaluation Harness

**Files:** `tree-heights/benchmarking/eval/`

#### `benchmark_eval.py`

Core metric computation. Signature:

```python
def evaluate(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> dict:
    """Returns {mae, rmse, bias, mae_canopy, mae_ground,
                pearson_r, gradient_error}"""
```

- `mae_canopy`: MAE over pixels where GT > 2 m
- `mae_ground`: MAE over pixels where GT <= 2 m
- `gradient_error`: MAE of Sobel(pred) vs Sobel(gt)

#### `corruption_sweep.py`

Orchestrator. For a given model + dataset, applies every regime and writes
result JSONs.

```python
REGIMES = {
    # Misalignment sweep
    "shift_0":  {"max_shift": 0},
    "shift_4":  {"max_shift": 4},
    "shift_8":  {"max_shift": 8},
    "shift_16": {"max_shift": 16},
    "shift_24": {"max_shift": 24},
    "shift_48": {"max_shift": 48},
    # Cutout sweep
    "cutout_10pct": {"cutout_prob": 1.0, "cutout_area_range": [0.1, 0.1]},
    "cutout_30pct": {"cutout_prob": 1.0, "cutout_area_range": [0.3, 0.3]},
    "cutout_50pct": {"cutout_prob": 1.0, "cutout_area_range": [0.5, 0.5]},
    # Dependency probes
    "zero_chm":   "ZERO_CHM",
    "zero_image": "ZERO_IMAGE",
    # Standard 6 regimes (reuse CHMCorruptor.for_regime)
    "clean":    "clean",
    "shifted":  "shifted",
    "masked":   "masked",
    "degraded": "degraded",
    "blurred":  "blurred",
    "zero":     "zero",
}
```

Each model adapter must implement:

```python
class ModelAdapter:
    def load(self, checkpoint_path: str, device: str): ...
    def predict(self, image: np.ndarray, chm: np.ndarray) -> np.ndarray: ...
    @property
    def name(self) -> str: ...
    @property
    def accepts_chm(self) -> bool: ...  # False for monocular models
```

**Inference tile size:** All models must be run on **512×512 tiles**.
If the input image is larger than 512×512, tile it into non-overlapping
512×512 patches (pad the right/bottom edges if needed), run inference
on each tile independently, and stitch the predictions back together.
Do not use overlapping tiles — each pixel should be predicted exactly
once to avoid smoothing artifacts from averaging.

Output per run: `results/{model}_{dataset}_{regime}.json`

#### `generate_tables.py`

Reads all result JSONs and produces:
- `results/table1_shift_sweep.md` — MAE per model per shift level
- `results/table2_dependency.md` — full / zero_chm / zero_image per model
- `results/table3_regimes.md` — 6 standard regimes per model
- `figures/mae_vs_shift.png` — line chart, one line per model
- `figures/dependency_probes.png` — grouped bar chart

---

### Phase 3: ResDepth Benchmark (no fine-tuning) — COMPLETE

**Code:** `tree-heights/benchmarking/adapters/resdepth_adapter.py`

**ResDepth's input/output format:**
- Input: initial DSM (GeoTIFF) + orthorectified panchromatic satellite image(s) (GeoTIFF)
- Output: refined DSM (initial + learned residual)
- Resolution: 0.25 m GSD (their default), configurable
- Architecture: U-Net with outer skip (residual learning), depth=5, 64 start kernels
- Framework: PyTorch 1.9, MIT license
- Paper: Stucker & Schindler, ISPRS J. 2022

**ResDepth training data:**
- **Regions:** Zurich (ZUR1, ZUR2, ZUR3) and Berlin (BER) — urban areas
- **Imagery:** WorldView-3 VHR panchromatic stereo imagery, 16-bit, 0.25 m GSD
- **Ground truth:** High-resolution LiDAR DSMs
- **Data availability:** Not publicly available — "due to the commercial
  nature of VHR imagery, we cannot share our complete datasets."
- **Checkpoint used:** `ResDepth-stereo_trained_on_BER_ZUR` (multi-city
  model, 3 input channels: DSM + stereo pair). This is their best
  model for geographical generalization.
- **Normalization:** DSM and image normalization parameters stored in
  `DSM_normalization_parameters.p` and `Image_normalization_parameters.p`.
  The image normalization was derived from 16-bit panchromatic data
  (mean/std for ~0–2048 range). We use per-image z-normalization
  instead to bridge the 8-bit RGB gap.

**Our CATT model:**
- **Training data:** SynRS3D synthetic (~45k, 12 v1 subsets) +
  Open-Canopy real stale pairs (~57k windows, SPOT 2023 + LiDAR 2021).
  Two-stream stratified batching.
- **Validation data (checkpoint selection only):** SynRS3D held-out
  (~22k, v2 subsets) + DFC18/DFC19 (~11k). DFC was used only for
  early stopping — the model never saw DFC during gradient updates.
- DINOv3 backbone + DPT decoder + CHM cross-attention fusion
- Checkpoint: `P1_v1_catt_synoc_dfcval` (epoch 10, val MAE=2.14)

**Strategy:** Both models evaluated on **DFC18/DFC19 val** data (RGB +
nDSM). No fine-tuning — use pretrained weights as-is. ResDepth receives
nDSM as its initial DSM input and grayscale RGB as its image input.

**Results:** See Tables 1–5 above. Key findings:
- shift_0: Our CATT model is 2x better than ResDepth even under
  perfect alignment (MAE 1.40 vs 2.86).
- shift_4+: ResDepth degrades rapidly (+57% MAE at 48px shift).
  Ours stays nearly flat (+21%).
- zero_chm: ResDepth collapses (MAE=4.86, R=−0.19). Ours degrades
  gracefully (MAE=2.91, R=0.58).
- zero_image: ResDepth partial function (3.53). Ours better (2.26).

**Data:** DFC18/DFC19 val (existing at `dfc/DFC18`, `dfc/DFC19`).

---

### Phase 4: Open-Canopy Benchmark (no fine-tuning)

**Code:** `tree-heights/benchmarking/adapters/opencanopy_adapter.py`

**Their input/output format:**
- Input: SPOT 6/7 image (1.5 m GSD, RGB+NIR)
- Output: canopy height map (no depth prompt accepted)
- Pretrained on: Open-Canopy dataset (87k km² France, SPOT + LiDAR-HD)
- Weights: HuggingFace `AI4Forest/Open-Canopy`

**Strategy:** Both models evaluated on **multiple datasets** using
their pretrained weights. Open-Canopy is monocular (no prompt), so
this is an asymmetric comparison showing that our CHM prompt adds
value over image-only approaches.

**Steps:**

1. **On Open-Canopy data** (existing at `open_canopy/`):
   - Run Open-Canopy PVTv2 on SPOT 2023 images → predict CHM
   - Run our CATT on same images + stale CHM 2021 as prompt
   - Evaluate both against GT CHM 2023

2. **On DFC18/DFC19 data** (existing at `dfc/`):
   - Run Open-Canopy PVTv2 on DFC RGB images → predict height
   - Run our CATT on same images + corrupted nDSM as prompt
   - Evaluate both against GT nDSM

3. **Asymmetric comparison table:**

   | Config | Model | Dataset |
   |---|---|---|
   | image only | Open-Canopy PVTv2 | Open-Canopy, DFC |
   | image + CHM (clean) | Ours | Open-Canopy, DFC |
   | image + CHM (shifted) | Ours | Open-Canopy, DFC |
   | image + CHM (degraded) | Ours | Open-Canopy, DFC |
   | image + zeros | Ours (zero_chm) | Open-Canopy, DFC |

**Expected results:**
- Ours (clean prompt) >> PVTv2 (the prompt provides substantial signal)
- Ours (degraded prompt) > PVTv2 (even a bad prompt helps)
- Ours (zero prompt) ≈ PVTv2 (both monocular; backbone difference)

**Data used:** `open_canopy/` (existing) + `dfc/` (existing).
No new downloads.

---

### Phase 5: Prompt Depth Anything Benchmark (train ours on their data)

**Code:** `tree-heights/benchmarking/adapters/promptda_adapter.py`

**Their input/output format:**
- Input: RGB image + low-res LiDAR depth (192x256, iPhone)
- Output: metric depth map (up to 4K)
- Architecture: Depth Anything v2 + multi-scale additive prompt fusion
- Pretrained on: ARKitScenes + HyperSim (indoor depth, iPhone LiDAR)
- Weights: ViT-Large (340M params), ViT-Small (25M params)
- Assumes: pixel-aligned LiDAR + image, same-time capture

**Strategy:** Train **our CATT model** on **PromptDA's datasets**
(ARKitScenes / HyperSim). Then both models are evaluated on
ARKitScenes val with corruption sweeps. This proves our cross-attention
architecture generalizes beyond remote sensing and handles misalignment
better than their additive fusion even on their home turf.

**Steps:**

1. **Download PromptDA's training data:**
   - ARKitScenes upsampling split → `data/tg/tree_height/benchmarkings/arkitscenes/`
   - HyperSim subset → `data/tg/tree_height/benchmarkings/hypersim/`
   - Total: ~150 GB

2. **Write ARKitScenes/HyperSim data loader** for our model:
   - Image = RGB
   - CHM prompt = LiDAR depth (with CHMCorruptor applied for
     misalignment augmentation during training)
   - GT = dense depth

3. **Train our CATT model** on ARKitScenes + HyperSim train splits.

4. **Evaluate both on ARKitScenes val:**
   - PromptDA-L with pretrained weights (their best result)
   - Our CATT trained on the same data
   - Corruption sweep on both: shift the LiDAR prompt, zero it out,
     degrade it

5. **Side-by-side comparison** with corruption sweep.

**Expected results:**
- shift_0: PromptDA competitive or better (optimized for aligned input,
  additive fusion is efficient when alignment holds)
- shift_4+: PromptDA degrades, ours stays flat (their additive fusion
  has no spatial flexibility; our cross-attention can attend across
  positions)
- zero_chm: Both degrade, but differently
- zero_image: PromptDA may struggle (foundation model is image-centric)

**Key insight:** PromptDA's paper explicitly tested cross-attention
fusion and found it worse (L1=0.0523 vs 0.0163 for additive). But they
only tested under pixel-aligned conditions. Our hypothesis: cross-
attention wins under misalignment, which is the real-world scenario.

---

## Results

See **[RESULTS.md](RESULTS.md)** for all benchmark results, tables, and
key takeaways.

---

## Competitor Model Details (reference)

### ResDepth (Stucker & Schindler, ISPRS J. Photogrammetry 2022)
- **Architecture:** U-Net with outer skip connection (residual learning)
- **Training data:** WorldView-3 VHR panchromatic stereo imagery (16-bit,
  0.25 m GSD) over Zurich (ZUR1–3) and Berlin (BER), with LiDAR DSM
  ground truth. Urban scenes — buildings, roads, vegetation.
- **Input:** Initial stereo DSM + orthorectified panchromatic image(s)
- **Output:** Refined DSM (input DSM + learned residual correction)
- **Key assumption:** DSM and image are pixel-aligned and contemporaneous.
  Residual learning target = (GT DSM − input DSM), so any spatial shift
  between input DSM and image makes the residual target meaningless.
- **Reported accuracy:** ~30% MAE reduction over raw stereo DSM on
  their Zurich/Berlin test regions.
- **Checkpoint used in benchmark:** `ResDepth-stereo_trained_on_BER_ZUR`
  (multi-city, 3 input channels). We run in geom-mono mode (2 channels:
  DSM + single grayscale image).

### Open-Canopy (Fajwel et al., CVPR 2025)
- **Architecture:** PVTv2-B3 backbone + ConvTranspose2d decoder
  (decoder_stride=32)
- **Training data:** 87,000 km² of France — SPOT 6/7 imagery (1.5 m
  GSD, 4-channel RGBNIR) + LiDAR-HD canopy height ground truth
- **Input:** SPOT 6/7 image (4 channels: RGB + NIR)
- **Output:** Canopy height map (no depth prompt accepted, monocular)
- **Reported accuracy:** ~2.5–3.0 m MAE on their France benchmark
- **Status:** Adapter written. Near-zero predictions on DFC data with
  timm 1.0.15 (domain gap + timm version mismatch). Local uv venv
  `.venv-opencanopy` with timm=0.9.16 available for re-testing.

### Prompt Depth Anything (Yin et al., CVPR 2025)
- **Architecture:** Depth Anything v2 (ViT-L, 340M params) + multi-scale
  additive prompt fusion blocks
- **Training data:** ARKitScenes (iPhone LiDAR + Faro laser GT, indoor)
  + HyperSim (synthetic indoor depth)
- **Input:** RGB image + low-resolution LiDAR depth (192×256, iPhone)
- **Output:** Metric depth map (up to 4K resolution)
- **Key assumption:** LiDAR prompt and image are pixel-aligned and
  captured at the same time (iPhone sensor fusion). Uses additive
  feature injection, not cross-attention.
- **Reported accuracy:**
  - ARKitScenes: L1=0.0132, RMSE=0.0315 (768×1024, non-zero-shot)
  - ScanNet++: L1=0.0250, RMSE=0.0829, F-score=0.7619
  - Cross-attention fusion tested and was WORSE: L1=0.0523 (vs 0.0163
    for additive) — but only under pixel-aligned conditions.
- **Status:** Pending (Phase 5 — train our CATT on ARKitScenes first)

### Depth Any Canopy (reference, not benchmarked directly)
- EarthView: MAE=0.1304 (ViT-B), IoU=0.5926
- HRCHM: MAE=0.1203 (ViT-B)
- 97.5M params, no depth prompt

### Meta CHMv2 (reference, not benchmarked directly)
- SatLidar v2 test: MAE=3.0 m (vs CHMv1's 4.3 m)
- DINOv3-Sat-L backbone, 1 m GSD
- No depth prompt
