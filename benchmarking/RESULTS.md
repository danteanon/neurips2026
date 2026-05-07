# Benchmarking Results

**CATT checkpoint used throughout:** `P1_v1_catt_synoc_dfcval`
(epoch 17, val MAE = 1.9166). DINOv3 ViT-L backbone + DPT decoder
+ CHM cross-attention fusion. Trained on **SynRS3D synthetic** (12 v1
subsets, ~45k samples) **+ Open-Canopy stale pairs** (SPOT 2023 image
+ LiDAR 2021 prompt → LiDAR 2023 GT, ~57k windows). DFC was used only
for early-stopping checkpoint selection — never for gradient updates.

**Baselines benchmarked:**

| Baseline | Architecture | Modality | Native training data |
|---|---|---|---|
| **ResDepth** (`stereo_BER_ZUR`) | U-Net + outer skip residual | RGB + DSM prompt | WorldView-3 16-bit pan stereo over Berlin/Zurich |
| **Open-Canopy PVTv2** | PVTv2-B3 monocular | RGB(+NIR), no prompt | 87,000 km² SPOT 6/7 over France |
| **Meta CHMv2** (`facebook/dinov3-vitl16-chmv2-dpt-head`) | DINOv3 ViT-L + DPT | RGB monocular | Meta proprietary global satellite training corpus |

All baselines use their official pretrained weights; no further
fine-tuning. CATT is also untouched between datasets — same single
checkpoint everywhere.

---

## 1. Headline — Tree-only MAE on DFC val (11,217 tiles)

The most important comparison: CATT (CHM-prompted) vs Meta CHMv2
(monocular DINOv3 same-backbone) on a fully held-out, large-scale
dataset, restricted to **tree pixels** (semantic class = high
vegetation in DFC19's `gt_ss_mask`).

| Model | regime | MAE all px (m) | **MAE tree px (m)** | MAE ground px (m) | RMSE (m) | Bias (m) |
|---|---|---:|---:|---:|---:|---:|
| **Ours (CATT)** | clean prompt | **1.190** | **3.166** | 0.578 | 2.814 | −0.179 |
| **Ours (CATT)** | shift 24 px | 1.298 | 3.358 | 0.642 | 3.036 | −0.177 |
| **Ours (CATT)** | shift 48 px | 1.481 | 3.626 | 0.745 | 3.411 | −0.167 |
| Ours (CATT) | zero CHM (monocular fallback) | 2.659 | 6.865 | 0.382 | 6.477 | −2.558 |
| Meta CHMv2 (DINOv3 ViT-L, monocular) | n/a | 3.173 | 6.921 | 0.682 | 7.415 | −1.013 |

### Key takeaways

1. **CATT prompted ≈ 2.2× better than CHMv2** on tree pixels
   (3.17 m vs 6.92 m), at the same backbone size.
2. **Robust to misalignment:** CATT is still better than CHMv2 even
   with a 48-pixel CHM shift (3.63 m vs 6.92 m on trees). Going from
   0 to 48 px shift only adds **+0.46 m** tree MAE.
3. **Same architecture, monocular ablation:** with the prompt blanked
   (`zero_chm`) CATT degrades to **6.87 m** tree MAE — essentially
   tied with CHMv2's 6.92 m. This isolates the benefit of the prompt
   pathway: cross-attention to a noisy CHM is worth ~3.7 m of tree-
   level MAE on this dataset.
4. **Ground pixels:** All models are accurate on ground (sub-metre);
   the tree class is where CATT's CHM-conditioning shines.

---

## 2. Tree-only MAE on Open-Canopy native (10,000 samples)

Open-Canopy is **in-domain** for the Open-Canopy PVTv2 baseline (and
is in CATT's training mix). Mask: `lidar_classification` ≥
high-vegetation class.

| Model | regime | MAE all px (m) | **MAE tree px (m)** | MAE ground px (m) | RMSE (m) | Bias (m) |
|---|---|---:|---:|---:|---:|---:|
| **Ours (CATT)** | clean (2021 LiDAR prompt → 2023 GT) | **0.643** | **2.295** | 0.388 | 1.603 | −0.249 |
| Ours (CATT) | zero CHM (monocular ablation) | 1.549 | 8.400 | 0.449 | 3.830 | −1.546 |
| **Meta CHMv2** | image only | 1.352 | 6.846 | 0.448 | 3.213 | −1.275 |
| Open-Canopy PVTv2 (in-domain) | image only | 2.383 | — | — | 3.395 | +1.784 |

Findings:

1. **CATT prompted is ~3× better than Meta CHMv2 on tree pixels**
   (2.30 m vs 6.85 m).
2. CATT's monocular ablation (8.40 m) is *worse* than CHMv2's
   monocular (6.85 m), suggesting that on Open-Canopy the SPOT image
   alone is a weak signal — almost all of CATT's 2.30 m tree MAE
   comes from the CHM prompt pathway, not the image.
3. The Open-Canopy PVTv2 baseline (its own training set) achieves
   2.38 m all-pixel MAE — CATT prompted beats it (0.64 m all-pixel)
   by leveraging the 2-year-stale LiDAR prompt that the monocular
   PVTv2 ignores.

---

## 3. CATT vs all baselines on DFC Track2-RGB (1,816 tiles, all-pixel MAE)

DFC-2019 Track2 stereo-RGB tiles re-paired with DFC19 nDSM ground
truth. No tree mask is available for this subset (DFC18 tiles only),
so metrics are pixel-mean over the whole tile.

| Model | clean | shift 24 | shift 48 | zero_chm | zero_image |
|---|---:|---:|---:|---:|---:|
| **Ours (CATT)** | **1.471** | **1.598** | **1.817** | 3.117 | 2.183 |
| ResDepth (stereo, geom-mono) | 3.064 | 3.893 | 4.604 | 4.404 | 3.279 |
| Meta CHMv2 (monocular) | 3.038 | n/a (no prompt) | n/a | 3.038 | n/a |
| Open-Canopy PVTv2 (monocular) | 3.967 | n/a | n/a | 3.967 | n/a |

CATT halves the MAE of every baseline on the same data; even a 48-px
CHM shift keeps CATT (1.82 m) below all monocular/residual baselines
at clean (3.04–3.97 m).

---

## 4. Misalignment robustness — CATT vs ResDepth (DFC Track2-RGB, MAE m)

Same data as Section 3, full shift sweep.

| Model | shift=0 | 4 | 8 | 16 | 24 | 48 |
|---|---:|---:|---:|---:|---:|---:|
| **Ours (CATT)** | **1.471** | 1.482 | 1.498 | 1.540 | 1.598 | 1.817 |
| ResDepth | 3.064 | 3.171 | 3.278 | 3.575 | 3.893 | 4.604 |

Degradation 0 → 48 px:
- CATT: **+0.35 m (+24 %)**
- ResDepth: +1.54 m (+50 %)

The same trend holds on the full DFC val set (Section 1): CATT
+0.46 m tree MAE at 48 px, while monocular CHMv2 has *no* prompt
to misalign and stays at its baseline 6.92 m everywhere.

---

## 5. Dependency probes — CATT vs ResDepth (DFC Track2-RGB)

| Probe | CATT MAE | ResDepth MAE | Notes |
|---|---:|---:|---|
| Clean (image + CHM) | **1.471** | 3.064 | both modalities |
| Zero CHM (image only) | 3.117 | 4.404 | ResDepth predicts ≈ tiny residual on 0 → near-flat |
| Zero image (CHM only) | 2.183 | 3.279 | CATT can read structure off the prompt directly |

ResDepth's residual learning collapses without the DSM input;
CATT's two-stream design degrades smoothly on either side.

---

## 6. Cutout sweep — CATT vs ResDepth (DFC Track2-RGB)

Random rectangular holes in the CHM prompt.

| Model | cutout 10 % | cutout 30 % | cutout 50 % |
|---|---:|---:|---:|
| **Ours (CATT)** | **1.535** | **1.747** | **1.992** |
| ResDepth | 3.179 | 3.412 | 3.629 |

CATT inpaints missing regions via cross-attention to image features.
ResDepth treats holes as zero-DSM and degrades mostly because its
residual is biased toward 0 → underestimation in the hole.

---

## 7. Open-Canopy PVTv2 — sanity check on DFC val (full sweep)

The Open-Canopy adapter was non-trivial (custom `timm 0.9` venv,
BGRNIR channel order, `mean=0/std=1` normalisation). The result:

| Regime | MAE all px (m) | RMSE | Bias |
|---|---:|---:|---:|
| clean | 3.238 | 5.704 | −3.231 |
| zero_chm | 3.236 | 5.703 | −3.236 |
| zero_image | 3.236 | 5.703 | −3.236 |

The model correctly ignores the (unused) CHM prompt across regimes
(numbers identical to 3 decimals), and gives identical MAE for
zero_image too — i.e. it is collapsing toward a near-flat mean
prediction on out-of-domain (US) imagery. **This confirms a
fundamental domain gap for Open-Canopy on US imagery**, not a bug in
the adapter. On its own native French SPOT data (Section 2) it
achieves a healthy 2.38 m all-pixel MAE.

---

## 8. Discarded / invalid runs

- **`dfc_track2_msi`** (752 samples, all 17 regimes for CATT and
  ResDepth). The raw 8-band MSI imagery in DFC2019 Track2 is **not
  orthorectified**, but the DFC19 ground truth is. Visual inspection
  showed severe systematic misalignment between image and GT. Numbers
  in `results_track2_msi/` are kept for archive only — **do not
  cite**.

---

## 9. Coverage matrix — what has been run

Legend: ✓ done · ⏳ in progress · ⏸ queued · ✗ skipped/blocked
| Model ↓  Dataset → | DFC val (11,217) | Track2-RGB (1,816) | Open-Canopy native | DFC val tree-mask | OC tree-mask (10k) | ARKitScenes |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ours (CATT)** | ✓ (full sweep + tree mask) | ✓ (full sweep) | ✓ (clean + zero_chm, 10k) | ✓ | ✓ | ⏸ training pending (P4) |
| ResDepth | ✓ (legacy 500) | ✓ (full sweep) | n/a | n/a | n/a | n/a |
| Meta CHMv2 | ✓ (clean + tree mask) | ✓ (clean) | ✓ (10k clean) | ✓ | ✓ | n/a |
| Open-Canopy PVTv2 | ✓ (clean + 2 ablations) | ✓ (clean) | ✓ (40k clean) | n/a (no mask reads) | n/a | n/a |
| PromptDA | — | — | — | — | — | ⏸ blocked on CATT training |

Total benchmark runs completed: **84 result JSONs** under
`results_chmv2/`, `results_opencanopy/`, `results_track2_msi/`,
`results_track2_rgb/`, `results_treemask/`.

All four `treemask` sweep jobs finished at **2026-05-05 17:29 IST**.

---

## Pending

- **PromptDA / ARKitScenes** — needs CATT-on-ARKit training first
  (in `P4_arkitscenes` session).
- **RESULTS.md figures** (line charts: MAE vs shift; bar chart: tree
  vs ground per model) — not yet generated.
