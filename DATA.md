# Datasets used in this submission

This file is a condensed copy of the upstream dataset README at
`data/tg/tree_height/synrs3d/README.md` in our internal repo. All
recipes are reproduced verbatim so reviewers can re-fetch the data
independently.

| Dir | Source | Size | Tiles / frames | Used in |
|---|---|---:|---:|---|
| `synrs3d/SynRS3D/`      | SynRS3D synthetic (NeurIPS 2024 spotlight)                            | 206 GB | 69,667 | training (CATT) |
| `synrs3d/dfc/`          | IEEE GRSS DFC 2018 + 2019, repackaged by SynRS3D authors              | 22 GB  | 11,191 | **eval (CATT, aerial)** |
| `synrs3d/open_canopy/`  | Open-Canopy (CVPR 2025) — French SPOT 6/7 + IGN LiDAR-HD CHM          | 393 GB | n/a    | training (CATT) |
| `synrs3d/ogc/`          | OGC ATL + ARG, repackaged by SynRS3D authors                          | 66 GB  | 26,741 | training (optional, CATT) |
| `synrs3d/geonrw/`       | GeoNRW (Aachen→Wuppertal, 42 NRW cities), repackaged by SynRS3D authors | 212 GB | 122,624 | training (optional, CATT) |
| `arkitscenes/upsampling/` | Apple ARKitScenes upsampling track (RGB + iPhone-LiDAR + highres-LiDAR) | ~80 GB Val / ~530 GB Train | 287 / 1,970 videos | **eval (Hp7 vs PromptDA, indoor)** |
| `hypersim/`             | Apple HyperSim photorealistic indoor scenes                            | ~250 GB | 461 scenes | training (Hp7) |

The minimum download to reproduce the aerial CATT benchmark (`bash
scripts/download_data.sh --minimal`) is just `synrs3d/dfc/` (22 GB).
To reproduce the indoor Hp7 vs PromptDA benchmark instead, use `bash
scripts/download_data.sh --arkitscenes` (~80 GB; only the Validation
split is fetched). `--full` pulls everything plus SynRS3D + Open-Canopy
for retraining.

All commands below assume the working directory is `data/` inside
this reviewer package.

---

## `synrs3d/SynRS3D/` — synthetic source (NeurIPS 2024)

Hugging Face dataset
[JTRNEO/SynRS3D](https://huggingface.co/datasets/JTRNEO/SynRS3D).
17 zips, one per terrain × GSD × height-class subset.

```bash
hf download JTRNEO/SynRS3D --repo-type dataset --local-dir synrs3d/SynRS3D
# Then unzip each *.zip in place; one folder per subset will appear.
```

## `synrs3d/dfc/` — DFC18 + DFC19 (JAX/OMA)

Google Drive zips published by the SynRS3D authors (see
[their repo](https://github.com/JTRNEO/SynRS3D)).
Note: DFC19 zip extracts flat (no `DFC19/` parent), so a small
post-processing step creates `DFC19_JAX/`, `DFC19_OMA/` aliases (city
is encoded in the filename prefix).

```bash
gdown 1rq8w7YT25y2kxxRhuIpI68QeZmX1GZ0F -O DFC18.zip   # DFC18
gdown 1eoF16sxIHOQ5928SrboMqbi686sfKFLF -O DFC19.zip   # DFC19 (JAX+OMA combined)
```

`scripts/download_data.sh` runs both `gdown` calls and unpacks the
archives automatically.

## `synrs3d/open_canopy/` — Open-Canopy (CVPR 2025)

Hugging Face dataset
[AI4Forest/Open-Canopy](https://huggingface.co/datasets/AI4Forest/Open-Canopy).
Contains `canopy_height/` (single-epoch SPOT + LiDAR),
`canopy_height_change/` (multi-year pairs), `lidar_v2/` (newer CHM),
and `pretrained_models/`.

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 \
  hf download AI4Forest/Open-Canopy --repo-type dataset \
    --local-dir synrs3d/open_canopy
```

## `synrs3d/ogc/` — OGC_ATL + OGC_ARG (optional)

```bash
gdown 1tWBfrGKPbrPT1CyXp0iUm6_KItKYuiWb -O OGC_ATL.zip
gdown 1eTRYKdqX1Qce0Gq6EsmVQcWoiEioOdSB -O OGC_ARG.zip
```

## `synrs3d/geonrw/` — GeoNRW (optional)

```bash
gdown 1yWOWpgcpW2JhmBGedNApuzuNspnnVngF -O GeoNRW.zip
```

## `arkitscenes/upsampling/` — ARKitScenes depth-upsampling track

The official downloader at https://github.com/apple/ARKitScenes is the
only authoritative source (Apple S3 buckets refuse anonymous range
requests, so do not try to `wget` directly). The Validation split (287
videos, ~80 GB) is enough to reproduce
`benchmarking/results/arkitscenes/results_hp7_vs_promptda_arkit_summary.md`.

```bash
git clone --depth 1 https://github.com/apple/ARKitScenes \
    benchmarking/repos/ARKitScenes
( cd benchmarking/repos/ARKitScenes && \
  python3 download_data.py upsampling \
    --video_id_csv depth_upsampling/upsampling_train_val_splits.csv \
    --split Validation \
    --download_dir "$PWD/../../../data/arkitscenes/upsampling" )
```

`scripts/download_data.sh --arkitscenes` wraps both steps. Pass
`ARKIT_FULL=1` to also fetch the 1 970-video Training split (~530 GB).

Each `upsampling/Validation/<video_id>/` contains three
synchronised PNG streams used by the benchmark:

| Subdir | Format | Units |
|---|---|---|
| `wide/` (or `color/`) | RGB PNG (192 × 256 → resized to 448 × 448) | 8-bit |
| `lowres_depth/`       | uint16 PNG (iPhone LiDAR upsampled prompt)  | mm |
| `highres_depth/`      | uint16 PNG (Faro Focus3D ground truth)      | mm |

Both depth streams are clipped to 10 m before evaluation (matching
PromptDA's preprocessing).

## `hypersim/` — Apple HyperSim (Hp7 training)

Hp7 was trained on HyperSim's RGB + ground-truth depth pairs at
448 × 448. We use the official downloader from
https://github.com/apple/ml-hypersim. See
`data/tg/tree_height/synrs3d/README.md` (mirrored in our internal
repo) for the exact list of scenes and the preprocessing recipe.

---

## Licences

| Dataset | Licence |
|---|---|
| SynRS3D                  | CC-BY-NC-4.0 (research only) |
| DFC18 / DFC19            | IEEE GRSS Data Fusion Contest terms (research only) |
| Open-Canopy              | CC-BY-4.0 |
| OGC ATL / ARG            | CC-BY-4.0 (per upstream IEEE GRSS) |
| GeoNRW                   | dl-de/by-2-0 (Land NRW open data) |
| ARKitScenes              | CC-BY-NC-SA-4.0 (research only, Apple) |
| HyperSim                 | CC-BY-NC-SA-3.0 (research only, Apple)  |

## Citations

```bibtex
@article{song2024synrs3d,
  title  = {SynRS3D: A Synthetic Dataset for Global 3D Semantic
            Understanding from Monocular Remote Sensing Imagery},
  author = {Song, Jian and Chen, Hongruixuan and Xuan, Weihao and
            Xia, Junshi and Yokoya, Naoto},
  journal = {arXiv preprint arXiv:2406.18151},
  year    = {2024}
}

@inproceedings{fajwel2025opencanopy,
  title     = {Open-Canopy: A Country-Scale Benchmark for Canopy
               Height Estimation at Very High Resolution},
  author    = {Fajwel and others},
  booktitle = {CVPR},
  year      = {2025}
}

@inproceedings{baruch2021arkitscenes,
  title     = {{ARKitScenes}: A Diverse Real-World Dataset for {3D}
               Indoor Scene Understanding using Mobile {RGB-D} Data},
  author    = {Baruch, Gilad and Chen, Zhuoyuan and Dehghan, Afshin
               and Dimry, Tal and Feigin, Yuri and Fu, Peter and
               Gebauer, Thomas and Joffe, Brandon and Kurz, Daniel
               and Schwartz, Arik and Shulman, Elad},
  booktitle = {NeurIPS Datasets and Benchmarks Track},
  year      = {2021}
}

@inproceedings{roberts2021hypersim,
  title     = {{Hypersim}: A Photorealistic Synthetic Dataset for
               Holistic Indoor Scene Understanding},
  author    = {Roberts, Mike and Ramapuram, Jason and Ranjan, Anurag
               and Kumar, Atulit and Bautista, Miguel Angel and
               Paczan, Nathan and Webb, Russ and Susskind, Joshua M.},
  booktitle = {ICCV},
  year      = {2021}
}

@article{lin2024promptda,
  title   = {Prompt Depth Anything},
  author  = {Lin, Haotong and Peng, Sida and Zhang, Jingxiao and
             Wang, Xiaowei and Wang, Hujun and Zhou, Xiaowei},
  journal = {arXiv preprint arXiv:2412.14015},
  year    = {2024}
}
```
