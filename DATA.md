# Datasets used in this submission

This file is a condensed copy of the upstream dataset README at
`data/tg/tree_height/synrs3d/README.md` in our internal repo. All
recipes are reproduced verbatim so reviewers can re-fetch the data
independently.

| Dir | Source | Size | Tiles | Used in |
|---|---|---:|---:|---|
| `synrs3d/SynRS3D/`      | SynRS3D synthetic (NeurIPS 2024 spotlight)                            | 206 GB | 69,667 | training |
| `synrs3d/dfc/`          | IEEE GRSS DFC 2018 + 2019, repackaged by SynRS3D authors              | 22 GB  | 11,191 | **eval** |
| `synrs3d/open_canopy/`  | Open-Canopy (CVPR 2025) — French SPOT 6/7 + IGN LiDAR-HD CHM          | 393 GB | n/a    | training |
| `synrs3d/ogc/`          | OGC ATL + ARG, repackaged by SynRS3D authors                          | 66 GB  | 26,741 | training (optional) |
| `synrs3d/geonrw/`       | GeoNRW (Aachen→Wuppertal, 42 NRW cities), repackaged by SynRS3D authors | 212 GB | 122,624 | training (optional) |

The minimum download to reproduce the headline benchmark (`bash
scripts/download_data.sh --minimal`) is just `synrs3d/dfc/` (22 GB).
The full training set (`--full`) pulls SynRS3D + Open-Canopy + DFC.

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

---

## Licences

| Dataset | Licence |
|---|---|
| SynRS3D                  | CC-BY-NC-4.0 (research only) |
| DFC18 / DFC19            | IEEE GRSS Data Fusion Contest terms (research only) |
| Open-Canopy              | CC-BY-4.0 |
| OGC ATL / ARG            | CC-BY-4.0 (per upstream IEEE GRSS) |
| GeoNRW                   | dl-de/by-2-0 (Land NRW open data) |

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
```
