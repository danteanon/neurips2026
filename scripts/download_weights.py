#!/usr/bin/env python
"""Download published model checkpoints from the Hugging Face Hub.

Two model families are bundled with this submission and hosted under
``mldatauser/model_weights/height_estimation/``:

* **CATT (P1)** — DINOv3 ViT-L + DPT + cross-attention CHM fusion,
  trained on SynRS3D + Open-Canopy stale pairs. The headline model in
  the paper. Best val MAE = 1.9166 m at epoch 17.

* **Hp7** — DINOv2 ViT-L + Depth-Anything-V2 DPT head + cross-attention
  CHM fusion, trained on HyperSim only. The PromptDA-aligned baseline
  used for the indoor metric-depth (ARKitScenes) prompt-corruption
  sweep. Training is ongoing; ``last.ckpt`` is the canonical artefact
  for now (best so far: val MAE 1.7132 m at epoch 15).

Layout on the Hub::

    mldatauser/model_weights/height_estimation/
    ├── P1_v1_catt_synoc_dfcval/
    │   ├── epoch=17-val_mae=1.9166.ckpt   ← default --variant catt
    │   ├── epoch=15-val_mae=1.9824.ckpt
    │   ├── epoch=18-val_mae=1.9715.ckpt
    │   ├── last.ckpt
    │   ├── config.yaml
    │   └── README.md
    └── Hp7_v1_hypersim_dav2/
        ├── last.ckpt                      ← default --variant hp7
        ├── config.yaml
        └── README.md

Usage
-----
    # default: pull the published epoch-17 CATT checkpoint into weights/
    python scripts/download_weights.py

    # pull the Hp7 last checkpoint (renamed to weights/Hp7_hypersim_last.ckpt)
    python scripts/download_weights.py --variant hp7

    # pull the *latest* epoch (last.ckpt) of CATT
    python scripts/download_weights.py --variant catt --filename last.ckpt

    # pull a specific other artefact (e.g. an alternate epoch)
    python scripts/download_weights.py \\
        --variant catt --filename "epoch=18-val_mae=1.9715.ckpt"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    level=logging.INFO)
log = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "mldatauser/model_weights"

# Per-variant defaults. The ``alias`` is the friendly local filename
# under ``weights/`` so reviewer-facing scripts (``run_*.sh``,
# ``infer.py``) can use a stable name regardless of which checkpoint
# epoch we end up publishing.
VARIANTS: dict[str, dict] = {
    "catt": {
        "subdir": "height_estimation/P1_v1_catt_synoc_dfcval",
        "default_filename": "epoch=17-val_mae=1.9166.ckpt",
        "alias": "P1_catt_epoch17.ckpt",
        "description": "CATT (DINOv3 + DPT + CHM cross-attn) — headline model.",
    },
    "hp7": {
        "subdir": "height_estimation/Hp7_v1_hypersim_dav2",
        # Currently published artefact. The training run is still
        # progressing — once it converges we'll add the best
        # epoch=*-val_mae=*.ckpt to the same subfolder; until then
        # last.ckpt is the canonical reference for reviewers.
        "default_filename": "last.ckpt",
        "alias": "Hp7_hypersim_last.ckpt",
        "description": "Hp7 (DINOv2 + DA-V2 + CHM cross-attn) — "
                       "PromptDA-aligned indoor baseline.",
    },
}


def _resolve_dest(variant: str, filename: str, dest_arg: Path | None) -> Path:
    """Pick the on-disk filename. If reviewer asked for the variant's
    default file, use the friendly alias; otherwise mirror the HF
    filename verbatim under ``weights/``.
    """
    if dest_arg is not None:
        return dest_arg
    weights_dir = PACKAGE_ROOT / "weights"
    if filename == VARIANTS[variant]["default_filename"]:
        return weights_dir / VARIANTS[variant]["alias"]
    return weights_dir / Path(filename).name


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--variant", choices=sorted(VARIANTS.keys()),
                        default="catt",
                        help="Which model to pull (default: %(default)s). "
                             "Use 'hp7' for the indoor ARKitScenes benchmark.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO,
                        help="HF repo id (default: %(default)s)")
    parser.add_argument("--subdir", default=None,
                        help="Override the subfolder inside the HF repo "
                             "(default: variant-specific).")
    parser.add_argument("--filename", default=None,
                        help="File inside <subdir>/ "
                             "(default: variant-specific best checkpoint)")
    parser.add_argument("--dest", type=Path, default=None,
                        help="Local destination path "
                             "(default: weights/<friendly alias> or "
                             "weights/<basename of --filename>).")
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    subdir = args.subdir or variant["subdir"]
    filename = args.filename or variant["default_filename"]
    dest = _resolve_dest(args.variant, filename, args.dest)

    log.info("Variant:  %s — %s", args.variant, variant["description"])
    log.info("HF path:  %s :: %s/%s", args.repo_id, subdir, filename)
    log.info("Local:    %s", dest)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit(
            "huggingface_hub is not installed. Run "
            "'pip install -r requirements.txt' first."
        )

    if dest.exists():
        log.info("Checkpoint already present at %s — nothing to do.", dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    path_in_repo = f"{subdir.rstrip('/')}/{filename}"

    # hf_transfer accelerates large multipart downloads when installed
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # token=None: huggingface_hub falls back to the cached credentials
    # written by `hf auth login`. The mldatauser/model_weights repo is
    # public, so anonymous downloads also work for reviewers who haven't
    # logged in.
    cached = hf_hub_download(
        repo_id=args.repo_id,
        filename=path_in_repo,
        local_dir=str(dest.parent),
    )
    cached_path = Path(cached)
    if cached_path.resolve() != dest.resolve():
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        try:
            dest.symlink_to(cached_path.resolve())
        except OSError:
            import shutil
            shutil.copy2(cached_path, dest)

    size_gb = dest.stat().st_size / 1e9 if dest.exists() else 0
    log.info("Done — %s (%.2f GB)", dest, size_gb)


if __name__ == "__main__":
    main()
