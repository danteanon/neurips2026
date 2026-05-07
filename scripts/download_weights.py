#!/usr/bin/env python
"""Download the published P1 (CATT) checkpoint from the Hugging Face Hub.

The reference checkpoint for the paper is
``P1_v1_catt_synoc_dfcval`` (epoch 17, val MAE = 1.9166 m). It is
hosted in the public ``mldatauser/model_weights`` repo under
``height_estimation/P1_v1_catt_synoc_dfcval/``.

The same repo also hosts:
  * ``last.ckpt``                              (final epoch)
  * ``epoch=15-val_mae=1.9824.ckpt``           (alt early-stop)
  * ``epoch=18-val_mae=1.9715.ckpt``           (alt late-stop)
  * ``config.yaml``                            (training config)
  * ``README.md``                              (model card)

Usage
-----
    # default: pulls the published epoch-17 checkpoint into weights/
    python scripts/download_weights.py

    # download the *latest* epoch (last.ckpt) instead
    python scripts/download_weights.py --filename last.ckpt

    # download a specific other artefact in the same subfolder
    python scripts/download_weights.py \
        --filename epoch=18-val_mae=1.9715.ckpt
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

# Public mirror — the same `mldatauser` org that hosts the DINOv3
# backbone weights pulled by code/model/dinov3_model.py.
DEFAULT_REPO = "mldatauser/model_weights"
HF_SUBDIR = "height_estimation/P1_v1_catt_synoc_dfcval"
DEFAULT_FILE = "epoch=17-val_mae=1.9166.ckpt"          # benchmark checkpoint
LAST_FILE    = "last.ckpt"                             # latest epoch

DEFAULT_DEST = PACKAGE_ROOT / "weights" / "P1_catt_epoch17.ckpt"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO,
                        help="HF repo id (default: %(default)s)")
    parser.add_argument("--subdir", default=HF_SUBDIR,
                        help="Subfolder inside the HF repo "
                             "(default: %(default)s)")
    parser.add_argument("--filename", default=DEFAULT_FILE,
                        help=f"File inside {HF_SUBDIR}/ "
                             f"(default: %(default)s — the published "
                             f"benchmark checkpoint)")
    parser.add_argument("--dest", type=Path, default=None,
                        help="Local destination path "
                             f"(default: weights/<basename of --filename>)")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit(
            "huggingface_hub is not installed. Run "
            "'pip install -r requirements.txt' first."
        )

    path_in_repo = f"{args.subdir.rstrip('/')}/{args.filename}"
    dest = args.dest or (PACKAGE_ROOT / "weights" / Path(args.filename).name)

    if dest.exists():
        log.info("Checkpoint already present at %s — nothing to do.", dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s::%s → %s", args.repo_id, path_in_repo, dest)

    # hf_transfer accelerates large multipart downloads when installed
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # `token` is omitted: huggingface_hub falls back to the cached credentials
    # written by `hf auth login` (~/.cache/huggingface/token). The
    # mldatauser/model_weights repo is public, so anonymous downloads also
    # work for reviewers who haven't logged in.
    cached = hf_hub_download(
        repo_id=args.repo_id,
        filename=path_in_repo,
        local_dir=str(dest.parent),
    )
    cached_path = Path(cached)
    if cached_path.resolve() != dest.resolve():
        # hf_hub_download may have put the file at <local_dir>/<path_in_repo>
        # (i.e. nested under height_estimation/...). Symlink the friendly
        # flat name reviewers expect.
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
