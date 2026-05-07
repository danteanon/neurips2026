"""Adapter for Open-Canopy PVTv2 (monocular canopy height, no depth prompt).

The pretrained checkpoint uses PVTv2-B3 with 4-channel input in the native
SPOT 6/7 band order: **B, G, R, NIR**.  The model was trained with
mean=0, std=1 (raw float32 pixel values in [0, 255]), despite the YAML
listing mean/std=124 — the GEODataModule hard-codes mean=0, std=1.

When the input image carries an ``_nir`` attribute (set by the MSI loader),
the real NIR band is used.  Otherwise the red channel is duplicated as a
synthetic NIR stand-in.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
OPENCANOPY_ROOT = PACKAGE_ROOT / "benchmarking" / "repos" / "Open-Canopy"
sys.path.insert(0, str(PACKAGE_ROOT / "benchmarking"))

from eval.corruption_sweep import ModelAdapter  # noqa: E402

log = logging.getLogger(__name__)

OC_MEAN = np.float32(0.0)
OC_STD = np.float32(1.0)


class _OpenCanopyNet(nn.Module):
    """PVTv2-B3 backbone + ConvTranspose2d seg_head matching the Open-Canopy
    checkpoint layout."""

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "pvt_v2_b3", pretrained=False, in_chans=4,
        )
        self.seg_head = nn.ConvTranspose2d(512, 1, 32, stride=32)

    def forward(self, x):
        feat = self.backbone.patch_embed(x)
        for stage in self.backbone.stages:
            feat = stage(feat)
        out = self.seg_head(feat)
        return out


class Adapter(ModelAdapter):

    def __init__(self):
        self._model = None
        self._device = None

    def load(self, checkpoint_path: str, device: str) -> None:
        self._device = torch.device(device)
        net = _OpenCanopyNet()

        for p in (str(OPENCANOPY_ROOT), str(OPENCANOPY_ROOT / "src")):
            if p not in sys.path:
                sys.path.insert(0, p)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)

        backbone_sd = {}
        seg_head_sd = {}
        for k, v in sd.items():
            if k.startswith("net.model."):
                backbone_sd[k[len("net.model."):]] = v
            elif k.startswith("net.seg_head.layers.0."):
                seg_head_sd[k[len("net.seg_head.layers.0."):]] = v

        result_bb = net.backbone.load_state_dict(backbone_sd, strict=False)
        result_sh = net.seg_head.load_state_dict(seg_head_sd, strict=True)

        if result_bb.missing_keys:
            log.warning("OC backbone: %d missing keys", len(result_bb.missing_keys))
        if result_bb.unexpected_keys:
            log.warning("OC backbone: %d unexpected keys", len(result_bb.unexpected_keys))

        net.eval().to(self._device)
        self._model = net
        log.info("Open-Canopy PVTv2 loaded on %s", device)

    def predict(self, image: np.ndarray, chm: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        img = image.astype(np.float32)  # (H, W, 3) in RGB order

        nir = getattr(image, '_nir', None)
        if nir is not None:
            nir_f = nir.astype(np.float32)
            if nir_f.ndim == 2:
                nir_f = nir_f[:, :, np.newaxis]
        else:
            nir_f = img[:, :, 0:1]

        # Model expects BGRNIR order (native SPOT band layout).
        # Input image is RGB, so reorder to B, G, R then append NIR.
        bgr = img[:, :, ::-1].copy()  # RGB -> BGR
        img4 = np.concatenate([bgr, nir_f], axis=2)  # (H, W, 4) = BGRNIR
        img4 = (img4 - OC_MEAN) / OC_STD

        img_t = (
            torch.from_numpy(img4)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self._device)
        )

        with torch.no_grad():
            pred = self._model(img_t)

        pred = torch.relu(pred)
        pred = F.interpolate(
            pred, size=(h, w), mode="bilinear", align_corners=False,
        )
        return pred.squeeze().cpu().numpy()

    @property
    def name(self) -> str:
        return "opencanopy_pvtv2"

    @property
    def accepts_chm(self) -> bool:
        return False
