"""Adapter for our CATT (cross-attention CHM-prompted) height model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PACKAGE_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "benchmarking"))

from eval.corruption_sweep import ModelAdapter  # noqa: E402
from model.get_model import get_model  # noqa: E402
from lightning_modules.height_module import HeightEstimationModule  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_CONFIG = str(CODE_ROOT / "configs" / "P1_v1_catt_synoc_dfcval.yaml")


class Adapter(ModelAdapter):
    """Wrap our Dinov3HeightModelDPT (CATT) for the benchmark harness."""

    def __init__(self, config_path: str = DEFAULT_CONFIG):
        self._config_path = config_path
        self._model = None
        self._device = None

    def load(self, checkpoint_path: str, device: str) -> None:
        self._device = torch.device(device)

        with open(self._config_path) as fh:
            yaml_config = yaml.safe_load(fh)

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        ckpt_config = ckpt.get("hyper_parameters", {}).get("config", None)
        build_config = ckpt_config if ckpt_config is not None else yaml_config
        if ckpt_config is None:
            print("[warn] checkpoint has no stored config; falling back to YAML.")

        net = get_model(build_config)
        module = HeightEstimationModule(model=net, config=build_config)
        module.load_state_dict(ckpt["state_dict"], strict=True)
        module.eval().to(self._device)
        self._model = module.model

    def predict(self, image: np.ndarray, chm: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img_t = (
            torch.from_numpy(img)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self._device)
        )

        chm_t = (
            torch.from_numpy(chm)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(self._device)
        )

        with torch.no_grad():
            pred = self._model(img_t, chm_t)

        return pred.squeeze().cpu().numpy()

    @property
    def name(self) -> str:
        return "ours_catt"
