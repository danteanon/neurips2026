# Lazy imports using __getattr__ to avoid loading heavy dependencies at import time
import importlib
from model.get_model import get_model
from utils.logging_config import setup_logging

def __getattr__(name):
    """
    Lazy loading of model modules to avoid importing heavy dependencies
    until they are actually needed.
    """
    try:
        module = importlib.import_module(f'model.{name}')
        return module
    except ImportError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Expose modules at the package level for direct import
__all__ = [
    "dino_models",
    "satlas_models",
    "get_model",
    "depth_dpt",
    "dinov3_models",
    "dinov3_detection",  # DINOv3 object detection models (DETR + DPT-FCOS)
    "dinov3_height_model",  # CHM-prompted height estimation
    "dinov2_height_model",  # CHM-prompted height estimation on DINOv2 + DA-V2 head (Hp7+)
    "m2f_seg",
    "canopy_models",
    "dinov3_sae_model",  # DINOv3 + SAE integrated segmentation models
    "dinov3_sae_topk_model",  # DINOv3 + TopK SAE integrated segmentation models
    "sae",               # Sparse Autoencoder subpackage
]

setup_logging()
