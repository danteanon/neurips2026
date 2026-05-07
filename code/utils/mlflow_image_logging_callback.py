"""MLflow image logging callback.

Supports two logging paths:

1. **Task-specific hook (preferred).** If the Lightning module exposes a
   ``mlflow_validation_images(batch, batch_idx)`` method returning a dict of
   ``{name: PIL.Image}``, the callback calls it and logs every returned image
   under ``images/{stage}_epoch_{E}_batch_{B}_{name}.png``. The module owns
   what to visualize (e.g. RGB | CHM | Pred | GT for height estimation, or a
   segmentation overlay). The callback only owns the cadence.

2. **Legacy segmentation fallback.** For segmentation modules that predate the
   hook, batches must be ``(images, labels)``, the module must accept
   ``pl_module(images)`` and expose ``generate_images(images, labels, logits)``
   returning two grids. This path is preserved for backward compatibility.
"""

from typing import Any, Dict

import lightning as pl
import mlflow
import torch
import torchvision
from PIL import Image


class MLflowImageLoggingCallback(pl.Callback):
    def __init__(self, log_freq: int = 100):
        super().__init__()
        self.log_freq = log_freq

    def _should_log(self, trainer: pl.Trainer, batch_idx: int) -> bool:
        dynamic_offset = trainer.current_epoch % self.log_freq
        return (batch_idx + dynamic_offset) % self.log_freq == 0

    def _log_images(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        stage: str,
    ) -> None:
        if not self._should_log(trainer, batch_idx):
            return
        if trainer.global_rank != 0:
            return

        # --- Preferred path: task-specific hook owns the visualization ---
        hook = getattr(pl_module, "mlflow_validation_images", None)
        if callable(hook):
            with torch.no_grad():
                images: Dict[str, Image.Image] = hook(batch, batch_idx)
            if not images:
                return
            for name, pil_img in images.items():
                mlflow.log_image(
                    pil_img,
                    f"images/{stage}_epoch_{trainer.current_epoch}"
                    f"_batch_{batch_idx}_{name}.png",
                )
            return

        # --- Legacy segmentation fallback ---
        try:
            images_tensor, labels = batch
        except (TypeError, ValueError):
            # Batch shape not (images, labels) and module has no hook — nothing to log.
            return

        with torch.no_grad():
            outputs = pl_module(images_tensor)

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs

        if not hasattr(pl_module, "generate_images"):
            return

        grid1, grid2 = pl_module.generate_images(
            images_tensor[:8], labels[:8], logits[:8]
        )
        mlflow.log_image(
            torchvision.transforms.ToPILImage()(grid1),
            f"images/{stage}_epoch_{trainer.current_epoch}_batch_{batch_idx}_images.png",
        )
        mlflow.log_image(
            torchvision.transforms.ToPILImage()(grid2),
            f"images/{stage}_epoch_{trainer.current_epoch}_batch_{batch_idx}_overlay.png",
        )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._log_images(trainer, pl_module, batch, batch_idx, "val")
