import os
import lightning as pl
from torch.utils.data import DataLoader
import logging
import data_loaders
from utils.metrics import compute_classes_weights

log_level = os.environ.get('LOGLEVEL', 'INFO')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(log_level)


class SegmentationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for all dataset types.

    Config must contain `train_data_loader` and `val_data_loader` sections,
    each fully self-contained:

        train_data_loader:
          class: "SegmentationDatasetFolders"
          data_dir: "/data/lulc"
          num_classes: 9
          channels: 3
          normalisation: "90p"
          folders: ["train"]
          transforms:
            augmentations: ["crop", "spatial"]
            image_size: 256

        val_data_loader:
          class: "SegmentationDatasetFolders"
          data_dir: "/data/lulc"
          num_classes: 9
          channels: 3
          normalisation: "90p"
          folders: ["val"]
          transforms:
            augmentations: []
            image_size: 512

        dataloader:
          batch_size: 8
          num_workers: 4
          pin_memory: true
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        dl_cfg = config.get("dataloader", {})
        self.batch_size = dl_cfg.get("batch_size", 8)
        # Val/test share a separate batch size when specified. Useful when val
        # is run at higher resolution than train (e.g. val 512 / train 256)
        # and VRAM caps the usable val batch size.
        self.val_batch_size = dl_cfg.get("val_batch_size", self.batch_size)
        self.num_workers = dl_cfg.get("num_workers", 4)
        self.pin_memory = dl_cfg.get("pin_memory", True)
        self.persistent_workers = dl_cfg.get("persistent_workers", True)
        self.prefetch_factor = dl_cfg.get("prefetch_factor", 2)
        if self.val_batch_size != self.batch_size:
            logger.info(
                f"Using separate val/test batch size: train={self.batch_size}, "
                f"val/test={self.val_batch_size}"
            )

        self.train_dataset = None
        self.test_dataset = None
        self.class_weights = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = self._create_dataset("train_data_loader")
            self.test_dataset = self._create_dataset("val_data_loader")

            if self.config.get("compute_class_weights", False):
                num_classes = self.config.get("classes", 2)
                logger.info("Computing class weights...")
                self.class_weights = compute_classes_weights(self.train_dataset, num_classes)
                logger.info(f"Class weights: {self.class_weights}")
                self.config["weights"] = self.class_weights

        if (stage == 'test' or stage == 'predict') and self.test_dataset is None:
            self.test_dataset = self._create_dataset("val_data_loader")

    def _create_dataset(self, config_key: str):
        """Instantiate a dataset from a self-contained config section."""
        cfg = dict(self.config[config_key])

        class_name = cfg.pop("class")
        transforms_config = cfg.pop("transforms", {})
        cfg["transforms_config"] = transforms_config

        data_dir = cfg.pop("data_dir")

        dataset_class = getattr(data_loaders, class_name)
        dataset = dataset_class(data_dir, **cfg)

        logger.info(f"Created {class_name} ({len(dataset)} samples, data_dir={data_dir})")
        return dataset

    def _get_collate_fn(self):
        collate_fn_name = self.config.get("dataloader", {}).get("collate_fn")
        if collate_fn_name:
            return getattr(data_loaders, collate_fn_name)
        return None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self._get_collate_fn(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._get_collate_fn(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._get_collate_fn(),
        )
