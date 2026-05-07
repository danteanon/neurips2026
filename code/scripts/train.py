#!/usr/bin/env python
import os
import json
# with open("../cfg/aws.json", 'r') as f:
#         config_data = json.load(f)
# ACCESS_KEY = config_data['ACCESS_KEY']
# SECRET_KEY = config_data['SECRET_KEY']
# os.environ["AWS_ACCESS_KEY_ID"] = ACCESS_KEY
# os.environ["AWS_SECRET_ACCESS_KEY"] = SECRET_KEY
import sys
import yaml
import argparse
import logging
from pathlib import Path

# Add parent directory to path to allow imports - MOVED TO TOP
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from datetime import datetime


# Import local modules
from model.get_model import get_model
from lightning_modules.model_module import SegmentationModule
from lightning_modules.data_module import SegmentationDataModule
from utils.gpu_utils import get_free_gpu

# Configure logging
log_level = os.environ.get('LOGLEVEL', 'INFO')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train a segmentation model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the configuration file')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU index to use. If not specified, will choose based on memory availability')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint to resume training from')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Run ID for the experiment')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (fast dev run)')
    parser.add_argument('--name', type=str, default=None,
                        help='Human-readable tag (e.g. ablation_id) prepended to the auto-generated '
                             'run_id. Does NOT change the experiment name — that is always taken '
                             'from the config (`name` / `mlflow_experiment` keys).')
    
    return parser.parse_args()

def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def train(config_path=None, run_id=None, gpu=None, resume=None, seed=42, debug=False, name=None):
    """
    Main training function
    Args:
        config_path: str or None
        experiment_name: str or None
        gpu: int or None
        resume: str or None
        seed: int
        debug: bool
        name: str or None
    """
    torch.cuda.empty_cache()
    config = load_config(config_path)
    # `config["name"]` and `config["mlflow_experiment"]` are the stable experiment identity
    # and must never be mutated by CLI args — checkpoint directories and TensorBoard
    # log paths derive from them. `--name` only tags the per-run identifier (`run_id`).
    if run_id is None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{name}_{now}" if name else f"experiment_{now}"
    logger.info(
        f"run_id={run_id} | experiment={config.get('mlflow_experiment', '?')} | "
        f"config_name={config.get('name', '?')} | tag={name or '(none)'}"
    )
    pl.seed_everything(seed)
    model = get_model(config)
    logger.info(f"Initialized {config['model']['name']} model of type {config['model']['type']}")
    data_module = SegmentationDataModule(config)
    
    # Dynamic lightning module loading
    lightning_config = config.get("lightning_module", {})
    lightning_module_name = lightning_config.get("class_name", "SegmentationModule")
    
    logger.info(f"Loading Lightning Module: {lightning_module_name}")

    if lightning_module_name == "ObjectDetectionModule":
        from lightning_modules.detection_module import ObjectDetectionModule
        lightning_module = ObjectDetectionModule(model, config)
    elif lightning_module_name == "MultiHeadedModule":
        from lightning_modules.multiheaded_module import MultiHeadedModule
        lightning_module = MultiHeadedModule(model, config)
    elif lightning_module_name == "FinetuneSegmentationModule":
        from lightning_modules.finetune_module import FinetuneSegmentationModule
        lightning_module = FinetuneSegmentationModule(model, config)
    elif lightning_module_name == "SAESegmentationModule":
        from lightning_modules.sae_module import SAESegmentationModule
        lightning_module = SAESegmentationModule(model, config)
    elif lightning_module_name == "SAEPretrainSegmentationModule":
        from lightning_modules.sae_pretrain_module import SAEPretrainSegmentationModule
        lightning_module = SAEPretrainSegmentationModule(model, config)
    elif lightning_module_name == "TopKSAEPretrainSegmentationModule":
        from lightning_modules.sae_pretrain_module import TopKSAEPretrainSegmentationModule
        lightning_module = TopKSAEPretrainSegmentationModule(model, config)
    elif lightning_module_name == "MultiLayerSAESegmentationModule":
        from lightning_modules.sae_module import MultiLayerSAESegmentationModule
        lightning_module = MultiLayerSAESegmentationModule(model, config)
    elif lightning_module_name == "MultiLayerTopKSAEPretrainModule":
        from lightning_modules.sae_pretrain_module import MultiLayerTopKSAEPretrainModule
        lightning_module = MultiLayerTopKSAEPretrainModule(model, config)
    elif lightning_module_name == "SingleClassFinetuneModule":
        from lightning_modules.single_class_finetune_module import SingleClassFinetuneModule
        lightning_module = SingleClassFinetuneModule(model, config)
    elif lightning_module_name == "FinetuneSAEModule":
        from lightning_modules.finetune_sae_module import FinetuneSAEModule
        lightning_module = FinetuneSAEModule(model, config)
    elif lightning_module_name == "HeightEstimationModule":
        from lightning_modules.height_module import HeightEstimationModule
        lightning_module = HeightEstimationModule(model, config)
    else:
        # Default to SegmentationModule
        lightning_module = SegmentationModule(model, config)
    callbacks = []
    # Detection tasks use mAP; height estimation uses MAE (lower is better); segmentation uses dice
    is_detection = lightning_module_name == "ObjectDetectionModule"
    is_height = lightning_module_name == "HeightEstimationModule"
    if is_detection:
        ckpt_monitor, ckpt_mode = "val_mAP", "max"
        ckpt_filename = "{epoch:02d}-{val_mAP:.4f}"
    elif is_height:
        ckpt_monitor, ckpt_mode = "val_mae", "min"
        ckpt_filename = "{epoch:02d}-{val_mae:.4f}"
    else:
        ckpt_monitor, ckpt_mode = "val_dice", "max"
        ckpt_filename = "{epoch:02d}-{val_dice:.4f}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.get("checkpoint_dir"), config.get("mlflow_experiment", "default"), run_id),
        filename=ckpt_filename,
        monitor=ckpt_monitor,
        mode=ckpt_mode,
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    if config.get("early_stopping", False):
        early_stop_config = config.get("early_stop", {})
        early_stop_callback = EarlyStopping(
            monitor=early_stop_config.get("monitor", "val_dice"),
            min_delta=early_stop_config.get("min_delta", 0.001),
            patience=early_stop_config.get("patience", 10),
            verbose=early_stop_config.get("verbose", True),
            mode=early_stop_config.get("mode", "max")
        )
        callbacks.append(early_stop_callback)
    # NOTE: MLflow image-logging callback was used internally during
    # development; removed from this reviewer package because the bundled
    # training entry point logs only TensorBoard metrics.

    # CHM diagnostics callback — streams Track A probes (P1–P5) to TB under
    # the `diag/` namespace every validation epoch. Auto-attached for height
    # runs; no-ops for any model without a `decoder.layers` attribute. See
    # docs/chm/ablation_plan.md §7.1 for the scalar-key table and
    # kill-a-run-early rules.
    if is_height:
        diag_cfg = config.get("chm_diagnostics", {}) or {}
        if diag_cfg.get("enabled", True):
            from lightning_modules.chm_diagnostics_callback import CHMDiagnosticsCallback
            callbacks.append(CHMDiagnosticsCallback(
                enabled=True,
                diagnostic_batch_size=(
                    None if diag_cfg.get("diagnostic_batch_size") is None
                    else int(diag_cfg.get("diagnostic_batch_size"))
                ),
                log_histograms=bool(diag_cfg.get("log_histograms", True)),
                log_images=bool(diag_cfg.get("log_images", True)),
                image_log_every_n_epochs=int(diag_cfg.get("image_log_every_n_epochs", 5)),
                # Train-time (tier A / tier B) channels. Tier A is a cheap
                # scalar-read pass (gate parameter + cross-attn weight norms
                # + prior bank) every N steps — key for catching fast-moving
                # divergence like the "W_v magnitude trap" (K5/T0 post-mortem)
                # between val epochs. Tier B runs P1-P3 on a cached train
                # batch every M steps. Defaults mirror the values in
                # docs/chm/insights/paradigm_shift_ln_removal.md.
                enable_train_logging=bool(diag_cfg.get("enable_train_logging", True)),
                train_log_tier_a_every_n_steps=int(
                    diag_cfg.get("train_log_tier_a_every_n_steps", 100)
                ),
                train_log_tier_b_every_n_steps=int(
                    diag_cfg.get("train_log_tier_b_every_n_steps", 500)
                ),
            ))

        # Health probes — adds grad-norms, cross-attn weight stats, CHM-zero
        # val probe, and input-sensitivity probe. Independent of the main
        # CHMDiagnosticsCallback; both can run together.
        probes_cfg = config.get("chm_health_probes", {}) or {}
        if probes_cfg.get("enabled", False):
            from lightning_modules.chm_health_probes import CHMHealthProbes
            callbacks.append(CHMHealthProbes(
                enabled=True,
                log_grad_every_n_steps=int(
                    probes_cfg.get("log_grad_every_n_steps", 50)
                ),
                log_attn_every_n_steps=int(
                    probes_cfg.get("log_attn_every_n_steps", 200)
                ),
                input_sens_every_n_epochs=int(
                    probes_cfg.get("input_sens_every_n_epochs", 1)
                ),
                attn_max_layers=int(
                    probes_cfg.get("attn_max_layers", 3)
                ),
                input_sens_image_size=probes_cfg.get(
                    "input_sens_image_size", None
                ),
            ))

    
    # Create output directory path with experiment name
    output_base_dir = config.get("log_dir", "./logs")
    experiment_name = config.get("mlflow_experiment", "experiment")
    output_dir = os.path.join(output_base_dir, experiment_name)
    
    logger_module = TensorBoardLogger(
        save_dir=output_dir,
        name=run_id
    )
    trainer_args = config.get("trainer", {})
    # Set dynamic values if needed
    if not torch.cuda.is_available():
        trainer_args["accelerator"] = "cpu"
        trainer_args["devices"] = None
    elif gpu is not None:
        # User specified a specific GPU via command line
        trainer_args["devices"] = [gpu]
        trainer_args.pop("strategy", None)  # Remove DDP strategy for single GPU
    elif trainer_args.get("devices") is None:
        # No devices specified in config, auto-select single free GPU
        gpu = get_free_gpu()
        trainer_args["devices"] = [gpu]
    # Otherwise, use devices from config (e.g., devices: 2 for multi-GPU)
    trainer = pl.Trainer(
        **trainer_args,
        logger=logger_module,
        callbacks=callbacks,
    )
    # Run training. MLflow integration was removed from this reviewer
    # package; metrics are logged to TensorBoard only (see ``logger_module``
    # above). Set ``--resume`` to continue from a checkpoint.
    if resume:
        logger.info(f"Resuming from checkpoint: {resume}")
        trainer.fit(lightning_module, data_module, ckpt_path=resume)
    else:
        trainer.fit(lightning_module, data_module)

    logger.info(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    logger.info(f"Best validation score:  {checkpoint_callback.best_model_score}")

    # Reload best (or use last) and emit a JIT trace next to the .ckpt so
    # downstream inference can skip the Lightning import path.
    if checkpoint_callback.best_model_path:
        checkpoint = torch.load(checkpoint_callback.best_model_path)
        lightning_module.load_state_dict(checkpoint["state_dict"])
    lightning_module.eval()
    model_to_log = lightning_module.model
    model_to_log.eval()

    channels = config.get("channels", 3)
    dummy_input = torch.randn(1, channels, 512, 512)
    try:
        if hasattr(model_to_log, "set_output_mode"):
            model_to_log.set_output_mode("inference")
        with torch.no_grad():
            traced_model = torch.jit.trace(model_to_log.cpu(), dummy_input.cpu())
        trace_path = os.path.join(
            config.get("checkpoint_dir", "checkpoints"),
            config.get("mlflow_experiment", "default"),
            run_id,
            "traced_model.pt",
        )
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        traced_model.save(trace_path)
        logger.info(f"Saved JIT trace (1, {channels}, 512, 512) → {trace_path}")
    except Exception as e:
        logger.error(f"Error tracing model: {e}")

if __name__ == "__main__":
    args = parse_args()
    train(
        config_path=args.config,
        gpu=args.gpu,
        resume=args.resume,
        seed=args.seed,
        debug=args.debug,
        name=args.name,
        run_id=args.run_id
    )