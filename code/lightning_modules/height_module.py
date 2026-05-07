"""
Lightning module for CHM-prompted height estimation.

Handles:
  - 3-element batches ``(image, chm, gt_height)``
  - Regression losses (L1, gradient matching)
  - MAE / RMSE / delta-1 metrics
  - TensorBoard visualisation of RGB | CHM | predicted | GT height
"""

import inspect
import io
import os
import logging
from typing import Dict

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchvision.utils import make_grid

import losses.loss_fun as losses
from utils.schedulers import (
    PolyLRWithWarmup,
    CosineAnnealingLRWithWarmup,
    FinetuneLRScheduler,
    FinetuneDropScheduler,
)

log_level = os.environ.get("LOGLEVEL", "INFO")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(log_level)


class HeightEstimationModule(pl.LightningModule):
    """Lightning module for dense height regression with optional CHM prompt."""

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=["model"])

        self._setup_loss_functions()
        self._setup_stability_regularizers()

        # Auxiliary CHM-reconstruction loss weight (Option B, ablation J).
        # Non-zero only when the model was built with ``aux_chm_recon=True``
        # AND the config carries ``aux_chm_recon_weight > 0``. We also keep
        # the model's own flag as the source of truth for whether the head
        # exists so mismatched configs surface as a clear error rather than
        # a silent no-op.
        self.aux_chm_recon_weight = float(config.get("aux_chm_recon_weight", 0.0))
        # Reconstruction target — autoencoder mode (target = corrupted input
        # CHM, default for backward compatibility with B2 and earlier runs)
        # vs. denoising mode (target = clean CHM = gt nDSM, used by CATT).
        # See :meth:`_compute_aux_chm_recon_loss` for the rationale.
        self.aux_chm_recon_target = str(
            config.get("aux_chm_recon_target", "input")
        )
        if self.aux_chm_recon_target not in ("input", "clean"):
            raise ValueError(
                "aux_chm_recon_target must be 'input' or 'clean', got "
                f"{self.aux_chm_recon_target!r}"
            )
        has_recon_head = getattr(self.model, "chm_recon_head", None) is not None
        if self.aux_chm_recon_weight > 0 and not has_recon_head:
            raise ValueError(
                "aux_chm_recon_weight > 0 but the model has no chm_recon_head. "
                "Set the model-level 'aux_chm_recon: true' in the config."
            )
        if has_recon_head and self.aux_chm_recon_weight == 0:
            logger.warning(
                "Model has aux_chm_recon head but aux_chm_recon_weight is 0 — "
                "the head will run but never contribute gradients."
            )
        self._use_aux_chm_recon = self.aux_chm_recon_weight > 0 and has_recon_head
        if self._use_aux_chm_recon:
            logger.info(
                "Enabled aux_chm_recon: weight=%.5f, target=%s",
                self.aux_chm_recon_weight, self.aux_chm_recon_target,
            )

        self._setup_chm_vicreg()
        self._setup_chm_catt()
        self._setup_chm_counterfactual()
        self._setup_chm_per_layer_probe()

        # Optional output activation. The DPT head is unbounded; when the
        # training target lives in [0, 1] (per-sample min-max normalisation
        # via ``minmax_normalise: true`` on the dataset) the output should
        # be clamped to the same range -- otherwise L1 alone weakly
        # constrains the head and any loss that pulls predictions apart
        # (e.g. counterfactual hinge with margin > 0) can drive the head
        # output to ±∞ at no cost. ``"sigmoid"`` is the natural choice in
        # that mode; for metric-units training (ARKitScenes / SynRS3D /
        # Open-Canopy) the default ``None`` keeps the head linear.
        self.output_activation = config.get("output_activation", None)
        if self.output_activation not in (None, "sigmoid"):
            raise ValueError(
                f"output_activation must be None or 'sigmoid', got "
                f"{self.output_activation!r}"
            )
        if self.output_activation == "sigmoid":
            logger.info("Enabled output_activation: sigmoid (predictions in [0, 1])")

        # running accumulators for epoch-level metrics
        self._val_mae_sum = 0.0
        self._val_mse_sum = 0.0
        self._val_delta1_sum = 0.0
        self._val_count = 0

    # ------------------------------------------------------------------
    # Loss setup
    # ------------------------------------------------------------------
    def _setup_loss_functions(self):
        self.loss_fns = []
        for name in self.config["loss"]:
            params = self.config.get(name, {})
            self.loss_fns.append(getattr(losses, name)(**params))
        self.loss_weights = self.config["loss_weights"]
        assert len(self.loss_weights) == len(self.loss_fns)
        # Cache, per-loss, whether the forward accepts an `image=` kwarg.
        # Used in `_compute_loss` to forward the RGB tensor only to losses
        # that need it (e.g. EdgeAwareSmoothnessLoss). Cheap once-at-init
        # introspection; avoids paying signature inspection cost per step.
        self._loss_accepts_image = []
        for fn in self.loss_fns:
            try:
                sig = inspect.signature(fn.forward)
                params = sig.parameters
                accepts = (
                    "image" in params
                    or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                )
            except (TypeError, ValueError):
                accepts = False
            self._loss_accepts_image.append(accepts)
        logger.info(
            f"Losses: {[type(l).__name__ for l in self.loss_fns]} "
            f"(image-aware: {self._loss_accepts_image})"
        )

        # Per-source loss routing (optional). When set, training_step
        # receives a list of sub-batches (one per source) and applies
        # only the designated losses to each sub-batch.
        self._per_source_loss = None
        loss_per_source_cfg = self.config.get("loss_per_source")
        if loss_per_source_cfg is not None:
            self._per_source_loss = {}
            for src_key, src_cfg in loss_per_source_cfg.items():
                src_idx = int(src_key)
                src_loss_names = src_cfg["loss"]
                src_loss_weights = src_cfg["loss_weights"]
                src_fns = []
                src_img_flags = []
                for name in src_loss_names:
                    params = self.config.get(name, {})
                    fn = getattr(losses, name)(**params)
                    src_fns.append(fn)
                    try:
                        sig = inspect.signature(fn.forward)
                        p = sig.parameters
                        accepts = (
                            "image" in p
                            or any(
                                v.kind == inspect.Parameter.VAR_KEYWORD
                                for v in p.values()
                            )
                        )
                    except (TypeError, ValueError):
                        accepts = False
                    src_img_flags.append(accepts)
                self._per_source_loss[src_idx] = {
                    "fns": src_fns,
                    "weights": src_loss_weights,
                    "image_flags": src_img_flags,
                }
            logger.info(
                "Per-source loss routing enabled for %d sources: %s",
                len(self._per_source_loss),
                {k: [type(f).__name__ for f in v["fns"]]
                 for k, v in self._per_source_loss.items()},
            )

    # ------------------------------------------------------------------
    # Stability regularisers (U-series)
    # ------------------------------------------------------------------
    def _setup_stability_regularizers(self):
        """Optionally instantiate the U-series rank-floor / activation
        regularisers from config.

        Both are *off* unless explicitly enabled by config. Off ⇒ no
        cost, no hooks installed (activation reg) and no module
        enumeration (rank reg).

        Config surface:

        .. code-block:: yaml

            rank_floor_reg:
              enabled: true
              target_stable_rank: 64.0
              weight: 1.0e-3
              target_module_filter: "decoder|lidar_prior"

            activation_reg:
              enabled: true
              mode: "norm_target"           # or "inter_layer_lipschitz"
              weight: 1.0e-2
              norm_target: 1.0
              lipschitz_threshold: 0.5
              hook_filter: "\\.layers\\.\\d+\\.ffn$"
        """
        rank_cfg = dict(self.config.get("rank_floor_reg", {}))
        self.rank_reg = None
        if rank_cfg.pop("enabled", False):
            from losses.rank_regularizer import RankFloorRegularizer
            self.rank_reg = RankFloorRegularizer(self.model, **rank_cfg)

        act_cfg = dict(self.config.get("activation_reg", {}))
        self.act_reg = None
        if act_cfg.pop("enabled", False):
            from losses.activation_regularizer import ActivationRegularizer
            self.act_reg = ActivationRegularizer(self.model, **act_cfg)

    # ------------------------------------------------------------------
    # CHM-token contrastive regulariser (VICReg, paper-faithful)
    # ------------------------------------------------------------------
    def _setup_chm_vicreg(self):
        """Optionally instantiate the paper-faithful VICReg loss on CHM
        encoder features.

        Architecture (Bardes et al. 2022, ICLR):

        ``CHM image → chm_encoder → [B, T, D] tokens
                    → mean over T → [B, D]
                    → VICRegExpander (3-layer MLP, BN+ReLU)
                    → [B, D_exp]
                    → VICRegLoss (per-channel batch-axis variance,
                                  raw off-diag covariance, MSE invariance)``

        The expander is the load-bearing component the paper uses to
        whiten and over-complete the embedding space so the per-channel
        variance hinge and the off-diagonal covariance penalty have
        somewhere meaningful to push. Mean-pooling over the token axis
        before the expander matches the paper's image-level
        formulation (one D-dim feature per image).

        The two augmented views ``chm_v1`` and ``chm_v2`` arrive in a
        5-tuple batch ``(image, chm, chm_v1, chm_v2, gt)`` produced by
        the dataset's ``chm_contrastive_corruption`` block.

        Config surface:

        .. code-block:: yaml

            chm_vicreg:
              enabled: true
              expander_dim: 2048      # paper uses 8192; 2048 is enough at our scale
              expander_layers: 3      # paper default
              gamma: 1.0              # paper default (calibrated to BN-stabilised expander)
              lambda_inv: 25.0        # paper default
              lambda_var: 25.0        # paper default
              lambda_cov: 1.0         # paper default
              eps: 1.0e-4             # paper default

        The expander is owned by the Lightning module rather than the
        model so it is naturally discarded when the trained checkpoint
        is later used for inference (matches the paper's "expander is
        thrown away after pretraining" pattern).
        """
        cfg = dict(self.config.get("chm_vicreg", {}))
        self.chm_vicreg = None
        self.chm_vicreg_expander = None
        self._use_chm_vicreg = bool(cfg.pop("enabled", False))
        # Granularity controls the tensor VICReg sees:
        #   "image" (default): mean-pool tokens over T → [B, D] before expander.
        #     Anchors against image-to-image collapse but is structurally blind
        #     to within-image rank collapse (per-image only B samples).
        #   "token": flatten tokens → [B*T, D] → expander → [B*T, D_exp].
        #     Variance/covariance computed over B*T samples; directly attacks
        #     within-image rank collapse and is the matrix the cross-attention
        #     actually queries.
        #   "both": run both, with image-level term scaled by ``image_weight``.
        self._vicreg_granularity = str(cfg.pop("granularity", "image"))
        self._vicreg_image_weight = float(cfg.pop("image_weight", 0.1))
        if self._vicreg_granularity not in ("image", "token", "both"):
            raise ValueError(
                f"chm_vicreg.granularity must be one of 'image', 'token', "
                f"'both', got {self._vicreg_granularity!r}"
            )
        if not self._use_chm_vicreg:
            return

        if not hasattr(self.model, "chm_encoder"):
            raise ValueError(
                "chm_vicreg.enabled=true but the model has no .chm_encoder "
                "attribute. Only CHM-prompted height models support VICReg."
            )
        encoder_dim = getattr(self.model, "embed_dim", None)
        if encoder_dim is None:
            raise ValueError(
                "chm_vicreg.enabled=true requires the model to expose "
                "`embed_dim` (CHM encoder output width)."
            )

        from losses.vicreg_loss import VICRegLoss, VICRegExpander
        expander_dim = int(cfg.pop("expander_dim", 2048))
        expander_layers = int(cfg.pop("expander_layers", 3))
        self.chm_vicreg_expander = VICRegExpander(
            in_dim=int(encoder_dim),
            expander_dim=expander_dim,
            num_layers=expander_layers,
        )
        self.chm_vicreg = VICRegLoss(**cfg)
        logger.info(
            "Enabled chm_vicreg (paper-faithful, granularity=%s, image_weight=%.3f): "
            "encoder_dim=%d, expander_dim=%d, layers=%d, "
            "γ=%.3f, λ_inv=%.3f, λ_var=%.3f, λ_cov=%.3f",
            self._vicreg_granularity,
            self._vicreg_image_weight,
            int(encoder_dim),
            expander_dim,
            expander_layers,
            self.chm_vicreg.gamma,
            self.chm_vicreg.lambda_inv,
            self.chm_vicreg.lambda_var,
            self.chm_vicreg.lambda_cov,
        )

    def _compute_chm_vicreg_loss(
        self,
        chm_v1: torch.Tensor,
        chm_v2: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run encoder + expander on two views and apply VICReg.

        The CHM encoder is called directly (not via ``self.model.forward``)
        for two reasons: (1) we need the pre-PE token bank, not the
        decoder's final feature map; (2) we must bypass the model's
        internal ``chm_dropout`` which would zero one branch's input
        and trivially solve the invariance term.

        Steps per branch (per granularity):

        * ``image``: ``[B, 1, H, W] → encoder [B, T, D] → mean over T
          → [B, D] → expander → [B, D_exp]``. ``N = B`` per call.
        * ``token``: ``[B, 1, H, W] → encoder [B, T, D] → flatten
          → [B*T, D] → expander → [B*T, D_exp]``. ``N = B*T`` — much
          larger pool, lets variance/covariance terms see the per-token
          structure that cross-attention actually queries.
        * ``both``: run both and combine: ``L = L_token + image_weight * L_image``.

        The augmentations on ``chm_v1`` / ``chm_v2`` are alignment-
        preserving (no rotation, no crop, only height-noise / blur /
        scale-jitter) so per-token correspondence between views is
        valid — invariance pulls aligned tokens together.
        """
        z1, _, _ = self.model.chm_encoder(chm_v1)              # [B, T, D]
        z2, _, _ = self.model.chm_encoder(chm_v2)
        B, T, D = z1.shape

        components: Dict[str, torch.Tensor] = {}
        loss_total = torch.zeros((), device=z1.device, dtype=z1.dtype)

        if self._vicreg_granularity in ("token", "both"):
            e1_t = self.chm_vicreg_expander(z1.reshape(B * T, D))   # [B*T, D_exp]
            e2_t = self.chm_vicreg_expander(z2.reshape(B * T, D))
            l_token, c_token = self.chm_vicreg(e1_t, e2_t)
            loss_total = loss_total + l_token
            components.update(
                {f"token_{k}": v for k, v in c_token.items()}
            )
            components["token_total"] = l_token.detach()

        if self._vicreg_granularity in ("image", "both"):
            z1_pooled = z1.mean(dim=1)                              # [B, D]
            z2_pooled = z2.mean(dim=1)
            e1_i = self.chm_vicreg_expander(z1_pooled)              # [B, D_exp]
            e2_i = self.chm_vicreg_expander(z2_pooled)
            l_image, c_image = self.chm_vicreg(e1_i, e2_i)
            weight = (
                self._vicreg_image_weight
                if self._vicreg_granularity == "both"
                else 1.0
            )
            loss_total = loss_total + weight * l_image
            components.update(
                {f"image_{k}": v for k, v in c_image.items()}
            )
            components["image_total"] = l_image.detach()

        # Back-compat: when running a single granularity, also expose the
        # bare {inv, var, cov} keys so downstream TB plots that referenced
        # them keep working.
        if self._vicreg_granularity == "token":
            for k in ("inv", "var", "cov"):
                components[k] = components[f"token_{k}"]
        elif self._vicreg_granularity == "image":
            for k in ("inv", "var", "cov"):
                components[k] = components[f"image_{k}"]

        return loss_total, components

    # ------------------------------------------------------------------
    # CHM-Aligned Token Training (CATT) — bespoke alternative to VICReg
    # ------------------------------------------------------------------
    def _setup_chm_catt(self):
        """Optionally instantiate the CATT loss on CHM encoder features.

        CATT is the in-house replacement for VICReg, designed for the
        small-batch / high-feature-dim regime where VICReg's variance and
        covariance estimators are noisy / rank-deficient. See
        ``losses/catt_loss.py`` and
        ``docs/chm/insights/cross_run.md`` for the design rationale.

        Two terms (no batch statistics, both supervised or scale-invariant):

        * **per-token local CHM regression** — a ``Linear(D, k*k)`` head
          on each token predicts the avg-pooled clean CHM in that token's
          receptive field. This is *the* anti-collapse mechanism: the
          target is per-token-distinct so a constant or a low-rank token
          bank cannot satisfy it.
        * **directional cross-view consistency** — ``1 − cos(z1, z2)`` per
          token between two corruptions of the same source. Cosine-only
          (scale-invariant) so it cannot incentivise the magnitude
          collapse VICReg's MSE invariance produced in B4.

        The optional dense supervised reconstruction (``aux_chm_recon`` with
        ``aux_chm_recon_target: clean``) is the third CATT term but is wired
        through the existing :class:`~model.dinov3_height_model.CHMReconHead`
        machinery — set both ``aux_chm_recon: true`` on the model and
        ``aux_chm_recon_target: clean`` on the LightningModule to enable it.

        Config surface::

            chm_catt:
              enabled: true
              patch_size: 16        # encoder downsampling factor
              sub_patch_k: 1        # 1 → mean-per-token, k>1 → k×k sub-grid
              lambda_local: 1.0     # weight on per-token regression
              lambda_consistency: 1.0  # weight on directional consistency

        The two augmented views ``chm_v1`` and ``chm_v2`` arrive in the
        same 5-tuple batch ``(image, chm, chm_v1, chm_v2, gt)`` that VICReg
        uses; CATT and VICReg can coexist on a single training run if
        desired (both will pull on the same encoder).
        """
        cfg = dict(self.config.get("chm_catt", {}))
        self.chm_catt = None
        self._use_chm_catt = bool(cfg.pop("enabled", False))
        if not self._use_chm_catt:
            return

        if not hasattr(self.model, "chm_encoder"):
            raise ValueError(
                "chm_catt.enabled=true but the model has no .chm_encoder "
                "attribute. Only CHM-prompted height models support CATT."
            )
        encoder_dim = getattr(self.model, "embed_dim", None)
        if encoder_dim is None:
            raise ValueError(
                "chm_catt.enabled=true requires the model to expose "
                "`embed_dim` (CHM encoder output width)."
            )
        patch_size = int(cfg.pop("patch_size", getattr(self.model, "patch_size", 16)))

        from losses.catt_loss import CATTLoss
        self.chm_catt = CATTLoss(
            embed_dim=int(encoder_dim),
            patch_size=patch_size,
            sub_patch_k=int(cfg.pop("sub_patch_k", 1)),
            lambda_local=float(cfg.pop("lambda_local", 1.0)),
            lambda_consistency=float(cfg.pop("lambda_consistency", 1.0)),
        )
        if cfg:
            raise ValueError(f"Unknown chm_catt keys: {sorted(cfg.keys())}")
        logger.info(
            "Enabled chm_catt: encoder_dim=%d, patch_size=%d, sub_patch_k=%d, "
            "λ_local=%.3f, λ_cons=%.3f",
            int(encoder_dim),
            self.chm_catt.patch_size,
            self.chm_catt.sub_patch_k,
            self.chm_catt.lambda_local,
            self.chm_catt.lambda_consistency,
        )

    def _compute_chm_catt_loss(
        self,
        chm_v1: torch.Tensor,
        chm_v2: torch.Tensor,
        chm_clean: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run encoder on two corrupted views and apply CATT.

        The CHM encoder is called directly (not via ``self.model.forward``)
        because we need the pre-PE token bank, not the decoder's final
        feature map, and we must bypass the model's internal
        ``chm_dropout`` which would zero one branch and trivialise the
        consistency term. ``chm_v1`` / ``chm_v2`` are alignment-preserving
        corruptions of the same clean CHM (no rotation / no crop / only
        height-noise / blur / scale-jitter), so per-token correspondence
        between views is preserved.
        """
        z1, _, _ = self.model.chm_encoder(chm_v1)              # [B, T, D]
        z2, _, _ = self.model.chm_encoder(chm_v2)
        return self.chm_catt(z1, z2, chm_clean)

    # ------------------------------------------------------------------
    # CHM counterfactual usage hinge
    # ------------------------------------------------------------------
    def _setup_chm_counterfactual(self):
        """Optionally enable the counterfactual CHM-usage hinge.

        Forces the height prediction to depend on the CHM input by
        running a *second* forward with the CHM channel zeroed and
        penalising the model when the two predictions agree::

            L_use = relu(margin − mean_masked(|pred − pred_zero|))

        The ``pred_zero`` branch is detached so all gradient flows
        through ``pred`` (which is the supervised forward we're
        optimising anyway). This is a domain-tailored counterfactual
        loss in the spirit of PairCFR / CounterCurate, adapted for
        a continuous regression target.

        Config surface::

            chm_counterfactual:
              enabled: true
              margin: 0.5             # metres
              weight: 1.0             # relative scale on the hinge
              every_k_steps: 1        # 2 → halves the extra forward cost
        """
        cfg = dict(self.config.get("chm_counterfactual", {}))
        self._cf_enabled = bool(cfg.pop("enabled", False))
        self._cf_margin = float(cfg.pop("margin", 0.5))
        self._cf_weight = float(cfg.pop("weight", 1.0))
        self._cf_every_k = int(cfg.pop("every_k_steps", 1))
        if self._cf_every_k < 1:
            raise ValueError(
                f"chm_counterfactual.every_k_steps must be >= 1, "
                f"got {self._cf_every_k}"
            )
        if cfg:
            raise ValueError(
                f"Unknown chm_counterfactual keys: {sorted(cfg.keys())}"
            )
        if self._cf_enabled:
            logger.info(
                "Enabled chm_counterfactual: margin=%.3f m, weight=%.3f, "
                "every_k_steps=%d",
                self._cf_margin, self._cf_weight, self._cf_every_k,
            )

    def _compute_chm_counterfactual_loss(
        self,
        pred: torch.Tensor,
        imgs: torch.Tensor,
        chm: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        """One extra forward with CHM=0; hinge if predictions agree.

        ``pred`` is the supervised forward already computed in
        ``training_step`` (gradient stays attached). ``pred_zero`` is the
        counterfactual forward — *detached* so its only role is as a
        constant target the hinge measures distance from.
        """
        # Bypass model-level chm_dropout for the counterfactual branch:
        # we want the encoder to see ``zero CHM`` deterministically so
        # the hinge has a well-defined comparison target. The dropout
        # randomness would otherwise produce a moving target.
        saved = getattr(self.model, "chm_dropout", 0.0)
        try:
            self.model.chm_dropout = 0.0
            with torch.no_grad():
                pred_zero = self.forward(imgs, torch.zeros_like(chm))
        finally:
            self.model.chm_dropout = saved

        diff = (pred - pred_zero.detach()).abs()             # [B, 1, H, W]
        mask = gt > 0
        if mask.any():
            diff_avg = diff[mask].mean()
        else:
            diff_avg = diff.mean()
        return F.relu(self._cf_margin - diff_avg)

    # ------------------------------------------------------------------
    # Per-layer cross-attn CHM-prediction probe (option E)
    # ------------------------------------------------------------------
    def _setup_chm_per_layer_probe(self):
        """Optionally enable the per-layer CHM-prediction probe loss.

        The model exposes per-decoder-layer CHM predictions in
        ``aux["chm_pred_per_layer"]`` (one ``[B, P]`` tensor per layer
        when ``model.chm_pred_per_layer=True``). For each layer we
        compute L1 against the per-image-patch mean CHM target, then
        average across layers and scale by ``weight``.

        Config surface::

            chm_per_layer_probe:
              enabled: true
              weight: 0.01           # per-layer L1 weight (× n_layers)

        ``weight`` is per-layer; total probe contribution to the loss
        is ``weight × n_layers`` after summation.
        """
        cfg = dict(self.config.get("chm_per_layer_probe", {}))
        self._probe_enabled = bool(cfg.pop("enabled", False))
        self._probe_weight = float(cfg.pop("weight", 0.01))
        if cfg:
            raise ValueError(
                f"Unknown chm_per_layer_probe keys: {sorted(cfg.keys())}"
            )
        if self._probe_enabled:
            has_heads = (
                getattr(self.model, "chm_pred_per_layer_enabled", False)
                and getattr(self.model, "chm_pred_heads", None) is not None
            )
            if not has_heads:
                raise ValueError(
                    "chm_per_layer_probe.enabled=true but the model has no "
                    "chm_pred_heads. Set the model-level "
                    "'chm_pred_per_layer: true' in the config."
                )
            logger.info(
                "Enabled chm_per_layer_probe: weight=%.4f per layer",
                self._probe_weight,
            )

    def _compute_chm_per_layer_probe_loss(
        self,
        chm_pred_per_layer: list,
        chm: torch.Tensor,
        chm_was_dropped: torch.Tensor,
    ) -> torch.Tensor:
        """L1 loss between per-layer CHM predictions and per-patch mean CHM.

        Predictions arrive as a list of ``[B, P]`` tensors (one per
        decoder layer); target is the per-image-patch mean CHM, shape
        ``[B, P]``. Samples where CHM was dropped are excluded so the
        probe only fires on real CHM inputs.
        """
        if chm_pred_per_layer is None or len(chm_pred_per_layer) == 0:
            return torch.zeros((), device=chm.device, dtype=chm.dtype)

        valid = ~chm_was_dropped                                       # [B]
        if valid.sum() == 0:
            return torch.zeros((), device=chm.device, dtype=chm.dtype)

        patch = self.model.patch_size
        # [B, 1, H, W] → [B, 1, h_img, w_img] via avg-pool over the patch.
        target = F.avg_pool2d(chm, kernel_size=patch, stride=patch)
        B = chm.shape[0]
        target = target.reshape(B, -1)                                  # [B, P]
        target = target[valid]                                          # [B', P]

        per_layer_losses = []
        for pred in chm_pred_per_layer:
            # ``pred``: [B, P]. Restrict to valid samples and L1.
            per_layer_losses.append(F.l1_loss(pred[valid], target))
        # Sum across layers (matches the ``weight × n_layers`` interpretation
        # in the docstring); mean would dilute it 10×.
        return torch.stack(per_layer_losses).sum()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _activate(self, t):
        """Apply the configured output activation (no-op when disabled).

        Centralised here so every consumer of model outputs (loss,
        validation metrics, counterfactual hinge, aux-recon, image
        panels) sees the *same* post-activation tensor regardless of
        which call path produced it.
        """
        if t is None or self.output_activation is None:
            return t
        if self.output_activation == "sigmoid":
            return torch.sigmoid(t)
        return t

    def forward(self, image, chm=None):
        return self._activate(self.model(image, chm))

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    def _compute_loss(self, pred, target, prefix="", image=None):
        """Sum over the configured loss list with per-loss image plumbing.

        Losses whose ``forward`` accepts an ``image`` kwarg (detected once
        at setup, cached in ``self._loss_accepts_image``) are called with
        ``fn(pred, target, image=image)``; all others receive
        ``fn(pred, target)``. ``image`` may be ``None`` (e.g. legacy
        callers); in that case image-aware losses fall back to their
        plain (pred, target) behaviour or raise -- it is the loss class's
        responsibility to decide.
        """
        total = 0.0
        for w, fn, needs_image in zip(
            self.loss_weights, self.loss_fns, self._loss_accepts_image
        ):
            if needs_image and image is not None:
                l = fn(pred, target, image=image)
            else:
                l = fn(pred, target)
            if prefix:
                self.log(f"{prefix}_{type(fn).__name__}", l,
                         on_step=True, on_epoch=True, sync_dist=True)
            total += w * l
        return total

    def _compute_aux_chm_recon_loss(
        self,
        aux: dict,
        chm: torch.Tensor,
        chm_clean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """L1 reconstruction of the input CHM from CHM-encoder tokens.

        Two modes, selected by ``aux_chm_recon_target`` in the config:

        * ``"input"`` (default, autoencoder mode): the head reconstructs the
          *corrupted* CHM the encoder was actually fed. This pressures the
          encoder to preserve the prompt signal end-to-end (Option B from
          the J-series ablations) — useful when the goal is to keep CHM
          information from being silently dropped, but it does not require
          the encoder to denoise.

        * ``"clean"`` (denoising / CATT mode): the head reconstructs the
          *clean* CHM (= ``gt`` nDSM). This is the dense supervised target
          version of CATT's per-token regression: every pixel of the
          encoder→reconstruction path is graded against ground truth. It
          forces the encoder both to preserve and to denoise the prompt.

        Fully-dropped samples (dataset-level ``full_dropout`` OR the model's
        internal ``chm_dropout``) are excluded — when the input was zeroed
        out on purpose there is nothing meaningful to reconstruct.

        Returns a zero tensor (still on-graph) when nothing is reconstructable
        for this batch so the optimizer step stays well-defined.
        """
        recon = aux.get("chm_recon")
        valid = ~aux["chm_was_dropped"]                       # [B] bool
        if recon is None or valid.sum() == 0:
            return torch.zeros((), device=chm.device, dtype=chm.dtype)

        # Same activation as the main prediction head — when training in
        # normalised [0, 1] mode the recon target also lives in [0, 1] so
        # the head must be bounded the same way to keep the loss honest.
        recon = self._activate(recon)
        recon_v = recon[valid]                                # [B', 1, H, W]
        if self.aux_chm_recon_target == "clean":
            if chm_clean is None:
                raise RuntimeError(
                    "aux_chm_recon_target='clean' but no clean CHM was passed "
                    "to _compute_aux_chm_recon_loss. This is a programming "
                    "error in training_step."
                )
            target_v = chm_clean[valid]
        else:
            target_v = chm[valid]                             # [B', 1, H, W]
        return F.l1_loss(recon_v, target_v)

    # ------------------------------------------------------------------
    # Batch unpacking
    # ------------------------------------------------------------------
    @staticmethod
    def _unpack_batch(batch):
        """Tolerate 3 / 4 / 5 / 6-tuple batches and return a uniform shape.

        The tuple-arity matrix is:

        * ``3``: ``(imgs, chm, gt)`` -- legacy, no contrastive views, no
          per-sample normalisation metadata.
        * ``4``: ``(imgs, chm, gt, meta)`` -- min-max-normalised dataset
          (e.g. :class:`HypersimDepthDataset` with
          ``minmax_normalise=True``); ``meta`` is a ``[B, 2]`` tensor of
          ``(shift, scale)``.
        * ``5``: ``(imgs, chm, chm_v1, chm_v2, gt)`` -- contrastive views
          (VICReg / CATT) without per-sample normalisation.
        * ``6``: ``(imgs, chm, chm_v1, chm_v2, gt, meta)`` -- both
          contrastive views and per-sample normalisation.

        Returns ``(imgs, chm, gt, chm_v1, chm_v2, meta)``, with
        ``chm_v1 / chm_v2 / meta = None`` when not present in the batch.
        """
        n = len(batch)
        if n == 3:
            imgs, chm, gt = batch
            return imgs, chm, gt, None, None, None
        if n == 4:
            imgs, chm, gt, meta = batch
            return imgs, chm, gt, None, None, meta
        if n == 5:
            imgs, chm, v1, v2, gt = batch
            return imgs, chm, gt, v1, v2, None
        if n == 6:
            imgs, chm, v1, v2, gt, meta = batch
            return imgs, chm, gt, v1, v2, meta
        raise ValueError(
            f"Unexpected batch tuple length {n} -- "
            "expected 3, 4 (with meta), 5 (with contrastive views), or "
            "6 (with both)."
        )

    @staticmethod
    def _meta_to_shift_scale(meta: torch.Tensor):
        """Reshape ``meta=[B, 2]`` to broadcastable ``[B, 1, 1, 1]``.

        ``meta[b, 0] = shift``, ``meta[b, 1] = scale``. The reshape lets
        ``pred_metric = pred_norm * scale + shift`` work without manual
        broadcasting at every call site.
        """
        shift = meta[:, 0].view(-1, 1, 1, 1)
        scale = meta[:, 1].view(-1, 1, 1, 1)
        return shift, scale

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        # Per-source split path: batch is a list of sub-batches (one per
        # source). Each sub-batch gets its own forward pass and loss
        # recipe; gradients accumulate into one backward pass.
        # Distinguish from flat collation: flat → batch[0] is a Tensor;
        # split → batch[0] is a list/tuple of Tensors.
        if (
            isinstance(batch, (list, tuple))
            and len(batch) > 0
            and isinstance(batch[0], (list, tuple))
            and self._per_source_loss is not None
        ):
            return self._training_step_per_source(batch, batch_idx)

        # Legacy single-batch path (flat collation or single-source run).
        imgs, chm, gt, chm_v1, chm_v2, meta = self._unpack_batch(batch)
        del meta  # unused in training; kept-shape consistency only

        # Surface a clear error if dataset-vs-module config drift —
        # any feature that needs the two contrastive views (VICReg or
        # CATT) requires the dataset to emit the 5-tuple, and conversely
        # the 5-tuple is wasted compute if no consumer is enabled.
        wants_views = self._use_chm_vicreg or self._use_chm_catt
        if wants_views and chm_v1 is None:
            raise ValueError(
                "chm_vicreg.enabled or chm_catt.enabled is true but the "
                "dataset emitted a 3-tuple. Add chm_contrastive_corruption "
                "to the train_data_loader config."
            )
        if not wants_views and chm_v1 is not None:
            if not getattr(self, "_warned_unused_views", False):
                logger.warning(
                    "Dataset emits contrastive views but no consumer "
                    "(chm_vicreg / chm_catt) is enabled — the auxiliary CHM "
                    "views will be ignored."
                )
                self._warned_unused_views = True

        if self.act_reg is not None:
            self.act_reg.start()

        need_aux = (
            self._use_aux_chm_recon
            or self._probe_enabled
        )
        try:
            if need_aux:
                pred, aux = self.model(imgs, chm, return_aux=True)
                pred = self._activate(pred)
            else:
                pred = self.forward(imgs, chm)
                aux = None
        finally:
            if self.act_reg is not None:
                self.act_reg.stop()

        loss = self._compute_loss(pred, gt, prefix="train", image=imgs)

        if aux is not None and self._use_aux_chm_recon:
            aux_loss = self._compute_aux_chm_recon_loss(aux, chm, chm_clean=gt)
            self.log("train_aux_chm_recon_loss", aux_loss,
                     on_step=True, on_epoch=True, sync_dist=True)
            loss = loss + self.aux_chm_recon_weight * aux_loss

        if aux is not None and self._probe_enabled:
            probe_loss = self._compute_chm_per_layer_probe_loss(
                aux.get("chm_pred_per_layer"),
                chm,
                aux["chm_was_dropped"],
            )
            self.log("train_chm_per_layer_probe_loss", probe_loss,
                     on_step=True, on_epoch=True, sync_dist=True)
            loss = loss + self._probe_weight * probe_loss

        if self._cf_enabled and (batch_idx % self._cf_every_k == 0):
            cf_loss = self._compute_chm_counterfactual_loss(
                pred, imgs, chm, gt,
            )
            self.log("train_chm_counterfactual_loss", cf_loss,
                     on_step=True, on_epoch=True, sync_dist=True)
            loss = loss + self._cf_weight * cf_loss

        if self._use_chm_vicreg:
            vic_loss, vic_components = self._compute_chm_vicreg_loss(chm_v1, chm_v2)
            self.log("train_chm_vicreg_loss", vic_loss,
                     on_step=True, on_epoch=True, sync_dist=True)
            for name, val in vic_components.items():
                self.log(f"train_chm_vicreg_{name}", val,
                         on_step=True, on_epoch=True, sync_dist=True)
            loss = loss + vic_loss

        if self._use_chm_catt:
            catt_loss, catt_components = self._compute_chm_catt_loss(
                chm_v1, chm_v2, chm_clean=gt
            )
            self.log("train_chm_catt_loss", catt_loss,
                     on_step=True, on_epoch=True, sync_dist=True)
            for name, val in catt_components.items():
                self.log(f"train_chm_catt_{name}", val,
                         on_step=True, on_epoch=True, sync_dist=True)
            loss = loss + catt_loss

        if self.rank_reg is not None:
            rank_loss = self.rank_reg.compute_loss()
            self.log("train_rank_floor_loss", rank_loss,
                     on_step=True, on_epoch=True, sync_dist=True)
            loss = loss + rank_loss

        if self.act_reg is not None:
            act_loss = self.act_reg.compute_loss()
            self.log("train_activation_reg_loss", act_loss,
                     on_step=True, on_epoch=True, sync_dist=True)
            loss = loss + act_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        opt = self.trainer.optimizers[0]
        self.log("lr", opt.param_groups[0]["lr"],
                 on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    # Per-source training (split collation)
    # ------------------------------------------------------------------
    def _training_step_per_source(self, batch_list: list, batch_idx: int):
        """Forward each source sub-batch with its own loss recipe.

        Called when ``per_source_collate_split`` is the collation function
        AND ``loss_per_source`` is configured. Each sub-batch gets an
        independent forward pass; the resulting losses are summed into a
        single scalar for one combined backward pass.
        """
        total_loss = torch.tensor(0.0, device=self.device)

        for src_idx, sub_batch in enumerate(batch_list):
            imgs, chm, gt, chm_v1, chm_v2, meta = self._unpack_batch(sub_batch)

            src_cfg = self._per_source_loss.get(src_idx)
            if src_cfg is None:
                # Source not in loss_per_source → use the default loss list
                pred = self.forward(imgs, chm)
                src_loss = self._compute_loss(
                    pred, gt, prefix=f"train_s{src_idx}", image=imgs
                )
            else:
                pred = self.forward(imgs, chm)
                src_loss = torch.tensor(0.0, device=self.device)
                for w, fn, needs_image in zip(
                    src_cfg["weights"], src_cfg["fns"], src_cfg["image_flags"]
                ):
                    if needs_image:
                        l = fn(pred, gt, image=imgs)
                    else:
                        l = fn(pred, gt)
                    self.log(
                        f"train_s{src_idx}_{type(fn).__name__}", l,
                        on_step=True, on_epoch=True, sync_dist=True,
                    )
                    src_loss = src_loss + w * l

            self.log(f"train_s{src_idx}_loss", src_loss,
                     on_step=True, on_epoch=True, sync_dist=True)
            total_loss = total_loss + src_loss

        self.log("train_loss", total_loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        opt = self.trainer.optimizers[0]
        self.log("lr", opt.param_groups[0]["lr"],
                 on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return total_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        # Handle per-source split collation: concatenate sub-batches back
        # into a single batch for unified validation metrics.
        if (
            isinstance(batch, (list, tuple))
            and len(batch) > 0
            and isinstance(batch[0], (list, tuple))
        ):
            parts = [self._unpack_batch(sb) for sb in batch]
            imgs = torch.cat([p[0] for p in parts], dim=0)
            chm = torch.cat([p[1] for p in parts], dim=0)
            gt = torch.cat([p[2] for p in parts], dim=0)
            meta_parts = [p[5] for p in parts]
            if all(m is not None for m in meta_parts):
                meta = torch.cat(meta_parts, dim=0)
            else:
                meta = None
        else:
            imgs, chm, gt, _, _, meta = self._unpack_batch(batch)

        with torch.no_grad():
            pred = self.forward(imgs, chm)

        loss = self._compute_loss(pred, gt, prefix="val", image=imgs)
        self.log("val_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        self._accumulate_metrics(pred, gt, meta=meta)

        if batch_idx == 0:
            self._log_visualisations(imgs, chm, pred, gt)

        return {"val_loss": loss}

    def _accumulate_metrics(self, pred, gt, meta=None):
        """Accumulate MAE, MSE, delta-1 across the epoch.

        When ``meta`` is provided (``[B, 2]`` tensor of per-sample
        ``(shift, scale)``), prediction and target are mapped back to
        metres before any metric is computed. This keeps MAE / RMSE /
        δ<1.25 numerically comparable across runs, irrespective of
        whether the dataset trains on metric or per-sample-normalised
        targets.
        """
        if meta is not None:
            shift, scale = self._meta_to_shift_scale(meta)
            pred = pred * scale + shift
            gt = gt * scale + shift

        mask = gt > 0  # evaluate only where there is actual height
        if mask.sum() == 0:
            return

        p = pred[mask]
        g = gt[mask]

        self._val_mae_sum += (p - g).abs().sum().item()
        self._val_mse_sum += ((p - g) ** 2).sum().item()

        ratio = torch.max(p / g.clamp(min=1e-6), g / p.clamp(min=1e-6))
        self._val_delta1_sum += (ratio < 1.25).float().sum().item()
        self._val_count += mask.sum().item()

    def on_validation_epoch_start(self):
        torch.cuda.empty_cache()
        self._val_mae_sum = 0.0
        self._val_mse_sum = 0.0
        self._val_delta1_sum = 0.0
        self._val_count = 0

    def on_validation_epoch_end(self):
        n = max(self._val_count, 1)
        mae = self._val_mae_sum / n
        rmse = (self._val_mse_sum / n) ** 0.5
        delta1 = self._val_delta1_sum / n

        self.log("val_mae", mae, prog_bar=True, sync_dist=True)
        self.log("val_rmse", rmse, prog_bar=True, sync_dist=True)
        self.log("val_delta1", delta1, prog_bar=True, sync_dist=True)

        if self.trainer.global_rank == 0:
            print(f"\nEpoch {self.current_epoch}: "
                  f"MAE={mae:.3f}  RMSE={rmse:.3f}  δ<1.25={delta1:.3f}",
                  flush=True)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def _log_visualisations(self, imgs, chm, pred, gt, max_images=4):
        n = min(max_images, imgs.size(0))
        rows = []
        for i in range(n):
            rgb = imgs[i]  # [3, H, W]
            # normalise each single-channel map to [0, 1] for display
            chm_vis = self._normalize_for_vis(chm[i, 0])
            pred_vis = self._normalize_for_vis(pred[i, 0].detach())
            gt_vis = self._normalize_for_vis(gt[i, 0])

            # stack to 3-channel for grid
            row = torch.cat([
                rgb.clamp(0, 1),
                chm_vis.unsqueeze(0).expand(3, -1, -1),
                pred_vis.unsqueeze(0).expand(3, -1, -1),
                gt_vis.unsqueeze(0).expand(3, -1, -1),
            ], dim=2)  # [3, H, 4*W]
            rows.append(row)

        grid = make_grid(torch.stack(rows), nrow=1, normalize=False)
        self.logger.experiment.add_image("val_rgb_chm_pred_gt", grid, self.global_step)

    @staticmethod
    def _normalize_for_vis(t: torch.Tensor) -> torch.Tensor:
        lo, hi = t.min(), t.max()
        if hi - lo < 1e-6:
            return torch.zeros_like(t)
        return (t - lo) / (hi - lo)

    # ------------------------------------------------------------------
    # MLflow image logging hook (called by MLflowImageLoggingCallback)
    # ------------------------------------------------------------------
    def mlflow_validation_images(
        self, batch, batch_idx: int, max_samples: int = 4
    ) -> Dict[str, Image.Image]:
        """Produce PIL images for MLflow from a validation batch.

        Rotates which validation batch is logged across epochs: epoch ``E``
        logs batch ``E % num_val_batches``. Returns an empty dict for every
        other batch so the callback's cadence doesn't flood MLflow with
        near-duplicate images (one panel per validation pass). Over enough
        epochs you eventually see every val sample. Relies on
        ``mlflow_image_log_freq: 1`` in the config — a larger cadence would
        let the callback reject the target batch before this hook ever sees
        it.

        The returned panel shows, per sample: RGB input, CHM prompt,
        predicted height, ground-truth height and the absolute error — all
        with a shared colour scale so the human eye can compare at a glance.
        """
        n_val = (self.trainer.num_val_batches or [1])[0]
        target_batch = self.trainer.current_epoch % max(1, int(n_val))
        if batch_idx != target_batch:
            return {}

        # Tolerate per-source split-collation: concatenate sub-batches
        # so the panel is a single mosaic across sources, matching the
        # layout used by ``validation_step`` for metric reduction.
        if (
            isinstance(batch, (list, tuple))
            and len(batch) > 0
            and isinstance(batch[0], (list, tuple))
        ):
            parts = [self._unpack_batch(sb) for sb in batch]
            imgs = torch.cat([p[0] for p in parts], dim=0)
            chm = torch.cat([p[1] for p in parts], dim=0)
            gt = torch.cat([p[2] for p in parts], dim=0)
            meta_parts = [p[5] for p in parts]
            meta = (
                torch.cat(meta_parts, dim=0)
                if all(m is not None for m in meta_parts)
                else None
            )
        else:
            # Tolerate all tuple arities: use _unpack_batch to normalise.
            imgs, chm, gt, _, _, meta = self._unpack_batch(batch)
        imgs = imgs[:max_samples]
        chm = chm[:max_samples]
        gt = gt[:max_samples]
        if meta is not None:
            meta = meta[:max_samples]

        with torch.no_grad():
            pred = self.forward(imgs, chm)

        # When training in per-sample normalised mode, un-apply the
        # affine ``(shift, scale)`` so the rendered colour-bar reads in
        # metres rather than dimensionless [0, 1] units. Without this
        # the panel says "max height 32 m" whenever the prediction
        # autoscales to its own dynamic range.
        if meta is not None:
            shift, scale = self._meta_to_shift_scale(meta)
            pred = pred * scale + shift
            gt = gt * scale + shift
            chm = chm * scale + shift

        panel = self._build_height_panel(imgs, chm, pred, gt)
        return {"rgb_chm_pred_gt_error": panel}

    # ImageNet normalisation stats applied by the dataloader. Visual
    # outputs need to undo this before display, otherwise RGB values are
    # in roughly [-2.1, 2.6] and a naive clip-to-[0,1] yields a heavily
    # saturated image where most pixels collapse to pure black or white.
    _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @staticmethod
    def _build_height_panel(imgs, chm, pred, gt) -> Image.Image:
        """Matplotlib panel: one row per sample, 5 columns (RGB/CHM/Pred/GT/Err)."""
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        n = imgs.size(0)

        imgs_np = imgs.detach().float().cpu().numpy()
        # De-normalise: the dataloader applies the standard ImageNet
        # transform ``x = (x/255 - mean) / std`` so the tensor entering
        # the model lives in roughly ``[-2.1, 2.6]``. Reverse it before
        # rendering — otherwise a plain ``clip(0, 1)`` on the normalised
        # values throws away most of the colour information.
        mean = HeightEstimationModule._IMAGENET_MEAN.reshape(1, 3, 1, 1)
        std = HeightEstimationModule._IMAGENET_STD.reshape(1, 3, 1, 1)
        imgs_np = imgs_np * std + mean
        chm_np = chm.detach().float().cpu().numpy()[:, 0]
        pred_np = pred.detach().float().cpu().numpy()[:, 0]
        gt_np = gt.detach().float().cpu().numpy()[:, 0]

        chm_np = np.clip(chm_np, 0, None)
        pred_np = np.clip(pred_np, 0, None)
        gt_np = np.clip(gt_np, 0, None)

        # Shared height scale across all samples and maps for an honest visual.
        height_vmax = max(float(gt_np.max()), float(pred_np.max()), 1.0)
        err_vmax = max(0.3 * height_vmax, 1.0)

        fig, axes = plt.subplots(n, 5, figsize=(15, 3 * n), squeeze=False)

        col_titles = ["RGB", "CHM prompt", "Pred", "GT", "|Pred − GT|"]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=11, fontweight="bold")

        last_pred_im = None
        last_err_im = None

        for i in range(n):
            rgb = imgs_np[i].transpose(1, 2, 0)
            rgb = np.clip(rgb, 0, 1)

            err = np.abs(pred_np[i] - gt_np[i])
            mask = gt_np[i] > 0
            mae = float(err[mask].mean()) if mask.any() else 0.0

            axes[i, 0].imshow(rgb)
            axes[i, 1].imshow(chm_np[i], cmap="viridis", vmin=0, vmax=height_vmax)
            last_pred_im = axes[i, 2].imshow(
                pred_np[i], cmap="viridis", vmin=0, vmax=height_vmax
            )
            axes[i, 3].imshow(gt_np[i], cmap="viridis", vmin=0, vmax=height_vmax)
            last_err_im = axes[i, 4].imshow(err, cmap="hot", vmin=0, vmax=err_vmax)

            axes[i, 0].set_ylabel(
                f"MAE={mae:.2f} m", rotation=0, ha="right",
                va="center", labelpad=45, fontsize=10,
            )
            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])

        # shared colour bars along the bottom
        fig.subplots_adjust(right=0.92)
        height_cbar_ax = fig.add_axes([0.93, 0.35, 0.012, 0.3])
        fig.colorbar(last_pred_im, cax=height_cbar_ax, label="height (m)")
        err_cbar_ax = fig.add_axes([0.965, 0.35, 0.012, 0.3])
        fig.colorbar(last_err_im, cax=err_cbar_ax, label="|err| (m)")

        fig.suptitle(
            f"val · rank {0} · max height {height_vmax:.1f} m",
            fontsize=12, fontweight="bold", y=0.995,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB").copy()

    # ------------------------------------------------------------------
    # Optimiser / scheduler (mirrors SegmentationModule patterns)
    # ------------------------------------------------------------------
    def _build_muon_optimizer(self, optimizer_params: Dict):
        """Construct a Muon + AdamW hybrid optimiser.

        Splits parameters into three groups using
        :func:`optim.muon.split_params_for_muon`:

        * 2-D matrices (Q, K, V, out_proj, FFN.fc{1,2}, DPT linears) →
          Muon update with spectral-bounded steps.
        * 3-D / 4-D weights (conv kernels in CHM prompt encoder, DPT
          fusion convs) → AdamW with weight decay.
        * 1-D parameters and the prior bank → AdamW without weight
          decay (matches the global default-AdamW path's policy: bias /
          LN γ-β / σReparam γ / gate α / prior bank are never decayed).

        Config surface (everything optional, sensible defaults applied):

        .. code-block:: yaml

            optimizer: "Muon"
            optimizer_params:
              # Muon group
              lr: 0.02              # spectral-norm of each Muon step
              momentum: 0.95        # SGD momentum (Nesterov)
              ns_steps: 5
              # Embedded AdamW group
              adamw_lr: 1.0e-4      # for non-2D params
              adamw_betas: [0.9, 0.95]
              adamw_eps: 1.0e-8
              adamw_weight_decay: 0.01
        """
        from optim.muon import Muon, split_params_for_muon

        groups = split_params_for_muon(self.named_parameters())

        opt_params = dict(optimizer_params)
        muon_lr = float(opt_params.pop("lr", 2e-2))
        momentum = float(opt_params.pop("momentum", 0.95))
        nesterov = bool(opt_params.pop("nesterov", True))
        ns_steps = int(opt_params.pop("ns_steps", 5))
        adamw_lr = float(opt_params.pop("adamw_lr", 1e-4))
        adamw_betas = tuple(opt_params.pop("adamw_betas", (0.9, 0.95)))
        adamw_eps = float(opt_params.pop("adamw_eps", 1e-8))
        adamw_wd = float(opt_params.pop("adamw_weight_decay",
                                        opt_params.pop("weight_decay", 0.01)))
        if opt_params:
            logger.warning(
                f"Unused optimizer_params keys for Muon: {list(opt_params.keys())}"
            )

        n_muon = len(groups["muon"])
        n_decay = len(groups["adamw_decay"])
        n_no_decay = len(groups["adamw_no_decay"])
        logger.info(
            f"Muon optimiser: {n_muon} 2D params (orthogonalised SGD, "
            f"lr={muon_lr}), {n_decay} non-2D w/ decay (lr={adamw_lr}, "
            f"wd={adamw_wd}), {n_no_decay} 1D / prior-bank w/o decay."
        )

        return Muon(
            muon_params=groups["muon"],
            lr=muon_lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_params=groups["adamw_decay"],
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
            adamw_no_decay_params=groups["adamw_no_decay"],
        )

    def configure_optimizers(self):
        optimizer_name = self.config.get("optimizer", "AdamW")
        optimizer_params = dict(self.config.get("optimizer_params", {"lr": 3e-4, "weight_decay": 0.01}))
        optimizer_params.pop("use_parameter_groups", None)

        # Optional per-parameter-group LR / weight-decay routing.
        # If set, each named parameter is bucketed into the FIRST group whose
        # ``match`` keyword list contains a substring of the parameter name;
        # parameters that match no group go into a default ``leftover`` group
        # using the top-level ``optimizer_params.lr`` / ``weight_decay``.
        # 1-D parameters (LayerNorm γ/β, biases, σReparam γ scalars, ...) are
        # forced to weight_decay=0 inside whichever group they land in.
        # See configs/height/production/Hp7_*.yaml for a worked example.
        parameter_groups_cfg = optimizer_params.pop("parameter_groups", None)

        if optimizer_name == "Muon":
            if parameter_groups_cfg is not None:
                raise ValueError(
                    "parameter_groups is only supported for AdamW; got Muon. "
                    "Use optimizer: AdamW when running with grouped LRs."
                )
            optimizer = self._build_muon_optimizer(optimizer_params)
        else:
            wd = optimizer_params.pop("weight_decay", 0.01)

            if parameter_groups_cfg is not None:
                groups_order = list(parameter_groups_cfg.keys())
                buckets = {
                    name: {"decay": [], "no_decay": []} for name in groups_order
                }
                buckets["__leftover__"] = {"decay": [], "no_decay": []}
                group_specs = {
                    name: dict(spec) for name, spec in parameter_groups_cfg.items()
                }
                for name, param in self.named_parameters():
                    if not param.requires_grad:
                        continue
                    no_decay = (
                        param.ndim <= 1
                        or "lidar_prior.prior" in name
                        or "slot_embed" in name
                    )
                    bucket_name = None
                    for gname in groups_order:
                        match_kws = group_specs[gname].get("match", [])
                        if any(kw in name for kw in match_kws):
                            bucket_name = gname
                            break
                    if bucket_name is None:
                        bucket_name = "__leftover__"
                    sub = "no_decay" if no_decay else "decay"
                    buckets[bucket_name][sub].append(param)

                # Build optimizer's param_groups in declaration order.
                param_groups = []
                for gname in groups_order:
                    spec = group_specs[gname]
                    extra = {
                        k: v
                        for k, v in spec.items()
                        if k not in ("match", "weight_decay", "name")
                    }
                    g_wd = spec.get("weight_decay", wd)
                    if buckets[gname]["decay"]:
                        param_groups.append({
                            "params": buckets[gname]["decay"],
                            "weight_decay": g_wd,
                            "name": gname,
                            **extra,
                        })
                    if buckets[gname]["no_decay"]:
                        param_groups.append({
                            "params": buckets[gname]["no_decay"],
                            "weight_decay": 0.0,
                            "name": f"{gname}_no_decay",
                            **extra,
                        })

                if buckets["__leftover__"]["decay"] or buckets["__leftover__"]["no_decay"]:
                    if buckets["__leftover__"]["decay"]:
                        param_groups.append({
                            "params": buckets["__leftover__"]["decay"],
                            "weight_decay": wd,
                            "name": "leftover",
                        })
                    if buckets["__leftover__"]["no_decay"]:
                        param_groups.append({
                            "params": buckets["__leftover__"]["no_decay"],
                            "weight_decay": 0.0,
                            "name": "leftover_no_decay",
                        })
                    leftover_total = sum(
                        len(buckets["__leftover__"][s]) for s in ("decay", "no_decay")
                    )
                    print(
                        f"[configure_optimizers] {leftover_total} trainable params "
                        f"did not match any parameter_group; routed to 'leftover' "
                        f"with lr={optimizer_params.get('lr')}"
                    )

                # Log group sizes for debugging.
                for pg in param_groups:
                    print(
                        f"[configure_optimizers] group '{pg['name']}': "
                        f"{len(pg['params'])} params, lr={pg.get('lr', optimizer_params.get('lr'))}, "
                        f"wd={pg['weight_decay']}"
                    )

                optimizer = AdamW(param_groups, **optimizer_params)
            else:
                decay_params = []
                no_decay_params = []
                for name, param in self.named_parameters():
                    if not param.requires_grad:
                        continue
                    # Skip weight decay for:
                    #   - 1-D params (LayerNorm γ/β, biases, σReparam γ scalars,
                    #     scalar gates, etc.)
                    #   - the prior bank (learned [M, d] memory)
                    #   - the slot identity embedding (DETR-style [M, d] —
                    #     same rationale as the prior bank: WD shrinks the
                    #     symmetry-breaking signal toward zero).
                    if (
                        param.ndim <= 1
                        or "lidar_prior.prior" in name
                        or "slot_embed" in name
                    ):
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)

                optimizer = AdamW(
                    [
                        {"params": decay_params, "weight_decay": wd},
                        {"params": no_decay_params, "weight_decay": 0.0},
                    ],
                    **optimizer_params,
                )

        scheduler_config = self.config.get("scheduler", {"name": "CosineAnnealingLR"})
        scheduler_name = scheduler_config.get("name")

        if scheduler_name == "CosineAnnealingLR":
            sched = CosineAnnealingLR(
                optimizer,
                **scheduler_config.get("params", {}).get("CosineAnnealingLR", {"T_max": 50}),
            )
            lr_dict = {"scheduler": sched, "interval": "epoch"}

        elif scheduler_name == "CosineAnnealingLRWithWarmup":
            total_steps = (
                scheduler_config.get("params", {})
                .get("CosineAnnealingLRWithWarmup", {})
                .get("total_iters", self.trainer.estimated_stepping_batches)
            )
            sched = CosineAnnealingLRWithWarmup(
                optimizer,
                total_iters=total_steps,
                **{
                    k: v
                    for k, v in scheduler_config.get("params", {})
                    .get("CosineAnnealingLRWithWarmup", {})
                    .items()
                    if k != "total_iters"
                },
            )
            lr_dict = {"scheduler": sched, "interval": "step"}

        elif scheduler_name == "CosineAnnealingWarmRestarts":
            sched = CosineAnnealingWarmRestarts(
                optimizer,
                **scheduler_config.get("params", {}).get("CosineAnnealingWarmRestarts", {
                    "T_0": 10,
                    "T_mult": 2,
                    "eta_min": 1e-7,
                }),
            )
            lr_dict = {"scheduler": sched, "interval": "step"}

        elif scheduler_name == "PolyLRWithWarmup":
            total_steps = (
                scheduler_config.get("params", {})
                .get("PolyLRWithWarmup", {})
                .get("total_iters", self.trainer.estimated_stepping_batches)
            )
            sched = PolyLRWithWarmup(
                optimizer,
                total_iters=total_steps,
                **{
                    k: v
                    for k, v in scheduler_config.get("params", {})
                    .get("PolyLRWithWarmup", {})
                    .items()
                    if k != "total_iters"
                },
            )
            lr_dict = {"scheduler": sched, "interval": "step"}

        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        return {"optimizer": optimizer, "lr_scheduler": lr_dict}
