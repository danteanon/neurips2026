"""DINOv2 + Depth-Anything-V2 + cross-attention CHM fusion (Hp7+).

Architecture (top-down)::

    image  ─► Dinov2Backbone (DINOv2 ViT-L / patch=14, frozen)
                └► 4 intermediate taps at [4, 11, 17, 23] → tokens_i [B, P, D]

    chm    ─► CHMPromptEncoder (stride=16) ─► chm_tokens [B, P', D]
                └► bilinear-resample to image grid (H/14 × W/14)
                └► + 2D sin PE                              ─► chm_memory

    ┌─ Parallel-batched refinement (shared weights across the 4 taps) ─┐
    │   stack ──► [4B, P, D]                                           │
    │   repeat chm_memory ──► [4B, P, D]                               │
    │   HeightDecoderStack(stacked, chm_memory_rep, window_mask)       │
    │   split ──► [tokens_i'  for i in 0..3]                           │
    └──────────────────────────────────────────────────────────────────┘

    └► DPTHead (Depth-Anything-V2-pretrained)
       ─► height_map [B, num_classes, H, W]

Why a separate file from ``dinov3_height_model``:
    * The backbone is the DINOv2 ViT-L used by Depth-Anything V2 (patch=14,
      no register tokens, ImageNet-mean RGB), not DINOv3 (patch=16, 4 storage
      tokens, RoPE position embeddings). The two backbones have incompatible
      state-dict keys and different position-embedding schemes.
    * The DPT head is the standard ``DPTHead`` (not ``DPTHeadLinear``) so the
      ``depth_head.*`` weights from ``model_weights/depth_anything_v2_vitl.pth``
      load with a one-to-one shape match.
    * The CHM token grid (H/16) and image token grid (H/14) differ, so cross-
      attention requires a bilinear resample of the CHM tokens before
      decoding. This is a 2-line tweak inside ``forward`` rather than a new
      shared decoder.

CHMPromptEncoder, CHMReconHead, HeightDecoderStack, and the σ-reparam helper
are reused unchanged from :mod:`model.dinov3_height_model`.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov2 import DINOv2
from .depth_dpt import DPTHead
from .dinov3_height_model import (
    CHMPromptEncoder,
    CHMReconHead,
    HeightDecoderStack,
    apply_sigma_reparam_to_all_linears,
    build_2d_sincos_pe,
    get_cached_window_mask,
)


_DA_V2_DEFAULT_PATH = "model_weights/depth_anything_v2_vitl.pth"


class Dinov2Backbone(nn.Module):
    """Frozen DINOv2 ViT-L feature extractor (patch=14).

    Mirrors the public API of :class:`model.dinov3_model.Dinov3Backbone` so
    :class:`Dinov2HeightModelDPT` can stay structurally close to
    :class:`model.dinov3_height_model.Dinov3HeightModelDPT`.

    Inputs MUST be ImageNet-normalised RGB tensors (mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)) — same convention as Depth-Anything V2.

    Note: ``DINOv2(model_name, pretrained=False)`` is used here so we do NOT
    download separate DINOv2 weights at construction time. The DA-V2
    checkpoint already contains the DINOv2 body under the ``pretrained.*``
    prefix; :meth:`Dinov2HeightModelDPT._load_da_v2_pretrained` loads those
    onto ``self.backbone`` directly.
    """

    intermediate_layer_idx = {
        "vitl": [4, 11, 17, 23],
    }

    def __init__(self, backbone_type: str = "vitl"):
        super().__init__()
        if backbone_type != "vitl":
            raise NotImplementedError(
                f"Dinov2Backbone currently only supports 'vitl', got '{backbone_type}'"
            )
        self.backbone_type = backbone_type
        # pretrained=False so DINOv2 doesn't fetch its own DINOv2 weights;
        # we'll load DA-V2 weights for both body and head from a single file.
        self.backbone = DINOv2("dinov2_vitl14", pretrained=False)

    def forward(self, x: torch.Tensor) -> tuple:
        with torch.no_grad():
            out = self.backbone.get_intermediate_layers(
                x,
                n=self.intermediate_layer_idx[self.backbone_type],
                reshape=False,
                return_class_token=False,
                norm=True,
            )
        return out


class Dinov2HeightModelDPT(nn.Module):
    """CHM-prompted height estimation built on a DINOv2 backbone + DA-V2 DPT head.

    Args:
        num_classes: output channel count (1 for height/depth).
        backbone_type: only ``"vitl"`` is currently supported.
        patch_size: DINOv2 patch size (must be 14 to match the backbone).
        load_pretrained_da_v2: if True (default), loads ``pretrained.*`` and
            ``depth_head.*`` keys from ``da_v2_weights_path`` into the
            backbone and DPT head respectively.
        da_v2_weights_path: path to the DA-V2 ViT-L checkpoint file.
        n_layers, n_heads, ffn_ratio, dropout, chm_dropout, aux_chm_recon,
        cross_attn_window, layer_scale_init, cross_attn_layer_scale_init,
        chm_pred_per_layer, decoder_*qk_norm, decoder_*qkv_spectral_norm,
        sigma_reparam_*: identical semantics to
            :class:`Dinov3HeightModelDPT`.
        dpt_features, dpt_out_channels, dpt_use_bn: standard DPTHead config.
            Defaults match Depth-Anything V2 ViT-L's head exactly so the
            checkpoint loads with no shape mismatches.
        reinit_final_classifier: if True (default), re-initialises the final
            1×1 conv (``head.scratch.output_conv2[2]``) with ``std=0.01``.
            REQUIRED when training with a downstream sigmoid (``output_activation
            = "sigmoid"``) on min-max-normalised targets: DA-V2's depth head
            outputs raw metric depth (~0–10 m), and ``sigmoid(depth)`` is
            ≈ 1 everywhere → instant saturation. Re-initing the final 1×1
            so the pre-sigmoid logit ≈ 0 keeps post-sigmoid step-0 output
            ≈ 0.5, which is then trainable.

    Forward:
        ``model(image, chm)`` returns ``[B, num_classes, H, W]``.
        ``model(image, chm, return_intermediates=True)`` and
        ``return_aux=True`` follow the same contract as
        :meth:`Dinov3HeightModelDPT.forward`.
    """

    def __init__(
        self,
        num_classes: int = 1,
        backbone_type: str = "vitl",
        patch_size: int = 14,
        load_pretrained_da_v2: bool = True,
        da_v2_weights_path: str = _DA_V2_DEFAULT_PATH,
        n_layers: int = 10,
        n_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        chm_dropout: float = 0.0,
        aux_chm_recon: bool = False,
        cross_attn_window: int = 3,
        layer_scale_init: Optional[float] = 1.0e-5,
        cross_attn_layer_scale_init: Optional[float] = None,
        chm_pred_per_layer: bool = False,
        decoder_cross_attn_qk_norm: bool = False,
        decoder_cross_attn_qkv_spectral_norm: bool = False,
        decoder_self_attn_qk_norm: bool = False,
        decoder_self_attn_qkv_spectral_norm: bool = False,
        sigma_reparam_learnable: bool = False,
        sigma_reparam_apply_to_all_linears: bool = False,
        sigma_reparam_gamma_init: str = "constant",
        dpt_features: int = 256,
        dpt_out_channels: Optional[list] = None,
        dpt_use_bn: bool = False,
        reinit_final_classifier: bool = True,
    ):
        super().__init__()

        if backbone_type != "vitl":
            raise NotImplementedError(
                f"Only 'vitl' is currently supported, got '{backbone_type}'"
            )
        if patch_size != 14:
            raise ValueError(
                f"DINOv2 ViT-L is patch=14 by construction, got patch_size={patch_size}"
            )

        embed_dim = 1024

        if cross_attn_window < 0:
            raise ValueError(
                f"cross_attn_window must be >= 0, got {cross_attn_window}"
            )
        if layer_scale_init is not None and layer_scale_init <= 0:
            raise ValueError(
                f"layer_scale_init must be None or a positive float, "
                f"got {layer_scale_init}"
            )
        if (
            cross_attn_layer_scale_init is not None
            and cross_attn_layer_scale_init <= 0
        ):
            raise ValueError(
                f"cross_attn_layer_scale_init must be None or a positive "
                f"float, got {cross_attn_layer_scale_init}. "
                f"Use a tiny positive value (e.g. 1e-8) to start the "
                f"cross-attn contribution near-zero while keeping a "
                f"learnable γ."
            )

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.chm_dropout = chm_dropout
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.cross_attn_window = cross_attn_window
        self.layer_scale_init = layer_scale_init
        self.cross_attn_layer_scale_init = cross_attn_layer_scale_init
        self.dpt_features = dpt_features
        self.aux_chm_recon = aux_chm_recon
        self.chm_pred_per_layer_enabled = bool(chm_pred_per_layer)
        self.sigma_reparam_learnable = bool(sigma_reparam_learnable)
        self.sigma_reparam_apply_to_all_linears = bool(
            sigma_reparam_apply_to_all_linears
        )
        self.sigma_reparam_gamma_init = str(sigma_reparam_gamma_init)

        self._window_mask_cache: dict = {}

        # ---- Backbone (DINOv2 ViT-L / patch 14) ----
        self.backbone = Dinov2Backbone(backbone_type=backbone_type)
        self.tap_indices = list(self.backbone.intermediate_layer_idx[backbone_type])

        # ---- CHM prompt encoder (stride 16) + optional recon head ----
        # The encoder produces an (H/16, W/16) grid; we resample to the
        # image grid (H/14, W/14) inside forward() before cross-attn.
        self.chm_encoder = CHMPromptEncoder(
            embed_dim=embed_dim,
            patch_size=16,
        )
        self.chm_recon_head = (
            CHMReconHead(in_dim=embed_dim) if aux_chm_recon else None
        )

        # ---- Cross-attention decoder stack (shared across the 4 taps) ----
        self.decoder = HeightDecoderStack(
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            layer_scale_init=layer_scale_init,
            cross_attn_layer_scale_init=cross_attn_layer_scale_init,
            cross_attn_qk_norm=decoder_cross_attn_qk_norm,
            cross_attn_qkv_spectral_norm=decoder_cross_attn_qkv_spectral_norm,
            self_attn_qk_norm=decoder_self_attn_qk_norm,
            self_attn_qkv_spectral_norm=decoder_self_attn_qkv_spectral_norm,
            qkv_spectral_learnable=self.sigma_reparam_learnable,
            qkv_spectral_gamma_init=self.sigma_reparam_gamma_init,
        )

        if self.sigma_reparam_apply_to_all_linears:
            apply_sigma_reparam_to_all_linears(
                self.decoder,
                learnable=self.sigma_reparam_learnable,
                init=1.0,
                gamma_init_mode=self.sigma_reparam_gamma_init,
                n_power_iterations=1,
            )

        # ---- Per-layer cross-attn CHM-prediction probes (optional) ----
        if self.chm_pred_per_layer_enabled:
            self.chm_pred_heads = nn.ModuleList(
                [nn.Linear(embed_dim, 1) for _ in range(n_layers)]
            )
            for h in self.chm_pred_heads:
                nn.init.zeros_(h.bias)
                nn.init.normal_(h.weight, mean=0.0, std=0.02)
        else:
            self.chm_pred_heads = None

        # ---- DPT head (DA-V2-compatible) ----
        if dpt_out_channels is None:
            dpt_out_channels = [256, 512, 1024, 1024]
        if len(dpt_out_channels) != 4:
            raise ValueError(
                f"dpt_out_channels must have length 4, got {len(dpt_out_channels)}"
            )
        # DPTHead's __init__ produces the standard DA-V2 layout
        # (projects, resize_layers, scratch.layer{1..4}_rn, refinenet{1..4},
        # output_conv1, output_conv2 Sequential). Keys/shapes match
        # depth_head.* in the DA-V2 ViT-L checkpoint exactly.
        self.head = DPTHead(
            in_channels=embed_dim,
            features=dpt_features,
            num_classes=num_classes,
            use_bn=dpt_use_bn,
            out_channels=dpt_out_channels,
            use_clstoken=False,
            use_auxiliary=False,
            patch_size=patch_size,
        )

        # ---- Pretrained weight loading (DA-V2 → backbone + head) ----
        if load_pretrained_da_v2:
            self._load_da_v2_pretrained(da_v2_weights_path)

        # ---- Re-init final classifier so post-sigmoid step-0 ≈ 0.5 ----
        if reinit_final_classifier:
            self._reinit_final_classifier()

        # Freeze backbone (Hp7 directive: DINOv2 encoder stays frozen).
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def _load_da_v2_pretrained(self, weights_path: str) -> None:
        """Split DA-V2 state_dict into backbone (`pretrained.*`) and head
        (`depth_head.*`) shards and load each with shape-strict mode.
        """
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"DA-V2 weights not found at {weights_path}. Place "
                f"depth_anything_v2_vitl.pth in model_weights/ or pass "
                f"da_v2_weights_path explicitly."
            )
        state = torch.load(weights_path, map_location="cpu", weights_only=True)

        backbone_state, head_state = {}, {}
        for k, v in state.items():
            if k.startswith("pretrained."):
                backbone_state[k[len("pretrained."):]] = v
            elif k.startswith("depth_head."):
                head_state[k[len("depth_head."):]] = v

        bb_missing, bb_unexpected = self.backbone.backbone.load_state_dict(
            backbone_state, strict=False,
        )
        hd_missing, hd_unexpected = self.head.load_state_dict(
            head_state, strict=False,
        )

        def _summary(name, missing, unexpected, expected_count):
            loaded = expected_count - len(unexpected)
            print(
                f"DA-V2 init: {name}: matched {loaded}/{expected_count} ckpt keys "
                f"({len(missing)} missing in target, {len(unexpected)} unexpected)"
            )
            if unexpected:
                print(
                    f"  unexpected: {unexpected[:3]}"
                    f"{'...' if len(unexpected) > 3 else ''}"
                )
            if missing:
                print(
                    f"  missing:    {missing[:3]}"
                    f"{'...' if len(missing) > 3 else ''}"
                )

        _summary("backbone", bb_missing, bb_unexpected, len(backbone_state))
        _summary("head", hd_missing, hd_unexpected, len(head_state))

    def _reinit_final_classifier(self) -> None:
        """Re-init the final 1×1 conv in the DPT head with std=0.01.

        ``DPTHead.scratch.output_conv2`` is::

            Sequential([
                [0] Conv2d(128, 32, 3),
                [1] ReLU(True),
                [2] Conv2d(32, num_classes, 1),  ← we re-init this only
                [3] Identity(),
            ])

        Re-initing only [2] keeps DA-V2's pretrained 3×3 conv (output_conv2[0])
        and output_conv1, while neutralising the final classifier so the
        post-sigmoid output starts near 0.5 instead of saturated near 1.
        """
        final = self.head.scratch.output_conv2[2]
        nn.init.normal_(final.weight, mean=0.0, std=0.01)
        if final.bias is not None:
            nn.init.zeros_(final.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        image: torch.Tensor,
        chm: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        return_aux: bool = False,
    ):
        B, _, H, W = image.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input H,W ({H},{W}) must be divisible by "
                f"patch_size={self.patch_size}. Use 224, 252, 448, 504, ..."
            )
        h_img = H // self.patch_size
        w_img = W // self.patch_size
        P = h_img * w_img
        D = self.embed_dim

        feats = self.backbone(image)
        if len(feats) != 4:
            raise RuntimeError(
                f"Expected 4 backbone taps, got {len(feats)}."
            )

        use_chm = chm is not None
        if (
            use_chm
            and self.training
            and torch.rand((), device=chm.device).item() < self.chm_dropout
        ):
            chm = torch.zeros_like(chm)

        chm_was_dropped = (
            (chm.reshape(B, -1) == 0).all(dim=1)
            if use_chm
            else torch.ones(B, dtype=torch.bool, device=image.device)
        )

        chm_recon: Optional[torch.Tensor] = None
        if use_chm:
            chm_tokens, h_chm, w_chm = self.chm_encoder(chm)

            if return_aux and self.chm_recon_head is not None:
                tokens_grid = (
                    chm_tokens.transpose(1, 2).reshape(B, D, h_chm, w_chm)
                )
                chm_recon = self.chm_recon_head(tokens_grid)
                if chm_recon.shape[-2:] != (H, W):
                    chm_recon = F.interpolate(
                        chm_recon, size=(H, W),
                        mode="bilinear", align_corners=False,
                    )

            # Bilinear-resample CHM tokens to the image-grid so the cross-attn
            # window mask built for (h_img, w_img) is valid.
            if (h_chm, w_chm) != (h_img, w_img):
                chm_grid = chm_tokens.transpose(1, 2).reshape(B, D, h_chm, w_chm)
                chm_grid = F.interpolate(
                    chm_grid, size=(h_img, w_img),
                    mode="bilinear", align_corners=False,
                )
                chm_tokens = chm_grid.flatten(2).transpose(1, 2)
                h_chm, w_chm = h_img, w_img

            pe = build_2d_sincos_pe(
                h_chm, w_chm, D,
                device=chm_tokens.device, dtype=chm_tokens.dtype,
            )
            chm_memory = chm_tokens + pe.unsqueeze(0)
        else:
            chm_memory = torch.zeros(
                B, P, D, device=feats[0].device, dtype=feats[0].dtype,
            )

        stacked = torch.cat(feats, dim=0)
        chm_memory_rep = chm_memory.repeat(4, 1, 1)

        cross_attn_mask = get_cached_window_mask(
            self._window_mask_cache, h_img, w_img,
            self.cross_attn_window, stacked.device,
        )

        need_per_layer = (
            self.chm_pred_per_layer_enabled and return_aux and use_chm
        )
        if return_intermediates or need_per_layer:
            refined_stacked, per_layer = self.decoder(
                stacked, chm_memory_rep, cross_attn_mask,
                return_intermediates=True,
            )
        else:
            refined_stacked = self.decoder(
                stacked, chm_memory_rep, cross_attn_mask,
            )
            per_layer = None

        refined_list = list(torch.split(refined_stacked, B, dim=0))

        chm_pred_per_layer: Optional[list] = None
        if need_per_layer and per_layer is not None:
            chm_pred_per_layer = []
            for li, head_li in enumerate(self.chm_pred_heads):
                delta_4b = per_layer[li]["cross_attn_delta"]
                pred_4b = head_li(delta_4b).squeeze(-1)
                pred_taps = pred_4b.reshape(4, B, -1)
                chm_pred_per_layer.append(pred_taps.mean(dim=0))

        out_features = [(t,) for t in refined_list]
        height_map = self.head(out_features, h_img, w_img)

        intermediates = None
        if return_intermediates:
            intermediates = {
                "tap_indices": self.tap_indices,
                "img_tokens_per_tap": feats,
                "chm_memory": chm_memory,
                "cross_attn_mask": cross_attn_mask,
                "refined_tokens_per_tap": refined_list,
                "decoder_per_layer": per_layer,
            }
        aux = (
            {
                "chm_recon": chm_recon,
                "chm_was_dropped": chm_was_dropped,
                "chm_pred_per_layer": chm_pred_per_layer,
            }
            if return_aux
            else None
        )

        if return_intermediates and return_aux:
            return height_map, intermediates, aux
        if return_intermediates:
            return height_map, intermediates
        if return_aux:
            return height_map, aux
        return height_map

    # ------------------------------------------------------------------
    # Keep frozen backbone in eval() across train()/eval() flips
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        # Backbone stays in eval so any internal Dropout / norm running
        # statistics behave deterministically. DINOv2 has no BatchNorm or
        # active dropouts in its default config, so this is mostly a
        # safety net for future config changes.
        self.backbone.eval()
        return self
