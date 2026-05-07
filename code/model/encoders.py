import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import logging
from .dinov2 import DINOv2

log_level = os.environ.get('LOGLEVEL', 'INFO')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(log_level)
logger.info(f"Logger level: {logger.getEffectiveLevel()}")
logger = logging.getLogger(__name__)

def get_last_k(vit: nn.Module, k: int = 4):
    """
    Return a list of the indices of the last `k` transformer blocks.
    Example: ViT-L has 24 blocks  ->  [20, 21, 22, 23]
    """
    n = len(vit.blocks)            # standard PyTorch ModuleList length
    return list(range(n - k, n))

class DinoPyramid(nn.Module):
    def __init__(self, backbone = 'dinov2_vitl14', out_dims=(128,256,512,1024), strides = [4,8,16,32]):
        super().__init__()
        self.vit = DINOv2(backbone, pretrained=True).eval()
        self.patch = self.vit.patch_embed.patch_size[0]          # 14 or 16
        embed_dim    = self.vit.embed_dim
        self.proj = nn.ModuleList(
            [nn.Conv2d(embed_dim, d, 1) for d in out_dims])
        self.strides = strides
        self.finetuning_enabled = False

    def enable_finetuning(self):
        """Enable finetuning mode - allows gradients to flow through the encoder"""
        self.finetuning_enabled = True
        self.vit.train()  # Set ViT to training mode for finetuning
        
    def disable_finetuning(self):
        """Disable finetuning mode - freezes the encoder"""
        self.finetuning_enabled = False
        self.vit.eval()  # Set ViT to eval mode when frozen

    def forward(self, x):                                   # BCHW (0-1 range)
        B,C,H,W = x.shape
        pad_h = (self.patch - H % self.patch) % self.patch  # pad to mult. of patch
        pad_w = (self.patch - W % self.patch) % self.patch
        if pad_h or pad_w:
            x = F.pad(x, (0,pad_w,0,pad_h))

        if self.finetuning_enabled:
            # Allow gradients to flow when finetuning
            feats = self.vit.forward_features(x)["x_norm_patchtokens"]
        else:
            # Use no_grad when frozen (original behavior)
            with torch.no_grad():
                feats = self.vit.forward_features(x)["x_norm_patchtokens"]
        
        h, w = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        feats = feats.reshape(B, h, w, -1).permute(0,3,1,2)     # BCHW
        
        # strides we want in output order [32,16,8,4]
        
        pyr = []
        for s, conv in zip(self.strides, self.proj):
            resized = F.interpolate(feats, size=(x.shape[-2]//s, x.shape[-1]//s),
                                    mode='bilinear', align_corners=False)
            pyr.append(conv(resized))
        return pyr

class DinoFullPyr(nn.Module):
    """
    ViT-L/H/G → 4-level feature pyramid.
    Uses *all* requested taps, fuses by simple concat.
    Deepest tap becomes stride-32 (‘res5’).
    """
    def __init__(
        self,                      # pre-trained frozen ViT
        backbone = 'dinov2_vitl14',
        n_encoder_blocks: int = 4,   # number of transformer blocks
        strides = (4,8,16,32),        # output strides
        out_dims=(128,256,512,1024),    # res2…res5 channels
    ):
        super().__init__()
        self.vit     = DINOv2(backbone, pretrained=True).eval()
        self.patch   = self.vit.patch_embed.patch_size[0]
        embed_dim    = self.vit.embed_dim
        self.tap_idx  = get_last_k(self.vit, n_encoder_blocks)          #  ← automatic list
        self.strides = strides
        # 4 lateral 1×1 projections
        self.proj = nn.ModuleList([nn.Conv2d(embed_dim, d, 1) for d in out_dims])
    
    @torch.no_grad()
    def forward(self, x):                               # x : B×C×H×W (0-1)
        B,_,H,W = x.shape
        pad_h = (self.patch - H % self.patch) % self.patch
        pad_w = (self.patch - W % self.patch) % self.patch
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        feats = self.vit.get_intermediate_layers(x, self.tap_idx, reshape=True)
        pyr   = []

        # list of output strides that match U-Net order (deep → shallow)
        for i, (feat, proj, s) in enumerate(
                zip(feats, self.proj, self.strides)):
            logger.debug(f"Pyramid {i} shape before reshaping: {feat.shape}")
            logger.debug(f"Resizing to {H // s} x {W // s} with stride {s}, input width {W}, height {H}")
            if s != self.patch:             # deepest grid keeps native size
                # >>> your requested strategy <<<
                feat = F.interpolate(
                    feat,
                    size=(H // s, W // s),
                    mode="bilinear",
                    align_corners=False,
                )         # PyTorch's bilinear interpolation docs :contentReference[oaicite:0]{index=0}

            pyr.append(proj(feat))          # 1×1 conv to 1024/512/256/128

        return pyr      # [stride-32, stride-16, stride-8, stride-4]
    
 