import torch.nn as nn
import torch
from .layers import conv_bn_relu,UpConvSkipBlock
import torch.nn.functional as F
import os
import logging
# Get the logging level from the environment variable
log_level = os.environ.get('LOGLEVEL', 'INFO')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(log_level)
logger.info(f"Logger level: {logger.getEffectiveLevel()}")


class ReassembleLayer(nn.Module):
    """
    1×1 conv (channel squeeze) followed by optional up‑sampling (ConvT or bilinear).
    `scale` = {1, 2, 4} so output spatial size matches the HR‑CH design.
    """
    def __init__(self, in_ch, out_ch, scale):
        super().__init__()
        self.up   = nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1)
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.scale = scale

    def forward(self, x):
        x = self.proj(x)
        if self.scale > 1:
            # HR‑CH uses ConvTranspose2d; bilinear is lighter and artifact‑free
            # x = F.interpolate(x, scale_factor=self.scale,
            #                   mode="bilinear", align_corners=False)
            x = self.up(x)
        return x


class DPTDecoder(nn.Module):
    """
    HR‑CH / DPT‑style decoder.
    Expects encoder feats in order [deepest … shallowest] with channels
    (1024, 512, 256, 128) and strides 32, 16, 8, 4.
    """
    def __init__(self,
                 enc_ch=(1024, 512, 256, 128),
                 fuse_ch=(1024, 512, 256, 128),
                 num_classes=1):
        super().__init__()

        # ❶  re‑assemble (channel squeeze + resize)
        scales = (16,8,4,2)      # make all maps stride‑8 after this stage
        self.reassemble = nn.ModuleList([
            ReassembleLayer(c_in, c_out, sc)
            for c_in, c_out, sc in zip(enc_ch, fuse_ch, scales)
        ])

        # ❷  feature‑fusion chain (DPT style: add → conv → upsample)
        # After reassembly, all 4 maps are stride‑8.
        self.fuse3 = conv_bn_relu(fuse_ch[0], 256)   # start from deepest
        self.fuse2 = conv_bn_relu(256 + fuse_ch[1], 256)
        self.fuse1 = conv_bn_relu(256 + fuse_ch[2], 256)
        self.fuse0 = conv_bn_relu(256 + fuse_ch[3], 256)

        # ❸  task head
        self.head  = nn.Conv2d(256, num_classes, 1)

    def forward(self, feats, out_size):
        # feats: [f5, f4, f3, f2] = strides 32/16/8/4
        feats = list(reversed(feats))
        r_feats = [layer(f) for layer, f in zip(self.reassemble, feats)] # deepest → shallowest
        r5, r4, r3, r2 = r_feats              # all stride‑8 now
        logger.debug(f"Reassemble output shapes: {[f.shape for f in r_feats]}")
        x = self.fuse3(r5)                    # deepest
        logger.debug(f"fuse3 output shape: {x.shape}")
        x = F.interpolate(x, size=r4.shape[-2:], mode="bilinear",
                          align_corners=False)
        x = self.fuse2(torch.cat([x, r4], dim=1))
        logger.debug(f"fuse2 output shape: {x.shape}")

        x = F.interpolate(x, size=r3.shape[-2:], mode="bilinear",
                          align_corners=False)
        x = self.fuse1(torch.cat([x, r3], dim=1))
        logger.debug(f"fuse1 output shape: {x.shape}")
        x = F.interpolate(x, size=r2.shape[-2:], mode="bilinear",
                          align_corners=False)
        x = self.fuse0(torch.cat([x, r2], dim=1))   # still stride‑8

        logger.debug(f"fuse0 output shape: {x.shape}")
        x = self.head(x)
        logger.debug(f"Model head output shape: {x.shape}")
        # final ×8 upsample → original (pre‑pad) size
        logits = F.interpolate(x, size=out_size,
                               mode="bilinear", align_corners=False)
        return logits
    
class UNetDecoderFlex(nn.Module):
    """
    Accepts a 4-tensor list [res5…res2] and returns stride-1 logits.
    `enc_ch` must match the channels that DinoPyramid / DinoFullPyr output.
    """
    def __init__(self, enc_ch=(1024, 512, 256, 128),
                 bottleneck=512, num_classes=1):
        super().__init__()
        c5, c4, c3, c2 = enc_ch
        self.center = nn.Sequential(
            conv_bn_relu(c5, bottleneck),
            conv_bn_relu(bottleneck, bottleneck),
        )
        self.up1   = UpConvSkipBlock(bottleneck,   c4, bottleneck // 2)   # ->1/16
        self.up2   = UpConvSkipBlock(bottleneck // 2, c3, bottleneck // 4)# ->1/8
        self.up3   = UpConvSkipBlock(bottleneck // 4, c2, bottleneck // 8)# ->1/4
        self.head  = nn.Sequential(
            nn.ConvTranspose2d(bottleneck // 8, bottleneck // 8, 4, stride=2, padding=1),
            conv_bn_relu(bottleneck // 8, bottleneck // 8),
            nn.Conv2d(bottleneck // 8, num_classes, 1),
        )

    def forward(self, feats, out_size):
        feats = list(reversed(feats))
        f5, f4, f3, f2 = feats                     # deepest → shallowest
        logger.debug(f"Input shape: {f5.shape}")
        x  = self.center(f5)
        logger.debug(f"Input shape: {x.shape}, skip shape: {f4.shape}")
        x  = self.up1(x, f4)
        logger.debug(f"Input shape: {x.shape}, skip shape: {f3.shape}")
        x  = self.up2(x, f3)
        logger.debug(f"Input shape: {x.shape}, skip shape: {f2.shape}")
        x  = self.up3(x, f2)                       # stride-4
        x  = self.head(x)
        logger.debug(f"Model head output shape: {x.shape}")
        x  = F.interpolate(x, size=out_size, mode="bilinear",
                           align_corners=False)    # back to pre-pad size
        return x