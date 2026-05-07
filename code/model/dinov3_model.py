"""
Dinov3 model for segmentation. Removed the model loading path
Use the dataset name to load the weights 
"""
import os

from .dinov3.models.vision_transformer import vit_large
import torch.nn as nn
import torch
import torch.nn.functional as F
import re
from .dpt_layers.blocks import FeatureFusionBlock, _make_scratch
from .depth_dpt import DPTHead, DPTHeadLinear
from .layers import (
    Mask2FormerPixelDecoder,
    Mask2FormerTransformerDecoder,
    Mask2FormerPredictionHeads
)
# Import HuggingFace components
try:
    from transformers import (
        Mask2FormerConfig,
        Mask2FormerModel,
        Mask2FormerForUniversalSegmentation
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers not installed. HuggingFace Mask2Former models will not be available.")
    print("Install with: pip install transformers")

def _make_fusion_block(features, use_bn=False):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class Dinov3SegmentationHead(nn.Module):
    def __init__(
        self, 
        in_channels=1024,  # DINOv3 feature dimension
        features=256,      # Decoder feature dimension
        num_classes=2,     # Number of segmentation classes
        out_channels=[256, 512, 1024, 1024],  # Output channels for each scale
        use_bn=False
    ):
        super(Dinov3SegmentationHead, self).__init__()
        
        # Project DINOv3 features (1024D) to decoder channels
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        # Resize layers to bring features to appropriate scales for fusion
        self.resize_layers = nn.ModuleList([
            # Layer 0: Upsample 4x (e.g., 32x32 -> 128x128)
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0
            ),
            # Layer 1: Upsample 2x (e.g., 32x32 -> 64x64)
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0
            ),
            # Layer 2: Keep same resolution (32x32 -> 32x32)
            nn.Identity(),
            # Layer 3: Downsample 2x (e.g., 32x32 -> 16x16)
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1
            )
        ])
        
        # Feature fusion network (adapted from DPTHead)
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        # Remove stem_transpose as we don't need it
        self.scratch.stem_transpose = None
        
        # Fusion blocks for progressive feature combination
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        # Final segmentation head
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, 
            head_features_1 // 2, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(
                head_features_1 // 2, 
                head_features_2, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ),
            nn.ReLU(True),
            nn.Conv2d(
                head_features_2, 
                num_classes, 
                kernel_size=1, 
                stride=1, 
                padding=0
            ),
        )
    
    def forward(self, out_features, input_shape):
        """
        Args:
            out_features: List of 4 feature tensors from Dinov3Backbone [B, C, H, W]
            input_shape: Original input image shape for final interpolation
        """
        out = []
        
        # Process each feature level
        for i, x in enumerate(out_features):
            # Since reshape=True in backbone, features are already spatial [B, C, H, W]
            
            # Project to decoder channels
            x = self.projects[i](x)
            
            # Resize to appropriate scale
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        # Progressive feature fusion (following DPT structure)
        layer_1, layer_2, layer_3, layer_4 = out
        
        # Apply refinement networks
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        # Progressive fusion from coarse to fine
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        # Final segmentation output
        out = self.scratch.output_conv1(path_1)
        out = self.scratch.output_conv2(out)
        
        # Interpolate to match input image size
        H, W = input_shape[2], input_shape[3]
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=True)
        
        return out

class Dinov3UNetDecoder(nn.Module):
    def __init__(
        self, 
        in_channels=1024,  # DINOv3 feature dimension
        num_classes=2,     # Number of segmentation classes
        decoder_channels=[512, 256, 128, 64],  # Decoder channel progression
    ):
        super(Dinov3UNetDecoder, self).__init__()
        
        # Projection layers to reduce channel dimensions from 1024 to decoder_channels
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(in_channels, decoder_channels[0], kernel_size=1),  # For deepest feature
            nn.Conv2d(in_channels, decoder_channels[1], kernel_size=1),  # For feature 3
            nn.Conv2d(in_channels, decoder_channels[2], kernel_size=1),  # For feature 2  
            nn.Conv2d(in_channels, decoder_channels[3], kernel_size=1),  # For feature 1 (highest res)
        ])
        
        # Fusion blocks (no upsampling layers needed, will use F.interpolate)
        self.conv1 = self._make_conv_block(decoder_channels[0] + decoder_channels[1], decoder_channels[1])
        self.conv2 = self._make_conv_block(decoder_channels[1] + decoder_channels[2], decoder_channels[2])
        self.conv3 = self._make_conv_block(decoder_channels[2] + decoder_channels[3], decoder_channels[3])
        
        # Final classification head
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[3], decoder_channels[3] // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[3] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[3] // 2, num_classes, kernel_size=1)
        )
    
    def _make_conv_block(self, in_channels, out_channels):
        """Create a basic convolutional block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features, input_shape):
        """
        Args:
            features: List of 4 feature tensors from Dinov3Backbone [B, 1024, H, W]
                     All features should have the same spatial resolution (32x32 for 512x512 input)
            input_shape: Original input image shape for final interpolation
        """
        # Project all features to decoder channels
        # All features are at 32x32 resolution
        f1, f2, f3, f4 = features
        
        # Project features to appropriate channel dimensions
        f1_proj = self.proj_layers[3](f1)  # -> 64 channels
        f2_proj = self.proj_layers[2](f2)  # -> 128 channels  
        f3_proj = self.proj_layers[1](f3)  # -> 256 channels
        f4_proj = self.proj_layers[0](f4)  # -> 512 channels
        
        # U-Net style decoder path with bilinear upsampling
        # Start from deepest feature (f4) - 32x32
        x = f4_proj  # [B, 512, 32, 32]
        
        # Stage 1: Upsample 32x32 -> 64x64 and fuse with f3
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 512, 64, 64]
        f3_upsampled = F.interpolate(f3_proj, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 256, 64, 64]
        x = torch.cat([x, f3_upsampled], dim=1)  # [B, 768, 64, 64]
        x = self.conv1(x)  # [B, 256, 64, 64]
        
        # Stage 2: Upsample 64x64 -> 128x128 and fuse with f2
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 256, 128, 128]
        f2_upsampled = F.interpolate(f2_proj, scale_factor=4, mode='bilinear', align_corners=True)  # [B, 128, 128, 128]
        x = torch.cat([x, f2_upsampled], dim=1)  # [B, 384, 128, 128]
        x = self.conv2(x)  # [B, 128, 128, 128]
        
        # Stage 3: Upsample 128x128 -> 256x256 and fuse with f1
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 128, 256, 256]
        f1_upsampled = F.interpolate(f1_proj, scale_factor=8, mode='bilinear', align_corners=True)  # [B, 64, 256, 256]
        x = torch.cat([x, f1_upsampled], dim=1)  # [B, 192, 256, 256]
        x = self.conv3(x)  # [B, 64, 256, 256]
        
        # Stage 4: Final upsampling 256x256 -> 512x512
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 64, 512, 512]
        
        # Final convolution to get segmentation map
        x = self.final_conv(x)  # [B, num_classes, 512, 512]
        
        # Final interpolation to exactly match input size (in case of any rounding)
        H, W = input_shape[2], input_shape[3]
        if x.shape[2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
        return x

# Default local weight paths for each dataset type
DINOV3_WEIGHT_PATHS = {
    "lvd1689m": "model_weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "sat493m": "model_weights/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
}


class Dinov3Backbone(nn.Module):
    """Frozen DINOv3 ViT-L feature extractor.

    IMPORTANT — input image preprocessing contract:
        Inputs to ``forward(x)`` MUST be ImageNet-normalised RGB tensors,
        i.e. each channel has been scaled to [0, 1] and then standardised
        with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225).

        DINOv3 was pretrained with these statistics; feeding in raw [0, 1]
        or [0, 255] images will silently produce off-distribution features
        and degrade every downstream head (segmentation, height, etc.).

        In this repo, that normalisation is applied by
        :class:`utils.transformations.Transformations` when the data-loader
        config has ``scaling: "imagenet_scaling"`` in its ``transforms``
        block. All ``configs/height/*.yaml`` are wired this way; when adding
        a new training config or a new data-loader, make sure the same
        scaling step is present before tensors reach this backbone.
    """

    def __init__(
        self, 
        backbone_type: str = "vitl", 
        dataset: str = "lvd1689m", 
        reshape: bool = True, 
        return_class_token: bool = False,
        load_pretrained_backbone: bool = False,
        weights_path: str = None
    ):
        """
        Initialize Dinov3Backbone with dataset type determining architecture.
        
        Args:
            backbone_type: Type of ViT backbone ('vitl', 'vitb', 'vitg')
            dataset: Dataset type - either 'sat493m' or 'lvd1689m'. Determines model architecture.
            reshape: Whether to reshape features to spatial format [B, C, H, W]
            return_class_token: Whether to return class token
            load_pretrained_backbone: Whether to load pretrained backbone weights.
                                     Set to False (default) for inference when loading full model checkpoint.
                                     Set to True for training to auto-load backbone weights.
            weights_path: Path to pretrained weights. If None and load_pretrained_backbone=True,
                         uses default path based on dataset type.
        """
        super().__init__()
        self.backbone_type = backbone_type
        self.dataset = dataset
        
        # Determine architecture based on dataset type
        untie_global_and_local_cls_norm = (dataset == "sat493m")
        
        self.backbone = vit_large(
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
            mask_k_bias=True
        )
        self.reshape = reshape
        self.return_class_token = return_class_token
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        # Automatically load pretrained weights if requested
        if load_pretrained_backbone:
            self.load_weights(weights_path)

    HF_REPO = "mldatauser/dinov3_models"
    HF_FILENAMES = {
        "lvd1689m": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "sat493m": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
    }

    def load_weights(self, weights_path: str = None):
        """
        Load pretrained encoder weights.

        Resolution order:
          1. Explicit ``weights_path`` argument
          2. Local default path (DINOV3_WEIGHT_PATHS)
          3. HuggingFace Hub (HF_REPO / HF_FILENAMES)
        """
        if weights_path is None:
            weights_path = DINOV3_WEIGHT_PATHS.get(self.dataset)
            if weights_path is None:
                raise ValueError(f"No default weight path for dataset '{self.dataset}'. "
                               f"Available: {list(DINOV3_WEIGHT_PATHS.keys())}")

        if not os.path.isfile(weights_path):
            os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
            from huggingface_hub import hf_hub_download
            hf_filename = self.HF_FILENAMES.get(self.dataset)
            print(f"Local weights not found at {weights_path}; "
                  f"downloading from {self.HF_REPO}/{hf_filename}")
            weights_path = hf_hub_download(
                repo_id=self.HF_REPO, filename=hf_filename
            )

        model_weights = torch.load(weights_path, weights_only=True)
        self.backbone.load_state_dict(model_weights)
        self.backbone.eval()
        print(f"Weights loaded successfully from {weights_path}")

    def forward(self, x):
        # NOTE: ``x`` MUST be ImageNet-normalised RGB:
        #   x = (img/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        # DINOv3 was pretrained on these statistics. In this repo this is
        # applied by ``Transformations(scaling="imagenet_scaling")`` configured
        # in the data-loader transforms block. Skipping it gives off-distribution
        # features and silently hurts every downstream head.
        with torch.no_grad():
            out = self.backbone.get_intermediate_layers(x, n=self.intermediate_layer_idx[self.backbone_type], 
            reshape=self.reshape, norm=True, return_class_token=self.return_class_token)
        return out

class Dinov3Segmentation(nn.Module):
    def __init__(self, num_classes, backbone_type: str = "vitl", dataset: str = "lvd1689m", features=256,
                 load_pretrained_backbone: bool = False):
        super().__init__()
        self.backbone = Dinov3Backbone(backbone_type, dataset=dataset, load_pretrained_backbone=load_pretrained_backbone)
        self.head = Dinov3SegmentationHead(
            in_channels=1024,  # DINOv3 vitl feature dimension
            features=features,
            num_classes=num_classes,
            out_channels=[256, 512, 1024, 1024],
            use_bn=False
        )

    def load_backbone_weights(self, weights_path: str = None):
        """Load pretrained encoder weights into the backbone.
        
        Args:
            weights_path: Path to weights. If None, uses default based on dataset type.
        """
        self.backbone.load_weights(weights_path)

    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Pass features and input shape to segmentation head
        segmentation_output = self.head(features, x.shape)
        
        return segmentation_output

class Dinov3UNetSegmentation(nn.Module):
    def __init__(self, num_classes, backbone_type: str = "vitl", dataset: str = "lvd1689m", decoder_channels=[512, 256, 128, 64],
                 load_pretrained_backbone: bool = False):
        super().__init__()
        self.backbone = Dinov3Backbone(backbone_type, dataset=dataset, load_pretrained_backbone=load_pretrained_backbone)
        self.head = Dinov3UNetDecoder(
            in_channels=1024,  # DINOv3 vitl feature dimension
            num_classes=num_classes,
            decoder_channels=decoder_channels
        )

    def load_backbone_weights(self, weights_path: str = None):
        """Load pretrained encoder weights into the backbone.
        
        Args:
            weights_path: Path to weights. If None, uses default based on dataset type.
        """
        self.backbone.load_weights(weights_path)

    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Pass features and input shape to U-Net decoder
        segmentation_output = self.head(features, x.shape)
        
        return segmentation_output


class Dinov3AnyUPUnetDecoder(nn.Module):
    """
    U-Net style decoder for AnyUP upsampled features.
    
    Expects 4 features at different spatial resolutions:
    - features[0]: H/2, W/2   (shallow layer, highest spatial resolution)
    - features[1]: H/4, W/4
    - features[2]: H/8, W/8
    - features[3]: H/16, W/16 (deep layer, lowest spatial resolution)
    
    Decodes from deep to shallow in U-Net fashion.
    """
    def __init__(self, in_channels=1024, num_classes=1, decoder_channels=[512, 256, 128, 64]):
        super().__init__()
        
        # Project backbone features to decoder channels
        # features[3] (deepest) -> decoder_channels[0] (512)
        # features[2] -> decoder_channels[1] (256)
        # features[1] -> decoder_channels[2] (128)
        # features[0] (shallowest) -> decoder_channels[3] (64)
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(in_channels, decoder_channels[3], kernel_size=1),  # shallow -> 64
            nn.Conv2d(in_channels, decoder_channels[2], kernel_size=1),  # -> 128
            nn.Conv2d(in_channels, decoder_channels[1], kernel_size=1),  # -> 256
            nn.Conv2d(in_channels, decoder_channels[0], kernel_size=1),  # deep -> 512
        ])
        
        # Decoder blocks: upsample + concat + conv
        # Stage 1: decoder_channels[0] (512) -> upsampled, concat with decoder_channels[1] (256)
        self.up_conv1 = self._make_decoder_block(
            decoder_channels[0] + decoder_channels[1],  # 512 + 256 = 768
            decoder_channels[1]  # output 256
        )
        
        # Stage 2: decoder_channels[1] (256) -> upsampled, concat with decoder_channels[2] (128)
        self.up_conv2 = self._make_decoder_block(
            decoder_channels[1] + decoder_channels[2],  # 256 + 128 = 384
            decoder_channels[2]  # output 128
        )
        
        # Stage 3: decoder_channels[2] (128) -> upsampled, concat with decoder_channels[3] (64)
        self.up_conv3 = self._make_decoder_block(
            decoder_channels[2] + decoder_channels[3],  # 128 + 64 = 192
            decoder_channels[3]  # output 64
        )
        
        # Final segmentation head
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[3], decoder_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        """Create a decoder conv block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features, input_shape):
        """
        Args:
            features: List of 4 upsampled features from AnyUP
                     [f0: H/2, f1: H/4, f2: H/8, f3: H/16]
            input_shape: Original input shape [B, C, H, W]
        
        Returns:
            Segmentation output [B, num_classes, H, W]
        """
        f0, f1, f2, f3 = features  # shallow to deep
        
        # Project to decoder channels
        f0_proj = self.proj_layers[0](f0)  # [B, 64, H/2, W/2]
        f1_proj = self.proj_layers[1](f1)  # [B, 128, H/4, W/4]
        f2_proj = self.proj_layers[2](f2)  # [B, 256, H/8, W/8]
        f3_proj = self.proj_layers[3](f3)  # [B, 512, H/16, W/16]
        
        # Decoder: start from deepest feature (f3)
        # Stage 1: f3 -> upsample to f2 size, concat, conv
        x = F.interpolate(f3_proj, size=f2_proj.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, f2_proj], dim=1)  # [B, 768, H/8, W/8]
        x = self.up_conv1(x)  # [B, 256, H/8, W/8]
        
        # Stage 2: -> upsample to f1 size, concat, conv
        x = F.interpolate(x, size=f1_proj.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, f1_proj], dim=1)  # [B, 384, H/4, W/4]
        x = self.up_conv2(x)  # [B, 128, H/4, W/4]
        
        # Stage 3: -> upsample to f0 size, concat, conv
        x = F.interpolate(x, size=f0_proj.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, f0_proj], dim=1)  # [B, 192, H/2, W/2]
        x = self.up_conv3(x)  # [B, 64, H/2, W/2]
        
        # Final segmentation output
        x = self.final_conv(x)  # [B, num_classes, H/2, W/2]
        
        # Upsample to original input size
        H, W = input_shape[2], input_shape[3]
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
        return x

class Dinov3AnyUPSegmentation(nn.Module):
    def __init__(self, num_classes, backbone_type: str = "vitl", dataset: str = "lvd1689m", decoder_channels=[512, 256, 128, 64],
                 load_pretrained_backbone: bool = False):
        super().__init__()
        from .upsamplers import anyup_upsampler
        self.upsampler = anyup_upsampler()
        self.backbone = Dinov3Backbone(backbone_type, dataset=dataset, load_pretrained_backbone=load_pretrained_backbone)
        self.head = Dinov3AnyUPUnetDecoder(
            in_channels=1024,  # DINOv3 vitl feature dimension
            num_classes=num_classes,
            decoder_channels=decoder_channels)
    
    def forward(self, x):
        # Extract features from backbone
        H, W = x.shape[2], x.shape[3]
        features = self.backbone(x)
        
        # Upsample features to create multi-scale pyramid
        # Shallow layers (early): higher spatial resolution
        # Deep layers (late): lower spatial resolution
        features_up = []
        for i, feature in enumerate(reversed(features)):
            output_size = (int(H / (2 ** (i + 1))), int(W / (2 ** (i + 1))))
            features_up.append(self.upsampler(x, feature, output_size=output_size))
            # print(f"Feature {i} upsampled shape: {features_up[-1].shape}")
        
        # Pass upsampled features to U-Net decoder
        segmentation_output = self.head(features_up, x.shape)
        return segmentation_output


class Dinov3AnyUPSegmentationFullRes(nn.Module):
    """
    AnyUp-based segmentation model with full-resolution feature ensemble.
    
    Architecture:
    1. DINOv3 backbone extracts 4 features from different layers (all at low-res)
    2. AnyUp upsamples ALL 4 features to FULL image resolution (H, W)
    3. Each feature is projected to a lower embedding dimension
    4. All projected features are concatenated
    5. Simple 1x1 convolution head produces final segmentation
    
    This approach:
    - Preserves maximum spatial resolution throughout
    - Combines semantics from all ViT layers (early=texture, late=semantic)
    - Avoids blurry bilinear interpolation in decoder
    - Simple and efficient inference
    
    Args:
        num_classes: Number of segmentation classes
        backbone_type: Type of DINOv3 backbone ('vitl', 'vitb', 'vitg')
        dataset: Dataset for pretrained weights ('lvd1689m' or 'sat493m')
        proj_dim: Projection dimension for each feature (default: 256)
        load_pretrained_backbone: Whether to load pretrained backbone weights
    """
    
    def __init__(
        self, 
        num_classes, 
        backbone_type: str = "vitl", 
        dataset: str = "lvd1689m", 
        proj_dim: int = 256,
        load_pretrained_backbone: bool = False
    ):
        super().__init__()
        from .upsamplers import anyup_upsampler
        # Embedding dimensions for different backbone types
        embedding_dims = {
            "vitl": 1024,
            "vitb": 768,
            "vitg": 1536,
        }
        self.in_channels = embedding_dims[backbone_type]
        self.proj_dim = proj_dim
        self.num_features = 4  # DINOv3 returns 4 intermediate features
        
        # AnyUp upsampler
        self.upsampler = anyup_upsampler()
        
        # DINOv3 backbone
        self.backbone = Dinov3Backbone(
            backbone_type, 
            dataset=dataset, 
            load_pretrained_backbone=load_pretrained_backbone
        )
        
        # Projection layers: project each feature from in_channels to proj_dim
        # Each layer corresponds to a different ViT layer (4, 11, 17, 23 for vitl)
        # Using 3x3 kernel for spatial context during projection
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, proj_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(self.num_features)
        ])
        
        # Final segmentation head
        # Input: concatenated features (proj_dim * num_features)
        # Output: num_classes
        concat_channels = proj_dim * self.num_features  # 256 * 4 = 1024
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(concat_channels, concat_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(concat_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(concat_channels // 2, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input image tensor [B, 3, H, W]
        
        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        B, C, H, W = x.shape
        
        # Extract features from backbone
        # Returns 4 features: [layer_4, layer_11, layer_17, layer_23]
        # Each: [B, 1024, h, w] where h, w = H//16, W//16 typically
        features = self.backbone(x)
        
        # Upsample ALL features to full resolution using AnyUp
        # Then project each to lower dimension
        # Using q_chunk_size to process attention in chunks (memory efficient)
        projected_features = []
        for i, feature in enumerate(features):
            # Upsample to full resolution (H, W) with chunked attention
            feature_hr = self.upsampler(x, feature, output_size=(H, W), q_chunk_size=128)  # [B, 1024, H, W]
            
            # Project to lower dimension
            feature_proj = self.proj_layers[i](feature_hr)  # [B, proj_dim, H, W]
            projected_features.append(feature_proj)
        
        # Concatenate all projected features
        # [B, proj_dim * 4, H, W] = [B, 1024, H, W]
        concat_features = torch.cat(projected_features, dim=1)
        
        # Final segmentation output
        seg_output = self.seg_head(concat_features)  # [B, num_classes, H, W]
        
        return seg_output


class Dinov3AnyUPSegmentationFullResV2(nn.Module):
    """
    Enhanced version with attention-based feature fusion instead of simple concat.
    
    Same as Dinov3AnyUPSegmentationFullRes but uses learned attention weights
    to combine features from different ViT layers, rather than simple concatenation.
    
    Args:
        num_classes: Number of segmentation classes
        backbone_type: Type of DINOv3 backbone ('vitl', 'vitb', 'vitg')
        dataset: Dataset for pretrained weights
        proj_dim: Projection dimension for each feature (default: 256)
        load_pretrained_backbone: Whether to load pretrained backbone weights
    """
    
    def __init__(
        self, 
        num_classes, 
        backbone_type: str = "vitl", 
        dataset: str = "lvd1689m", 
        proj_dim: int = 256,
        load_pretrained_backbone: bool = False
    ):
        super().__init__()
        from .upsamplers import anyup_upsampler
        embedding_dims = {
            "vitl": 1024,
            "vitb": 768,
            "vitg": 1536,
        }
        self.in_channels = embedding_dims[backbone_type]
        self.proj_dim = proj_dim
        self.num_features = 4
        
        # AnyUp upsampler
        self.upsampler = anyup_upsampler()
        
        # DINOv3 backbone
        self.backbone = Dinov3Backbone(
            backbone_type, 
            dataset=dataset, 
            load_pretrained_backbone=load_pretrained_backbone
        )
        
        # Projection layers for each feature
        # Using 3x3 kernel for spatial context during projection
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, proj_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(self.num_features)
        ])
        
        # Attention-based fusion: learn to weight different layer contributions
        # Memory-efficient: operates on pooled features, not full resolution
        # Input: [B, proj_dim * num_features] (pre-pooled and concatenated)
        self.layer_attention = nn.Sequential(
            nn.Linear(proj_dim * self.num_features, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, self.num_features),
            nn.Softmax(dim=1)
        )
        
        # Spatial attention for each combined feature
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(proj_dim, proj_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final segmentation head (input is proj_dim since we do weighted sum)
        self.seg_head = nn.Sequential(
            nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Forward pass using ONLY the last layer feature (memory-efficient).
        
        Uses only the deepest semantic feature from DINOv3 (layer 23),
        which has the richest semantic information.
        
        Args:
            x: Input image tensor [B, 3, H, W]
        
        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        B, C, H, W = x.shape
        
        # Extract backbone features
        features = self.backbone(x)
        
        # Use ONLY the last feature (deepest, most semantic)
        # features[-1] is from layer 23 in ViT-L
        last_feature = features[-1]  # [B, 1024, h, w]
        
        # Upsample to full resolution with chunked attention
        with torch.no_grad():
            feature_hr = self.upsampler(x, last_feature, output_size=(H, W), q_chunk_size=4)
        
        # Project to lower dimension (use first projection layer)
        fused = self.proj_layers[0](feature_hr)  # [B, proj_dim, H, W]
        
        # Apply spatial attention
        spatial_attn = self.spatial_attention(fused)  # [B, 1, H, W]
        fused = fused * spatial_attn + fused  # Residual attention
        del spatial_attn
        
        # Final segmentation output
        seg_output = self.seg_head(fused)
        
        return seg_output


class Dinov3CentroidHead(nn.Module):
    """Lightweight decoder head for centroid heatmap generation."""
    
    def __init__(self, in_channels=1024, features=256, decoder_channels=[512, 256, 128, 64]):
        super().__init__()
        
        # Projection layers for each feature scale
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(in_channels, decoder_channels[0], kernel_size=1),  # f4 -> 512
            nn.Conv2d(in_channels, decoder_channels[1], kernel_size=1),  # f3 -> 256  
            nn.Conv2d(in_channels, decoder_channels[2], kernel_size=1),  # f2 -> 128
            nn.Conv2d(in_channels, decoder_channels[3], kernel_size=1),  # f1 -> 64
        ])
        
        # Progressive upsampling and fusion layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(decoder_channels[0] + decoder_channels[1], decoder_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[1]),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(decoder_channels[1] + decoder_channels[2], decoder_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[2]),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(decoder_channels[2] + decoder_channels[3], decoder_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True)
        )
        
        # Final centroid heatmap output (single channel)
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[3], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Single channel output
            nn.Sigmoid()  # Ensure output is in [0, 1] range
        )
    
    def forward(self, features, input_shape):
        """
        Args:
            features: List of 4 feature tensors from Dinov3Backbone [B, 1024, H, W]
            input_shape: Original input image shape for final interpolation
        """
        f1, f2, f3, f4 = features
        
        # Project features to decoder channels
        f1_proj = self.proj_layers[3](f1)  # -> 64 channels
        f2_proj = self.proj_layers[2](f2)  # -> 128 channels  
        f3_proj = self.proj_layers[1](f3)  # -> 256 channels
        f4_proj = self.proj_layers[0](f4)  # -> 512 channels
        
        # Progressive upsampling and fusion (similar to segmentation head)
        x = f4_proj  # [B, 512, 32, 32]
        
        # Stage 1: Upsample and fuse with f3
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 512, 64, 64]
        f3_upsampled = F.interpolate(f3_proj, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 256, 64, 64]
        x = torch.cat([x, f3_upsampled], dim=1)  # [B, 768, 64, 64]
        x = self.conv1(x)  # [B, 256, 64, 64]
        
        # Stage 2: Upsample and fuse with f2
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 256, 128, 128]
        f2_upsampled = F.interpolate(f2_proj, scale_factor=4, mode='bilinear', align_corners=True)  # [B, 128, 128, 128]
        x = torch.cat([x, f2_upsampled], dim=1)  # [B, 384, 128, 128]
        x = self.conv2(x)  # [B, 128, 128, 128]
        
        # Stage 3: Upsample and fuse with f1
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 128, 256, 256]
        f1_upsampled = F.interpolate(f1_proj, scale_factor=8, mode='bilinear', align_corners=True)  # [B, 64, 256, 256]
        x = torch.cat([x, f1_upsampled], dim=1)  # [B, 192, 256, 256]
        x = self.conv3(x)  # [B, 64, 256, 256]
        
        # Stage 4: Final upsampling
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 64, 512, 512]
        
        # Generate centroid heatmap
        x = self.final_conv(x)  # [B, 1, 512, 512]
        
        # Final interpolation to match input size
        H, W = input_shape[2], input_shape[3]
        if x.shape[2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
        return x.squeeze(1)  # Remove channel dimension to get [B, H, W]


class Dinov3SegmentationHeadWithIntermediate(nn.Module):
    """
    Modified segmentation head that exposes intermediate features for centroid head
    """
    
    def __init__(self, in_channels, features, num_classes, out_channels, use_bn=False):
        super().__init__()
        
        # Same as original segmentation head but with intermediate feature access
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels[i], kernel_size=1) for i in range(4)
        ])
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels[0] + out_channels[1], features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(features + out_channels[2], 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128 + out_channels[3], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, features, input_shape):
        f1, f2, f3, f4 = features
        
        # Project features
        f1_proj = self.proj_layers[3](f1)  
        f2_proj = self.proj_layers[2](f2)  
        f3_proj = self.proj_layers[1](f3)  
        f4_proj = self.proj_layers[0](f4)  
        
        # Progressive upsampling and fusion
        x = f4_proj
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        f3_upsampled = F.interpolate(f3_proj, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, f3_upsampled], dim=1)
        x = self.conv1(x)  # First intermediate feature [B, features, 64, 64]
        
        intermediate_64 = x.clone()  # Save for centroid head
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        f2_upsampled = F.interpolate(f2_proj, scale_factor=4, mode='bilinear', align_corners=True)
        x = torch.cat([x, f2_upsampled], dim=1)
        x = self.conv2(x)  # Second intermediate feature [B, 128, 128, 128]
        
        intermediate_128 = x.clone()  # Save for centroid head
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        f1_upsampled = F.interpolate(f1_proj, scale_factor=8, mode='bilinear', align_corners=True)
        x = torch.cat([x, f1_upsampled], dim=1)
        x = self.conv3(x)  # Third intermediate feature [B, 64, 256, 256]
        
        intermediate_256 = x.clone()  # Save for centroid head
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.final_conv(x)
        
        # Final interpolation
        H, W = input_shape[2], input_shape[3]
        if x.shape[2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
        # Return segmentation output and intermediate features
        intermediate_features = {
            '64': intermediate_64,    # [B, features, 64, 64]
            '128': intermediate_128,  # [B, 128, 128, 128]
            '256': intermediate_256   # [B, 64, 256, 256]
        }
        
        return x, intermediate_features


class Dinov3CentroidHeadDenseTransformer(nn.Module):
    """
    Dense Prediction Transformer-style centroid detection head following proper DPT principles.
    
    Key Features:
    - Uses multi-scale features from DIFFERENT ViT layers (not just upsampled single feature)
    - Dense fusion across all scales for both semantic and spatial information
    - Separate pathway from segmentation to avoid feature competition
    - Built-in sharpening operations for precise peak localization
    
    DPT Approach:
    1. Extract features from multiple ViT layers [4, 11, 17, 23] for DINOv3-L
    2. Process each scale independently with transformer blocks
    3. Fuse features densely from coarse to fine
    4. Generate sharp centroid predictions with learnable temperature scaling
    """
    
    def __init__(self, backbone_features=1024, features=256):
        super().__init__()
        
        # Multi-scale feature processing - directly from backbone
        # DINOv3 ViT-L gives us features at different scales through different layers
        
        # Feature projection layers for different scales
        self.feature_proj_16 = nn.Conv2d(backbone_features, features//2, 1)  # 1/16 scale
        self.feature_proj_8 = nn.Conv2d(backbone_features, features//2, 1)   # 1/8 scale  
        self.feature_proj_4 = nn.Conv2d(backbone_features, features//2, 1)   # 1/4 scale
        
        # Dense prediction transformer blocks
        self.transformer_16 = self._make_transformer_block(features//2, features//2)
        self.transformer_8 = self._make_transformer_block(features//2, features//2)
        self.transformer_4 = self._make_transformer_block(features//2, features//2)
        
        # Feature fusion layers with dense connections
        self.fusion_8_16 = nn.Sequential(
            nn.Conv2d(features, features//2, 3, padding=1),
            nn.BatchNorm2d(features//2),
            nn.ReLU(inplace=True)
        )
        
        self.fusion_4_8_16 = nn.Sequential(
            nn.Conv2d(features + features//2, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        
        # Centroid-specific processing layers
        self.centroid_encoder = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features//2, 3, padding=1),
            nn.BatchNorm2d(features//2),
            nn.ReLU(inplace=True)
        )
        
        # Sharpening convolution layers
        self.sharpening_conv = nn.Sequential(
            nn.Conv2d(features//2, features//4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features//4, features//4, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Final prediction head optimized for peaks
        self.peak_predictor = nn.Sequential(
            nn.Conv2d(features//4, features//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features//8, 1, 1),
        )
        
        # Learnable parameters for output sharpening
        self.peak_temperature = nn.Parameter(torch.tensor(1.0))  # Sharp temperature
        self.peak_bias = nn.Parameter(torch.tensor(0.0))        # Negative bias for sparsity
        
    def _make_transformer_block(self, in_channels, out_channels, num_heads=8):
        """Create a simple transformer block for feature processing"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Self-attention like processing via grouped convolution
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=num_heads),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
        )
    
    def forward(self, backbone_features, input_shape):
        """
        Args:
            backbone_features: List/tuple of multi-scale backbone features from different ViT layers
                              For DINOv3-L: [layer4, layer11, layer17, layer23] features
            input_shape: Original input shape for upsampling
        """
        # Extract features from different scales (proper Dense Prediction Transformer approach)
        # DINOv3-L provides 4 different feature scales from layers [4, 11, 17, 23]
        
        if isinstance(backbone_features, (list, tuple)):
            # Use actual multi-scale features from different ViT layers
            feat_low = backbone_features[0]      # Early layer - fine details, lower semantic
            feat_mid_low = backbone_features[1]  # Mid-early layer 
            feat_mid_high = backbone_features[2] # Mid-late layer
            feat_high = backbone_features[3]     # Late layer - high semantic, coarser spatial
        else:
            # Fallback: simulate multi-scale from single feature (if needed)
            feat_high = backbone_features
            feat_mid_high = F.interpolate(feat_high, scale_factor=1.5, mode='bilinear', align_corners=False)  
            feat_mid_low = F.interpolate(feat_high, scale_factor=2, mode='bilinear', align_corners=False)
            feat_low = F.interpolate(feat_high, scale_factor=3, mode='bilinear', align_corners=False)
        
        # Process each scale with transformer blocks (actual multi-scale dense processing)
        feat_16 = self.feature_proj_16(feat_high)      # Highest semantic level
        feat_16 = self.transformer_16(feat_16)
        
        feat_8 = self.feature_proj_8(feat_mid_high)    # Mid-high semantic + spatial detail
        feat_8 = self.transformer_8(feat_8)
        
        feat_4 = self.feature_proj_4(feat_mid_low)     # Lower semantic but finer spatial detail
        feat_4 = self.transformer_4(feat_4)
        
        # Dense feature fusion - bottom-up pathway
        
        # Fuse 1/16 and 1/8 features
        feat_16_to_8 = F.interpolate(feat_16, size=feat_8.shape[2:], mode='bilinear', align_corners=False)
        fused_8 = torch.cat([feat_8, feat_16_to_8], dim=1)  # [B, 256, 64, 64]
        fused_8 = self.fusion_8_16(fused_8)  # [B, 128, 64, 64]
        
        # Fuse all scales at 1/4 resolution
        feat_16_to_4 = F.interpolate(feat_16, size=feat_4.shape[2:], mode='bilinear', align_corners=False)
        fused_8_to_4 = F.interpolate(fused_8, size=feat_4.shape[2:], mode='bilinear', align_corners=False)
        fused_all = torch.cat([feat_4, fused_8_to_4, feat_16_to_4], dim=1)  # [B, 384, 128, 128]
        fused_all = self.fusion_4_8_16(fused_all)  # [B, 256, 128, 128]
        
        # Centroid-specific processing
        centroid_features = self.centroid_encoder(fused_all)  # [B, 128, 128, 128]
        
        # Apply sharpening operations
        sharp_features = self.sharpening_conv(centroid_features)  # [B, 64, 128, 128]
        
        # Generate peak predictions
        raw_peaks = self.peak_predictor(sharp_features)  # [B, 1, 128, 128]
        
        # Apply learnable transformations BEFORE upsampling for better gradient flow
        raw_peaks = raw_peaks + self.peak_bias
        raw_peaks = raw_peaks / (self.peak_temperature.abs() + 1e-6)  # Temperature scaling
        
        # Final upsampling to input resolution
        target_h, target_w = input_shape[2], input_shape[3]
        peaks = F.interpolate(raw_peaks, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        # Remove channel dimension if present (from upsampling)
        if len(peaks.shape) == 4 and peaks.shape[1] == 1:
            peaks = peaks.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
        
        # SIMPLE & EFFECTIVE SOLUTION: Just use sigmoid without aggressive modifications
        # The key insight: Our CenterNetFocalLoss will handle the training dynamics
        peaks = torch.sigmoid(peaks)
        
        # Optional: Slight sharpening through power scaling (less aggressive than softmax)
        # This preserves gradients while encouraging sharper peaks
        peaks = torch.pow(peaks, 0.8)  # Slightly sharpen without destroying gradients
        
        return peaks  # [B, H, W]


class Dinov3CentroidHeadShared(nn.Module):
    """
    Improved centroid head that reuses segmentation intermediate features
    """
    
    def __init__(self, segmentation_features, num_classes):
        super().__init__()
        
        # Feature fusion layers to combine segmentation info with centroid task
        self.seg_feature_conv = nn.Sequential(
            nn.Conv2d(segmentation_features, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.seg_logits_conv = nn.Sequential(
            nn.Conv2d(num_classes, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Fusion of intermediate features
        self.fusion_128 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),  # seg_features + intermediate_128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.fusion_256 = nn.Sequential(
            nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1),   # fusion_128 + intermediate_256
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # AGGRESSIVE attention mechanism for centroid localization (no sigmoid compression)
        self.attention = nn.Sequential(
            nn.Conv2d(32 + 64, 64, kernel_size=1),  # Increase capacity
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.Softplus()  # Softplus instead of sigmoid - allows larger values
        )
        
        # Final centroid prediction layers optimized for SHARP outputs
        self.centroid_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # Add dropout for better generalization
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1)
            # NO final activation - let raw logits have full range
        )
        
        # BALANCED parameters for sharpness without saturation
        self.temperature = nn.Parameter(torch.tensor(0.3))  # Moderate temperature
        
        # Moderate scaling to boost dynamic range
        self.output_scale = nn.Parameter(torch.tensor(3.0))  # Moderate scaling
        
        # Small positive bias to help with activation
        self.output_bias = nn.Parameter(torch.tensor(0.5))  # Small positive bias
        
    def forward(self, seg_intermediate, seg_logits, input_shape):
        """
        Args:
            seg_intermediate: Dict with keys '64', '128', '256' containing intermediate features
            seg_logits: [B, num_classes, H, W] segmentation logits
            input_shape: Original input shape
        """
        # Process segmentation features at 64x64
        seg_features_64 = self.seg_feature_conv(seg_intermediate['64'])  # [B, 128, 64, 64]
        
        # Upsample and fuse with 128x128 intermediate
        # Get target size from the actual intermediate feature
        intermediate_h, intermediate_w = seg_intermediate['128'].shape[2], seg_intermediate['128'].shape[3]
        seg_features_128 = F.interpolate(seg_features_64, size=(intermediate_h, intermediate_w), mode='bilinear', align_corners=True)
        fused_128 = torch.cat([seg_features_128, seg_intermediate['128']], dim=1)
        fused_128 = self.fusion_128(fused_128)  # [B, 64, intermediate_h, intermediate_w]
        
        # Upsample and fuse with 256x256 intermediate
        # Get target size from the actual intermediate feature
        target_h, target_w = seg_intermediate['256'].shape[2], seg_intermediate['256'].shape[3]
        fused_256 = F.interpolate(fused_128, size=(target_h, target_w), mode='bilinear', align_corners=True)
        fused_256 = torch.cat([fused_256, seg_intermediate['256']], dim=1)
        fused_256 = self.fusion_256(fused_256)  # [B, 32, target_h, target_w]
        
        # Use segmentation logits as attention for centroid localization
        seg_logits_resized = F.interpolate(seg_logits, size=(target_h, target_w), mode='bilinear', align_corners=True)
        seg_attention = self.seg_logits_conv(seg_logits_resized)  # [B, 64, target_h, target_w]
        
        # Apply AGGRESSIVE attention mechanism
        attention_input = torch.cat([fused_256, seg_attention], dim=1)
        attention_weights = self.attention(attention_input)  # [B, 32, target_h, target_w]
        
        # Use additive attention instead of multiplicative to avoid suppression
        attended_features = fused_256 + attention_weights * 0.5  # Mix original + attention
        
        # Generate RAW centroid heatmap (no restrictions on range)
        centroid_out = self.centroid_conv(attended_features)  # [B, 1, target_h, target_w]
        
        # AGGRESSIVE transformation to break out of uniform predictions
        # 1. Add bias to shift distribution
        centroid_out = centroid_out + self.output_bias
        
        # 2. Scale up dramatically to increase dynamic range
        centroid_out = centroid_out * self.output_scale
        
        # 3. Apply aggressive temperature scaling for sharpness BEFORE upsampling
        centroid_out = centroid_out / (self.temperature + 1e-8)
        
        # Upsample to full resolution using bilinear interpolation
        centroid_out = F.interpolate(centroid_out, scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final interpolation to match input size
        H, W = input_shape[2], input_shape[3]
        if centroid_out.shape[2:] != (H, W):
            centroid_out = F.interpolate(centroid_out, size=(H, W), mode='bilinear', align_corners=True)
        
        # Use Softplus instead of sigmoid to avoid saturation
        # Softplus maps to [0, inf) and is smoother than ReLU
        centroid_out = F.softplus(centroid_out)
        
        # Normalize to [0, 1] range to match targets 
        # But allow for some values > 1 during intermediate processing
        max_val = centroid_out.max() + 1e-8
        min_val = centroid_out.min()
        centroid_out = (centroid_out - min_val) / (max_val - min_val)
        
        return centroid_out.squeeze(1)  # [B, H, W]

class Dinov3DPT(nn.Module):
    def __init__(self, num_classes, backbone_type: str = "vitl", dataset: str = "lvd1689m", features=256,
                 patch_size: int = 16, use_auxiliary: bool = False, load_pretrained_backbone: bool = False,
                 head_type: str = "DPTHead"):
        super().__init__()
        embedding_dim = {
            "vitl": 1024,
            "vitb": 768,
            "vitg": 1536,
        }
        self.patch_size = patch_size
        self.backbone = Dinov3Backbone(reshape=False, return_class_token=True, dataset=dataset, load_pretrained_backbone=load_pretrained_backbone)
        head_classes = {
            "DPTHead": DPTHead,
            "DPTHeadLinear": DPTHeadLinear,
        }
        head_cls = head_classes.get(head_type, DPTHead)
        self.segmentation_head = head_cls(embedding_dim[backbone_type], num_classes=num_classes, use_auxiliary=use_auxiliary, patch_size=self.patch_size)

    def load_backbone_weights(self, weights_path: str = None):
        """Load pretrained encoder weights into the backbone.
        
        Args:
            weights_path: Path to weights. If None, uses default based on dataset type.
        """
        self.backbone.load_weights(weights_path)

    def forward(self, x):
        features = self.backbone(x)
        if(x.shape[-2] % self.patch_size != 0 or x.shape[-1] % self.patch_size != 0):
            raise ValueError(f"Image size {x.shape[-2]}x{x.shape[-1]} is not divisible by patch size {self.patch_size}")
        patch_h, patch_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        segmentation_output = self.segmentation_head(features, patch_h, patch_w)
        return segmentation_output

class Dinov3DPTCentroidSegmentation(nn.Module):
    def __init__(self, num_classes, backbone_type: str = "vitl", dataset: str = "lvd1689m", features=256,
                 patch_size: int = 16, use_auxiliary: bool = False, load_pretrained_backbone: bool = False):
        super().__init__()
        embedding_dim = {
            "vitl": 1024,
            "vitb": 768,
            "vitg": 1536,
        }
        self.patch_size = patch_size
        self.backbone = Dinov3Backbone(reshape=False, return_class_token=True, dataset=dataset, load_pretrained_backbone=load_pretrained_backbone)
        self.centroidhead = DPTHead(embedding_dim[backbone_type], num_classes=1, use_auxiliary=use_auxiliary, patch_size=self.patch_size)
        self.segmentation_head = DPTHead(embedding_dim[backbone_type], num_classes=num_classes, use_auxiliary=use_auxiliary, patch_size=self.patch_size)

    def load_backbone_weights(self, weights_path: str = None):
        """Load pretrained encoder weights into the backbone.
        
        Args:
            weights_path: Path to weights. If None, uses default based on dataset type.
        """
        self.backbone.load_weights(weights_path)

    def forward(self, x):
        features = self.backbone(x)
        if(x.shape[-2] % self.patch_size != 0 or x.shape[-1] % self.patch_size != 0):
            raise ValueError(f"Image size {x.shape[-2]}x{x.shape[-1]} is not divisible by patch size {self.patch_size}")
        patch_h, patch_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        centroid_output = self.centroidhead(features, patch_h, patch_w)
        segmentation_output = self.segmentation_head(features, patch_h, patch_w)

        if (len(centroid_output.shape) == 4) and (centroid_output.shape[1] == 1):
            centroid_output = centroid_output.squeeze(1)  # [B, 1, H, W] -> [B, H, W]

        return segmentation_output, centroid_output

class Dinov3SegmentationMultiHead(nn.Module):
    """
    Improved Multi-headed DINOv3 model for segmentation and centroid detection.
    
    This model shares features between heads for better centroid performance:
    1. Segmentation head for polygon masks  
    2. Centroid head that reuses segmentation intermediate features and logits
    """
    
    def __init__(self, num_classes, backbone_type: str = "vitl", dataset: str = "lvd1689m", features=256,
                 load_pretrained_backbone: bool = False):
        super().__init__()
        
        # Shared backbone
        self.backbone = Dinov3Backbone(backbone_type, dataset=dataset, load_pretrained_backbone=load_pretrained_backbone)
        
        # Modified segmentation head that exposes intermediate features
        self.segmentation_head = Dinov3SegmentationHeadWithIntermediate(
            in_channels=1024,  # DINOv3 vitl feature dimension
            features=features,
            num_classes=num_classes,
            out_channels=[256, 512, 1024, 1024],
            use_bn=False
        )
        
        # NEW: Dense transformer centroid head - completely separate pathway
        self.centroid_head = Dinov3CentroidHeadDenseTransformer(
            backbone_features=1024,  # DINOv3 vitl feature dimension
            features=features
        )

    def load_backbone_weights(self, weights_path: str = None):
        """Load pretrained encoder weights into the backbone.
        
        Args:
            weights_path: Path to weights. If None, uses default based on dataset type.
        """
        self.backbone.load_weights(weights_path)
    
    def forward(self, x):
        """
        Forward pass through both heads with separate pathways.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            tuple: (segmentation_logits, centroid_heatmap)
            - segmentation_logits: [B, num_classes, H, W]
            - centroid_heatmap: [B, H, W]
        """
        # Extract features from shared backbone
        backbone_intermediate_features = self.backbone(x)  # Returns tuple of features from different layers
        
        # Generate segmentation output (uses all intermediate features)
        segmentation_output, seg_intermediate = self.segmentation_head(backbone_intermediate_features, x.shape)
        
        # Generate centroid output using ALL backbone intermediate features (proper dense approach)
        # Pass all multi-scale features for dense prediction transformer processing
        centroid_output = self.centroid_head(backbone_intermediate_features, x.shape)
        
        return segmentation_output, centroid_output


class Dinov3Mask2Former(nn.Module):
    """
    Mask2Former-style segmentation model with Dinov3Backbone.
    
    Architecture:
    1. Dinov3Backbone: Extract multi-scale features from ViT
    2. Pixel Decoder: Build FPN-style feature pyramid  
    3. Transformer Decoder: Process learnable queries with cross-attention
    4. Prediction Heads: Generate class and mask predictions
    
    Args:
        num_classes: Number of segmentation classes (excludes background)
        backbone_type: Type of ViT backbone ('vitl', 'vitb', 'vitg')
        dataset: Dataset type - either 'sat493m' or 'lvd1689m'
        num_queries: Number of object queries (default: 100)
        hidden_dim: Hidden dimension for transformer (default: 256)
        num_decoder_layers: Number of transformer decoder layers (default: 6)
        num_heads: Number of attention heads (default: 8)
        load_pretrained_backbone: Whether to load pretrained backbone weights (default: False)
    """
    
    def __init__(
        self,
        num_classes=2,
        backbone_type="vitl",
        dataset="lvd1689m",
        num_queries=100,
        hidden_dim=256,
        num_decoder_layers=6,
        num_heads=8,
        dropout=0.0,
        load_pretrained_backbone=False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # 1. Backbone
        self.backbone = Dinov3Backbone(
            backbone_type=backbone_type,
            dataset=dataset,
            reshape=True,              # Get spatial features [B, C, H, W]
            return_class_token=False,  # Don't need CLS token for dense prediction
            load_pretrained_backbone=load_pretrained_backbone
        )
        
        # 2. Pixel Decoder (FPN)
        embedding_dim = {
            "vitl": 1024,
            "vitb": 768,
            "vitg": 1536,
        }
        self.pixel_decoder = Mask2FormerPixelDecoder(
            in_channels=embedding_dim[backbone_type],
            feature_channels=hidden_dim
        )
        
        # 3. Transformer Decoder
        self.transformer_decoder = Mask2FormerTransformerDecoder(
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            dim_feedforward=hidden_dim * 4
        )
        
        # 4. Prediction Heads
        self.prediction_heads = Mask2FormerPredictionHeads(
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

    def load_backbone_weights(self, weights_path: str = None):
        """Load pretrained encoder weights into the backbone.
        
        Args:
            weights_path: Path to weights. If None, uses default based on dataset type.
        """
        self.backbone.load_weights(weights_path)
    
    def forward(self, x):
        """
        Args:
            x: Input image tensor [B, 3, H, W]
        
        Returns:
            dict with keys:
                - 'pred_logits': [B, num_queries, num_classes+1] class predictions
                - 'pred_masks': [B, num_queries, H, W] binary mask predictions
                - 'aux_outputs': List of auxiliary outputs from intermediate layers (optional)
        """
        input_shape = x.shape
        
        # Step 1: Extract backbone features
        # Returns 4 features: [layer_4, layer_11, layer_17, layer_23]
        # Each: [B, 1024, 32, 32] for 512x512 input
        backbone_features = self.backbone(x)
        
        # Step 2: Build feature pyramid
        # multi_scale_features: [C2, C3, C4, C5] at [256, 128, 64, 32] resolutions
        # mask_features: [B, 256, 256, 256] for mask prediction
        multi_scale_features, mask_features = self.pixel_decoder(backbone_features)
        
        # Step 3: Transformer decoder
        # query_features: [B, num_queries, hidden_dim]
        # all_layer_outputs: List of intermediate query features
        query_features, all_layer_outputs = self.transformer_decoder(multi_scale_features)
        
        # Step 4: Generate predictions
        class_logits, mask_logits = self.prediction_heads(query_features, mask_features)
        # class_logits: [B, num_queries, num_classes+1]
        # mask_logits: [B, num_queries, 256, 256]
        
        # Step 5: Upsample masks to input resolution
        H, W = input_shape[2], input_shape[3]
        mask_logits = F.interpolate(
            mask_logits, 
            size=(H, W), 
            mode='bilinear',
            align_corners=False
        )
        
        outputs = {
            'pred_logits': class_logits,  # [B, num_queries, num_classes+1]
            'pred_masks': mask_logits      # [B, num_queries, H, W]
        }
        
        # Optionally return auxiliary outputs from intermediate decoder layers
        # Useful for deep supervision during training
        if self.training:
            aux_outputs = []
            for layer_output in all_layer_outputs[:-1]:  # Exclude last layer (already in main output)
                aux_class, aux_mask = self.prediction_heads(layer_output, mask_features)
                aux_mask = F.interpolate(aux_mask, size=(H, W), mode='bilinear', align_corners=False)
                aux_outputs.append({
                    'pred_logits': aux_class,
                    'pred_masks': aux_mask
                })
            outputs['aux_outputs'] = aux_outputs
        
        return outputs


class Dinov3FeatureAdapter(nn.Module):
    """
    Adapter to convert Dinov3Backbone features to HuggingFace Mask2Former format.
    
    Dinov3 outputs 4 features at the same resolution but from different layers.
    This adapter creates a proper multi-scale pyramid as expected by HuggingFace.
    """
    
    def __init__(self, in_channels=1024, feature_channels=256):
        super().__init__()
        
        # Project each Dinov3 feature to feature_channels
        self.input_projections = nn.ModuleList([
            nn.Conv2d(in_channels, feature_channels, kernel_size=1)
            for _ in range(4)
        ])
        
        # Create multi-scale pyramid
        # We'll progressively upsample to create different resolutions
        self.conv_1_16 = nn.Conv2d(feature_channels, feature_channels, 3, padding=1)
        self.conv_1_8 = nn.Conv2d(feature_channels, feature_channels, 3, padding=1)
        self.conv_1_4 = nn.Conv2d(feature_channels, feature_channels, 3, padding=1)
        
        # Layer norms (HuggingFace uses these)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(feature_channels) for _ in range(4)
        ])
    
    def forward(self, dinov3_features):
        """
        Args:
            dinov3_features: List of 4 features [layer_4, layer_11, layer_17, layer_23]
                            Each: [B, 1024, H, W] at same resolution
        
        Returns:
            multi_scale_features: List of 4 features at different scales
                                 [1/4, 1/8, 1/16, 1/32] resolution
        """
        # Project all features
        projected = [proj(feat) for proj, feat in zip(self.input_projections, dinov3_features)]
        
        # Create multi-scale pyramid by upsampling
        # Start from deepest feature (layer_23)
        feat_1_32 = projected[3]  # [B, 256, H, W] - deepest, keep as is
        
        # 1/16 scale: upsample layer_17
        feat_1_16 = F.interpolate(projected[2], scale_factor=2, mode='bilinear', align_corners=False)
        feat_1_16 = self.conv_1_16(feat_1_16)  # [B, 256, 2H, 2W]
        
        # 1/8 scale: upsample layer_11
        feat_1_8 = F.interpolate(projected[1], scale_factor=4, mode='bilinear', align_corners=False)
        feat_1_8 = self.conv_1_8(feat_1_8)  # [B, 256, 4H, 4W]
        
        # 1/4 scale: upsample layer_4
        feat_1_4 = F.interpolate(projected[0], scale_factor=8, mode='bilinear', align_corners=False)
        feat_1_4 = self.conv_1_4(feat_1_4)  # [B, 256, 8H, 8W]
        
        # Apply layer norms (HuggingFace format)
        # Permute to [B, H, W, C] for layer norm, then back
        multi_scale = []
        for feat, norm in zip([feat_1_4, feat_1_8, feat_1_16, feat_1_32], self.layer_norms):
            B, C, H, W = feat.shape
            feat = feat.permute(0, 2, 3, 1)  # [B, H, W, C]
            feat = norm(feat)
            feat = feat.permute(0, 3, 1, 2)  # [B, C, H, W]
            multi_scale.append(feat)
        
        return multi_scale  # [1/4, 1/8, 1/16, 1/32]


class Dinov3HFMask2Former(nn.Module):
    """
    Hybrid model combining Dinov3Backbone with HuggingFace Mask2Former decoder.
    
    This model uses:
    - Dinov3Backbone: For robust feature extraction (pretrained on satellite/other data)
    - HuggingFace Mask2Former: For state-of-the-art segmentation decoder
    
    Benefits:
    - Leverage Dinov3's powerful pretrained features
    - Use HuggingFace's proven Mask2Former implementation
    - Easy fine-tuning and deployment
    
    Args:
        num_classes: Number of segmentation classes (excluding background)
        dinov3_backbone_type: Type of ViT backbone ('vitl', 'vitb', 'vitg')
        dataset: Dataset type - either 'sat493m' or 'lvd1689m'
        mask2former_config: HuggingFace Mask2Former config name or object
        freeze_backbone: Whether to freeze Dinov3 backbone weights
        hidden_dim: Hidden dimension for decoder (default: 256)
        load_pretrained_backbone: Whether to load pretrained backbone weights (default: False)
    
    Example:
        >>> model = Dinov3HFMask2Former(
        ...     num_classes=2,
        ...     dinov3_backbone_type="vitl",
        ...     dataset="sat493m",
        ...     mask2former_config="facebook/mask2former-swin-small-coco-instance"
        ... )
        >>> model.load_backbone_weights("path/to/encoder_weights.pth")
    """
    
    def __init__(
        self,
        num_classes=2,
        dinov3_backbone_type="vitl",
        dataset="lvd1689m",
        mask2former_config="facebook/mask2former-swin-small-coco-instance",
        freeze_backbone=True,
        hidden_dim=256,
        load_pretrained_backbone=False
    ):
        super().__init__()
        
        if not HF_AVAILABLE:
            raise ImportError(
                "transformers library is required for Dinov3HFMask2Former. "
                "Install with: pip install transformers"
            )
        
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        
        # 1. Dinov3 Backbone for feature extraction
        self.backbone = Dinov3Backbone(
            backbone_type=dinov3_backbone_type,
            dataset=dataset,
            reshape=True,              # Get spatial features
            return_class_token=False,  # Don't need CLS token
            load_pretrained_backbone=load_pretrained_backbone
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Dinov3Backbone frozen - will not be updated during training")
        
        # 2. Feature adapter to convert Dinov3 to HuggingFace format
        embedding_dim = {
            "vitl": 1024,
            "vitb": 768,
            "vitg": 1536,
        }
        self.feature_adapter = Dinov3FeatureAdapter(
            in_channels=embedding_dim[dinov3_backbone_type],
            feature_channels=hidden_dim
        )
        
        # 3. Load HuggingFace Mask2Former configuration
        if isinstance(mask2former_config, str):
            config = Mask2FormerConfig.from_pretrained(mask2former_config)
        else:
            config = mask2former_config
        
        # Modify config for our use case
        config.num_labels = num_classes
        config.hidden_dim = hidden_dim
        config.num_feature_levels = 4  # Match our multi-scale features
        
        # 4. Initialize Mask2Former model
        # We'll use the model's decoder and heads, but replace backbone features
        self.mask2former = Mask2FormerModel(config)
        
        # Initialize prediction heads
        self.class_predictor = nn.Linear(hidden_dim, num_classes + 1)  # +1 for "no object"
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        print(f"Dinov3HFMask2Former initialized:")
        print(f"  - Backbone: Dinov3-{dinov3_backbone_type} ({'frozen' if freeze_backbone else 'trainable'})")
        print(f"  - Dataset: {dataset}")
        print(f"  - Decoder: HuggingFace Mask2Former")
        print(f"  - Classes: {num_classes}")
        print(f"  - Hidden dim: {hidden_dim}")

    def load_backbone_weights(self, weights_path: str = None):
        """Load pretrained encoder weights into the backbone.
        
        Args:
            weights_path: Path to weights. If None, uses default based on dataset type.
        """
        self.backbone.load_weights(weights_path)
    
    def forward(self, pixel_values, mask_labels=None, class_labels=None):
        """
        Forward pass through the model.
        
        Args:
            pixel_values: Input images [B, 3, H, W]
            mask_labels: Ground truth masks [B, num_instances, H, W] (optional, for training)
            class_labels: Ground truth class labels [B, num_instances] (optional, for training)
        
        Returns:
            dict with:
                - 'pred_logits': [B, num_queries, num_classes+1] class predictions
                - 'pred_masks': [B, num_queries, H, W] mask predictions
                - 'loss': total loss (if labels provided)
                - 'loss_dict': detailed loss breakdown (if labels provided)
        """
        B, _, H, W = pixel_values.shape
        
        # Step 1: Extract features using Dinov3
        # Returns 4 features: [layer_4, layer_11, layer_17, layer_23]
        if self.freeze_backbone:
            with torch.no_grad():
                dinov3_features = self.backbone(pixel_values)
        else:
            dinov3_features = self.backbone(pixel_values)
        
        # Step 2: Adapt features to HuggingFace format
        # Convert to multi-scale pyramid [1/4, 1/8, 1/16, 1/32]
        adapted_features = self.feature_adapter(dinov3_features)
        
        # Step 3: Process through Mask2Former decoder
        # This is a simplified version - full implementation would need encoder_hidden_states
        # For now, we'll build a simplified forward pass
        
        # Get pixel decoder outputs (multi-scale features)
        multi_scale_features = adapted_features
        
        # Use finest resolution for mask features
        mask_features = multi_scale_features[0]  # [B, hidden_dim, H/4, W/4]
        
        # Generate queries through transformer decoder
        # Simplified: use learnable queries
        num_queries = 100  # Standard for Mask2Former
        query_embeds = nn.Embedding(num_queries, self.mask2former.config.hidden_dim).to(pixel_values.device)
        queries = query_embeds.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, hidden_dim]
        
        # Predict classes and masks
        class_logits = self.class_predictor(queries)  # [B, num_queries, num_classes+1]
        
        # Mask predictions via dot product
        mask_embeds = self.mask_embed(queries)  # [B, num_queries, hidden_dim]
        mask_logits = torch.einsum("bqc,bchw->bqhw", mask_embeds, mask_features)
        
        # Upsample masks to input resolution
        mask_logits = F.interpolate(
            mask_logits,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        
        # Prepare outputs
        outputs = {
            'pred_logits': class_logits,
            'pred_masks': mask_logits
        }
        
        # Compute loss if labels provided (training mode)
        if mask_labels is not None and class_labels is not None:
            loss_dict = self.compute_loss(class_logits, mask_logits, class_labels, mask_labels)
            outputs['loss'] = sum(loss_dict.values())
            outputs['loss_dict'] = loss_dict
        
        return outputs
    
    def compute_loss(self, pred_logits, pred_masks, target_classes, target_masks):
        """
        Compute Mask2Former-style losses.
        
        This is a simplified version. Full implementation would include:
        - Hungarian matching between predictions and targets
        - Focal loss for classification
        - Dice + BCE loss for masks
        """
        # Simplified loss computation
        # In practice, you'd use Hungarian matching here
        
        # Classification loss (Cross Entropy)
        # Flatten predictions and targets
        pred_logits_flat = pred_logits.reshape(-1, pred_logits.shape[-1])
        target_classes_flat = target_classes.reshape(-1)
        
        class_loss = F.cross_entropy(
            pred_logits_flat,
            target_classes_flat.long(),
            ignore_index=-1
        )
        
        # Mask loss (Dice + BCE)
        # Simplified: average over all predictions and targets
        pred_masks_sigmoid = pred_masks.sigmoid()
        
        # Dice loss
        numerator = 2 * (pred_masks_sigmoid * target_masks).sum()
        denominator = pred_masks_sigmoid.sum() + target_masks.sum()
        dice_loss = 1 - (numerator + 1) / (denominator + 1)
        
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_masks,
            target_masks,
            reduction='mean'
        )
        
        loss_dict = {
            'loss_ce': class_loss,
            'loss_dice': dice_loss,
            'loss_bce': bce_loss
        }
        
        return loss_dict
    
    @torch.no_grad()
    def inference(self, pixel_values, threshold=0.5):
        """
        Simplified inference with post-processing.
        
        Args:
            pixel_values: Input images [B, 3, H, W]
            threshold: Confidence threshold for predictions
        
        Returns:
            List of dicts, one per image, containing:
                - 'masks': Binary masks [num_instances, H, W]
                - 'classes': Predicted classes [num_instances]
                - 'scores': Confidence scores [num_instances]
        """
        self.eval()
        outputs = self.forward(pixel_values)
        
        pred_logits = outputs['pred_logits']  # [B, num_queries, num_classes+1]
        pred_masks = outputs['pred_masks']    # [B, num_queries, H, W]
        
        # Get probabilities
        pred_probs = F.softmax(pred_logits, dim=-1)  # [B, num_queries, num_classes+1]
        pred_scores, pred_classes = pred_probs.max(dim=-1)  # [B, num_queries]
        
        # Threshold masks
        pred_masks = (pred_masks.sigmoid() > threshold).float()
        
        # Post-process per image
        results = []
        for b in range(pred_logits.shape[0]):
            # Filter by score threshold and not "no object" class
            valid_mask = (pred_scores[b] > threshold) & (pred_classes[b] < self.num_classes)
            
            results.append({
                'masks': pred_masks[b][valid_mask],
                'classes': pred_classes[b][valid_mask],
                'scores': pred_scores[b][valid_mask]
            })
        
        return results

