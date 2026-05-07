import torch
import torch.nn as nn
import torch.nn.functional as F

class linear_head(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 5):
        super(linear_head, self).__init__()
        self.convolution_block = nn.Sequential(
            nn.Upsample(scale_factor= 4),
            nn.Conv2d(embedding_size, num_classes, (1,1)),
            nn.Upsample(scale_factor= 4, mode='bilinear')
        )

    def forward(self, x):
        return self.convolution_block(x)

class conv_block(nn.Module):
    def __init__(self, input_size, filter_size):
        super(conv_block, self).__init__()
        self.convolution_block = nn.Sequential(
            nn.Conv2d(input_size, filter_size, (3,3), padding=1),
            nn.BatchNorm2d(filter_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.convolution_block(x)
        return x

class ConvDecoderSkip(nn.Module):
    def __init__(self, input_size, filter_size, upsample = "ConvTranspose2d", is_upsample = True):
        super().__init__()
        self.convolution_block = nn.Sequential(
            nn.BatchNorm2d(input_size),
            nn.ReLU(),
            nn.Conv2d(input_size, filter_size, (3,3), padding=1),

            nn.BatchNorm2d(filter_size),
            nn.ReLU(),
            nn.Conv2d(filter_size, filter_size, (3,3), padding=1), 

            nn.BatchNorm2d(filter_size),
            nn.ReLU()            
        )
        self.skip_layer = nn.Conv2d(input_size, filter_size, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.is_upsample = is_upsample

    def forward(self, inp):
        skip_x = self.skip_layer(inp)
        x = self.convolution_block(inp)
        x = x+skip_x
        if self.is_upsample is True: 
            x = self.upsample(x)
        return x

class conv_block_up(nn.Module):
    def __init__(self, input_size, filter_size):
        super().__init__()
        self.convolution_block = nn.Sequential(
            nn.Conv2d(input_size, filter_size, (3,3), padding=1),
            nn.BatchNorm2d(filter_size),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

    def forward(self, x):
        x = self.convolution_block(x)
        return x

class conv_head(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 5):
        super(conv_head, self).__init__()
        from .layers import conv_block  # Avoid circular import
        self.segmentation_conv = nn.Sequential(
            conv_block(embedding_size, 1024),
            nn.ConvTranspose2d(1024, 1024, kernel_size=4, padding=1, stride=2),
            conv_block(1024, 512),
            nn.ConvTranspose2d(512, 512, kernel_size=4, padding=1, stride=2),
            conv_block(512, 256),
            nn.ConvTranspose2d(256, 256, kernel_size=4, padding=1, stride=2),
            conv_block(256, 128),
            conv_block(128, 64),
            nn.ConvTranspose2d(64, 64, kernel_size=4, padding=1, stride=2),
            nn.Conv2d(64, num_classes, (3,3), padding = 1)
        )
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.segmentation_conv(x)
        return x

class conv_head_upsample(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 5):
        super().__init__()
        from .layers import conv_block  # Avoid circular import
        self.segmentation_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv_block(embedding_size, 1024),       
            conv_block(1024, 512),
            nn.Upsample(scale_factor=2),
            conv_block(512, 256),
            conv_block(256, 128),
            conv_block(128, 64),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, num_classes, (3,3), padding = 1)
        )
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.segmentation_conv(x)
        return x

class conv_head_upsample_full(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 5):
        super().__init__()
        from .layers import conv_block  # Avoid circular import
        self.segmentation_conv = nn.Sequential(
            conv_block(embedding_size, 1024),  
            nn.Upsample(scale_factor=2),
            conv_block(1024, 512),
            nn.Upsample(scale_factor=2),
            conv_block(512, 256),
            nn.Upsample(scale_factor=2),
            conv_block(256, 128),
            conv_block(128, 64),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, num_classes, (3,3), padding = 1)
        )
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.segmentation_conv(x)
        return x

class featup_conv_head(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 5):
        super(featup_conv_head, self).__init__()
        from .layers import conv_block  # Avoid circular import
        self.segmentation_conv = nn.Sequential(
            conv_block(embedding_size, 1024),
            conv_block(1024, 512),
            conv_block(512, 256),
            conv_block(256, 128),
            conv_block(128, 64),
            nn.Conv2d(64, num_classes, (3,3))
        )

    def forward(self, x):
        x = self.segmentation_conv(x)
        return x
    
def conv_bn_relu(in_ch, out_ch, k=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UpConvSkipBlock(nn.Module):
    """
    Upsample by ×2 (learnable) → concat skip → two convs.
    Resize is done to *exact* skip size to cope with 129×129 etc.
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.conv = nn.Sequential(
            conv_bn_relu(out_ch + skip_ch, out_ch),
            conv_bn_relu(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x   = self.up(x)
        x   = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                            align_corners=False)      # safety reshape
        x   = torch.cat([x, skip], dim=1)
        x   = self.conv(x)
        return x


# ============================================================================
# Mask2Former Components
# ============================================================================

class Mask2FormerPixelDecoder(nn.Module):
    """
    FPN-style pixel decoder that converts same-resolution ViT features 
    to multi-scale feature pyramid for Mask2Former.
    
    Input: 4 features at same resolution from different ViT layers
    Output: Multi-scale feature pyramid [C2, C3, C4, C5] + mask features
    """
    
    def __init__(self, in_channels=1024, feature_channels=256):
        super().__init__()
        
        self.feature_channels = feature_channels
        
        # Project each input feature to feature_channels
        self.input_projections = nn.ModuleList([
            nn.Conv2d(in_channels, feature_channels, kernel_size=1)
            for _ in range(4)
        ])
        
        # Lateral convolutions for FPN fusion
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
            for _ in range(4)
        ])
        
        # Output convolutions after fusion
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                conv_bn_relu(feature_channels, feature_channels, k=3),
                conv_bn_relu(feature_channels, feature_channels, k=3)
            )
            for _ in range(4)
        ])
        
        # Mask feature projection (for finest resolution)
        self.mask_features = nn.Sequential(
            conv_bn_relu(feature_channels, feature_channels, k=3),
            conv_bn_relu(feature_channels, feature_channels, k=3)
        )
    
    def forward(self, backbone_features):
        """
        Args:
            backbone_features: List of 4 features [layer_4, layer_11, layer_17, layer_23]
                              Each: [B, 1024, 32, 32] for 512x512 input
        
        Returns:
            multi_scale_features: List of 4 features at different resolutions
                                 [C2, C3, C4, C5] = [256x256, 128x128, 64x64, 32x32]
            mask_features: [B, 256, 256, 256] - finest resolution for mask prediction
        """
        # Project all features to feature_channels
        projected = [proj(feat) for proj, feat in zip(self.input_projections, backbone_features)]
        # projected[0-3]: [B, 256, 32, 32]
        
        # Build multi-scale pyramid
        # Start from deepest feature (layer_23) as base
        features = []
        
        # C5 (1/32): Keep layer_23 at original resolution
        c5 = self.lateral_convs[3](projected[3])  # [B, 256, 32, 32]
        c5 = self.output_convs[3](c5)
        features.append(c5)
        
        # C4 (1/16): Upsample layer_17 to 64x64
        layer_17_up = F.interpolate(projected[2], scale_factor=2, mode='bilinear', align_corners=False)
        c5_up = F.interpolate(c5, scale_factor=2, mode='bilinear', align_corners=False)
        c4 = self.lateral_convs[2](layer_17_up) + c5_up  # [B, 256, 64, 64]
        c4 = self.output_convs[2](c4)
        features.append(c4)
        
        # C3 (1/8): Upsample layer_11 to 128x128
        layer_11_up = F.interpolate(projected[1], scale_factor=4, mode='bilinear', align_corners=False)
        c4_up = F.interpolate(c4, scale_factor=2, mode='bilinear', align_corners=False)
        c3 = self.lateral_convs[1](layer_11_up) + c4_up  # [B, 256, 128, 128]
        c3 = self.output_convs[1](c3)
        features.append(c3)
        
        # C2 (1/4): Upsample layer_4 to 256x256
        layer_4_up = F.interpolate(projected[0], scale_factor=8, mode='bilinear', align_corners=False)
        c3_up = F.interpolate(c3, scale_factor=2, mode='bilinear', align_corners=False)
        c2 = self.lateral_convs[0](layer_4_up) + c3_up  # [B, 256, 256, 256]
        c2 = self.output_convs[0](c2)
        features.append(c2)
        
        # Reverse to get [C2, C3, C4, C5] (finest to coarsest)
        multi_scale_features = list(reversed(features))
        
        # Generate mask features from finest resolution (C2)
        mask_feats = self.mask_features(c2)  # [B, 256, 256, 256]
        
        return multi_scale_features, mask_feats


class Mask2FormerTransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer for Mask2Former.
    Performs: Self-Attention → Cross-Attention → FFN
    """
    
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.0, dim_feedforward=2048):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention  
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
    
    def forward(self, queries, memory, pos_embed=None):
        """
        Args:
            queries: [B, num_queries, hidden_dim] - learnable object queries
            memory: [B, HW, hidden_dim] - flattened spatial features
            pos_embed: Optional positional embedding
        
        Returns:
            queries: [B, num_queries, hidden_dim] - updated queries
        """
        # Self-attention
        q = k = queries
        queries2 = self.self_attn(q, k, queries)[0]
        queries = queries + self.dropout1(queries2)
        queries = self.norm1(queries)
        
        # Cross-attention (queries attend to spatial features)
        queries2 = self.cross_attn(
            query=queries,
            key=memory,
            value=memory
        )[0]
        queries = queries + self.dropout2(queries2)
        queries = self.norm2(queries)
        
        # FFN
        queries2 = self.ffn(queries)
        queries = queries + queries2
        queries = self.norm3(queries)
        
        return queries


class Mask2FormerTransformerDecoder(nn.Module):
    """
    Transformer decoder with learnable object queries for Mask2Former.
    """
    
    def __init__(
        self,
        num_queries=100,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.0,
        dim_feedforward=2048
    ):
        super().__init__()
        
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            Mask2FormerTransformerDecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                dim_feedforward=dim_feedforward
            )
            for _ in range(num_layers)
        ])
        
        # Layer to project spatial features to hidden_dim
        self.memory_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
    
    def forward(self, multi_scale_features):
        """
        Args:
            multi_scale_features: List of [C2, C3, C4, C5] features
                                 For simplicity, we'll use C3 (1/8 resolution)
        
        Returns:
            query_features: [B, num_queries, hidden_dim]
            all_layer_outputs: List of query features from each decoder layer
        """
        # Use mid-resolution features (C3) for cross-attention
        # C3: [B, 256, 128, 128]
        spatial_features = multi_scale_features[1]  # Use C3
        B, C, H, W = spatial_features.shape
        
        # Project and flatten spatial features for cross-attention
        memory = self.memory_proj(spatial_features)  # [B, 256, H, W]
        memory = memory.flatten(2).permute(0, 2, 1)  # [B, H*W, 256]
        
        # Initialize queries for this batch
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, hidden_dim]
        
        # Pass through decoder layers
        all_layer_outputs = []
        for layer in self.layers:
            queries = layer(queries, memory)
            all_layer_outputs.append(queries)
        
        return queries, all_layer_outputs


class Mask2FormerPredictionHeads(nn.Module):
    """
    Prediction heads for Mask2Former: class classification + mask generation.
    """
    
    def __init__(self, hidden_dim=256, num_classes=2):
        super().__init__()
        
        # Class prediction head (predict one of num_classes or "no object")
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        
        # Mask embedding head
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, query_features, mask_features):
        """
        Args:
            query_features: [B, num_queries, hidden_dim]
            mask_features: [B, hidden_dim, H, W]
        
        Returns:
            class_logits: [B, num_queries, num_classes+1]
            mask_logits: [B, num_queries, H, W]
        """
        # Class predictions
        class_logits = self.class_embed(query_features)  # [B, num_queries, num_classes+1]
        
        # Mask predictions via dot product
        mask_embed = self.mask_embed(query_features)  # [B, num_queries, hidden_dim]
        
        # Compute mask logits: dot product between query embeddings and pixel embeddings
        # mask_embed: [B, Q, C], mask_features: [B, C, H, W]
        mask_logits = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        
        return class_logits, mask_logits