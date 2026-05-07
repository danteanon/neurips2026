import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dpt_layers.blocks import FeatureFusionBlock, _make_scratch
from .dpt_layers.transform import Resize, NormalizeImage, PrepareForNet
from .dinov2 import DINOv2


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        num_classes = 2,
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False,
        use_auxiliary=True,
        patch_size=14
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        self.use_auxiliary = use_auxiliary
        self.patch_size = patch_size
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, num_classes, kernel_size=1, stride=1, padding=0),
            nn.Identity(),
        )
        
        # Auxiliary classifiers for intermediate supervision during training
        if self.use_auxiliary:
            # Auxiliary classifier after refinenet4 (lowest resolution)
            self.aux_classifier_4 = nn.Sequential(
                nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(features // 2),
                nn.ReLU(True),
                nn.Conv2d(features // 2, num_classes, kernel_size=1, stride=1, padding=0)
            )
            
            # Auxiliary classifier after refinenet3 (mid-low resolution)
            self.aux_classifier_3 = nn.Sequential(
                nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(features // 2),
                nn.ReLU(True),
                nn.Conv2d(features // 2, num_classes, kernel_size=1, stride=1, padding=0)
            )
            
            # Auxiliary classifier after refinenet2 (mid-high resolution)
            self.aux_classifier_2 = nn.Sequential(
                nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(features // 2),
                nn.ReLU(True),
                nn.Conv2d(features // 2, num_classes, kernel_size=1, stride=1, padding=0)
            )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        # Main output
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * self.patch_size), int(patch_w * self.patch_size)), mode="bilinear", align_corners=True)
        main_out = self.scratch.output_conv2(out)
        # Return auxiliary outputs only during training
        if self.training and self.use_auxiliary:
            # Generate auxiliary outputs at different scales (keep original resolutions)
            aux_out_4 = self.aux_classifier_4(path_4)
            aux_out_3 = self.aux_classifier_3(path_3)
            aux_out_2 = self.aux_classifier_2(path_2)
            
            return {
                'main': main_out,
                'aux_4': aux_out_4,  # Lowest resolution
                'aux_3': aux_out_3,  # Mid-low resolution  
                'aux_2': aux_out_2   # Mid-high resolution
            }
        else:
            return main_out


class DPTHeadLinear(DPTHead):
    """DPT head with a single 1x1 conv classifier instead of the multi-conv bottleneck.
    
    Relies on refinenet fusion blocks for spatial refinement and uses a direct
    linear projection from the fused feature space to class logits.
    """

    def __init__(self, in_channels, features=256, num_classes=2, use_bn=False,
                 out_channels=[256, 512, 1024, 1024], use_clstoken=False,
                 use_auxiliary=True, patch_size=14):
        super().__init__(
            in_channels, features, num_classes, use_bn,
            out_channels, use_clstoken, use_auxiliary, patch_size,
        )
        self.scratch.output_conv1 = nn.Identity()
        self.scratch.output_conv2 = nn.Conv2d(features, num_classes, kernel_size=1)

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = F.interpolate(path_1, (int(patch_h * self.patch_size), int(patch_w * self.patch_size)),
                            mode="bilinear", align_corners=True)
        main_out = self.scratch.output_conv2(out)

        if self.training and self.use_auxiliary:
            aux_out_4 = self.aux_classifier_4(path_4)
            aux_out_3 = self.aux_classifier_3(path_3)
            aux_out_2 = self.aux_classifier_2(path_2)
            return {
                'main': main_out,
                'aux_4': aux_out_4,
                'aux_3': aux_out_3,
                'aux_2': aux_out_2,
            }
        return main_out


class DepthAnythingV2(nn.Module):
    def __init__(
        self, 
        encoder='dinov2_vitl14', 
        features=256, 
        num_classes = 2,
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        use_auxiliary=True,
        head_type='dpt',
    ):
        super(DepthAnythingV2, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'dinov2_vitl14': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        self.patch = 14
        self.encoder = encoder
        self.pretrained = DINOv2(encoder, pretrained=True).eval()
        
        head_cls = DPTHeadLinear if head_type == 'dpt_linear' else DPTHead
        self.depth_head = head_cls(self.pretrained.embed_dim, features, num_classes, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, use_auxiliary=use_auxiliary)
    
    def get_parameter_groups(self, lr_config):
        """
        Get parameter groups for different parts of the model with specified learning rates.
        
        This method categorizes model parameters into logical groups:
        - proj_params: Projection layers (projects, resize_layers, readout_projects)
        - fuse_head_params: Fusion blocks and head layers (scratch layers, refinenet, output layers)
        - aux_params: Auxiliary classifiers for intermediate supervision
        - encoder_params: Pretrained encoder backbone (usually frozen)
        
        Args:
            lr_config (dict): Learning rate configuration for different parameter groups.
                Expected format:
                {
                    'proj_params': {'lr': 1e-5, 'weight_decay': 0.01},
                    'fuse_head_params': {'lr': 1e-4, 'weight_decay': 0.01},
                    'aux_params': {'lr': 5e-5, 'weight_decay': 0.01},
                    'encoder_params': {'lr': 0.0, 'weight_decay': 0.01}  # Usually frozen
                }
                
        Returns:
            list: List of parameter groups for optimizer, each containing:
                - 'params': List of parameters
                - 'name': Group name for identification
                - Additional optimizer parameters (lr, weight_decay, etc.)
        """
        param_groups = []
        all_params = dict(self.named_parameters())
        assigned_params = set()
        
        # Define parameter group mappings with their keywords and default configs
        group_definitions = [
            {
                'name': 'proj_params',
                'keywords': ['projects', 'resize_layers', 'readout_projects'],
                'default_config': {'lr': 1e-5, 'weight_decay': 0.01}
            },
            {
                'name': 'fuse_head_params', 
                'keywords': [
                    'scratch.refinenet', 'scratch.layer', 'scratch.output_conv', 
                    'scratch.bn', 'depth_head.scratch'
                ],
                'default_config': {'lr': 1e-4, 'weight_decay': 0.01}
            },
            {
                'name': 'aux_params',
                'keywords': ['aux_classifier'],
                'default_config': {'lr': 5e-5, 'weight_decay': 0.01}
            },
            {
                'name': 'encoder_params',
                'keywords': ['pretrained'],
                'default_config': {'lr': 0.0, 'weight_decay': 0.01}  # Usually frozen
            }
        ]
        
        # Create parameter groups based on definitions
        for group_def in group_definitions:
            group_name = group_def['name']
            keywords = group_def['keywords']
            default_config = group_def['default_config']
            
            # Find parameters matching this group's keywords
            group_params = []
            for param_name, param in all_params.items():
                if any(keyword in param_name for keyword in keywords):
                    group_params.append(param)
                    assigned_params.add(param_name)
            
            # Add parameter group if it has parameters
            if group_params:
                group_config = lr_config.get(group_name, default_config)
                param_groups.append({
                    'params': group_params,
                    'name': group_name,
                    **group_config
                })
        
        # Handle any remaining unassigned parameters
        remaining_params = []
        for param_name, param in all_params.items():
            if param_name not in assigned_params:
                remaining_params.append(param)
        
        if remaining_params:
            # Use default config or fall back to reasonable defaults
            default_config = lr_config.get('default_params', {'lr': 1e-4, 'weight_decay': 0.01})
            param_groups.append({
                'params': remaining_params,
                'name': 'default_params',
                **default_config
            })
        
        return param_groups
    
    def debug_parameter_groups(self, lr_config=None):
        """
        Debug method to print parameter group assignments and statistics.
        
        Args:
            lr_config (dict, optional): Learning rate configuration. If None, uses defaults.
        """
        if lr_config is None:
            lr_config = {}
            
        print("=" * 60)
        print("DepthAnythingV2 Parameter Group Analysis")
        print("=" * 60)
        
        param_groups = self.get_parameter_groups(lr_config)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Total model parameters: {total_params:,}")
        print(f"Number of parameter groups: {len(param_groups)}")
        print()
        
        for i, group in enumerate(param_groups):
            group_params = sum(p.numel() for p in group['params'])
            percentage = (group_params / total_params) * 100
            
            print(f"Group {i+1}: {group['name']}")
            print(f"  Parameters: {group_params:,} ({percentage:.1f}%)")
            print(f"  Learning rate: {group['lr']}")
            print(f"  Weight decay: {group['weight_decay']}")
            
            # Show some example parameter names
            example_names = []
            for param_name, param in self.named_parameters():
                if param in group['params']:
                    example_names.append(param_name)
                if len(example_names) >= 3:  # Show up to 3 examples
                    break
            
            if example_names:
                print(f"  Example parameters:")
                for name in example_names:
                    param_shape = dict(self.named_parameters())[name].shape
                    print(f"    - {name}: {tuple(param_shape)}")
            print()
        
        print("=" * 60)

    def forward(self, x):
        # print(x.shape)
        B,C,H,W = x.shape
        pad_h = (self.patch - H % self.patch) % self.patch  # pad to mult. of patch
        pad_w = (self.patch - W % self.patch) % self.patch
        if pad_h or pad_w:
            x = F.pad(x, (0,pad_w,0,pad_h))
        # print(x.shape)
        patch_h, patch_w = x.shape[-2] // self.patch, x.shape[-1] // self.patch

        with torch.no_grad():
            features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        
        depth_output = self.depth_head(features, patch_h, patch_w)
        
        # Handle both dictionary (training with auxiliary) and tensor (inference) outputs
        if isinstance(depth_output, dict):
            # Training mode with auxiliary outputs
            result = {}
            for key, output in depth_output.items():
                if key == 'main':
                    # Only resize main output to match input dimensions
                    result[key] = F.interpolate(output, size=(H, W), mode="bilinear", align_corners=True)
                else:
                    # Keep auxiliary outputs at their original resolution for downsampled ground truth comparison
                    result[key] = output
            return result
        else:
            # Inference mode or training without auxiliary
            depth = F.interpolate(depth_output, size=(H, W), mode="bilinear", align_corners=True)
            return depth
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        
        depth = self.forward(image)
        
        depth = F.interpolate(depth.squeeze(1)[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)

class DepthAnythingV2Plus(nn.Module):
    def __init__(
        self, 
        encoder='dinov2_vitl14', 
        features=256, 
        num_classes = 2,
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        use_auxiliary=True,
        head_type='dpt',
    ):
        super(DepthAnythingV2Plus, self).__init__()
        
        self.model = DepthAnythingV2(encoder, features, num_classes, out_channels, use_bn, use_clstoken, use_auxiliary, head_type=head_type)
        weights_path = "model_weights/depth_anything_v2_vitl.pth"
        
        # Load pretrained weights with selective loading
        self.load_pretrained_weights(weights_path)
        
        # Re-initialize classification layers AFTER loading pretrained weights
        # This ensures our careful initialization isn't overwritten by pretrained weights
        # that were trained for depth estimation (1 channel) not segmentation (num_classes)
        self._initialize_classification_layers()
    
    def _initialize_classification_layers(self):
        """
        Initialize classification layers with small weights to prevent gradient explosion.
        
        This method re-initializes the layers that cause scale mismatches when transferring
        from depth estimation (1 channel) to segmentation (num_classes channels).
        """
        head = self.model.depth_head
        
        if isinstance(head, DPTHeadLinear):
            # DPTHeadLinear: output_conv1 is Identity, output_conv2 is a single 1x1 Conv2d
            nn.init.normal_(head.scratch.output_conv2.weight, mean=0.0, std=0.01)
            nn.init.constant_(head.scratch.output_conv2.bias, 0.0)
            print("DepthAnythingV2Plus: Re-initialized DPTHeadLinear classification layers:")
            print(f"  - output_conv2 (1x1): Normal with std=0.01")
        else:
            nn.init.xavier_uniform_(head.scratch.output_conv1.weight, gain=0.1)
            nn.init.constant_(head.scratch.output_conv1.bias, 0.0)
            final_conv = head.scratch.output_conv2[2]
            nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)
            nn.init.constant_(final_conv.bias, 0.0)
            print("DepthAnythingV2Plus: Re-initialized DPTHead classification layers:")
            print(f"  - output_conv1: Xavier uniform with gain=0.1")
            print(f"  - final conv: Normal with std=0.01")
        
        if hasattr(head, 'use_auxiliary') and head.use_auxiliary:
            for name in ('aux_classifier_4', 'aux_classifier_3', 'aux_classifier_2'):
                if hasattr(head, name):
                    final_layer = getattr(head, name)[3]
                    nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
                    nn.init.constant_(final_layer.bias, 0.0)
            print(f"  - aux classifiers: Normal with std=0.01")
    
    def get_parameter_groups(self, lr_config):
        """
        Delegate parameter groups creation to the underlying model.
        
        Args:
            lr_config (dict): Learning rate configuration for different parameter groups.
                
        Returns:
            list: List of parameter groups for optimizer.
        """
        return self.model.get_parameter_groups(lr_config)
    
    def debug_parameter_groups(self, lr_config=None):
        """
        Delegate parameter groups debugging to the underlying model.
        
        Args:
            lr_config (dict, optional): Learning rate configuration. If None, uses defaults.
        """
        return self.model.debug_parameter_groups(lr_config)
    
    def load_pretrained_weights(self, weights_path):
        """
        Load pretrained weights while skipping incompatible layers.
        Specifically skips the final classification layer (scratch.output_conv2).
        """
        try:
            # Load the pretrained state dict
            pretrained_dict = torch.load(weights_path, map_location='cuda')
            
            # Get current model state dict
            model_dict = self.model.state_dict()
            
            # Filter out incompatible keys (final classification layer)
            incompatible_keys = []
            compatible_dict = {}
            
            for key, value in pretrained_dict.items():
                if key in model_dict:
                    # Check if shapes match
                    if model_dict[key].shape == value.shape:
                        compatible_dict[key] = value
                    else:
                        incompatible_keys.append(key)
                        print(f"Skipping layer {key}: shape mismatch "
                              f"(pretrained: {value.shape}, current: {model_dict[key].shape})")
                else:
                    incompatible_keys.append(key)
                    print(f"Skipping layer {key}: not found in current model")
            
            # Check for keys in current model that are not in pretrained
            missing_keys = []
            for key in model_dict.keys():
                if key not in pretrained_dict:
                    missing_keys.append(key)
            
            # Load the compatible weights
            model_dict.update(compatible_dict)
            self.model.load_state_dict(model_dict)
            
            # Print loading summary
            print(f"Successfully loaded pretrained weights from {weights_path}")
            print(f"Loaded {len(compatible_dict)} layers")
            print(f"Skipped {len(incompatible_keys)} incompatible layers: {incompatible_keys}")
            if missing_keys:
                print(f"Missing {len(missing_keys)} layers (will use random initialization): {missing_keys}")
            
        except FileNotFoundError:
            print(f"Warning: Pretrained weights file not found at {weights_path}")
            print("Model will use random initialization")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Model will use random initialization")
    
    def forward(self, x):
        return self.model(x)
