import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import kornia
import kornia.morphology as morph
import kornia.filters as filters
from typing import Optional, List
from scipy.optimize import linear_sum_assignment


def _hungarian_assignment(cost_matrix: torch.Tensor):
    """
    Pure-torch Jonker-Volgenant Hungarian algorithm. Drop-in replacement
    for scipy.optimize.linear_sum_assignment that requires only torch.
    """
    cost = cost_matrix.detach().float().cpu()
    orig_n, orig_m = cost.shape

    # Pad to square
    size = max(orig_n, orig_m)
    if orig_n != orig_m:
        pad_val = cost.abs().max() + 1.0
        a = torch.full((size, size), pad_val, dtype=cost.dtype)
        a[:orig_n, :orig_m] = cost
    else:
        a = cost.clone()

    # All arrays are 1-indexed (index 0 is a sentinel)
    INF = float("inf")
    u = torch.zeros(size + 1, dtype=cost.dtype)   # row potentials
    v = torch.zeros(size + 1, dtype=cost.dtype)   # col potentials
    p = torch.zeros(size + 1, dtype=torch.long)   # p[j] = row matched to col j (0 = free)
    way = torch.zeros(size + 1, dtype=torch.long)  # predecessor column in augmenting path

    for i in range(1, size + 1):
        p[0] = i
        j0 = 0  # virtual column linked to row i
        minv = torch.full((size + 1,), INF, dtype=cost.dtype)
        used = torch.zeros(size + 1, dtype=torch.bool)

        while True:
            used[j0] = True
            i0 = p[j0].item()
            delta = INF
            j1 = -1

            for j in range(1, size + 1):
                if used[j]:
                    continue
                cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                if cur.item() < minv[j].item():
                    minv[j] = cur
                    way[j] = j0
                if minv[j].item() < delta:
                    delta = minv[j].item()
                    j1 = j

            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0].item() == 0:
                break

        # Trace augmenting path back to virtual column 0
        while j0 != 0:
            p[j0] = p[way[j0].item()]
            j0 = way[j0].item()

    # Extract assignment: p[j] = row assigned to column j (1-indexed)
    row_ind = []
    col_ind = []
    for j in range(1, size + 1):
        r = p[j].item() - 1   # 0-indexed row
        c = j - 1              # 0-indexed col
        if r < orig_n and c < orig_m:
            row_ind.append(r)
            col_ind.append(c)

    return (
        torch.tensor(row_ind, dtype=torch.long),
        torch.tensor(col_ind, dtype=torch.long),
    )


FocalLoss = smp.losses.FocalLoss
LovaszLoss = smp.losses.LovaszLoss
DiceLoss = smp.losses.DiceLoss
SoftCrossEntropyLoss = smp.losses.SoftCrossEntropyLoss
SoftBCEWithLogitsLoss = smp.losses.SoftBCEWithLogitsLoss
TverskyLoss = smp.losses.TverskyLoss
MSELoss = torch.nn.MSELoss



class GradMatchLoss(torch.nn.Module):
    """
    Multi-class gradient-matching loss (Depth-Anything V2 style) for segmentation.
    
    Promotes sharp recovery of boundaries and fine structures by aligning the gradients
    of predicted probability maps with ground truth one-hot encoded targets.
    Applied at multiple scales to capture both fine and coarse transitions.
    
    Uses per-scale weighting to emphasize fine-resolution edges, following MiDaS-style
    multi-scale gradient matching where finer scales receive higher weight.
    
    Original use: Depth estimation, adapted here for multi-class segmentation
    
    Args:
        weight: Global weight multiplier for this loss
        scales: Tuple of downsampling factors for multi-scale application
        ignore_index: Optional class index to ignore in loss computation
        scale_weights: How to weight different scales. Options:
            'uniform': Equal weight for all scales
            'inverse': Weight by 1/scale (emphasizes fine scales)
            'exponential': Weight by 1/2^(scale_idx) (stronger emphasis on fine scales)
            or a tuple of weights matching scales length
        boundary_threshold: Threshold for identifying boundary pixels based on
            target gradient magnitude. Only boundary pixels contribute to loss,
            avoiding dilution from interior pixels. Set to 0.0 to use all pixels.
            Default: 0.1 (recommended for normalized gradients)
    """
    def __init__(self, weight=10, scales=(1, 2, 4, 8), ignore_index=None,
                 scale_weights='exponential', boundary_threshold=0.1):
        super().__init__()
        self.weight = weight
        self.scales = scales
        self.ignore_index = ignore_index
        self.boundary_threshold = boundary_threshold
        
        # Compute per-scale weights
        if isinstance(scale_weights, (tuple, list)):
            assert len(scale_weights) == len(scales), "scale_weights must match scales length"
            self.scale_weights = torch.tensor(scale_weights, dtype=torch.float32)
        elif scale_weights == 'uniform':
            self.scale_weights = torch.ones(len(scales), dtype=torch.float32)
        elif scale_weights == 'inverse':
            # Weight by 1/scale - emphasizes finer scales
            self.scale_weights = torch.tensor([1.0 / s for s in scales], dtype=torch.float32)
        elif scale_weights == 'exponential':
            # Weight by 1/2^(idx) - stronger emphasis on fine scales
            self.scale_weights = torch.tensor([1.0 / (2 ** i) for i in range(len(scales))], dtype=torch.float32)
        else:
            raise ValueError(f"Unknown scale_weights: {scale_weights}")
        
        # Normalize weights to sum to 1
        self.scale_weights = self.scale_weights / self.scale_weights.sum()

    @staticmethod
    def _grad(t):
        gx, gy = filters.spatial_gradient(t, mode="sobel", order=1).unbind(dim=2)
        return gx, gy

    def forward(self, logits, target):
        """
        logits : (B,C,H,W)  — raw network output
        target : (B,H,W)    — int labels
        """
        probs = F.softmax(logits, 1)

        onehot = F.one_hot(target, logits.size(1)).permute(0,3,1,2).float()
        mask   = (target != self.ignore_index).unsqueeze(1) if self.ignore_index is not None else None

        total = 0.
        scale_weights = self.scale_weights.to(probs.device)
        
        for scale_idx, s in enumerate(self.scales):
            if s > 1:
                # Use consistent downsampling for both predictions and targets
                # avg_pool2d treats both as continuous distributions, which is appropriate
                # for gradient matching (we care about the gradient of the distribution)
                probs_s  = F.avg_pool2d(probs,  s, stride=s)
                onehot_s = F.avg_pool2d(onehot, s, stride=s)
                mask_s = F.avg_pool2d(mask.float(), s, stride=s) > 0.99 if mask is not None else None
            else:
                probs_s, onehot_s = probs, onehot
                mask_s = mask

            # Compute gradients BEFORE applying mask
            gx_p, gy_p = self._grad(probs_s)
            gx_t, gy_t = self._grad(onehot_s)

            # Identify boundary pixels where target has significant gradient
            # This focuses the loss on edges rather than diluting over interior pixels
            target_grad_mag = gx_t.abs() + gy_t.abs()  # (B, C, H, W)
            
            if self.boundary_threshold > 0:
                # Boundary mask: pixels where target gradient exceeds threshold
                boundary_mask = (target_grad_mag > self.boundary_threshold).float()
            else:
                # Use all pixels (original behavior)
                boundary_mask = torch.ones_like(target_grad_mag)

            # Apply ignore mask if specified
            if mask_s is not None:
                mask_expanded = mask_s.expand_as(gx_p)
                boundary_mask = boundary_mask * mask_expanded

            # Compute loss only over boundary pixels
            num_boundary = boundary_mask.sum().clamp(min=1.0)
            
            # Gradient differences weighted by boundary mask
            gx_diff = (gx_p - gx_t).abs() * boundary_mask
            gy_diff = (gy_p - gy_t).abs() * boundary_mask
            
            # Normalize by number of boundary pixels (not all pixels)
            scale_loss = (gx_diff.sum() + gy_diff.sum()) / num_boundary

            # Apply per-scale weight
            total += scale_weights[scale_idx] * scale_loss

        return self.weight * total

class CombinedSegmentationLoss(_Loss):
    """
    Combined loss function that integrates multiple loss components for fine-grained segmentation.
    
    Combines:
    1. Dice Loss (overlap-based)
    2. Cross-Entropy Loss (pixel-wise classification)
    3. Gradient Matching Loss (boundary preservation)
    4. Optional: Focal Loss (hard example mining)
    
    Args:
        dice_weight (float): Weight for Dice loss. Default: 0.4
        ce_weight (float): Weight for Cross-Entropy loss. Default: 0.3
        gradient_weight (float): Weight for Gradient Matching loss. Default: 0.2
        focal_weight (float): Weight for Focal loss. Default: 0.1
        use_focal (bool): Whether to include Focal loss. Default: True
        class_weights (Optional[List[float]]): Class weights for CE and Focal losses. Default: None
    """
    
    def __init__(self,
                 dice_weight: float = 0.4,
                 ce_weight: float = 0.3,
                 gradient_weight: float = 0.2,
                 focal_weight: float = 0.1,
                 use_focal: bool = True,
                 class_weights: Optional[List[float]] = None,
                 **gradient_loss_kwargs):
        super(CombinedSegmentationLoss, self).__init__()
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.gradient_weight = gradient_weight
        self.focal_weight = focal_weight
        self.use_focal = use_focal
        
        # Initialize loss functions
        self.dice_loss = DiceLoss(mode='multiclass')
        self.gradient_loss = GradMatchLoss(**gradient_loss_kwargs)
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
        
        if use_focal:
            self.focal_loss = FocalLoss(mode='multiclass')
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Forward pass returning both total loss and individual components.
        
        Args:
            input: Model predictions [B, C, H, W]
            target: Ground truth labels [B, H, W]
            
        Returns:
            dict: Dictionary containing 'loss' and individual loss components
        """
        # Compute individual losses
        dice_loss = self.dice_loss(input, target)
        ce_loss = F.cross_entropy(input, target, weight=self.class_weights)
        gradient_loss = self.gradient_loss(input, target)
        
        # Combine losses
        total_loss = (self.dice_weight * dice_loss + 
                     self.ce_weight * ce_loss + 
                     self.gradient_weight * gradient_loss)
        
        loss_dict = {
            'loss': total_loss,
            'dice_loss': dice_loss,
            'ce_loss': ce_loss,
            'gradient_loss': gradient_loss
        }
        
        if self.use_focal:
            focal_loss = self.focal_loss(input, target)
            total_loss += self.focal_weight * focal_loss
            loss_dict['loss'] = total_loss
            loss_dict['focal_loss'] = focal_loss
        
        return total_loss


class WeightedDice(_Loss):
    def __init__(self, weights, eps: float = 1e-7, smooth: float = 0.0, F_BETA: float = 0.9, GAMMA = 2.0):
        super(WeightedDice, self).__init__()
        # self.weights = torch.tensor(weights, device = self.device)
        self.register_buffer('weights', torch.tensor(weights))
        self.smooth = smooth
        self.eps = eps
        self.F_BETA = F_BETA
        self.GAMMA = GAMMA
        

    def compute_score(self, y_pred, y_true, smooth, eps, dims):
        assert y_pred.size() == y_true.size(), f"pred shape is {y_pred.shape} and target shape is {y_true.shape}"
        if dims is not None:
            intersection = torch.sum(y_pred * y_true, dim=dims)
            cardinality = torch.sum(self.F_BETA*y_pred + (1.0-self.F_BETA)*y_true, dim=dims)
        else:
            intersection = torch.sum(y_pred * y_true)
            cardinality = torch.sum(self.F_BETA*y_pred + (1.0-self.F_BETA)*y_true)

        dice_score = ( intersection + smooth) / (cardinality + smooth).clamp_min(eps)
        return dice_score
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        weights = self.weights.to(y_pred.device)
        y_pred = y_pred.log_softmax(dim=1).exp()
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        y_true = y_true.view(bs, -1)
        y_pred = y_pred.view(bs, num_classes, -1)
        y_true = F.one_hot((y_true).to(torch.long), num_classes)
        y_true = y_true.permute(0, 2, 1)
        
        scores = self.compute_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)
        losses = (1.0 - scores)**(self.GAMMA)
        weighted_losses = losses * weights
        # weighted_losses = [weight*loss for weight, loss in zip(self.weights, losses)]
        weighted_dice_loss = torch.sum(weighted_losses)/(torch.sum(weights))

        return weighted_dice_loss

class PixelWeightedCrossEntropyLoss(_Loss):
    """
    Pixel-weighted cross entropy loss that emphasizes boundary regions.
    Based on the approach from "Continental-scale building detection from high resolution satellite imagery"
    but generalized for any segmentation task where pixels are weighted more around the boundaries 
    using distance transforms. Uses Kornia for GPU-accelerated operations.
    """
    def __init__(self, 
                 w0: float = 10.0,  # weight parameter for boundary emphasis
                 sigma: float = 5.0,  # standard deviation for Gaussian weighting
                 class_weights: Optional[List[float]] = None,  # optional class weights
                 ignore_index: int = -100):
        super(PixelWeightedCrossEntropyLoss, self).__init__()
        self.w0 = w0
        self.sigma = sigma
        self.ignore_index = ignore_index
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
    
    def compute_weight_map(self, target: torch.Tensor) -> torch.Tensor:
        """
        Compute pixel weight map emphasizing boundaries between object instances using
        Gaussian convolution approach from Continental-Scale Building Detection paper.
        This is much more efficient than iterative distance computation.
        
        Args:
            target: Ground truth segmentation mask [B, H, W]
            
        Returns:
            weight_map: Pixel weights [B, H, W] with higher weights at boundaries
        """
        batch_size, height, width = target.shape
        device = target.device
        weight_maps = torch.ones_like(target, dtype=torch.float32, device=device)
        
        # Create Gaussian kernel for convolution
        kernel_size = int(self.sigma * 6) + 1  # 6 sigma covers ~99.7% of distribution
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        
        # Create 2D Gaussian kernel
        gaussian_kernel = self._create_gaussian_kernel(kernel_size, self.sigma, device)
        
        for b in range(batch_size):
            target_b = target[b]  # [H, W]
            
            # Create edge map - detect boundaries within building regions
            edge_map = self._create_edge_map(target_b, device)
            
            if edge_map.sum() > 0:
                # Apply Gaussian convolution to edge map
                # Add batch and channel dimensions for conv2d: [1, 1, H, W]
                edge_4d = edge_map.unsqueeze(0).unsqueeze(0).float()
                
                # Apply Gaussian convolution
                with torch.no_grad():
                    weight_map = F.conv2d(
                        edge_4d, 
                        gaussian_kernel.unsqueeze(0).unsqueeze(0), 
                        padding=kernel_size//2
                    ).squeeze()  # Remove batch and channel dims
                
                # Scale the weights: base weight of 1.0 + weighted boundary emphasis
                weight_map = 1.0 + self.w0 * weight_map
                
                weight_maps[b] = weight_map
            else:
                # No boundaries found, use uniform weighting
                weight_maps[b] = torch.ones_like(target_b, dtype=torch.float32)
        
        return weight_maps
    
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
        """Create 2D Gaussian kernel for convolution."""
        # Create coordinate grids
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
        coords = coords - kernel_size // 2
        
        # Create 2D coordinate grids
        y_coords, x_coords = torch.meshgrid(coords, coords, indexing='ij')
        
        # Compute Gaussian kernel
        kernel = torch.exp(-(x_coords**2 + y_coords**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()  # Normalize
        
        return kernel
    
    def _create_edge_map(self, target: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Create edge map detecting boundaries within foreground regions.
        Uses morphological operations and gradient detection.
        """
        # Create binary mask for foreground (non-background)
        foreground_mask = (target > 0).float()  # [H, W]
        
        if foreground_mask.sum() == 0:
            return torch.zeros_like(target, dtype=torch.float32)
        
        # Add dimensions for morphological operations [1, 1, H, W]
        foreground_4d = foreground_mask.unsqueeze(0).unsqueeze(0)
        
        # Create structuring element (3x3 kernel)
        kernel = torch.ones(3, 3, device=device)
        
        # Morphological operations to find boundaries
        dilated = morph.dilation(foreground_4d, kernel)
        eroded = morph.erosion(foreground_4d, kernel)
        
        # Morphological gradient (boundaries)
        morph_boundaries = (dilated - eroded).squeeze()  # [H, W]
        
        # Also detect internal boundaries using spatial gradient
        target_4d = target.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Compute gradients using Kornia's spatial_gradient function
        gradients = filters.spatial_gradient(target_4d, mode='diff')  # [1, 1, 2, H, W]
        grad_x = gradients[0, 0, 0]  # [H, W] - x gradient
        grad_y = gradients[0, 0, 1]  # [H, W] - y gradient
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Combine morphological and gradient boundaries
        gradient_boundaries = (gradient_magnitude > 0.1).float()
        
        # Final edge map: focus on boundaries within foreground regions
        edge_map = torch.clamp(morph_boundaries + gradient_boundaries, 0, 1)
        edge_map = edge_map * foreground_mask  # Only within foreground regions
        
        return edge_map
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of pixel-weighted cross entropy loss.
        
        Args:
            input: Model predictions [B, C, H, W]
            target: Ground truth labels [B, H, W]
            
        Returns:
            loss: Weighted cross entropy loss
        """
        # Compute pixel weight map based on boundaries
        weight_map = self.compute_weight_map(target)  # [B, H, W]
        
        # Compute standard cross entropy loss (unreduced)
        ce_loss = F.cross_entropy(
            input, 
            target, 
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction='none'
        )  # [B, H, W]
        
        # Apply pixel weights
        weighted_loss = ce_loss * weight_map
        
        # Return mean loss
        if self.ignore_index != -100:
            # Only average over valid pixels
            valid_mask = (target != self.ignore_index)
            if valid_mask.sum() > 0:
                return weighted_loss[valid_mask].mean()
            else:
                return torch.tensor(0.0, device=input.device, requires_grad=True)
        else:
            return weighted_loss.mean()


class AdaptivePixelWeightedCrossEntropyLoss(_Loss):
    """
    Adaptive version that computes weights per class channel using Kornia.
    This version creates separate weight maps for each class to handle multi-class scenarios better.
    Generalized for any multi-class segmentation task, not just building detection.
    """
    def __init__(self, 
                 w0: float = 10.0,
                 sigma: float = 5.0,
                 class_weights: Optional[List[float]] = None,
                 ignore_index: int = -100,
                 per_class_weighting: bool = True):
        super(AdaptivePixelWeightedCrossEntropyLoss, self).__init__()
        self.w0 = w0
        self.sigma = sigma
        self.ignore_index = ignore_index
        self.per_class_weighting = per_class_weighting
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
    
    def compute_class_weight_maps(self, target: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Compute weight maps for each class separately emphasizing boundaries between instances
        using Gaussian convolution approach from Continental-Scale Building Detection paper.
        This is much more efficient than iterative distance computation.
        
        Args:
            target: Ground truth segmentation mask [B, H, W]
            num_classes: Number of classes
            
        Returns:
            weight_maps: Class-specific pixel weights [B, C, H, W] with higher weights at boundaries
        """
        batch_size, height, width = target.shape
        device = target.device
        weight_maps = torch.ones(batch_size, num_classes, height, width, 
                                dtype=torch.float32, device=device)
        
        # Create Gaussian kernel for convolution
        kernel_size = int(self.sigma * 6) + 1  # 6 sigma covers ~99.7% of distribution
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        
        # Create 2D Gaussian kernel
        gaussian_kernel = self._create_gaussian_kernel(kernel_size, self.sigma, device)
        
        for b in range(batch_size):
            target_b = target[b]  # [H, W]
            
            for class_idx in range(num_classes):
                # Create class-specific edge map
                edge_map = self._create_class_edge_map(target_b, class_idx, device)
                
                if edge_map.sum() > 0:
                    # Apply Gaussian convolution to edge map
                    # Add batch and channel dimensions for conv2d: [1, 1, H, W]
                    edge_4d = edge_map.unsqueeze(0).unsqueeze(0).float()
                    
                    # Apply Gaussian convolution
                    with torch.no_grad():
                        weight_map = F.conv2d(
                            edge_4d, 
                            gaussian_kernel.unsqueeze(0).unsqueeze(0), 
                            padding=kernel_size//2
                        ).squeeze()  # Remove batch and channel dims
                    
                    # Scale the weights: base weight of 1.0 + weighted boundary emphasis
                    weight_map = 1.0 + self.w0 * weight_map
                    
                    weight_maps[b, class_idx] = weight_map
                else:
                    # No boundaries found for this class, use uniform weighting
                    weight_maps[b, class_idx] = torch.ones_like(target_b, dtype=torch.float32)
        
        return weight_maps
    
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
        """Create 2D Gaussian kernel for convolution."""
        # Create coordinate grids
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
        coords = coords - kernel_size // 2
        
        # Create 2D coordinate grids
        y_coords, x_coords = torch.meshgrid(coords, coords, indexing='ij')
        
        # Compute Gaussian kernel
        kernel = torch.exp(-(x_coords**2 + y_coords**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()  # Normalize
        
        return kernel
    
    def _create_class_edge_map(self, target: torch.Tensor, class_idx: int, device: torch.device) -> torch.Tensor:
        """
        Create edge map for a specific class, detecting boundaries between instances
        and inter-class boundaries.
        """
        # Create binary mask for current class
        class_mask = (target == class_idx).float()  # [H, W]
        
        if class_mask.sum() == 0:
            return torch.zeros_like(target, dtype=torch.float32)
        
        # Add dimensions for morphological operations [1, 1, H, W]
        class_4d = class_mask.unsqueeze(0).unsqueeze(0)
        
        # Create structuring element (3x3 kernel)
        kernel = torch.ones(3, 3, device=device)
        
        # Morphological operations to find boundaries
        dilated = morph.dilation(class_4d, kernel)
        eroded = morph.erosion(class_4d, kernel)
        
        # Morphological gradient (boundaries)
        morph_boundaries = (dilated - eroded).squeeze()  # [H, W]
        
        # Detect inter-class boundaries using spatial gradient
        target_4d = target.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Compute gradients using Kornia's spatial_gradient function
        gradients = filters.spatial_gradient(target_4d, mode='diff')  # [1, 1, 2, H, W]
        grad_x = gradients[0, 0, 0]  # [H, W] - x gradient
        grad_y = gradients[0, 0, 1]  # [H, W] - y gradient
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Inter-class boundaries (where gradient is high)
        inter_class_boundaries = (gradient_magnitude > 0.1).float()
        
        # Combine morphological and gradient boundaries
        edge_map = torch.clamp(morph_boundaries + inter_class_boundaries, 0, 1)
        
        # Focus on boundaries relevant to current class:
        # 1. Boundaries within class regions (intra-class)
        # 2. Boundaries at the edge of class regions (inter-class)
        class_dilated = dilated.squeeze()  # Slightly expanded class region
        edge_map = edge_map * class_dilated  # Keep boundaries near this class
        
        return edge_map
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with per-class weight computation.
        
        Args:
            input: Model predictions [B, C, H, W]
            target: Ground truth labels [B, H, W]
            
        Returns:
            loss: Weighted cross entropy loss
        """
        batch_size, num_classes, height, width = input.shape
        
        if self.per_class_weighting:
            # Compute per-class weight maps
            class_weight_maps = self.compute_class_weight_maps(target, num_classes)  # [B, C, H, W]
            
            # Convert target to one-hot for per-class loss computation
            target_one_hot = F.one_hot(target.clamp(0, num_classes-1), num_classes)  # [B, H, W, C]
            target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
            
            # Compute log probabilities
            log_probs = F.log_softmax(input, dim=1)  # [B, C, H, W]
            
            # Compute per-class losses
            class_losses = -target_one_hot * log_probs  # [B, C, H, W]
            
            # Apply class-specific weights
            weighted_losses = class_losses * class_weight_maps  # [B, C, H, W]
            
            # Sum over classes and average over spatial dimensions
            total_loss = weighted_losses.sum(dim=1)  # [B, H, W]
            
            # Handle ignore index
            if self.ignore_index != -100:
                valid_mask = (target != self.ignore_index)
                if valid_mask.sum() > 0:
                    return total_loss[valid_mask].mean()
                else:
                    return torch.tensor(0.0, device=input.device, requires_grad=True)
            else:
                return total_loss.mean()
        else:
            # Use simpler boundary-based weighting
            weight_map = self.compute_weight_map(target)
            ce_loss = F.cross_entropy(input, target, weight=self.class_weights, 
                                    ignore_index=self.ignore_index, reduction='none')
            weighted_loss = ce_loss * weight_map
            
            if self.ignore_index != -100:
                valid_mask = (target != self.ignore_index)
                if valid_mask.sum() > 0:
                    return weighted_loss[valid_mask].mean()
                else:
                    return torch.tensor(0.0, device=input.device, requires_grad=True)
            else:
                return weighted_loss.mean()
    
    def compute_weight_map(self, target: torch.Tensor) -> torch.Tensor:
        """
        Compute pixel weight map emphasizing boundaries between object instances using
        Gaussian convolution approach from Continental-Scale Building Detection paper.
        This is much more efficient than iterative distance computation.
        
        Args:
            target: Ground truth segmentation mask [B, H, W]
            
        Returns:
            weight_map: Pixel weights [B, H, W] with higher weights at boundaries
        """
        batch_size, height, width = target.shape
        device = target.device
        weight_maps = torch.ones_like(target, dtype=torch.float32, device=device)
        
        # Create Gaussian kernel for convolution
        kernel_size = int(self.sigma * 6) + 1  # 6 sigma covers ~99.7% of distribution
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        
        # Create 2D Gaussian kernel
        gaussian_kernel = self._create_gaussian_kernel(kernel_size, self.sigma, device)
        
        for b in range(batch_size):
            target_b = target[b]  # [H, W]
            
            # Create edge map - detect boundaries within building regions
            edge_map = self._create_edge_map(target_b, device)
            
            if edge_map.sum() > 0:
                # Apply Gaussian convolution to edge map
                # Add batch and channel dimensions for conv2d: [1, 1, H, W]
                edge_4d = edge_map.unsqueeze(0).unsqueeze(0).float()
                
                # Apply Gaussian convolution
                with torch.no_grad():
                    weight_map = F.conv2d(
                        edge_4d, 
                        gaussian_kernel.unsqueeze(0).unsqueeze(0), 
                        padding=kernel_size//2
                    ).squeeze()  # Remove batch and channel dims
                
                # Scale the weights: base weight of 1.0 + weighted boundary emphasis
                weight_map = 1.0 + self.w0 * weight_map
                
                weight_maps[b] = weight_map
            else:
                # No boundaries found, use uniform weighting
                weight_maps[b] = torch.ones_like(target_b, dtype=torch.float32)
        
        return weight_maps
    
    def _create_edge_map(self, target: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Create edge map detecting boundaries within foreground regions.
        Uses morphological operations and gradient detection.
        """
        # Create binary mask for foreground (non-background)
        foreground_mask = (target > 0).float()  # [H, W]
        
        if foreground_mask.sum() == 0:
            return torch.zeros_like(target, dtype=torch.float32)
        
        # Add dimensions for morphological operations [1, 1, H, W]
        foreground_4d = foreground_mask.unsqueeze(0).unsqueeze(0)
        
        # Create structuring element (3x3 kernel)
        kernel = torch.ones(3, 3, device=device)
        
        # Morphological operations to find boundaries
        dilated = morph.dilation(foreground_4d, kernel)
        eroded = morph.erosion(foreground_4d, kernel)
        
        # Morphological gradient (boundaries)
        morph_boundaries = (dilated - eroded).squeeze()  # [H, W]
        
        # Also detect internal boundaries using spatial gradient
        target_4d = target.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Compute gradients using Kornia's spatial_gradient function
        gradients = filters.spatial_gradient(target_4d, mode='diff')  # [1, 1, 2, H, W]
        grad_x = gradients[0, 0, 0]  # [H, W] - x gradient
        grad_y = gradients[0, 0, 1]  # [H, W] - y gradient
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Combine morphological and gradient boundaries
        gradient_boundaries = (gradient_magnitude > 0.1).float()
        
        # Final edge map: focus on boundaries within foreground regions
        edge_map = torch.clamp(morph_boundaries + gradient_boundaries, 0, 1)
        edge_map = edge_map * foreground_mask  # Only within foreground regions
        
        return edge_map

class GaussianEdgeWeightedCrossEntropyLoss(_Loss):
    """
    Cross-entropy loss with per-pixel weights computed on-the-fly:

        ω(i) = w_c + c · (G_σ * E)(i)

    where
        E(i)    = 1  if pixel i is on an instance boundary, else 0
        G_σ     = 2-D Gaussian (std = σ)
        w_c     = static class-balance weight for class c = y(i)
        c       = scalar scale factor

    This matches the “alternative weighting scheme” (steps 1-2) in the paper.
    It is differentiable, GPU-friendly, and compatible with data augmentation.
    """

    def __init__(
        self,
        sigma: float = 3.0,                  # σ in the paper
        scale_c: float = 20.0,              # c in the paper
        class_weights = None,   # w_c (inverse class freq.)
        ignore_index: int = -100,
    ):
        super().__init__()
        self.sigma = float(sigma)
        self.scale_c = float(scale_c)
        self.ignore_index = ignore_index

        if class_weights is not None:
            weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", weight_tensor)
        else:
            self.class_weights = None  # treated as w_c = 1 in forward()

        # cache Gaussian kernel per device / dtype
        self._kernel_cache = {}

    # ------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------
    @staticmethod
    def _boundary_mask(lbl: torch.Tensor) -> torch.Tensor:
        """
        lbl : (H,W) int tensor of instance or semantic labels
        returns binary edge mask (H,W)
        """
        # Kornia spatial_gradient returns (B,C,2,H,W); we use magnitude>0
        grad = filters.spatial_gradient(lbl.float().unsqueeze(0).unsqueeze(0),
                                        mode="diff")[0, 0]
        return (grad.abs().sum(0) > 0).float()   # 1 where label changes

    def _gaussian_kernel(self, device, dtype) -> torch.Tensor:
        """Return cached 2-D Gaussian kernel G_σ."""
        key = (device, dtype)
        if key in self._kernel_cache:
            return self._kernel_cache[key]

        radius = int(3 * self.sigma)
        ksize = 2 * radius + 1
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel1d = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        kernel1d = kernel1d / kernel1d.sum()
        kernel2d = kernel1d[:, None] * kernel1d[None, :]
        self._kernel_cache[key] = kernel2d
        return kernel2d

    # ------------------------------------------------------------
    def _weight_map(self, target: torch.Tensor) -> torch.Tensor:
        """
        target : (B,H,W) int labels
        returns : (B,H,W) per-pixel weights ω
        """
        B, H, W = target.shape
        device, dtype = target.device, torch.float32
        kernel = self._gaussian_kernel(device, dtype).unsqueeze(0).unsqueeze(0)

        # build boundary masks for the whole batch (vectorised)
        edges = torch.stack([self._boundary_mask(t) for t in target], dim=0)   # (B,H,W)
        edges = edges.unsqueeze(1)                                             # (B,1,H,W)

        # Gaussian blur (depth-wise conv)
        blurred = F.conv2d(edges, kernel, padding=kernel.shape[-1]//2)         # (B,1,H,W)
        blurred = blurred.squeeze(1)                                           # (B,H,W)

        weights = self.scale_c * blurred                                        # c · conv
        if self.class_weights is not None:
            weights += self.class_weights[target]                              # + w_c
        else:
            weights += 1.0                                                     # default w_c

        return weights

    # ------------------------------------------------------------
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits : (B,C,H,W) raw network outputs
        target : (B,H,W)   integer labels
        """
        # build per-pixel weights
        ω = self._weight_map(target)                                           # (B,H,W)

        # unreduced CE
        loss = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="none",
        )                                                                       # (B,H,W)

        if self.ignore_index != -100:
            valid = target != self.ignore_index
            loss = loss[valid] * ω[valid]
        else:
            loss = loss * ω

        return loss.mean()


class FocalHeatmapLoss(nn.Module):
    """
    Focal Loss adapted for heatmap regression to produce sharper centroid predictions.
    Focuses training on hard examples and suppresses easy background pixels.
    """
    
    def __init__(self, alpha=2.0, beta=4.0, reduction='mean'):
        """
        Args:
            alpha: Controls the importance of positive vs negative pixels
            beta: Focusing parameter (higher beta = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted heatmap [B, H, W]
            target: Ground truth heatmap [B, H, W]
        """
        # Ensure predictions are in valid range [0, 1]
        pred = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)
        
        # Positive and negative weights
        pos_weight = torch.pow(1 - pred, self.alpha)
        neg_weight = torch.pow(pred, self.alpha)
        
        # Focus on hard examples
        pos_loss = torch.pow(1 - pred, self.beta) * torch.log(pred)
        neg_loss = torch.pow(pred, self.beta) * torch.log(1 - pred)
        
        # Combine losses
        loss = -(target * pos_weight * pos_loss + (1 - target) * neg_weight * neg_loss)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class PeakAwareLoss(nn.Module):
    """
    Loss function that emphasizes peaks in heatmaps.
    Combines standard regression loss with peak-specific penalty.
    """
    
    def __init__(self, peak_threshold=0.5, peak_weight=2.0, base_loss='l1', reduction='mean'):
        """
        Args:
            peak_threshold: Values above this are considered peaks
            peak_weight: Weight multiplier for peak regions
            base_loss: 'mse' or 'l1'
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.peak_threshold = peak_threshold
        self.peak_weight = peak_weight
        self.reduction = reduction
        
        if base_loss == 'mse':
            self.base_loss_fn = nn.MSELoss(reduction='none')
        elif base_loss == 'l1':
            self.base_loss_fn = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported base_loss: {base_loss}")
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted heatmap [B, H, W]
            target: Ground truth heatmap [B, H, W]
        """
        # Compute base loss
        base_loss = self.base_loss_fn(pred, target)
        
        # Identify peak regions in ground truth
        peak_mask = target > self.peak_threshold
        
        # Apply higher weight to peak regions
        weighted_loss = torch.where(peak_mask, base_loss * self.peak_weight, base_loss)
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class CenterNetLoss(nn.Module):
    """
    CenterNet-style loss for object centroid detection.
    Uses focal loss with additional penalties for offset errors.
    """
    
    def __init__(self, alpha=2.0, beta=4.0, reduction='mean'):
        """
        Args:
            alpha: Focal loss alpha parameter
            beta: Focal loss beta parameter  
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted heatmap [B, H, W]
            target: Ground truth heatmap [B, H, W]
        """
        # Normalize predictions to [0, 1]
        pred = torch.sigmoid(pred) if pred.max() > 1.0 else pred
        pred = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)
        
        # Separate positive and negative samples
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()
        
        # Number of positive samples per batch item
        num_pos = pos_mask.sum(dim=[1, 2], keepdim=True)
        num_pos = torch.clamp(num_pos, min=1)
        
        # Positive loss (focal loss for positives)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_mask
        pos_loss = pos_loss.sum(dim=[1, 2], keepdim=True) / num_pos
        
        # Negative loss (focal loss for negatives, weighted by distance to positive)
        neg_weights = torch.pow(1 - target, self.beta)
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_mask
        neg_loss = neg_loss.sum(dim=[1, 2], keepdim=True) / (neg_mask.sum(dim=[1, 2], keepdim=True) + 1e-7)
        
        # Combine losses
        total_loss = -(pos_loss + neg_loss)
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss


class HybridCentroidLoss(nn.Module):
    """
    Hybrid loss combining multiple objectives for better centroid detection:
    - Base regression loss (L1 or MSE)
    - Focal loss for hard example mining
    - Peak-aware loss for sharp predictions
    """
    
    def __init__(self, 
                 base_weight=1.0, 
                 focal_weight=0.5, 
                 peak_weight=0.3,
                 base_loss='l1',
                 focal_alpha=2.0,
                 focal_beta=4.0,
                 peak_threshold=0.5):
        super().__init__()
        
        self.base_weight = base_weight
        self.focal_weight = focal_weight 
        self.peak_weight = peak_weight
        
        # Base regression loss
        if base_loss == 'mse':
            self.base_loss_fn = nn.MSELoss()
        elif base_loss == 'l1':
            self.base_loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported base_loss: {base_loss}")
        
        # Focal loss for hard examples
        self.focal_loss_fn = FocalHeatmapLoss(alpha=focal_alpha, beta=focal_beta)
        
        # Peak-aware loss for sharpness
        self.peak_loss_fn = PeakAwareLoss(peak_threshold=peak_threshold, base_loss=base_loss)
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted heatmap [B, H, W]
            target: Ground truth heatmap [B, H, W]
        """
        base_loss = self.base_loss_fn(pred, target)
        focal_loss = self.focal_loss_fn(pred, target)
        peak_loss = self.peak_loss_fn(pred, target)
        
        total_loss = (self.base_weight * base_loss + 
                     self.focal_weight * focal_loss + 
                     self.peak_weight * peak_loss)
        
        return total_loss


class GradientSharpnessLoss(nn.Module):
    """
    Loss that encourages sharp predictions by penalizing low gradients in high-intensity regions.
    Promotes steep gradients around peaks for more focused centroids.
    """
    
    def __init__(self, intensity_threshold=0.3, gradient_weight=1.0, reduction='mean'):
        """
        Args:
            intensity_threshold: Minimum intensity to consider for sharpness penalty
            gradient_weight: Weight for gradient magnitude loss
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.intensity_threshold = intensity_threshold
        self.gradient_weight = gradient_weight
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted heatmap [B, H, W]
            target: Ground truth heatmap [B, H, W]
        """
        # Compute gradients using Sobel operators
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], 
                              dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], 
                              dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        
        # Add channel dimension for conv2d
        pred_expanded = pred.unsqueeze(1)  # [B, 1, H, W]
        target_expanded = target.unsqueeze(1)
        
        # Compute gradients
        pred_grad_x = F.conv2d(pred_expanded, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_expanded, sobel_y, padding=1)
        target_grad_x = F.conv2d(target_expanded, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_expanded, sobel_y, padding=1)
        
        # Gradient magnitudes
        pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        # Focus on high-intensity regions
        high_intensity_mask = target > self.intensity_threshold
        
        # Gradient loss - encourage similar gradient magnitudes in high-intensity regions
        gradient_loss = F.mse_loss(
            pred_grad_mag * high_intensity_mask.unsqueeze(1),
            target_grad_mag * high_intensity_mask.unsqueeze(1),
            reduction='none'
        )
        
        if self.reduction == 'mean':
            return gradient_loss.mean()
        elif self.reduction == 'sum':
            return gradient_loss.sum()
        else:
            return gradient_loss.squeeze(1)


class SuperSharpCentroidLoss(nn.Module):
    """
    Ultra-sharp centroid loss combining multiple sharpening techniques:
    - Hybrid loss (regression + focal + peak-aware)
    - Gradient sharpness penalty
    - High-frequency emphasis
    """
    
    def __init__(self, 
                 hybrid_weight=1.0,
                 gradient_weight=0.8,
                 laplacian_weight=0.3,
                 **hybrid_kwargs):
        super().__init__()
        
        self.hybrid_weight = hybrid_weight
        self.gradient_weight = gradient_weight 
        self.laplacian_weight = laplacian_weight
        
        # Base hybrid loss
        self.hybrid_loss = HybridCentroidLoss(**hybrid_kwargs)
        
        # Gradient sharpness loss
        self.gradient_loss = GradientSharpnessLoss()
        
        # Laplacian kernel for high-frequency emphasis
        laplacian_kernel = torch.tensor([[[
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ]]], dtype=torch.float32)
        self.register_buffer('laplacian_kernel', laplacian_kernel)
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted heatmap [B, H, W]
            target: Ground truth heatmap [B, H, W]
        """
        # Base hybrid loss
        hybrid_loss = self.hybrid_loss(pred, target)
        
        # Gradient sharpness loss
        gradient_loss = self.gradient_loss(pred, target)
        
        # Laplacian (high-frequency) loss
        pred_expanded = pred.unsqueeze(1)  # [B, 1, H, W]
        target_expanded = target.unsqueeze(1)
        
        pred_laplacian = F.conv2d(pred_expanded, self.laplacian_kernel, padding=1)
        target_laplacian = F.conv2d(target_expanded, self.laplacian_kernel, padding=1)
        
        laplacian_loss = F.mse_loss(pred_laplacian, target_laplacian)
        
        # Combine losses
        total_loss = (self.hybrid_weight * hybrid_loss + 
                     self.gradient_weight * gradient_loss + 
                     self.laplacian_weight * laplacian_loss)
        
        return total_loss

class ImprovedCenterNetLoss(nn.Module):
    """
    Improved CenterNet loss that properly penalizes zero predictions.
    Based on web research findings about focal loss for sparse heatmaps.
    """
    def __init__(self, alpha=2.0, beta=4.0, positive_weight=10.0):
        super().__init__()
        self.alpha = alpha  # Focus on hard examples
        self.beta = beta    # Down-weight easy negatives  
        self.positive_weight = positive_weight  # Boost positive samples
        
    def forward(self, pred, target):
        """
        pred: [N, H, W] predicted heatmap
        target: [N, H, W] target heatmap
        """
        # Ensure predictions are in valid range
        pred = torch.clamp(pred, min=1e-7, max=1.0-1e-7)
        
        # Separate positive and negative samples
        positive_mask = target > 0.5
        negative_mask = target <= 0.5
        
        # Focal loss components
        pos_loss = -self.positive_weight * (1 - pred)**self.alpha * torch.log(pred)
        neg_loss = -(1 - target)**self.beta * pred**self.alpha * torch.log(1 - pred)
        
        # Apply masks and reduce
        pos_loss = pos_loss[positive_mask].sum()
        neg_loss = neg_loss[negative_mask].sum()
        
        # Normalize by number of positive samples (avoid division by zero)
        num_pos = positive_mask.sum().clamp(min=1)
        
        total_loss = (pos_loss + neg_loss) / num_pos
        
        # Add MSE component for peak sharpness
        mse_loss = nn.functional.mse_loss(pred, target, reduction='mean')
        
        return total_loss + 0.1 * mse_loss


class PositiveWeightedMSE(nn.Module):
    """Simple MSE loss with extra weight for positive regions"""
    def __init__(self, positive_weight=5.0):
        super().__init__()
        self.positive_weight = positive_weight
        
    def forward(self, pred, target):
        # Basic MSE
        mse = (pred - target) ** 2
        
        # Weight positive regions more heavily
        weights = torch.ones_like(target)
        weights[target > 0.1] = self.positive_weight
        
        weighted_mse = mse * weights
        return weighted_mse.mean()


class CenterNetFocalLoss(nn.Module):
    """
    CenterNet-style focal loss specifically designed for sharp heatmap predictions.
    Based on the original CenterNet paper implementation.
    """
    def __init__(self, alpha=3.5, beta=4.0):
        super().__init__()
        self.alpha = alpha  # Focus on hard examples
        self.beta = beta    # Down-weight easy negatives
        
    def forward(self, pred, target):
        """
        pred: [B, H, W] predicted heatmap
        target: [B, H, W] target heatmap
        """
        pred = torch.sigmoid(pred)
        pred = torch.clamp(pred, min=1e-4, max=1.0-1e-4)
        
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        
        # Compute positive loss
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        
        # Compute negative loss with penalty reduction
        neg_weights = torch.pow(1 - target, self.beta)
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds
        
        # Normalize by number of positive samples
        num_pos = pos_inds.sum().clamp(min=1)
        
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        
        loss = -(pos_loss + neg_loss) / num_pos
        return loss


class SpatialSoftmaxLoss(nn.Module):
    """
    Loss designed for spatial softmax outputs where peaks compete spatially.
    Encourages sharp, localized peaks.
    """
    def __init__(self, temperature=1.0, focus_weight=2.0):
        super().__init__()
        self.temperature = temperature
        self.focus_weight = focus_weight
        
    def forward(self, pred, target):
        """
        pred: [B, H, W] predicted spatial softmax heatmap
        target: [B, H, W] target heatmap  
        """
        # Convert target to soft distribution (normalize per sample)
        batch_size = target.shape[0]
        target_soft = torch.zeros_like(target)
        
        for b in range(batch_size):
            target_b = target[b]
            if target_b.sum() > 1e-6:
                target_soft[b] = target_b / (target_b.sum() + 1e-8)
            else:
                # If no peaks, create uniform distribution
                target_soft[b] = torch.ones_like(target_b) / target_b.numel()
        
        # KL divergence between predicted and target distributions
        pred_log = torch.log(pred + 1e-8)
        kl_loss = F.kl_div(pred_log, target_soft, reduction='batchmean')
        
        # Add focus loss to encourage sharp peaks
        # Penalize entropy (reward low entropy = sharp peaks)
        entropy = -torch.sum(pred * pred_log, dim=(1, 2)).mean()
        focus_loss = self.focus_weight * entropy
        
        return kl_loss + focus_loss


class SharpPeakRegressionLoss(nn.Module):
    """
    Ultimate loss for sharp peak regression combining multiple techniques:
    - CenterNet focal loss for handling sparse targets
    - Spatial softmax loss for competitive peak selection  
    - Peak sharpness penalty for encouraging local variation
    """
    def __init__(self, focal_weight=1.0, softmax_weight=0.5, sharpness_weight=0.2):
        super().__init__()
        
        self.focal_loss = CenterNetFocalLoss(alpha=2.0, beta=4.0)
        self.softmax_loss = SpatialSoftmaxLoss(temperature=1.0, focus_weight=2.0) 
        
        self.focal_weight = focal_weight
        self.softmax_weight = softmax_weight  
        self.sharpness_weight = sharpness_weight
        
    def compute_sharpness_penalty(self, pred):
        """Penalize overly smooth predictions by measuring local variance"""
        # Compute local standard deviation using sliding window
        kernel = torch.ones(1, 1, 3, 3, device=pred.device) / 9.0
        pred_unsqueezed = pred.unsqueeze(1)
        
        # Local mean
        local_mean = F.conv2d(pred_unsqueezed, kernel, padding=1)
        
        # Local variance  
        local_var = F.conv2d(pred_unsqueezed**2, kernel, padding=1) - local_mean**2
        local_std = torch.sqrt(local_var + 1e-8)
        
        # Penalize low standard deviation (smooth regions)
        # But only in regions where we expect peaks
        peak_regions = (pred > 0.1).float().unsqueeze(1)
        sharpness_penalty = -torch.mean(local_std * peak_regions)
        
        return sharpness_penalty
        
    def forward(self, pred, target):
        # Main losses
        focal = self.focal_loss(pred, target)
        softmax = self.softmax_loss(pred, target) 
        sharpness = self.compute_sharpness_penalty(pred)
        
        total_loss = (self.focal_weight * focal + 
                     self.softmax_weight * softmax +
                     self.sharpness_weight * sharpness)
        
        return total_loss


class ClassDiscriminativeLoss(_Loss):
    """
    Class discriminative loss to encourage monosemantic SAE features.
    
    For each SAE feature, compute the entropy of its class distribution.
    Low entropy = feature is class-specific (good)
    High entropy = feature is polysemantic (bad)
    
    The loss is the mean normalized entropy across all active features,
    encouraging each feature to specialize for a single class.
    
    Args:
        num_classes: Number of segmentation classes
        eps: Small value for numerical stability (default: 1e-8)
    
    Example:
        >>> loss_fn = ClassDiscriminativeLoss(num_classes=9)
        >>> h_sparse = torch.randn(2, 64, 16384)  # (B, N, d_hidden)
        >>> class_labels = torch.randint(0, 9, (2, 64))  # (B, N)
        >>> loss = loss_fn(h_sparse, class_labels)
    
    Note:
        This loss should be used during SAE training alongside reconstruction
        and other losses. It does NOT replace segmentation losses.
    """
    
    def __init__(self, num_classes: int, eps: float = 1e-8):
        super(ClassDiscriminativeLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
    
    def forward(
        self, 
        h_sparse: torch.Tensor, 
        class_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute class discriminative loss.
        
        Args:
            h_sparse: Sparse SAE features (B, N, d_hidden) where N is spatial tokens
            class_labels: Class labels at feature resolution (B, N)
        
        Returns:
            Scalar loss (mean normalized entropy across features)
        """
        B, N, d_hidden = h_sparse.shape
        device = h_sparse.device
        eps = self.eps
        num_classes = self.num_classes
        
        # Initialize accumulators for class-feature activation sums
        # class_feature_sums[c, f] = sum of feature f activations for class c
        class_feature_sums = torch.zeros(num_classes, d_hidden, device=device)
        
        # Flatten batch and spatial dimensions
        h_flat = h_sparse.reshape(B * N, d_hidden)  # (B*N, d_hidden)
        h_normalized = h_flat / (h_flat.sum(dim=-1, keepdim=True) + eps)
        labels_flat = class_labels.reshape(B * N)    # (B*N,)
        
        # For each class, sum the feature activations
        for c in range(num_classes):
            mask = (labels_flat == c)  # (B*N,)
            if mask.any():
                # Sum activations for this class
                class_feature_sums[c] = h_normalized[mask].sum(dim=0)  # (d_hidden,)
        
        # Normalize to get probability distribution per feature
        # probs[c, f] = probability that feature f is used for class c
        total_per_feature = class_feature_sums.sum(dim=0, keepdim=True)  # (1, d_hidden)
        probs = class_feature_sums / (total_per_feature + eps)  # (num_classes, d_hidden)
        
        # Compute entropy per feature: H = -sum(p * log(p))
        # clamp probs to avoid log(0)
        log_probs = torch.log(probs + eps)
        entropy_per_feature = -(probs * log_probs).sum(dim=0)  # (d_hidden,)
        
        # Only consider features that were actually used (avoid penalizing dead features)
        active_mask = total_per_feature.squeeze() > eps  # (d_hidden,)
        if active_mask.any():
            loss = entropy_per_feature[active_mask].mean()
        else:
            loss = (h_sparse * 0.0).sum()
        
        # Normalize by log(num_classes) so loss is in [0, 1] range
        # Maximum entropy is log(num_classes) when uniform distribution
        max_entropy = torch.log(torch.tensor(float(num_classes), device=device))
        loss = loss / max_entropy
        
        return loss


class LabelConsistencyLoss(_Loss):
    """
    Label consistency loss to encourage same-class pixels to use similar sparse codes.
    
    Uses cosine similarity to measure angular distance between same-class feature vectors.
    This is ideal for TopK SAE since it's invariant to activation magnitude and focuses
    on WHICH features are active, not their exact values.
    
    Only computes similarity over active features (non-zero activations) to avoid
    dilution from the ~99.8% zero values in TopK sparse codes.
    
    Loss: L = 1 - mean(cosine_similarity(h_i, mu_c)) over active features
    
    Args:
        num_classes: Number of segmentation classes
        eps: Small value for numerical stability (default: 1e-8)
        max_samples_per_class: Maximum samples per class for efficiency (default: None = use all)
    
    Example:
        >>> loss_fn = LabelConsistencyLoss(num_classes=9)
        >>> h_sparse = torch.randn(2, 64, 16384)  # (B, N, d_hidden)
        >>> class_labels = torch.randint(0, 9, (2, 64))  # (B, N)
        >>> loss = loss_fn(h_sparse, class_labels)
    
    Note:
        This loss complements ClassDiscriminativeLoss:
        - ClassDiscriminativeLoss: each feature should be class-specific
        - LabelConsistencyLoss: same-class pixels should use similar features
    """
    
    def __init__(
        self, 
        num_classes: int,
        eps: float = 1e-8,
        max_samples_per_class: Optional[int] = None,
    ):
        super(LabelConsistencyLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.max_samples_per_class = max_samples_per_class
    
    def forward(
        self, 
        h_sparse: torch.Tensor, 
        class_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label consistency loss.
        
        Args:
            h_sparse: Sparse SAE features (B, N, d_hidden) where N is spatial tokens
            class_labels: Class labels at feature resolution (B, N)
        
        Returns:
            Scalar loss (mean within-class angular dissimilarity)
        """
        B, N, d_hidden = h_sparse.shape
        device = h_sparse.device
        
        # Flatten batch and spatial dimensions
        h_flat = h_sparse.reshape(B * N, d_hidden)  # (B*N, d_hidden)
        h_normalized = h_flat / (h_flat.sum(dim=-1, keepdim=True) + self.eps)
        labels_flat = class_labels.reshape(B * N)    # (B*N,)
        
        # Build global active mask: features that have non-zero activation somewhere
        class_feature_sums = torch.zeros(self.num_classes, d_hidden, device=device)
        for c in range(self.num_classes):
            mask = (labels_flat == c)
            if mask.any():
                class_feature_sums[c] = h_normalized[mask].abs().sum(dim=0)
        total_per_feature = class_feature_sums.sum(dim=0)  # (d_hidden,)
        active_mask = total_per_feature > self.eps  # (d_hidden,)
        
        # If no active features, return zero
        if not active_mask.any():
            return (h_sparse * 0.0).sum()
        
        # Collect per-class losses
        class_losses = []
        
        for c in range(self.num_classes):
            mask = (labels_flat == c)
            n_samples = mask.sum()
            
            if n_samples < 2:
                continue
            
            # Get samples for this class
            h_class = h_normalized[mask]  # (N_c, d_hidden)
            
            # Optionally subsample for efficiency
            if self.max_samples_per_class is not None and n_samples > self.max_samples_per_class:
                stride = n_samples // self.max_samples_per_class
                indices = torch.arange(0, n_samples, stride, device=device)[:self.max_samples_per_class]
                h_class = h_class[indices]
            
            # Only use active features for cosine similarity
            h_class_active = h_class[:, active_mask]  # (N_c, num_active)
            
            # Compute class mean (prototype) over active features
            mu_c = h_class_active.mean(dim=0, keepdim=True)  # (1, num_active)
            
            # Cosine similarity: measures angular distance (direction, not magnitude)
            # Perfect for TopK SAE where we care about WHICH features, not their values
            similarity = F.cosine_similarity(h_class_active, mu_c, dim=-1, eps=self.eps)  # (N_c,)
            
            # Loss = 1 - mean similarity (minimize angular distance to prototype)
            loss_c = 1.0 - similarity.mean()
            
            class_losses.append(loss_c)
        
        if len(class_losses) == 0:
            return (h_sparse * 0.0).sum()
        
        loss = torch.stack(class_losses).mean()
        
        # Clamp
        loss = torch.clamp(loss, min=0.0, max=1e6)
        
        return loss


class LCKSVDLoss(_Loss):
    """
    True LC-KSVD (Label Consistent K-SVD) loss with learnable discriminative transform.
    
    Original formulation from Jiang et al. (CVPR 2011):
        L = ||Q - A @ X||^2
    
    Where:
        - Q: One-hot class labels (num_classes, num_samples)
        - A: Learned linear transform (num_classes, d_hidden) - projects sparse codes to labels
        - X: Sparse codes (d_hidden, num_samples)
    
    This loss learns a transform A such that sparse codes can be linearly projected
    to class labels. This forces same-class samples to lie in a discriminative
    linear subspace.
    
    Args:
        num_classes: Number of segmentation classes
        d_hidden: SAE hidden dimension (dictionary size)
        eps: Small value for numerical stability
    
    Example:
        >>> loss_fn = LCKSVDLoss(num_classes=9, d_hidden=16384)
        >>> h_sparse = torch.randn(2, 64, 16384)  # (B, N, d_hidden)
        >>> class_labels = torch.randint(0, 9, (2, 64))  # (B, N)
        >>> loss = loss_fn(h_sparse, class_labels)
    
    Note:
        The transform A is learned jointly with SAE parameters. After training,
        A can be used to interpret which features are most discriminative for
        each class (by examining rows of A).
    
    Reference:
        Jiang, Lin, Davis. "Learning a Discriminative Dictionary for Sparse Coding
        via Label Consistent K-SVD." CVPR 2011.
    """
    
    def __init__(
        self, 
        num_classes: int,
        d_hidden: int,
        eps: float = 1e-8,
    ):
        super(LCKSVDLoss, self).__init__()
        self.num_classes = num_classes
        self.d_hidden = d_hidden
        self.eps = eps
        
        # Learnable linear transform: projects sparse codes to class space
        # A: (d_hidden) -> (num_classes)
        self.A = nn.Linear(d_hidden, num_classes, bias=False)
        
        # Initialize with small weights for stability
        nn.init.xavier_uniform_(self.A.weight)
    
    def forward(
        self, 
        h_sparse: torch.Tensor, 
        class_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute LC-KSVD loss.
        
        Args:
            h_sparse: Sparse SAE features (B, N, d_hidden) where N is spatial tokens
            class_labels: Class labels at feature resolution (B, N)
        
        Returns:
            Scalar loss (MSE between projected codes and one-hot labels)
        """
        B, N, d_hidden = h_sparse.shape
        device = h_sparse.device
        
        # Flatten batch and spatial dimensions
        h_flat = h_sparse.reshape(B * N, d_hidden)  # (B*N, d_hidden)
        labels_flat = class_labels.reshape(B * N)    # (B*N,)
        
        # Project sparse codes to label space
        pred_labels = self.A(h_flat)  # (B*N, num_classes)
        
        # Create one-hot target labels
        # Clamp to valid range to handle ignore_index or invalid labels
        labels_clamped = labels_flat.clamp(0, self.num_classes - 1)
        Q = F.one_hot(labels_clamped, self.num_classes).float()  # (B*N, num_classes)
        
        # MSE loss between projection and one-hot labels
        loss = F.mse_loss(pred_labels, Q)
        
        return loss
    
    def get_class_feature_weights(self) -> torch.Tensor:
        """
        Get the learned feature weights for each class.
        
        Returns:
            Tensor of shape (num_classes, d_hidden) where each row shows
            which features are most important for that class.
        """
        return self.A.weight.data  # (num_classes, d_hidden)


# =====================================================================
#  Object Detection Losses
# =====================================================================

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) -> (x1, y1, x2, y2). All normalised [0,1]."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (x1, y1, x2, y2) -> (cx, cy, w, h)."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalised IoU between two sets of boxes in (x1, y1, x2, y2) format.

    Args:
        boxes1: (N, 4)
        boxes2: (M, 4)

    Returns:
        GIoU matrix (N, M) in range [-1, 1].
    """
    # Intersection
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2[None, :] - inter_area
    iou = inter_area / union.clamp(min=1e-6)

    # Enclosing box
    enc_x1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    enc_y1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    enc_x2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    enc_y2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])

    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)

    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
    return giou


class GIoULoss(nn.Module):
    """
    Generalised IoU loss for bounding box regression.

    Args:
        reduction: 'mean', 'sum', or 'none'

    Forward:
        pred_boxes : (N, 4)  in (x1, y1, x2, y2) or (cx, cy, w, h) format
        target_boxes : (N, 4)  same format
        box_fmt : 'xyxy' or 'cxcywh'
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        box_fmt: str = "cxcywh",
    ) -> torch.Tensor:
        if box_fmt == "cxcywh":
            pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
            target_xyxy = box_cxcywh_to_xyxy(target_boxes)
        else:
            pred_xyxy = pred_boxes
            target_xyxy = target_boxes

        # Pairwise GIoU on matched pairs (diagonal)
        giou_matrix = generalized_box_iou(pred_xyxy, target_xyxy)
        giou_diag = torch.diag(giou_matrix)
        loss = 1 - giou_diag

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class HungarianMatcher(nn.Module):
    """DETR-style bipartite matcher. Uses scipy by default; set use_scipy=False
    to use the pure-torch fallback (_hungarian_assignment)."""

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        use_scipy: bool = True,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_scipy = use_scipy

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list):
        """
        Parameters
        ----------
        outputs : dict
            pred_logits : (B, Q, num_classes+1)
            pred_boxes  : (B, Q, 4)  normalised (cx, cy, w, h)
        targets : list[dict]  length B
            Each dict has:
                labels : LongTensor(N,)
                boxes  : FloatTensor(N, 4)  normalised (cx, cy, w, h)

        Returns
        -------
        list of (pred_indices, tgt_indices) tuples, one per image.
        """
        B, Q, _ = outputs["pred_logits"].shape

        # Flatten across batch for efficiency
        out_prob = outputs["pred_logits"].softmax(-1)  # (B, Q, C+1)
        out_bbox = outputs["pred_boxes"]                # (B, Q, 4)

        indices = []
        for b in range(B):
            tgt_labels = targets[b]["labels"]  # (N,)
            tgt_boxes = targets[b]["boxes"]    # (N, 4)

            if tgt_labels.numel() == 0:
                indices.append((torch.tensor([], dtype=torch.long),
                                torch.tensor([], dtype=torch.long)))
                continue

            # Classification cost: -prob of correct class
            cost_cls = -out_prob[b, :, tgt_labels]  # (Q, N)

            # L1 box cost
            cost_box = torch.cdist(out_bbox[b], tgt_boxes, p=1)  # (Q, N)

            # GIoU cost
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[b]),
                box_cxcywh_to_xyxy(tgt_boxes),
            )  # (Q, N)

            # Combined cost
            C = (
                self.cost_class * cost_cls
                + self.cost_bbox * cost_box
                + self.cost_giou * cost_giou
            )

            # Replace NaN/Inf with large values so assignment still works
            C = C.nan_to_num(nan=1e4, posinf=1e4, neginf=-1e4)

            if self.use_scipy:
                pred_idx, tgt_idx = linear_sum_assignment(C.cpu().numpy())
                pred_idx = torch.as_tensor(pred_idx, dtype=torch.long)
                tgt_idx = torch.as_tensor(tgt_idx, dtype=torch.long)
            else:
                pred_idx, tgt_idx = _hungarian_assignment(C)
            indices.append((pred_idx, tgt_idx))

        return indices


class DETRLoss(nn.Module):
    """
    Full DETR set-prediction loss.

    Combines:
      - Cross-entropy classification (with "no-object" class)
      - L1 bounding-box regression
      - Generalised IoU loss

    Compatible with the standard loss_fun.py interface: instantiated via
    config and called as ``loss(preds, targets)``.

    Config example:
        loss: ["DETRLoss"]
        loss_weights: [1.0]
        DETRLoss:
          num_classes: 1
          cost_class: 1.0
          cost_bbox: 5.0
          cost_giou: 2.0
          cls_weight: 1.0
          bbox_weight: 5.0
          giou_weight: 2.0
          no_object_weight: 0.1
    """

    def __init__(
        self,
        num_classes: int = 1,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        cls_weight: float = 1.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
        no_object_weight: float = 0.1,
        use_scipy: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.no_object_weight = no_object_weight

        self.matcher = HungarianMatcher(cost_class, cost_bbox, cost_giou, use_scipy=use_scipy)
        self.giou_loss = GIoULoss(reduction="mean")

        # CE weights: down-weight the "no-object" class
        ce_weight = torch.ones(num_classes + 1)
        ce_weight[-1] = no_object_weight
        self.register_buffer("ce_weight", ce_weight)

    def forward(self, preds: dict, targets: list) -> torch.Tensor:
        """
        Parameters
        ----------
        preds : dict from Dinov3DetectionDETR
            pred_logits : (B, Q, C+1)
            pred_boxes  : (B, Q, 4)
        targets : list[dict]
            Each dict has ``labels`` (N,) and ``boxes`` (N, 4) in (cx, cy, w, h).

        Returns
        -------
        Scalar total loss.
        """
        indices = self.matcher(preds, targets)

        # ---------- Classification loss ----------
        B, Q, _ = preds["pred_logits"].shape
        device = preds["pred_logits"].device

        # Default target class = "no-object" (last class index)
        target_classes = torch.full(
            (B, Q), self.num_classes, dtype=torch.long, device=device
        )
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if pred_idx.numel() > 0:
                target_classes[b, pred_idx] = targets[b]["labels"][tgt_idx].to(device)

        cls_loss = F.cross_entropy(
            preds["pred_logits"].reshape(-1, self.num_classes + 1),
            target_classes.reshape(-1),
            weight=self.ce_weight.to(device),
        )

        # ---------- Box losses (only for matched pairs) ----------
        src_boxes_list = []
        tgt_boxes_list = []
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if pred_idx.numel() > 0:
                src_boxes_list.append(preds["pred_boxes"][b, pred_idx])
                tgt_boxes_list.append(targets[b]["boxes"][tgt_idx].to(device))

        if len(src_boxes_list) > 0:
            src_boxes = torch.cat(src_boxes_list, dim=0)
            tgt_boxes = torch.cat(tgt_boxes_list, dim=0)

            l1_loss = F.l1_loss(src_boxes, tgt_boxes, reduction="mean")
            giou_loss = self.giou_loss(src_boxes, tgt_boxes, box_fmt="cxcywh")
        else:
            l1_loss = torch.tensor(0.0, device=device)
            giou_loss = torch.tensor(0.0, device=device)

        total = (
            self.cls_weight * cls_loss
            + self.bbox_weight * l1_loss
            + self.giou_weight * giou_loss
        )
        return total

    def forward_with_components(self, preds: dict, targets: list) -> dict:
        """Same as forward() but returns individual loss components for logging."""
        indices = self.matcher(preds, targets)
        B, Q, _ = preds["pred_logits"].shape
        device = preds["pred_logits"].device

        target_classes = torch.full(
            (B, Q), self.num_classes, dtype=torch.long, device=device
        )
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if pred_idx.numel() > 0:
                target_classes[b, pred_idx] = targets[b]["labels"][tgt_idx].to(device)

        cls_loss = F.cross_entropy(
            preds["pred_logits"].reshape(-1, self.num_classes + 1),
            target_classes.reshape(-1),
            weight=self.ce_weight.to(device),
        )

        src_boxes_list, tgt_boxes_list = [], []
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if pred_idx.numel() > 0:
                src_boxes_list.append(preds["pred_boxes"][b, pred_idx])
                tgt_boxes_list.append(targets[b]["boxes"][tgt_idx].to(device))

        if src_boxes_list:
            src_boxes = torch.cat(src_boxes_list, dim=0)
            tgt_boxes = torch.cat(tgt_boxes_list, dim=0)
            l1_loss = F.l1_loss(src_boxes, tgt_boxes, reduction="mean")
            giou_loss = self.giou_loss(src_boxes, tgt_boxes, box_fmt="cxcywh")
        else:
            l1_loss = torch.tensor(0.0, device=device)
            giou_loss = torch.tensor(0.0, device=device)

        total = (
            self.cls_weight * cls_loss
            + self.bbox_weight * l1_loss
            + self.giou_weight * giou_loss
        )

        return {
            "loss": total,
            "cls_loss": cls_loss,
            "l1_loss": l1_loss,
            "giou_loss": giou_loss,
        }


class FCOSDetectionLoss(nn.Module):
    """
    Anchor-free FCOS-style loss for Dinov3DetectionDPT.

    Combines:
      - Focal loss for classification (per-pixel)
      - IoU / GIoU loss for bounding-box regression
      - Binary cross-entropy for centerness

    The ground-truth encoding follows FCOS: each foreground pixel is
    assigned to the object whose bbox contains it, and regresses the
    distances (l, t, r, b) to the four sides of that box.

    Config example:
        loss: ["FCOSDetectionLoss"]
        loss_weights: [1.0]
        FCOSDetectionLoss:
          num_classes: 1
          focal_alpha: 0.25
          focal_gamma: 2.0
          cls_weight: 1.0
          bbox_weight: 1.0
          centerness_weight: 1.0
    """

    def __init__(
        self,
        num_classes: int = 1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cls_weight: float = 1.0,
        bbox_weight: float = 1.0,
        centerness_weight: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.centerness_weight = centerness_weight

    # ------------------------------------------------------------------
    #  Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_targets(
        targets: list,
        H: int,
        W: int,
        num_classes: int,
        device: torch.device,
    ):
        """
        Encode list[dict] ground truth into dense per-pixel maps (vectorized).

        Parameters
        ----------
        targets : list[dict]
            Each dict has ``labels`` (N,), ``boxes`` (N, 4) in normalised
            (cx, cy, w, h) format.
        H, W : spatial dims of the prediction maps.

        Returns
        -------
        cls_targets   : (B, H, W)        long – class index (0 = background)
        reg_targets   : (B, 4, H, W)     float – (l, t, r, b) pixel distances
        ctr_targets   : (B, H, W)        float – centerness in [0, 1]
        fg_mask       : (B, H, W)        bool – foreground mask
        """
        B = len(targets)
        cls_targets = torch.zeros(B, H, W, dtype=torch.long, device=device)
        reg_targets = torch.zeros(B, 4, H, W, dtype=torch.float32, device=device)
        ctr_targets = torch.zeros(B, H, W, dtype=torch.float32, device=device)

        # Create meshgrid once (not per batch or per object)
        ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
        xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

        for b in range(B):
            boxes = targets[b]["boxes"].to(device)   # (N, 4) cxcywh
            labels = targets[b]["labels"].to(device)  # (N,)

            if boxes.numel() == 0:
                continue

            N = boxes.shape[0]
            x1 = boxes[:, 0] - boxes[:, 2] / 2  # (N,)
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            x2 = boxes[:, 0] + boxes[:, 2] / 2
            y2 = boxes[:, 1] + boxes[:, 3] / 2

            # Broadcast: (N, 1, 1) vs (H, W) -> (N, H, W)
            l = xx.unsqueeze(0) - x1[:, None, None]
            t = yy.unsqueeze(0) - y1[:, None, None]
            r = x2[:, None, None] - xx.unsqueeze(0)
            bot = y2[:, None, None] - yy.unsqueeze(0)

            # Inside mask for all objects at once: (N, H, W)
            inside = (l > 0) & (t > 0) & (r > 0) & (bot > 0)

            any_fg = inside.any(dim=0)  # (H, W)
            if not any_fg.any():
                continue

            # Centerness for all objects: (N, H, W)
            lr = torch.min(l, r) / torch.max(l, r).clamp(min=1e-6)
            tb = torch.min(t, bot) / torch.max(t, bot).clamp(min=1e-6)
            ctr = torch.sqrt((lr * tb).clamp(min=0)).clamp(0, 1)

            # Last-wins: highest object index covering each pixel
            obj_priority = torch.arange(N, device=device)[:, None, None].expand_as(inside)
            obj_priority = obj_priority.masked_fill(~inside, -1)
            winner = obj_priority.max(dim=0).values  # (H, W)

            fg_h, fg_w = torch.where(any_fg)
            win_idx = winner[fg_h, fg_w]

            cls_targets[b, fg_h, fg_w] = labels[win_idx] + 1
            reg_targets[b, 0, fg_h, fg_w] = l[win_idx, fg_h, fg_w]
            reg_targets[b, 1, fg_h, fg_w] = t[win_idx, fg_h, fg_w]
            reg_targets[b, 2, fg_h, fg_w] = r[win_idx, fg_h, fg_w]
            reg_targets[b, 3, fg_h, fg_w] = bot[win_idx, fg_h, fg_w]
            ctr_targets[b, fg_h, fg_w] = ctr[win_idx, fg_h, fg_w]

        fg_mask = cls_targets > 0
        return cls_targets, reg_targets, ctr_targets, fg_mask

    # ------------------------------------------------------------------
    #  Focal loss
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Sigmoid focal loss (binary per-class)."""
        p = torch.sigmoid(inputs)
        ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** gamma
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * focal_weight * ce
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        return loss

    # ------------------------------------------------------------------
    #  IoU loss for LTRB regression
    # ------------------------------------------------------------------

    @staticmethod
    def _iou_loss(pred_ltrb: torch.Tensor, target_ltrb: torch.Tensor) -> torch.Tensor:
        """IoU loss between predicted and target (l,t,r,b) vectors."""
        pred_area = (pred_ltrb[:, 0] + pred_ltrb[:, 2]) * (pred_ltrb[:, 1] + pred_ltrb[:, 3])
        tgt_area = (target_ltrb[:, 0] + target_ltrb[:, 2]) * (target_ltrb[:, 1] + target_ltrb[:, 3])

        w_inter = torch.min(pred_ltrb[:, 0], target_ltrb[:, 0]) + torch.min(pred_ltrb[:, 2], target_ltrb[:, 2])
        h_inter = torch.min(pred_ltrb[:, 1], target_ltrb[:, 1]) + torch.min(pred_ltrb[:, 3], target_ltrb[:, 3])

        inter = w_inter * h_inter
        union = pred_area + tgt_area - inter
        iou = inter / union.clamp(min=1e-6)
        return (1 - iou).mean()

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, preds: dict, targets: list) -> torch.Tensor:
        """
        Parameters
        ----------
        preds : dict from Dinov3DetectionDPT
            cls_logits  : (B, num_classes, H, W)
            bbox_pred   : (B, 4, H, W)  positive (l, t, r, b)
            centerness  : (B, 1, H, W)
        targets : list[dict]
            Each has ``labels`` (N,) and ``boxes`` (N, 4) in (cx, cy, w, h).

        Returns
        -------
        Scalar total loss.
        """
        cls_logits = preds["cls_logits"]   # (B, C, H, W)
        bbox_pred = preds["bbox_pred"]     # (B, 4, H, W)
        centerness = preds["centerness"].squeeze(1)  # (B, H, W)

        B, C, H, W = cls_logits.shape
        device = cls_logits.device

        # Encode ground truth
        cls_targets, reg_targets, ctr_targets, fg_mask = self._encode_targets(
            targets, H, W, self.num_classes, device
        )

        # --- Classification (focal) loss ---
        # Convert cls_targets to one-hot for focal loss (ignore bg class 0)
        # cls_targets are 1-indexed (0=bg), class logits are 0-indexed
        cls_one_hot = torch.zeros(B, C, H, W, device=device)
        for c in range(C):
            cls_one_hot[:, c] = (cls_targets == c + 1).float()

        cls_loss = self._sigmoid_focal_loss(
            cls_logits, cls_one_hot, self.focal_alpha, self.focal_gamma, reduction="mean"
        )

        # --- Box regression loss (only on foreground pixels) ---
        num_fg = fg_mask.sum().clamp(min=1).float()

        if fg_mask.sum() > 0:
            pred_ltrb_fg = bbox_pred.permute(0, 2, 3, 1)[fg_mask]  # (num_fg, 4)
            tgt_ltrb_fg = reg_targets.permute(0, 2, 3, 1)[fg_mask]  # (num_fg, 4)
            bbox_loss = self._iou_loss(pred_ltrb_fg, tgt_ltrb_fg)
        else:
            bbox_loss = torch.tensor(0.0, device=device)

        # --- Centerness loss (only on foreground pixels) ---
        if fg_mask.sum() > 0:
            ctr_loss = F.binary_cross_entropy_with_logits(
                centerness[fg_mask], ctr_targets[fg_mask], reduction="mean"
            )
        else:
            ctr_loss = torch.tensor(0.0, device=device)

        total = (
            self.cls_weight * cls_loss
            + self.bbox_weight * bbox_loss
            + self.centerness_weight * ctr_loss
        )
        return total

    def forward_with_components(self, preds: dict, targets: list) -> dict:
        """Same as forward() but returns individual loss components for logging."""
        cls_logits = preds["cls_logits"]
        bbox_pred = preds["bbox_pred"]
        centerness = preds["centerness"].squeeze(1)

        B, C, H, W = cls_logits.shape
        device = cls_logits.device

        cls_targets, reg_targets, ctr_targets, fg_mask = self._encode_targets(
            targets, H, W, self.num_classes, device
        )

        cls_one_hot = torch.zeros(B, C, H, W, device=device)
        for c in range(C):
            cls_one_hot[:, c] = (cls_targets == c + 1).float()

        cls_loss = self._sigmoid_focal_loss(
            cls_logits, cls_one_hot, self.focal_alpha, self.focal_gamma, reduction="mean"
        )

        if fg_mask.sum() > 0:
            pred_ltrb_fg = bbox_pred.permute(0, 2, 3, 1)[fg_mask]
            tgt_ltrb_fg = reg_targets.permute(0, 2, 3, 1)[fg_mask]
            bbox_loss = self._iou_loss(pred_ltrb_fg, tgt_ltrb_fg)
        else:
            bbox_loss = torch.tensor(0.0, device=device)

        if fg_mask.sum() > 0:
            ctr_loss = F.binary_cross_entropy_with_logits(
                centerness[fg_mask], ctr_targets[fg_mask], reduction="mean"
            )
        else:
            ctr_loss = torch.tensor(0.0, device=device)

        total = (
            self.cls_weight * cls_loss
            + self.bbox_weight * bbox_loss
            + self.centerness_weight * ctr_loss
        )

        return {
            "loss": total,
            "cls_loss": cls_loss,
            "bbox_loss": bbox_loss,
            "centerness_loss": ctr_loss,
        }


class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss (QFL) from "Generalized Focal Loss".
    
    Jointly optimizes classification score and IoU quality. The classification
    target is a continuous quality score (0-1) instead of binary 0/1.
    
    Args:
        beta: Modulating factor for quality weighting (default: 2.0)
        reduction: 'none', 'mean', or 'sum'
    """
    def __init__(self, beta: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_quality: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_logits: (N, C) classification logits
            target_quality: (N, C) quality scores in [0, 1] (IoU for positives, 0 for negatives)
            weight: (N,) optional per-sample weights
        """
        pred_sigmoid = pred_logits.sigmoid()
        scale_factor = pred_sigmoid
        zerolabel = scale_factor.new_zeros(pred_logits.shape)
        
        loss = F.binary_cross_entropy_with_logits(
            pred_logits, zerolabel, reduction="none"
        ) * scale_factor.pow(self.beta)

        # Apply quality weighting
        loss = loss * target_quality + (1 - target_quality) * loss

        if weight is not None:
            loss = loss * weight.unsqueeze(1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class VarifocalLoss(nn.Module):
    """
    Varifocal Loss (VFL) from "VarifocalNet".
    
    Asymmetric weighting for positive and negative samples. Uses target IoU
    quality for positives and 0 for negatives.
    
    Args:
        alpha: Weighting factor for positive samples (default: 0.75)
        gamma: Focusing parameter (default: 2.0)
        iou_weighted: If True, weight positives by IoU quality (default: True)
        reduction: 'none', 'mean', or 'sum'
    """
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        iou_weighted: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.reduction = reduction

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_quality: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_logits: (N, C) classification logits
            target_quality: (N, C) quality scores (IoU for positives, 0 for negatives)
            weight: (N,) optional per-sample weights
        """
        pred_sigmoid = pred_logits.sigmoid()
        target = target_quality.float()
        
        # Focal weight: for positives use target, for negatives use sigmoid
        focal_weight = target * (target > 0.0).float() + \
                       self.alpha * (pred_sigmoid - target).abs().pow(self.gamma) * \
                       (target <= 0.0).float()
        
        # BCE loss
        loss = F.binary_cross_entropy_with_logits(
            pred_logits, target, reduction="none"
        )
        loss = loss * focal_weight

        if self.iou_weighted and (target > 0).any():
            # Weight positive samples by IoU quality
            pos_mask = target > 0
            loss[pos_mask] = loss[pos_mask] * target[pos_mask]

        if weight is not None:
            loss = loss * weight.unsqueeze(1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DIoULoss(nn.Module):
    """
    Distance-IoU Loss from "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression".
    
    Adds normalized distance between box centers to IoU loss for faster convergence.
    
    Args:
        eps: Small value to avoid division by zero (default: 1e-7)
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_boxes: (N, 4) boxes in xyxy or ltrb format
            target_boxes: (N, 4) target boxes in same format
        
        Returns:
            Scalar loss (1 - DIoU)
        """
        # IoU
        iou = self._compute_iou(pred_boxes, target_boxes)
        
        # Center distance
        pred_ctr = self._box_center(pred_boxes)
        tgt_ctr = self._box_center(target_boxes)
        center_dist_sq = ((pred_ctr - tgt_ctr) ** 2).sum(dim=-1)
        
        # Diagonal of smallest enclosing box
        enclose_x1 = torch.min(pred_boxes[..., 0], target_boxes[..., 0])
        enclose_y1 = torch.min(pred_boxes[..., 1], target_boxes[..., 1])
        enclose_x2 = torch.max(pred_boxes[..., 2], target_boxes[..., 2])
        enclose_y2 = torch.max(pred_boxes[..., 3], target_boxes[..., 3])
        
        enclose_w = (enclose_x2 - enclose_x1).clamp(min=0)
        enclose_h = (enclose_y2 - enclose_y1).clamp(min=0)
        diagonal_sq = enclose_w ** 2 + enclose_h ** 2 + self.eps
        
        # DIoU
        diou = iou - (center_dist_sq / diagonal_sq)
        loss = 1 - diou
        
        return loss.mean()

    def _compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between two sets of boxes."""
        x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
        y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
        x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
        y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
        
        inter_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        union_area = boxes1_area + boxes2_area - inter_area + self.eps
        iou = inter_area / union_area
        
        return iou

    def _box_center(self, boxes: torch.Tensor) -> torch.Tensor:
        """Compute box centers (cx, cy)."""
        return torch.stack([
            (boxes[..., 0] + boxes[..., 2]) / 2,
            (boxes[..., 1] + boxes[..., 3]) / 2,
        ], dim=-1)


class CIoULoss(nn.Module):
    """
    Complete-IoU Loss from "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression".
    
    Extends DIoU by adding aspect ratio consistency penalty.
    
    Args:
        eps: Small value to avoid division by zero (default: 1e-7)
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_boxes: (N, 4) boxes in xyxy or ltrb format
            target_boxes: (N, 4) target boxes in same format
        
        Returns:
            Scalar loss (1 - CIoU)
        """
        # IoU
        iou = self._compute_iou(pred_boxes, target_boxes)
        
        # Center distance
        pred_ctr = self._box_center(pred_boxes)
        tgt_ctr = self._box_center(target_boxes)
        center_dist_sq = ((pred_ctr - tgt_ctr) ** 2).sum(dim=-1)
        
        # Diagonal of smallest enclosing box
        enclose_x1 = torch.min(pred_boxes[..., 0], target_boxes[..., 0])
        enclose_y1 = torch.min(pred_boxes[..., 1], target_boxes[..., 1])
        enclose_x2 = torch.max(pred_boxes[..., 2], target_boxes[..., 2])
        enclose_y2 = torch.max(pred_boxes[..., 3], target_boxes[..., 3])
        
        enclose_w = (enclose_x2 - enclose_x1).clamp(min=0)
        enclose_h = (enclose_y2 - enclose_y1).clamp(min=0)
        diagonal_sq = enclose_w ** 2 + enclose_h ** 2 + self.eps
        
        # Aspect ratio consistency
        pred_w = (pred_boxes[..., 2] - pred_boxes[..., 0]).clamp(min=self.eps)
        pred_h = (pred_boxes[..., 3] - pred_boxes[..., 1]).clamp(min=self.eps)
        tgt_w = (target_boxes[..., 2] - target_boxes[..., 0]).clamp(min=self.eps)
        tgt_h = (target_boxes[..., 3] - target_boxes[..., 1]).clamp(min=self.eps)
        
        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(tgt_w / tgt_h) - torch.atan(pred_w / pred_h), 2
        )
        
        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)
        
        # CIoU
        ciou = iou - (center_dist_sq / diagonal_sq) - alpha * v
        loss = 1 - ciou
        
        return loss.mean()

    def _compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between two sets of boxes."""
        x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
        y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
        x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
        y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
        
        inter_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        union_area = boxes1_area + boxes2_area - inter_area + self.eps
        iou = inter_area / union_area
        
        return iou

    def _box_center(self, boxes: torch.Tensor) -> torch.Tensor:
        """Compute box centers (cx, cy)."""
        return torch.stack([
            (boxes[..., 0] + boxes[..., 2]) / 2,
            (boxes[..., 1] + boxes[..., 3]) / 2,
        ], dim=-1)


class FCOSDetectionLossV2(nn.Module):
    """
    Enhanced FCOS Detection Loss with VarifocalLoss and CIoU/DIoU options.
    
    Compared to FCOSDetectionLoss, this version supports:
    - VarifocalLoss (uses IoU quality as classification target)
    - CIoU/DIoU for better bbox regression
    - GIoU as an option
    
    Args:
        num_classes: Number of object classes
        cls_loss_type: 'focal' or 'varifocal' (default: 'varifocal')
        bbox_loss_type: 'iou', 'giou', 'diou', or 'ciou' (default: 'ciou')
        focal_alpha: Alpha for focal loss (default: 0.25)
        focal_gamma: Gamma for focal loss (default: 2.0)
        vfl_alpha: Alpha for varifocal loss (default: 0.75)
        vfl_gamma: Gamma for varifocal loss (default: 2.0)
        cls_weight: Weight for classification loss (default: 1.0)
        bbox_weight: Weight for box regression loss (default: 1.0)
        centerness_weight: Weight for centerness loss (default: 1.0)
    """
    def __init__(
        self,
        num_classes: int,
        cls_loss_type: str = "varifocal",
        bbox_loss_type: str = "ciou",
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        vfl_alpha: float = 0.75,
        vfl_gamma: float = 2.0,
        cls_weight: float = 1.0,
        bbox_weight: float = 1.0,
        centerness_weight: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_loss_type = cls_loss_type
        self.bbox_loss_type = bbox_loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.centerness_weight = centerness_weight
        
        # Initialize loss functions
        if cls_loss_type == "varifocal":
            self.cls_loss_fn = VarifocalLoss(alpha=vfl_alpha, gamma=vfl_gamma)
        
        if bbox_loss_type == "diou":
            self.bbox_loss_fn = DIoULoss()
        elif bbox_loss_type == "ciou":
            self.bbox_loss_fn = CIoULoss()
        elif bbox_loss_type == "giou":
            self.bbox_loss_fn = GIoULoss()

    def _encode_targets(self, targets, H, W, num_classes, device):
        """Encode targets to per-pixel format (vectorized, no per-object loop)."""
        B = len(targets)
        cls_targets = torch.zeros(B, H, W, dtype=torch.long, device=device)
        reg_targets = torch.zeros(B, 4, H, W, dtype=torch.float32, device=device)
        ctr_targets = torch.zeros(B, H, W, dtype=torch.float32, device=device)

        # Create meshgrid once (not per batch or per object)
        yy, xx = torch.meshgrid(
            torch.linspace(0.5 / H, 1 - 0.5 / H, H, device=device),
            torch.linspace(0.5 / W, 1 - 0.5 / W, W, device=device),
            indexing="ij",
        )

        for b, tgt in enumerate(targets):
            labels = tgt["labels"]
            boxes = tgt["boxes"]
            if boxes.numel() == 0:
                continue

            N = boxes.shape[0]
            x1 = (boxes[:, 0] - boxes[:, 2] / 2).clamp(0, 1)
            y1 = (boxes[:, 1] - boxes[:, 3] / 2).clamp(0, 1)
            x2 = (boxes[:, 0] + boxes[:, 2] / 2).clamp(0, 1)
            y2 = (boxes[:, 1] + boxes[:, 3] / 2).clamp(0, 1)

            # Broadcast: (N, 1, 1) vs (H, W) -> (N, H, W)
            l = xx.unsqueeze(0) - x1[:, None, None]
            t = yy.unsqueeze(0) - y1[:, None, None]
            r = x2[:, None, None] - xx.unsqueeze(0)
            bot = y2[:, None, None] - yy.unsqueeze(0)

            # Inside mask for all objects at once: (N, H, W)
            inside = (l > 0) & (t > 0) & (r > 0) & (bot > 0)

            any_fg = inside.any(dim=0)  # (H, W)
            if not any_fg.any():
                continue

            # Centerness for all objects: (N, H, W)
            lr = torch.min(l, r) / torch.max(l, r).clamp(min=1e-6)
            tb = torch.min(t, bot) / torch.max(t, bot).clamp(min=1e-6)
            ctr = torch.sqrt((lr * tb).clamp(min=0)).clamp(0, 1)

            # Last-wins assignment: highest object index that covers each pixel
            # Multiply inside by object index; argmax gives last True
            obj_priority = torch.arange(N, device=device)[:, None, None].expand_as(inside)
            obj_priority = obj_priority.masked_fill(~inside, -1)
            winner = obj_priority.max(dim=0).values  # (H, W), -1 where no object

            # Gather winning object's values using advanced indexing
            fg_h, fg_w = torch.where(any_fg)
            win_idx = winner[fg_h, fg_w]  # object index per fg pixel

            cls_targets[b, fg_h, fg_w] = labels[win_idx] + 1
            reg_targets[b, 0, fg_h, fg_w] = l[win_idx, fg_h, fg_w]
            reg_targets[b, 1, fg_h, fg_w] = t[win_idx, fg_h, fg_w]
            reg_targets[b, 2, fg_h, fg_w] = r[win_idx, fg_h, fg_w]
            reg_targets[b, 3, fg_h, fg_w] = bot[win_idx, fg_h, fg_w]
            ctr_targets[b, fg_h, fg_w] = ctr[win_idx, fg_h, fg_w]

        fg_mask = cls_targets > 0
        return cls_targets, reg_targets, ctr_targets, fg_mask

    def _compute_iou_quality(self, pred_ltrb, target_ltrb):
        """Compute IoU for quality targets."""
        pred_area = (pred_ltrb[:, 0] + pred_ltrb[:, 2]) * (pred_ltrb[:, 1] + pred_ltrb[:, 3])
        tgt_area = (target_ltrb[:, 0] + target_ltrb[:, 2]) * (target_ltrb[:, 1] + target_ltrb[:, 3])

        w_inter = torch.min(pred_ltrb[:, 0], target_ltrb[:, 0]) + torch.min(pred_ltrb[:, 2], target_ltrb[:, 2])
        h_inter = torch.min(pred_ltrb[:, 1], target_ltrb[:, 1]) + torch.min(pred_ltrb[:, 3], target_ltrb[:, 3])

        inter = w_inter * h_inter
        union = pred_area + tgt_area - inter
        iou = inter / union.clamp(min=1e-6)
        return iou

    def _ltrb_box_loss(self, pred_ltrb: torch.Tensor, tgt_ltrb: torch.Tensor) -> torch.Tensor:
        """Compute IoU/GIoU/DIoU/CIoU directly in ltrb space (no meshgrid needed).
        
        Since both pred and target are measured from the same pixel,
        all geometric quantities can be derived from ltrb alone.
        """
        eps = 1e-6
        pl, pt, pr, pb = pred_ltrb[:, 0], pred_ltrb[:, 1], pred_ltrb[:, 2], pred_ltrb[:, 3]
        tl, tt, tr, tb = tgt_ltrb[:, 0], tgt_ltrb[:, 1], tgt_ltrb[:, 2], tgt_ltrb[:, 3]

        # Box dimensions
        pred_w = (pl + pr).clamp(min=eps)
        pred_h = (pt + pb).clamp(min=eps)
        tgt_w = (tl + tr).clamp(min=eps)
        tgt_h = (tt + tb).clamp(min=eps)

        pred_area = pred_w * pred_h
        tgt_area = tgt_w * tgt_h

        # Intersection
        inter_w = (torch.min(pl, tl) + torch.min(pr, tr)).clamp(min=0)
        inter_h = (torch.min(pt, tt) + torch.min(pb, tb)).clamp(min=0)
        inter = inter_w * inter_h

        union = pred_area + tgt_area - inter
        iou = inter / union.clamp(min=eps)

        if self.bbox_loss_type == "iou":
            return (1 - iou).mean()

        # Enclosing box (for GIoU, DIoU, CIoU)
        enclose_w = torch.max(pl, tl) + torch.max(pr, tr)
        enclose_h = torch.max(pt, tt) + torch.max(pb, tb)

        if self.bbox_loss_type == "giou":
            enclose_area = (enclose_w * enclose_h).clamp(min=eps)
            giou = iou - (enclose_area - union) / enclose_area
            return (1 - giou).mean()

        # Center distance (for DIoU, CIoU)
        # pred center offset from pixel: ((r-l)/2, (b-t)/2)
        # target center offset from pixel: ((tr-tl)/2, (tb-tt)/2)
        dx = (pr - pl) / 2 - (tr - tl) / 2
        dy = (pb - pt) / 2 - (tb - tt) / 2
        center_dist_sq = dx ** 2 + dy ** 2

        diagonal_sq = (enclose_w ** 2 + enclose_h ** 2).clamp(min=eps)

        if self.bbox_loss_type == "diou":
            diou = iou - center_dist_sq / diagonal_sq
            return (1 - diou).mean()

        # CIoU: add aspect ratio penalty
        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(tgt_w / tgt_h) - torch.atan(pred_w / pred_h), 2
        )
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)

        ciou = iou - center_dist_sq / diagonal_sq - alpha * v
        return (1 - ciou).mean()

    def forward(self, preds: dict, targets: list) -> torch.Tensor:
        cls_logits = preds["cls_logits"]
        bbox_pred = preds["bbox_pred"]
        centerness = preds["centerness"].squeeze(1)

        B, C, H, W = cls_logits.shape
        device = cls_logits.device

        cls_targets, reg_targets, ctr_targets, fg_mask = self._encode_targets(
            targets, H, W, self.num_classes, device
        )

        # --- Classification loss ---
        if self.cls_loss_type == "varifocal":
            # Compute IoU quality for positives
            quality_targets = torch.zeros(B, C, H, W, device=device)
            
            if fg_mask.sum() > 0:
                with torch.no_grad():
                    pred_ltrb_fg = bbox_pred.permute(0, 2, 3, 1)[fg_mask]
                    tgt_ltrb_fg = reg_targets.permute(0, 2, 3, 1)[fg_mask]
                    iou_quality = self._compute_iou_quality(pred_ltrb_fg, tgt_ltrb_fg)
                    
                    # Vectorized assignment via advanced indexing
                    cls_indices = cls_targets[fg_mask] - 1  # 0-indexed
                    fg_b, fg_h, fg_w = torch.where(fg_mask)
                    quality_targets[fg_b, cls_indices, fg_h, fg_w] = iou_quality
            
            cls_loss = self.cls_loss_fn(
                cls_logits.permute(0, 2, 3, 1).reshape(-1, C),
                quality_targets.permute(0, 2, 3, 1).reshape(-1, C)
            )
        else:
            # Standard focal loss (vectorized one-hot)
            class_range = torch.arange(1, C + 1, device=device).view(1, C, 1, 1)
            cls_one_hot = (cls_targets.unsqueeze(1) == class_range).float()
            
            p = torch.sigmoid(cls_logits)
            ce = F.binary_cross_entropy_with_logits(cls_logits, cls_one_hot, reduction="none")
            p_t = p * cls_one_hot + (1 - p) * (1 - cls_one_hot)
            focal_weight = (1 - p_t) ** self.focal_gamma
            alpha_t = self.focal_alpha * cls_one_hot + (1 - self.focal_alpha) * (1 - cls_one_hot)
            cls_loss = (alpha_t * focal_weight * ce).mean()

        # --- Box regression loss (computed directly in ltrb space, no meshgrid) ---
        if fg_mask.sum() > 0:
            pred_ltrb_fg = bbox_pred.permute(0, 2, 3, 1)[fg_mask]
            tgt_ltrb_fg = reg_targets.permute(0, 2, 3, 1)[fg_mask]
            bbox_loss = self._ltrb_box_loss(pred_ltrb_fg, tgt_ltrb_fg)
        else:
            bbox_loss = torch.tensor(0.0, device=device)

        # --- Centerness loss ---
        if fg_mask.sum() > 0:
            ctr_loss = F.binary_cross_entropy_with_logits(
                centerness[fg_mask], ctr_targets[fg_mask], reduction="mean"
            )
        else:
            ctr_loss = torch.tensor(0.0, device=device)

        total = (
            self.cls_weight * cls_loss
            + self.bbox_weight * bbox_loss
            + self.centerness_weight * ctr_loss
        )
        return total


# ============================================================================
# Height regression losses
# ============================================================================

class L1HeightLoss(_Loss):
    """Masked L1 loss for height regression.

    Only evaluates pixels where ``target > 0`` (i.e. ignores background) when
    ``ignore_zero=True``.
    """

    def __init__(self, ignore_zero: bool = True):
        super().__init__()
        self.ignore_zero = ignore_zero

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.ignore_zero:
            mask = target > 0
            if mask.sum() == 0:
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
            return F.l1_loss(pred[mask], target[mask])
        return F.l1_loss(pred, target)


class GradientMatchingLoss(_Loss):
    """Sobel-based gradient matching loss for sharp height boundaries.

    Computes the L1 difference between Sobel gradients of predicted and
    target height maps.  Evaluated only where ``target > 0`` when
    ``ignore_zero=True``.
    """

    def __init__(self, ignore_zero: bool = True):
        super().__init__()
        self.ignore_zero = ignore_zero

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradient(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        sx = self.sobel_x.to(device=x.device, dtype=x.dtype)
        sy = self.sobel_y.to(device=x.device, dtype=x.dtype)
        gx = F.conv2d(x, sx, padding=1)
        gy = F.conv2d(x, sy, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        grad_pred = self._gradient(pred)
        grad_target = self._gradient(target)

        if self.ignore_zero:
            if target.ndim == 3:
                mask = target.unsqueeze(1) > 0
            else:
                mask = target > 0
            if mask.sum() == 0:
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
            return F.l1_loss(grad_pred[mask], grad_target[mask])
        return F.l1_loss(grad_pred, grad_target)


class ScaleInvariantLogLoss(_Loss):
    """Scale-invariant log depth loss (Eigen et al., NIPS 2014).

    Defined as

        d_i  = log(pred_i) - log(target_i)
        L    = mean(d^2) - lambda_si * mean(d)^2

    The second term subtracts the squared mean of the residual (the
    "scale" component), so the network is rewarded for getting the
    *relative* depth structure right even if its absolute scale is off
    by a multiplicative constant. Recovers MSE-in-log-space when
    ``lambda_si=0`` and is fully scale-invariant when ``lambda_si=1``;
    Eigen used 0.5, BTS / AdaBins / NeWCRFs use 0.85, which is what we
    default to.

    Why we want this for ARKitScenes depth (and not L1 alone):
      * L1 minimisation predicts the conditional **median**, which over
        depth distributions is well-known to produce blurry, low-contrast
        outputs (Mathis et al. 2024; cf. notes in Depth-Anything-V2).
      * SI-log regularises the variance of the log-residual, which
        rewards correct *gradients* of depth even when the absolute
        offset is small — hence sharper outputs without amping the
        gradient term to the point of edge-noise.

    Pixels with ``target <= 0`` are no-return / sentinel and are skipped
    when ``ignore_zero=True``. Predictions are clamped to ``[eps, +inf)``
    before ``log`` to keep the loss finite when the head's raw
    regression dips negative early in training.
    """

    def __init__(
        self,
        lambda_si: float = 0.85,
        eps: float = 1.0e-3,
        ignore_zero: bool = True,
    ):
        super().__init__()
        if not 0.0 <= lambda_si <= 1.0:
            raise ValueError(
                f"lambda_si must be in [0, 1], got {lambda_si}"
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.lambda_si = float(lambda_si)
        self.eps = float(eps)
        self.ignore_zero = ignore_zero

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.ignore_zero:
            mask = target > 0
            if mask.sum() == 0:
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
            pred_v = pred[mask].clamp(min=self.eps)
            tgt_v = target[mask].clamp(min=self.eps)
        else:
            pred_v = pred.clamp(min=self.eps)
            tgt_v = target.clamp(min=self.eps)

        d = torch.log(pred_v) - torch.log(tgt_v)
        # Variance form: E[d^2] - lambda * E[d]^2. With lambda < 1 the
        # second term is strictly smaller than the first, so the loss
        # stays non-negative.
        return d.pow(2).mean() - self.lambda_si * d.mean().pow(2)


class EdgeAwareGradientLoss(_Loss):
    """Sobel-gradient matching with no-return mask erosion.

    Functionally a drop-in replacement for :class:`GradientMatchingLoss`,
    but corrects a subtle bug that bites depth datasets with sentinel
    "no-return" pixels (ARKitScenes ``highres_depth`` Faro scans store
    invalid pixels as ``0``):

      * :class:`GradientMatchingLoss` runs Sobel over the *raw* target
        depth map. Wherever a no-return zero borders a real depth, the
        target gradient is dominated by the depth -> 0 transition — an
        artefact of the sentinel, not of any real surface boundary.
      * If we then mask with ``target > 0`` we *keep* pixels at the
        boundary and tell the model to predict that artificial edge.
        The model dutifully learns to draw sharp ridges around every
        no-return hole, which looks crisp but is wrong everywhere.

    The fix is to erode the valid mask by ``erode_iters`` pixels before
    selecting which pixels contribute to the loss. With ``erode_iters=1``
    (default), every surviving pixel is guaranteed to have all eight
    Sobel-3x3 neighbours valid, so its target gradient is computed from
    real depths only. The boundary halo is silently excluded.

    Erosion is implemented as ``-max_pool2d(-mask, 3)`` (8-connected),
    differentiable-irrelevant since masks are bool.

    Args:
        ignore_zero:    Whether to mask out pixels where ``target == 0``.
                        ``False`` reproduces :class:`GradientMatchingLoss`
                        verbatim (no-return pixels participate in both
                        Sobel and the loss reduction).
        erode_iters:    Number of 3x3 erosion passes applied to the
                        valid mask before pixel selection.  ``1`` is
                        sufficient for a 3x3 Sobel kernel; bump to 2 if
                        you switch to a 5x5 gradient operator.
    """

    def __init__(
        self,
        ignore_zero: bool = True,
        erode_iters: int = 1,
    ):
        super().__init__()
        if erode_iters < 0:
            raise ValueError(f"erode_iters must be >= 0, got {erode_iters}")
        self.ignore_zero = ignore_zero
        self.erode_iters = int(erode_iters)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradient(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        sx = self.sobel_x.to(device=x.device, dtype=x.dtype)
        sy = self.sobel_y.to(device=x.device, dtype=x.dtype)
        gx = F.conv2d(x, sx, padding=1)
        gy = F.conv2d(x, sy, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    @staticmethod
    def _erode(mask: torch.Tensor, iters: int) -> torch.Tensor:
        """8-connected morphological erosion via min-pool."""
        m = mask.float()
        for _ in range(iters):
            m = -F.max_pool2d(-m, kernel_size=3, stride=1, padding=1)
        return m > 0.5

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        grad_pred = self._gradient(pred)
        grad_target = self._gradient(target)

        if not self.ignore_zero:
            return F.l1_loss(grad_pred, grad_target)

        if target.ndim == 3:
            valid = (target.unsqueeze(1) > 0)
        else:
            valid = (target > 0)
        mask = self._erode(valid, self.erode_iters) if self.erode_iters > 0 else valid

        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return F.l1_loss(grad_pred[mask], grad_target[mask])


class EdgeAwareSignedGradientLoss(_Loss):
    """PromptDA-style signed-component gradient loss with mask erosion.

    Replaces :class:`EdgeAwareGradientLoss`'s magnitude form
    ``L1(|nabla D_pred|, |nabla D_gt|)`` with the signed-component form
    ``L1(d D_pred/dx, d D_gt/dx) + L1(d D_pred/dy, d D_gt/dy)`` used by
    Prompt Depth Anything (Lin et al., CVPR 2025, eq. 1-2). The key
    difference is that this form preserves the direction of the depth
    change at every pixel, not just its magnitude, which gives a much
    stronger gradient signal at sharp edges -- a blurry prediction that
    happens to match the gradient *magnitude* of the target (e.g. a ramp
    integrated over a few pixels matching a sharp step) is still
    penalised heavily because its per-pixel signed components disagree.

    By linearity ``d(D_pred - D_gt)/dx = d D_pred/dx - d D_gt/dx`` so
    PromptDA's published form
    ``|d(D_pred - D_gt)/dx| + |d(D_pred - D_gt)/dy|`` is identical to
    ``L1(d D_pred/dx, d D_gt/dx) + L1(d D_pred/dy, d D_gt/dy)`` -- both
    return the same per-pixel scalar -- and the latter expresses the
    intent more transparently.

    Multi-scale extension (DepthAnythingV2 / PromptDA recipe):
    when ``scales`` contains more than one factor, the loss is summed
    across average-pooled versions of (pred, target) at each downscale
    factor ``s in scales``. Scale ``s`` first avg-pools both inputs by
    factor ``s``, then applies the same Sobel + signed-L1 protocol, and
    contributes ``L_s / len(scales)`` to the total. This delivers
    gradient signal across multiple frequency bands, which is empirically
    the single biggest sharpness lever in monocular / prompted depth
    losses (compare DA-v2's "gradient matching loss" applied at scales
    {1, 2, 4, 8} on the final output -- they do *not* use intermediate
    aux classifiers; the multi-scale loss on the final output is the
    canonical mechanism). Scale 1 means full resolution; scale 8 means
    the loss compares 8x-downsampled (pred, target).

    The no-return-mask erosion behaviour is identical to
    :class:`EdgeAwareGradientLoss`: pixels within ``erode_iters`` of any
    sentinel-zero target are excluded from the loss, so the model is
    not asked to predict the artificial step-edge that bordering a
    no-return hole creates in the raw target gradient. With multi-scale,
    the mask is also avg-pooled at each scale (a downsampled pixel is
    valid iff its parent block was majority-valid).

    Args:
        ignore_zero:    Whether to mask out pixels where ``target == 0``.
                        ``False`` skips the mask entirely (Sobel runs over
                        the full image).
        erode_iters:    Number of 3x3 erosion passes applied to the
                        valid mask before pixel selection.  ``1`` is
                        sufficient for a 3x3 Sobel kernel.
        scales:         List of integer downscale factors (``1`` =
                        full-res). Default ``[1]`` reproduces single-scale
                        behaviour. ``[1, 2, 4, 8]`` matches DA-v2.
    """

    def __init__(
        self,
        ignore_zero: bool = True,
        erode_iters: int = 1,
        scales: list[int] | tuple[int, ...] | None = None,
    ):
        super().__init__()
        if erode_iters < 0:
            raise ValueError(f"erode_iters must be >= 0, got {erode_iters}")
        if scales is None:
            scales = [1]
        scales = [int(s) for s in scales]
        if any(s < 1 for s in scales):
            raise ValueError(f"scales must all be >= 1, got {scales}")
        self.ignore_zero = ignore_zero
        self.erode_iters = int(erode_iters)
        self.scales = scales

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    @staticmethod
    def _erode(mask: torch.Tensor, iters: int) -> torch.Tensor:
        m = mask.float()
        for _ in range(iters):
            m = -F.max_pool2d(-m, kernel_size=3, stride=1, padding=1)
        return m > 0.5

    def _signed_grads(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        sx = self.sobel_x.to(device=x.device, dtype=x.dtype)
        sy = self.sobel_y.to(device=x.device, dtype=x.dtype)
        gx = F.conv2d(x, sx, padding=1)
        gy = F.conv2d(x, sy, padding=1)
        return gx, gy

    def _loss_at_scale(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        scale: int,
    ) -> torch.Tensor:
        """Compute the single-scale signed-gradient loss after avg-pooling
        both pred and target by ``scale``. ``scale=1`` is the
        identity / native-resolution case."""
        if scale > 1:
            # Use avg_pool2d for symmetric downsampling. We need 4D
            # tensors for pool; restore at end.
            p = pred if pred.ndim == 4 else pred.unsqueeze(1)
            t = target if target.ndim == 4 else target.unsqueeze(1)
            p = F.avg_pool2d(p, kernel_size=scale, stride=scale)
            t = F.avg_pool2d(t, kernel_size=scale, stride=scale)
        else:
            p = pred
            t = target

        gx_pred, gy_pred = self._signed_grads(p)
        gx_target, gy_target = self._signed_grads(t)

        if not self.ignore_zero:
            return F.l1_loss(gx_pred, gx_target) + F.l1_loss(gy_pred, gy_target)

        # Build the valid mask at this scale. Avg-pooling the boolean
        # mask gives a "fraction of valid pixels in the block"; we
        # threshold at >=0.5 so a block that was majority-valid is
        # treated as valid post-pool.
        if t.ndim == 3:
            valid = (t.unsqueeze(1) > 0).float()
        else:
            valid = (t > 0).float()
        if scale > 1 and target.shape != t.shape:
            # Already pooled (because we pooled t above); use it directly.
            pass
        valid_bool = valid > 0.5
        mask = self._erode(valid_bool, self.erode_iters) if self.erode_iters > 0 else valid_bool

        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return (
            F.l1_loss(gx_pred[mask], gx_target[mask])
            + F.l1_loss(gy_pred[mask], gy_target[mask])
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(self.scales) == 1 and self.scales[0] == 1:
            return self._loss_at_scale(pred, target, 1)
        # Average across scales so the contribution is independent of
        # scales-list length (otherwise the loss magnitude scales with
        # the number of bands and the user has to retune the weight
        # every time scales is changed).
        total = 0.0
        for s in self.scales:
            total = total + self._loss_at_scale(pred, target, s)
        return total / float(len(self.scales))


class EdgeAwareSmoothnessLoss(_Loss):
    """RGB-edge-aware smoothness regulariser (Godard et al. 2017).

    Penalises spatial gradients of the prediction except where the
    *input RGB* image has an edge. Used in essentially every modern
    monocular depth paper since SfMLearner / Monodepth (Godard et al.
    2017, "Unsupervised Monocular Depth Estimation with Left-Right
    Consistency", eq. 7) and adopted by depth-completion methods
    including PromptDA, NLSPN, GuideNet.

    Form
    ----
    For a 1-channel prediction ``D`` and a 3-channel RGB image
    ``I``:

    .. math::

        L_\\text{smooth} = \\frac{1}{N}\\sum_{p}\\;
            \\Big(
                |\\partial_x D|_p \\,e^{-\\beta\\,\\|\\partial_x I\\|_p}
              + |\\partial_y D|_p \\,e^{-\\beta\\,\\|\\partial_y I\\|_p}
            \\Big)

    The exponential weight collapses to ~0 at strong RGB edges
    (object boundaries -- the prediction is *allowed* to be sharp
    there) and to ~1 in homogeneous RGB regions (sky, grass, road --
    the prediction *should* be flat). It is the cleanest available
    "snap depth boundaries to RGB edges" signal.

    For *prompted* depth where the model occasionally also has CHM,
    this loss does not look at the CHM at all; it only penalises
    the prediction's gradient via the RGB image. This is the
    correct framing because (a) CHM may itself be smooth (low-res
    iPhone LiDAR for ARKit) and (b) we want sharpness anchored in
    the *RGB* boundary structure, which is what the qualitative
    failure mode in the NeurIPS qualitative figure shows -- soft
    blob-shaped trees instead of RGB-aligned canopies.

    Args:
        beta:           Sensitivity of the gradient gate. Larger
                        beta = sharper gate (loss vanishes faster
                        at RGB edges). Standard value 10.0 (Godard
                        et al.).
        rgb_grad_norm:  How to combine the 3 RGB gradient channels
                        into a scalar weight per pixel. ``"l1"``
                        (default) sums absolute gradient over the 3
                        RGB channels; ``"max"`` takes the per-pixel
                        max. ``"l1"`` is the original paper's
                        choice.
        ignore_zero:    If True, mask out pixels where ``target == 0``.
                        Mostly cosmetic for this loss because the
                        smoothness term doesn't reference the target,
                        but it lets us skip no-return holes that
                        would otherwise be regularised toward zero
                        gradient -- which is wrong because the
                        no-return *boundary* is a step edge.
    """

    def __init__(
        self,
        beta: float = 10.0,
        rgb_grad_norm: str = "l1",
        ignore_zero: bool = True,
    ):
        super().__init__()
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        if rgb_grad_norm not in ("l1", "max"):
            raise ValueError(
                f"rgb_grad_norm must be one of 'l1', 'max', got {rgb_grad_norm!r}"
            )
        self.beta = float(beta)
        self.rgb_grad_norm = rgb_grad_norm
        self.ignore_zero = bool(ignore_zero)

    @staticmethod
    def _grad_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward-difference gradients along x (last dim) and y
        (second-to-last). Returns same H,W shape as input via padding
        the last column / row with zeros."""
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        # Pad last col / row with zeros so shapes line up downstream.
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        return gx, gy

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image is None:
            raise RuntimeError(
                "EdgeAwareSmoothnessLoss requires the input RGB image. "
                "Pass it via the `image` kwarg from training_step / "
                "validation_step. The HeightEstimationModule._compute_loss "
                "machinery must inspect each loss's signature and forward "
                "the RGB tensor when accepted."
            )

        # Normalise shapes to [B, C, H, W].
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Pred gradients (1 channel).
        gx_d, gy_d = self._grad_xy(pred)

        # RGB gradients aggregated to a per-pixel scalar weight.
        gx_i, gy_i = self._grad_xy(image)
        if self.rgb_grad_norm == "l1":
            w_x = torch.exp(-self.beta * gx_i.abs().mean(dim=1, keepdim=True))
            w_y = torch.exp(-self.beta * gy_i.abs().mean(dim=1, keepdim=True))
        else:  # "max"
            w_x = torch.exp(-self.beta * gx_i.abs().amax(dim=1, keepdim=True))
            w_y = torch.exp(-self.beta * gy_i.abs().amax(dim=1, keepdim=True))

        loss_x = gx_d.abs() * w_x
        loss_y = gy_d.abs() * w_y

        if not self.ignore_zero:
            return loss_x.mean() + loss_y.mean()

        # Mask out no-return holes; we want the smoothness term to be
        # silent across the no-return step edge rather than dragging
        # predictions to zero gradient there.
        if target.ndim == 3:
            valid = (target.unsqueeze(1) > 0)
        else:
            valid = (target > 0)

        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # We mask both the per-pixel loss tensor and average only over
        # valid pixels so the loss is correctly scaled by the number
        # of valid sites instead of the full image.
        return (
            (loss_x * valid).sum() / valid.sum().clamp(min=1)
            + (loss_y * valid).sum() / valid.sum().clamp(min=1)
        )


class GradientWeightedL1Loss(_Loss):
    """L1 loss with per-pixel weights proportional to ``|nabla target|``.

    Form
    ----

    .. math::

        L = \\frac{1}{N_\\text{valid}}\\sum_p\\;
            (1 + \\alpha\\,\\|\\nabla g\\|_p)\\,|p - g|_p

    where ``\\nabla g`` is the per-pixel target gradient magnitude
    (Sobel L2). The base ``1`` keeps the loss everywhere a sane L1
    even where the target is locally flat; the ``alpha * ||grad g||``
    term up-weights pixels near object edges where boundary fidelity
    matters most. With ``alpha = 0`` this reduces to plain
    :class:`L1HeightLoss`.

    Used in DepthFormer / GLPN-style supervisory recipes; combined
    with multi-scale signed-gradient and edge-aware smoothness, it
    is the standard "boundary emphasis" tool in modern monocular
    depth.

    Args:
        alpha:          Edge-emphasis strength. ``5.0`` works well as
                        a default for tree-height (gradients are in
                        meters/pixel, mostly < 0.5). For ARKit-scale
                        depth (meters but sub-meter gradients) try
                        ``10.0``.
        ignore_zero:    Mask out pixels where ``target == 0`` (sentinel
                        for no-return holes / synthetic gaps). The
                        gradient computation does *not* mask
                        the no-return step edge -- ``alpha`` is small
                        enough that the spurious gradient at a
                        sentinel boundary contributes negligibly,
                        and erosion would slow the loss substantially
                        for marginal benefit.
    """

    def __init__(
        self,
        alpha: float = 5.0,
        ignore_zero: bool = True,
    ):
        super().__init__()
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        self.alpha = float(alpha)
        self.ignore_zero = bool(ignore_zero)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _target_grad_mag(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 3:
            t = t.unsqueeze(1)
        sx = self.sobel_x.to(device=t.device, dtype=t.dtype)
        sy = self.sobel_y.to(device=t.device, dtype=t.dtype)
        gx = F.conv2d(t, sx, padding=1)
        gy = F.conv2d(t, sy, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        grad_mag = self._target_grad_mag(target)
        weight = 1.0 + self.alpha * grad_mag
        l1 = (pred - target).abs() * weight

        if not self.ignore_zero:
            return l1.mean()

        valid = target > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return (l1 * valid).sum() / valid.sum().clamp(min=1)
