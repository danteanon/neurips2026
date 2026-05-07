import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import torchmetrics
import kornia
from tqdm import tqdm
# Import torchmetrics detection MAP and regression metrics
try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    TORCHMETRICS_DETECTION_AVAILABLE = True
except ImportError:
    TORCHMETRICS_DETECTION_AVAILABLE = False
    # Note: Will raise error when BBoxmAP is instantiated

try:
    from torchmetrics.regression import MeanSquaredError
    TORCHMETRICS_REGRESSION_AVAILABLE = True
except ImportError:
    TORCHMETRICS_REGRESSION_AVAILABLE = False

def compute_classes_weights(dataset, num_classes):
    n_classes = np.zeros(num_classes)
    n_pixels = 0
    for img, label in tqdm(dataset):
        c = torch.bincount(label.view(-1), minlength=num_classes)
        n = label.nelement()
        n_classes += c.data.numpy()
        n_pixels += n
    weights = 1 / np.log(1.02 + (n_classes / n_pixels))  # cf https://arxiv.org/pdf/1606.02147.pdf
    return weights.round(3, out=weights).tolist()

class Dice(torchmetrics.Metric):
    """
    Computes the Dice score for segmentation.
    
    Args:
        num_classes (int): Number of classes (including background)
        mode (str): 'multiclass' or 'binary'. Default: 'multiclass'
        ignore_index (int, optional): Class index to ignore (typically 0 for background). Default: None
    """
    def __init__(self, num_classes, mode='multiclass', ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.mode = mode
        self.ignore_index = ignore_index
        self.add_state("intersection", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("union", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        
    def update(self, preds, target):
        """
        Update metric states with new predictions and targets.
        
        Args:
            preds (torch.Tensor): Predictions [B, C, H, W] (logits) or [B, H, W] (class indices)
            target (torch.Tensor): Ground truth [B, H, W] (class indices)
        """
        if isinstance(preds, list):
            preds = torch.stack(preds)
            
        if isinstance(target, list):
            target = torch.stack(target)
        
        # Apply argmax if predictions are logits [B, C, H, W]
        if preds.dim() == 4:
            preds = torch.argmax(preds, dim=1)  # [B, H, W]
        
        # Also convert target if it's in one-hot format [B, C, H, W]
        if target.dim() == 4:
            target = torch.argmax(target, dim=1)  # [B, H, W]
        
        # Calculate per-class intersection and union
        for cls_idx in range(self.num_classes):
            if cls_idx == self.ignore_index:
                continue
            
            pred_mask = (preds == cls_idx)
            target_mask = (target == cls_idx)
            
            intersection = (pred_mask & target_mask).sum()
            union = pred_mask.sum() + target_mask.sum()
            
            self.intersection[cls_idx] += intersection
            self.union[cls_idx] += union

    def _compute(self):
        """
        Internal compute method that always returns per-class results.
        
        Returns:
            torch.Tensor: Per-class Dice scores of shape [num_classes]
        """
        if self.mode == 'binary':
            # Binary mode: return Dice for non-background class only
            # Assumes ignore_index is the background class
            if self.ignore_index is not None:
                # Find the foreground class (not ignore_index)
                foreground_idx = 1 if self.ignore_index == 0 else 0
                return 2 * self.intersection[foreground_idx] / (self.union[foreground_idx] + 1e-7)
            else:
                # No ignore_index, use class 1 as foreground
                return 2 * self.intersection[1] / (self.union[1] + 1e-7)
        
        elif self.mode == 'multiclass':
            # Multi-class mode: calculate per-class Dice scores
            return 2 * self.intersection / (self.union + 1e-7)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'binary' or 'multiclass'.")
    
    def compute(self, reduction='mean'):
        """
        Compute the Dice score with optional reduction.
        
        Args:
            reduction (str): 'mean' or 'none'. Default: 'mean'
                           'mean' returns average across valid classes
                           'none' returns per-class scores
        
        Returns:
            torch.Tensor: Dice score(s)
        """
        dice_per_class = self._compute()
        
        if reduction == 'none':
            return dice_per_class
        
        # Apply reduction for multiclass
        if self.mode == 'multiclass':
            # Mean across valid classes (excluding ignore_index)
            valid_mask = torch.ones(self.num_classes, dtype=torch.bool, device=dice_per_class.device)
            if self.ignore_index is not None:
                valid_mask[self.ignore_index] = False
            
            return dice_per_class[valid_mask].mean()
        else:
            # Binary mode already returns scalar
            return dice_per_class

class MeanIoU(torchmetrics.Metric):
    """
    Computes the IoU (Intersection over Union) score for segmentation tasks.
    
    Args:
        num_classes (int): Number of classes
        ignore_index (int, optional): Index to ignore in calculation (typically 0 for background). Default: None
    """
    def __init__(self, num_classes=1, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.add_state("intersection", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("union", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        
    def update(self, preds, target):
        """
        Update metric states with new predictions and targets.
        
        Args:
            preds (torch.Tensor): Predictions [B, C, H, W] (logits) or [B, H, W] (class indices)
            target (torch.Tensor): Ground truth [B, C, H, W] (one-hot) or [B, H, W] (class indices)
        """
        if isinstance(preds, list):
            preds = torch.stack(preds)
        
        if isinstance(target, list):
            target = torch.stack(target)
        
        # Apply argmax if predictions are logits [B, C, H, W]
        if preds.dim() == 4:
            preds = torch.argmax(preds, dim=1)  # [B, H, W]
        
        # Also convert target if it's in one-hot format [B, C, H, W]
        if target.dim() == 4:
            target = torch.argmax(target, dim=1)  # [B, H, W]
        
        # Ensure target is long type for class indices
        target = target.long()
        
        # Calculate IoU for each class
        for cls_idx in range(self.num_classes):
            if self.ignore_index is not None and cls_idx == self.ignore_index:
                continue
                
            pred_cls = (preds == cls_idx)
            target_cls = (target == cls_idx)
            
            intersection = (pred_cls & target_cls).sum()
            union = pred_cls.sum() + target_cls.sum() - intersection
            
            self.intersection[cls_idx] += intersection
            self.union[cls_idx] += union

    def _compute(self):
        """
        Internal compute method that always returns per-class results.
        
        Returns:
            torch.Tensor: Per-class IoU scores of shape [num_classes]
        """
        return self.intersection / (self.union + 1e-7)
    
    def compute(self, reduction='mean'):
        """
        Compute the IoU with optional reduction.
        
        Args:
            reduction (str): 'mean' or 'none'. Default: 'mean'
                           'mean' returns average across valid classes
                           'none' returns per-class IoU scores
        
        Returns:
            torch.Tensor: IoU score(s)
        """
        iou_per_class = self._compute()
        
        if reduction == 'none':
            return iou_per_class
        
        # Mean across valid classes (excluding ignore_index)
        valid_classes = torch.ones(self.num_classes, dtype=torch.bool, device=iou_per_class.device)
        if self.ignore_index is not None:
            valid_classes[self.ignore_index] = False
            
        return iou_per_class[valid_classes].mean()

class PolygonObjectMetric(torchmetrics.Metric):
    """
    Computes precision, recall and F1 score based on object detection principles for segmentation.
    Uses connected components to identify objects in predictions and ground truth.
    """
    def __init__(self, threshold=0.5, connectivity=2):
        super().__init__()
        self.threshold = threshold
        self.connectivity = connectivity
        self.add_state("true_positives", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0.), dist_reduce_fx="sum")
    
    def _find_connected_components(self, mask):
        """Find connected components in binary mask using Kornia"""
        # Ensure mask is float and has correct dimensions for kornia
        if mask.dim() == 3:  # [B, H, W]
            mask = mask.unsqueeze(1)  # [B, 1, H, W]
        elif mask.dim() == 2:  # [H, W]
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Use kornia's connected components
        labels = kornia.contrib.connected_components(mask.float(), num_iterations=100)
        
        # Remove channel dimension if it was added
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)  # [B, H, W]
            
        return labels.long()
    
    def _gpu_object_matching(self, pred_components, target_components, batch_idx):
        """Completely GPU-based object matching using vectorized operations"""
        pred_comp = pred_components[batch_idx]
        target_comp = target_components[batch_idx]
        
        # Get unique labels on GPU (no CPU transfer)
        pred_unique = torch.unique(pred_comp)
        target_unique = torch.unique(target_comp)
        
        # Remove background (0) on GPU
        pred_objects = pred_unique[pred_unique > 0]
        target_objects = target_unique[target_unique > 0]
        
        if len(pred_objects) == 0 or len(target_objects) == 0:
            return len(pred_objects), 0, len(target_objects)
        
        pred_objects_length = len(pred_objects) 
        gt_objects_length = len(target_objects)
        if pred_objects_length > 10 * gt_objects_length:
            print(f"Warning: Skipping object matching for batch {batch_idx}. "
                f"Total objects ({pred_objects_length}) exceeds threshold (10 * {gt_objects_length} = {10 * gt_objects_length})")
            return 0, 0, 0  # Return zeros to avoid computation
        # Create masks for all objects at once - vectorized
        pred_masks = pred_comp.unsqueeze(0) == pred_objects.unsqueeze(1).unsqueeze(2)  # [n_pred, H, W]
        target_masks = target_comp.unsqueeze(0) == target_objects.unsqueeze(1).unsqueeze(2)  # [n_target, H, W]
        
        # Vectorized IoU computation for all pairs
        intersection_matrix = torch.zeros(len(pred_objects), len(target_objects), device=pred_comp.device)
        union_matrix = torch.zeros_like(intersection_matrix)
        
        for i, pred_mask in enumerate(pred_masks):
            for j, target_mask in enumerate(target_masks):
                intersection = torch.logical_and(pred_mask, target_mask).sum()
                union = torch.logical_or(pred_mask, target_mask).sum()
                intersection_matrix[i, j] = intersection
                union_matrix[i, j] = union
        
        # Calculate IoU matrix on GPU
        iou_matrix = intersection_matrix / (union_matrix + 1e-7)
        
        # Find best matches using GPU operations
        matched_pairs = []
        used_targets = torch.zeros(len(target_objects), dtype=torch.bool, device=pred_comp.device)
        
        for i in range(len(pred_objects)):
            available_targets = ~used_targets
            if not available_targets.any():
                break
                
            iou_row = iou_matrix[i]
            iou_row = torch.where(available_targets, iou_row, torch.tensor(-1.0, device=pred_comp.device))
            
            best_target_idx = torch.argmax(iou_row)
            best_iou = iou_row[best_target_idx]
            
            if best_iou > self.threshold:
                matched_pairs.append((i, best_target_idx.item()))
                used_targets[best_target_idx] = True
        
        tp = len(matched_pairs)
        fp = len(pred_objects) - tp
        fn = len(target_objects) - tp
        
        return tp, fp, fn

    def update(self, preds, targets):
        """
        Update metrics with new predictions and targets.
        
        Args:
            preds (torch.Tensor): Class predictions from model
            targets (torch.Tensor): Ground truth labels
        """
        # Convert to binary for component analysis (assuming class 0 is background)
        preds_binary = (preds > 0).float()
        targets_binary = (targets > 0).float()
        
        # Find connected components
        with torch.no_grad():
            pred_components = self._find_connected_components(preds_binary)
            target_components = self._find_connected_components(targets_binary)

        tp, fp, fn = self._gpu_object_matching(pred_components, target_components, 0)
            
        # Update counts
        self.true_positives += tp
        self.false_positives += fp
        self.false_negatives += fn
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute precision, recall and F1 score"""
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-7)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        return {
            "val_precision_poly": precision,
            "val_recall_poly": recall,
            "val_F1_poly": f1
        }

class AveragePrecisionMetric(torchmetrics.Metric):
    """
    Computes Average Precision (AP) for segmentation, similar to COCO mAP.
    """
    def __init__(self, iou_thresholds=None):
        super().__init__()
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.add_state("ious", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")
    
    def update(self, preds, targets):
        """
        Update metrics with new predictions and targets.
        
        Args:
            preds (torch.Tensor): Class predictions from model
            targets (torch.Tensor): Ground truth labels
        """
        # Convert to binary for AP calculation (background vs foreground)
        preds_binary = (preds > 0).float()
        targets_binary = (targets > 0).float()
        
        # Calculate IoU for each prediction
        batch_size = preds.shape[0]
        for b in range(batch_size):
            pred = preds_binary[b]
            target = targets_binary[b]
            
            intersection = torch.logical_and(pred, target).sum().float()
            union = torch.logical_or(pred, target).sum().float()
            
            iou = intersection / (union + 1e-7)
            
            self.ious.append(iou.unsqueeze(0))
            self.labels.append(torch.as_tensor(1 if target.sum() > 0 else 0, device=target.device).unsqueeze(0))
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute Average Precision at different IoU thresholds"""
        if not self.ious:
            return {"mAP": torch.tensor(0.0)}
            
        ious = torch.cat(self.ious)
        labels = torch.cat(self.labels)
        
        ap_results = {}
        for threshold in self.iou_thresholds:
            # Predictions that meet the IoU threshold
            matches = (ious >= threshold).float()
            
            # Calculate precision and recall
            tp = (matches * labels).sum()
            fp = (matches * (1 - labels)).sum()
            fn = ((1 - matches) * labels).sum()
            
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            
            ap_results[f"AP_{int(threshold*100)}"] = precision * recall
        
        # Calculate mean AP
        ap_values = torch.stack(list(ap_results.values()))
        mean_ap = torch.mean(ap_values)
        ap_results["mAP"] = mean_ap
        
        return ap_results


# =============================================================================
# Bounding Box Validation Metrics
# =============================================================================

def bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate IoU between two bounding boxes in xywh format.
    
    Args:
        bbox1, bbox2: [x, y, width, height] format
        
    Returns:
        IoU value between 0 and 1
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)
    
    if left >= right or top >= bottom:
        return 0.0
    
    intersection = (right - left) * (bottom - top)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


class BBoxIoU(torchmetrics.Metric):
    """IoU calculation for bounding boxes"""
    
    def __init__(self):
        super().__init__()
        self.add_state("total_iou", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, pred_bboxes: List[Dict], gt_bboxes: List[Dict]):
        """
        Update IoU metric with predicted and ground truth bboxes.
        
        Args:
            pred_bboxes: List of {"bbox": [x, y, w, h], "class": int, "confidence": float}
            gt_bboxes: List of {"bbox": [x, y, w, h], "class": int}
        """
        if not pred_bboxes or not gt_bboxes:
            return
        
        total_iou = 0.0
        count = 0
        
        # Match predictions to ground truth (simple greedy matching)
        for pred in pred_bboxes:
            best_iou = 0.0
            for gt in gt_bboxes:
                if pred.get("class", 0) == gt.get("class", 0):  # Same class
                    iou = bbox_iou(pred["bbox"], gt["bbox"])
                    best_iou = max(best_iou, iou)
            
            total_iou += best_iou
            count += 1
        
        self.total_iou += torch.tensor(total_iou)
        self.count += torch.tensor(count)
    
    def compute(self):
        """Compute mean IoU"""
        return self.total_iou / (self.count + 1e-7)


def _convert_to_torchmetrics_format(pred_bboxes: List[Dict], gt_bboxes: List[Dict]):
    """
    Convert our bbox format to torchmetrics detection format.
    
    Args:
        pred_bboxes: List of {"bbox": [x, y, w, h], "class": int, "confidence": float}
        gt_bboxes: List of {"bbox": [x, y, w, h], "class": int}
        
    Returns:
        preds, targets in torchmetrics format
    """
    # Convert predictions
    if pred_bboxes:
        pred_boxes = torch.tensor([pred["bbox"] for pred in pred_bboxes], dtype=torch.float32)
        pred_scores = torch.tensor([pred.get("confidence", 1.0) for pred in pred_bboxes], dtype=torch.float32)
        pred_labels = torch.tensor([pred.get("class", 1) for pred in pred_bboxes], dtype=torch.int64)
        
        pred_dict = {
            "boxes": pred_boxes,
            "scores": pred_scores,
            "labels": pred_labels
        }
    else:
        pred_dict = {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "scores": torch.empty(0, dtype=torch.float32),
            "labels": torch.empty(0, dtype=torch.int64)
        }
    
    # Convert ground truth
    if gt_bboxes:
        gt_boxes = torch.tensor([gt["bbox"] for gt in gt_bboxes], dtype=torch.float32)
        gt_labels = torch.tensor([gt.get("class", 1) for gt in gt_bboxes], dtype=torch.int64)
        
        gt_dict = {
            "boxes": gt_boxes,
            "labels": gt_labels
        }
    else:
        gt_dict = {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty(0, dtype=torch.int64)
        }
    
    return pred_dict, gt_dict


class BBoxmAP(torchmetrics.Metric):
    """
    Mean Average Precision for bbox detection using torchmetrics.
    Requires torchvision >= 0.8.0 for proper functionality.
    """
    
    def __init__(self, iou_thresholds=None, box_format='xywh', class_metrics=False):
        super().__init__()
        
        if not TORCHMETRICS_DETECTION_AVAILABLE:
            raise ImportError(
                "torchmetrics detection module not available. "
                "Please install torchvision >= 0.8.0: pip install torchvision>=0.8.0"
            )
        
        self.box_format = box_format
        self.class_metrics = class_metrics
        
        # Use official torchmetrics implementation
        self.map_metric = MeanAveragePrecision(
            box_format=box_format,
            iou_thresholds=iou_thresholds,
            class_metrics=class_metrics
        )
    
    def update(self, pred_bboxes: List[Dict], gt_bboxes: List[Dict]):
        """
        Update mAP metric with predictions and ground truth.
        
        Args:
            pred_bboxes: List of {"bbox": [x, y, w, h], "class": int, "confidence": float}
            gt_bboxes: List of {"bbox": [x, y, w, h], "class": int}
        """
        # Convert to torchmetrics format and update
        pred_dict, gt_dict = _convert_to_torchmetrics_format(pred_bboxes, gt_bboxes)
        self.map_metric.update([pred_dict], [gt_dict])
    
    def compute(self):
        """Compute mAP using official torchmetrics implementation"""
        results = self.map_metric.compute()
        return {k: v for k, v in results.items()}


class BBoxPrecisionRecall(torchmetrics.Metric):
    """Precision and Recall for bounding box detection"""
    
    def __init__(self, iou_threshold=0.5):
        super().__init__()
        self.iou_threshold = iou_threshold
        
        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, pred_bboxes: List[Dict], gt_bboxes: List[Dict]):
        """
        Update precision/recall with predictions and ground truth.
        
        Args:
            pred_bboxes: List of {"bbox": [x, y, w, h], "class": int, "confidence": float}
            gt_bboxes: List of {"bbox": [x, y, w, h], "class": int}
        """
        if not gt_bboxes:
            self.false_positives += len(pred_bboxes)
            return
        
        if not pred_bboxes:
            self.false_negatives += len(gt_bboxes)
            return
        
        # Match predictions to ground truth
        used_gt = set()
        matched_preds = 0
        
        for pred in pred_bboxes:
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_bboxes):
                if gt_idx in used_gt:
                    continue
                
                if pred.get("class", 0) == gt.get("class", 0):
                    iou = bbox_iou(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_iou >= self.iou_threshold and best_gt_idx != -1:
                used_gt.add(best_gt_idx)
                matched_preds += 1
        
        self.true_positives += matched_preds
        self.false_positives += len(pred_bboxes) - matched_preds
        self.false_negatives += len(gt_bboxes) - matched_preds
    
    def compute(self):
        """Compute precision, recall, and F1 score"""
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-7)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }




def calculate_bbox_metrics(predictions: List[List[Dict]], ground_truths: List[List[Dict]], 
                          iou_thresholds=None, class_metrics=False):
    """
    Calculate comprehensive bounding box validation metrics using torchmetrics.
    
    Args:
        predictions: List of prediction lists, each containing {"bbox": [x, y, w, h], "class": int, "confidence": float}
        ground_truths: List of ground truth lists, each containing {"bbox": [x, y, w, h], "class": int}
        iou_thresholds: IoU thresholds for evaluation. If None, uses default [0.1, 0.2, ..., 0.95]
        class_metrics: Whether to compute per-class metrics
    
    Returns:
        Dictionary with all calculated metrics including:
        - Standard COCO mAP evaluation (IoU 0.5:0.95)
        - mAP@0.5, mAP@0.75 for common thresholds  
        - Individual AP values for each IoU threshold (ap_10, ap_20, ap_30, etc.)
        - Area-based metrics (small, medium, large objects)
        - Mean Average Recall (mAR) at different detection limits
        - Per-class metrics (if enabled)
        - Custom IoU, precision, recall, F1 metrics
        - Count-based metrics: MSE and RMSE between predicted and ground truth counts
    
    Raises:
        ImportError: If torchmetrics detection module is not available
    """
    # Set default thresholds if not provided
    if iou_thresholds is None:
        iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    # Initialize main metrics
    bbox_iou_metric = BBoxIoU()
    bbox_pr_metric = BBoxPrecisionRecall()
    bbox_map_metric = BBoxmAP(
        box_format='xywh', 
        class_metrics=class_metrics
    )
    
    # Initialize count metrics
    if TORCHMETRICS_REGRESSION_AVAILABLE:
        count_mse_metric = MeanSquaredError(squared=True)   # MSE
        count_rmse_metric = MeanSquaredError(squared=False) # RMSE
        count_metrics_available = True
    else:
        count_metrics_available = False
    
    # Initialize individual threshold metrics for getting individual AP values
    individual_map_metrics = {}
    for threshold in iou_thresholds:
        threshold_key = f"ap_{int(threshold * 100):02d}"
        individual_map_metrics[threshold_key] = BBoxmAP(
            box_format='xywh',
            iou_thresholds=[threshold],  # Single threshold for this metric
            class_metrics=False  # Keep individual metrics simple
        )
    
    # Update all metrics with all data
    for pred_list, gt_list in zip(predictions, ground_truths):
        bbox_iou_metric.update(pred_list, gt_list)
        bbox_map_metric.update(pred_list, gt_list)
        bbox_pr_metric.update(pred_list, gt_list)
        
        # Update count metrics
        if count_metrics_available:
            pred_count = torch.tensor(float(len(pred_list)))
            gt_count = torch.tensor(float(len(gt_list)))
            count_mse_metric.update(pred_count.unsqueeze(0), gt_count.unsqueeze(0))
            count_rmse_metric.update(pred_count.unsqueeze(0), gt_count.unsqueeze(0))
        
        # Update individual threshold metrics
        for metric in individual_map_metrics.values():
            metric.update(pred_list, gt_list)
    
    # Compute main metrics
    results = {
        "mean_iou": bbox_iou_metric.compute(),
        **bbox_map_metric.compute(),
        **bbox_pr_metric.compute()
    }
    
    # Add count metrics if available
    if count_metrics_available:
        results["count_mse"] = count_mse_metric.compute()
        results["count_rmse"] = count_rmse_metric.compute()
    
    # Add individual AP values for each IoU threshold
    for threshold_key, metric in individual_map_metrics.items():
        individual_result = metric.compute()
        results[threshold_key] = individual_result['map']
    
    return results


# =============================================================================
# SAE Monosemanticity Metrics
# =============================================================================

class DeadFeatureMetric(torchmetrics.Metric):
    """
    Track SAE feature usage statistics across training.
    
    Measures:
    - Dead feature ratio: Features that never activate
    - Rare feature ratio: Features that activate < threshold
    - Feature usage entropy: How uniformly features are used
    - Mean feature usage: Average activation frequency
    
    Args:
        d_hidden: SAE dictionary size (number of features)
        dead_threshold: Usage rate below this = dead (default: 0.0)
        rare_threshold: Usage rate below this = rare (default: 0.001)
    
    Example:
        >>> metric = DeadFeatureMetric(d_hidden=16384)
        >>> metric.update(h_sparse)  # (B, N, d_hidden)
        >>> stats = metric.compute()
        >>> print(stats['dead_feature_ratio'])
    """
    
    def __init__(
        self, 
        d_hidden: int, 
        dead_threshold: float = 0.0,
        rare_threshold: float = 0.001
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.dead_threshold = dead_threshold
        self.rare_threshold = rare_threshold
        
        # Cumulative feature activation counts
        self.add_state("feature_counts", default=torch.zeros(d_hidden), dist_reduce_fx="sum")
        self.add_state("total_tokens", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def update(self, h_sparse: torch.Tensor):
        """
        Update feature usage statistics.
        
        Args:
            h_sparse: Sparse SAE features (B, N, d_hidden) or (B*N, d_hidden)
        """
        if h_sparse.dim() == 3:
            B, N, d_hidden = h_sparse.shape
            h_flat = h_sparse.reshape(B * N, d_hidden)
        else:
            h_flat = h_sparse
            
        # Count activations per feature (any non-zero activation counts)
        active = (h_flat.abs() > 1e-8).float()  # (tokens, d_hidden)
        self.feature_counts += active.sum(dim=0).to(self.feature_counts.device)
        self.total_tokens += torch.tensor(h_flat.shape[0], device=self.total_tokens.device, dtype=torch.float32)
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute feature usage statistics."""
        if self.total_tokens == 0:
            return {
                'dead_feature_ratio': torch.tensor(0.0),
                'rare_feature_ratio': torch.tensor(0.0),
                'feature_usage_entropy': torch.tensor(0.0),
                'mean_feature_usage': torch.tensor(0.0),
            }
        
        usage_rate = self.feature_counts / self.total_tokens
        
        # Dead features: never activated
        dead_mask = usage_rate <= self.dead_threshold
        dead_ratio = dead_mask.float().mean()
        
        # Rare features: activated < threshold
        rare_mask = usage_rate < self.rare_threshold
        rare_ratio = rare_mask.float().mean()
        
        # Feature usage entropy (higher = more uniform usage)
        # Normalize to probability distribution
        usage_prob = usage_rate / (usage_rate.sum() + 1e-8)
        entropy = -(usage_prob * torch.log(usage_prob + 1e-8)).sum()
        # Normalize by max entropy (log(d_hidden))
        max_entropy = torch.log(torch.tensor(float(self.d_hidden), device=entropy.device))
        normalized_entropy = entropy / max_entropy
        
        return {
            'dead_feature_ratio': dead_ratio,
            'rare_feature_ratio': rare_ratio,
            'feature_usage_entropy': normalized_entropy,
            'mean_feature_usage': usage_rate.mean(),
        }


class ClusterPurityMetric(torchmetrics.Metric):
    """
    Cluster patches by their sparse codes and measure coherence.
    
    Uses GPU-based KMeans clustering on sparse codes and computes:
    - Silhouette score: Cluster separation quality (approximated for speed)
    - NMI with ground truth: Alignment with class labels
    
    Note: Uses pure PyTorch for GPU acceleration, no sklearn dependency.
    
    Args:
        num_classes: Number of segmentation classes
        n_clusters: Number of clusters for KMeans (default: 50)
        max_samples: Maximum samples to store for clustering (default: 5000)
        kmeans_iters: Number of KMeans iterations (default: 10)
    
    Example:
        >>> metric = ClusterPurityMetric(num_classes=9, n_clusters=50)
        >>> metric.update(h_sparse, class_labels)
        >>> stats = metric.compute()
        >>> print(stats['silhouette_score'])
    """
    
    def __init__(
        self,
        num_classes: int,
        n_clusters: int = 50,
        max_samples: int = 5000,
        kmeans_iters: int = 10
    ):
        super().__init__()
        self.num_classes = num_classes
        self.n_clusters = n_clusters
        self.max_samples = max_samples
        self.kmeans_iters = kmeans_iters
        
        # Store samples as lists (can't use tensors for variable-length accumulation)
        self.add_state("codes_list", default=[], dist_reduce_fx=None)
        self.add_state("labels_list", default=[], dist_reduce_fx=None)
        self.add_state("sample_count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, h_sparse: torch.Tensor, class_labels: torch.Tensor):
        """
        Update with new samples.
        
        Args:
            h_sparse: Sparse SAE features (B, N, d_hidden)
            class_labels: Class labels (B, N) at feature resolution
        """
        if self.sample_count >= self.max_samples:
            return
            
        B, N, d_hidden = h_sparse.shape
        
        # Flatten and sample
        h_flat = h_sparse.reshape(B * N, d_hidden)
        labels_flat = class_labels.reshape(B * N)
        
        # Random sample if too many
        remaining = self.max_samples - self.sample_count
        if h_flat.shape[0] > remaining:
            indices = torch.randperm(h_flat.shape[0], device=h_flat.device)[:remaining]
            h_flat = h_flat[indices]
            labels_flat = labels_flat[indices]
        
        self.codes_list.append(h_flat.detach().cpu())
        self.labels_list.append(labels_flat.detach().cpu())
        self.sample_count += h_flat.shape[0]
    
    def _gpu_kmeans(self, data: torch.Tensor, n_clusters: int, n_iters: int) -> torch.Tensor:
        """
        Fast GPU-based KMeans clustering.
        
        Args:
            data: (N, D) tensor of data points
            n_clusters: Number of clusters
            n_iters: Number of iterations
            
        Returns:
            (N,) tensor of cluster assignments
        """
        N, D = data.shape
        device = data.device
        
        # Initialize centroids using kmeans++ style (random from data)
        indices = torch.randperm(N, device=device)[:n_clusters]
        centroids = data[indices].clone()  # (K, D)
        
        for _ in range(n_iters):
            # Compute distances to all centroids: (N, K)
            # Using squared L2 distance for efficiency
            dists = torch.cdist(data, centroids, p=2)  # (N, K)
            
            # Assign to nearest centroid
            assignments = dists.argmin(dim=1)  # (N,)
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(n_clusters, device=device)
            
            for k in range(n_clusters):
                mask = assignments == k
                if mask.any():
                    new_centroids[k] = data[mask].mean(dim=0)
                    counts[k] = mask.sum()
                else:
                    # Keep old centroid if cluster is empty
                    new_centroids[k] = centroids[k]
            
            centroids = new_centroids
        
        # Final assignment
        dists = torch.cdist(data, centroids, p=2)
        assignments = dists.argmin(dim=1)
        
        return assignments
    
    def _fast_silhouette_score(
        self, 
        data: torch.Tensor, 
        assignments: torch.Tensor,
        sample_size: int = 1000
    ) -> torch.Tensor:
        """
        Fast approximation of silhouette score on GPU.
        
        Uses sampling for efficiency.
        
        Args:
            data: (N, D) tensor
            assignments: (N,) cluster assignments
            sample_size: Number of samples for approximation
            
        Returns:
            Scalar silhouette score
        """
        N = data.shape[0]
        device = data.device
        n_clusters = assignments.max().item() + 1
        
        # Sample for efficiency
        if N > sample_size:
            indices = torch.randperm(N, device=device)[:sample_size]
            data = data[indices]
            assignments = assignments[indices]
            N = sample_size
        
        # Compute pairwise distances (can be memory intensive)
        # Use chunked computation for large N
        if N > 2000:
            # Too large, use approximation based on centroid distances
            centroids = torch.zeros(n_clusters, data.shape[1], device=device)
            for k in range(n_clusters):
                mask = assignments == k
                if mask.any():
                    centroids[k] = data[mask].mean(dim=0)
            
            # Approximate: a(i) = distance to own centroid, b(i) = min distance to other centroids
            a_scores = torch.zeros(N, device=device)
            b_scores = torch.full((N,), float('inf'), device=device)
            
            for i in range(N):
                own_cluster = assignments[i]
                a_scores[i] = (data[i] - centroids[own_cluster]).norm()
                
                for k in range(n_clusters):
                    if k != own_cluster:
                        dist = (data[i] - centroids[k]).norm()
                        b_scores[i] = torch.min(b_scores[i], dist)
            
            silhouette = (b_scores - a_scores) / torch.max(a_scores, b_scores).clamp(min=1e-8)
            return silhouette.mean()
        
        # Full pairwise computation for smaller N
        dists = torch.cdist(data, data, p=2)  # (N, N)
        
        a_scores = torch.zeros(N, device=device)
        b_scores = torch.full((N,), float('inf'), device=device)
        
        for k in range(n_clusters):
            mask = assignments == k
            cluster_size = mask.sum()
            
            if cluster_size > 1:
                # a(i) = mean distance to same cluster (excluding self)
                cluster_dists = dists[mask][:, mask]
                a_scores[mask] = cluster_dists.sum(dim=1) / (cluster_size - 1)
            
            # b(i) = min mean distance to other clusters
            for j in range(n_clusters):
                if j != k:
                    other_mask = assignments == j
                    if other_mask.any():
                        other_dists = dists[mask][:, other_mask].mean(dim=1)
                        b_scores[mask] = torch.min(b_scores[mask], other_dists)
        
        # Silhouette score
        silhouette = (b_scores - a_scores) / torch.max(a_scores, b_scores).clamp(min=1e-8)
        
        return silhouette.mean()
    
    def _nmi_score(self, labels_true: torch.Tensor, labels_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute Normalized Mutual Information on GPU.
        
        Args:
            labels_true: (N,) ground truth labels
            labels_pred: (N,) predicted cluster assignments
            
        Returns:
            Scalar NMI score
        """
        N = labels_true.shape[0]
        device = labels_true.device
        
        # Get unique labels
        classes = labels_true.unique()
        clusters = labels_pred.unique()
        n_classes = len(classes)
        n_clusters = len(clusters)
        
        # Build contingency matrix
        contingency = torch.zeros(n_classes, n_clusters, device=device)
        for i, c in enumerate(classes):
            for j, k in enumerate(clusters):
                contingency[i, j] = ((labels_true == c) & (labels_pred == k)).sum()
        
        # Marginals
        row_sum = contingency.sum(dim=1)  # (n_classes,)
        col_sum = contingency.sum(dim=0)  # (n_clusters,)
        
        # Entropy of true labels
        p_true = row_sum / N
        H_true = -(p_true * torch.log(p_true + 1e-10)).sum()
        
        # Entropy of predicted labels
        p_pred = col_sum / N
        H_pred = -(p_pred * torch.log(p_pred + 1e-10)).sum()
        
        # Mutual information
        # MI = sum_ij p(i,j) * log(p(i,j) / (p(i) * p(j)))
        p_joint = contingency / N
        p_outer = p_true.unsqueeze(1) * p_pred.unsqueeze(0)
        
        # Avoid log(0)
        mask = p_joint > 0
        mi = (p_joint[mask] * torch.log(p_joint[mask] / (p_outer[mask] + 1e-10))).sum()
        
        # Normalized MI
        nmi = 2 * mi / (H_true + H_pred + 1e-10)
        
        return nmi.clamp(0, 1)
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute clustering metrics using GPU-based implementations."""
        if len(self.codes_list) == 0:
            return {
                'silhouette_score': torch.tensor(0.0),
                'nmi_with_gt': torch.tensor(0.0),
            }
        
        # Concatenate all samples
        codes = torch.cat(self.codes_list, dim=0)
        labels = torch.cat(self.labels_list, dim=0)
        
        # Need at least n_clusters samples
        if codes.shape[0] < self.n_clusters:
            return {
                'silhouette_score': torch.tensor(0.0),
                'nmi_with_gt': torch.tensor(0.0),
            }
        
        try:
            # Move to GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            codes = codes.to(device)
            labels = labels.to(device)
            
            # GPU-based KMeans
            cluster_assignments = self._gpu_kmeans(
                codes, 
                self.n_clusters, 
                self.kmeans_iters
            )
            
            # Fast silhouette score
            sil_score = self._fast_silhouette_score(codes, cluster_assignments)
            
            # NMI with ground truth
            nmi = self._nmi_score(labels, cluster_assignments)
            
            return {
                'silhouette_score': sil_score.cpu(),
                'nmi_with_gt': nmi.cpu(),
            }
        except Exception:
            return {
                'silhouette_score': torch.tensor(0.0),
                'nmi_with_gt': torch.tensor(0.0),
            }


class FeatureConsistencyMetric(torchmetrics.Metric):
    """
    Measure if visually similar patches use similar sparse codes.
    
    Computes correlation between:
    - Visual similarity (cosine similarity of flattened patches)
    - Code similarity (Jaccard similarity of active features)
    
    High correlation = features are semantically meaningful (monosemantic)
    
    Args:
        max_samples: Maximum samples to store (default: 2000)
        
    Example:
        >>> metric = FeatureConsistencyMetric(max_samples=2000)
        >>> metric.update(h_sparse, images)
        >>> stats = metric.compute()
        >>> print(stats['visual_code_correlation'])
    """
    
    def __init__(self, max_samples: int = 2000):
        super().__init__()
        self.max_samples = max_samples
        
        # Store embeddings and binary codes
        self.add_state("embeddings_list", default=[], dist_reduce_fx=None)
        self.add_state("codes_list", default=[], dist_reduce_fx=None)
        self.add_state("sample_count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(
        self, 
        h_sparse: torch.Tensor, 
        images: torch.Tensor,
        patch_size: int = 16
    ):
        """
        Update with new samples.
        
        Args:
            h_sparse: Sparse SAE features (B, N, d_hidden)
            images: Original images (B, C, H, W)
            patch_size: Size of each patch (default: 16 for ViT)
        """
        if self.sample_count >= self.max_samples:
            return
        
        B, N, d_hidden = h_sparse.shape
        _, C, H, W = images.shape
        
        # Compute patch grid size
        h_patches = H // patch_size
        w_patches = W // patch_size
        
        # Ensure N matches expected patches
        if N != h_patches * w_patches:
            # Mismatch, skip this batch
            return
        
        # Extract patch embeddings (simple: average pool each patch region)
        # Reshape to (B, C, h_patches, patch_size, w_patches, patch_size)
        patches = images.reshape(B, C, h_patches, patch_size, w_patches, patch_size)
        # Average over patch dimensions -> (B, C, h_patches, w_patches)
        patch_embed = patches.mean(dim=(3, 5))
        # Reshape to (B, N, C)
        patch_embed = patch_embed.reshape(B, C, -1).permute(0, 2, 1)
        
        # Flatten batch
        patch_embed_flat = patch_embed.reshape(B * N, C)
        h_flat = h_sparse.reshape(B * N, d_hidden)
        
        # Binary codes (which features are active)
        codes_binary = (h_flat.abs() > 1e-8).float()
        
        # Random sample if too many
        remaining = self.max_samples - self.sample_count
        if patch_embed_flat.shape[0] > remaining:
            indices = torch.randperm(patch_embed_flat.shape[0])[:remaining]
            patch_embed_flat = patch_embed_flat[indices]
            codes_binary = codes_binary[indices]
        
        self.embeddings_list.append(patch_embed_flat.detach().cpu())
        self.codes_list.append(codes_binary.detach().cpu())
        self.sample_count += patch_embed_flat.shape[0]
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute visual-code correlation."""
        if len(self.embeddings_list) == 0 or self.sample_count < 100:
            return {
                'visual_code_correlation': torch.tensor(0.0),
                'mean_code_similarity': torch.tensor(0.0),
            }
        
        # Concatenate all samples
        embeddings = torch.cat(self.embeddings_list, dim=0)  # (N, C)
        codes = torch.cat(self.codes_list, dim=0)  # (N, d_hidden)
        
        # Subsample for efficiency (pairwise computation is O(N^2))
        n_samples = min(500, embeddings.shape[0])
        if embeddings.shape[0] > n_samples:
            indices = torch.randperm(embeddings.shape[0])[:n_samples]
            embeddings = embeddings[indices]
            codes = codes[indices]
        
        # Compute pairwise visual similarity (cosine)
        embeddings_norm = F.normalize(embeddings, dim=-1)
        visual_sim = embeddings_norm @ embeddings_norm.T  # (N, N)
        
        # Compute pairwise code similarity (Jaccard on active features)
        # Jaccard = intersection / union
        intersection = codes @ codes.T  # (N, N) - dot product of binary = intersection count
        # Union = sum_i + sum_j - intersection
        code_sums = codes.sum(dim=-1, keepdim=True)  # (N, 1)
        union = code_sums + code_sums.T - intersection
        code_sim = intersection / (union + 1e-8)  # Jaccard similarity
        
        # Get upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones_like(visual_sim), diagonal=1).bool()
        visual_flat = visual_sim[mask]
        code_flat = code_sim[mask]
        
        # Compute Pearson correlation
        visual_mean = visual_flat.mean()
        code_mean = code_flat.mean()
        visual_centered = visual_flat - visual_mean
        code_centered = code_flat - code_mean
        
        correlation = (visual_centered * code_centered).sum() / (
            torch.sqrt((visual_centered ** 2).sum() * (code_centered ** 2).sum()) + 1e-8
        )
        
        return {
            'visual_code_correlation': correlation,
            'mean_code_similarity': code_flat.mean(),
        }