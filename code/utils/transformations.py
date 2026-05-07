import torch.nn.functional as F
import albumentations as A
import cv2
import numpy as np
import torch

def ensure_channel_last_decorator(fn):
    def wrapper(self, image, labels, *args, **kwargs):
        image, is_channel_first = self._ensure_channel_last(image)
        
        # Handle both single label and multiple labels
        is_single_label = not isinstance(labels, (list, tuple))
        if is_single_label:
            labels = [labels]  # Convert single label to list
            
        # Convert all labels to channel-last format
        labels_converted = []
        labels_channel_first_flags = []
        for label in labels:
            label_converted, is_label_channel_first = self._ensure_channel_last(label)
            labels_converted.append(label_converted)
            labels_channel_first_flags.append(is_label_channel_first)
        
        result = fn(self, image, labels_converted, *args, **kwargs)
        
        # result can be (image, labels) or (image, labels, ...)
        if isinstance(result, tuple) and len(result) >= 2:
            image_out = self._restore_format(result[0], is_channel_first)
            labels_out = []
            
            # Restore format for all labels
            for i, label in enumerate(result[1]):
                label_restored = self._restore_format(label, labels_channel_first_flags[i])
                labels_out.append(label_restored)
            
            # If input was single label, return single label; otherwise return list
            if is_single_label:
                labels_out = labels_out[0]
            
            # Return with original format restored
            return (image_out, labels_out) + result[2:]
        else:
            return result
    return wrapper

class Transformations:
    """
    A class to handle image transformations dynamically based on specified names.

    Available Augmentations:
    --------------------------
    - crop: Apply random cropping.
    - spatial: Apply spatial transformations (scaling, flipping, rotating).
    - color: Apply color transformations (sharpening, brightness, gamma, etc.).
    - noise: Apply noise transformations (blur, noise, downscaling, etc.).
    - super_resolution: Downsample image to 0.75x then upsample back (labels unchanged).
                       Teaches model to predict high-res segmentation from lower-res input.
                       Probability controlled by super_resolution_p parameter (default: 0.5).
    
    Available Scaling Operations:
    --------------------------
    - imagenet_scaling: Normalize the image using ImageNet mean and standard deviation.
    
    Available Resizing Operations:
    --------------------------
    - pad: Pad images and labels to the nearest multiple of a specified value.
    """

    def __init__(self, augmentations=None, scaling=None, resizing=None, image_size: int= 512, pad_multiple=14, 
                 super_resolution_p=0.3, **kwargs):
        """
        Initialize the Transformations class.

        Parameters:
        -----------
        augmentations : list of str
            The names of the augmentations to apply (e.g., ["crop", "spatial", "color", "noise"]).
        scaling : str or None
            The type of scaling to apply (e.g., "imagenet_scaling").
        resizing : str or None
            The type of resizing to apply (e.g., "pad").
        image_size : int or None
            The target size for cropping or resizing.
        pad_multiple : int, optional
            The multiple to pad to when using the "pad" resizing.
        super_resolution_p : float, optional
            Probability of applying super-resolution transform (default: 0.5).
        """
        self.image_size = image_size
        self.augmentations = augmentations or []
        self.scaling = scaling
        self.resizing = resizing
        self.pad_multiple = pad_multiple
        self.super_resolution_p = super_resolution_p
        
        self.augmentation_map = {
            "resize": self.resize_image,
            "crop": self.crop_transform,
            "spatial": self.spatial_transforms,
            "color": self.color_transforms,
            "noise": self.noise_transforms,
            "super_resolution": self.super_resolution_transform,
        }

        self.bbox_augmentation_map = {
            "resize": self.bbox_resize_transform,
            "crop": self.bbox_crop_transform,
            "spatial": self.bbox_spatial_transforms,
            "color": self.bbox_color_transforms,
            "noise": self.bbox_noise_transforms,
        }
        
        self.scaling_map = {
            "imagenet_scaling": self.imagenet_scaling,
        }
        
        self.resizing_map = {
            "pad": self.pad_to_nearest_multiple_of_n,
            "resize_image": self.resize_image
        }

    def apply(self, image, labels):
        """
        Apply all transformations (augmentations, scaling, and resizing) to the image and labels.
        Only applies transformations if the corresponding parameters were provided.

        Parameters:
        -----------
        image : numpy.ndarray
            The input image.
        labels : numpy.ndarray or list of numpy.ndarray
            The corresponding label(s)/mask(s). Can be a single mask or a list of masks
            (e.g., [segmentation_mask, centroid_heatmap]).

        Returns:
        --------
        tuple:
            Transformed image and label(s). Returns same format as input (single or list).
        """
        # Apply augmentations only if they were provided
        if self.augmentations:
            image, labels = self.apply_augmentation(image, labels)
        
        # Apply scaling only if it was provided
        if self.scaling:
            image, labels = self.apply_scaling(image, labels)
        
        # Apply resizing only if it was provided
        if self.resizing:
            image, labels = self.apply_resizing(image, labels)
            
        return image, labels

    def apply_augmentation(self, image, labels):
        """
        Apply only the specified augmentations to the image and labels.

        Parameters:
        -----------
        image : numpy.ndarray
            The input image.
        labels : numpy.ndarray or list of numpy.ndarray
            The corresponding label(s)/mask(s).

        Returns:
        --------
        tuple:
            Augmented image and labels.
        """
        for aug_name in self.augmentations:
            if aug_name in self.augmentation_map:
                image, labels = self.augmentation_map[aug_name](image, labels)
            else:
                raise ValueError(f"Unknown augmentation: {aug_name}")
        return image, labels

    def apply_scaling(self, image, labels):
        """
        Apply the specified scaling to the image and labels.

        Parameters:
        -----------
        image : numpy.ndarray
            The input image.
        labels : numpy.ndarray or list of numpy.ndarray
            The corresponding label(s)/mask(s).

        Returns:
        --------
        tuple:
            Scaled image and labels.
        """
        if self.scaling is None:
            return image, labels
            
        if self.scaling in self.scaling_map:
            image, labels = self.scaling_map[self.scaling](image, labels)
        else:
            raise ValueError(f"Unknown scaling type: {self.scaling}")
        return image, labels
        
    def apply_resizing(self, image, labels):
        """
        Apply the specified resizing to the image and labels.

        Parameters:
        -----------
        image : numpy.ndarray
            The input image.
        labels : numpy.ndarray or list of numpy.ndarray
            The corresponding label(s)/mask(s).

        Returns:
        --------
        tuple:
            Resized image and labels.
        """
        if self.resizing is None:
            return image, labels
            
        if self.resizing in self.resizing_map:
            if self.resizing == "pad":
                image, labels = self.resizing_map[self.resizing](image, labels, self.pad_multiple)
            else:
                image, labels = self.resizing_map[self.resizing](image, labels)
        else:
            raise ValueError(f"Unknown resizing type: {self.resizing}")
        return image, labels

    def _ensure_channel_last(self, image):
        """
        Ensure that the image is in channel-last format (HWC) for albumentations.
        
        Parameters:
        -----------
        image : numpy.ndarray
            The input image, either in CHW or HWC format
            
        Returns:
        --------
        tuple:
            (converted_image, is_channel_first) where is_channel_first indicates
            whether the original image was in channel-first format
        """
        is_channel_first = False
        
        # Check if image is in channel-first format (CHW)
        if image.ndim == 3 and (image.shape[0] == 1 or image.shape[0] == 3):
            image = np.transpose(image, (1, 2, 0))
            is_channel_first = True
            
        return image, is_channel_first

    def _restore_format(self, image, is_channel_first):
        """
        Restore the image format if it was originally channel-first.
        
        Parameters:
        -----------
        image : numpy.ndarray
            The transformed image in channel-last format (HWC)
        is_channel_first : bool
            Whether the original image was in channel-first format
            
        Returns:
        --------
        numpy.ndarray:
            Image in the original format
        """
        if is_channel_first:
            image = np.transpose(image, (2, 0, 1))
        return image

    @ensure_channel_last_decorator
    def crop_transform(self, image, labels):
        """
        Apply random cropping to the image and labels.
        """
        transform = A.Compose([ # type: ignore
            A.RandomScale(scale_limit=(0.0, 0.3), p = 0.5),
            A.RandomCrop(width=self.image_size, height=self.image_size, p=1.0)
        ])
        transformed = transform(image=image, masks=labels)
        return transformed['image'], transformed['masks']

    @ensure_channel_last_decorator
    def spatial_transforms(self, image, labels):
        """
        Apply spatial transformations (scaling, flipping, rotating) to the image and labels.
        Ensures output is always (image_size, image_size).
        """
        transform = A.Compose([ # type: ignore
            A.RandomScale((-0.2, 0.2), p=0.5),
            A.OneOf([
                A.HorizontalFlip(p=0.8),
                A.VerticalFlip(p=0.8),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.2, 0.2),
                                   rotate_limit=(-90, 90), border_mode=cv2.BORDER_CONSTANT, value=0)
            ], p=0.8),
            A.Resize(self.image_size, self.image_size)  # Ensures output size
        ], p=1.0)
        transformed = transform(image=image, masks=labels)
        return transformed['image'], transformed['masks']

    @ensure_channel_last_decorator
    def color_transforms(self, image, labels):
        """
        Apply color transformations (sharpening, brightness, gamma, etc.) to the image.
        These transforms only affect the color properties without adding noise.
        """
        transform = A.Compose([
            A.OneOf([
                A.Sharpen(),
                A.RandomGamma(),
                A.RandomBrightnessContrast(0.1, 0.2),
                A.Emboss()
            ], p=0.5)
        ], p=1.0)
        transformed = transform(image=image, masks=labels)
        return transformed['image'], transformed['masks']
        
    @ensure_channel_last_decorator
    def noise_transforms(self, image, labels):
        """
        Apply noise transformations (blur, noise, downscaling, etc.) to the image.
        These transforms introduce various forms of noise and degradation.
        """
        transform = A.Compose([
            A.OneOf([
                A.Downscale(scale_max=0.5),
                A.Blur(blur_limit=3),
                A.MultiplicativeNoise(multiplier=(0.8, 1.1)),
                A.GaussNoise(var_limit=0.003),
            ], p=0.5)
        ], p=1.0)
        transformed = transform(image=image, masks=labels)
        return transformed['image'], transformed['masks']

    @ensure_channel_last_decorator
    def super_resolution_transform(self, image, labels):
        """
        Apply super-resolution style degradation to the image only.
        
        Downsamples the image to 0.75x of its size, then upsamples back to original.
        This forces the model to learn high-resolution segmentation from lower-resolution input.
        Labels remain at full resolution and are not modified.
        
        The probability of applying this transform is controlled by self.super_resolution_p.
        
        Parameters:
        -----------
        image : numpy.ndarray
            The input image (H, W, C)
        labels : list of numpy.ndarray
            The corresponding labels/masks (passed through unchanged)
        
        Returns:
        --------
        tuple:
            Degraded image (downsampled then upsampled) and unchanged labels
        """
        h, w = image.shape[:2]
        
        # Calculate downsampled size (0.75x)
        down_h = int(h * 0.75)
        down_w = int(w * 0.75)
        
        # Use Sequential to apply downsample -> upsample with probability p
        transform = A.Compose([
            A.Sequential([
                A.Resize(height=down_h, width=down_w, interpolation=cv2.INTER_LINEAR),
                A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR)
            ], p=self.super_resolution_p)
        ])
        
        # Apply transform only to image (additional_targets not needed, masks ignored)
        transformed = transform(image=image)
        
        # Return transformed image with original high-resolution labels
        return transformed['image'], labels

    @ensure_channel_last_decorator
    def imagenet_scaling(self, image, labels):
        """
        Normalize the image using ImageNet mean and std with albumentations.

        Parameters:
        -----------
        image : numpy.ndarray
            The input image.
        labels : list of numpy.ndarray
            The corresponding labels/masks (passed through unchanged).

        Returns:
        --------
        tuple:
            Normalized image and unchanged labels.
        """
        transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0)
        ])
        transformed = transform(image=image)
        return transformed['image'], labels

    def pad_to_nearest_multiple_of_n(self, images, labels, n=None):
        """
        Pad images and labels to the nearest multiple of `n` using numpy functions.
        Assumes images are always channel-first (C, H, W).
        Handles both single label and multiple labels.
        """
        if n is None:
            n = self.pad_multiple

        # Check for channel-first format
        if images.ndim != 3 or images.shape[0] not in (1, 3):
            raise ValueError("Input image must be channel-first (C, H, W) with 1 or 3 channels.")

        c, h, w = images.shape
        new_h = ((h + n - 1) // n) * n
        new_w = ((w + n - 1) // n) * n
        pad_h = new_h - h
        pad_w = new_w - w
        pad_top = 0
        pad_bottom = pad_h
        pad_left = 0
        pad_right = pad_w

        images_padded = np.pad(images, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

        # Handle both single label and multiple labels
        is_single_label = not isinstance(labels, (list, tuple))
        if is_single_label:
            labels = [labels]  # Convert single label to list

        labels_padded_list = []
        for label in labels:
            # Pad each label (assume (H, W) or (1, H, W))
            if label.ndim == 3 and label.shape[0] == 1:
                label_channel_first = True
                label = label[0]
            else:
                label_channel_first = False

            label_padded = np.pad(label, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

            if label_channel_first:
                label_padded = label_padded[np.newaxis, ...]
            
            labels_padded_list.append(label_padded)
        
        # Return single label if input was single, otherwise return list
        if is_single_label:
            labels_padded = labels_padded_list[0]
        else:
            labels_padded = labels_padded_list

        return images_padded, labels_padded

    # ------------------------------------------------------------------
    #  Bbox-aware transforms for object detection
    # ------------------------------------------------------------------

    def _bbox_params(self, min_visibility=0.3):
        return A.BboxParams(
            format="yolo",
            min_visibility=min_visibility,
            label_fields=["class_labels"],
            clip=True,
            filter_invalid_bboxes=True,
        )

    def apply_bbox(self, image, bboxes, class_labels):
        """Apply configured augmentations to image + bounding boxes.

        Parameters
        ----------
        image : np.ndarray (H, W, C) uint8 or float
        bboxes : list of [cx, cy, w, h]  normalised YOLO format
        class_labels : list of int

        Returns
        -------
        image, bboxes, class_labels  (transformed)
        """
        for aug_name in self.augmentations:
            if aug_name in self.bbox_augmentation_map:
                image, bboxes, class_labels = self.bbox_augmentation_map[aug_name](
                    image, bboxes, class_labels
                )
            else:
                raise ValueError(f"Unknown bbox augmentation: {aug_name}")
        return image, bboxes, class_labels

    def bbox_resize_transform(self, image, bboxes, class_labels):
        """Resize image to image_size x image_size. Normalised YOLO boxes are unchanged."""
        image = A.Resize(self.image_size, self.image_size)(image=image)["image"]
        return image, bboxes, class_labels

    def bbox_crop_transform(self, image, bboxes, class_labels):
        """AtLeastOneBboxRandomCrop — guarantees at least one box in the crop.
        Pads small images first, then crops and resizes to image_size."""
        transform = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=self.image_size, min_width=self.image_size,
                    border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0,
                ),
                A.AtLeastOneBBoxRandomCrop(
                    height=self.image_size, width=self.image_size, p=1.0
                ),
            ],
            bbox_params=self._bbox_params(),
        )
        out = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        return out["image"], out["bboxes"], out["class_labels"]

    def bbox_spatial_transforms(self, image, bboxes, class_labels):
        """Flips, rotation, shift-scale-rotate with bbox awareness."""
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=(-0.1, 0.1),
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                A.Resize(self.image_size, self.image_size),
            ],
            bbox_params=self._bbox_params(),
        )
        out = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        return out["image"], out["bboxes"], out["class_labels"]

    def bbox_color_transforms(self, image, bboxes, class_labels):
        """Color augmentations — boxes pass through unchanged."""
        transform = A.Compose(
            [
                A.OneOf(
                    [
                        A.Sharpen(),
                        A.RandomGamma(),
                        A.RandomBrightnessContrast(0.1, 0.2),
                        A.Emboss(),
                    ],
                    p=0.5,
                )
            ],
            bbox_params=self._bbox_params(),
        )
        out = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        return out["image"], out["bboxes"], out["class_labels"]

    def bbox_noise_transforms(self, image, bboxes, class_labels):
        """Noise augmentations — boxes pass through unchanged."""
        transform = A.Compose(
            [
                A.OneOf(
                    [
                        A.Downscale(scale_max=0.5),
                        A.Blur(blur_limit=3),
                        A.MultiplicativeNoise(multiplier=(0.8, 1.1)),
                        A.GaussNoise(var_limit=0.003),
                    ],
                    p=0.5,
                )
            ],
            bbox_params=self._bbox_params(),
        )
        out = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        return out["image"], out["bboxes"], out["class_labels"]

    # ------------------------------------------------------------------
    #  Segmentation helpers (original)
    # ------------------------------------------------------------------

    @ensure_channel_last_decorator
    def resize_image(self, image, labels, image_size=None):
        """
        Resize image and labels to the specified size using albumentations.
        Args:
            image: numpy array, shape [H, W, C] (channel-last)
            labels: numpy array or list of numpy arrays, shape [H, W] or [1, H, W]
            image_size: int (for square resize)
        Returns:
            Resized image and labels
        """
        if image_size is None:
            image_size = self.image_size
        if image_size is None:
            return image, labels
        transform = A.Compose([
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR)
        ])
        transformed = transform(image=image, masks=labels)
        return transformed['image'], transformed['masks']