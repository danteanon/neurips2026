import numpy as np

class Normalization:
    """
    A class to normalize images using different strategies.

    Factory Method:
    ---------------
    Use `Normalization(method="90p")`, `Normalization(method="min_max")`, `Normalization(method="8bit")`, or `Normalization(method="auto")` 
    to create an instance with the desired normalization strategy.

    Methods:
    --------
    apply(img, channels=3):
        Apply the selected normalization method to the input image.
    """

    def __init__(self, method="90p"):
        """
        Initialize the Normalization class with a specific method.

        Parameters:
        -----------
        method : str
            The normalization method to use. Options are:
            - "90p": Normalize using the 2nd and 98th percentiles.
            - "min_max": Normalize using min-max scaling.
            - "8bit": Normalize by dividing by 255 (for 8-bit images).
            - "auto": Auto-detect if image is 8-bit and apply appropriate normalization.
        """
        if method == "90p":
            self.method = self._normalize_img_90p
            self.__doc__ = self._normalize_img_90p.__doc__
        elif method == "99p_pre_computed":
            self.method = self._normalize_img_99p_pre_computed
            self.__doc__ = self._normalize_img_99p_pre_computed.__doc__
        elif method == "min_max":
            self.method = self._normalize_min_max
            self.__doc__ = self._normalize_min_max.__doc__
        elif method == "8bit":
            self.method = self._normalize_8bit
            self.__doc__ = self._normalize_8bit.__doc__
        elif method == "auto":
            self.method = self._normalize_auto
            self.__doc__ = self._normalize_auto.__doc__
        elif method == "none":
            self.method = self._no_normalisation
            self.__doc__ = self._no_normalisation.__doc__
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def apply(self, img, channels=3, p_2=None, p_99=None):
        """
        Apply the selected normalization method to the input image.

        Parameters:
        -----------
        img : numpy.ndarray
            The input image to normalize.
        channels : int
            The number of channels in the image.

        Returns:
        --------
        numpy.ndarray
            The normalized image.
        """
        # Check if this is the pre-computed percentile method
        if hasattr(self.method, '__name__') and self.method.__name__ == '_normalize_img_99p_pre_computed':
            if p_2 is None or p_99 is None:
                raise ValueError("p_2 and p_99 must be provided for '99p_pre_computed' method")
            return self.method(img, np.array(p_2), np.array(p_99), channels)
        else:
            # All other methods use standard signature
            return self.method(img, channels)
    @staticmethod
    def _normalize_img_90p(img, channels=3, per_channel=False):
        """
        Normalize the image using the 2nd and 98th percentiles.

        Parameters:
        -----------
        img : numpy.ndarray
            The input image to normalize.
        channels : int
            The number of channels in the image.

        Returns:
        --------
        numpy.ndarray
            The normalized image.
        """
        mask = np.any(img > 0, axis=0)
        if mask.sum() == 0:
            return img
        if per_channel:
            p_2 = np.percentile(img[:, mask], 2, axis=-1)[0:channels, np.newaxis, np.newaxis]
            p_90 = np.percentile(img[:, mask], 98, axis=-1)[0:channels, np.newaxis, np.newaxis]
        else:
            p_2 = np.max(np.percentile(img[0:channels, mask], 2, axis=-1))
            p_90 = np.max(np.percentile(img[0:channels, mask], 98, axis=-1))
        if np.all(p_90 - p_2) > 0:
            image = (np.float32(img)) / (p_90)
        else:
            image = np.zeros_like(img)
        image = np.clip(image, 0.0, 1.0)
        return image.astype(np.float32)
    
    @staticmethod
    def _normalize_img_99p_pre_computed(img, p_2, p_99, channels=3):
        """
        Normalize the image using the 2nd and 99th percentiles pre-computed. Use with lmdb dataset

        Parameters:
        -----------
        img : numpy.ndarray
            The input image to normalize.
        channels : int
            The number of channels in the image.

        Returns:
        --------
        numpy.ndarray
            The normalized image.
        """
        
        if np.any(p_99-p_2) < 1e-6:
            return img[0:channels]
        
        if np.all(p_99 - p_2) > 1e-6:
            image = (np.float32(img[0:channels]) - p_2[:channels, np.newaxis, np.newaxis]) / (p_99[:channels, np.newaxis, np.newaxis] - p_2[:channels, np.newaxis, np.newaxis])
        else:
            image = np.zeros_like(img[0:channels])
        image = np.clip(image, 0.0, 1.0)
        return image.astype(np.float32)


    @staticmethod
    def _normalize_min_max(img, channels=3):
        """
        Normalize the image using min-max scaling.

        Parameters:
        -----------
        img : numpy.ndarray
            The input image to normalize.
        channels : int
            The number of channels in the image.

        Returns:
        --------
        numpy.ndarray
            The normalized image.
        """
        mask = np.any(img > 0, axis=0)
        if mask.sum() == 0:
            return img
        p_min = np.min(img[:, mask])
        p_max = np.max(img[:, mask])
        if (p_max - p_min) > 0:
            image = (np.float32(img) - p_min) / (p_max - p_min)
        else:
            image = np.zeros_like(img)
        image = np.clip(image, 0.0, 1.0)
        return image.astype(np.float32)

    @staticmethod
    def _normalize_8bit(img, channels=3):
        """
        Normalize the image by dividing by 255 (for 8-bit images).

        Parameters:
        -----------
        img : numpy.ndarray
            The input image to normalize.
        channels : int
            The number of channels in the image.

        Returns:
        --------
        numpy.ndarray
            The normalized image.
        """
        return (np.float32(img) / 255.0).astype(np.float32)
    
    @staticmethod
    def _no_normalisation(img, channels=3):
        """
        Do not normalize the image.
        """
        return img

    @staticmethod
    def _normalize_auto(img, channels=3):
        """
        Auto-detect if image is 8-bit and apply appropriate normalization.
        If the image appears to have 8-bit values (0-255 range), apply 8-bit normalization.
        Otherwise, apply 90p normalization.

        Parameters:
        -----------
        img : numpy.ndarray
            The input image to normalize.
        channels : int
            The number of channels in the image.

        Returns:
        --------
        numpy.ndarray
            The normalized image.
        """
        # Check if image appears to be 8-bit (values between 0-255)
        if img.dtype == np.uint8:
            return Normalization._normalize_8bit(img, channels)
        else:
            return Normalization._normalize_img_90p(img, channels)