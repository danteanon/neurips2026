#!/usr/bin/env python3
"""
tiling_classes.py

Object-oriented refactoring of tiling_utils.py functionality.
Provides class-based interfaces for all tiling operations with clear inheritance structure.

Classes:
- BaseTiler: Base class for all tiling operations
- BaseImageLabelTiler: Base class for image-label tiling
- BaseLMDBTiler: Mixin for LMDB functionality
- GeospatialTiler: Geospatial image tiling
- NonGeospatialTiler: Non-geospatial image tiling
- Various specialized tilers with LMDB support
"""

# Standard library imports
import glob
import json
import lmdb
import math
import multiprocessing
import os
import pathlib
import shutil
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from typing import List, Tuple, Optional, Union, Dict, Any

# Third-party imports
import geopandas as gpd
import h3
import numpy as np
import rasterio
import shapely
from geopy.distance import geodesic
from PIL import Image
from pyproj import Transformer
from rasterio import features
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform
from skimage.morphology import erosion, disk
from tqdm import tqdm
from shapely.geometry import Polygon, box

# =============================================================================
# Utility Classes
# =============================================================================

class TilingUtils:
    """Static utility methods for tiling operations"""
    
    @staticmethod
    def calculate_resolution(raster_path: str) -> Optional[float]:
        """
        Calculate the ground resolution of a raster in meters per pixel.
        
        Parameters
        ----------
        raster_path : str
            Path to the raster file
            
        Returns
        -------
        float or None
            Resolution in meters per pixel, or None if calculation fails
        """
        try:
            with rasterio.open(raster_path) as raster:
                bounds = raster.bounds
                width = raster.width
                
                # Convert bounds to EPSG:4326
                transformer = Transformer.from_crs(raster.crs, "EPSG:4326", always_xy=True)
                min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
                max_lon, max_lat = transformer.transform(bounds.right, bounds.top)
                
                # Calculate distance in meters
                distance = geodesic((min_lat, min_lon), (min_lat, max_lon)).meters
                
                # Calculate resolution
                resolution = distance / width
                return resolution
        except Exception as e:
            print(f"Could not calculate resolution: {e}")
            return None

    @staticmethod
    def get_h3_index_for_raster(raster_path: str, resolution: int = 7) -> str:
        """
        Get H3 index of the centroid of a raster at specified resolution.
        
        Parameters
        ----------
        raster_path : str
            Path to the raster file
        resolution : int, optional
            H3 resolution (default: 7)
            
        Returns
        -------
        str
            H3 index string
        """
        with rasterio.open(raster_path) as src:
            # Get raster bounds
            bounds = src.bounds
            
            # Calculate centroid in raster CRS
            centroid_x = (bounds.left + bounds.right) / 2
            centroid_y = (bounds.bottom + bounds.top) / 2
            
            # Convert to lat/lon if not already
            if src.crs and src.crs.to_epsg() != 4326:
                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                lon, lat = transformer.transform(centroid_x, centroid_y)
            else:
                lon, lat = centroid_x, centroid_y
            
            # Get H3 index
            h3_index = h3.latlng_to_cell(lat, lon, resolution)
            return h3_index

    @staticmethod
    def calculate_label_statistics(label_array: np.ndarray) -> Dict[int, int]:
        """
        Calculate pixel area statistics for each label value in the array.
        
        Parameters
        ----------
        label_array : numpy.ndarray
            2D label array
            
        Returns
        -------
        dict
            Dictionary with label values as keys and pixel counts as values
        """
        unique_labels, counts = np.unique(label_array.flatten(), return_counts=True)
        return {int(label): int(count) for label, count in zip(unique_labels, counts)}

    @staticmethod
    def find_image_files(directory: str, extensions: Optional[List[str]] = None, recursive: bool = True) -> List[str]:
        """
        Find image files in directory with specified extensions.
        
        Parameters
        ----------
        directory : str
            Directory to search
        extensions : list, optional
            List of file extensions to search for (default: ['.tif', '.tiff'])
        recursive : bool, optional
            Whether to search recursively (default: False)
            
        Returns
        -------
        list
            List of image file paths
        """
        if extensions is None:
            extensions = ['.tif', '.tiff']
        
        directory = pathlib.Path(directory)
        files = []

        case_variations = []
        for ext in extensions:
            case_variations.extend([
                ext.lower(),
                ext.upper(), 
                ext.capitalize()
            ])
        
        # Remove duplicates
        case_variations = list(set(case_variations))
        
        for ext in case_variations:
            if recursive:
                files.extend(directory.rglob(f"*{ext}"))
            else:
                files.extend(directory.glob(f"*{ext}"))
        
        return [str(f) for f in files]

    @staticmethod
    def calculate_image_percentiles(image_data: np.ndarray, percentiles: List[float]) -> Dict[str, List[float]]:
        """
        Calculate percentiles for image data with masking.
        
        Parameters
        ----------
        image_data : numpy.ndarray
            Image data array
        percentiles : list
            List of percentiles to calculate
            
        Returns
        -------
        dict
            Dictionary mapping percentile values
        """
        # Use masking for normalization
        mask = np.any(image_data > 0, axis=0)
        if mask.sum() == 0:
            # If all pixels are zero, use default approach
            percentile_values = np.percentile(image_data, percentiles, axis=(1, 2))
        else:
            # Apply mask to exclude zero pixels
            percentile_values = np.percentile(image_data[:, mask], percentiles, axis=-1)
        
        # Create a dictionary to map percentile values
        return {str(p): percentile_values[i].tolist() for i, p in enumerate(percentiles)}

    @staticmethod
    def calculate_tile_windows(width: int, height: int, tile_size: int, overlap: float = 0, min_coverage: float = 0.2) -> List[Tuple[int, int, int, int]]:
        """
        Calculate tile windows with overlap handling.
        
        Parameters
        ----------
        width : int
            Width of the source image/raster
        height : int
            Height of the source image/raster
        tile_size : int
            Size of the square tiles
        overlap : float, optional
            Fraction of overlap between tiles (default: 0)
        min_coverage : float, optional
            Minimum coverage fraction for edge tiles (default: 0.2)
            
        Returns
        -------
        list
            List of (x, y, actual_width, actual_height) tuples for valid tiles
        """
        step_size = int(tile_size * (1 - overlap))
        windows = []
        
        for y in range(0, height, step_size):
            for x in range(0, width, step_size):
                # Calculate actual tile dimensions
                actual_width = min(tile_size, width - x)
                actual_height = min(tile_size, height - y)
                
                # Check minimum coverage
                width_coverage = actual_width / tile_size
                height_coverage = actual_height / tile_size
                
                if width_coverage >= min_coverage and height_coverage >= min_coverage:
                    windows.append((x, y, tile_size, tile_size))
        
        return windows

    @staticmethod
    def create_label_mask_from_shapefile(label_gdf: gpd.GeoDataFrame, tile_bounds: Tuple[float, float, float, float], 
                                       tile_height: int, tile_width: int) -> np.ndarray:
        """
        Create label mask by rasterizing shapefile for given tile bounds.
        
        Parameters
        ----------
        label_gdf : geopandas.GeoDataFrame
            Shapefile data
        tile_bounds : tuple
            Tile bounds as (left, bottom, right, top)
        tile_height : int
            Height of the tile in pixels
        tile_width : int
            Width of the tile in pixels
            
        Returns
        -------
        numpy.ndarray
            Label mask array
        """
        left, bottom, right, top = tile_bounds
        
        # Create a box geometry for the tile
        tile_box = shapely.geometry.box(left, bottom, right, top)
        
        # Find labels that intersect with this tile
        intersecting_labels = label_gdf[label_gdf.intersects(tile_box)]
        
        # Create label mask by rasterizing the shapefile
        if len(intersecting_labels) > 0:
            # Create transform from bounds
            tile_transform = rasterio.transform.from_bounds(left, bottom, right, top, tile_width, tile_height)
            
            shapes = [(geom, 1) for geom in intersecting_labels.geometry]
            label_mask = rasterio.features.rasterize(
                shapes=shapes,
                out_shape=(tile_height, tile_width),
                transform=tile_transform,
                fill=0,
                dtype='uint8'
            )
        else:
            label_mask = np.zeros((tile_height, tile_width), dtype=np.uint8)
        
        return label_mask

    @staticmethod
    def process_instance_label_with_erosion(label_array: np.ndarray, erosion_kernel_size: int) -> np.ndarray:
        """
        Process instance label array with erosion to separate touching instances.
        
        Parameters
        ----------
        label_array : numpy.ndarray
            Label array (single channel with values 1-255 or RGB with unique instance colors)
        erosion_kernel_size : int
            Size of the erosion kernel
        
        Returns
        -------
        numpy.ndarray
            Binary mask with separated instances (0 for background, 1 for instances)
        """
        # Handle different label formats
        if len(label_array.shape) == 3:  # RGB format
            # Each unique RGB combination represents a different instance
            h, w, c = label_array.shape
            
            # Create a single channel representation for unique identification
            if c == 3:  # RGB
                single_channel = (label_array[:, :, 0].astype(np.uint32) << 16) + \
                               (label_array[:, :, 1].astype(np.uint32) << 8) + \
                               label_array[:, :, 2].astype(np.uint32)
            else:  # RGBA or other
                single_channel = label_array[:, :, 0]  # Use first channel
            
            # Get unique instance values (excluding background - typically 0 or black)
            background_value = 0  # Assume black (0,0,0) is background
            unique_instances = np.unique(single_channel)
            unique_instances = unique_instances[unique_instances != background_value]
            
        else:  # Single channel format
            # Values 1-255 are instances, 0 is background
            single_channel = label_array
            background_value = 0
            unique_instances = np.unique(single_channel)
            unique_instances = unique_instances[unique_instances != background_value]
        
        # Initialize final binary mask
        final_mask = np.zeros(single_channel.shape, dtype=np.uint8)
        
        # Apply erosion to each instance separately if erosion is requested
        if erosion_kernel_size > 0:
            # Create erosion kernel
            kernel = disk(erosion_kernel_size)
            
            # Process each instance individually
            for instance_value in unique_instances:
                # Create binary mask for this specific instance
                instance_mask = (single_channel == instance_value).astype(np.uint8)
                
                # Apply erosion to this instance
                eroded_instance = erosion(instance_mask, kernel)
                
                # Add the eroded instance to the final mask
                final_mask = np.logical_or(final_mask, eroded_instance).astype(np.uint8)
        else:
            # No erosion - just create binary mask for all instances
            final_mask = (single_channel != background_value).astype(np.uint8)
        
        return final_mask

# =============================================================================
# Base Classes
# =============================================================================

class BaseTiler:
    """Base class for all tiling operations"""
    
    def __init__(self, tile_size: int = 512, overlap: float = 0, processes: Optional[int] = None, 
                 output_format: str = 'files', min_data_percentage: float = 0.05, min_coverage: float = 0.0,
                 channels: int = 3):
        """
        Initialize base tiler.
        
        Parameters
        ----------
        tile_size : int, optional
            Size of square tiles (default: 512)
        overlap : float, optional
            Fraction of overlap between tiles (default: 0)
        processes : int, optional
            Number of processes for multiprocessing (default: CPU count)
        output_format : str, optional
            Output format: 'files' or 'lmdb' (default: 'files')
        min_data_percentage : float, optional
            Minimum percentage of non-zero data to keep a tile (default: 0.05)
        min_coverage : float, optional
            Minimum coverage fraction for edge tiles (default: 0.2)
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.processes = processes or multiprocessing.cpu_count()
        self.output_format = output_format
        self.min_data_percentage = min_data_percentage
        self.min_coverage = min_coverage
        self.channels = channels

    def process_single(self, **kwargs) -> Union[List[str], int]:
        """Process a single item - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement process_single")
    
    def process_directory(self, **kwargs) -> Union[List[str], int]:
        """Process a directory - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement process_directory")
    
    def _setup_output_dirs(self, output_dir: str, create_img_label_dirs: bool = False) -> Union[str, Tuple[str, str]]:
        """Setup output directories"""
        os.makedirs(output_dir, exist_ok=True)
        
        if create_img_label_dirs:
            img_output_dir = os.path.join(output_dir, "img")
            label_output_dir = os.path.join(output_dir, "label")
            os.makedirs(img_output_dir, exist_ok=True)
            os.makedirs(label_output_dir, exist_ok=True)
            return img_output_dir, label_output_dir
        
        return output_dir


class BaseImageLabelTiler(BaseTiler):
    """Base class for image-label tiling operations"""
    
    def __init__(self, binary: bool = False, **kwargs):
        """
        Initialize image-label tiler.
        
        Parameters
        ----------
        binary : bool, optional
            Convert labels to binary masks (default: False)
        **kwargs
            Additional arguments passed to BaseTiler
        """
        super().__init__(**kwargs)
        self.binary = binary
    
    def process_single(self, **kwargs) -> Union[Tuple[List[str], List[str]], int]:
        """Process a single image-label pair - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement process_single")
    
    def process_directory(self, **kwargs) -> Union[Tuple[List[str], List[str]], int]:
        """Process a directory of image-label pairs - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement process_directory")


class BaseLMDBTiler:
    """Mixin class for LMDB functionality"""
    
    def __init__(self, client_name: str = "default", percentiles: List[float] = None):
        """
        Initialize LMDB tiler.
        
        Parameters
        ----------
        client_name : str, optional
            Client name for LMDB organization (default: "default")
        percentiles : list, optional
            Percentiles to calculate (default: [1, 2, 25, 50, 75, 98, 99])
        """
        if client_name is None:
            client_name = "client"
        self.client_name = client_name
        self.percentiles = percentiles or [1, 2, 25, 50, 75, 98, 99]
        
    def _setup_lmdb_environment(self, output_dir: str) -> Tuple[Any, List[str], str, str, pathlib.Path]:
        """Setup LMDB environment"""
        # Convert paths to pathlib.Path objects
        output_dir = pathlib.Path(output_dir)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create LMDB environment
        env = lmdb.open(str(output_dir/"pixels.lmdb"), map_size=2**44)
        
        # Create a key index to keep track of all tiles
        key_index = []
        current_date = datetime.now().strftime("%Y/%m/%d")
        
        return env, key_index, current_date, self.client_name, output_dir
        
    def _store_tile_in_lmdb(self, env, tile_key: str, image_tile: np.ndarray, 
                           label_data: np.ndarray, metadata: Dict[str, Any], band_count: int):
        """Store tile data and metadata in LMDB"""
        with env.begin(write=True) as tx:
            # Store each band separately
            for band_idx in range(band_count):
                band_data = image_tile[band_idx]
                tx.put(f"{tile_key}:img:{band_idx+1}".encode(), band_data.tobytes())
            
            # Store label data
            tx.put(f"{tile_key}:lbl".encode(), label_data.tobytes())
            
            # Store metadata
            tx.put(f"{tile_key}:meta".encode(), json.dumps(metadata).encode())
    
    def _finalize_lmdb(self, env, key_index: List[str]):
        """Finalize LMDB by storing key index and closing environment"""
        # Store the key index for efficient access
        with env.begin(write=True) as tx:
            tx.put("key_index".encode(), json.dumps(key_index).encode())
        
        env.close()
        
        print(f"Tiling complete for client {self.client_name} with {len(key_index)} tiles")
    
    def _create_tile_metadata(self, image_name: str, dtype: str, h3_index: str, percentile_dict: Dict[str, List[float]], 
                             row: int, col: int, window: Union[Window, Dict[str, int]], band_count: int, 
                             image_tile_shape: Tuple[int, ...], label_stats: Dict[int, int], source_id: str, 
                             bounds: Optional[Dict[str, float]], crs: Optional[Dict[str, Any]], 
                             resolution: Optional[float], current_date: str, label_band_count: int, 
                             label_height: int, label_width: int, **kwargs) -> Dict[str, Any]:
        """Create standardized metadata dictionary for tiles"""
        metadata = {
            "src": image_name,
            "dtype": str(dtype),
            "label_dtype": "uint8",
            "client": self.client_name,
            "h3_index": h3_index,
            "percentiles": percentile_dict,
            "row": row,
            "col": col,
            "window": {
                "col_off": int(window.col_off) if hasattr(window, 'col_off') else window.get('col_off', 0),
                "row_off": int(window.row_off) if hasattr(window, 'row_off') else window.get('row_off', 0),
                "width": int(window.width) if hasattr(window, 'width') else window.get('width', label_width),
                "height": int(window.height) if hasattr(window, 'height') else window.get('height', label_height)
            },
            "image": {
                "count": band_count,
                "width": image_tile_shape[2] if len(image_tile_shape) > 2 else image_tile_shape[1],
                "height": image_tile_shape[1] if len(image_tile_shape) > 2 else image_tile_shape[0]
            },
            "label": {
                "count": label_band_count,
                "width": label_width,
                "height": label_height,
                "statistics": label_stats
            },
            "source_id": source_id,
            "bounds": bounds,
            "crs": crs,
            "resolution": resolution,
            "date_created": current_date
        }
        
        # Add any additional metadata from kwargs
        metadata.update(kwargs)
        
        return metadata 

# =============================================================================
# Geospatial Tiling Classes
# =============================================================================

class GeospatialTiler(BaseTiler):
    """Tiler for geospatial images (GeoTIFF, etc.)"""
    
    def process_single(self, image_path: str, output_dir: str, prefix: str = "tile", **kwargs) -> List[str]:
        """
        Process a single geospatial image.
        
        Parameters
        ----------
        image_path : str
            Path to the geospatial image
        output_dir : str
            Output directory
        prefix : str, optional
            Prefix for tile filenames (default: "tile")
            
        Returns
        -------
        list
            List of tile file paths
        """
        output_dir = self._setup_output_dirs(output_dir)
        
        with rasterio.open(image_path) as src:
            width, height = src.width, src.height
            meta = src.meta.copy()
            
            # Calculate tile windows
            windows = TilingUtils.calculate_tile_windows(width, height, self.tile_size, self.overlap, self.min_coverage)
            
            # Generate tasks
            tasks = []
            for i, (x, y, actual_width, actual_height) in enumerate(windows):
                tasks.append((
                    image_path, output_dir, self.tile_size, x, y, 
                    actual_width, actual_height, meta, prefix, self.min_data_percentage, i
                ))
        
        # Process tiles in parallel
        tile_paths = []
        if tasks:
            with ProcessPoolExecutor(max_workers=self.processes) as executor:
                futures = [executor.submit(self._process_raster_tile, task) for task in tasks]
                
                for future in tqdm(as_completed(futures), total=len(futures), 
                                  desc=f"Tiling {os.path.basename(image_path)} with {self.processes} processes"):
                    tile_path = future.result()
                    if tile_path:
                        tile_paths.append(tile_path)
        
        return tile_paths
    
    def process_directory(self, input_dir: str, output_dir: str, extensions: Optional[List[str]] = None, **kwargs) -> List[str]:
        """
        Process all geospatial images in directory.
        
        Parameters
        ----------
        input_dir : str
            Input directory containing images
        output_dir : str
            Output directory
        extensions : list, optional
            File extensions to process (default: ['.tif', '.tiff'])
            
        Returns
        -------
        list
            List of all tile file paths
        """
        if extensions is None:
            extensions = ['.tif', '.tiff']
            
        img_files = TilingUtils.find_image_files(input_dir, extensions)
        all_tiles = []
        
        for img_path in tqdm(img_files, desc="Processing images"):
            basename = os.path.splitext(os.path.basename(img_path))[0]+"_"+str(uuid.uuid4())[0:8]
            tiles = self.process_single(img_path, output_dir, prefix=basename, **kwargs)
            all_tiles.extend(tiles)
        
        return all_tiles
    
    def _process_raster_tile(self, args: Tuple) -> Optional[str]:
        """Worker function for raster tiling"""
        (raster_path, output_dir, tile_size, x, y, actual_width, actual_height, 
         meta, prefix, min_data_percentage, tile_id) = args
        
        try:
            with rasterio.open(raster_path) as src:
                # Create the window
                window = Window(x, y, actual_width, actual_height)
                transform = src.window_transform(window)
                
                # Read the data
                data = src.read(window=window, boundless=True, fill_value=0)[0:self.channels]
                data = np.where(np.isnan(data), 0, data)
                # Skip tiles with too little data
                non_zero_percentage = np.count_nonzero(data) / data.size
                if non_zero_percentage < min_data_percentage:
                    return None
                
                # Create output path
                output_path = os.path.join(output_dir, f"{prefix}_{tile_id}_{y}_{x}.tif")
                
                # Update metadata for this tile
                tile_meta = meta.copy()
                tile_meta.update({
                    'height': actual_height,
                    'width': actual_width,
                    'transform': transform,
                    'count': self.channels
                })
                
                with rasterio.open(output_path, 'w', **tile_meta) as dst:
                    dst.write(data)
                
                return output_path
        except Exception as e:
            print(f"Error processing tile at x={x}, y={y}: {e}")
            return None


class GeospatialImageLabelTiler(BaseImageLabelTiler):
    """Tiler for geospatial image-label pairs"""
    
    def process_single(self, image_path: str, label_path: str, output_dir: str, **kwargs) -> Tuple[List[str], List[str]]:
        """
        Process geospatial image-label pair.
        
        Parameters
        ----------
        image_path : str
            Path to the image file
        label_path : str
            Path to the label file
        output_dir : str
            Output directory
            
        Returns
        -------
        tuple
            (List of image tile paths, List of label tile paths)
        """
        img_output_dir, label_output_dir = self._setup_output_dirs(output_dir, create_img_label_dirs=True)
        
        # Verify dimensions match
        with rasterio.open(image_path) as img_src, rasterio.open(label_path) as label_src:
            if img_src.height != label_src.height or img_src.width != label_src.width:
                print(f"Warning: Image and label dimensions don't match for {os.path.basename(image_path)}, skipping...")
                return [], []
            
            # Get dimensions and calculate tile windows
            height, width = img_src.height, img_src.width
            file_id = os.path.splitext(os.path.basename(image_path))[0]
            
            # Calculate tile windows
            tile_windows = TilingUtils.calculate_tile_windows(width, height, self.tile_size, self.overlap, self.min_coverage)
            
            # Generate tasks
            tasks = []
            for x, y, actual_width, actual_height in tile_windows:
                tasks.append((
                    image_path, label_path, img_output_dir, label_output_dir,
                    self.tile_size, x, y, file_id, self.binary
                ))
            
            # Process tiles in parallel
            img_tiles = []
            label_tiles = []
            if tasks:
                with ProcessPoolExecutor(max_workers=self.processes) as executor:
                    futures = [executor.submit(self._process_image_label_tile, task) for task in tasks]
                    
                    for future in tqdm(as_completed(futures), total=len(futures), 
                                      desc=f"Processing {file_id} with {self.processes} processes"):
                        img_tile_path, label_tile_path = future.result()
                        if img_tile_path and label_tile_path:
                            img_tiles.append(img_tile_path)
                            label_tiles.append(label_tile_path)
        
        return img_tiles, label_tiles
    
    def process_directory(self, img_dir: str, label_dir: str, output_dir: str, 
                         extensions: Optional[List[str]] = None, **kwargs) -> Tuple[List[str], List[str]]:
        """
        Process all image-label pairs in directories.
        
        Parameters
        ----------
        img_dir : str
            Directory containing images
        label_dir : str
            Directory containing labels
        output_dir : str
            Output directory
        extensions : list, optional
            File extensions to process
            
        Returns
        -------
        tuple
            (List of image tile paths, List of label tile paths)
        """
        if extensions is None:
            extensions = ['.tif', '.tiff']
            
        img_files = TilingUtils.find_image_files(img_dir, extensions)
        
        all_img_tiles = []
        all_label_tiles = []
        
        for img_path in tqdm(img_files, desc="Processing image-label pairs"):
            # Get corresponding label path
            basename = os.path.basename(img_path)
            label_path = os.path.join(label_dir, basename)
            
            if not os.path.exists(label_path):
                print(f"Warning: No matching label found for {basename}, skipping...")
                continue
            
            img_tiles, label_tiles = self.process_single(img_path, label_path, output_dir, **kwargs)
            all_img_tiles.extend(img_tiles)
            all_label_tiles.extend(label_tiles)
        
        return all_img_tiles, all_label_tiles
    
    def _process_image_label_tile(self, args: Tuple) -> Tuple[Optional[str], Optional[str]]:
        """Worker function to process a single image-label tile pair"""
        img_path, label_path, img_output_dir, label_output_dir, tile_size, x, y, file_id, binary = args
        
        try:
            with rasterio.open(img_path) as img_src, rasterio.open(label_path) as label_src:
                # Define window for this tile
                window = Window(x, y, tile_size, tile_size)
                transform = img_src.window_transform(window)
                
                # Read data for both image and label with boundless=True for boundary tiles
                img_tile_data = img_src.read(window=window, boundless=True, fill_value=0)
                label_tile_data = label_src.read(window=window, boundless=True, fill_value=0)
                
                # Skip if image tile is empty (all zeros)
                if np.max(img_tile_data) == 0:
                    return None, None
                
                # Process label data based on binary flag and number of channels
                if label_tile_data.shape[0] > 1:  # Multi-channel label
                    if binary:
                        # For binary case with multiple channels, take any positive value across all channels
                        binary_mask = np.max(label_tile_data > 0, axis=0).astype(np.uint8)
                        label_tile_data = np.expand_dims(binary_mask, axis=0)
                elif binary:
                    # Single channel binary conversion
                    label_tile_data = (label_tile_data > 0).astype(np.uint8)
                
                # Create image tile
                img_tile_path = os.path.join(img_output_dir, f"{file_id}_tile_{y}_{x}.tif")
                img_meta = img_src.meta.copy()
                img_meta.update({
                    'height': tile_size,
                    'width': tile_size,
                    'transform': transform
                })
                
                with rasterio.open(img_tile_path, 'w', **img_meta) as dst:
                    dst.write(img_tile_data)
                
                # Create label tile
                label_tile_path = os.path.join(label_output_dir, f"{file_id}_tile_{y}_{x}.tif")
                label_meta = label_src.meta.copy()
                label_meta.update({
                    'height': tile_size,
                    'width': tile_size,
                    'transform': transform,
                    'count': label_tile_data.shape[0],  # Update count based on processed data
                    'dtype': 'uint8'  # Ensure dtype is uint8 for processed labels
                })
                
                with rasterio.open(label_tile_path, 'w', **label_meta) as dst:
                    dst.write(label_tile_data)
                
                return img_tile_path, label_tile_path
        except Exception as e:
            print(f"Error processing tile {file_id} at x={x}, y={y}: {e}")
            return None, None


class GeospatialShapefileTiler(BaseImageLabelTiler):
    """Tiler for geospatial images with shapefile labels"""
    
    def __init__(self, buffer_path: Optional[str] = None, **kwargs):
        """
        Initialize shapefile tiler.
        
        Parameters
        ----------
        buffer_path : str, optional
            Path to buffer shapefile for clipping (default: None)
        **kwargs
            Additional arguments passed to BaseImageLabelTiler
        """
        super().__init__(**kwargs)
        self.buffer_path = buffer_path
    
    def process_single(self, image_path: str, output_dir: str, shapefile_path: str, **kwargs) -> Tuple[List[str], List[str]]:
        """
        Process image and generate labels from shapefile.
        
        Parameters
        ----------
        image_path : str
            Path to the image file
        output_dir : str
            Output directory
        shapefile_path : str
            Path to the shapefile
            
        Returns
        -------
        tuple
            (List of image tile paths, List of label tile paths)
        """
        img_output_dir, label_output_dir = self._setup_output_dirs(output_dir, create_img_label_dirs=True)
        
        # Load shapefile
        label_gdf = gpd.read_file(shapefile_path)
        
        with rasterio.open(image_path) as src:
            # Ensure shapefile is in the same CRS as the raster
            if label_gdf.crs != src.crs:
                label_gdf = label_gdf.to_crs(src.crs)
            
            # Get image properties
            width, height = src.width, src.height
            file_id = os.path.splitext(os.path.basename(image_path))[0]
            
            # Calculate tile windows
            tile_windows = TilingUtils.calculate_tile_windows(width, height, self.tile_size, self.overlap, self.min_coverage)
            
            # Create windows_transforms list for label generation
            windows_transforms = []
            for i, (x, y, actual_width, actual_height) in enumerate(tile_windows):
                # Create window and transform for this tile
                tile_window = Window(x, y, actual_width, actual_height)
                tile_transform = src.window_transform(tile_window)
                tile_id = f"{file_id}_{i}"
                windows_transforms.append((tile_window, tile_transform, tile_id))
            
            # Process image tiles
            img_tiles = self._create_image_tiles(image_path, img_output_dir, tile_windows, file_id)
            
            # Process label tiles
            label_tiles = self._create_label_tiles_from_shapefile(
                windows_transforms, label_gdf, label_output_dir, file_id
            )
        
        return img_tiles, label_tiles
    
    def process_directory(self, img_dir: str, output_dir: str, shapefile_path: str, 
                         extensions: Optional[List[str]] = None, **kwargs) -> Tuple[List[str], List[str]]:
        """
        Process all images in directory with shapefile labels.
        
        Parameters
        ----------
        img_dir : str
            Directory containing images
        output_dir : str
            Output directory
        shapefile_path : str
            Path to the shapefile
        extensions : list, optional
            File extensions to process
            
        Returns
        -------
        tuple
            (List of image tile paths, List of label tile paths)
        """
        if extensions is None:
            extensions = ['.tif', '.tiff']
            
        img_files = TilingUtils.find_image_files(img_dir, extensions)
        
        all_img_tiles = []
        all_label_tiles = []
        
        for img_path in tqdm(img_files, desc="Processing images with shapefile"):
            img_tiles, label_tiles = self.process_single(img_path, output_dir, shapefile_path, **kwargs)
            all_img_tiles.extend(img_tiles)
            all_label_tiles.extend(label_tiles)
        
        return all_img_tiles, all_label_tiles
    
    def _create_image_tiles(self, image_path: str, output_dir: str, tile_windows: List[Tuple[int, int, int, int]], 
                           file_id: str) -> List[str]:
        """Create image tiles from windows"""
        with rasterio.open(image_path) as src:
            meta = src.meta.copy()
            
            tasks = []
            for i, (x, y, actual_width, actual_height) in enumerate(tile_windows):
                tasks.append((
                    image_path, output_dir, self.tile_size, x, y, 
                    actual_width, actual_height, meta, file_id, self.min_data_percentage, i
                ))
            
            # Process tiles in parallel
            tile_paths = []
            if tasks:
                with ProcessPoolExecutor(max_workers=self.processes) as executor:
                    futures = [executor.submit(self._process_raster_tile, task) for task in tasks]
                    
                    for future in tqdm(as_completed(futures), total=len(futures), 
                                      desc=f"Creating image tiles for {file_id}"):
                        tile_path = future.result()
                        if tile_path:
                            tile_paths.append(tile_path)
            
            return tile_paths
    
    def _create_label_tiles_from_shapefile(self, windows_transforms: List[Tuple], label_gdf: gpd.GeoDataFrame, 
                                         output_dir: str, file_id: str) -> List[str]:
        """Create label tiles from shapefile"""
        tasks = []
        for window, transform, tile_id in windows_transforms:
            tasks.append((
                window, transform, tile_id, label_gdf, output_dir, 
                self.tile_size, 'tif', file_id
            ))
        
        # Process tiles in parallel
        label_paths = []
        if tasks:
            with ProcessPoolExecutor(max_workers=self.processes) as executor:
                futures = [executor.submit(self._process_label_tile, task) for task in tasks]
                
                for future in tqdm(as_completed(futures), total=len(futures), 
                                  desc=f"Creating label tiles for {file_id}"):
                    label_path = future.result()
                    if label_path:
                        label_paths.append(label_path)
        
        return label_paths
    
    def _process_raster_tile(self, args: Tuple) -> Optional[str]:
        """Worker function for raster tiling"""
        (raster_path, output_dir, tile_size, x, y, actual_width, actual_height, 
         meta, prefix, min_data_percentage, tile_id) = args
        
        try:
            with rasterio.open(raster_path) as src:
                # Create the window
                window = Window(x, y, actual_width, actual_height)
                transform = src.window_transform(window)
                
                # Read the data
                data = src.read(window=window, boundless=True, fill_value=0)
                
                # Skip tiles with too little data
                non_zero_percentage = np.count_nonzero(data) / data.size
                if non_zero_percentage < min_data_percentage:
                    return None
                
                # Create output path
                output_path = os.path.join(output_dir, f"{prefix}_{tile_id}_{y}_{x}.tif")
                
                # Update metadata for this tile
                tile_meta = meta.copy()
                tile_meta.update({
                    'height': actual_height,
                    'width': actual_width,
                    'transform': transform
                })
                
                with rasterio.open(output_path, 'w', **tile_meta) as dst:
                    dst.write(data)
                
                return output_path
        except Exception as e:
            print(f"Error processing tile at x={x}, y={y}: {e}")
            return None
    
    def _process_label_tile(self, args: Tuple) -> Optional[str]:
        """Worker function for label tile creation"""
        (window, transform, tile_id, shapefile_gdf, output_dir, 
         tile_size, output_format, prefix) = args
        
        try:
            # Get tile bounds using rasterio window bounds function
            tile_bounds = window_bounds(window, transform)
            
            # Create label mask using utility
            label_mask = TilingUtils.create_label_mask_from_shapefile(shapefile_gdf, tile_bounds, tile_size, tile_size)
            
            # Create output path
            ext = '.tif' if output_format == 'tif' else '.png'
            output_path = os.path.join(output_dir, f"{prefix}_{tile_id}{ext}")
            
            if output_format == 'tif':
                # Save as GeoTIFF with spatial reference
                with rasterio.open(output_path, 'w', 
                                 driver='GTiff',
                                 height=tile_size,
                                 width=tile_size,
                                 count=1,
                                 dtype='uint8',
                                 transform=transform) as dst:
                    dst.write(label_mask, 1)
            else:
                # Save as PNG
                Image.fromarray(label_mask, mode='L').save(output_path)
            
            return output_path
        except Exception as e:
            print(f"Error processing label tile {tile_id}: {e}")
            return None


class GeospatialBufferTiler(GeospatialShapefileTiler):
    """Tiler for geospatial images with shapefile labels and buffer clipping"""
    
    def process_single(self, image_path: str, output_dir: str, shapefile_path: str, buffer_path: str, **kwargs) -> Tuple[List[str], List[str]]:
        """
        Process image with shapefile labels, clipped by buffer.
        
        Parameters
        ----------
        image_path : str
            Path to the image file
        output_dir : str
            Output directory
        shapefile_path : str
            Path to the shapefile
        buffer_path : str
            Path to the buffer shapefile
            
        Returns
        -------
        tuple
            (List of image tile paths, List of label tile paths)
        """
        import rasterio.mask
        
        img_output_dir, label_output_dir = self._setup_output_dirs(output_dir, create_img_label_dirs=True)
        
        # Load shapefiles
        buffer_gdf = gpd.read_file(buffer_path)
        label_gdf = gpd.read_file(shapefile_path)
        
        img_tiles = []
        label_tiles = []
        
        # Process each buffer geometry
        for idx, buffer_geom in enumerate(buffer_gdf.geometry):
            try:
                with rasterio.open(image_path) as src:
                    # Ensure buffer and label are in the same CRS as the raster
                    if buffer_gdf.crs != src.crs:
                        buffer_gdf = buffer_gdf.to_crs(src.crs)
                    if label_gdf.crs != src.crs:
                        label_gdf = label_gdf.to_crs(src.crs)
                    
                    # Get the bounds of the buffer in pixel coordinates
                    buffer_bounds = rasterio.features.bounds(buffer_geom)
                    window = src.window(*buffer_bounds)
                    
                    # Convert window to integer pixels
                    window = Window(int(window.col_off), int(window.row_off), 
                                   int(window.width), int(window.height))
                    
                    # Read the clipped data to check if empty
                    clipped_data = src.read(window=window, boundless=True, fill_value=0)
                    if np.max(clipped_data) == 0:
                        continue
                    
                    # Get clipped dimensions and transform
                    width = clipped_data.shape[2]
                    height = clipped_data.shape[1]
                    clipped_transform = src.window_transform(window)
                    
                    # Calculate tile windows
                    tile_windows = TilingUtils.calculate_tile_windows(width, height, self.tile_size, self.overlap, self.min_coverage)
                    
                    # Create windows_transforms list for label generation
                    windows_transforms = []
                    for i, (x, y, actual_width, actual_height) in enumerate(tile_windows):
                        # Create window and transform for this tile
                        tile_window = Window(x, y, actual_width, actual_height)
                        # Use rasterio's window_transform method for proper transform calculation
                        tile_transform = window_transform(tile_window, clipped_transform)
                        tile_id = f"buffer_{idx}_{i}"
                        windows_transforms.append((tile_window, tile_transform, tile_id))
                    
                    # Create a temporary raster file for the clipped data
                    temp_clipped_path = os.path.join(output_dir, f"temp_clipped_{idx}.tif")
                    
                    # Write clipped data to temporary file
                    clipped_meta = src.meta.copy()
                    clipped_meta.update({
                        'height': height,
                        'width': width,
                        'transform': clipped_transform
                    })
                    
                    with rasterio.open(temp_clipped_path, 'w', **clipped_meta) as dst:
                        dst.write(clipped_data)
                    
                    try:
                        # Process image tiles
                        prefix = f"buffer_{idx}"
                        img_tile_paths = self._create_image_tiles(temp_clipped_path, img_output_dir, tile_windows, prefix)
                        
                        # Process label tiles
                        label_tile_paths = self._create_label_tiles_from_shapefile(
                            windows_transforms, label_gdf, label_output_dir, prefix
                        )
                        
                        # Combine results
                        img_tiles.extend(img_tile_paths)
                        label_tiles.extend(label_tile_paths)
                        
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_clipped_path):
                            os.remove(temp_clipped_path)
                    
            except Exception as e:
                print(f"Error processing buffer {idx}: {e}")
                continue
        
        return img_tiles, label_tiles
    
    def process_directory(self, img_dir: str, output_dir: str, shapefile_path: str, buffer_path: str,
                         extensions: Optional[List[str]] = None, **kwargs) -> Tuple[List[str], List[str]]:
        """
        Process all images in directory with buffer clipping.
        
        Parameters
        ----------
        img_dir : str
            Directory containing images
        output_dir : str
            Output directory
        shapefile_path : str
            Path to the shapefile
        buffer_path : str
            Path to the buffer shapefile
        extensions : list, optional
            File extensions to process
            
        Returns
        -------
        tuple
            (List of image tile paths, List of label tile paths)
        """
        if extensions is None:
            extensions = ['.tif', '.tiff']
            
        img_files = TilingUtils.find_image_files(img_dir, extensions)
        
        all_img_tiles = []
        all_label_tiles = []
        
        for img_path in tqdm(img_files, desc="Processing images with buffer"):
            img_tiles, label_tiles = self.process_single(img_path, output_dir, shapefile_path, buffer_path, **kwargs)
            all_img_tiles.extend(img_tiles)
            all_label_tiles.extend(label_tiles)
        
        return all_img_tiles, all_label_tiles

# =============================================================================
# LMDB Tiling Classes (Geospatial)
# =============================================================================

class GeospatialLMDBTiler(BaseTiler, BaseLMDBTiler):
    """LMDB tiler for geospatial images"""
    
    def __init__(self, client_name: str = "default", **kwargs):
        BaseTiler.__init__(self, **kwargs)
        BaseLMDBTiler.__init__(self, client_name=client_name)
    
    def process_single(self, image_path: str, output_dir: str, **kwargs) -> int:
        """
        Process a single geospatial image to LMDB.
        
        Parameters
        ----------
        image_path : str
            Path to the geospatial image
        output_dir : str
            Output directory for LMDB
            
        Returns
        -------
        int
            Number of tiles created
        """
        env, key_index, current_date, client_name, output_path = self._setup_lmdb_environment(output_dir)
        
        try:
            with rasterio.open(image_path) as src:
                # Get image properties
                width, height = src.width, src.height
                band_count = src.count
                dtype = src.dtypes[0]
                
                # Calculate H3 index and resolution
                h3_index = TilingUtils.get_h3_index_for_raster(image_path)
                resolution = TilingUtils.calculate_resolution(image_path)
                
                # Calculate image percentiles
                image_data = src.read()
                percentile_dict = TilingUtils.calculate_image_percentiles(image_data, self.percentiles)
                
                # Calculate tile windows
                windows = TilingUtils.calculate_tile_windows(width, height, self.tile_size, self.overlap, self.min_coverage)
                
                # Process each tile
                for row, (x, y, actual_width, actual_height) in enumerate(windows):
                    # Create window
                    window = Window(x, y, actual_width, actual_height)
                    
                    # Read image tile
                    image_tile = src.read(window=window, boundless=True, fill_value=0)
                    
                    # Skip if empty
                    if np.max(image_tile) == 0:
                        continue
                    
                    # Resize to standard tile size if needed
                    if image_tile.shape[1] != self.tile_size or image_tile.shape[2] != self.tile_size:
                        resized_tile = np.zeros((band_count, self.tile_size, self.tile_size), dtype=image_tile.dtype)
                        resized_tile[:, :image_tile.shape[1], :image_tile.shape[2]] = image_tile
                        image_tile = resized_tile
                    
                    # Create dummy label (all zeros for image-only tiling)
                    label_data = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
                    
                    # Create tile key
                    image_name = os.path.basename(image_path)
                    source_id = str(uuid.uuid4())
                    tile_key = f"{h3_index}_{client_name}/{image_name}_{source_id}_{row}_{x//self.tile_size}_{y//self.tile_size}"
                    
                    # Create metadata - use tile bounds, not source image bounds
                    tile_bounds = window_bounds(window, src.transform)
                    bounds = {
                        'left': tile_bounds[0],
                        'bottom': tile_bounds[1],
                        'right': tile_bounds[2],
                        'top': tile_bounds[3]
                    } if src.bounds else None
                    
                    crs = {'init': str(src.crs)} if src.crs else None
                    
                    metadata = self._create_tile_metadata(
                        image_name=image_name,
                        dtype=dtype,
                        h3_index=h3_index,
                        percentile_dict=percentile_dict,
                        row=row,
                        col=x//self.tile_size,
                        window=window,
                        band_count=band_count,
                        image_tile_shape=image_tile.shape,
                        label_stats={0: self.tile_size * self.tile_size},
                        source_id=source_id,
                        bounds=bounds,
                        crs=crs,
                        resolution=resolution,
                        current_date=current_date,
                        label_band_count=1,
                        label_height=self.tile_size,
                        label_width=self.tile_size
                    )
                    
                    # Store in LMDB
                    self._store_tile_in_lmdb(env, tile_key, image_tile, label_data, metadata, band_count)
                    key_index.append(tile_key)
            
            # Finalize LMDB
            self._finalize_lmdb(env, key_index)
            return len(key_index)
            
        except Exception as e:
            env.close()
            raise e
    
    def process_directory(self, input_dir: str, output_dir: str, extensions: Optional[List[str]] = None, **kwargs) -> int:
        """
        Process all geospatial images in directory to LMDB.
        
        Parameters
        ----------
        input_dir : str
            Input directory containing images
        output_dir : str
            Output directory for LMDB
        extensions : list, optional
            File extensions to process (default: ['.tif', '.tiff'])
            
        Returns
        -------
        int
            Total number of tiles created
        """
        if extensions is None:
            extensions = ['.tif', '.tiff']
            
        img_files = TilingUtils.find_image_files(input_dir, extensions)
        total_tiles = 0
        
        for img_path in tqdm(img_files, desc="Processing images to LMDB"):
            tiles_created = self.process_single(img_path, output_dir, **kwargs)
            total_tiles += tiles_created
        
        return total_tiles


class GeospatialImageLabelLMDBTiler(BaseImageLabelTiler, BaseLMDBTiler):
    """LMDB tiler for geospatial image-label pairs"""
    
    def __init__(self, client_name: str = "default", **kwargs):
        BaseImageLabelTiler.__init__(self, **kwargs)
        BaseLMDBTiler.__init__(self, client_name=client_name)
    
    def process_directory(self, img_dir: str, label_dir: str, output_dir: str, **kwargs) -> int:
        """
        Process directory of image-label pairs to LMDB.
        """
        # Find image-label pairs
        img_files = list(pathlib.Path(img_dir).rglob("*.tif"))
        key_index = []
        
        for img_path in tqdm(img_files, desc="Processing image-label pairs to LMDB"):
            label_path = pathlib.Path(label_dir) / img_path.name
            if label_path.exists():
                key_index_img = self.process_single(str(img_path), str(label_path), output_dir, **kwargs)
                key_index.extend(key_index_img)
            else:
                print(f"Warning: No matching label found for {img_path.name}")

        env, _, _, _, _ = self._setup_lmdb_environment(output_dir)
        self._finalize_lmdb(env, key_index)
        return len(key_index)
    
    def process_single(self, image_path: str, label_path: str, output_dir: str, **kwargs) -> List[str]:
        """
        Process geospatial image-label pair to LMDB.
        
        Parameters
        ----------
        image_path : str
            Path to the image file
        label_path : str
            Path to the label file
        output_dir : str
            Output directory for LMDB
            
        Returns
        -------
        list
            List of tile keys created
        """
        env, key_index, current_date, client_name, output_path = self._setup_lmdb_environment(output_dir)
        
        try:
            with rasterio.open(image_path) as img_src, rasterio.open(label_path) as label_src:
                # Verify dimensions match
                if img_src.height != label_src.height or img_src.width != label_src.width:
                    print(f"Warning: Image and label dimensions don't match for {os.path.basename(image_path)}, skipping...")
                    self._finalize_lmdb(env, key_index)
                    return 0
                
                # Get image properties
                width, height = img_src.width, img_src.height
                band_count = img_src.count
                dtype = img_src.dtypes[0]
                
                # Calculate H3 index and resolution
                h3_index = TilingUtils.get_h3_index_for_raster(image_path)
                resolution = TilingUtils.calculate_resolution(image_path)
                
                # Calculate image percentiles
                image_data = img_src.read()
                percentile_dict = TilingUtils.calculate_image_percentiles(image_data, self.percentiles)
                
                # Calculate tile windows
                windows = TilingUtils.calculate_tile_windows(width, height, self.tile_size, self.overlap, self.min_coverage)
                
                # Process each tile
                for row, (x, y, actual_width, actual_height) in enumerate(windows):
                    # Create window
                    window = Window(x, y, actual_width, actual_height)
                    
                    # Read image and label tiles
                    image_tile = img_src.read(window=window, boundless=True, fill_value=0)
                    label_tile = label_src.read(window=window, boundless=True, fill_value=0)
                    
                    # Skip if image tile is empty
                    if np.max(image_tile) == 0:
                        continue
                    
                    # Process label data
                    if self.binary:
                        if label_tile.shape[0] > 1:
                            # Multi-channel to binary
                            label_data = np.max(label_tile > 0, axis=0).astype(np.uint8)
                        else:
                            # Single channel to binary
                            label_data = (label_tile[0] > 0).astype(np.uint8)
                    else:
                        # Keep original label format
                        label_data = label_tile[0] if label_tile.shape[0] == 1 else label_tile
                        if len(label_data.shape) > 2:
                            label_data = label_data[0]  # Take first channel if multi-channel
                        label_data = label_data.astype(np.uint8)
                    
                    # Resize tiles to standard size if needed
                    if image_tile.shape[1] != self.tile_size or image_tile.shape[2] != self.tile_size:
                        resized_tile = np.zeros((band_count, self.tile_size, self.tile_size), dtype=image_tile.dtype)
                        resized_tile[:, :image_tile.shape[1], :image_tile.shape[2]] = image_tile
                        image_tile = resized_tile
                    
                    if label_data.shape[0] != self.tile_size or label_data.shape[1] != self.tile_size:
                        resized_label = np.zeros((self.tile_size, self.tile_size), dtype=label_data.dtype)
                        resized_label[:label_data.shape[0], :label_data.shape[1]] = label_data
                        label_data = resized_label
                    
                    # Calculate label statistics
                    label_stats = TilingUtils.calculate_label_statistics(label_data)
                    
                    # Create tile key
                    image_name = os.path.basename(image_path)
                    source_id = str(uuid.uuid4())
                    tile_key = f"{h3_index}_{client_name}/{image_name}_{source_id}_{row}_{x//self.tile_size}_{y//self.tile_size}"
                    
                    # Create metadata - use tile bounds, not source image bounds
                    tile_bounds = window_bounds(window, img_src.transform)
                    bounds = {
                        'left': tile_bounds[0],
                        'bottom': tile_bounds[1],
                        'right': tile_bounds[2],
                        'top': tile_bounds[3]
                    } if img_src.bounds else None
                    
                    crs = {'init': str(img_src.crs)} if img_src.crs else None
                    
                    metadata = self._create_tile_metadata(
                        image_name=image_name,
                        dtype=dtype,
                        h3_index=h3_index,
                        percentile_dict=percentile_dict,
                        row=row,
                        col=x//self.tile_size,
                        window=window,
                        band_count=band_count,
                        image_tile_shape=image_tile.shape,
                        label_stats=label_stats,
                        source_id=source_id,
                        bounds=bounds,
                        crs=crs,
                        resolution=resolution,
                        current_date=current_date,
                        label_band_count=1,
                        label_height=self.tile_size,
                        label_width=self.tile_size
                    )
                    
                    # Store in LMDB
                    self._store_tile_in_lmdb(env, tile_key, image_tile, label_data, metadata, band_count)
                    key_index.append(tile_key)
            
            # Finalize LMDB
            self._finalize_lmdb(env, key_index)
            env.close()
            return key_index
            
        except Exception as e:
            env.close()
            raise e


class GeospatialShapefileLMDBTiler(GeospatialShapefileTiler, BaseLMDBTiler):
    """LMDB tiler for geospatial images with shapefile labels"""
    
    def __init__(self, client_name: str = "default", **kwargs):
        GeospatialShapefileTiler.__init__(self, **kwargs)
        BaseLMDBTiler.__init__(self, client_name=client_name)
    
    def process_single(self, image_path: str, output_dir: str, shapefile_path: str, **kwargs) -> int:
        """
        Process image with shapefile labels to LMDB.
        
        Parameters
        ----------
        image_path : str
            Path to the image file
        output_dir : str
            Output directory for LMDB
        shapefile_path : str
            Path to the shapefile
            
        Returns
        -------
        int
            Number of tiles created
        """
        env, key_index, current_date, client_name, output_path = self._setup_lmdb_environment(output_dir)
        
        try:
            # Load shapefile
            label_gdf = gpd.read_file(shapefile_path)
            
            with rasterio.open(image_path) as src:
                # Ensure shapefile is in the same CRS as the raster
                if label_gdf.crs != src.crs:
                    label_gdf = label_gdf.to_crs(src.crs)
                
                # Get image properties
                width, height = src.width, src.height
                band_count = src.count
                dtype = src.dtypes[0]
                
                # Calculate H3 index and resolution
                h3_index = TilingUtils.get_h3_index_for_raster(image_path)
                resolution = TilingUtils.calculate_resolution(image_path)
                
                # Calculate image percentiles
                image_data = src.read()
                percentile_dict = TilingUtils.calculate_image_percentiles(image_data, self.percentiles)
                
                # Calculate tile windows
                windows = TilingUtils.calculate_tile_windows(width, height, self.tile_size, self.overlap, self.min_coverage)
                
                # Process each tile
                for row, (x, y, actual_width, actual_height) in enumerate(windows):
                    # Create window
                    window = Window(x, y, actual_width, actual_height)
                    tile_transform = src.window_transform(window)
                    
                    # Read image tile
                    image_tile = src.read(window=window, boundless=True, fill_value=0)
                    
                    # Skip if image tile is empty
                    if np.max(image_tile) == 0:
                        continue
                    
                    # Get tile bounds using rasterio window bounds function
                    tile_bounds = window_bounds(window, src.transform)
                    
                    # Create label mask from shapefile
                    label_data = TilingUtils.create_label_mask_from_shapefile(label_gdf, tile_bounds, self.tile_size, self.tile_size)
                    
                    # Resize tiles to standard size if needed
                    if image_tile.shape[1] != self.tile_size or image_tile.shape[2] != self.tile_size:
                        resized_tile = np.zeros((band_count, self.tile_size, self.tile_size), dtype=image_tile.dtype)
                        resized_tile[:, :image_tile.shape[1], :image_tile.shape[2]] = image_tile
                        image_tile = resized_tile
                    
                    # Calculate label statistics
                    label_stats = TilingUtils.calculate_label_statistics(label_data)
                    
                    # Create tile key
                    image_name = os.path.basename(image_path)
                    source_id = str(uuid.uuid4())
                    tile_key = f"{h3_index}_{client_name}/{image_name}_{source_id}_{row}_{x//self.tile_size}_{y//self.tile_size}"
                    
                    # Create metadata - use tile_bounds (already computed above), not source image bounds
                    bounds = {
                        'left': tile_bounds[0],
                        'bottom': tile_bounds[1],
                        'right': tile_bounds[2],
                        'top': tile_bounds[3]
                    } if src.bounds else None
                    
                    crs = {'init': str(src.crs)} if src.crs else None
                    
                    metadata = self._create_tile_metadata(
                        image_name=image_name,
                        dtype=dtype,
                        h3_index=h3_index,
                        percentile_dict=percentile_dict,
                        row=row,
                        col=x//self.tile_size,
                        window=window,
                        band_count=band_count,
                        image_tile_shape=image_tile.shape,
                        label_stats=label_stats,
                        source_id=source_id,
                        bounds=bounds,
                        crs=crs,
                        resolution=resolution,
                        current_date=current_date,
                        label_band_count=1,
                        label_height=self.tile_size,
                        label_width=self.tile_size
                    )
                    
                    # Store in LMDB
                    self._store_tile_in_lmdb(env, tile_key, image_tile, label_data, metadata, band_count)
                    key_index.append(tile_key)
            
            # Finalize LMDB
            self._finalize_lmdb(env, key_index)
            return len(key_index)
            
        except Exception as e:
            env.close()
            raise e


class GeospatialBufferLMDBTiler(GeospatialBufferTiler, BaseLMDBTiler):
    """LMDB tiler for geospatial images with shapefile labels and buffer clipping"""
    
    def __init__(self, client_name: str = "default", **kwargs):
        GeospatialBufferTiler.__init__(self, **kwargs)
        BaseLMDBTiler.__init__(self, client_name=client_name)
    
    def tile_image_with_buffer_to_lmdb(self, image_path: str, shapefile_path: str, buffer_path: str, output_dir: str) -> int:
        """
        Tile image with shapefile labels, clipped by buffer, to LMDB.
        
        Parameters
        ----------
        image_path : str
            Path to the image file
        shapefile_path : str
            Path to the shapefile
        buffer_path : str
            Path to the buffer shapefile
        output_dir : str
            Output directory for LMDB
            
        Returns
        -------
        int
            Number of tiles created
        """
        import rasterio.mask
        
        env, key_index, current_date, client_name, output_path = self._setup_lmdb_environment(output_dir)
        
        try:
            # Load shapefiles
            buffer_gdf = gpd.read_file(buffer_path)
            label_gdf = gpd.read_file(shapefile_path)
            
            # Process each buffer geometry
            for buffer_idx, buffer_geom in enumerate(buffer_gdf.geometry):
                try:
                    with rasterio.open(image_path) as src:
                        # Ensure buffer and label are in the same CRS as the raster
                        if buffer_gdf.crs != src.crs:
                            buffer_gdf = buffer_gdf.to_crs(src.crs)
                        if label_gdf.crs != src.crs:
                            label_gdf = label_gdf.to_crs(src.crs)
                        
                        # Get the bounds of the buffer in pixel coordinates
                        buffer_bounds = rasterio.features.bounds(buffer_geom)
                        buffer_window = src.window(*buffer_bounds)
                        
                        # Convert window to integer pixels
                        buffer_window = Window(int(buffer_window.col_off), int(buffer_window.row_off), 
                                             int(buffer_window.width), int(buffer_window.height))
                        
                        # Read the clipped data to check if empty
                        clipped_data = src.read(window=buffer_window, boundless=True, fill_value=0)
                        if np.max(clipped_data) == 0:
                            continue
                        
                        # Get clipped dimensions and transform
                        clipped_width = clipped_data.shape[2]
                        clipped_height = clipped_data.shape[1]
                        clipped_transform = src.window_transform(buffer_window)
                        
                        # Get image properties
                        band_count = src.count
                        dtype = src.dtypes[0]
                        
                        # Calculate H3 index and resolution
                        h3_index = TilingUtils.get_h3_index_for_raster(image_path)
                        resolution = TilingUtils.calculate_resolution(image_path)
                        
                        # Calculate image percentiles for the clipped data
                        percentile_dict = TilingUtils.calculate_image_percentiles(clipped_data, self.percentiles)
                        
                        # Calculate tile windows for the clipped area
                        tile_windows = TilingUtils.calculate_tile_windows(clipped_width, clipped_height, self.tile_size, self.overlap, self.min_coverage)
                        
                        # Process each tile within this buffer
                        for tile_idx, (x, y, actual_width, actual_height) in enumerate(tile_windows):
                            # Create window for this tile within the clipped area
                            tile_window = Window(x, y, actual_width, actual_height)
                            tile_transform = window_transform(tile_window, clipped_transform)
                            
                            # Read image tile from the clipped data
                            image_tile = clipped_data[:, y:y+actual_height, x:x+actual_width]
                            
                            # Skip if image tile is empty
                            if np.max(image_tile) == 0:
                                continue
                            
                            # Get tile bounds for label generation
                            tile_bounds = window_bounds(tile_window, clipped_transform)
                            
                            # Create label mask from shapefile
                            label_data = TilingUtils.create_label_mask_from_shapefile(label_gdf, tile_bounds, self.tile_size, self.tile_size)
                            
                            # Resize tiles to standard size if needed
                            if image_tile.shape[1] != self.tile_size or image_tile.shape[2] != self.tile_size:
                                resized_tile = np.zeros((band_count, self.tile_size, self.tile_size), dtype=image_tile.dtype)
                                resized_tile[:, :image_tile.shape[1], :image_tile.shape[2]] = image_tile
                                image_tile = resized_tile
                            
                            # Resize label if needed
                            if label_data.shape[0] != self.tile_size or label_data.shape[1] != self.tile_size:
                                resized_label = np.zeros((self.tile_size, self.tile_size), dtype=label_data.dtype)
                                resized_label[:label_data.shape[0], :label_data.shape[1]] = label_data
                                label_data = resized_label
                            
                            # Calculate label statistics
                            label_stats = TilingUtils.calculate_label_statistics(label_data)
                            
                            # Create tile key with buffer information
                            image_name = os.path.basename(image_path)
                            source_id = str(uuid.uuid4())
                            tile_key = f"{h3_index}_{client_name}/buffer_{buffer_idx}_{image_name}_{source_id}_{tile_idx}_{x//self.tile_size}_{y//self.tile_size}"
                            
                            # Create metadata - use tile_bounds (already computed above), not source image bounds
                            bounds = {
                                'left': tile_bounds[0],
                                'bottom': tile_bounds[1],
                                'right': tile_bounds[2],
                                'top': tile_bounds[3]
                            } if src.bounds else None
                            
                            crs = {'init': str(src.crs)} if src.crs else None
                            
                            # Add buffer-specific metadata
                            metadata = self._create_tile_metadata(
                                image_name=image_name,
                                dtype=dtype,
                                h3_index=h3_index,
                                percentile_dict=percentile_dict,
                                row=tile_idx,
                                col=x//self.tile_size,
                                window=tile_window,
                                band_count=band_count,
                                image_tile_shape=image_tile.shape,
                                label_stats=label_stats,
                                source_id=source_id,
                                bounds=bounds,
                                crs=crs,
                                resolution=resolution,
                                current_date=current_date,
                                label_band_count=1,
                                label_height=self.tile_size,
                                label_width=self.tile_size,
                                buffer_idx=buffer_idx,
                                buffer_geometry=str(buffer_geom)
                            )
                            
                            # Store in LMDB
                            self._store_tile_in_lmdb(env, tile_key, image_tile, label_data, metadata, band_count)
                            key_index.append(tile_key)
                            
                except Exception as e:
                    print(f"Error processing buffer {buffer_idx}: {e}")
                    continue
            
            # Finalize LMDB
            self._finalize_lmdb(env, key_index)
            return len(key_index)
            
        except Exception as e:
            env.close()
            raise e
    
    def tile_directory_with_buffer_to_lmdb(self, img_dir: str, shapefile_path: str, buffer_path: str, output_dir: str, 
                                         extensions: Optional[List[str]] = None) -> int:
        """
        Tile all images in directory with buffer clipping to LMDB.
        
        Parameters
        ----------
        img_dir : str
            Directory containing images
        shapefile_path : str
            Path to the shapefile
        buffer_path : str
            Path to the buffer shapefile
        output_dir : str
            Output directory for LMDB
        extensions : list, optional
            File extensions to process (default: ['.tif', '.tiff'])
            
        Returns
        -------
        int
            Total number of tiles created
        """
        if extensions is None:
            extensions = ['.tif', '.tiff']
            
        img_files = TilingUtils.find_image_files(img_dir, extensions)
        
        env, key_index, current_date, client_name, output_path = self._setup_lmdb_environment(output_dir)
        
        try:
            total_tiles = 0
            
            for img_path in tqdm(img_files, desc="Processing images with buffer to LMDB"):
                # Close and reopen environment to avoid keeping it open too long
                env.close()
                env, _, _, _, _ = self._setup_lmdb_environment(output_dir)
                
                tiles_created = self._process_single_image_buffer_lmdb(
                    img_path, shapefile_path, buffer_path, env, key_index, 
                    current_date, client_name
                )
                total_tiles += tiles_created
            
            # Finalize LMDB
            self._finalize_lmdb(env, key_index)
            return total_tiles
            
        except Exception as e:
            env.close()
            raise e
    
    def _process_single_image_buffer_lmdb(self, image_path: str, shapefile_path: str, buffer_path: str,
                                        env, key_index: List[str], current_date: str, client_name: str) -> int:
        """Process a single image with buffer clipping for LMDB storage"""
        import rasterio.mask
        
        tiles_created = 0
        
        try:
            # Load shapefiles
            buffer_gdf = gpd.read_file(buffer_path)
            label_gdf = gpd.read_file(shapefile_path)
            
            # Process each buffer geometry
            for buffer_idx, buffer_geom in enumerate(buffer_gdf.geometry):
                try:
                    with rasterio.open(image_path) as src:
                        # Ensure buffer and label are in the same CRS as the raster
                        if buffer_gdf.crs != src.crs:
                            buffer_gdf = buffer_gdf.to_crs(src.crs)
                        if label_gdf.crs != src.crs:
                            label_gdf = label_gdf.to_crs(src.crs)
                        
                        # Get the bounds of the buffer in pixel coordinates
                        buffer_bounds = rasterio.features.bounds(buffer_geom)
                        buffer_window = src.window(*buffer_bounds)
                        
                        # Convert window to integer pixels
                        buffer_window = Window(int(buffer_window.col_off), int(buffer_window.row_off), 
                                             int(buffer_window.width), int(buffer_window.height))
                        
                        # Read the clipped data to check if empty
                        clipped_data = src.read(window=buffer_window, boundless=True, fill_value=0)
                        if np.max(clipped_data) == 0:
                            continue
                        
                        # Get clipped dimensions and transform
                        clipped_width = clipped_data.shape[2]
                        clipped_height = clipped_data.shape[1]
                        clipped_transform = src.window_transform(buffer_window)
                        
                        # Get image properties
                        band_count = src.count
                        dtype = src.dtypes[0]
                        
                        # Calculate H3 index and resolution
                        h3_index = TilingUtils.get_h3_index_for_raster(image_path)
                        resolution = TilingUtils.calculate_resolution(image_path)
                        
                        # Calculate image percentiles for the clipped data
                        percentile_dict = TilingUtils.calculate_image_percentiles(clipped_data, self.percentiles)
                        
                        # Calculate tile windows for the clipped area
                        tile_windows = TilingUtils.calculate_tile_windows(clipped_width, clipped_height, self.tile_size, self.overlap, self.min_coverage)
                        
                        # Process each tile within this buffer
                        for tile_idx, (x, y, actual_width, actual_height) in enumerate(tile_windows):
                            # Create window for this tile within the clipped area
                            tile_window = Window(x, y, actual_width, actual_height)
                            tile_transform = window_transform(tile_window, clipped_transform)
                            
                            # Read image tile from the clipped data
                            image_tile = clipped_data[:, y:y+actual_height, x:x+actual_width]
                            
                            # Skip if image tile is empty
                            if np.max(image_tile) == 0:
                                continue
                            
                            # Get tile bounds for label generation
                            tile_bounds = window_bounds(tile_window, clipped_transform)
                            
                            # Create label mask from shapefile
                            label_data = TilingUtils.create_label_mask_from_shapefile(label_gdf, tile_bounds, self.tile_size, self.tile_size)
                            
                            # Resize tiles to standard size if needed
                            if image_tile.shape[1] != self.tile_size or image_tile.shape[2] != self.tile_size:
                                resized_tile = np.zeros((band_count, self.tile_size, self.tile_size), dtype=image_tile.dtype)
                                resized_tile[:, :image_tile.shape[1], :image_tile.shape[2]] = image_tile
                                image_tile = resized_tile
                            
                            # Resize label if needed
                            if label_data.shape[0] != self.tile_size or label_data.shape[1] != self.tile_size:
                                resized_label = np.zeros((self.tile_size, self.tile_size), dtype=label_data.dtype)
                                resized_label[:label_data.shape[0], :label_data.shape[1]] = label_data
                                label_data = resized_label
                            
                            # Calculate label statistics
                            label_stats = TilingUtils.calculate_label_statistics(label_data)
                            
                            # Create tile key with buffer information
                            image_name = os.path.basename(image_path)
                            source_id = str(uuid.uuid4())
                            tile_key = f"{h3_index}_{client_name}/buffer_{buffer_idx}_{image_name}_{source_id}_{tile_idx}_{x//self.tile_size}_{y//self.tile_size}"
                            
                            # Create metadata - use tile_bounds (already computed above), not source image bounds
                            bounds = {
                                'left': tile_bounds[0],
                                'bottom': tile_bounds[1],
                                'right': tile_bounds[2],
                                'top': tile_bounds[3]
                            } if src.bounds else None
                            
                            crs = {'init': str(src.crs)} if src.crs else None
                            
                            # Add buffer-specific metadata
                            metadata = self._create_tile_metadata(
                                image_name=image_name,
                                dtype=dtype,
                                h3_index=h3_index,
                                percentile_dict=percentile_dict,
                                row=tile_idx,
                                col=x//self.tile_size,
                                window=tile_window,
                                band_count=band_count,
                                image_tile_shape=image_tile.shape,
                                label_stats=label_stats,
                                source_id=source_id,
                                bounds=bounds,
                                crs=crs,
                                resolution=resolution,
                                current_date=current_date,
                                label_band_count=1,
                                label_height=self.tile_size,
                                label_width=self.tile_size,
                                buffer_idx=buffer_idx,
                                buffer_geometry=str(buffer_geom)
                            )
                            
                            # Store in LMDB
                            self._store_tile_in_lmdb(env, tile_key, image_tile, label_data, metadata, band_count)
                            key_index.append(tile_key)
                            tiles_created += 1
                            
                except Exception as e:
                    print(f"Error processing buffer {buffer_idx} for {os.path.basename(image_path)}: {e}")
                    continue
            
        except Exception as e:
            print(f"Error processing image {os.path.basename(image_path)}: {e}")
        
        return tiles_created


# =============================================================================
# Non-Geospatial Tiling Classes
# =============================================================================

class NonGeospatialTiler(BaseTiler):
    """Tiler for non-geospatial images (PNG, JPG, etc.)"""
    
    def tile_single_image(self, image_path: str, output_dir: str, prefix: str = "tile") -> List[str]:
        """
        Tile a single non-geospatial image.
        
        Parameters
        ----------
        image_path : str
            Path to the image
        output_dir : str
            Output directory
        prefix : str, optional
            Prefix for tile filenames (default: "tile")
            
        Returns
        -------
        list
            List of tile file paths
        """
        output_dir = self._setup_output_dirs(output_dir)
        
        # Load image
        image = Image.open(image_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        height, width = image_array.shape[:2]
        
        # Calculate tile windows
        windows = TilingUtils.calculate_tile_windows(width, height, self.tile_size, self.overlap, self.min_coverage)
        
        # Generate tasks
        tasks = []
        for i, (x, y, actual_width, actual_height) in enumerate(windows):
            tasks.append((
                image_path, output_dir, self.tile_size, x, y, 
                actual_width, actual_height, prefix, self.min_data_percentage, i
            ))
        
        # Process tiles in parallel
        tile_paths = []
        if tasks:
            with ProcessPoolExecutor(max_workers=self.processes) as executor:
                futures = [executor.submit(self._process_image_tile, task) for task in tasks]
                
                for future in tqdm(as_completed(futures), total=len(futures), 
                                  desc=f"Tiling {os.path.basename(image_path)} with {self.processes} processes"):
                    tile_path = future.result()
                    if tile_path:
                        tile_paths.append(tile_path)
        
        return tile_paths
    
    def tile_directory(self, input_dir: str, output_dir: str, extensions: Optional[List[str]] = None) -> List[str]:
        """
        Tile all non-geospatial images in directory.
        
        Parameters
        ----------
        input_dir : str
            Input directory containing images
        output_dir : str
            Output directory
        extensions : list, optional
            File extensions to process (default: ['.png', '.jpg', '.jpeg'])
            
        Returns
        -------
        list
            List of all tile file paths
        """
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg']
            
        img_files = TilingUtils.find_image_files(input_dir, extensions)
        all_tiles = []
        
        for img_path in tqdm(img_files, desc="Processing images"):
            basename = os.path.splitext(os.path.basename(img_path))[0]
            tiles = self.tile_single_image(img_path, output_dir, prefix=basename)
            all_tiles.extend(tiles)
        
        return all_tiles
    
    def _process_image_tile(self, args: Tuple) -> Optional[str]:
        """Worker function for image tiling"""
        (image_path, output_dir, tile_size, x, y, actual_width, actual_height, 
         prefix, min_data_percentage, tile_id) = args
        
        try:
            # Load image
            image = Image.open(image_path)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # Extract tile
            tile = image.crop((x, y, x + actual_width, y + actual_height))
            
            # Convert to array to check data
            tile_array = np.array(tile)
            
            # Skip tiles with too little data (check if mostly black/empty)
            non_zero_percentage = np.count_nonzero(tile_array) / tile_array.size
            if non_zero_percentage < min_data_percentage:
                return None
            
            # Resize to standard tile size if needed
            if actual_width != tile_size or actual_height != tile_size:
                tile = tile.resize((tile_size, tile_size), Image.LANCZOS)
            
            # Create output path
            output_path = os.path.join(output_dir, f"{prefix}_{tile_id}_{y}_{x}.png")
            
            # Save tile
            tile.save(output_path)
            
            return output_path
        except Exception as e:
            print(f"Error processing tile at x={x}, y={y}: {e}")
            return None


class NonGeospatialImageLabelTiler(BaseImageLabelTiler):
    """Tiler for non-geospatial image-label pairs"""
    
    def tile_image_label_pair(self, image_path: str, label_path: str, output_dir: str) -> Tuple[List[str], List[str]]:
        """
        Tile non-geospatial image-label pair.
        
        Parameters
        ----------
        image_path : str
            Path to the image file
        label_path : str
            Path to the label file
        output_dir : str
            Output directory
            
        Returns
        -------
        tuple
            (List of image tile paths, List of label tile paths)
        """
        img_output_dir, label_output_dir = self._setup_output_dirs(output_dir, create_img_label_dirs=True)
        
        # Load images
        image = Image.open(image_path)
        label = Image.open(label_path)
        
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Verify dimensions match
        if image.size != label.size:
            print(f"Warning: Image and label dimensions don't match for {os.path.basename(image_path)}, skipping...")
            return [], []
        
        # Get dimensions
        width, height = image.size
        file_id = os.path.splitext(os.path.basename(image_path))[0]
        
        # Calculate tile windows
        tile_windows = TilingUtils.calculate_tile_windows(width, height, self.tile_size, self.overlap, self.min_coverage)
        
        # Generate tasks
        tasks = []
        for x, y, actual_width, actual_height in tile_windows:
            tasks.append((
                image_path, label_path, img_output_dir, label_output_dir,
                self.tile_size, x, y, file_id, self.binary
            ))
        
        # Process tiles in parallel
        img_tiles = []
        label_tiles = []
        if tasks:
            with ProcessPoolExecutor(max_workers=self.processes) as executor:
                futures = [executor.submit(self._process_image_label_tile, task) for task in tasks]
                
                for future in tqdm(as_completed(futures), total=len(futures), 
                                  desc=f"Processing {file_id} with {self.processes} processes"):
                    img_tile_path, label_tile_path = future.result()
                    if img_tile_path and label_tile_path:
                        img_tiles.append(img_tile_path)
                        label_tiles.append(label_tile_path)
        
        return img_tiles, label_tiles
    
    def _process_image_label_tile(self, args: Tuple) -> Tuple[Optional[str], Optional[str]]:
        """Worker function to process a single image-label tile pair"""
        img_path, label_path, img_output_dir, label_output_dir, tile_size, x, y, file_id, binary = args
        
        try:
            # Load images
            image = Image.open(img_path)
            label = Image.open(label_path)
            
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # Extract tiles
            img_tile = image.crop((x, y, x + tile_size, y + tile_size))
            label_tile = label.crop((x, y, x + tile_size, y + tile_size))
            
            # Check if image tile is empty
            img_array = np.array(img_tile)
            if np.max(img_array) == 0:
                return None, None
            
            # Process label tile
            label_array = np.array(label_tile)
            if binary:
                # Convert to binary mask
                if len(label_array.shape) == 3:  # RGB label
                    binary_mask = np.max(label_array > 0, axis=2).astype(np.uint8) * 255
                    label_tile = Image.fromarray(binary_mask, mode='L')
                else:  # Grayscale label
                    binary_mask = (label_array > 0).astype(np.uint8) * 255
                    label_tile = Image.fromarray(binary_mask, mode='L')
            
            # Create output paths
            img_tile_path = os.path.join(img_output_dir, f"{file_id}_tile_{y}_{x}.png")
            label_tile_path = os.path.join(label_output_dir, f"{file_id}_tile_{y}_{x}.png")
            
            # Save tiles
            img_tile.save(img_tile_path)
            label_tile.save(label_tile_path)
            
            return img_tile_path, label_tile_path
        except Exception as e:
            print(f"Error processing tile {file_id} at x={x}, y={y}: {e}")
            return None, None


class NonGeospatialInstanceTiler(BaseImageLabelTiler):
    """Tiler for non-geospatial instance segmentation"""
    
    def __init__(self, erosion_kernel_size: int = 0, **kwargs):
        """
        Initialize instance tiler.
        
        Parameters
        ----------
        erosion_kernel_size : int, optional
            Size of erosion kernel for separating instances (default: 0)
        **kwargs
            Additional arguments passed to BaseImageLabelTiler
        """
        super().__init__(**kwargs)
        self.erosion_kernel_size = erosion_kernel_size
    
    def tile_image_label_pair(self, image_path: str, label_path: str, output_dir: str) -> Tuple[List[str], List[str]]:
        """
        Tile image-instance label pair.
        
        Parameters
        ----------
        image_path : str
            Path to the image file
        label_path : str
            Path to the instance label file
        output_dir : str
            Output directory
            
        Returns
        -------
        tuple
            (List of image tile paths, List of label tile paths)
        """
        img_output_dir, label_output_dir = self._setup_output_dirs(output_dir, create_img_label_dirs=True)
        
        # Load images
        image = Image.open(image_path)
        label = Image.open(label_path)
        
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Verify dimensions match
        if image.size != label.size:
            print(f"Warning: Image and label dimensions don't match for {os.path.basename(image_path)}, skipping...")
            return [], []
        
        # Get dimensions
        width, height = image.size
        file_id = os.path.splitext(os.path.basename(image_path))[0]
        
        # Calculate tile windows
        tile_windows = TilingUtils.calculate_tile_windows(width, height, self.tile_size, self.overlap, self.min_coverage)
        
        # Generate tasks
        tasks = []
        for x, y, actual_width, actual_height in tile_windows:
            tasks.append((
                image_path, label_path, img_output_dir, label_output_dir,
                self.tile_size, x, y, file_id, self.erosion_kernel_size
            ))
        
        # Process tiles in parallel
        img_tiles = []
        label_tiles = []
        if tasks:
            with ProcessPoolExecutor(max_workers=self.processes) as executor:
                futures = [executor.submit(self._process_instance_tile, task) for task in tasks]
                
                for future in tqdm(as_completed(futures), total=len(futures), 
                                  desc=f"Processing {file_id} with {self.processes} processes"):
                    img_tile_path, label_tile_path = future.result()
                    if img_tile_path and label_tile_path:
                        img_tiles.append(img_tile_path)
                        label_tiles.append(label_tile_path)
        
        return img_tiles, label_tiles
    
    def _process_instance_tile(self, args: Tuple) -> Tuple[Optional[str], Optional[str]]:
        """Worker function to process a single instance tile pair"""
        img_path, label_path, img_output_dir, label_output_dir, tile_size, x, y, file_id, erosion_kernel_size = args
        
        try:
            # Load images
            image = Image.open(img_path)
            label = Image.open(label_path)
            
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # Extract tiles
            img_tile = image.crop((x, y, x + tile_size, y + tile_size))
            label_tile = label.crop((x, y, x + tile_size, y + tile_size))
            
            # Check if image tile is empty
            img_array = np.array(img_tile)
            if np.max(img_array) == 0:
                return None, None
            
            # Process instance label
            label_array = np.array(label_tile)
            processed_label = TilingUtils.process_instance_label_with_erosion(label_array, erosion_kernel_size)
            
            # Convert back to PIL Image
            processed_label_img = Image.fromarray((processed_label * 255).astype(np.uint8), mode='L')
            
            # Create output paths
            img_tile_path = os.path.join(img_output_dir, f"{file_id}_tile_{y}_{x}.png")
            label_tile_path = os.path.join(label_output_dir, f"{file_id}_tile_{y}_{x}.png")
            
            # Save tiles
            img_tile.save(img_tile_path)
            processed_label_img.save(label_tile_path)
            
            return img_tile_path, label_tile_path
        except Exception as e:
            print(f"Error processing instance tile {file_id} at x={x}, y={y}: {e}")
            return None, None


# =============================================================================
# Non-Geospatial LMDB Tiling Classes
# =============================================================================

class NonGeospatialLMDBTiler(BaseTiler, BaseLMDBTiler):
    """LMDB tiler for non-geospatial images"""
    
    def __init__(self, client_name: str = "default", **kwargs):
        BaseTiler.__init__(self, **kwargs)
        BaseLMDBTiler.__init__(self, client_name=client_name)
    
    def tile_single_image_to_lmdb(self, image_path: str, output_dir: str) -> int:
        """
        Tile a single non-geospatial image to LMDB.
        
        Parameters
        ----------
        image_path : str
            Path to the image
        output_dir : str
            Output directory for LMDB
            
        Returns
        -------
        int
            Number of tiles created
        """
        env, key_index, current_date, client_name, output_path = self._setup_lmdb_environment(output_dir)
        
        try:
            # Load image
            image = Image.open(image_path)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            image_array = np.array(image)
            height, width = image_array.shape[:2]
            
            # For non-geospatial images, we don't have bands in the same way
            if len(image_array.shape) == 3:
                band_count = image_array.shape[2]
                # Reshape to (bands, height, width) format
                image_data = np.transpose(image_array, (2, 0, 1))
            else:
                band_count = 1
                image_data = np.expand_dims(image_array, axis=0)
            
            # Calculate image percentiles
            percentile_dict = TilingUtils.calculate_image_percentiles(image_data, self.percentiles)
            
            # Calculate tile windows
            windows = TilingUtils.calculate_tile_windows(width, height, self.tile_size, self.overlap, self.min_coverage)
            
            # Process each tile
            for row, (x, y, actual_width, actual_height) in enumerate(windows):
                # Extract tile
                if len(image_array.shape) == 3:
                    tile_data = image_array[y:y+actual_height, x:x+actual_width, :]
                    # Transpose to (channels, height, width)
                    image_tile = np.transpose(tile_data, (2, 0, 1))
                else:
                    tile_data = image_array[y:y+actual_height, x:x+actual_width]
                    image_tile = np.expand_dims(tile_data, axis=0)
                
                # Skip if empty
                if np.max(image_tile) == 0:
                    continue
                
                # Resize to standard tile size if needed
                if image_tile.shape[1] != self.tile_size or image_tile.shape[2] != self.tile_size:
                    resized_tile = np.zeros((band_count, self.tile_size, self.tile_size), dtype=image_tile.dtype)
                    resized_tile[:, :image_tile.shape[1], :image_tile.shape[2]] = image_tile
                    image_tile = resized_tile
                
                # Create dummy label (all zeros for image-only tiling)
                label_data = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
                
                # Create tile key (no H3 for non-geospatial)
                image_name = os.path.basename(image_path)
                source_id = str(uuid.uuid4())
                tile_key = f"non_geo_{client_name}/{image_name}_{source_id}_{row}_{x//self.tile_size}_{y//self.tile_size}"
                
                # Create metadata (no geospatial info)
                metadata = self._create_tile_metadata(
                    image_name=image_name,
                    dtype=str(image_tile.dtype),
                    h3_index="",  # Empty for non-geospatial
                    percentile_dict=percentile_dict,
                    row=row,
                    col=x//self.tile_size,
                    window={'col_off': x, 'row_off': y, 'width': actual_width, 'height': actual_height},
                    band_count=band_count,
                    image_tile_shape=image_tile.shape,
                    label_stats={0: self.tile_size * self.tile_size},
                    source_id=source_id,
                    bounds=None,  # No bounds for non-geospatial
                    crs=None,     # No CRS for non-geospatial
                    resolution=None,  # No resolution for non-geospatial
                    current_date=current_date,
                    label_band_count=1,
                    label_height=self.tile_size,
                    label_width=self.tile_size
                )
                
                # Store in LMDB
                self._store_tile_in_lmdb(env, tile_key, image_tile, label_data, metadata, band_count)
                key_index.append(tile_key)
            
            # Finalize LMDB
            self._finalize_lmdb(env, key_index)
            return len(key_index)
            
        except Exception as e:
            env.close()
            raise e


class NonGeospatialImageLabelLMDBTiler(BaseImageLabelTiler, BaseLMDBTiler):
    """LMDB tiler for non-geospatial image-label pairs"""
    
    def __init__(self, client_name: str = "default", **kwargs):
        BaseImageLabelTiler.__init__(self, **kwargs)
        BaseLMDBTiler.__init__(self, client_name=client_name)

    def tile_directory_pairs(self, img_dir, label_dir, output_dir, **kwargs):
        """
        Override to use tile_image_label_pair_to_lmdb instead of tile_image_label_pair
        """
        # Find image-label pairs
        img_files = list(pathlib.Path(img_dir).rglob("*.png"))
        key_index = []
        
        for img_path in tqdm(img_files):
            label_path = pathlib.Path(label_dir) / img_path.name
            if label_path.exists():
                key_index_img = self.tile_image_label_pair_to_lmdb(img_path, label_path, output_dir)
                key_index.extend(key_index_img)
            else:
                print(f"Warning: No matching label found for {img_path.name}")

        env, _, _, _, _ = self._setup_lmdb_environment(output_dir)
        self._finalize_lmdb(env, key_index)
    
    def tile_image_label_pair_to_lmdb(self, image_path: str, label_path: str, output_dir: str) -> int:
        """
        Tile non-geospatial image-label pair to LMDB.
        
        Parameters
        ----------
        image_path : str
            Path to the image file
        label_path : str
            Path to the label file
        output_dir : str
            Output directory for LMDB
            
        Returns
        -------
        int
            Number of tiles created
        """
        env, key_index, current_date, client_name, output_path = self._setup_lmdb_environment(output_dir)
        
        try:
            # Load images
            image = Image.open(image_path)
            label = Image.open(label_path)
            
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # Verify dimensions match
            if image.size != label.size:
                print(f"Warning: Image and label dimensions don't match for {os.path.basename(image_path)}, skipping...")
                self._finalize_lmdb(env, key_index)
                return 0
            
            # Convert to arrays
            image_array = np.array(image)
            label_array = np.array(label)
            
            height, width = image_array.shape[:2]
            
            # Handle image channels
            if len(image_array.shape) == 3:
                band_count = image_array.shape[2]
                image_data = np.transpose(image_array, (2, 0, 1))
            else:
                band_count = 1
                image_data = np.expand_dims(image_array, axis=0)
            
            # Calculate image percentiles
            percentile_dict = TilingUtils.calculate_image_percentiles(image_data, self.percentiles)
            
            # Calculate tile windows
            windows = TilingUtils.calculate_tile_windows(width, height, self.tile_size, self.overlap, self.min_coverage)
            
            # Process each tile
            for row, (x, y, actual_width, actual_height) in enumerate(windows):
                # Extract image tile
                if len(image_array.shape) == 3:
                    img_tile_data = image_array[y:y+actual_height, x:x+actual_width, :]
                    image_tile = np.transpose(img_tile_data, (2, 0, 1))
                else:
                    img_tile_data = image_array[y:y+actual_height, x:x+actual_width]
                    image_tile = np.expand_dims(img_tile_data, axis=0)
                
                # Extract label tile
                label_tile_data = label_array[y:y+actual_height, x:x+actual_width]
                
                # Skip if image tile is empty
                if np.max(image_tile) == 0:
                    continue
                
                # Process label data
                if self.binary:
                    if len(label_tile_data.shape) == 3:  # RGB label
                        label_data = np.max(label_tile_data > 0, axis=2).astype(np.uint8)
                    else:  # Grayscale label
                        label_data = (label_tile_data > 0).astype(np.uint8)
                else:
                    # Keep original label format
                    if len(label_tile_data.shape) == 3:
                        label_data = label_tile_data[:, :, 0]  # Take first channel
                    else:
                        label_data = label_tile_data
                    label_data = label_data.astype(np.uint8)
                
                # Resize tiles to standard size if needed
                if image_tile.shape[1] != self.tile_size or image_tile.shape[2] != self.tile_size:
                    resized_tile = np.zeros((band_count, self.tile_size, self.tile_size), dtype=image_tile.dtype)
                    resized_tile[:, :image_tile.shape[1], :image_tile.shape[2]] = image_tile
                    image_tile = resized_tile
                
                if label_data.shape[0] != self.tile_size or label_data.shape[1] != self.tile_size:
                    resized_label = np.zeros((self.tile_size, self.tile_size), dtype=label_data.dtype)
                    resized_label[:label_data.shape[0], :label_data.shape[1]] = label_data
                    label_data = resized_label
                
                # Calculate label statistics
                label_stats = TilingUtils.calculate_label_statistics(label_data)
                
                # Create tile key (no H3 for non-geospatial)
                image_name = os.path.basename(image_path)
                source_id = str(uuid.uuid4())
                tile_key = f"non_geo_{client_name}/{image_name}_{source_id}_{row}_{x//self.tile_size}_{y//self.tile_size}"
                
                # Create metadata (no geospatial info)
                metadata = self._create_tile_metadata(
                    image_name=image_name,
                    dtype=str(image_tile.dtype),
                    h3_index="",  # Empty for non-geospatial
                    percentile_dict=percentile_dict,
                    row=row,
                    col=x//self.tile_size,
                    window={'col_off': x, 'row_off': y, 'width': actual_width, 'height': actual_height},
                    band_count=band_count,
                    image_tile_shape=image_tile.shape,
                    label_stats=label_stats,
                    source_id=source_id,
                    bounds=None,  # No bounds for non-geospatial
                    crs=None,     # No CRS for non-geospatial
                    resolution=None,  # No resolution for non-geospatial
                    current_date=current_date,
                    label_band_count=1,
                    label_height=self.tile_size,
                    label_width=self.tile_size
                )
                
                # Store in LMDB
                self._store_tile_in_lmdb(env, tile_key, image_tile, label_data, metadata, band_count)
                key_index.append(tile_key)
            
            # Finalize LMDB
            self._finalize_lmdb(env, key_index)
            return len(key_index)
            
        except Exception as e:
            env.close()
            raise e


# =============================================================================
# JSON Annotation Tiling Classes
# =============================================================================

class JsonAnnotationTiler(BaseTiler):
    """Tiler for aerial (8-bit) images with JSON annotations containing polygon and bbox objects"""
    
    def __init__(self, tile_size: int = 1000, **kwargs):
        """
        Initialize JSON annotation tiler.
        
        Parameters
        ----------
        tile_size : int, optional
            Size of square tiles (default: 1000)
        **kwargs
            Additional arguments passed to BaseTiler
        """
        super().__init__(tile_size=tile_size, **kwargs)
    
    def _calculate_tile_positions(self, dimension_size: int) -> List[Tuple[int, int]]:
        """
        Calculate start/end positions for tiles with 50% merge rule.
        
        If remaining space after a tile is < 50% of tile_size, merge it with current tile.
        
        Parameters
        ----------
        dimension_size : int
            Total dimension size (width or height)
            
        Returns
        -------
        list
            List of (start, end) positions for tiles
        """
        positions = []
        current_pos = 0
        
        while current_pos < dimension_size:
            tile_end = min(current_pos + self.tile_size, dimension_size)
            remaining = dimension_size - tile_end
            
            # If remaining space < 50% of tile_size, include it in current tile
            if remaining > 0 and remaining < (self.tile_size * 0.5):
                tile_end = dimension_size  # Extend current tile to end
            
            positions.append((current_pos, tile_end))
            current_pos = tile_end
        
        return positions
    
    def process_directory(self, img_dir: str, annotation_dir: str, output_dir: str, 
                                 image_extensions: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
        """
        Tile all images and their corresponding JSON annotations.
        
        Parameters
        ----------
        img_dir : str
            Directory containing images
        annotation_dir : str
            Directory containing JSON annotation files
        output_dir : str
            Output directory for tiled data
        image_extensions : list, optional
            Image file extensions to process (default: ['.tif', '.tiff', '.png', '.jpg'])
            
        Returns
        -------
        tuple
            (List of image tile paths, List of JSON annotation paths)
        """
        if image_extensions is None:
            image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        
        # Setup output directories
        img_output_dir, json_output_dir = self._setup_output_dirs(output_dir, create_img_label_dirs=True)
        
        # Find all image files
        img_files = TilingUtils.find_image_files(img_dir, image_extensions)
        
        all_img_tiles = []
        all_json_tiles = []
        
        for img_path in tqdm(img_files, desc="Tiling images with JSON annotations"):
            # Find corresponding JSON file
            img_basename = os.path.splitext(os.path.basename(img_path))[0]
            json_path = os.path.join(annotation_dir, f"{img_basename}.json")
            
            if not os.path.exists(json_path):
                print(f"Warning: No JSON annotation found for {img_basename}, skipping...")
                continue
            
            try:
                img_tiles, json_tiles = self.process_single(img_path, json_path, img_output_dir, json_output_dir)
                all_img_tiles.extend(img_tiles)
                all_json_tiles.extend(json_tiles)
            except Exception as e:
                print(f"Error processing {img_basename}: {e}")
                continue
        
        print(f"Successfully created {len(all_img_tiles)} image tiles and {len(all_json_tiles)} JSON annotations")
        return all_img_tiles, all_json_tiles
    
    def process_single(self, img_path: str, json_path: str, img_output_dir: str, 
                              json_output_dir: str) -> Tuple[List[str], List[str]]:
        """
        Tile a single image and its JSON annotation.
        
        Parameters
        ----------
        img_path : str
            Path to the image file
        json_path : str
            Path to the JSON annotation file
        img_output_dir : str
            Output directory for image tiles
        json_output_dir : str
            Output directory for JSON annotation tiles
            
        Returns
        -------
        tuple
            (List of image tile paths, List of JSON annotation paths)
        """
        # Load JSON annotation
        with open(json_path, 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)
        
        # Extract objects
        objects = annotation_data.get('objects', [])
        image_info = annotation_data.get('image', {})
        
        # Separate polygon and bbox objects
        polygon_objects = [obj for obj in objects if 'polygon' in obj]
        bbox_objects = [obj for obj in objects if 'bbox' in obj]
        
        # Load image to get dimensions
        try:
            # Try loading as raster first (for TIFF)
            with rasterio.open(img_path) as src:
                img_width, img_height = src.width, src.height
                img_data = src.read()
                # Convert to HWC format for PIL
                if img_data.shape[0] <= 4:  # Assume channels first
                    img_array = np.transpose(img_data[:3], (1, 2, 0))  # Take first 3 channels
                else:
                    img_array = img_data
                use_pil = False
        except:
            # Fall back to PIL for other formats
            with Image.open(img_path) as img:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img_array = np.array(img)
                img_height, img_width = img_array.shape[:2]
                use_pil = True
        
        # Get base filename
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Calculate tile positions with 50% merge rule
        x_positions = self._calculate_tile_positions(img_width)
        y_positions = self._calculate_tile_positions(img_height)
        
        img_tile_paths = []
        json_tile_paths = []
        
        # Process each tile
        for tile_row, (y_start, y_end) in enumerate(y_positions):
            for tile_col, (x_start, x_end) in enumerate(x_positions):
                # Calculate actual tile dimensions
                actual_width = x_end - x_start
                actual_height = y_end - y_start
                
                # Extract image tile
                img_tile = img_array[y_start:y_end, x_start:x_end]
                
                # Pad tile to minimum tile_size if needed
                # Tiles can be larger than tile_size (50% merge rule) but never smaller
                final_width = max(actual_width, self.tile_size)
                final_height = max(actual_height, self.tile_size)
                
                if actual_width < self.tile_size or actual_height < self.tile_size:
                    # Need padding to reach minimum tile_size
                    if len(img_tile.shape) == 3:
                        padded_tile = np.zeros((final_height, final_width, img_tile.shape[2]), dtype=img_tile.dtype)
                        padded_tile[:actual_height, :actual_width] = img_tile
                    else:
                        padded_tile = np.zeros((final_height, final_width), dtype=img_tile.dtype)
                        padded_tile[:actual_height, :actual_width] = img_tile
                    img_tile = padded_tile
                    
                    # Update actual dimensions to reflect padding
                    actual_width = final_width
                    actual_height = final_height
                
                # Skip empty tiles
                if np.max(img_tile) == 0:
                    continue
                
                # Create tile bounds for clipping
                tile_bounds = box(x_start, y_start, x_end, y_end)
                
                # Process annotations for this tile
                tile_polygons = self._clip_polygons_to_tile(polygon_objects, tile_bounds, x_start, y_start, actual_width, actual_height)
                tile_bboxes = self._filter_bboxes_for_tile(bbox_objects, tile_bounds, x_start, y_start, actual_width, actual_height)
                
                # Skip tiles with no annotations if desired
                if len(tile_polygons) == 0 and len(tile_bboxes) == 0:
                    # You might want to skip empty annotation tiles or include them
                    # For now, we'll include them for completeness
                    pass
                
                # Save image tile as PNG
                tile_filename = f"{base_filename}_tile_{tile_row}_{tile_col}.png"
                img_tile_path = os.path.join(img_output_dir, tile_filename)
                
                # Convert to PIL Image and save
                if len(img_tile.shape) == 3:
                    tile_img = Image.fromarray(img_tile.astype(np.uint8))
                else:
                    tile_img = Image.fromarray(img_tile.astype(np.uint8), mode='L')
                tile_img.save(img_tile_path)
                
                # Create JSON annotation for this tile
                tile_annotation = {
                    "image": {
                        "id": f"{base_filename}_tile_{tile_row}_{tile_col}",
                        "file_name": tile_filename,
                        "width": actual_width,
                        "height": actual_height
                    },
                    "conventions": {
                        "bbox_format": "xywh",
                        "polygon_space": "pixel"
                    },
                    "objects": tile_polygons + tile_bboxes,
                    "ignore_regions": [],
                    "tile_info": {
                        "source_image": os.path.basename(img_path),
                        "tile_row": tile_row,
                        "tile_col": tile_col,
                        "x_offset": x_start,
                        "y_offset": y_start,
                        "original_width": img_width,
                        "original_height": img_height
                    }
                }
                
                # Add spatial reference if available from original
                if "spatial_ref" in annotation_data:
                    tile_annotation["spatial_ref"] = annotation_data["spatial_ref"]
                
                # Save JSON annotation
                json_tile_filename = f"{base_filename}_tile_{tile_row}_{tile_col}.json"
                json_tile_path = os.path.join(json_output_dir, json_tile_filename)
                
                with open(json_tile_path, 'w', encoding='utf-8') as f:
                    json.dump(tile_annotation, f, indent=2)
                
                img_tile_paths.append(img_tile_path)
                json_tile_paths.append(json_tile_path)
        
        return img_tile_paths, json_tile_paths
    
    def _clip_polygons_to_tile(self, polygon_objects: List[Dict], tile_bounds: box, 
                              x_offset: int, y_offset: int, tile_width: int, tile_height: int) -> List[Dict]:
        """
        Clip polygon objects to tile boundaries and transform coordinates.
        
        Parameters
        ----------
        polygon_objects : list
            List of polygon objects from JSON
        tile_bounds : shapely.geometry.box
            Tile boundary box
        x_offset : int
            Tile x offset in original image
        y_offset : int
            Tile y offset in original image
        tile_width : int
            Actual width of the tile
        tile_height : int
            Actual height of the tile
            
        Returns
        -------
        list
            List of clipped polygon objects with tile-local coordinates
        """
        clipped_polygons = []
        
        for obj in polygon_objects:
            try:
                # Get polygon data
                polygon_data = obj.get('polygon', {})
                exterior = polygon_data.get('exterior', [])
                holes = polygon_data.get('holes', [])
                
                if len(exterior) < 3:
                    continue
                
                # Create shapely polygon
                exterior_coords = [(pt[0], pt[1]) for pt in exterior]
                hole_coords = [[(pt[0], pt[1]) for pt in hole] for hole in holes if len(hole) >= 3]
                
                try:
                    original_polygon = Polygon(exterior_coords, hole_coords)
                except:
                    # If polygon creation fails, skip
                    continue
                
                # Check if polygon intersects with tile
                if not original_polygon.intersects(tile_bounds):
                    continue
                
                # Clip polygon to tile bounds
                clipped = original_polygon.intersection(tile_bounds)
                
                # Skip if clipping resulted in empty geometry
                if clipped.is_empty:
                    continue
                
                # Handle different geometry types from clipping
                clipped_geoms = []
                if clipped.geom_type == 'Polygon':
                    clipped_geoms = [clipped]
                elif clipped.geom_type == 'MultiPolygon':
                    clipped_geoms = list(clipped.geoms)
                elif clipped.geom_type == 'GeometryCollection':
                    # Extract only polygons from the collection
                    clipped_geoms = [geom for geom in clipped.geoms 
                                   if geom.geom_type in ['Polygon', 'MultiPolygon']]
                    # Flatten MultiPolygons
                    flattened_geoms = []
                    for geom in clipped_geoms:
                        if geom.geom_type == 'Polygon':
                            flattened_geoms.append(geom)
                        elif geom.geom_type == 'MultiPolygon':
                            flattened_geoms.extend(list(geom.geoms))
                    clipped_geoms = flattened_geoms
                else:
                    continue  # Skip other geometry types
                
                # Process each clipped polygon
                for geom in clipped_geoms:
                    if geom.area < 1.0:  # Skip very small polygons
                        continue
                    
                    # Transform coordinates to tile-local space
                    exterior_local = [[pt[0] - x_offset, pt[1] - y_offset] for pt in geom.exterior.coords[:-1]]  # Remove last duplicate point
                    holes_local = []
                    
                    for interior in geom.interiors:
                        hole_local = [[pt[0] - x_offset, pt[1] - y_offset] for pt in interior.coords[:-1]]
                        holes_local.append(hole_local)
                    
                    # Close the polygon if not already closed
                    if exterior_local[0] != exterior_local[-1]:
                        exterior_local.append(exterior_local[0])
                    
                    for hole in holes_local:
                        if hole[0] != hole[-1]:
                            hole.append(hole[0])
                    
                    # Create new polygon object
                    new_polygon_obj = obj.copy()
                    new_polygon_obj['polygon'] = {
                        "type": "Polygon",
                        "exterior": exterior_local,
                        "holes": holes_local
                    }
                    
                    clipped_polygons.append(new_polygon_obj)
                    
            except Exception as e:
                print(f"Error processing polygon: {e}")
                continue
        
        return clipped_polygons
    
    def _filter_bboxes_for_tile(self, bbox_objects: List[Dict], tile_bounds: box, 
                               x_offset: int, y_offset: int, tile_width: int, tile_height: int) -> List[Dict]:
        """
        Filter and transform bbox objects for tile.
        
        Parameters
        ----------
        bbox_objects : list
            List of bbox objects from JSON
        tile_bounds : shapely.geometry.box
            Tile boundary box
        x_offset : int
            Tile x offset in original image
        y_offset : int
            Tile y offset in original image
        tile_width : int
            Actual width of the tile
        tile_height : int
            Actual height of the tile
            
        Returns
        -------
        list
            List of bbox objects with tile-local coordinates
        """
        tile_bboxes = []
        
        for obj in bbox_objects:
            try:
                # Get bbox coordinates [x, y, width, height]
                bbox = obj.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                x, y, w, h = bbox
                
                # Create bbox geometry
                bbox_geom = box(x, y, x + w, y + h)
                
                # Check if bbox intersects with tile
                if not bbox_geom.intersects(tile_bounds):
                    continue
                
                # Clip bbox to tile bounds
                clipped_bbox = bbox_geom.intersection(tile_bounds)
                
                if clipped_bbox.is_empty or clipped_bbox.area < 1.0:
                    continue
                
                # Get clipped bounds
                minx, miny, maxx, maxy = clipped_bbox.bounds
                
                # Transform to tile-local coordinates
                local_x = max(0, minx - x_offset)
                local_y = max(0, miny - y_offset)
                local_w = min(tile_width - local_x, maxx - minx)
                local_h = min(tile_height - local_y, maxy - miny)
                
                # Skip very small bboxes
                if local_w < 1 or local_h < 1:
                    continue
                
                # Create new bbox object
                new_bbox_obj = obj.copy()
                new_bbox_obj['bbox'] = [float(local_x), float(local_y), float(local_w), float(local_h)]
                
                tile_bboxes.append(new_bbox_obj)
                
            except Exception as e:
                print(f"Error processing bbox: {e}")
                continue
        
        return tile_bboxes


class JsonAnnotationTilerSeparateFolders(JsonAnnotationTiler):
    """Tiler for aerial (8-bit) images with JSON annotations from separate bbox and polygon folders"""
    
    def __init__(self, bbox_folder: str = "bbox", segmentation_folder: str = "segmentation", 
                 bbox: bool = True, segmentation: bool = True, **kwargs):
        """
        Initialize JSON annotation tiler with separate folders.
        
        Parameters
        ----------
        bbox_folder : str, default="bbox"
            Name of the folder containing bbox JSON files
        segmentation_folder : str, default="segmentation"
            Name of the folder containing segmentation JSON files
        bbox : bool, default=True
            Whether to load bbox annotations. If False, only polygon data will be processed.
        segmentation : bool, default=True
            Whether to load segmentation annotations. If False, only bbox data will be processed.
        **kwargs
            Additional arguments passed to JsonAnnotationTiler
        """
        super().__init__(**kwargs)
        self.bbox_folder = bbox_folder
        self.segmentation_folder = segmentation_folder
        self.load_bbox = bbox
        self.load_segmentation = segmentation
        
        # Validate that at least one type is requested
        if not bbox and not segmentation:
            raise ValueError("At least one of 'bbox' or 'segmentation' must be True")
    
    def process_directory(self, img_dir: str, annotation_dir: str, output_dir: str, 
                         image_extensions: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
        """
        Tile all images and their corresponding JSON annotations from separate folders.
        
        Parameters
        ----------
        img_dir : str
            Directory containing images
        annotation_dir : str
            Root directory containing bbox and segmentation subfolders
        output_dir : str
            Output directory for tiled data
        image_extensions : list, optional
            Image file extensions to process (default: ['.tif', '.tiff', '.png', '.jpg'])
            
        Returns
        -------
        tuple
            (List of image tile paths, List of JSON annotation paths)
        """
        if image_extensions is None:
            image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        
        # Setup output directories
        img_output_dir = os.path.join(output_dir, "img")
        os.makedirs(img_output_dir, exist_ok=True)
        
        # Find all image files
        img_files = TilingUtils.find_image_files(img_dir, image_extensions)
        
        all_img_tiles = []
        all_json_tiles = []
        
        for img_path in tqdm(img_files, desc="Tiling images with separate JSON annotations"):
            # Find corresponding annotation files
            img_basename = os.path.splitext(os.path.basename(img_path))[0]
            
            # Build paths to annotation files
            segmentation_path = os.path.join(annotation_dir, self.segmentation_folder, f"{img_basename}.json") if self.load_segmentation else None
            bbox_path = os.path.join(annotation_dir, self.bbox_folder, f"{img_basename}.json") if self.load_bbox else None
            
            # Check if required files exist
            files_exist = True
            
            if self.load_segmentation and not os.path.exists(segmentation_path):
                print(f"Warning: No segmentation annotation found for {img_basename}, skipping...")
                files_exist = False
            if self.load_bbox and not os.path.exists(bbox_path):
                print(f"Warning: No bbox annotation found for {img_basename}, skipping...")
                files_exist = False
            
            if not files_exist:
                continue
            
            try:
                img_tiles, json_tiles = self.process_single_separate_files(
                    img_path, segmentation_path, bbox_path, output_dir
                )
                all_img_tiles.extend(img_tiles)
                all_json_tiles.extend(json_tiles)
            except Exception as e:
                print(f"Error processing {img_basename}: {e}")
                continue
        
        # Log what was processed
        load_types = []
        if self.load_segmentation:
            load_types.append("segmentation")
        if self.load_bbox:
            load_types.append("bbox")
        
        print(f"Successfully created {len(all_img_tiles)} image tiles and {len(all_json_tiles)} JSON annotations")
        print(f"Processed data types: {'-'.join(load_types)}")
        return all_img_tiles, all_json_tiles
    
    def process_single_separate_files(self, img_path: str, segmentation_path: Optional[str], 
                                    bbox_path: Optional[str], output_dir: str) -> Tuple[List[str], List[str]]:
        """
        Tile a single image with separate segmentation and bbox JSON files.
        
        Parameters
        ----------
        img_path : str
            Path to the image file
        segmentation_path : str or None
            Path to the segmentation JSON annotation file
        bbox_path : str or None
            Path to the bbox JSON annotation file
        output_dir : str
            Root output directory (img, bbox, and segmentation subdirectories will be created here)
            
        Returns
        -------
        tuple
            (List of image tile paths, List of JSON annotation paths)
        """
        # Create output directories
        img_output_dir = os.path.join(output_dir, "img")
        os.makedirs(img_output_dir, exist_ok=True)
        
        # Load annotation data from separate files
        polygon_objects = []
        bbox_objects = []
        image_info = {}
        spatial_ref = None
        
        # Load segmentation data if requested
        if self.load_segmentation and segmentation_path is not None:
            with open(segmentation_path, 'r', encoding='utf-8') as f:
                segmentation_data = json.load(f)
            
            # Extract polygon objects
            seg_objects = segmentation_data.get('objects', [])
            polygon_objects = [obj for obj in seg_objects if 'polygon' in obj]
            
            # Get image info from segmentation file
            image_info = segmentation_data.get('image', {})
            spatial_ref = segmentation_data.get('spatial_ref')
        
        # Load bbox data if requested
        if self.load_bbox and bbox_path is not None:
            with open(bbox_path, 'r', encoding='utf-8') as f:
                bbox_data = json.load(f)
            
            # Extract bbox objects
            bbox_obj_list = bbox_data.get('objects', [])
            bbox_objects = [obj for obj in bbox_obj_list if 'bbox' in obj]
            
            # Get image info from bbox file if not already available
            if not image_info:
                image_info = bbox_data.get('image', {})
            if spatial_ref is None:
                spatial_ref = bbox_data.get('spatial_ref')
        
        # Load image to get dimensions
        try:
            # Try loading as raster first (for TIFF)
            with rasterio.open(img_path) as src:
                img_width, img_height = src.width, src.height
                img_data = src.read()
                # Convert to HWC format for PIL
                if img_data.shape[0] <= 4:  # Assume channels first
                    img_array = np.transpose(img_data[:3], (1, 2, 0))  # Take first 3 channels
                else:
                    img_array = img_data
                use_pil = False
        except:
            # Fall back to PIL for other formats
            with Image.open(img_path) as img:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img_array = np.array(img)
                img_height, img_width = img_array.shape[:2]
                use_pil = True
        
        # Get base filename
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Calculate tile positions with 50% merge rule
        x_positions = self._calculate_tile_positions(img_width)
        y_positions = self._calculate_tile_positions(img_height)
        
        img_tile_paths = []
        json_tile_paths = []
        
        # Process each tile
        for tile_row, (y_start, y_end) in enumerate(y_positions):
            for tile_col, (x_start, x_end) in enumerate(x_positions):
                # Calculate actual tile dimensions
                actual_width = x_end - x_start
                actual_height = y_end - y_start
                
                # Extract image tile
                img_tile = img_array[y_start:y_end, x_start:x_end]
                
                # Pad tile to minimum tile_size if needed
                # Tiles can be larger than tile_size (50% merge rule) but never smaller
                final_width = max(actual_width, self.tile_size)
                final_height = max(actual_height, self.tile_size)
                
                if actual_width < self.tile_size or actual_height < self.tile_size:
                    # Need padding to reach minimum tile_size
                    if len(img_tile.shape) == 3:
                        padded_tile = np.zeros((final_height, final_width, img_tile.shape[2]), dtype=img_tile.dtype)
                        padded_tile[:actual_height, :actual_width] = img_tile
                    else:
                        padded_tile = np.zeros((final_height, final_width), dtype=img_tile.dtype)
                        padded_tile[:actual_height, :actual_width] = img_tile
                    img_tile = padded_tile
                    
                    # Update actual dimensions to reflect padding
                    actual_width = final_width
                    actual_height = final_height
                
                # Skip empty tiles
                if np.max(img_tile) == 0:
                    continue
                
                # Create tile bounds for clipping
                tile_bounds = box(x_start, y_start, x_end, y_end)
                
                # Process annotations for this tile based on what's loaded
                tile_polygons = []
                tile_bboxes = []
                
                if self.load_segmentation:
                    tile_polygons = self._clip_polygons_to_tile(polygon_objects, tile_bounds, x_start, y_start, actual_width, actual_height)
                
                if self.load_bbox:
                    tile_bboxes = self._filter_bboxes_for_tile(bbox_objects, tile_bounds, x_start, y_start, actual_width, actual_height)
                
                # Skip tiles with no annotations if desired
                if len(tile_polygons) == 0 and len(tile_bboxes) == 0:
                    # You might want to skip empty annotation tiles or include them
                    # For now, we'll include them for completeness
                    pass
                
                # Save image tile as PNG
                tile_filename = f"{base_filename}_tile_{tile_row}_{tile_col}.png"
                img_tile_path = os.path.join(img_output_dir, tile_filename)
                
                # Convert to PIL Image and save
                if len(img_tile.shape) == 3:
                    tile_img = Image.fromarray(img_tile.astype(np.uint8))
                else:
                    tile_img = Image.fromarray(img_tile.astype(np.uint8), mode='L')
                tile_img.save(img_tile_path)
                
                # Create JSON annotation for this tile
                tile_annotation = {
                    "image": {
                        "id": f"{base_filename}_tile_{tile_row}_{tile_col}",
                        "file_name": tile_filename,
                        "width": actual_width,
                        "height": actual_height
                    },
                    "conventions": {
                        "bbox_format": "xywh",
                        "polygon_space": "pixel"
                    },
                    "objects": tile_polygons + tile_bboxes,
                    "ignore_regions": [],
                    "tile_info": {
                        "source_image": os.path.basename(img_path),
                        "tile_row": tile_row,
                        "tile_col": tile_col,
                        "x_offset": x_start,
                        "y_offset": y_start,
                        "original_width": img_width,
                        "original_height": img_height
                    }
                }
                
                # Add spatial reference if available
                if spatial_ref is not None:
                    tile_annotation["spatial_ref"] = spatial_ref
                
                # Save JSON annotations separately for bbox and segmentation
                json_tile_filename = f"{base_filename}_tile_{tile_row}_{tile_col}.json"
                
                # Save segmentation annotation if requested
                if self.load_segmentation:
                    segmentation_annotation = tile_annotation.copy()
                    segmentation_annotation["objects"] = tile_polygons
                    segmentation_json_dir = os.path.join(output_dir, self.segmentation_folder)
                    os.makedirs(segmentation_json_dir, exist_ok=True)
                    segmentation_json_path = os.path.join(segmentation_json_dir, json_tile_filename)
                    
                    with open(segmentation_json_path, 'w', encoding='utf-8') as f:
                        json.dump(segmentation_annotation, f, indent=2)
                    json_tile_paths.append(segmentation_json_path)
                
                # Save bbox annotation if requested
                if self.load_bbox:
                    bbox_annotation = tile_annotation.copy()
                    bbox_annotation["objects"] = tile_bboxes
                    bbox_json_dir = os.path.join(output_dir, self.bbox_folder)
                    os.makedirs(bbox_json_dir, exist_ok=True)
                    bbox_json_path = os.path.join(bbox_json_dir, json_tile_filename)
                    
                    with open(bbox_json_path, 'w', encoding='utf-8') as f:
                        json.dump(bbox_annotation, f, indent=2)
                    json_tile_paths.append(bbox_json_path)
                
                img_tile_paths.append(img_tile_path)
        
        return img_tile_paths, json_tile_paths


# =============================================================================
# Example Usage and Factory Functions
# =============================================================================

def create_tiler(tiler_type: str, **kwargs):
    """
    Factory function to create appropriate tiler based on type.
    
    Parameters
    ----------
    tiler_type : str
        Type of tiler to create. Options:
        - 'geospatial': GeospatialTiler
        - 'geospatial_image_label': GeospatialImageLabelTiler
        - 'geospatial_shapefile': GeospatialShapefileTiler
        - 'geospatial_buffer': GeospatialBufferTiler
        - 'geospatial_lmdb': GeospatialLMDBTiler
        - 'geospatial_image_label_lmdb': GeospatialImageLabelLMDBTiler
        - 'geospatial_shapefile_lmdb': GeospatialShapefileLMDBTiler
        - 'geospatial_buffer_lmdb': GeospatialBufferLMDBTiler
        - 'non_geospatial': NonGeospatialTiler
        - 'non_geospatial_image_label': NonGeospatialImageLabelTiler
        - 'non_geospatial_instance': NonGeospatialInstanceTiler
        - 'non_geospatial_lmdb': NonGeospatialLMDBTiler
        - 'non_geospatial_image_label_lmdb': NonGeospatialImageLabelLMDBTiler
        - 'json_annotation': JsonAnnotationTiler
        - 'json_annotation_separate_folders': JsonAnnotationTilerSeparateFolders
    **kwargs
        Additional arguments passed to the tiler constructor
        
    Returns
    -------
    BaseTiler
        Appropriate tiler instance
    """
    tiler_map = {
        'geospatial': GeospatialTiler,
        'geospatial_image_label': GeospatialImageLabelTiler,
        'geospatial_shapefile': GeospatialShapefileTiler,
        'geospatial_buffer': GeospatialBufferTiler,
        'geospatial_lmdb': GeospatialLMDBTiler,
        'geospatial_image_label_lmdb': GeospatialImageLabelLMDBTiler,
        'geospatial_shapefile_lmdb': GeospatialShapefileLMDBTiler,
        'geospatial_buffer_lmdb': GeospatialBufferLMDBTiler,
        'non_geospatial': NonGeospatialTiler,
        'non_geospatial_image_label': NonGeospatialImageLabelTiler,
        'non_geospatial_instance': NonGeospatialInstanceTiler,
        'non_geospatial_lmdb': NonGeospatialLMDBTiler,
        'non_geospatial_image_label_lmdb': NonGeospatialImageLabelLMDBTiler,
        'json_annotation': JsonAnnotationTiler,
        'json_annotation_separate_folders': JsonAnnotationTilerSeparateFolders,
    }
    
    if tiler_type not in tiler_map:
        raise ValueError(f"Unknown tiler type: {tiler_type}. Available types: {list(tiler_map.keys())}")
    
    return tiler_map[tiler_type](**kwargs)


if __name__ == "__main__":
    tiler = GeospatialTiler(tile_size=2048)
    input_dir = "/home/prod-gpu-3/prod-gpu-3/nfs/node_data/th/dominion_images"
    output_dir = "/home/prod-gpu-3/prod-gpu-3/nfs/node_data/th/dominion_images_2048"
    tiler.tile_directory(input_dir, output_dir)