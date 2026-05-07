import rasterio
from abc import abstractmethod, ABC
from rasterio.windows import Window
import numpy as np
import os
import logging
from tqdm import tqdm
from rasterio import features
from shapely.geometry import Polygon
from rasterio.windows import Window
from rasterio.transform import from_origin
import geopandas as gpd 
import rasterio.merge
import glob
import shapely
from multiprocessing import Pool

TEMP_DATA_PATH = "temp/session"

# Configure logging
logger = logging.getLogger(__name__)


class DatasetCreator(ABC):
    def __init__(self, file_path: str = None, batch_size: int = 1, image_height: int = 256, image_width: int = 256, channels: int = 3):
        """ Dataset creater for a raster file

        Parameters
        ----------
        file_path : str
            File path of the tiff file, local or s3
        batch_size : int, optional
            _description_, by default 1
        image_height : int, optional
            _description_, by default 256
        image_width : int, optional
            _description_, by default 256
        channels : int, optional
            _description_, by default 3
        """
        self.batch_size = int(batch_size)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.file_path = file_path
        self.channels = channels

        self.labels = None
        self.img_list = None

    def _load_image(self, path, channels=3):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(
            image, channels=channels, dtype=np.float32, expand_animations=False)
        return image

    def add_diff_layer(self, image):
        r_channel = image[0, :, :]
        g_channel = image[1, :, :]
        diff_layer = r_channel-g_channel
        diff_layer = tf.expand_dims(diff_layer, 0)
        image = tf.concat([image, diff_layer], 0)
        return image

    @abstractmethod
    def _load_labeled_data(self, path):
        pass

    def iterator(self):
        return iter(self.loaded_dataset)

    def get_batch(self):
        return next(iter(self.loaded_dataset))

    def write_results(self, idx, img):
        pass

    def close_files(self):
        pass

    @abstractmethod
    def _load_labeled_data(self, path):
        pass

    @abstractmethod
    def load_process(self, config, csv_file, shuffle_size=1000):
        pass

def load_rasters(path, channels):
    raster = None
    try: raster = rasterio.open(str(path, 'utf-8'))
    except: raster = rasterio.open(str(path))
    image = raster.read(range(1, channels+1)).astype(np.float32)
    image = np.transpose(image, [1, 2, 0])
    # t_99 = np.percentile(image[:,:,0:3], 99).astype(np.float32)
    # img_max = np.max(image[:,:,0:3])
    # if(t_99>EPSILON):
    #     image = image/(t_99)
    # else:
    # image = image/(2**12-1)
    image = normalize_img(image, channels)
    # image[image >= 1.0] = 1.0
    # image = ((image.astype(np.uint8))/255.0).astype(np.float32)
    return image.astype(np.float32)

# def normalize_img(img, channels = 3):
#     image = np.log(img*0.005 + 1)
#     mask = img[...,0]!=0
#     mean_max = np.max(np.median(image[mask], axis = 0)[0:channels])
#     per_70 = np.max(np.percentile(image[mask], 70, axis = 0)[0:channels])
#     # image_exp = np.exp(((image- NORM_PERCENTILES[0:channels, 0]) / NORM_PERCENTILES[0:channels, 1])*5- 1)
#     if(per_70<1e-6):
#         return img
#     image_exp = np.exp(((image- mean_max) / per_70)*5- 1)
#     return image_exp/(image_exp+1)

class RasterDataset(DatasetCreator):
    def _load_labeled_data(self, path):
        pass

    def _load_data(self, path):
        img = tf.numpy_function(load_raster_img, [path, self.channels, self.bgr], tf.float32)
        img = tf.numpy_function(NORMALIZATION_FUN[self.normalize], [img, self.channels], tf.float32)
        return img

    def init_result_raster(self, write_dir = TEMP_DATA_PATH):
        result_raster_path = write_dir+'/results.tif'
        create_result_raster(self.file_path, result_raster_path)
        self.result_raster_path = result_raster_path
    
    def tile_rasters(self, write_dir = TEMP_DATA_PATH, overlap = 0.3):
        if(is_s3_path(self.file_path)):
            file_path = 'temp/session/file.tif'
            download_file_s3(self.file_path, file_path)
        else:
            file_path = self.file_path
        img_list_file_path = tile_rasters(file_path, self.image_height, self.image_height, write_dir, overlap=overlap)
        with open(img_list_file_path) as f:
            img_list = f.readlines()
            img_list = list(map(lambda x: x.strip(), img_list))
        self.img_list = img_list
    
    def process_image(self, img):
        # if(self.resize is not None):
        #     shape = tf.shape(img)
        #     img = tf.reshape(img, [shape[0], shape[1], self.channels])
        #     img = tf.image.resize(img, [self.resize, self.resize])
        img = img*2-1
        return img

    def load_process(self, folder_path = None, augment=False, shuffle_data = False, normalize = 'lnadaptive', resize_to = None, bgr = False):
        self.resize = resize_to
        self.normalize = normalize
        self.bgr = bgr
        if(self.img_list is None):
            assert folder_path is not None, "At least one of file path that is to be tiled or a folder path that has tiled images needs to be given"
            self.img_list = glob.glob(os.path.join(folder_path,"*.tif"), 
                   recursive = False)
            print("Scanned folder {}, found {} files".format(folder_path, str(len(self.img_list))))

        img_list = self.img_list
        self.raster_tiles = list(map(lambda x: rasterio.open(x), img_list))
        

        self.num = len(img_list)
        self.dataset = tf.data.Dataset.from_tensor_slices(
            img_list)
        self.img_list = img_list
        self.beta_array = np.array([ 1.0, 1.0, 5.553436, 0.8, 1.466958, 3.860696], dtype = np.float32)

        self.loaded_dataset = self.dataset.map(
            self._load_data, num_parallel_calls=AUTOTUNE)
        self.loaded_dataset = self.loaded_dataset.map(
            self.process_image, num_parallel_calls=AUTOTUNE)
        self.loaded_dataset = self.loaded_dataset.batch(self.batch_size)
        self.loaded_dataset = self.loaded_dataset.prefetch(
            buffer_size=AUTOTUNE)
    
    def write_results(self, idx, img, index):
        assert self.result_raster_path is not None, "Result raster not initalized"
        write_tiles(self.result_raster_path, self.raster_tiles[idx], img, index)


def create_result_raster(base_raster_path, result_raster_path, n_classes = 4):
    """ Create a result raster file

    Parameters
    ----------
    base_raster : str
        Base raster file path that has the template for creating result raster
    result_raster_path : str
        path of the result raster file

    Returns
    -------
    _type_
        _description_
    """
    raster = rasterio.open(base_raster_path)
    raster_profile = (raster.meta.copy())
    raster_profile.update({"count":n_classes})
    result_raster = rasterio.open(result_raster_path, 'w', **raster_profile)
    result_raster.close()
    print("Result raster created at: {}".format(result_raster_path))

def write_img_to_raster(img, raster_meta, dest_path, transformation = None, write_mode = 'w', dtype = None):
    """ Writes cropped image to raster using the meta information from the source raster to
    destination path

    Parameters
    ----------
    img : np.ndarray
        Image that will be written
    transformation : affine function
        image to raster transformation function
    source_raster : rasterio raster
        Source raster whose meta will be used to create the destination raster
    dest_path : _type_
        _description_
    """
    out_meta = raster_meta.copy()
    if(transformation is None):
        transformation = raster_meta['transform']
    if dtype is None:
        dtype = out_meta['dtype']

    out_meta.update({"driver": "GTiff",
                    "count": img.shape[0],
                     "height": img.shape[1],
                     "width": img.shape[2],
                     "dtype": dtype,
                     "transform": transformation})
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with rasterio.open(dest_path, write_mode, **out_meta) as dest:
        dest.write(img)

def crop_raster(raster, region, channels = 3):
    raser_cropped, t = rasterio.mask.mask(raster, [region], crop=True)
    raser_cropped = raser_cropped[0:channels]
    write_img_to_raster(raser_cropped, raster.meta, "temp/region.tif", t)
    region_raster = rasterio.open("temp/region.tif")
    region_img = region_raster.read()
    region_img = np.transpose(region_img, [1, 2, 0])
    # t_95 = np.percentile(region_img[:,:,0:3], 99)
    # print(t_95, region_img[:,:,0:3].max())
    # t_95 = 2**12-1
    # region_img = region_img/(t_95)*255
    # region_img[region_img > 255] = 255
    return region_img, region_raster

def split_images_with_overlap(raster, tile_height = 1024, tile_width = 1024, overlap = 0.5, dest_folder = "tmp/img_tiles", dest_file = "img_tiles.txt", file_name_label = '', compression = 'lzw'):
    filenames = []
    overlap_height = int(overlap*tile_height)
    overlap_width = int(overlap*tile_width)
    img_height, img_width = raster.height, raster.width

    img_meta = raster.meta.copy()

    img_meta.update({"driver": "GTiff",
                     "nodata": 0,
                     "height": tile_height,
                     "width": tile_width})
    
    tile_area = tile_width*tile_height

    for y_idx in tqdm(range(0, img_height, tile_height-overlap_height)):
        for x_idx in range(0, img_width, tile_width-overlap_width):
            current_tile_area = (min(tile_height, img_height-y_idx))*(min(tile_width, img_width-x_idx))
            if((current_tile_area/tile_area)<0.6):
                continue
            window = Window(x_idx, y_idx, tile_width, tile_height)
            window_transform = raster.window_transform(window)

            w = raster.read(window=window, boundless=True, fill_value = 0)
            if (raster.nodata != 0):
                w[w == raster.nodata] = 0
            
            if(w.max()==0): continue
            
            img_meta.update({
                     "transform": window_transform,
                     "height": w.shape[1],
                     "width": w.shape[2],
                     "compression": compression})


            # Saving raster tiles
            file_path = "{}/img_{}_{}_{}.tif".format(
                dest_folder, file_name_label, str(y_idx), str(x_idx))
            filenames.append(file_path)
            with rasterio.open(file_path, "w", **img_meta) as dest:
                dest.write(w)

    with open(dest_folder+"/"+dest_file, "a+") as outfile:
        outfile.write("\n".join(filenames))


def save_tile(args):
    src_path, x_start, y_start, tile_width, tile_height, output_dir, tile_id = args
    with rasterio.open(src_path) as src:
        window = Window(x_start, y_start, tile_width, tile_height)
        
        transform = src.window_transform(window)
        meta = src.meta.copy()
        meta.update({
            'driver': 'GTiff',
            'height': tile_height,
            'width': tile_width,
            'transform': transform
        })
        
        tile_data = src.read(window=window, boundless=True, fill_value = 0)
        # tile_uuid = uuid.uuid4()
        output_path = os.path.join(output_dir, f"{tile_id}_{x_start}_{y_start}.tif")
        if(tile_data.max() <= 0): return output_path
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(tile_data)
    return output_path

def tile_raster(src_path, output_dir, tile_size=512):
    os.makedirs(output_dir, exist_ok = True)
    with rasterio.open(src_path) as src:
        width = src.width
        height = src.height
        
        tile_width = tile_size
        tile_height = tile_size
        
        x_offsets = range(0, width, tile_width)
        y_offsets = range(0, height, tile_height)

        filename = os.path.basename(src_path).split(".")[0]
        
        tasks = [(src_path, x, y, tile_width, tile_height, output_dir, filename) 
                 for x in x_offsets for y in y_offsets]
        
        with Pool() as pool:
            result = pool.map(save_tile, tasks)
    return result



# def tile_rasters(raster_path, tile_width:int, tile_height:int, write_dir:str, img_file="img.txt", overlap = 0.3, channel_count = 3, write_path_to_file = True, filename = None, normalize = None, **normalize_args):
#     """ Tile bitemporal rasters

#     Parameters
#     ----------
#     raster1 : str
#          Path to raster 1 projected in the format "EPSG:4326"
#     raster2 : str
#         Path to raster 2 projected in the format "EPSG:4326"
#     tile_width : int
#         Tile width
#     tile_height : int
#         Tile height
#     write_dir : str
#         Path to the directory where tiles are written
#     """
#     raster = rasterio.open(raster_path)
#     if filename is None: filename = os.path.splitext(os.path.basename(raster_path))[0]
#     img_meta = raster.meta.copy()

#     img_meta.update({"driver": "GTiff",
#                      "nodata": 0,
#                      "height": tile_height,
#                      "width": tile_width})
    
#     if(normalize is not None):
#         img_meta.update({"dtype": "float32",
#                          "count": channel_count,})

#     os.makedirs(write_dir, exist_ok = True)

#     overlap_height = int(overlap*tile_height)
#     overlap_width = int(overlap*tile_width)

#     img_height, img_width = raster.height, raster.width
#     tile_locations1 = []
#     print("Tiling rasters")
#     for y_idx in tqdm(range(0, img_height, tile_height-overlap_height)):
#         for x_idx in range(0, img_width, tile_width-overlap_width):
#             # Windowed tiling in raster 1
#             window = Window(x_idx, y_idx, tile_width, tile_height)
#             window_transform = raster.window_transform(window)
#             img_meta.update({"transform": window_transform})
            

#             w = raster.read(window=window, boundless=True, fill_value = 0)
#             if (raster.nodata != 0):
#                 w[w == raster.nodata] = 0

#             if((w.max()==0)):
#                 continue
#             if(normalize is not None):
#                 w = np.transpose(w, [1,2,0])
#                 w = NORMALIZATION_FUN[normalize](w[...,0:channel_count], **normalize_args)
#                 w = np.transpose(w, [2,0,1])
#             # Saving raster tiles
#             file_path1 = "{}/{}_tile_{}_{}.tif".format(
#                 write_dir, filename, str(y_idx), str(x_idx))
#             tile_locations1.append(file_path1)
#             with rasterio.open(file_path1, "w", **img_meta) as dest:
#                 dest.write(w)

#     if(write_path_to_file):
#         with open(write_dir+"/"+img_file, "w") as outfile:
#             outfile.write("\n".join(tile_locations1))

#     return write_dir+"/"+img_file


def merging_tiff(res_tiffs, save_path):
    """Merge TIFF files using rasterio merge - transforms are now normalized at tile creation."""
    if len(res_tiffs) == 0:
        logger.warning("No tiles provided for merging")
        return
    
    # Use rasterio merge with consistent transforms
    rasterio.merge.merge(res_tiffs, dst_path=save_path, method='max')
    logger.info(f"Successfully merged {len(res_tiffs)} tiles using rasterio.merge")

def get_raster_geom(raster):
    img = raster.read()
    raster_mask = np.any(img, axis = 0).astype(np.uint8)
    shapes = features.shapes(raster_mask, transform=raster.transform)
    geom = []
    values = []
    for g, v in shapes:
        values.append(v)
        geom.append(g)
    print(values)
    values = np.array(values)
    geom = np.array(geom)
    predicted_array = values ==1.0  
    geom = geom[predicted_array]
    values = values[predicted_array]
    # geom_poly = list(map(lambda x: Polygon(x['coordinates'][0]), geom))
    geom_poly = list(map(lambda x: shapely.geometry.shape(x), geom))
    columns = {"values": values}
    gdf = gpd.GeoDataFrame(columns, crs= str(raster.crs), geometry = geom_poly)
    return gdf

def generate_shapefile_from_mask(raster_mask_path, mask_index, mode = 'w', label_path = 'temp/session/result_shp'):
    try:
        raster = rasterio.open(raster_mask_path)
    except:
        return None
    raster_mask = raster.read(mask_index).astype(np.uint8)
    shapes = features.shapes(raster_mask, transform=raster.transform)
    geom = []
    values = []
    for g, v in shapes:
        values.append(v)
        geom.append(g)
    values = np.array(values)
    geom = np.array(geom)
    predicted_array = values > 0
    geom = geom[predicted_array]
    values = values[predicted_array]
    geom_poly = list(map(lambda x: shapely.geometry.shape(x), geom))
    columns = {"values": values}
    
    # Handle missing CRS: use None instead of string "None" to create CRS-less shapefile
    crs = raster.crs if raster.crs is not None else None
    if crs is None:
        logger.warning(f"Raster {raster_mask_path} has no CRS information. Creating shapefile without CRS.")
    
    gdf = gpd.GeoDataFrame(columns, crs=crs, geometry = geom_poly)
    os.makedirs(label_path, exist_ok=True)
    label_file_path = label_path+"/result.shp"
    if(not os.path.exists(label_file_path)): mode = 'w'
    if(len(columns['values'])>0):
        gdf.to_file(label_file_path, mode = mode)
    return gdf

def reproject_rasters(raster_path, dest_raster, dst_crs):
    """ Reproject a raster to destination crs

    Parameters
    ----------
    raster_path : str
        Path of the source raster
    dest_raster : str
        Path of the destination raster
    dst_crs : str
        crs of the form 'EPSG:4326'. Source raster is projected to this raster
    """
    from osgeo import gdal
    if(str(rasterio.open(raster_path).crs) != dst_crs):
        print("Reprojecting raster to {}".format(dst_crs))
        gdal.Warp(dest_raster,raster_path,dstSRS=dst_crs)
    else:
        print("Source raster already in the destination crs format. No reprojection needed")
    # print("Reprojecting raster to {}".format(dst_crs))
    # src = rasterio.open(raster)
    # transform, width, height = calculate_default_transform(
    #         src.crs, dst_crs, src.width, src.height, *src.bounds)
    # kwargs = src.meta.copy()
    # kwargs.update({
    #     'crs': dst_crs,
    #     'transform': transform,
    #     'width': width,
    #     'height': height
    # })

    # with rasterio.open(dest_raster, 'w+', **kwargs) as dst:
    #     for i in range(1, src.count + 1):
    #         reproject(
    #             source=rasterio.band(src, i),
    #             destination=rasterio.band(dst, i),
    #             src_transform=src.transform,
    #             src_crs=src.crs,
    #             dst_transform=transform,
    #             dst_crs=dst_crs,
    #             resampling=Resampling.nearest)

def gen_labels_for_tiles(tile_folder_path, shapefile, label_type = None, output_dir = "temp/labels", output_file_name = "labels.txt"):
    """ Generate labels using shapefiles and tiles

    Parameters
    ----------
    tile_folder_path : str
        Path of the folder containing the rasters
    shapefile : str
        Path of the shapefile
    output_dir : str, optional
        _description_, by default "temp/labels"
    output_file_name: str, optional
        Name of the file containing the list of labels, by default "labels.txt"
    """
    gdf = gpd.read_file(shapefile)
    if(label_type is not None):
        gdf = gdf[gdf[label_type['label_id']].isin(label_type['value'])]
    file_names = os.listdir(tile_folder_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    label_locations = []
    for file_name in file_names:
        raster = rasterio.open(os.path.join(tile_folder_path, file_name))
        dest_meta = raster.meta.copy()
        
        raster_mask, _, _ = rasterio.mask.raster_geometry_mask(raster, gdf.geometry, invert = True)
        dest_meta.update({"count":1})
        file_path = output_dir+"/"+file_name
        label_locations.append(file_path)
        with rasterio.open(file_path, "w", **dest_meta) as dest:
                dest.write(raster_mask, indexes = 1)
    
    with open(output_dir+"/"+output_file_name, "w") as outfile:
        outfile.write("\n".join(label_locations))

def cust_merge_method(merged_data, new_data, merged_mask, new_mask, **kwargs):
    """Returns the first available pixel."""
    mask = np.empty_like(merged_mask, dtype="bool")
    np.logical_not(new_mask, out=mask)
    np.logical_and(merged_mask, mask, out=mask)
    mask[:,:, :20] = False
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")

def merging_tiff(res_tiffs, save_path):
    """Merge TIFF files using rasterio merge - transforms are now normalized at tile creation."""
    if len(res_tiffs) == 0:
        logger.warning("No tiles provided for merging")
        return 0
    
    # Use rasterio merge with consistent transforms
    rasterio.merge.merge(res_tiffs, dst_path=save_path, method='max')
    logger.info(f"Successfully merged {len(res_tiffs)} tiles using rasterio.merge")
    return 1

def write_tiles(raster_path, tile, img, indexes = 1):
    raster = rasterio.open(raster_path, 'r+')
    tile_bounds = tile.bounds
    tile_window = raster.window(*tile_bounds)
    try:
        raster.write(np.squeeze(img), window = tile_window, indexes = indexes)
    except Exception as e:
        print(e)
    raster.close()
