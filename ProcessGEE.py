# -*- coding: utf-8 -*-
"""
Created on Tue Mar 4 10:30:00 2025

@authors: Gregory A. Greene, Dong Zhao
"""

import os
import glob
import ee
import geemap
import geopandas as gpd

import ProcessRasters as pr
import gc
from typing import Optional


### LAND COVER DATA
def _get_dynworld_col(aoi, start_date, end_date):
    """
    Retrieve the Dynamic World land cover dataset for the specified AOI and time range.

    :param aoi: The area of interest as an Earth Engine geometry.
    :param start_date: List of start dates in YYYY-MM-DD format.
    :param end_date: List of end dates in YYYY-MM-DD format.
    :return: An Earth Engine ImageCollection.
    """
    landcover_col = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
                     .filterBounds(aoi)
                     .filterDate(start_date[0], end_date[0]))

    for i in range(1, len(start_date)):
        landcover_col1 = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
                          .filterBounds(aoi)
                          .filterDate(start_date[i], end_date[i]))
        landcover_col = landcover_col.merge(landcover_col1)

    return landcover_col


def _get_aoi(shp_path) -> tuple:
    """
    Load AOI from a shapefile and return bounding box, Earth Engine shape, and geometry.

    :param shp_path: Path to the shapefile.
    :return: Tuple containing bounding box, Earth Engine shape, and geometry.
    """
    ee_shp = geemap.shp_to_ee(shp_path)
    ee_geometry = ee_shp.geometry()
    shp_gdf = gpd.read_file(shp_path)
    bbox = shp_gdf.total_bounds  # (xmin, ymin, xmax, ymax)

    return bbox, ee_shp, ee_geometry


def get_dynworld_landcover(ee_project_id: str,
                           aoi_shp_path: str,
                           start_date: list,
                           end_date: list,
                           out_folder: str,
                           bands: Optional[list] = None,
                           out_epsg: Optional[int] = 4326,
                           out_res: float = 10,
                           prob_type: Optional[str] = None) -> None:
    """
    Downloads and processes Dynamic World Land Cover data from Google Earth Engine (GEE).

    This function:
        - Retrieves Dynamic World land cover classification data for a given AOI and date range.
        - Computes the most frequently occurring land cover class (mode) across images.
        - Optionally downloads probability statistics (mean or median) for selected land cover classes.
        - Outputs data as GeoTIFFs in the specified coordinate system and resolution.
        - Handles large AOIs by tiling and mosaicking datasets.

    :param ee_project_id: Google Earth Engine project ID for authentication.
    :param aoi_shp_path: Path to the shapefile defining the area of interest (AOI).
    :param start_date: List of start dates (YYYY-MM-DD) for imagery retrieval.
    :param end_date: List of end dates (YYYY-MM-DD) corresponding to the start dates.
    :param out_folder: Directory where output land cover rasters will be saved.
    :param bands: List of land cover probability bands to download. If None, all bands are used.
        Options: ['water', 'trees', 'grass', 'flooded_vegetation', 'crops',
                  'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']
    :param out_epsg: EPSG code for the output projection. Default: 4326 (WGS 84).
    :param out_res: Spatial resolution of the output data in units matching `out_epsg`. Default: 10 meters.
    :param prob_type: Type of probability statistic to compute and download.
        Options: 'mean', 'median'. If None, probability data is not downloaded.
    :return: None. Processed land cover data is saved as GeoTIFF files.

    ### Processing Steps:
        1. **Authenticate & Initialize GEE**: Logs into the Google Earth Engine environment.
        2. **AOI & Image Collection**:
           - Reads the AOI from the shapefile and determines its bounding box.
           - Fetches Dynamic World land cover images within the specified date range.
        3. **Land Cover Classification**:
           - Computes the most frequent land cover class using a mode reducer.
           - Downloads the classification map as a GeoTIFF.
        4. **Handling Large AOIs**:
           - If the AOI exceeds 1° in any dimension, the function tiles the area into smaller sections.
           - Tiles are downloaded separately and mosaicked into a single raster.
        5. **Downloading Probability Data (if specified)**:
           - Computes the mean or median probability for selected land cover types.
           - Downloads probability rasters for each band as separate GeoTIFFs.

    ### Output Structure:
        - **Land cover classification**:
          - Small AOIs: `landcover.tif`
          - Large AOIs: Tiled images merged into `landcover.tif`
        - **Land cover probability data** (if `prob_type` is specified):
          - Small AOIs: `landcover_prob_{band}.tif`
          - Large AOIs: Tiled images merged into probability datasets.

    ### Example Usage:
        get_dynworld_landcover(\n
            ee_project_id="your_project_id",\n
            aoi_shp_path="path/to/aoi.shp",\n
            start_date=["2023-06-01"],\n
            end_date=["2023-08-31"],\n
            out_folder="output_directory",\n
            bands=['trees', 'grass', 'built'],\n
            out_epsg=32610,  # UTM Zone 10N\n
            out_res=30,  # 30-meter resolution\n
            prob_type='mean'\n
        )\n
    """
    ee.Authenticate()
    ee.Initialize(project=ee_project_id)

    # Set the output EPSG
    dst_crs = f'EPSG:{out_epsg}'

    # Set bands if None
    if bands is None:
        bands = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub',
                 'built', 'bare', 'snow_and_ice']

    # Get the AOI shapefile bounding box, earth engine shape, and earth engine geometry
    bbox, ee_shp, ee_geometry = _get_aoi(shp_path=aoi_shp_path)
    ymin, xmin, ymax, xmax = bbox
    ee_shp = ee_geometry

    # Check if the AOI is large (greater than 1 degree in X or Y direction)
    large_aoi = False
    if (abs((ymax - ymin)) > 1) or (abs((xmax - xmin)) > 1):
        large_aoi = True

    # Build the data
    landcover_col = _get_dynworld_col(ee_geometry, start_date, end_date)

    # Print a list of images in the collection
    print('Images in the Collection:\n\t', landcover_col.aggregate_array('system:index').getInfo())

    #get the most frequently occuring class label (using mode) based on example code from here
    #(https://developers.google.com/earth-engine/tutorials/community/introduction-to-dynamic-world-pt-1)
    classification = landcover_col.select('label')
    dw_composite = classification.reduce(ee.Reducer.mode())

    # Get the land cover data
    if large_aoi:
        # Download landcover class
        tile_dir = os.path.join(out_folder, 'temp_tiles')
        if not os.path.exists(tile_dir):
            os.makedirs(tile_dir)
        tile_features = geemap.fishnet(ee_shp, rows=2, cols=2)
        geemap.download_ee_image_tiles(image=dw_composite,
                                       features=tile_features,
                                       out_dir=tile_dir,
                                       prefix='landcover_',
                                       crs=dst_crs,
                                       scale=out_res,
                                       dtype='int8',
                                       num_threads=os.cpu_count())

        # Mosaic tiles into single dataset
        mosaic_list = glob.glob(os.path.join(tile_dir, '*.tif'))
        mosaic_out = os.path.join(out_folder, 'landcover.tif')
        mosaic_ras = pr.mosaicRasters(mosaic_list=mosaic_list,
                                      out_file=mosaic_out,
                                      extent_mode='trim')

        # Delete temporary data
        mosaic_ras.close()
        del mosaic_ras
        gc.collect()
        for path in mosaic_list:
            os.remove(path)
        os.rmdir(tile_dir)

    else:
        # Download landcover class (one raster)
        geemap.download_ee_image(image=dw_composite,
                                 filename=os.path.join(out_folder, 'landcover.tif'),
                                 region=ee_geometry,
                                 crs=dst_crs,
                                 scale=out_res,
                                 dtype='int8',
                                 num_threads=os.cpu_count())

    if prob_type is not None:
        # Get the path to the probability output folder
        out_dir = os.path.join(out_folder, 'probability')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # Build probability request
        probs = landcover_col.select(bands)
        if prob_type == 'mean':
            probability = probs.reduce(ee.Reducer.mean())
        else:
            probability = probs.reduce(ee.Reducer.median())
        probs_multiband = probability.select(bands)

        if large_aoi:
            # Download probability class (LARGE AOI)
            tile_features = geemap.fishnet(ee_shp, rows=2, cols=2)
            for band in bands:
                geemap.download_ee_image_tiles(probs_multiband.select(band),
                                               features=tile_features,
                                               out_dir=out_dir,
                                               prefix=f'landcover_{band}_',
                                               scale=out_res,
                                               crs=dst_crs)

        else:
            # Download probability class (SMALL AOI)
            for band in bands:
                geemap.download_ee_image(probs_multiband.select(band),
                                         filename=os.path.join(out_dir, f'landcover_prob_{band}.tif'),
                                         region=ee_geometry,
                                         scale=out_res,
                                         crs=dst_crs)

    return


### SENTINEL 2 DATA
def _get_s2_sr_cld_col(aoi, start_date, end_date, cloud_filter):
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(aoi)
                 .filterDate(start_date[0], end_date[0])
                 .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_filter)))
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                        .filterBounds(aoi)
                        .filterDate(start_date[0], end_date[0]))
    s2_sr_cloudless_col = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))
    if len(start_date) != 1:
        for i in range(1, len(start_date)):
            s2_sr_col1 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                          .filterBounds(aoi)
                          .filterDate(start_date[i], end_date[i])
                          .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_filter)))
            s2_cloudless_col1 = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                                 .filterBounds(aoi)
                                 .filterDate(start_date[i], end_date[i]))
            s2_sr_cloudless_col1 = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
                'primary': s2_sr_col1,
                'secondary': s2_cloudless_col1,
                'condition': ee.Filter.equals(**{
                    'leftField': 'system:index',
                    'rightField': 'system:index'
                })
            }))
            s2_sr_cloudless_col = s2_sr_cloudless_col.merge(s2_sr_cloudless_col1)
    ##########
    return s2_sr_cloudless_col


def _add_cloud_bands(img, cloud_prob_thres):
    """Adds cloud probability and cloud mask bands to a Sentinel-2 image."""

    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(cloud_prob_thres).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(cld_prb).addBands(is_cloud)


def _add_cloud_bands_to_collection(img_col, cloud_prob_thres):
    """Maps the _add_cloud_bands function over an ImageCollection."""
    return img_col.map(lambda img: _add_cloud_bands(img, cloud_prob_thres))


def _add_shadow_bands(img, nir_dark_thresh, cloud_proj_dist):  # cloud shadow components
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(nir_dark_thresh * SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

    # Project shadows from clouds for the distance specified by the cloud_proj_dist input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, cloud_proj_dist * 10)
                .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
                .select('distance')
                .mask()
                .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


def _add_cld_shdw_mask(img, buffer, cloud_prob_thresh, nir_dark_thresh, cloud_proj_dist):
    """Adds cloud and cloud shadow bands to an image."""

    # Add cloud component bands
    img_cloud = _add_cloud_bands(img, cloud_prob_thresh)

    # Add cloud shadow component bands.
    img_cloud_shadow = _add_shadow_bands(img_cloud, nir_dark_thresh, cloud_proj_dist)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(buffer * 2 / 20)
                   .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
                   .rename('cloudmask'))

    return img_cloud_shadow.addBands(is_cld_shdw)


def _add_cld_shdw_mask_to_collection(img_col, buffer, cloud_prob_thresh, nir_dark_thresh, cloud_proj_dist):
    """Maps the _add_cld_shdw_mask function over an ImageCollection."""
    return img_col.map(lambda img: _add_cld_shdw_mask(img, buffer, cloud_prob_thresh, nir_dark_thresh, cloud_proj_dist))


def _apply_cld_shdw_mask(img):
    """Applies the cloud and shadow mask to a Sentinel-2 image."""

    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks.
    return img.select('B.*').updateMask(not_cld_shdw)


def _sort_by_substring_order(file_paths: list[str], order_list: list[str]) -> list[str]:
    """
    Sort file paths based on the order of substrings in another list.

    :param file_paths: List of file paths.
    :param order_list: List of substrings defining the order.
    :return: Sorted list of file paths.
    """
    order_dict = {substring: index for index, substring in enumerate(order_list)}

    def get_order(file_path: str) -> int:
        for substring, index in order_dict.items():
            if substring in file_path:
                return index
        return len(order_list)  # Default to last if no match

    return sorted(file_paths, key=get_order)


def get_sentinel2(ee_project_id: str,
                  aoi_shp_path: str,
                  start_date: list,
                  end_date: list,
                  out_folder: str,
                  bands: Optional[list] = None,
                  out_epsg: Optional[int] = 4326,
                  out_res: float = 0.0001,
                  cloud_filter: int = 15,
                  cloud_prob_thresh: int = 40,
                  nir_dark_thresh: float = 0.15,
                  cloud_proj_dist: int = 2,
                  buffer: int = 0) -> None:
    """
    Retrieves and processes Sentinel-2 imagery from Google Earth Engine (GEE) with cloud and shadow masking.

    This function:
        - Retrieves Sentinel-2 surface reflectance data within a specified AOI and date range.
        - Applies cloud and shadow masking using Sentinel-2 cloud probability and dark pixel thresholding.
        - Generates a median composite of the filtered images.
        - Downloads the processed dataset as GeoTIFFs, either as a single raster or tiled outputs if the AOI is large.
        - Stacks tiled outputs into multiband raster datasets when needed.

    :param ee_project_id: Google Earth Engine project ID for authentication.
    :param aoi_shp_path: Path to the shapefile defining the area of interest (AOI).
    :param start_date: Start date for image retrieval (list format: [YYYY, MM, DD]).
    :param end_date: End date for image retrieval (list format: [YYYY, MM, DD]).
    :param out_folder: Path to the output directory for storing downloaded images.
    :param bands: List of Sentinel-2 bands to retrieve. Defaults to ['B2', 'B3', 'B4', 'B8', 'B12'].
    :param out_epsg: EPSG code for the output projection. Defaults to EPSG:4326.
    :param out_res: Output spatial resolution in degrees. Defaults to 0.0001 (~10m at the equator).
    :param cloud_filter: Maximum allowable cloud coverage percentage for image selection. Defaults to 15%.
    :param cloud_prob_thresh: Cloud probability threshold for masking clouds. Defaults to 40.
    :param nir_dark_thresh: NIR dark pixel threshold for shadow detection. Defaults to 0.15.
    :param cloud_proj_dist: Maximum cloud projection distance (km) for shadow masking. Defaults to 2 km.
    :param buffer: Buffer distance (in pixels) applied to cloud and shadow masks. Defaults to 0 (no buffer).
    :return: None. Processed Sentinel-2 images are saved as GeoTIFFs in the specified output folder.

    ### Processing Steps:
        1. **Authenticate & Initialize GEE**: Logs into the Google Earth Engine environment.
        2. **AOI & Image Collection**:
            - Reads the AOI from the shapefile.
            - Filters Sentinel-2 images by date range and cloud coverage.
        3. **Cloud & Shadow Masking**:
           - Adds cloud probability and shadow bands.
           - Applies a cloud-shadow mask to the dataset.
           - Reduces the dataset to a median composite image.
        4. **Image Validation**:
           - Checks if valid bands exist after processing.
        5. **Downloading & Saving**:
           - If AOI is large (>1° in width or height), the dataset is split into tiles and merged into multiband rasters.
           - Otherwise, individual band images are downloaded directly.
           - Final outputs are saved in the specified projection and resolution.

    ### Output Structure:
        - **For small AOIs**: Single-band images are saved as `sentinel2_B{band}.tif`.
        - **For large AOIs**: Tiled images are stacked and saved as `sentinel2_tileX.tif`, where `X` is the tile index.

    ### Example Usage:
        get_sentinel2(
            ee_project_id="your_project_id",\n
            aoi_shp_path="path/to/aoi.shp",\n
            start_date=[2023, 6, 1],\n
            end_date=[2023, 8, 31],\n
            out_folder="output_directory",\n
            bands=['B2', 'B3', 'B4', 'B8'],\n
            out_epsg=32610,  # UTM WGS84 Zone 10N\n
            out_res=10,  # 10-meter resolution\n
            cloud_filter=20,\n
            cloud_prob_thresh=50,\n
            nir_dark_thresh=0.2,\n
            cloud_proj_dist=3,\n
            buffer=1\n
        )
    """
    ee.Authenticate()
    ee.Initialize(project=ee_project_id)

    dst_crs = f'EPSG:{out_epsg}'

    if bands is None:
        bands = ['B2', 'B3', 'B4', 'B8', 'B12']

    bbox, ee_shp, ee_geometry = _get_aoi(shp_path=aoi_shp_path)
    ymin, xmin, ymax, xmax = bbox
    large_aoi = (abs((ymax - ymin)) > 1) or (abs((xmax - xmin)) > 1)

    s2_sr_cld_col = _get_s2_sr_cld_col(ee_geometry, start_date, end_date, cloud_filter)

    # Ensure collection is not empty before proceeding
    num_images = s2_sr_cld_col.size().getInfo()
    if num_images == 0:
        raise ValueError(
            'No Sentinel-2 images found for the given parameters. Check AOI, date range, and cloud filter.')

    print(f'Number of images found: {num_images}')

    # Process the collection
    s2_sr_cld_col = _add_cloud_bands_to_collection(s2_sr_cld_col, cloud_prob_thresh)
    s2_sr_cld_col = _add_cld_shdw_mask_to_collection(s2_sr_cld_col, buffer, cloud_prob_thresh, nir_dark_thresh,
                                                    cloud_proj_dist)

    # Reduce with median
    s2_sr_median = s2_sr_cld_col.map(_apply_cld_shdw_mask).reduce(ee.Reducer.median())

    # Ensure the final image has bands before proceeding
    if s2_sr_median.bandNames().size().getInfo() == 0:
        raise ValueError('Processed Sentinel-2 image has no valid bands. Adjust filtering or cloud masking.')

    # Get available band names
    available_bands = s2_sr_median.bandNames().getInfo()
    print(f'Available bands in final image: {available_bands}')

    if large_aoi:
        tile_dir = os.path.join(out_folder, 'temp_tiles')
        os.makedirs(tile_dir, exist_ok=True)
        tile_features = geemap.fishnet(ee_shp, rows=2, cols=2)

        for band in bands:
            band_median = f'{band}_median'
            if band_median not in available_bands:
                print(f'Skipping {band} as {band_median} is not available.')
                continue

            print(f'Downloading band: {band_median}')
            geemap.download_ee_image_tiles(image=s2_sr_median.select(band_median),
                                           features=tile_features,
                                           out_dir=tile_dir,
                                           prefix=f'sentinel2_{band}_',
                                           scale=out_res,
                                           crs=dst_crs)

        # Stacking tiles into a single dataset
        print('Merging Sentinel 2 tiles into stacked datasets')
        tile_list = glob.glob(os.path.join(tile_dir, f'sentinel2_{bands[0]}_*.tif'))

        for i in range(1, len(tile_list) + 1):
            print(f'\tProcessing tile {i}')
            stack_list = glob.glob(os.path.join(tile_dir, f'*_{i}.tif'))
            stack_list = _sort_by_substring_order(file_paths=stack_list, order_list=bands)
            out_file = os.path.join(out_folder, f'sentinel2_tile{i}.tif')
            band_names = [band.replace('B', 'Band_') for band in bands]
            stack_ras = pr.toMultiband(path_list=stack_list, out_file=out_file, band_names=band_names)
            stack_ras.close()
            del stack_ras

            # Cleanup temp files
            for path in stack_list:
                os.remove(path)

        # Cleanup temp directory
        os.rmdir(tile_dir)

    else:
        for band in bands:
            band_median = f'{band}_median'
            if band_median not in available_bands:
                print(f'Skipping {band} as {band_median} is not available.')
                continue

            print(f'Downloading band: {band_median}')
            geemap.download_ee_image(s2_sr_median.select(band_median),
                                     filename=os.path.join(out_folder, f'sentinel2_{band}.tif'),
                                     region=ee_geometry,
                                     scale=out_res,
                                     crs=dst_crs)

    return
