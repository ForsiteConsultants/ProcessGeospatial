# -*- coding: utf-8 -*-
"""
Created on Tue Mar 4 10:30:00 2025

@authors: Gregory A. Greene, Dong Zhao
"""

import os
import glob
import ee
import geemap
import os
import geopandas as gpd
import ProcessRasters as pr
from typing import Optional


### LAND COVER DATA
def _get_landcover_col(aoi, start_date, end_date):
    """
    Retrieve the Dynamic World land cover dataset for the specified AOI and time range.

    :param aoi: The area of interest as an Earth Engine geometry.
    :param start_date: List of start dates in YYYY-MM-DD format.
    :param end_date: List of end dates in YYYY-MM-DD format.
    :return: An Earth Engine ImageCollection.
    """
    landcover_col = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                     .filterBounds(aoi)
                     .filterDate(start_date[0], end_date[0]))

    for i in range(1, len(start_date)):
        landcover_col1 = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
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


def get_landcover(ee_project_id: str,
                  aoi_shp_path: str,
                  start_date: str,
                  end_date: str,
                  out_folder: str,
                  bands: Optional[list] = None,
                  out_epsg: Optional[int] = 4326,
                  out_res: float = None,
                  prob_type: Optional[str] = None) -> None:
    """
    Function to download Google Earth Engine Land Cover data.

    :param ee_project_id: The id of the Google Earth Engine project to use.
    :param aoi_shp_path: Path to a shapefile to use as an AOI (area of interest) for data download.
    :param start_date: The start date of imagery to process. Must be a string formatted as YYYY-MM-DD.
    :param end_date: The end date of imagery to process. Must be a string formatted as YYYY-MM-DD.
    :param bands: A list of strings containing the land cover bands to download. If None, all bands are downloaded.
        Options: ['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub',
                 'built', 'bare', 'snow_and_ice']
    :param out_epsg: The EPSG to apply to the output data.
        Default: 4326 (Geographic WGS84)
    :param out_res: The resolution of the output data. Should be in units that match out_epsg.
    :param out_folder: The folder to save the output data.
    :param prob_type: The type of probability statistics to calculate and download.
        If None, no probability data are downloaded. Options: 'mean', 'median'.
    :return: None
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

    # Check if the AOI is large (greater than 1 degree in X or Y direction)
    large_aoi = False
    if (abs((ymax - ymin)) > 1) or (abs((xmax - xmin)) > 1):
        large_aoi = True

    # Build the data
    landcover_col = _get_landcover_col(ee_geometry, [start_date], [end_date])

    # Print a list of images in the collection
    print('Images in the Collection: /n',landcover_col.aggregate_array('system:index').getInfo())

    #get the most frequently occuring class label based on example code from here
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
        geemap.download_ee_image_tiles(dw_composite,
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
        for path in mosaic_list:
            os.remove(path)
        os.remove(tile_dir)
        mosaic_ras.close()
        del mosaic_ras

    else:
        # Download landcover class (one raster)
        geemap.download_ee_image(dw_composite,
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


def _add_cloud_bands(img, cloud_prob_thres):  # cloud components
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(cloud_prob_thres).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))


def _add_shadow_bands(img, nir_dark_thresh, cloud_proj_dist):  # cloud shadow components
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(nir_dark_thresh * SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

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


def _add_cld_shdw_mask(img, buffer, cloud_prob_thresh, nir_dark_thresh, cloud_proj_dist):  # cloud shadow mask
    # Add cloud component bands
    img_cloud = _add_cloud_bands(img, cloud_prob_thresh)

    # Add cloud shadow component bands.
    img_cloud_shadow = _add_shadow_bands(img_cloud, nir_dark_thresh, cloud_proj_dist)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(buffer * 2 / 20)
                   .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
                   .rename('cloudmask'))

    # Include only the final cloud/cloud shadow mask along with the original image bands
    # return img.addBands(is_cld_shdw)

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)


# Define a function to apply the cloud mask to each image in the collection.
def _apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)

def get_sentinel2(ee_project_name: str,
                  aoi_shp_path: str,
                  start_date: str,
                  end_date: str,
                  out_folder: str,
                  bands: Optional[list] = None,
                  out_epsg: Optional[int] = 4326,
                  out_res: float = 0.0001,
                  cloud_filter: int = 15,
                  cloud_prob_thresh: int = 40,
                  nir_dark_thresh: float = 0.15,
                  cloud_proj_dist: int = 2,
                  buffer: int = 0):
    """
    Function to get Sentinel 2 data from Google Earth Engine.

    :param ee_project_name: The name of the Google Earth Engine project to use.
    :param aoi_shp_path: Path to a shapefile to use as an AOI (area of interest) for data download.
    :param start_date: The start date of imagery to process. Must be a string formatted as 'YYYY-MM-DD'.
    :param end_date: The end date of imagery to process. Must be a string formatted as 'YYYY-MM-DD'.
    :param out_folder: The folder to save the output data.
    :param bands: A list of strings containing the land cover bands to download. If None, all bands are downloaded.
        Options: ['B2', 'B3', 'B4', 'B8', 'B12']
    :param out_epsg: The EPSG to apply to the output data.
        Default: 4326 (Geographic WGS84)
    :param out_res: The resolution of the output data. Should be in units that match out_epsg.
        Default: 0.0001 (~11m at the equator)
    :param cloud_filter:
    :param cloud_prob_thresh:
    :param nir_dark_thresh:
    :param cloud_proj_dist:
    :param buffer:
    :return: None
    """
    ee.Authenticate()
    ee.Initialize(project=ee_project_name)

    # Set the output EPSG
    dst_crs = f'EPSG:{out_epsg}'

    # Set bands if None
    if bands is None:
        bands = ['B2', 'B3', 'B4', 'B8', 'B12']

    # Get the Earth Engine equivalent of the AOI shapefile bounding box, the shape itself, and the geometry
    bbox, ee_shp, ee_geometry = _get_aoi(shp_path=aoi_shp_path)
    ymin, xmin, ymax, xmax = bbox

    # Check if the AOI is large (greater than 1 degree in X or Y direction)
    large_aoi = False
    if (abs((ymax - ymin)) > 1) or (abs((xmax - xmin)) > 1):
        large_aoi = True

    # Build an image collection
    s2_sr_cld_col_eval = _get_s2_sr_cld_col(ee_geometry, start_date, end_date, cloud_filter)

    # Print the number of images in the collection
    print('%d images found.'%(len(s2_sr_cld_col_eval.aggregate_array('system:index').getInfo())))

    # Build cloud free data
    # Apply cloud and cloud-shadow masking, and use median method to calculate reflectances of the missing pixels
    s2_sr_median = (s2_sr_cld_col_eval.map(_add_cld_shdw_mask(s2_sr_cld_col_eval, buffer, cloud_prob_thresh,
                                                              nir_dark_thresh, cloud_proj_dist))
                    .map(_apply_cld_shdw_mask)
                    .median())

    if large_aoi:
        # Download images (LARGE AOI)
        tile_dir = os.path.join(out_folder, 'temp_tiles')
        if not os.path.exists(tile_dir):
            os.makedirs(tile_dir)
        tileFeatures = geemap.fishnet(ee_shp, rows=2, cols=2)
        for band in bands:
            geemap.download_ee_image_tiles(s2_sr_median.select(band), tileFeatures,
                                           out_dir=tile_dir,
                                           prefix=f'sentinel2_{band}_',
                                           scale=out_res,
                                           crs=dst_crs)

        # Mosaic tiles into single dataset
        mosaic_list = glob.glob(os.path.join(tile_dir, '*.tif'))
        mosaic_out = os.path.join(out_folder, 'landcover.tif')
        mosaic_ras = pr.mosaicRasters(mosaic_list=mosaic_list,
                                      out_file=mosaic_out,
                                      extent_mode='Union')

        # Delete temporary data
        for path in mosaic_list:
            os.remove(path)
        os.remove(tile_dir)
        mosaic_ras.close()
        del mosaic_ras
    else:
        # Download images (SMALL AOI)
        for band in bands:
            geemap.download_ee_image(s2_sr_median.select(band),
                                     filename=os.path.join(out_folder, f'sentinel2_{band}.tif'),
                                     region=ee_geometry,
                                     scale=out_res,
                                     crs=dst_crs)
