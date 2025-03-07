# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 13:25:52 2024

@author: Gregory A. Greene
"""

import os
import numpy as np
import fiona
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import rasterio as rio
from rasterio.mask import mask
from rasterio.crs import CRS
from rasterio.features import shapes, geometry_window, geometry_mask, rasterize
from rasterio.merge import merge
from rasterio.transform import xy, from_origin, from_bounds, rowcol
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
# from rasterio.enums import Resampling
# from rasterio.io import MemoryFile
from rasterio.transform import Affine
from pyproj import Transformer
from shapely.geometry import Point, box, shape, mapping
from shapely.affinity import translate
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from typing import Union, Optional
import warnings

try:
    from rasterio import shutil
except ImportError:
    import shutil


def _calculate_cosine_incidence(slope, aspect, zenith, azimuth):
    """
    Calculate the cosine of the solar incidence angle.

    :param slope: Slope in radians.
    :param aspect: Aspect in radians.
    :param zenith: Solar zenith angle in radians.
    :param azimuth: Solar azimuth angle in radians.
    :return: Cosine of the solar incidence angle.
    """
    return (
        np.sin(zenith) * np.cos(slope)
        + np.cos(zenith) * np.sin(slope) * np.cos(azimuth - aspect)
    )


def _process_block(block: np.ndarray,
                   transform: rio.io.DatasetReaderBase,
                   window: rio.io.WindowMethodsMixin) -> list:
    """
    Process a block of the raster array and return shapes.

    :param block:
    :param transform:
    :param window:
    :return: list of shapes
    """
    data = block
    # Calculate the translation values for the window
    dx, dy = transform * (window[1][0], window[0][0])
    # Adjust the translation values to account for the original transform's translation
    dx -= transform.c
    dy -= transform.f
    # Convert the block data into shapes and adjust the coordinates
    return [(translate(shape(s), xoff=dx, yoff=dy), v) for s, v in shapes(data, transform=transform)]


def arrayToRaster(array: np.ndarray,
                  out_file: str,
                  ras_profile: dict,
                  nodata_val: Optional[Union[int, float]] = None,
                  dtype: np.dtype = np.float32) -> rio.DatasetReader:
    """
    Function to convert a numpy array to a raster.

    :param array: input numpy array
    :param out_file: path (with name) to save output raster
    :param ras_profile: profile of reference rasterio dataset reader object
    :param nodata_val: new integer or floating point value to assign as "no data" (default = None)
    :param dtype: the numpy data type of new raster
    :return: rasterio dataset reader object in r+ mode
    """
    # Get profile and verify array shape matches reference raster profile
    profile = ras_profile.copy()

    # Check if the dimensions match (height, width, bands)
    required_shape = (profile['count'], profile['height'], profile['width'])
    if array.shape != required_shape:
        raise ValueError(f'Array shape {array.shape} does not match required shape {required_shape}')

    # Update profile
    profile.update(
        compress='lzw'  # Specify LZW compression
    )
    if nodata_val:
        profile.update(
            nodata=nodata_val  # Specify nodata value
        )
    if dtype:
        profile.update(
            dtype=dtype
        )

    # Create new raster file
    with rio.open(out_file, 'w', **profile) as dst:
        # Write data to new raster
        dst.write(array)
        # Calculate new statistics
        calculateStatistics(dst)

    # Return new raster as "readonly" rasterio openfile object
    return rio.open(out_file, 'r+')


def asciiToTiff(ascii_path: str,
                out_file: str,
                out_crs: str = 'EPSG:4326') -> rio.DatasetReader:
    """
    Function to convert a TIFF to an ASCII file.

    :param ascii_path:  path to ASCII dataset
    :param out_file: path (with name) to save TIF file
    :param out_crs: string defining new projection (e.g., 'EPSG:4326')
    :return: new rasterio dataset reader object in 'r+' mode
    """
    # Read the ASCII file
    with open(ascii_path, 'r') as f:
        # Skip header lines and get metadata
        header = {}
        for _ in range(6):
            key, value = f.readline().strip().split()
            header[key] = float(value)

        # Read the raster data
        data = np.loadtxt(f)

    # Extract metadata
    ncols = int(header['ncols'])
    nrows = int(header['nrows'])
    xllcorner = header['xllcorner']
    yllcorner = header['yllcorner']
    cellsize = header['cellsize']
    nodata_value = header['NODATA_value']

    # Define the transformation
    # Adjust the yllcorner to the top left y coordinate
    yllcorner_top = yllcorner + nrows * cellsize
    transform = from_origin(xllcorner, yllcorner_top, cellsize, cellsize)

    # Write the data to a GeoTIFF file
    with rio.open(out_file,
                  mode='w',
                  driver='GTiff',
                  height=nrows,
                  width=ncols,
                  count=1,
                  dtype=data.dtype,
                  crs=out_crs,
                  transform=transform,
                  nodata=nodata_value
                  ) as dst:
        dst.write(data, 1)

    return rio.open(out_file, 'r+')


def calcSolarRadiation(
    slope: np.ndarray,
    aspect: np.ndarray,
    elevation: np.ndarray,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    out_file: str,
    transform: Affine,
    time_step: int = 60,
    nodata: float = -9999
):
    """
    Calculate solar radiation using pvlib for a given raster grid.
    <<<<NOTE: THIS IS EXPERIMENTAL & UNTESTED>>>>

    :param slope: Slope array (degrees).
    :param aspect: Aspect array (degrees, clockwise from north).
    :param elevation: Elevation array (meters).
    :param lat: Latitude of the raster's region (degrees).
    :param lon: Longitude of the raster's region (degrees).
    :param start_date: Start date in 'YYYY-MM-DD' format.
    :param end_date: End date in 'YYYY-MM-DD' format.
    :param out_file: Path to save the solar radiation raster.
    :param transform: GeoTransform for spatial reference.
    :param time_step: Time step in minutes for radiation simulation (default 60).
    :param nodata: Nodata value for the input arrays (default -9999).
    :return: Path to the output solar radiation raster.
    """
    import pvlib

    # Mask nodata values
    elevation = np.ma.masked_equal(elevation, nodata)
    slope = np.ma.masked_equal(slope, nodata)
    aspect = np.ma.masked_equal(aspect, nodata)

    # Generate a time range
    times = pd.date_range(start=start_date, end=end_date, freq=f'{time_step}T', tz='UTC')

    # Precompute solar radiation components
    solar_radiation = np.zeros(elevation.shape, dtype=np.float32)

    for current_time in times:
        # Get solar position
        solar_position = pvlib.solarposition.get_solarposition(current_time, lat, lon, elevation.mean())
        zenith = np.radians(solar_position['apparent_zenith'])
        azimuth = np.radians(solar_position['azimuth'])

        # Calculate the cosine of the solar incidence angle
        cos_incidence = _calculate_cosine_incidence(np.radians(slope), np.radians(aspect), zenith, azimuth)

        # Ignore night times (zenith > 90°)
        cos_incidence[zenith > np.pi / 2] = 0

        # Estimate direct normal irradiance (DNI)
        clearsky = pvlib.clearsky.ineichen(solar_position['apparent_zenith'], altitude=elevation.mean())
        dni = clearsky['dni'].values

        # Calculate solar radiation
        direct_radiation = dni[:, None, None] * cos_incidence
        diffuse_radiation = 0.3 * direct_radiation  # Estimate diffuse as 30% of direct

        # Accumulate radiation over time
        solar_radiation += np.clip(direct_radiation + diffuse_radiation, 0, None)

    # Normalize by time range to get average radiation
    solar_radiation /= len(times)

    # Save the output raster
    profile = {
        'driver': 'GTiff',
        'dtype': rio.float32,
        'nodata': nodata,
        'width': elevation.shape[1],
        'height': elevation.shape[0],
        'count': 1,
        'crs': 'EPSG:4326',  # Assuming lat/lon input
        'transform': transform,
    }

    with rio.open(out_file, 'w', **profile) as dst:
        dst.write(solar_radiation.astype(rio.float32), 1)

    return out_file


def calculateStatistics(src: rio.DatasetReader) -> Union[rio.DatasetReader, None]:
    """
    Function to recalculate statistics for each band of a rasterio dataset reader object.

    :param src: input rasterio dataset reader object in 'r+' mode
    :return: rasterio dataset reader object in 'r+' mode
    """
    try:
        # Calculate statistics for all bands
        stats = src.stats()

        # Update dataset tags with the new statistics
        for i, band in enumerate(src.indexes):
            # Convert the Statistics object to a dictionary
            stats_dict = {
                'min': stats[i].min,
                'max': stats[i].max,
                'mean': stats[i].mean,
                'std': stats[i].std
            }
            src.update_tags(band, **stats_dict)

        return src
    except Exception:
        for bidx in src.indexes:
            try:
                src.statistics(bidx, clear_cache=True)
            except rio.errors.StatisticsError as e:
                print(f'Rasterio Calculate Statistics Error: {e}')
                continue
        return


def changeDtype(src: rio.DatasetReader,
                dtype: np.dtype,
                nodata_val: Optional[Union[int, float]] = None) -> rio.DatasetReader:
    """
    Function to change a raster's datatype to int or float.

    :param src: input rasterio dataset reader object
    :param dtype: new numpy data type (e.g., np.int32, np.float32)
    :param nodata_val: value to assign as "no data" (default = None)
    :return: rasterio dataset reader object in 'r+' mode
    """
    if nodata_val is None:
        nodata_val = src.profile['nodata']

    # Convert array values to integer type
    src_array = src.read()
    src_array[src_array == src.nodata] = nodata_val
    src_array = np.asarray(src_array, dtype=dtype)

    # Get file path and profile of the dataset object
    src_path = src.name
    profile = src.profile

    # Specify LZW compression and assign integer datatype
    profile.update(
        compress='lzw',
        nodata=nodata_val,
        dtype=dtype)

    src.close()

    # Create new raster file
    with rio.open(src_path, 'w', **profile) as dst:
        # Write data to new raster
        dst.write(src_array)
        # Calculate new statistics
        calculateStatistics(dst)

    return rio.open(src_path, 'r+')


def clipRaster_wRas(src: rio.DatasetReader,
                    mask_src: rio.DatasetReader,
                    out_file: str,
                    all_touched: Optional[bool] = True,
                    crop: Optional[bool] = True) -> rio.DatasetReader:
    """
    Function to clip a raster with the extent of another raster.

    :param src: rasterio dataset reader object being masked
    :param mask_src: rasterio dataset reader object used as a mask
    :param out_file: location and name to save output raster
    :param all_touched: include all cells that touch the raster boundary (in addition to inside the boundary)
        (default = True)
    :param crop: crop output extent to match the extent of the data (default = True)
    :return: rasterio dataset reader object in 'r+' mode
    """
    geometry = [box(*mask_src.bounds)]

    out_array, out_transform = mask(src, geometry, all_touched=all_touched, crop=crop)
    out_profile = src.profile
    src.close()

    out_profile.update(
        height=out_array.shape[1],
        width=out_array.shape[2],
        transform=out_transform
    )

    # Create new raster file
    with rio.open(out_file, 'w', **out_profile) as dst:
        # Write data to new raster
        dst.write(out_array)
        # Calculate new statistics
        calculateStatistics(dst)

    return rio.open(out_file, 'r+')


def clipRaster_wShape(src: rio.DatasetReader,
                      shape_path: str,
                      out_file: str,
                      select_field: Optional[str] = None,
                      select_value: Optional[Union[any, list[any]]] = None,
                      all_touched: Optional[bool] = True,
                      crop: Optional[bool] = True,
                      use_extent: Optional[bool] = False,
                      new_nodata_val: Optional[float] = None) -> rio.DatasetReader:
    """
    Function to clip a raster with a shapefile or its extent.

    :param src: input rasterio dataset reader object
    :param shape_path: file path to clip shapefile
    :param out_file: location and name to save output raster
    :param select_field: name of field to use to select specific features for clipping
    :param select_value: value(s) in select_field to use for the feature selection
    :param all_touched: If True, all cells touching the shapefile boundary are included (default = True)
    :param crop: crop output extent to match the extent of the data (default = True)
    :param use_extent: If True, clip raster using the shapefile's bounding box (default = False)
    :param new_nodata_val: Value to replace the existing no data value in the output raster (default = None)
    :return: rasterio dataset reader object in 'r+' mode
    """
    if select_field is not None:
        if not isinstance(select_field, str):
            raise ValueError('Parameter "select_field" must be str type')
        if select_value is None:
            raise ValueError('Parameter "select_value" requires a value when selecting features')

    src_path = src.name
    src.close()

    # Get the shapefile geometries
    with fiona.open(shape_path, 'r') as shapefile:
        if select_field is not None:
            if isinstance(select_value, list):
                filtered_features = [
                    feature for feature in shapefile
                    if feature['properties'][select_field] in select_value
                ]
            else:
                filtered_features = [
                    feature for feature in shapefile
                    if feature['properties'][select_field] == select_value
                ]
            geometries = [feature['geometry'] for feature in filtered_features]

            if not geometries:
                raise RuntimeWarning(f'No features found with {select_field} = {select_value}')
        else:
            geometries = [feature['geometry'] for feature in shapefile]

        if use_extent:
            # Get the bounding box of the shapefile
            shp_bounds = shapefile.bounds
            extent_geom = {
                'type': 'Polygon',
                'coordinates': [[
                    [shp_bounds[0], shp_bounds[1]],
                    [shp_bounds[0], shp_bounds[3]],
                    [shp_bounds[2], shp_bounds[3]],
                    [shp_bounds[2], shp_bounds[1]],
                    [shp_bounds[0], shp_bounds[1]]
                ]]
            }
            geometries = [extent_geom]

    # Open the raster file
    with rio.open(src_path) as new_src:
        if new_src.nodata is None:
            nodata_val = np.nan
        else:
            nodata_val = new_src.nodata
        # Clip the raster using the geometries or extent
        out_image, out_transform = mask(new_src, geometries, nodata=nodata_val, all_touched=all_touched, crop=crop)
        out_meta = new_src.meta.copy()

        # Replace the existing no data value if new_nodata_val is specified
        if new_nodata_val is not None:
            # Convert the existing no data value to the new one
            out_image[~np.isfinite(out_image)] = new_nodata_val
            out_image[out_image == nodata_val] = new_nodata_val
            out_meta['nodata'] = new_nodata_val

        # Update the metadata with the new dimensions, transform, and CRS
        out_meta.update(
            {
                'height': out_image.shape[1],
                'width': out_image.shape[2],
                'transform': out_transform
            }
        )

    with rio.open(out_file, 'w', **out_meta) as dst:
        dst.write(out_image)

        # Calculate new statistics
        calculateStatistics(dst)

    return rio.open(out_file, 'r+')


def copyRaster(src: rio.DatasetReader,
               out_file: str,
               band: int = None) -> rio.DatasetReader:
    """
    Function to copy a raster to a new location or export a specific band.

    :param src: input rasterio dataset reader object
    :param out_file: location and name to save output raster
    :param band: specific band number to export (1-based index), optional
    :return: rasterio dataset reader object in 'r+' mode
    """
    if band is not None:
        # Validate band number
        if band < 1 or band > src.count:
            raise ValueError(f'Band number {band} is out of range. Source has {src.count} bands.')

        # Read the specific band
        band_data = src.read(band)

        # Create a new raster file with the selected band
        profile = src.profile
        profile.update(count=1)  # Set the number of bands to 1
        with rio.open(out_file, 'w', **profile) as dst:
            dst.write(band_data, 1)
    else:
        try:
            shutil.copyfiles(src.name, out_file)
        except AttributeError:
            try:
                shutil.copyfile(src.name, out_file)
            except AttributeError:
                raise RuntimeError('Unable to copy file. Both shutil and raster.shutil methods failed.')

    return rio.open(out_file, 'r+')


def defineProjection(src: rio.DatasetReader,
                     crs: Optional[str] = 'EPSG:4326') -> rio.DatasetReader:
    """
    Function to define the projection of a raster when the projection is missing (i.e., not already defined).

    :param src: input rasterio dataset reader object
    :param crs: location and name to save output raster (default = 'EPSG:4326')
    :return: rasterio dataset reader object in 'r+' mode
    """
    # Get path of input source dataset
    src_path = src.name

    # Close input source dataset
    src.close()

    # Convert CRS string to a rasterio CRS object
    crs_obj = CRS.from_string(crs)

    with rio.open(src_path, 'r+') as src:
        # Assign the CRS
        src.crs = crs_obj

        # Ensure ArcGIS compatibility by updating metadata tags
        wkt = crs_obj.to_wkt()
        src.update_tags(ns='gdal', SRS_WKT=wkt)

    # Return new raster as "readonly" rasterio openfile object
    return rio.open(src_path, 'r+')


def exportRaster(src: rio.DatasetReader,
                 out_file: str) -> rio.DatasetReader:
    """
    Function to export a raster to a new location.

    :param src: input rasterio dataset reader object
    :param out_file: location and name to save output raster
    :return: rasterio dataset reader object in 'r+' mode
    """
    # Get profile
    src_profile = src.profile

    # Specify LZW compression
    src_profile.update(
        compress='lzw'
    )

    # Get raster array
    src_array = src.read()

    # Create new raster file
    with rio.open(out_file, 'w', **src_profile) as dst:
        # Write data to new raster
        dst.write(src_array)
        # Calculate new statistics
        calculateStatistics(dst)

    # Return new raster as "readonly" rasterio openfile object
    return rio.open(out_file, 'r+')


def extractRasterBand(src_path: str, out_path: str, band: int) -> rio.DatasetReader:
    """
    Extract a specific band from a raster and save it as a new raster.

    :param src_path: Path to the source raster file.
    :param out_path: Path to save the extracted band raster file.
    :param band: The band number to extract (1-based index).
    :return: rasterio dataset reader object in 'r+' mode
    """
    with rio.open(src_path) as src:
        profile = src.profile

        # Ensure the band number is valid
        if band < 1 or band > src.count:
            raise ValueError(f"Invalid band number: {band}. The raster has {src.count} bands.")

        # Read the specified band
        band_data = src.read(band)

        # Update the profile for a single-band output
        profile.update(count=1, dtype=band_data.dtype)

        # Save the extracted band as a new raster
        with rio.open(out_path, 'w', **profile) as dst:
            dst.write(band_data, 1)

    # Return new raster as "readonly" rasterio openfile object
    return rio.open(out_path, 'r+')


def extractValuesAtPoints(in_pts: Union[str, GeoDataFrame],
                          src: rio.DatasetReader,
                          value_field: str,
                          out_type: str = 'series',
                          new_pts_path: Optional[str] = None) -> Union[pd.Series, None]:
    """
    Function to extract raster values at shapefile point locations.

    :param in_pts: path to, or a GeoDataFrame object of, the point shapefile
    :param src: input rasterio dataset reader object
    :param value_field: name of field to contain raster values
    :param out_type: type of output ("gdf", "series", "shp", "csv")
    :param new_pts_path: path to new point shapefile, if out_type == 'shp' (default = None)
    :return: a GeoDataFrame object (if out_type is "gdf"),
        a pandas series object (if out_type is "series"), or None (if out_type is "shp" or "csv")
    """
    # If in_pts is not str or GeoDataFrame type, raise error
    if not isinstance(in_pts, (str, GeoDataFrame)):
        raise TypeError('Parameter "in_pts" must be a str or GeoPandas GeoDataFrame type')

    # If in_pts is str type, get GeoDataFrame object from path
    if isinstance(in_pts, str):
        gdf = gpd.read_file(in_pts)
    elif isinstance(in_pts, GeoDataFrame):
        gdf = in_pts

    # Extract raster values at point locations, and assign to "value_field"
    gdf[f'{value_field}'] = next(src.sample(zip(gdf['geometry'].x, gdf['geometry'].y)))[0]

    # Return raster point values
    if out_type == 'gdf':
        return gdf
    elif out_type == 'series':
        return gdf[f'{value_field}']
    elif out_type == 'shp':
        gdf.to_file(new_pts_path)
    elif out_type == 'csv':
        gdf.to_csv(new_pts_path)
    else:
        raise TypeError('[ProcessRasters] extractValuesAtPoints() function parameter "out_type" '
                        'must be either "series", "shp" or "csv"')

    return


def extractRowsColsWithPoly(in_poly: Union[str, fiona.Collection],
                            src: rio.DatasetReader,
                            id_field: str) -> list:
    """
    Function to extract a list of global row and column numbers from a raster (numpy array) with a polygon shapefile.

    :param in_poly: path to, or a fiona collection object of, a polygon shapefile
    :param src: input rasterio dataset reader object
    :param id_field: name of field in shapefile to use as feature ID
    :return: a list of row and column numbers [(row1, col1), (row2, col2), ...]
    """
    # If in_pts is str type, get fiona dataset object from path
    if isinstance(in_poly, str):
        in_poly = fiona.open(in_poly)

    # Get the transform and projection of the raster file
    crs = src.crs
    nodata_value = src.nodata

    # Check if the shapefile's CRS matches the raster's CRS
    shapefile_crs = in_poly.crs
    if shapefile_crs != crs:
        raise ValueError('Shapefile and raster CRS do not match')

    output_list = []

    for feature in in_poly:
        geom = shape(feature['geometry'])
        district_id = feature['properties'][id_field]

        # Calculate the window of the raster that intersects with the geometry
        window = geometry_window(src, [mapping(geom)], pad_x=1, pad_y=1)

        # Read the data in the window
        raster_data = src.read(1, window=window, masked=True)

        # Skip if the window data is empty
        if (raster_data.size == 0) or np.all(raster_data.mask):
            continue

        # Get the affine transformation for the window
        window_transform = src.window_transform(window)

        # Create a mask for the geometry
        geom_mask = geometry_mask(
            geometries=[mapping(geom)],
            transform=window_transform,
            invert=True,
            out_shape=(raster_data.shape[0],
                       raster_data.shape[1])
        )

        # Apply the geometry mask and check for NoData values
        valid_mask = (raster_data.data != nodata_value) * geom_mask

        # Get the row and column indices of the masked cells
        rows, cols = np.where(valid_mask)

        # Skip if no rows/cols are found
        if len(rows) == 0 or len(cols) == 0:
            continue

        # Convert to global indices
        global_rows = rows + window.row_off
        global_cols = cols + window.col_off
        row_col_pairs = list(zip(global_rows, global_cols))

        # Append the district ID and the row/column indices to the output list
        output_list.append([district_id, row_col_pairs])

    return output_list


def featureToRaster(feature_path: str,
                    out_path: str,
                    ref_ras_src: rio.DatasetReader,
                    value_field: str) -> rio.DatasetReader:
    """
    Function creates a raster from a shapefile.

    :param feature_path: path to feature dataset
    :param out_path: path to output raster
    :param ref_ras_src: rasterio DatasetReader object to use as reference
    :param value_field: name of the field/column to use for the raster values
    :return: rasterio dataset reader object in 'r+' mode
    """

    def _infer_raster_dtype(dtype):
        """Map pandas dtype to rasterio dtype."""
        if dtype.startswith('int'):
            return 'int32'
        elif dtype.startswith('float'):
            return 'float32'
        elif dtype == 'bool':
            return 'uint8'
        else:
            return 'str'

    # Get GeoPandas object of the feature dataset
    src = gpd.read_file(feature_path)

    # Ensure the value_field contains numeric values
    if not pd.api.types.is_numeric_dtype(src[value_field]):
        raise ValueError(f"The field '{value_field}' contains non-numeric values.")

    # Infer the appropriate raster dtype based on the feature data type
    pandas_dtype = pd.api.types.infer_dtype(src[value_field])
    raster_dtype = _infer_raster_dtype(pandas_dtype)

    # Get the schema of the reference raster, including the crs
    meta = ref_ras_src.meta.copy()
    meta.update(compress='lzw', dtype=raster_dtype)

    with rio.open(out_path, 'w+', **meta) as dst:
        # Get the raster array with the correct dtype
        dst_arr = dst.read(1).astype(raster_dtype)

        # Create a generator of geom, value pairs to use while rasterizing
        shapes = ((geom, value) for geom, value in zip(src['geometry'], src[value_field]))

        # Replace the raster array with new data
        burned = rasterize(shapes=shapes,
                           fill=0,
                           out=dst_arr,
                           transform=dst.transform)

        # Write the new data to band 1
        dst.write_band(1, burned)

    return rio.open(out_path, 'r+')


def getArea(src: rio.DatasetReader,
            search_value: Union[int, float, list[int, float]]) -> float:
    """
    Function to calculate the area of all cells matching a value or values.

    :param src: input rasterio dataset reader object
    :param search_value: raster value(s) to get area of
    :return: area (in units of raster)
    """
    if not isinstance(search_value, list):
        search_value = [search_value]

    x, y = getResolution(src)
    src_array = src.read()
    value_count = 0
    for value in search_value:
        value_count += np.count_nonzero(src_array == value)

    return x * y * value_count


def getAspect(src: rio.DatasetReader,
              out_file: str) -> rio.DatasetReader:
    """
    Function to generate a slope aspect raster from an elevation dataset.

    :param src: a rasterio dataset reader object
    :param out_file: the path and name of the output file
    :return: rasterio dataset reader object in r+ mode
    """
    from osgeo import gdal

    # Enable exceptions in GDAL to handle potential errors
    gdal.UseExceptions()

    # Get file path of dataset object
    src_path = src.name

    gdal.DEMProcessing(out_file,
                       src_path,
                       'aspect')

    # Calculate new statistics
    temp_src = getRaster(out_file)
    calculateStatistics(temp_src)
    temp_src.close()

    return rio.open(out_file, 'r+')


def getFirstLast(in_rasters: list[rio.DatasetReader],
                 out_file: str,
                 first_last: str,
                 full_extent: bool = True) -> rio.DatasetReader:
    """
    Function creates a new raster using the first valid values per cell across all input rasters.

    :param in_rasters: list of rasterio dataset reader objects
    :param out_file: the path and name of the output file
    :param first_last: output the first or last value (Options: "first", "last")
    :param full_extent: boolean indicating whether to use the (True) full extent of all rasters,
        or (False) only the overlapping extent
    :return: rasterio dataset reader object in 'r+' mode
    """
    # Verify inputs
    if not isinstance(in_rasters, list):
        raise TypeError('[ProcessRasters] getFirstLast() param "in_rasters" must be a list of rasterio objects')
    if not isinstance(out_file, str):
        raise TypeError('[ProcessRasters] getFirstLast() param "out_file" must be a string data type')
    if not isinstance(first_last, str):
        raise TypeError('[ProcessRasters] getFirstLast() param "first_last" must be a string data type')
    elif first_last not in ['first', 'last']:
        raise ValueError('[ProcessRasters] getFirstLast() param "first_last" must be either "first" or "last"')
    if not isinstance(full_extent, bool):
        raise TypeError('[ProcessRasters] getFirstLast() param "full_extent" must be a boolean data type')

    # Use the appropriate bounds option based on full_extent parameter
    bounds = None if full_extent else 'intersection'

    # Merge rasters to get the first valid values
    mosaic, out_trans = merge(in_rasters, method='first', bounds=bounds)

    # Update the profile with new dimensions and transform
    profile = in_rasters[0].profile
    profile.update({
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': out_trans
    })

    # Create new raster file and write data
    with rio.open(out_file, 'w', **profile) as dst:
        dst.write(mosaic)

    return rio.open(out_file, 'r+')


def getGridCoordinates(src: rio.DatasetReader,
                       out_file_x: str,
                       out_file_y: str,
                       out_crs: str = 'EPSG:4326',
                       dtype: np.dtype = np.float32) -> tuple[rio.DatasetReader, rio.DatasetReader]:
    """
    Function returns two X and Y rasters with cell values matching the grid cell coordinates (one for Xs, one for Ys).

    :param src: a rasterio dataset reader object
    :param out_file_x: path to output raster for X coordinates
    :param out_file_y: path to output raster for Y coordinates
    :param out_crs: string defining new projection (e.g., 'EPSG:4326')
    :param dtype: numpy data type for output rasters (default is np.float32)
    :return: a tuple (X, Y) of rasterio dataset reader objects in 'r+' mode
    """
    # Get the affine transformation matrix
    transform = src.transform

    # Get the coordinate reference system (CRS) of the input raster
    src_crs = src.crs

    # Get the number of rows and columns in the raster
    rows, cols = src.height, src.width

    # Get the geolocation of the top-left corner and pixel size
    x_start, y_start = transform * (0, 0)
    x_end, y_end = transform * (cols, rows)
    pixel_size_x = (x_end - x_start) / cols
    pixel_size_y = (y_end - y_start) / rows

    # Create a new affine transformation matrix for EPSG:4326
    transformer = Transformer.from_crs(src_crs, out_crs, always_xy=True)

    # Calculate the x & y coordinates for each cell
    x_coords = np.linspace(x_start + pixel_size_x / 2, x_end - pixel_size_x / 2, cols)
    y_coords = np.linspace(y_start + pixel_size_y / 2, y_end - pixel_size_y / 2, rows)
    lon, lat = np.meshgrid(x_coords, y_coords)
    lon, lat = transformer.transform(lon.flatten(), lat.flatten())

    # Reshape the lon and lat arrays to match the shape of the raster
    lon = lon.reshape(rows, cols).astype(dtype)
    lat = lat.reshape(rows, cols).astype(dtype)

    # Create output profiles for x and y coordinate rasters, setting dtype to the specified parameter
    profile = src.profile.copy()
    profile.update(dtype=dtype)

    # Write X coordinate data to out_path_x
    with rio.open(out_file_x, 'w', **profile) as dst:
        dst.write(lon, 1)
        calculateStatistics(dst)

    # Write Y coordinate data to out_path_y
    with rio.open(out_file_y, 'w', **profile) as dst:
        dst.write(lat, 1)
        calculateStatistics(dst)

    return rio.open(out_file_x, 'r+'), rio.open(out_file_y, 'r+')


def getHillshade(src: rio.DatasetReader,
                 out_file: str) -> rio.DatasetReader:
    """
    Function to generate a hillshade from an elevation dataset.

    :param src: a rasterio dataset reader object
    :param out_file: the path and name of the output file
    :return: rasterio dataset reader object
    """
    from osgeo import gdal

    # Get file path of dataset object
    src_path = src.name

    gdal.DEMProcessing(out_file,
                       src_path,
                       'hillshade')

    # Calculate new statistics
    temp_src = getRaster(out_file)
    calculateStatistics(temp_src)
    temp_src.close()

    return rio.open(out_file, 'r+')


def getMean(in_rasters: list[rio.DatasetReader],
            out_file: str,
            full_extent: bool = True) -> rio.DatasetReader:
    """
    Function creates a new raster from the mean value per cell across all input rasters.

    :param in_rasters: list of rasterio dataset reader objects
    :param out_file: the path and name of the output file
    :param full_extent: boolean indicating whether to use (True) the full extent of all rasters,
        or (False) only the overlapping extent (default = True)
    :return: rasterio dataset reader object in 'r+' mode
    """
    # Verify inputs
    if not isinstance(in_rasters, list):
        raise TypeError('[ProcessRasters] getMean() param "inrasters" must be a list of rasterio objects')
    if not isinstance(out_file, str):
        raise TypeError('[ProcessRasters] getMean() param "out_file" must be a string data type')
    if not isinstance(full_extent, bool):
        raise TypeError('[ProcessRasters] getMean() param "full_extent" must be a boolean data type')

    # Use the appropriate bounds option based on full_extent parameter
    bounds = None if full_extent else 'intersection'

    # Merge rasters to get the mean values
    mosaic, out_trans = merge(in_rasters, method='mean', bounds=bounds)

    # Update the profile with new dimensions and transform
    profile = in_rasters[0].profile
    profile.update({
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': out_trans
    })

    # Create new raster file and write data
    with rio.open(out_file, 'w', **profile) as dst:
        dst.write(mosaic)

    return rio.open(out_file, 'r+')


def getMinMax(in_rasters: Union[list[str], list[rio.DatasetReader]],
              out_file: str,
              min_max: str,
              full_extent: bool = True) -> rio.DatasetReader:
    """
    Function creates a new raster from the minimum or maximum values per cell across all input rasters.

    :param in_rasters: list of file paths or rasterio dataset reader objects
    :param out_file: the path and name of the output file
    :param min_max: output the minimum or maximum value (Options: "min", "max")
    :param full_extent: boolean indicating whether to use (True) the full extent of all rasters,
        or (False) only the overlapping extent (default = True)
    :return: rasterio dataset reader object in 'r+' mode
    """
    # Verify inputs
    if not isinstance(in_rasters, list):
        raise TypeError('[ProcessRasters] getMinMax() param "inrasters" must be '
                        'a list of file paths or rasterio dataset reader objects')
    if not isinstance(out_file, str):
        raise TypeError('[ProcessRasters] getMinMax() param "out_file" must be a string data type')
    if not isinstance(min_max, str):
        raise TypeError('[ProcessRasters] getMinMax() param "min_max" must be a string data type')
    elif min_max not in ['min', 'max']:
        raise ValueError('[ProcessRasters] getMinMax() param "min_max" must be either "min" or "max"')
    if not isinstance(full_extent, bool):
        raise TypeError('[ProcessRasters] getMinMax() param "full_extent" must be a boolean data type')

    # Get list of rasterio dataset reader objects if list of file paths was provided
    if isinstance(in_rasters[0], str):
        in_rasters = [getRaster(path) for path in in_rasters]

    # Use the appropriate bounds option based on full_extent parameter
    bounds = None if full_extent else 'intersection'

    if 'min' in min_max:
        # Merge rasters to get the minimum values
        mosaic, out_trans = merge(in_rasters, method='min', bounds=bounds)
    else:
        # Merge rasters to get the maximum values
        mosaic, out_trans = merge(in_rasters, method='max', bounds=bounds)

    # Update the profile with new dimensions and transform
    profile = in_rasters[0].profile
    profile.update({
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': out_trans
    })

    # Create new raster file and write data
    with rio.open(out_file, 'w', **profile) as dst:
        dst.write(mosaic)

    return rio.open(out_file, 'r+')


def getOverlapWindow(large_src, small_src):
    """
    Generate a rasterio window from the overlap of a smaller raster with a larger raster.

    :param large_src: A rasterio.DatasetReader object for the larger raster.
    :param small_src: A rasterio.DatasetReader object for the smaller raster.
    :return: A rasterio Window object representing the overlapping region.
    """
    # Extract the bounds of the smaller raster
    small_bounds = small_src.bounds
    min_x, min_y, max_x, max_y = small_bounds.left, small_bounds.bottom, small_bounds.right, small_bounds.top

    # Ensure valid extents
    min_x, max_x = sorted([min_x, max_x])
    min_y, max_y = sorted([min_y, max_y])

    # Convert the smaller raster's bounds to grid coordinates in the larger raster
    min_col, max_row = rowcol(large_src.transform, min_x, min_y)  # Bottom-left corner
    max_col, min_row = rowcol(large_src.transform, max_x, max_y)  # Top-right corner

    # Clip to the bounds of the larger raster
    min_row = max(0, min_row)
    min_col = max(0, min_col)
    max_row = min(large_src.height, max_row)
    max_col = min(large_src.width, max_col)

    # Create a window for rasterio to subset data
    window = Window.from_slices((min_row, max_row), (min_col, max_col))

    return window


def getRaster(in_path: str) -> rio.DatasetReader:
    """
    Function to get a rasterio dataset reader object from a raster file path.

    :param in_path: path to raster dataset
    :return: rasterio dataset reader object in 'r+' mode
    """
    return rio.open(in_path, 'r+')


def getResolution(src: rio.DatasetReader) -> tuple:
    """
    Function to get the x & y grid cell resolution of a raster dataset.

    :param src: input rasterio dataset reader object
    :return: tuple of floats; (x resolution, y resolution)
    """
    return src.res


def getSlope(src: rio.DatasetReader,
             out_file: str,
             slope_format: str) -> rio.DatasetReader:
    """
    Function to calculate slope from an elevation raster.

    :param src: input rasterio dataset reader object
    :param out_file: the path and name of the output file
    :param slope_format: slope format ("degree" or "percent")
    :return: rasterio dataset reader object in 'r+' mode
    """
    from osgeo import gdal

    # Enable exceptions in GDAL to handle potential errors
    gdal.UseExceptions()

    # Get file path of dataset object
    src_path = src.name

    # Process the slope calculation with GDAL
    gdal.DEMProcessing(out_file,
                       src_path,
                       'slope',
                       slopeFormat=slope_format)

    # Calculate new statistics
    temp_src = getRaster(out_file)
    calculateStatistics(temp_src)
    temp_src.close()

    return rio.open(out_file, 'r+')


def Integer(src: rio.DatasetReader,
            dtype: np.dtype,
            nodata_val: int,
            round_values: bool = False) -> rio.DatasetReader:
    """
    Function to convert a raster to an Integer data type. Updates the source raster file in place.

    :param src: input rasterio dataset reader object
    :param dtype: numpy dtype; e.g., signed (np.int8, np.int32) or unsigned (np.uint8, np.unit32)
    :param nodata_val: integer value to assign as no data
    :param round_values: round values before converting to Integer
    :return: rasterio dataset reader object in 'r+' mode
    """
    # Convert array values to integer type
    int_array = src.read()
    if round_values is not None:
        round_values = np.round(round_values)
    if src.nodata != nodata_val:
        int_array[int_array == src.nodata] = nodata_val
    int_array = np.asarray(int_array, dtype=int)

    # Get file path of dataset object
    src_path = src.name

    # Get profile of dataset object
    profile = src.profile

    # Specify LZW compression and assign integer datatype
    profile.update(
        compress='lzw',
        nodata=nodata_val,
        dtype=dtype)

    src.close()
    del src

    # Create new raster file
    with rio.open(src_path, 'w', **profile) as dst:
        # Write data to new raster
        dst.write(int_array)

        # Set the nodata value explicitly in the dataset
        dst.nodata = nodata_val

        # Update the metadata with the new nodata value
        dst.update_tags(nodata=nodata_val)

    return rio.open(src_path, 'r+')


def matchExtents(src: rio.DatasetReader,
                 ref_src: rio.DatasetReader,
                 out_file: str,
                 match_res: bool = False) -> rio.DatasetReader:
    """
    Match the extent of the source raster to the target raster.
    """
    # Check if extents and resolution already match
    if (src.bounds == ref_src.bounds and
            src.width == ref_src.width and
            src.height == ref_src.height and
            src.crs == ref_src.crs):
        return src

    # Extract metadata from src
    src_transform = src.transform
    src_crs = src.crs
    src_res_x, src_res_y = src_transform.a, -src_transform.e
    src_count = src.count
    src_dtype = src.read(1).dtype

    # Extract metadata from ref_src
    ref_transform = ref_src.transform
    ref_crs = ref_src.crs
    ref_bounds = ref_src.bounds

    # Determine the new resolution and bounds
    if match_res:
        # Match both resolution and extent
        new_transform = ref_transform
        new_width, new_height = ref_src.width, ref_src.height
    else:
        # Retain source resolution, adjust extent
        new_res_x, new_res_y = src_res_x, src_res_y
        new_width = round((ref_bounds.right - ref_bounds.left) / new_res_x)
        new_height = round((ref_bounds.top - ref_bounds.bottom) / new_res_y)
        new_transform = Affine(new_res_x, 0, ref_bounds.left, 0, -new_res_y, ref_bounds.top)

    # Prepare metadata for the new raster
    dest_meta = src.meta.copy()
    dest_meta.update({
        'crs': ref_crs,
        'transform': new_transform,
        'width': new_width,
        'height': new_height,
        'nodata': src.meta.get('nodata', -9999)  # Ensure nodata is explicitly set
    })

    # Create a temporary file if src == out_file
    if src.name == out_file:
        temp_file = out_file.replace('.tif', '_.tif')
    else:
        temp_file = out_file

    # Write the new raster to the temporary file
    with rio.open(temp_file, 'w', **dest_meta) as dest:
        for i in range(1, src_count + 1):
            destination_array = np.full((new_height, new_width), dest_meta['nodata'], dtype=src_dtype)
            reproject(
                source=rio.band(src, i),
                destination=destination_array,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=new_transform,
                dst_crs=ref_crs,
                dst_width=new_width,
                dst_height=new_height,
                resampling=Resampling.nearest,
                dst_nodata=dest_meta['nodata']
            )
            dest.write(destination_array, i)

    # If using a temporary file, replace the original file
    if src.name == out_file:
        src.close()  # Ensure the source file is closed
        os.replace(temp_file, out_file)  # Replace the original file with the temporary file

    return rio.open(out_file, 'r+')


def matchTopLeftCoord(src: rio.DatasetReader,
                      ref_src: rio.DatasetReader) -> rio.DatasetReader:
    """
    Align the top-left coordinate of the source raster to match the target raster,
    while maintaining the original resolution of the source raster. Updates the source raster file in place.

    :param src: source rasterio dataset reader object to alter (must be in "r+" mode)
    :param ref_src: target rasterio dataset reader object to use as reference
    :return: the altered source rasterio dataset reader object
    """
    # Get the top-left coordinate of the target raster
    target_transform = ref_src.transform
    target_top_left_x, target_top_left_y = target_transform[2], target_transform[5]

    # Get the original resolution of the source raster
    src_transform = src.transform
    pixel_size_x = src_transform[0]
    pixel_size_y = src_transform[4]

    # Create a new transform for the source raster, keeping its resolution but adjusting the top-left corner
    new_transform = from_origin(target_top_left_x, target_top_left_y, pixel_size_x, -pixel_size_y)

    # Update the transform of the source raster in place
    src.transform = new_transform
    src.update_tags(transform=str(new_transform))

    return src


def mosaicRasters(mosaic_list: list[Union[str, rio.DatasetReader]],
                  out_file: str,
                  extent_mode: str = 'union') -> rio.DatasetReader:
    """
    Function mosaics a list of rasterio objects to a new TIFF raster or updates an existing raster if specified.

    :param mosaic_list: List of paths to raster datasets or rasterio Dataset objects.
    :param out_file: Location and name to save the output raster.
    :param extent_mode: Determines the extent of the mosaic:
                        - 'union': Uses the union of all input extents (default).
                        - 'max': Uses the total merged extent that encompasses all extents, including nodata areas.
                        - 'trim': Minimizes the extent to match the data (trims nodata areas).
    :return: rasterio dataset reader object in 'r+' mode.
    """
    # Open files as rasterio objects if they are paths
    datasets = []
    for data in mosaic_list:
        if isinstance(data, str):
            datasets.append(rio.open(data))
        else:
            datasets.append(data)

    # Initialize bounds and resolution variables
    max_bounds = None
    resolution = None

    for dataset in datasets:
        # Update the overall bounds
        bounds = dataset.bounds
        if max_bounds is None:
            max_bounds = bounds
        else:
            max_bounds = (
                min(max_bounds[0], bounds[0]),  # minX
                min(max_bounds[1], bounds[1]),  # minY
                max(max_bounds[2], bounds[2]),  # maxX
                max(max_bounds[3], bounds[3])  # maxY
            )

        # Check and set resolution
        res = dataset.res
        if resolution is None:
            resolution = res
        elif res != resolution:
            warnings.warn('Warning: Not all input rasters have the same resolution.')

    # Calculate bounds based on the extent_mode
    if extent_mode == 'union':
        # Default merge behavior (union of extents)
        mosaic, transform = merge(datasets)
    elif extent_mode == 'max':
        # Explicitly set bounds to max_bounds
        mosaic, transform = merge(datasets, bounds=max_bounds, res=resolution)
    elif extent_mode == 'trim':
        # Trim to valid data bounds (calculate the extent based on non-nodata values)
        valid_bounds = None
        for dataset in datasets:
            data = dataset.read(1, masked=True)  # Read as masked array
            shapes_generator = rio.features.shapes(data, transform=dataset.transform)

            # Extract geometries from the generator
            geometries = [geom for geom, _ in shapes_generator]

            if geometries:
                # Get the window covering the valid geometries
                window = rio.features.geometry_window(dataset, geometries)
                bounds = rio.windows.bounds(window, dataset.transform)

                if valid_bounds is None:
                    valid_bounds = bounds
                else:
                    valid_bounds = (
                        min(valid_bounds[0], bounds[0]),  # minX
                        min(valid_bounds[1], bounds[1]),  # minY
                        max(valid_bounds[2], bounds[2]),  # maxX
                        max(valid_bounds[3], bounds[3])  # maxY
                    )

        if valid_bounds is not None:
            mosaic, transform = merge(datasets, bounds=valid_bounds, res=resolution)
        else:
            raise ValueError("No valid data found in input rasters to determine trimmed extent.")
    else:
        raise ValueError(f'Invalid extent_mode "{extent_mode}". Choose from "union", "max", or "trim".')

    # Update metadata based on the mosaic
    out_meta = datasets[0].meta.copy()
    out_meta.update({
        'driver': 'GTiff',
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': transform,
        'count': mosaic.shape[0]
    })

    # Close all datasets
    for ds in datasets:
        ds.close()

    # Write to the output file
    with rio.open(out_file, 'w', **out_meta) as dst:
        dst.write(mosaic)

    return rio.open(out_file, 'r+')


def normalizeRaster(src: rio.DatasetReader,
                    out_file: str) -> rio.DatasetReader:
    """
    Function to normalize a raster by dividing each cell by the sum of all values in the raster,
    so that all values must sum to 1 over the entire raster.

    :param src: input rasterio dataset reader object with a single band of data
    :param out_file: path to save the output raster
    :return: rasterio dataset reader object in 'r+' mode
    """
    # Read the raster values into a numpy array
    raster_array = src.read(1)

    # Mask out invalid (NaN) values
    raster_array = np.ma.masked_invalid(raster_array)

    # Sum of all valid raster values
    total_sum = raster_array.sum()

    # Normalize the raster values to make them sum to 1
    normalized_raster = raster_array / total_sum

    # Write the normalized raster to a new file
    profile = src.profile
    with rio.open(out_file, 'w', **profile) as dst:
        dst.write(normalized_raster.filled(np.nan), 1)

    return rio.open(out_file, 'r+')


def rasterExtentToPoly(src: rio.DatasetReader,
                       out_file: str) -> None:
    """
    Function to convert a raster to a polygon shapefile.

    :param src: input rasterio dataset reader object
    :param out_file: location and name to save output polygon extent shapefile
    :return: None
    """
    # Get the bounding box (extent) of the raster
    raster_bounds = src.bounds
    crs = src.crs  # Get the coordinate reference system of the raster

    # Create a polygon from the bounding box
    raster_polygon = box(raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top)

    # Define the schema for the shapefile
    schema = {
        'geometry': 'Polygon',
        'properties': {}
    }

    # Save the polygon as a shapefile
    with fiona.open(out_file, 'w', driver='ESRI Shapefile', crs=crs, schema=schema) as shp:
        shp.write({
            'geometry': mapping(raster_polygon),
            'properties': {}
        })

    return


def rasterToPoints(src: rio.DatasetReader,
                   out_file: str,
                   band: int = 1) -> None:
    """
    Function to generate a point shapefile from the center of each raster grid cell containing valid data.
    No data values will be ignored.

    :param src: input rasterio dataset reader object
    :param out_file: location and name to save output polygon shapefile
    :param band: integer representing a specific band to extract points from (default = 1)
    :return: None
    """
    # Get the transform, dimensions, projection, and no data value from the raster
    transform = src.transform
    width = src.width
    height = src.height
    crs = src.crs
    nodata = src.nodata

    # Read the raster data
    data = src.read(band)  # Use the first band, unless otherwise specified

    # Prepare a list to store point geometries
    points = []

    # Loop through each cell in the raster
    for row in range(height):
        for col in range(width):
            # Check if the cell value is not NoData
            if data[row, col] != nodata:
                # Calculate the center of the cell
                x, y = xy(transform, row, col, offset='center')
                # Create a point geometry and add it to the list
                points.append(Point(x, y))

    # Define the schema of the shapefile
    schema = {
        'geometry': 'Point',
        'properties': {}
    }

    # Write points to a shapefile
    with fiona.open(out_file, 'w', driver='ESRI Shapefile', crs=crs, schema=schema) as shp:
        for point in points:
            shp.write({
                'geometry': mapping(point),
                'properties': {}
            })

    return


def rasterToPoly(src: rio.DatasetReader,
                 out_file: str,
                 shp_value_field: str = 'Value',
                 multiprocess: bool = False,
                 num_cores: int = 2,
                 block_size: int = 256) -> None:
    """
    Function to convert a raster to a polygon shapefile.

    :param src: input rasterio dataset reader object
    :param out_file: location and name to save output polygon shapefile
    :param shp_value_field: name of the shapefile field that will contain the raster values (Default = "Value")
    :param multiprocess: use multiprocessing for raster to polygon conversion (True, False)
    :param num_cores: number of cores for multiprocessing
    :param block_size: size of blocks (# raster cells) for multiprocessing
    :return: None
    """
    if not multiprocess:
        # Create shape generator
        # print('[rasterToPoly - Creating shape generator]')
        shape_gen = ((shape(s), v) for s, v in shapes(src.read(masked=True), transform=src.transform))

        # Build a GeoDataFrame from unpacked shapes
        # print('[rasterToPoly - Building GeoDataFrame]')
        gdf = GeoDataFrame(dict(zip(['geometry', f'{shp_value_field}'], zip(*shape_gen))), crs=src.crs)
    else:
        # ### Code for multiprocessing
        # Get raster dimensions
        height, width = src.height, src.width

        # Function to generate blocks
        # print('[rasterToPoly - Generating data blocks from raster]')

        def gen_blocks():
            for i in range(0, height, block_size):
                for j in range(0, width, block_size):
                    window = ((i, min(i + block_size, height)), (j, min(j + block_size, width)))
                    yield src.read(masked=True, window=window), src.transform, window

        # Set up parallel processing
        # print('[rasterToPoly - Setting up and running parallel processing]')
        shapes_list = Parallel(n_jobs=num_cores)(
            delayed(_process_block)(*block) for block in gen_blocks()
        )

        # Flatten the list of shapes
        # print('[rasterToPoly - Flattening shapes]')
        shapes_flat = [shp for shapes_sublist in shapes_list for shp in shapes_sublist]

        # Build a GeoDataFrame from unpacked shapes
        # print('[rasterToPoly - Building GeoDataFrame]')
        gdf = GeoDataFrame({'geometry': [s for s, _ in shapes_flat],
                            f'{shp_value_field}': [v for _, v in shapes_flat]},
                           crs=src.crs)

    # Save to shapefile
    # print('[rasterToPoly - Saving shapefile to out_file]')
    gdf.to_file(out_file)

    return


def reprojRaster(src: rio.DatasetReader,
                 out_file: str,
                 out_crs: int = 4326) -> rio.DatasetReader:
    """
    Function to reproject a raster to a different coordinate system.

    :param src: input rasterio dataset reader object
    :param out_file: location and name to save output raster
    :param out_crs: the EPSG number of the new projection (e.g., 4326)
    :return: rasterio dataset reader object in 'r+' mode
    """
    out_crs = f'EPSG:{out_crs}'
    # Calculate transformation needed to reproject raster to out_crs
    transform, width, height = calculate_default_transform(
        src.crs, out_crs, src.width, src.height, *src.bounds
    )
    meta = src.meta.copy()
    meta.update({
        'crs': out_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    # Reproject raster and write to out_file
    with rio.open(out_file, 'w', **meta) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rio.band(src, i),
                destination=rio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=out_crs,
                resampling=Resampling.nearest)

    # Return new raster as "readonly" rasterio openfile object
    return rio.open(out_file, 'r+')


def resampleRaster(src: rio.DatasetReader,
                   ref_src: rio.DatasetReader,
                   out_file: str,
                   num_bands: int = 1,
                   match_extents: bool = False,
                   new_nodata_val: Optional[float] = None,
                   method: str = 'nearest') -> rio.DatasetReader:
    """
    Function to resample the resolution of one raster to match that of a reference raster.
    This function will also reproject the source projection to match the reference if needed.

    :param src: input rasterio dataset reader object
    :param ref_src: reference rasterio dataset reader object
    :param out_file: location and name to save output raster
    :param num_bands: number of bands in the output dataset (default = 1)
    :param match_extents: if True, extents of the ref and src rasters will be matched
    :param new_nodata_val: new no-data value; if None, a value compatible with ref_src's dtype will be used
    :param method: resampling method to use (default = "nearest"). Options: "nearest", "bilinear", "cubic", "average", "mode"
    :return: rasterio dataset reader object in 'r+' mode
    """
    # Define a dictionary to map method strings to rasterio resampling methods
    resampling_methods = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic,
        'average': Resampling.average,
        'mode': Resampling.mode
    }

    # Get the resampling method, default to nearest if invalid
    resampling_method = resampling_methods.get(method.lower(), Resampling.nearest)

    # Get the transform, dimensions, and projection of the reference dataset
    ref_transform = ref_src.transform
    ref_width = ref_src.width
    ref_height = ref_src.height
    ref_crs = ref_src.crs

    # Determine the data type of the reference raster
    dtype = ref_src.dtypes[0]  # Assuming all bands have the same dtype

    # Determine nodata value
    if new_nodata_val is None:
        if np.issubdtype(dtype, np.integer):
            nodata_val = np.iinfo(dtype).min
        else:
            nodata_val = np.nan
    else:
        nodata_val = new_nodata_val

    # Prepare the profile for the output
    profile = ref_src.profile
    profile.update(
        {
            'count': num_bands,  # Number of bands
            'height': ref_height,
            'width': ref_width,
            'transform': ref_transform,
            'nodata': nodata_val,  # Update nodata value
        }
    )

    # Prepare the output arrays for all bands
    out_arrays = [
        np.full((ref_height, ref_width), nodata_val, dtype=dtype) for _ in range(num_bands)
    ]

    # Loop over each band and resample
    for b in range(1, num_bands + 1):
        src_data = src.read(b)

        # Replace existing no-data values in the source data with the new no-data value
        if src.nodata is not None:
            src_data = np.where(src_data == src.nodata, nodata_val, src_data)

        # Replace invalid values in the source data with the new no-data value
        src_data = np.where(~np.isfinite(src_data), nodata_val, src_data).astype(dtype)

        if match_extents:
            ref_bounds = ref_src.bounds
            reproject(
                source=src_data,
                destination=out_arrays[b - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                dst_width=ref_width,
                dst_height=ref_height,
                dst_bounds=ref_bounds,  # Ensure that the destination matches the reference extent
                resampling=resampling_method
            )
        else:
            # Reproject to match resolution but maintain the source extent
            reproject(
                source=src_data,
                destination=out_arrays[b - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=resampling_method
            )

    # Write the resampled data to a new raster file
    with rio.open(out_file, 'w', **profile) as dst:
        for b in range(1, num_bands + 1):
            dst.write(out_arrays[b - 1], b)

    return rio.open(out_file, 'r+')


def samplePointsFromProbDensRaster(src: rio.DatasetReader,
                                   num_points: int,
                                   min_distance: float = 0,
                                   out_file: str = None,
                                   normalize_data: bool = False,
                                   random_seed: int = None) -> Union[None, gpd.GeoDataFrame]:
    """
    Sample points from a probability density raster while ensuring a minimum spacing between points.

    :param src: input rasterio dataset reader object with a single band of data
    :param num_points: number of points to sample
    :param min_distance: minimum spacing (meters) between points (Default = 0)
    :param out_file: path to save the output shapefile or GeoPackage (.shp or .gpkg)
    :param normalize_data: whether to normalize probability data
    :param random_seed: seed value for repeatable results
    :return: Either None or a GeoDataFrame of the resulting points.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Read raster and mask invalid values
    raster_array = src.read(1)
    raster_array = np.ma.masked_invalid(raster_array)
    raster_array = np.ma.masked_where((raster_array < 0) | (raster_array > 100), raster_array)

    # Get transform and valid pixel indices
    transform = src.transform
    valid_indices = np.argwhere(~raster_array.mask)

    # Flatten raster and get valid probabilities
    probabilities = raster_array.compressed()

    if normalize_data:
        probabilities /= probabilities.sum()  # Normalize to sum to 1

    if min_distance > 0:
        # Sample more points than needed to ensure filtering success
        oversample_factor = 3  # Try generating 3x more points initially
        sample_count = int(num_points * oversample_factor)
    else:
        sample_count = num_points

    # Sample valid points based on probability
    sampled_indices = np.random.choice(len(probabilities), size=sample_count, p=probabilities)

    # Convert sampled indices to real-world coordinates
    sampled_coords = valid_indices[sampled_indices]
    sampled_points = [Point(rio.transform.xy(transform, row, col)) for row, col in sampled_coords]

    # Convert to NumPy array for fast processing
    point_array = np.array([[p.x, p.y] for p in sampled_points])

    # Filter points using KDTree for efficient min-distance checks
    if (min_distance > 0) and (len(point_array) > 1):
        tree = cKDTree(point_array)
        valid_indices = tree.query_ball_tree(tree, min_distance)

        # Keep only unique points (no points closer than `min_distance`)
        unique_mask = np.ones(len(valid_indices), dtype=bool)
        for i, neighbors in enumerate(valid_indices):
            if len(neighbors) > 1:  # If a point is too close to another, mark it as invalid
                unique_mask[i] = False

        # Select only the unique points
        filtered_points = point_array[unique_mask]
    else:
        filtered_points = point_array  # If only one point, return it

    # Ensure we don't return more than the requested number of points
    filtered_points = filtered_points[:num_points]

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in filtered_points], crs=src.crs)

    # Save or return result
    if out_file is None:
        return gdf
    elif out_file.endswith('.shp'):
        gdf.to_file(out_file)
    elif out_file.endswith('.gpkg'):
        gdf.to_file(out_file, driver='GPKG')
    else:
        raise ValueError('Unsupported output format. Use .shp or .gpkg.')

    return


def setNull(src_path: str, nodata_val: Union[int, float]) -> None:
    """
    Function to set a new nodata value in an existing raster dataset.

    :param src_path: Path to the raster dataset (TIFF only)
    :param nodata_val: New nodata value to set in the dataset
    """
    # Open the dataset in 'r+' mode to allow modifications
    with rio.open(src_path, 'r+') as dst:
        # Set the nodata value in the dataset
        dst.nodata = nodata_val
        # Update the metadata with the new nodata value
        dst.update_tags(nodata=nodata_val)

    return


def sumRasters(src: rio.DatasetReader,
               inrasters: Union[rio.DatasetReader, list[rio.DatasetReader]]) -> rio.DatasetReader:
    """
    Function to sum values with an existing dataset.

    :param src: input rasterio dataset reader object
    :param inrasters: raster or list of rasters to add to src raster
    :return: rasterio dataset reader object in 'r+' mode
    """
    sum_result = src.read()

    if isinstance(inrasters, list):
        for ras in inrasters:
            sum_result += ras.read()
    else:
        sum_result += inrasters.read()

    profile = src.profile
    profile.update(
        compress='lzw'  # Specify LZW compression
    )
    src_path = src.name  # Get path of input source dataset
    src.close()  # Close input source dataset

    # Write new data to src_path (replace original data with new data)
    with rio.open(src_path, 'w', **profile) as dst:
        # Write sum data to source raster
        dst.write(sum_result.astype(src.dtypes[0]))
        # Calculate new statistics
        calculateStatistics(dst)

    # Return new raster as "readonly" rasterio openfile object
    return rio.open(src_path, 'r+')


def tifToASCII(src: rio.DatasetReader,
               out_file: str) -> rio.DatasetReader:
    """
    Function to convert a TIFF to an ASCII file.

    :param src: input rasterio dataset reader object
    :param out_file: path (with name) to save ASCII file
    :return: new rasterio dataset reader object in 'r+' mode
    """
    # Read raster data as a 2D NumPy array
    data = src.read(1)

    # Get the transform (georeferencing information)
    transform = src.transform

    # Get the number of rows and columns
    rows, cols = data.shape

    # Get the origin (top-left corner)
    x_ll_corner, y_ll_corner = transform * (0, rows)

    # Compute the pixel size
    pixel_size = transform[0]

    # Write the data to the ASCII file
    with open(out_file, 'w') as outfile:
        # Write header information
        outfile.write(f'ncols\t{cols}\n')
        outfile.write(f'nrows\t{rows}\n')
        outfile.write(f'xllcorner\t{x_ll_corner}\n')
        outfile.write(f'yllcorner\t{y_ll_corner}\n')
        outfile.write(f'cellsize\t{pixel_size}\n')
        outfile.write('NODATA_value\t-9999.000000\n')

        # Write the data values
        for row in data:
            for value in row:
                outfile.write(f'{value}\t')
            outfile.write('\n')

    return rio.open(out_file, 'r+')


def toMultiband(path_list: list[str],
                out_file: str) -> rio.DatasetReader:
    """
    Function to merge multiple single-band rasters into a new multiband raster.

    :param path_list: list of paths to single-band raster datasets
    :param out_file: location and name to save output raster
    :return: rasterio dataset reader object in 'r+' mode
    """
    array_list = [getRaster(path).read(1, masked=True) for path in path_list]

    out_meta = getRaster(path_list[0]).meta.copy()
    out_meta.update({'count': len(path_list)})

    shutil.copyfiles(getRaster(path_list[0]).name, out_file)

    with rio.open(out_file, 'w', **out_meta) as dest:
        for band, src in enumerate(array_list, start=1):
            dest.write(src, band)

    return rio.open(out_file, 'r+')


def trimNoDataExtent(src):
    """
    Function to trim the extent of a raster by removing areas with NoData values and adjusting the bounds
    to fit the actual data. Updates the original raster file in place.

    :param src: input rasterio dataset reader object (must be in "r+" mode)
    :return: the altered source rasterio dataset reader object
    """
    data = src.read(1, masked=True)
    # Read all bands into a list of arrays
    bands = src.count
    all_data = [src.read(band + 1) for band in range(bands)]

    # Ensure the mask has at least one dimension
    if data.mask.ndim == 0:
        data.mask = np.atleast_1d(data.mask)

    # Get the bounding box of valid data
    valid_rows, valid_cols = np.where(~data.mask)
    if valid_rows.size == 0 or valid_cols.size == 0:
        raise RuntimeWarning('No valid data found. No changes made.')

    top_left_row = min(valid_rows)
    bottom_right_row = max(valid_rows)
    top_left_col = min(valid_cols)
    bottom_right_col = max(valid_cols)

    # Crop the data for each band
    cropped_data = [data[top_left_row:bottom_right_row + 1, top_left_col:bottom_right_col + 1] for data in all_data]

    # Update the transform to reflect the cropped extent
    transform = src.transform
    new_left = transform[2] + top_left_col * transform[0]
    new_right = transform[2] + (bottom_right_col + 1) * transform[0]
    new_top = transform[5] + top_left_row * transform[4]
    new_bottom = transform[5] + (bottom_right_row + 1) * transform[4]

    # Create a new transform for the trimmed raster
    new_transform = from_bounds(new_left, new_bottom, new_right, new_top,
                                cropped_data[0].shape[1],  # new width
                                cropped_data[0].shape[0])  # new height

    # Update metadata
    out_meta = src.meta.copy()
    out_meta.update({
        'transform': new_transform,
        'height': cropped_data[0].shape[0],
        'width': cropped_data[0].shape[1]
    })

    out_file = src.name
    src.close()

    # Re-open the raster file in write mode ('w') to overwrite it with the trimmed data
    with rio.open(out_file, 'w', **out_meta) as dst:
        for band in range(bands):
            dst.write(cropped_data[band], band + 1)

    return rio.open(out_file, 'r+')


def updateRaster(src: rio.DatasetReader,
                 array: np.ndarray,
                 band: Optional[int] = None,
                 nodata_val: Optional[Union[int, float]] = None) -> rio.DatasetReader:
    """
    Function to update values in a raster with an input array.

    :param src: input rasterio dataset reader object (TIFF only)
    :param array: numpy array object
    :param band: integer representing a specific band to update
    :param nodata_val: value to assign as "No Data"
    :return: rasterio dataset reader object in 'r+' mode
    """
    if array is None or not isinstance(array, np.ndarray):
        raise ValueError('Input "array" must be a non-empty NumPy array.')

    # Get profile of source dataset
    profile = src.profile

    # Update profile
    profile.update(compress='lzw')  # Specify LZW compression
    src_path = src.name  # Get path of input source dataset
    src.close()  # Close input source dataset

    # Ensure array shape matches expected raster shape
    if len(array.shape) == 2:  # If 2D, add an axis to make it (1, height, width)
        array = array[np.newaxis, :, :]

    with rio.open(src_path, 'r+', **profile) as dst:
        if dst.count == 1:  # Single-band raster
            dst.write(array[0] if array.shape[0] == 1 else array, 1)
        elif band is not None:  # Multi-band raster, update a single band
            dst.write(array[0], band)
        else:  # Multi-band raster, update all bands
            dst.write(array)

        if nodata_val is not None:
            dst.nodata = nodata_val
            dst.update_tags(nodata=nodata_val)

        # Calculate new statistics
        calculateStatistics(dst)

    # Return new raster as "readonly" rasterio openfile object
    return rio.open(src_path, 'r+')


# STILL WORKING ON THIS FUNCTION
# def updateLargeRas_wSmallRas(src_lrg, src_small, nodata_val=None):
#     """
#     Function to update values in a large raster with values from a smaller raster.
#     The smaller raster must fit within the extent of the large raster,
#     and have the same resolution and grid alignment.
#     :param src_lrg: input rasterio dataset reader object of larger raster
#     :param src_small: input rasterio dataset reader object of smaller raster
#     :param nodata_val: the value corresponding to no data; used to exclude those values from the update
#     :return: None
#     """
#     # Get the transform and profile of the large raster
#     src_lrg_transform = src_lrg.transform
#
#     # Iterate over windows of the large raster
#     for _, window in src_lrg.block_windows(0):
#         # Read the corresponding window from the large raster
#         large_data = src_lrg.read(window=window)
#
#         # Get the window's coordinates in the large raster
#         window_coords = src_lrg.window_transform(window)
#
#         # Calculate the corresponding window in the small raster
#         small_window = Window(col_off=int(window_coords[0]), row_off=int(window_coords[1]),
#                               width=window.width, height=window.height)
#
#         # Read the data from the small raster for the current window
#         small_data = src_small.read(window=small_window)
#
#         # Calculate the offset for the overlap in the small raster
#         overlap_offset_small = (small_window.col_off - window_coords[0], small_window.row_off - window_coords[1])
#
#         # Mask for valid values in the small raster
#         valid_mask = ~np.isnan(small_data) & (small_data != nodata_val)
#
#         # Update the large raster with values from the small raster where valid
#         large_data[:, int(overlap_offset_small[1]):int(overlap_offset_small[1] + small_data.shape[1]),
#         int(overlap_offset_small[0]):int(overlap_offset_small[0] + small_data.shape[2])] = \
#             np.where(valid_mask, small_data,
#                      large_data[:, int(overlap_offset_small[1]):int(overlap_offset_small[1] + small_data.shape[1]),
#                      int(overlap_offset_small[0]):int(overlap_offset_small[0] + small_data.shape[2])])
#
#         # Write the updated data back to the large raster
#         src_lrg.write(large_data, window=window)
#
#     return src_lrg
