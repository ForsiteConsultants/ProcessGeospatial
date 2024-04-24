# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 13:25:52 2024

@author: Gregory A. Greene
"""
# import os
from typing import Union
import numpy as np
import fiona
import pandas
import pyproj as pp
# from osgeo import gdal
import geopandas as gpd
from geopandas import GeoDataFrame
import rasterio as rio
import rasterio.mask
# from rasterio import CRS
from rasterio import shutil
from rasterio.features import shapes
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
# from rasterio.windows import Window
# from rasterio.io import MemoryFile
# from rasterio.transform import Affine
from shapely.geometry import box, shape
from shapely.affinity import translate
from joblib import Parallel, delayed


def _process_block(block: np.ndarray,
                   transform: rio.io.DatasetReaderBase,
                   window: rio.io.WindowMethodsMixin) -> list:
    """
    Process a block of the raster array and return shapes
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
                  nodata_val: Union[int, float] = None, data_type=None) -> rio.DatasetReader:
    """
    Function to convert a numpy array to a raster
    :param array: input numpy array
    :param out_file: path (with name) to save output raster
    :param ras_profile: profile of reference rasterio dataset reader object
    :param nodata_val: value to assign as "no data"
    :param data_type: data type of new raster (int or float)
    :return: rasterio dataset reader object
    """
    # Get profile
    profile = ras_profile

    # Update profile
    profile.update(
        compress='lzw'  # Specify LZW compression
    )
    if nodata_val:
        profile.update(
            nodata=nodata_val   # Specify nodata value
        )
    if data_type:
        profile.update(
            dtype=data_type   # Specify nodata value
        )

    # Create new raster file
    with rio.open(out_file, 'w', **profile) as dst:
        # Write data to new raster
        dst.write(array)
    # Close new raster
    dst.close()

    # Return new raster as "readonly" rasterio openfile object
    return rio.open(out_file, 'r+')


def calculateStatistics(src: rio.DatasetReader) -> None:
    """
    Function to calculate statistics of a rasterio dataset reader object
    :param src: input rasterio dataset reader object
    :return: rasterio dataset reader object in 'r+' mode
    """
    for bidx in src.indexes:
        src.statistics(bidx, clear_cache=True)

    return


def changeDtype(src: rio.DatasetReader,
                datatype: str,
                nodata_val: Union[int, float] = None) -> rio.DatasetReader:
    """
    Function to change a raster's datatype to int or float
    :param src: input rasterio dataset reader object
    :param datatype: new data type (int or float)
    :param nodata_val: value to assign as "no data"
    :return: rasterio dataset reader object in 'r+' mode
    """
    if nodata_val is None:
        nodata_val = src.profile['nodata']

    # Convert array values to integer type
    src_array = src.read()
    src_array[src_array == src.nodata] = nodata_val
    if 'int' in datatype:
        src_array = np.asarray(src_array, dtype=int)
    else:
        src_array = np.asarray(src_array, dtype=float)

    # Get file path of dataset object
    src_path = src.name

    # Get profile of dataset object
    profile = src.profile

    # Specify LZW compression and assign integer datatype
    profile.update(
        compress='lzw',
        nodata=nodata_val,
        dtype=datatype)

    src.close()

    # Create new raster file
    with rio.open(src_path, 'w', **profile) as dst:
        # Write data to new raster
        dst.write(src_array)
    # Close new raster
    dst.close()

    return rio.open(src_path, 'r+')


def clipRaster_wRas(src: rio.DatasetReader,
                    mask_src: rio.DatasetReader,
                    out_file: str,
                    all_touched: bool = True,
                    crop: bool = True) -> rio.DatasetReader:
    """
    Function to clip a raster with the extent of another raster
    :param src: rasterio dataset reader object being masked
    :param mask_src: rasterio dataset reader object used as a mask
    :param out_file: location and name to save output raster
    :param all_touched: include all cells that touch the raster boundary (in addition to inside the boundary)
    :param crop: crop output extent to match the extent of the data
    :return: rasterio dataset reader object in 'r+' mode
    """
    geometry = [box(*mask_src.bounds)]

    out_array, out_transform = rio.mask.mask(src, geometry, all_touched=all_touched, crop=crop)
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
    # Close new raster
    dst.close()

    return rio.open(out_file, 'r+')


def clipRaster_wShape(src: rio.DatasetReader,
                      shape_path: str,
                      out_file: str,
                      all_touched: bool = True,
                      crop: bool = True) -> rio.DatasetReader:
    """
    Function to clip a raster with a shapefile
    :param src: input rasterio dataset reader object
    :param shape_path: file path to clip shapefile
    :param out_file: location and name to save output raster
    :param all_touched: include all cells that touch the shapefile boundary (in addition to inside the boundary)
    :param crop: crop output extent to match the extent of the data
    :return: rasterio dataset reader object in 'r+' mode
    """
    src_path = src.name
    src.close()

    # Get geometry of shapefile
    with fiona.open(shape_path, 'r') as shapefile:
        shapes = [feature['geometry'] for feature in shapefile]

    with rasterio.open(src_path) as new_src:
        out_image, out_transform = rasterio.mask.mask(new_src, shapes, all_touched=all_touched, crop=crop)
        out_meta = new_src.meta

    out_meta.update(
        {
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform
         }
    )

    with rasterio.open(out_file, 'w', **out_meta) as dst:
        dst.write(out_image)

    return rio.open(out_file, 'r+')


def copyRaster(src: rio.DatasetReader,
               out_file: str) -> rio.DatasetReader:
    """
    Function to copy a raster to a new location
    :param src: input rasterio dataset reader object
    :param out_file: location and name to save output raster
    :return: rasterio dataset reader object in 'r+' mode
    """
    shutil.copyfiles(src.name, out_file)
    return rio.open(out_file, 'r+')


def exportRaster(src: rio.DatasetReader,
                 out_file: str) -> rio.DatasetReader:
    """
    Function to export a raster to a new location
    :param src: input rasterio dataset reader object
    :param out_file: location and name to save output raster
    :return: rasterio dataset reader object
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
    # Close new raster
    dst.close()

    return


def extractValuesAtPoints(in_pts: Union[str, GeoDataFrame],
                          src: rio.DatasetReader,
                          value_field: str,
                          out_type: str = 'series',
                          new_pts_path: Union[str, None] = None) -> Union[pandas.Series, None]:
    """
    Function to extract raster values at shapefile point locations
    :param in_pts: path to, or a GeoDataFrame object of, the point shapefile
    :param src: input rasterio dataset reader object
    :param value_field: name of field to contain raster values
    :param out_type: type of output ("series", "shp", "csv")
    :param new_pts_path: path to new point shapefile
    :return: a pandas series object, or None (if out_type is "shp" or "csv")
    """
    # If in_pts is not str or GeoDataFrame type, raise error
    if not isinstance(in_pts, (str, GeoDataFrame)):
        raise TypeError('Parameter "in_pts" must be a str or GeoPandas GeoDataFrame type')

    # If in_pts is str type, get GeoDataFrame object from path
    if isinstance(in_pts, str):
        gdf = gpd.read_file(in_pts)

    # Extract raster values at point locations, and assign to "value_field"
    gdf[f'{value_field}'] = next(src.sample(zip(gdf['geometry'].x, gdf['geometry'].y)))[0]

    # Return raster point values
    if out_type == 'series':
        return gdf[f'{value_field}']
    elif out_type == 'shp':
        gdf.to_file(new_pts_path)
    elif out_type == 'csv':
        gdf.to_csv(new_pts_path)
    else:
        raise TypeError('[ProcessRasters] extractValuesAtPoints() function parameter "out_type" '
                        'must be either "series", "shp" or "csv"')

    return


def getArea(src: rio.DatasetReader,
            search_value: Union[int, float, list[int, float]]) -> float:
    """
    Function to calculate the area of all cells matching a value or values
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
    Function to generate a slope aspect raster from an elevation dataset
    :param src: a rasterio dataset reader object
    :param out_file: the path and name of the output file
    :return: rasterio dataset reader object
    """
    # Get file path of dataset object
    src_path = src.name

    gdal.DEMProcessing(out_file,
                       src_path,
                       'aspect')

    return rio.open(out_file, 'r+')


def getGridCoordinates(src: rio.DatasetReader,
                       out_file_x: str,
                       out_file_y: str,
                       out_crs: str = 'EPSG:4326') -> (rio.DatasetReader, rio.DatasetReader):
    """
    Function returns two X and Y rasters with cell values matching the grid cell coordinates (one for Xs, one for Ys)
    :param src: a rasterio dataset reader object
    :param out_file_x: path to output raster for X coordinates
    :param out_file_y: path to output raster for Y coordinates
    :param out_crs: string defining new projection (e.g., 'EPSG:4326')
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
    transformer = pp.Transformer.from_crs(src_crs, out_crs, always_xy=True)
    # new_transform = Affine.translation(x_start, y_start) * Affine.scale(pixel_size_x, pixel_size_y)

    # Calculate the x & y coordinates for each cell
    x_coords = np.linspace(x_start + pixel_size_x / 2, x_end - pixel_size_x / 2, cols)
    y_coords = np.linspace(y_start + pixel_size_y / 2, y_end - pixel_size_y / 2, rows)
    lon, lat = np.meshgrid(x_coords, y_coords)
    lon, lat = transformer.transform(lon.flatten(), lat.flatten())

    # Reshape the lon and lat arrays to match the shape of the raster
    lon = lon.reshape(rows, cols)
    lat = lat.reshape(rows, cols)

    # Create output profiles for x and y coordinate rasters
    profile = src.profile.copy()

    # Write X coordinate data to out_path_x
    with rio.open(out_file_x, 'w', **profile) as dst:
        # Write data to out_path_x
        dst.write(lon, 1)
    dst.close()

    # Write Y coordinate data to out_path_y
    with rio.open(out_file_y, 'w', **profile) as dst:
        # Write data to out_path_y
        dst.write(lat, 1)
    dst.close()

    return rio.open(out_file_x, 'r+'), rio.open(out_file_y, 'r+')


def getHillshade(src: rio.DatasetReader,
                 out_file: str) -> rio.DatasetReader:
    """
    Function to generate a hillshade from an elevation dataset
    :param src: a rasterio dataset reader object
    :param out_file: the path and name of the output file
    :return: rasterio dataset reader object
    """
    # Get file path of dataset object
    src_path = src.name

    gdal.DEMProcessing(out_file,
                       src_path,
                       'hillshade')

    return rio.open(out_file, 'r+')


def getRaster(in_path: str) -> rio.DatasetReader:
    """
    Function to get a rasterio dataset reader object from a raster file path
    :param in_path: path to raster dataset
    :return: rasterio dataset reader object in 'r+' mode
    """
    return rio.open(in_path, 'r+')


def getResolution(src: rio.DatasetReader) -> tuple:
    """
    Function to get the x & y grid cell resolution of a raster dataset
    :param src: input rasterio dataset reader object
    :return: tuple of floats; (x resolution, y resolution)
    """
    return src.res


def getMax(in_rasters: list[rio.DatasetReader],
           out_file: str) -> rio.DatasetReader:
    """
    Function creates a new raster from the maximum values of all input rasters
    :param in_rasters: list of rasterio dataset reader objects
    :param out_file: the path and name of the output file
    :return: rasterio dataset reader object in 'r+' mode
    """
    if not isinstance(in_rasters, list):
        raise TypeError('[ProcessRasters] getMax() param "inrasters" must be a list of rasterio objects')

    profile = in_rasters[0].profile
    array_list = [ras.read() for ras in in_rasters]

    max_array = np.maximum.reduce(array_list)

    # Create new raster file
    with rio.open(out_file, 'w', **profile) as dst:
        # Write data to new raster
        dst.write(max_array)

    return rio.open(out_file, 'r+')


def getMin(in_rasters: list[rio.DatasetReader],
           out_file: str) -> rio.DatasetReader:
    """
    Function creates a new raster from the minimum values of all input rasters
    :param in_rasters: list of rasterio dataset reader objects
    :param out_file: the path and name of the output file
    :return: rasterio dataset reader object in 'r+' mode
    """
    if not isinstance(in_rasters, list):
        raise TypeError('[ProcessRasters] getMax() param "inrasters" must be a list of rasterio objects')

    profile = in_rasters[0].profile
    array_list = [ras.read() for ras in in_rasters]

    max_array = np.minimum.reduce(array_list)

    # Create new raster file
    with rio.open(out_file, 'w', **profile) as dst:
        # Write data to new raster
        dst.write(max_array)

    return rio.open(out_file, 'r+')


def getSlope(src: rio.DatasetReader,
             out_file: str,
             slopeformat: str) -> rio.DatasetReader:
    """
    Function to calculate slope from an elevation raster
    :param src: input rasterio dataset reader object
    :param out_file: the path and name of the output file
    :param slopeformat: slope format ("degree" or "percent")
    :return: rasterio dataset reader object in 'r+' mode
    """
    # Get file path of dataset object
    src_path = src.name

    gdal.DEMProcessing(out_file,
                       src_path,
                       'slope',
                       slopeFormat=slopeformat)

    return rio.open(out_file, 'r+')


def Integer(src: rio.DatasetReader,
            datatype: np.dtype,
            nodata_val: int) -> rio.DatasetReader:
    """
    Function to convert a raster to an Integer data type
    :param src: input rasterio dataset reader object
    :param datatype: numpy dtype; eg., signed (np.int8, np.int32) or unsigned (np.uint8, np.unit32)
    :param nodata_val: integer value to assign as no data
    :return: rasterio dataset reader object in 'r+' mode
    """
    # Convert array values to integer type
    int_array = src.read()
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
        dtype=datatype)

    src.close()

    # Create new raster file
    with rio.open(src_path, 'w', **profile) as dst:
        # Write data to new raster
        dst.write(int_array)

    return rio.open(src_path, 'r+')


def mosaicRasters(path_list: list[str],
                  out_file: str) -> rio.DatasetReader:
    """
    Function mosaics a list of rasterio objects to a new raster
    :param path_list: list of paths to raster datasets
    :param out_file: location and name to save output raster
    :return: rasterio dataset reader object in 'r+' mode
    """
    mosaic_list = []

    for path in path_list:
        mosaic_list.extend([rio.open(path)])

    mosaic, output = merge(mosaic_list)

    out_meta = mosaic_list[0].meta
    out_meta.update(
        {
            'driver': 'GTiff',
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'transform': output
         }
    )

    with rio.open(out_file, 'w', **out_meta) as dst:
        dst.write(mosaic)
    dst.close()

    return rio.open(out_file, 'r+')


def rasterToPoly(src: rio.DatasetReader,
                 out_file: str,
                 shp_value_field: str = 'Value',
                 multiprocess: bool = False,
                 num_cores: int = 2,
                 block_size: int = 256) -> None:
    """
    Function to convert a raster to a polygon shapefile
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
        print('[rasterToPoly - Creating shape generator]')
        shape_gen = ((shape(s), v) for s, v in shapes(src.read(masked=True), transform=src.transform))

        # Build a GeoDataFrame from unpacked shapes
        print('[rasterToPoly - Building GeoDataFrame]')
        gdf = GeoDataFrame(dict(zip(['geometry', f'{shp_value_field}'], zip(*shape_gen))), crs=src.crs)
    else:
        # ### Code for multiprocessing
        # Get raster dimensions
        height, width = src.height, src.width

        # Function to generate blocks
        print('[rasterToPoly - Generating data blocks from raster]')

        def gen_blocks():
            for i in range(0, height, block_size):
                for j in range(0, width, block_size):
                    window = ((i, min(i + block_size, height)), (j, min(j + block_size, width)))
                    yield src.read(masked=True, window=window), src.transform, window

        # Set up parallel processing
        print('[rasterToPoly - Setting up and running parallel processing]')
        shapes_list = Parallel(n_jobs=num_cores)(
            delayed(_process_block)(*block) for block in gen_blocks()
        )

        # Flatten the list of shapes
        print('[rasterToPoly - Flattening shapes]')
        shapes_flat = [shp for shapes_sublist in shapes_list for shp in shapes_sublist]

        # Build a GeoDataFrame from unpacked shapes
        print('[rasterToPoly - Building GeoDataFrame]')
        gdf = GeoDataFrame({'geometry': [s for s, _ in shapes_flat],
                            f'{shp_value_field}': [v for _, v in shapes_flat]},
                           crs=src.crs)

    # Save to shapefile
    print('[rasterToPoly - Saving shapefile to out_file]')
    gdf.to_file(out_file)

    return


def reprojRaster(src: rio.DatasetReader,
                 out_file: str,
                 out_crs: str = 'EPSG:4326') -> rio.DatasetReader:
    """
    Function to reproject a raster to a different coordinate system
    :param src: input rasterio dataset reader object
    :param out_file: location and name to save output raster
    :param out_crs: string defining new projection (e.g., 'EPSG:4326')
    :return: rasterio dataset reader object in 'r+' mode
    """
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
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=out_crs,
                resampling=Resampling.nearest)

    # Return new raster as "readonly" rasterio openfile object
    return rio.open(out_file, 'r+')


def sumRasters(src: rio.DatasetReader,
               inrasters: Union[rio.DatasetReader, list[rio.DatasetReader]]) -> rio.DatasetReader:
    """
    Function to sum values with an existing dataset
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
        compress='lzw'      # Specify LZW compression
    )
    src_path = src.name     # Get path of input source dataset
    src.close()             # Close input source dataset

    # Write new data to src_path (replace original data with new data)
    with rio.open(src_path, 'w', **profile) as dst:
        # Write sum data to source raster
        dst.write(sum_result.astype(src.dtypes[0]))
    # Close source raster
    dst.close()

    # Return new raster as "readonly" rasterio openfile object
    return rio.open(src_path, 'r+')


def tifToASCII(src: rio.DatasetReader,
               out_file: str) -> rio.DatasetReader:
    """
    Function to convert a TIFF to an ASCII file
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
    Function to merge multiple single-band rasters into a new multiband raster
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


def updateRaster(src: rio.DatasetReader,
                 array: np.ndarray,
                 band: int = None,
                 nodata_val: Union[int, float] = None) -> rio.DatasetReader:
    """
    Function to update values in a raster with an input array
    :param src: input rasterio dataset reader object (TIFF only)
    :param array: numpy array object
    :param band: integer representing a specific band to update
    :param nodata_val: value to assign as "No Data"
    :return: rasterio dataset reader object in 'r+' mode
    """
    # Get profile of source dataset
    profile = src.profile

    # Update profile
    if nodata_val:
        profile.update(
            compress='lzw',     # Specify LZW compression
            nodata=nodata_val   # Specify nodata value
        )
    else:
        profile.update(
            compress='lzw'      # Specify LZW compression
        )
    src_path = src.name     # Get path of input source dataset
    src.close()             # Close input source dataset

    # Write new data to source out_path (replace original data)
    with rio.open(src_path, 'r+', **profile) as dst:
        if band is not None:
            # Write data to source raster
            dst.write(array[0], band)
        else:
            dst.write(array)

    # Return new raster as "readonly" rasterio openfile object
    return rio.open(src_path, 'r+')


# STILL WORKING ON THIS FUNCTION
# def updateLargeRas_wSmallRas(src_lrg, src_small, nodata_val=None):
#     """
#     Function to update values in a large raster with values from a smaller raster.
#     The smaller raster must fit within the extent of the large raster.
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
