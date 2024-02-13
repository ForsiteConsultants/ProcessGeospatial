# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 13:25:52 2024

@author: Gregory A. Greene
"""
import os
import fiona
from osgeo import gdal
import rasterio as rio
import rasterio.mask
#from rasterio.io import MemoryFile
#from rasterio import CRS
from rasterio import shutil
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
import numpy as np


def getRaster(inPath):
    """
    :param inPath: path to raster dataset
    :return: rasterio dataset object in 'r+' mode
    """
    return rio.open(inPath, 'r+')


def copyRaster(src, out_file):
    """
    :param src: input rasterio dataset object
    :param out_file:
    :return: rasterio dataset object in 'r+' mode
    """
    shutil.copyfiles(src.name, out_file)
    return rio.open(out_file, 'r+')


def clipRaster_wRas(src, mask_src, out_file):
    """
    Function to clip a raster with the extent of another raster
    :param src: rasterio dataset object being masked
    :param mask_src: rasterio dataset object used as a mask
    :param out_file: location and name to save output raster
    :return: rasterio dataset object in 'r+' mode
    """
    #src_path = src.name

    geometry = [box(*mask_src.bounds)]

    #with rasterio.open(src_path) as new_src:
    out_array, out_transform = rio.mask.mask(src, geometry, all_touched=True, crop=True)
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


def clipRaster_wPoly(src, shape_path, out_file):
    """
    Function to clip a raster with a polygon shapefile
    :param src: input rasterio dataset object
    :param shape_path: file path to clip feature dataset
    :param out_file: location and name to save output raster
    :return: rasterio dataset object in 'r+' mode
    """
    src_path = src.name
    src.close()

    # Get geometry of shapefile
    with fiona.open(shape_path, 'r') as shapefile:
        shapes = [feature['geometry'] for feature in shapefile]

    with rasterio.open(src_path) as new_src:
        out_image, out_transform = rasterio.mask.mask(new_src, shapes, all_touched=True, crop=True)
        out_meta = new_src.meta

    out_meta.update({'height': out_image.shape[1],
                     'width': out_image.shape[2],
                     'transform': out_transform})

    with rasterio.open(out_file, 'w', **out_meta) as dst:
        dst.write(out_image)
    dst.close()

    return rio.open(out_file, 'r+')


def reprojRaster(src, out_file, out_crs):
    """
    Function to reproject a raster to a different coordinate system
    :param src: input rasterio dataset object
    :param out_file: location and name to save output raster
    :param out_crs: string defining new projection (e.g., 'EPSG:4326')
    :return: rasterio dataset object
    """
    # Calculate transformation needed to reproject raster to out_crs
    transform, width, height = calculate_default_transform(
        src.crs, out_crs, src.width, src.height, *src.bounds
    )
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': out_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    # Reproject raster and write to out_file
    with rio.open(out_file, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=out_crs,
                resampling=Resampling.nearest)
    # Close source raster
    dst.close()

    # Return new raster as "readonly" rasterio openfile object
    return rio.open(out_file, 'r+')


def sumRasters(src, inrasters):
    """
    :param src: input rasterio dataset object
    :param inrasters: raster or list or rasters to add to src raster
    :return: rasterio dataset object
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


def updateRaster(src, array, nodata_val=None):
    """
    :param src: input rasterio dataset object
    :param array:
    :param nodata_val:
    :return: rasterio dataset object
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
    os.remove(src_path)     # Delete input source dataset

    # Write new data to source out_path (replace original data)
    with rio.open(src_path, 'w', **profile) as dst:
        # Write data to source raster
        dst.write(array)
    # Close source raster
    dst.close()

    # Return new raster as "readonly" rasterio openfile object
    return rio.open(src_path, 'r+')


def arrayToRaster(array, out_file, rasprofile, nodata_val=None, data_type=None):
    """
    :param array: input numpy array
    :param out_file:
    :param rasprofile:
    :param nodata_val:
    :param data_type:
    :return: rasterio dataset object
    """
    # Get profile
    profile = rasprofile

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


def exportRaster(src, out_file):
    """
    :param src: input rasterio dataset object
    :param out_file:
    :return: rasterio dataset object
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


def Integer(src, datatype, nodata_val):
    """
    :param src:
    :param datatype:
    :param nodata_val:
    :return:
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
    # Close new raster
    dst.close()

    return rio.open(src_path, 'r+')


def changeDtype(src, datatype, nodata_val=None):
    """
    :param src: input rasterio dataset object
    :param nodata_val:
    :return: rasterio dataset object in 'r+' mode
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


def tifToASCII(src, out_file):
    """
    :param src: input rasterio dataset object
    :param out_file:
    :return: new rasterio dataset object in 'r+' mode
    """
    #src_path = in_src.name
    #in_src.close()

    #with rasterio.open(src_path) as src:
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


def getSlope(src, out_file, slopeformat):
    """
    :param src: input rasterio dataset object
    :param out_file: the path and name of the output file
    :param slopeformat: slope format ("degree" or "percent")
    :return: rasterio dataset object
    """
    # Get file path of dataset object
    src_path = src.name

    gdal.DEMProcessing(out_file,
                       src_path,
                       'slope',
                       slopeFormat=slopeformat)

    return rio.open(out_file, 'r+')


def getAspect(src, out_file):
    """
    :param src: a rasterio dataset object
    :param out_file: the path and name of the output file
    :return: rasterio dataset object
    """
    # Get file path of dataset object
    src_path = src.name

    gdal.DEMProcessing(out_file,
                       src_path,
                       'aspect')

    return rio.open(out_file, 'r+')


def getHillshade(src, out_file):
    """
    :param src: a rasterio dataset object
    :param out_file: the path and name of the output file
    :return: rasterio dataset object
    """
    # Get file path of dataset object
    src_path = src.name

    gdal.DEMProcessing(out_file,
                       src_path,
                       'hillshade')

    return rio.open(out_file, 'r+')
