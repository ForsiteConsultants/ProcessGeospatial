# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 13:25:52 2024

@author: Gregory A. Greene
"""
from typing import Union, Optional
import numpy as np
import fiona
import pandas as pd
import pyproj as pp
from osgeo import gdal
import geopandas as gpd
from geopandas import GeoDataFrame
import rasterio as rio
from rasterio.mask import mask
# from rasterio import CRS
from rasterio.features import shapes, geometry_window, geometry_mask, rasterize
from rasterio.merge import merge
from rasterio.transform import xy, from_origin, from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
# from rasterio.windows import Window
# from rasterio.windows import Window
# from rasterio.io import MemoryFile
# from rasterio.transform import Affine
from shapely.geometry import Point, box, shape, mapping
from shapely.affinity import translate
from joblib import Parallel, delayed

try:
    from rasterio import shutil
except ImportError:
    import shutil


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
                  nodata_val: Optional[Union[int, float]] = None,
                  dtype: np.dtype = np.float32) -> rio.DatasetReader:
    """
    Function to convert a numpy array to a raster
    :param array: input numpy array
    :param out_file: path (with name) to save output raster
    :param ras_profile: profile of reference rasterio dataset reader object
    :param nodata_val: new integer or floating point value to assign as "no data" (default = None)
    :param dtype: the numpy data type of new raster
    :return: rasterio dataset reader object in r+ mode
    """
    # Get profile
    profile = ras_profile

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
    Function to convert a TIFF to an ASCII file
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


def calculateStatistics(src: rio.DatasetReader) -> None:
    """
    Function to recalculate statistics for each band of a rasterio dataset reader object
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
    except:

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
    Function to change a raster's datatype to int or float
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
    Function to clip a raster with the extent of another raster
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
                      crop: Optional[bool] = True) -> rio.DatasetReader:
    """
    Function to clip a raster with a shapefile
    :param src: input rasterio dataset reader object
    :param shape_path: file path to clip shapefile
    :param out_file: location and name to save output raster
    :param select_field: name of field to use to select specific features for clipping
    :param select_value: value(s) in select_field to use for the feature selection
    :param all_touched: If True, all cells touching the shapefile boundary are also included in the output.
        If False, only cells inside the boundary are included. (default = True)
    :param crop: crop output extent to match the extent of the data (default = True)
    :return: rasterio dataset reader object in 'r+' mode
    """
    # Verify select_field and select_value parameters
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

            # Check if any features were found
            if not geometries:
                raise RuntimeWarning(f'No features found with {select_field} = {select_value}')
        else:
            geometries = [feature['geometry'] for feature in shapefile]

    # Open the raster file
    with rio.open(src_path) as new_src:
        # Clip the raster using the geometries
        out_image, out_transform = mask(new_src, geometries, all_touched=all_touched, crop=crop)
        out_meta = new_src.meta.copy()

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
               out_file: str) -> rio.DatasetReader:
    """
    Function to copy a raster to a new location
    :param src: input rasterio dataset reader object
    :param out_file: location and name to save output raster
    :return: rasterio dataset reader object in 'r+' mode
    """
    shutil.copyfiles(src.name, out_file)
    return rio.open(out_file, 'r+')


def defineProjection(src: rio.DatasetReader,
                     crs: Optional[str] = 'EPSG:4326') -> rio.DatasetReader:
    """
    Function to define the projection of a raster when the projection is missing (i.e., not already defined)
    :param src: input rasterio dataset reader object
    :param crs: location and name to save output raster (default = 'EPSG:4326')
    :return: rasterio dataset reader object in 'r+' mode
    """
    # Get path of input source dataset
    src_path = src.name

    # Close input source dataset
    src.close()

    # Reproject raster and write to out_file
    with rio.open(src_path, 'r+') as dst:
        # Update the CRS in the dataset's metadata
        dst.crs = crs

        # Calculate new statistics
        calculateStatistics(dst)

    # Return new raster as "readonly" rasterio openfile object
    return rio.open(src_path, 'r+')


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
        # Calculate new statistics
        calculateStatistics(dst)

    return


def extractValuesAtPoints(in_pts: Union[str, GeoDataFrame],
                          src: rio.DatasetReader,
                          value_field: str,
                          out_type: str = 'series',
                          new_pts_path: Optional[str] = None) -> Union[pd.Series, None]:
    """
    Function to extract raster values at shapefile point locations
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
    Function to extract a list of global row and column numbers from a raster (numpy array) with a polygon shapefile
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
    Function creates a raster from a shapefile
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
    :return: rasterio dataset reader object in r+ mode
    """
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
    Function creates a new raster using the first valid values per cell across all input rasters
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
    Function returns two X and Y rasters with cell values matching the grid cell coordinates (one for Xs, one for Ys)
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
    transformer = pp.Transformer.from_crs(src_crs, out_crs, always_xy=True)

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

    # Calculate new statistics
    temp_src = getRaster(out_file)
    calculateStatistics(temp_src)
    temp_src.close()

    return rio.open(out_file, 'r+')


def getMean(in_rasters: list[rio.DatasetReader],
            out_file: str,
            full_extent: bool = True) -> rio.DatasetReader:
    """
    Function creates a new raster from the mean value per cell across all input rasters
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
    Function creates a new raster from the minimum or maximum values per cell across all input rasters
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


def getSlope(src: rio.DatasetReader,
             out_file: str,
             slope_format: str) -> rio.DatasetReader:
    """
    Function to calculate slope from an elevation raster
    :param src: input rasterio dataset reader object
    :param out_file: the path and name of the output file
    :param slope_format: slope format ("degree" or "percent")
    :return: rasterio dataset reader object in 'r+' mode
    """
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
    :param src: source rasterio dataset reader object to alter
    :param ref_src: rasterio dataset reader object to use as reference
    :param out_file: path to save the output raster with matched extent
    :param match_res: if True, match the reference resolution,
        if False, retain the source resolution.
    :return: rasterio dataset reader object in 'r+' mode
    """
    # Calculate transform, width, and height of the new raster based on match_res
    if match_res:
        # Match reference resolution
        transform, width, height = calculate_default_transform(
            src.crs, ref_src.crs, ref_src.width, ref_src.height, *ref_src.bounds
        )
    else:
        # Retain source resolution
        transform, width, height = calculate_default_transform(
            src.crs, ref_src.crs, src.width, src.height, *ref_src.bounds
        )

    # Update the metadata of the output raster to match the reference
    dest_meta = src.meta.copy()
    dest_meta.update({
        'crs': ref_src.crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    # Open the output raster for writing
    with rio.open(out_file, 'w', **dest_meta) as dest:
        # Reproject the source raster into the target's extent and resolution
        for i in range(1, src.count + 1):
            reproject(
                source=rio.band(src, i),
                destination=rio.band(dest, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=ref_src.crs,
                resampling=Resampling.nearest
            )

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


def mosaicRasters(mosaic_list: list[str, rio.DatasetReader],
                  out_file: str) -> rio.DatasetReader:
    """
    Function mosaics a list of rasterio objects to a new TIFF raster
    :param mosaic_list: list of rasterio Dataset objects, or paths to raster datasets
    :param out_file: location and name to save output raster
    :return: rasterio dataset reader object in 'r+' mode
    """
    for data in mosaic_list:
        if isinstance(data, str):
            mosaic_list.extend([rio.open(data)])
        else:
            break

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
        # Calculate new statistics
        calculateStatistics(dst)

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
    Function to convert a raster to a polygon shapefile
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
    No data values will be ignored
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
    Function to reproject a raster to a different coordinate system
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
                   band: int = 1,
                   match_extents: bool = False) -> rio.DatasetReader:
    """
    Function to resample the resolution of one raster to match that of a reference raster.
    This function will also reproject the source projection to match the reference if needed.

    :param src: input rasterio dataset reader object
    :param ref_src: reference rasterio dataset reader object
    :param out_file: location and name to save output raster
    :param band: integer representing a specific band to extract points from (default = 1)
    :param match_extents: if True, extents of the ref and src rasters will be matched
    :return: rasterio dataset reader object in 'r+' mode
    """
    # Get the transform, dimensions, and projection of the reference dataset
    ref_transform = ref_src.transform
    ref_width = ref_src.width
    ref_height = ref_src.height
    ref_crs = ref_src.crs

    # Prepare the metadata for the output
    out_meta = ref_src.meta.copy()

    # Read and resample the source raster data
    src_transform = src.transform
    src_crs = src.crs

    # Prepare the destination array
    out_array = np.empty((ref_height, ref_width), dtype=src.read(1).dtype)

    # If match_extents is True, match the extents of the reference raster
    if match_extents:
        ref_bounds = ref_src.bounds
        reproject(
            source=src.read(band),
            destination=out_array,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_width=ref_width,
            dst_height=ref_height,
            dst_bounds=ref_bounds,  # Ensure that the destination matches the reference extent
            resampling=Resampling.nearest
        )
    else:
        # Reproject to match resolution but maintain the source extent
        reproject(
            source=src.read(band),
            destination=out_array,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest
        )

    # Update the metadata with new dimensions and transform
    out_meta.update(
        {
            'height': ref_height,
            'width': ref_width,
            'transform': ref_transform
        }
    )

    # Write the resampled data to a new raster file
    with rio.open(out_file, 'w', **out_meta) as dst:
        dst.write(out_array, 1)

    return rio.open(out_file, 'r+')


def samplePointsFromProbDensRaster(src: rio.DatasetReader,
                                   num_points: int,
                                   out_file: str,
                                   normalize_data: bool = False) -> None:
    """
    Function to sample points from a probability density raster.

    Note: The probability density data must be normalized (i.e., all values must sum to 1 over the entire raster).
    If the data are not normalized, set normalize_data to True to have the function normalize the data
    before processing. E.g., a kernel density raster can be used as input, with normalized_data set to True.

    :param src: input rasterio dataset reader object with a single band of data
    :param num_points: number of points to sample
    :param out_file: path to save the output shapefile or GeoPackage (.shp or .gpkg)
    :param normalize_data: switch to normalize the data if it hasn't already been normalized
    :return: None
    """
    # Read the raster values into a numpy array
    raster_array = src.read(1)

    # Mask out invalid (NaN) values
    raster_array = np.ma.masked_invalid(raster_array)

    # Get the affine transformation to convert pixel indices to coordinates
    transform = src.transform

    # Flatten the raster and get the corresponding probabilities
    probabilities = raster_array.compressed()  # Only valid (non-NaN) values

    if normalize_data:
        probabilities /= probabilities.sum()  # Normalize data to sum to 1

    # Get the indices of valid pixels
    valid_indices = np.argwhere(~raster_array.mask)

    # Randomly sample indices based on probabilities
    sampled_indices = np.random.choice(range(len(probabilities)), size=num_points, replace=True, p=probabilities)
    # sampled_indices = random.choices(range(len(probabilities)), weights=probabilities, k=num_points)

    # Convert sampled indices back to row, col coordinates
    sampled_coords = valid_indices[sampled_indices]

    # Convert the row, col indices into real-world coordinates using the raster's affine transform
    sampled_points = [Point(rio.transform.xy(transform, row, col)) for row, col in sampled_coords]

    # Extract CRS from the raster dataset
    raster_crs = src.crs

    # Create a GeoDataFrame for the sampled points and set the CRS to match the raster
    gdf = gpd.GeoDataFrame(geometry=sampled_points, crs=raster_crs)

    # Save the result to a shapefile or GeoPackage based on the output extension
    if out_file.endswith('.shp'):
        gdf.to_file(out_file)
    elif out_file.endswith('.gpkg'):
        gdf.to_file(out_file, driver='GPKG')
    else:
        raise ValueError('Unsupported output format. Use .shp or .gpkg.')

    return


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

    # Find valid data indices across all bands
    # valid_mask = np.zeros_like(all_data[0], dtype=bool)
    # for data in all_data:
    #     valid_mask |= ~np.isnan(data)  # Update the valid mask

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
    Function to update values in a raster with an input array

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
    profile.update(
        compress='lzw'  # Specify LZW compression
    )
    src_path = src.name  # Get path of input source dataset
    src.close()  # Close input source dataset

    # Write new data to source out_path (replace original data)
    with rio.open(src_path, 'r+', **profile) as dst:
        if band is not None:
            # Write data to source raster
            dst.write(array[0], band)
        else:
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
