# -*- coding: utf-8 -*-
"""
Created on Mon Feb  21 09:00:00 2024

@author: Gregory A. Greene
"""

import os
import numpy
import numpy as np
import fiona as fio
from fiona.crs import CRS, from_epsg
# from fiona import Feature, Geometry
from shapely.geometry import mapping, shape, Point, box
from shapely.ops import unary_union
from pyproj import Transformer
import geopandas as gpd
import pandas as pd
from pandas.api.types import infer_dtype
import itertools
from typing import Union, Optional


def addField(src: fio.Collection,
             out_path: str,
             new_field: Union[str, list[str]],
             dtype: Union[str, list[str]],
             field_value: any = np.nan) -> fio.Collection:
    """
    Function to add a field to an existing shapefile.
    At present, this function only works by creating a new shapefile.
    :param src: fiona collection object
    :param out_path: path to new output shapefile
    :param new_field: name of new field
    :param dtype: data type of new field (default: 'float')
    :param field_value: value to assign to new field; must match dtype
    :return: updated fiona collection object in read mode
    """
    src_crs = src.crs
    src_schema = src.schema.copy()
    if isinstance(new_field, list):
        for i, field in enumerate(new_field):
            src_schema['properties'][field] = dtype[i]
    else:
        src_schema['properties'][new_field] = dtype

    with fio.open(out_path, 'w', 'ESRI Shapefile', src_schema, src_crs) as dst:
        for elem in src:
            if isinstance(new_field, list):
                for i, field in enumerate(new_field):
                    elem['properties'][field] = field_value
            else:
                elem['properties'][new_field] = field_value

            dst.write(
                {
                    'properties': elem['properties'],
                    'geometry': mapping(shape(elem['geometry']))
                }
            )

    return fio.open(out_path, 'r')


def bufferPoints(src: fio.Collection,
                 buffer_value: Union[int, float],
                 out_path: str,
                 use_extent: bool = False,
                 epsg: Optional[int] = None) -> fio.Collection:
    """
    Buffers each point in the input shapefile by the "buffer_value". If "use_extent" is True, a square polygon
    based on the extent of each buffer will be generated. Outputs results as a new shapefile.
    :param src: fiona collection object
    :param buffer_value: buffer value in the same units as the CRS of the input shapefile
    :param out_path: path to the output shapefile
    :param use_extent: if True, generate a square polygon based on the extent of each buffer
    :param epsg: EPSG code for the output CRS (defaults to input shapefile CRS)
    :return: fiona collection object in read mode
    """
    # Use the input shapefile's CRS if not provided
    crs = src.crs if epsg is None else from_epsg(epsg)

    # Define the schema for the output shapefile (Polygon geometry)
    schema = {
        'geometry': 'Polygon',
        'properties': {key: val for key, val in src.schema['properties'].items()}
    }

    # Create the output shapefile
    with fio.open(out_path, 'w', driver='ESRI Shapefile', crs=crs, schema=schema) as output_shp:
        # Loop through each feature (point) in the input shapefile
        for feature in src:
            # Extract the point geometry and properties
            point_geom = shape(feature['geometry'])  # Convert to shapely geometry
            properties = feature['properties']  # Get the feature's properties

            # Buffer the point geometry by the buffer_value (creates a circular buffer)
            buffered_point = point_geom.buffer(buffer_value)

            if use_extent:
                # Get the bounding box (extent) of the buffered point
                bounds = buffered_point.bounds  # (minx, miny, maxx, maxy)

                # Create a square polygon from the bounding box
                buffered_point = box(bounds[0], bounds[1], bounds[2], bounds[3])

            # Write the square polygon to the output shapefile, keeping the original properties
            output_shp.write({
                'geometry': mapping(buffered_point),
                'properties': properties
            })

    return fio.open(out_path, 'r')


def copyShapefile(src: fio.Collection,
                  out_path: str) -> fio.Collection:
    """
    Function to copy a shapefile to a new location
    :param src: fiona collection object
    :param out_path: path to new output shapefile
    :return: new fiona collection object in read mode
    """
    src_schema = src.schema.copy()

    with fio.open(out_path, 'w', 'ESRI Shapefile', src_schema, src.crs) as dst:
        for elem in src:
            dst.write(
                {
                    'properties': elem['properties'],
                    'geometry': mapping(shape(elem['geometry']))
                }
            )

    return fio.open(out_path, 'r')


def dissolveFeatures(src: fio.Collection,
                     out_path: str,
                     dissolve_field: str) -> fio.Collection:
    """
    Function returns a GeoDataFrame of the shapefile
    :param src: fiona collection object
    :param out_path: path to new output shapefile
    :param dissolve_field: name of the field/column to dissolve
    :return: dissolved fiona collection object in read mode
    """
    # Get the schema of the original shapefile, including the crs
    meta = src.meta
    with fio.open(out_path, 'w', **meta) as dst:
        # Group consecutive elements within the dissolve_field that have the same key
        element_groups = sorted(src, key=lambda k: k['properties'][dissolve_field])

        # Dissolve the features by group
        for key, group in itertools.groupby(element_groups,
                                            key=lambda x: x['properties'][dissolve_field]):
            properties, geom = zip(*[(feature['properties'],
                                      shape(feature['geometry'])) for feature in group])

            # Write the dissolved feature
            # Compute the unary_union of the elements in the group
            # using the properties of the first element in the group
            dst.write({'geometry': mapping(unary_union(geom)),
                       'properties': properties[0]})

    return fio.open(out_path, 'r')


def featureClassToDataframe(gdb_path: str,
                            feature_class: str,
                            keep_fields: Optional[list] = None) -> pd.DataFrame:
    """
    Function to convert an ESRI feature class (in a file GDB) to a Pandas Dataframe
    :param gdb_path: path to the ESRI File GeoDatabase
    :param feature_class: name of the feature class
    :param keep_fields: list of fields to keep in the output shapefile
    :return: Pandas DataFrame containing the attribute data of the feature class
    """
    from osgeo import ogr
    # Open the geodatabase
    driver = ogr.GetDriverByName('OpenFileGDB')
    gdb = driver.Open(gdb_path, 0)  # 0 means read-only
    if not gdb:
        raise Exception('Could not open geodatabase')

    # Get the layer
    layer = gdb.GetLayerByName(feature_class)
    if not layer:
        raise Exception('Could not find layer')

    # Extract field names
    field_names = [layer.GetLayerDefn().GetFieldDefn(i).GetName()
                   for i in range(layer.GetLayerDefn().GetFieldCount())]
    if keep_fields is not None:
        field_names = [field for field in field_names if field in keep_fields]

    # Extract features
    features = []
    for feature in layer:
        attrs = [feature.GetField(i) for i in range(feature.GetFieldCount())]
        features.append(attrs)

    # Convert to DataFrame
    df = pd.DataFrame(features, columns=field_names)

    # Close the data source
    gdb = None

    return df


def featureClassToShapefile(gdb_path: str,
                            feature_class: str,
                            shapefile_path: str,
                            keep_fields: Optional[list] = None) -> fio.Collection:
    """
    Function to convert an ESRI feature class (in a file GDB) to a shapefile
    :param gdb_path: path to the ESRI File GeoDatabase
    :param feature_class: name of the feature class
    :param shapefile_path: fiona collection object in read mode
    :param keep_fields: list of fields to keep in the output shapefile
    :return:
    """
    from osgeo import ogr
    # Open the source geodatabase
    gdb = ogr.Open(gdb_path)
    if not gdb:
        raise FileNotFoundError(f"Could not open geodatabase: {gdb_path}")

    layer = gdb.GetLayer(feature_class)
    if not layer:
        raise ValueError(f"Feature class '{feature_class}' not found in geodatabase.")

    # Create a new shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(shapefile_path):
        driver.DeleteDataSource(shapefile_path)
    shapefile = driver.CreateDataSource(shapefile_path)
    new_layer = shapefile.CreateLayer(feature_class, geom_type=layer.GetGeomType())

    # Create fields to keep in the new shapefile
    layer_defn = layer.GetLayerDefn()
    for i in range(layer_defn.GetFieldCount()):
        field_defn = layer_defn.GetFieldDefn(i)
        if (keep_fields is not None) and (field_defn.GetName() in keep_fields):
            new_layer.CreateField(field_defn)

    # Iterate through the source features and copy geometries and specified fields
    for feature in layer:
        new_feature = ogr.Feature(new_layer.GetLayerDefn())
        new_feature.SetGeometry(feature.GetGeometryRef().Clone())
        for field in keep_fields:
            new_feature.SetField(field, feature.GetField(field))
        new_layer.CreateFeature(new_feature)
        new_feature = None

    # Cleanup
    layer = None
    gdb = None
    shapefile = None

    return fio.open(shapefile_path, mode='r')


def featureExtentToPoly(gdf: gpd.GeoDataFrame,
                        buffer_size: float) -> gpd.GeoDataFrame:
    """
    Function buffers each polygon in the GeoDataFrame by the buffer_size and returns a new GeoDataFrame
    with polygons representing the extent (bounding box) of the buffered polygons.
    :param gdf: input GeoDataFrame object
    :param buffer_size: the buffer size to apply around the polygon
    :return: a GeoDataFrame with polygons based on the extents (bounding boxes) of the buffered polygons.
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError('The gdf parameter must be a GeoDataFrame data type.')

    # Function to buffer the polygon and get the extent (bounding box)
    def buffer_and_get_extent(polygon):
        buffered_polygon = polygon.buffer(buffer_size)
        minx, miny, maxx, maxy = buffered_polygon.bounds
        return box(minx, miny, maxx, maxy)

    # Apply the buffer_and_get_extent function to each geometry in the GeoDataFrame
    gdf['extent'] = gdf['geometry'].apply(buffer_and_get_extent)

    # Create a new GeoDataFrame with the extents as geometry
    new_gdf = gpd.GeoDataFrame(gdf.drop(columns='geometry'), geometry='extent', crs=gdf.crs)

    return new_gdf


def getCoordinates(src: fio.Collection) -> list:
    """
    Function to add a field to an existing shapefile
    :param src: fiona collection object
    :return: list of coordinate pairs for each point or vertex in a shapefile
    """
    # List to hold all coordinates
    coordinates = []

    # Iterate over each feature in the shapefile
    for feature in src:
        # Extract the geometry
        geom = feature['geometry']

        # Depending on the geometry type, extract coordinates
        if geom['type'] == 'Point':
            coordinates.append(geom['coordinates'])
        elif geom['type'] in ['LineString', 'MultiPoint']:
            coordinates.extend(geom['coordinates'])
        elif geom['type'] in ['Polygon', 'MultiLineString']:
            for part in geom['coordinates']:
                coordinates.extend(part)
        elif geom['type'] == 'MultiPolygon':
            for polygon in geom['coordinates']:
                for part in polygon:
                    coordinates.extend(part)

    return coordinates


def getShapeGDF(in_path: str) -> gpd.GeoDataFrame:
    """
    Function returns a GeoDataFrame of the shapefile
    :param in_path: path to shapefile
    :return: GeoDataFrame object
    """
    return gpd.read_file(in_path)


def getShapefile(in_path: str,
                 mode: str = 'r') -> fio.Collection:
    """
    Function returns a fiona collection object representing the shapefile
    :param in_path: path to shapefile
    :param mode: mode to open the shapefile with ('r', 'a', 'w'; default = 'r')
    :return: fiona collection object in read mode
    """
    return fio.open(in_path, mode=mode)


def joinTable(src: fio.Collection,
              table_path: str,
              out_path: str,
              key_field: str) -> fio.Collection:
    """
    Function to join a CSV or Excel table to a shapefile
    :param src: fiona collection object
    :param table_path: path to the join table
    :param out_path: output path to new shapefile
    :param key_field: common field between the shapefile and table to use for joining
    :return: fiona collection object in read mode
    """
    def _fiona_dtype(dtype):
        """Map pandas dtype to Fiona field type."""
        if dtype.startswith('int'):
            return 'int'
        elif dtype.startswith('float'):
            return 'float'
        elif dtype == 'bool':
            return 'bool'
        elif dtype.startswith('datetime'):
            return 'date'
        else:
            return 'str'

    # Read the table (CSV or Excel) into a DataFrame
    if table_path.endswith('.csv'):
        df = pd.read_csv(table_path)
    elif table_path.endswith('.xls') or table_path.endswith('.xlsx'):
        df = pd.read_excel(table_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    # Convert the DataFrame to a dictionary for easy lookup
    table_data = df.set_index(key_field).to_dict(orient='index')

    # Get the fiona collection metadata
    meta = src.meta

    # Update the schema with new fields from the table and infer data types
    for field_name in df.columns:
        if field_name != key_field and field_name not in meta['schema']['properties']:
            pandas_dtype = infer_dtype(df[field_name])
            fiona_dtype = _fiona_dtype(pandas_dtype)
            meta['schema']['properties'][field_name] = fiona_dtype

    # Create a new shapefile with updated schema
    with fio.open(out_path, 'w', **meta) as dst:
        for feature in src:
            feature_id = feature['properties'][key_field]
            if feature_id in table_data:
                # Update feature properties with table data
                for key, value in table_data[feature_id].items():
                    if key != key_field:
                        feature['properties'][key] = value
            dst.write(feature)

    return fio.open(out_path, 'r')


def projectGDF(gdf: gpd.GeoDataFrame,
               new_crs: int,
               out_path: str) -> gpd.GeoDataFrame:
    """
    Function returns a reprojected GeoDataFrame of the shapefile
    :param gdf: input GeoDataFrame object
    :param new_crs: EPSG code for new projection
    :param out_path: output path to new shapefile
    :return: GeoDataFrame object
    """
    # Change CRS to new_crs
    new_gdf = gdf.to_crs(epsg=new_crs)

    # Save new_gdf to out_path
    new_gdf.to_file(out_path)

    return new_gdf


def projectShapefile(src: fio.Collection,
                     new_crs: int,
                     out_path: str) -> fio.Collection:
    """
    Function returns a fiona collection object representing the shapefile
    :param src: fiona collection object
    :param new_crs: EPSG code for new projection
    :param out_path: output path to new shapefile
    :return: fiona collection object in read mode
    """
    src_crs = src.crs
    out_crs = CRS.from_epsg(new_crs)
    new_feats = []

    # Transform coordinates with new projection
    transformer = Transformer.from_crs(src_crs, out_crs)
    for feat in src:
        x, y = feat['geometry']['coordinates']
        x_, y_ = transformer.transform(x, y)
        new_feats.append({'geometry': mapping(Point(x_, y_)),
                          'properties': feat.properties})

    # Create new shapefile
    schema = src.schema
    with fio.open(out_path, mode='w', driver='ESRI Shapefile', schema=schema, crs=out_crs) as dst:
        for feat in new_feats:
            dst.write(feat)
    return fio.open(out_path, 'r')


def saveShapeGDF(gdf: gpd.GeoDataFrame,
                 out_path: str) -> gpd.GeoDataFrame:
    """
    Function to save a GeoDataFrame as a shapefile
    :param gdf: Geopandas GeoDataFrame object
    :param out_path: location and name to save output shapefile
    :return: GeoDataFrame object
    """
    gdf.to_file(out_path)
    return gpd.read_file(out_path)


def shapefileToNumpyArray(in_path: str,
                          out_type: Optional[str] = None,
                          fields: Optional[list[str]] = None) -> numpy.ndarray:
    """
    Function to convert a shapefile to a NumPy array
    :param in_path: path to the shapefile
    :param out_type: type of output to return (options: geometries, attributes, None).
        If None, both geometries and attributes will be returned.
    :param fields: list of fields to return from the attribute table.
        If None, all fields are returned.
    :return: NumPy array containing the geometries and attribute data from the shapefile
    """
    if out_type == 'geometries':
        fields = None

    geometries = []
    attributes = []

    # Open the shapefile using fiona
    with fio.open(in_path, 'r') as shapefile:
        for feature in shapefile:
            # Extract the geometry (as a list of coordinates)
            geometry = feature['geometry']['coordinates']
            geometries.append(geometry)

            # Extract the selected attribute data
            if fields:
                attribute = {field: feature['properties'][field] for field in fields}
            else:
                attribute = feature['properties']
            attributes.append(attribute)

    # Convert geometries and attributes to NumPy arrays
    geometries_array = np.array(geometries, dtype=object)
    attributes_array = np.array(attributes, dtype=object)

    if out_type == 'geometries':
        # Return an array of the geometries
        return geometries_array
    elif out_type == 'attributes':
        # Return an array of the attributes
        return attributes_array
    else:
        # Combine geometries and attributes into a single NumPy array
        return np.array(list(zip(geometries_array, attributes_array)), dtype=object)


def subsetPointsByBufferDistance(gdf: gpd.GeoDataFrame,
                                 buffer_dist: float,
                                 out_path: str,
                                 new_crs: Optional[int] = None) -> gpd.GeoDataFrame:
    """
    Function to generate a subset of points from a shapefile, such that each point
    is at least a given distance from all others.

    Note: The gdf must be in a projected coordinate system, and the distance value must be in the
    same units as the coordinate system. The new_crs parameter can be used to reproject the point
    dataset if it is not currently in a projected coordinate system.
    :param gdf: input GeoDataFrame object representing points
    :param buffer_dist: minimum distance between points
    :param out_path: output path to new shapefile
    :param new_crs: EPSG code to reproject the point dataset
    :return: a GeoDataFrame containing the subset of points.
    """
    # Ensure the CRS is projected in meters (for example, UTM)
    gdf = gdf.to_crs(epsg=new_crs)

    # Create an empty GeoDataFrame to store selected points
    selected_points = gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)

    # Iteratively select points
    for i, point in gdf.iterrows():
        if selected_points.empty:
            selected_points = selected_points.append(point)
        else:
            # Check the minimum distance to already selected points
            min_distance = selected_points.distance(point.geometry).min()
            if min_distance >= buffer_dist:
                selected_points = selected_points.append(point)

    # Save the resulting subset as a new shapefile
    selected_points.to_file(out_path)

    return gpd.read_file(out_path)
