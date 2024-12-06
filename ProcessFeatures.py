# -*- coding: utf-8 -*-
"""
Created on Mon Feb  21 09:00:00 2024

@author: Gregory A. Greene
"""

import os
import numpy as np
import json
import fiona as fio
from shapely.geometry import mapping, shape, Point, box
from shapely.ops import unary_union
from pyproj import Transformer
from pyproj import CRS
import geopandas as gpd
import pandas as pd
from pandas.api.types import infer_dtype
import itertools
from typing import Union, Optional, Literal


def _verifyGeoJSON(geojson_obj):
    """
    Checks if the input is a valid GeoJSON object and identifies its type.

    :param geojson_obj: A Python object representing a GeoJSON structure.
    :return: The GeoJSON type if valid, or None if it's not a GeoJSON.
    """
    if isinstance(geojson_obj, dict):
        geojson_type = geojson_obj.get('type', None)

        if geojson_type in ['FeatureCollection', 'Feature']:
            return geojson_type
        elif isinstance(geojson_obj.get('geometry', None), dict):
            geometry_type = geojson_obj['geometry'].get('type', None)
            return geometry_type

    return None


def addFieldsToShapefile(src_path: str,
                         new_fields: Union[str, list[str]],
                         dtype: Union[str, list[str]],
                         field_value: any = np.nan,
                         out_path: str = None,
                         overwrite: bool = False) -> fio.Collection:
    """
    Function to add field(s) to an existing shapefile.
    Creates a new shapefile or optionally overwrites the original.

    :param src_path: Path to the original shapefile
    :param out_path: Path to the output shapefile; if None, uses a temporary file
    :param new_fields: Name of new field(s) as string or list of strings
    :param dtype: Data type of new field(s) (e.g., 'float', 'str', 'int')
    :param field_value: Value to assign to new field(s); must match dtype
    :param overwrite: If True, overwrite the original shapefile with the updated version
    :return: Updated fiona collection object in read mode
    """
    # Determine the output path
    if out_path is None:
        if overwrite:
            # Create a temporary file in the same directory as the source file
            src_dir = os.path.dirname(src_path)
            out_path = os.path.join(src_dir, "temp_output.shp")
        else:
            # Create a standard temporary file if not overwriting
            out_path = src_path.replace('.shp', '_updated.shp')

    # Open the original shapefile
    with fio.open(src_path, "r") as src:
        src_crs = src.crs
        src_schema = src.schema.copy()

        # Modify the schema to include the new field(s)
        if isinstance(new_fields, list):
            for i, field in enumerate(new_fields):
                src_schema['properties'][field] = dtype[i]
        else:
            src_schema['properties'][new_fields] = dtype

        # Write to the output shapefile with the updated schema
        with fio.open(out_path, 'w', 'ESRI Shapefile', schema=src_schema, crs=src_crs) as dst:
            for elem in src:
                # Add new field(s) to properties
                if isinstance(new_fields, list):
                    for i, field in enumerate(new_fields):
                        elem['properties'][field] = field_value
                else:
                    elem['properties'][new_fields] = field_value

                dst.write({
                    'properties': elem['properties'],
                    'geometry': mapping(shape(elem['geometry']))
                })

    # Optionally overwrite the original file with the output file
    if overwrite:
        for ext in ['.shp', '.shx', '.dbf', '.prj']:
            original_file = src_path.replace('.shp', ext)
            if os.path.exists(original_file):
                os.remove(original_file)

        for ext in ['.shp', '.shx', '.dbf', '.prj']:
            temp_file = out_path.replace('.shp', ext)
            os.rename(temp_file, src_path.replace('.shp', ext))

        return fio.open(src_path, 'r')
    else:
        return fio.open(out_path, 'r')


def addFieldsToGeoJSON(src: dict,
                       new_fields: dict) -> dict:
    """
    Function to add new fields to each feature in a GeoJSON object.

    :param src: The GeoJSON object (as a Python dictionary).
    :param new_fields: A dictionary of field names and their corresponding values to add.
        e.g., {'new_field1': 'default_value', 'new_field2': 100}
    :return: The modified GeoJSON object.
    """
    # Verify GeoJSON data type
    if _verifyGeoJSON(src) is None:
        raise TypeError('Invalid data type. The "src" parameter must be a valid GeoJSON dictionary')

    for feature in src['features']:
        # Add each new field to the 'properties' of the feature
        for field_name, field_value in new_fields.items():
            feature['properties'][field_name] = field_value

    return src


def assignDefaultFieldValue_GeoJSON(src: dict,
                                    field_name: str,
                                    new_value: Union[float, int, str, None]):
    """
    Assigns a default value to a specific field in each feature's properties in a GeoJSON object.

    :param src: The GeoJSON object (as a Python dictionary).
    :param field_name: The name of the field to assign the default value to.
    :param new_value: The new value to assign to all features.
    :return: The modified GeoJSON object with updated field values.
    """
    # Verify GeoJSON data type
    if _verifyGeoJSON(src) is None:
        raise TypeError('Invalid data type. The "src" parameter must be a valid GeoJSON dictionary')

    # Update field with new value
    for feature in src['features']:
        # Check if the field exists, if not, create it, and assign the default value
        feature['properties'][field_name] = feature['properties'].get(field_name, new_value)

    return src


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
    crs = src.crs if epsg is None else CRS.from_epsg(epsg)

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


def bufferFeatures(in_path: str,
                   out_path: str,
                   buffer_dist: float,
                   get_extent: Optional[bool] = False) -> None:
    """
    Buffers features in the input geospatial file (shapefile or GeoJSON) by a specified distance and
    saves the result to a new file. Optionally returns a single feature representing the extent of
    all buffered features.

    :param in_path: Path to the input geospatial file (shapefile or GeoJSON).
    :param out_path: Path to save the buffered output file (shapefile or GeoJSON).
    :param buffer_dist: Buffer distance. Positive for outward buffering, negative for inward buffering.
    :param get_extent: If True, saves a single feature representing the extent of all buffered features.
    :return: None
    """
    with fio.open(in_path, 'r') as src:
        # Copy schema and CRS from input file
        input_schema = src.schema
        input_crs = src.crs

        # Modify schema for the buffered output
        output_schema = input_schema.copy()
        output_schema['geometry'] = 'Polygon'

        # Initialize extent bounds
        overall_bounds = None

        # Write the buffered features to the output file
        with fio.open(
            out_path,
            'w',
            driver=src.driver,
            crs=input_crs,
            schema=output_schema
        ) as dst:
            for feature in src:
                geom = shape(feature['geometry'])  # Convert to shapely geometry
                if geom.is_empty or not geom.is_valid:
                    # Skip empty or invalid geometries
                    continue

                # Apply buffer
                buffered_geom = geom.buffer(buffer_dist)

                # Ensure buffered geometry is valid and not empty
                if buffered_geom.is_valid and not buffered_geom.is_empty:
                    # Update the overall bounds
                    minx, miny, maxx, maxy = buffered_geom.bounds
                    if overall_bounds is None:
                        overall_bounds = [minx, miny, maxx, maxy]
                    else:
                        overall_bounds = [
                            min(overall_bounds[0], minx),
                            min(overall_bounds[1], miny),
                            max(overall_bounds[2], maxx),
                            max(overall_bounds[3], maxy)
                        ]

                    # If not calculating extent, save each buffered geometry
                    if not get_extent:
                        dst.write({
                            'geometry': mapping(buffered_geom),  # Convert back to GeoJSON format
                            'properties': feature['properties']  # Preserve feature properties
                        })

            # If calculating extent, save a single feature representing the bounding box
            if get_extent and overall_bounds:
                extent_geom = box(*overall_bounds)  # Create bounding box polygon
                dst.write({
                    'geometry': mapping(extent_geom),
                    'properties': {}  # Empty properties for the extent feature
                })

    return


def calcGeometryAttributes(
        src_path: str,
        attributes: list[Literal['length', 'area', 'perimeter']],
        field_names: list[str],
        units: list[Literal['meters', 'kilometers', 'miles', 'square_kilometers', 'hectares', 'acres']] = None,
        out_path: str = None,
        overwrite: bool = False,
        target_crs: str = 'EPSG:3395',
        mode: str = 'r') -> fio.collection:
    """
    Calculates specified geometry attributes (length, area, perimeter) for features in a shapefile.
    Reprojects unprojected data if needed for accurate measurements.

    :param src_path: Path to the input shapefile
    :param attributes: List of geometry attributes to calculate ('length', 'area', 'perimeter')
    :param field_names: List of field names to store each attribute's calculated values
    :param units: List of units for each attribute (e.g., 'meters', 'square_kilometers'). Default is None.
    :param out_path: Path for the output shapefile with calculated attributes; if None, uses a temporary file
    :param overwrite: If True, overwrite the original shapefile with the updated version
    :param mode: mode to open the returned output shapefile with ('r', 'a', 'w'; default = 'r')
    :param target_crs: EPSG code for the projected CRS to use if the input is unprojected
    :return: fiona collection object
    """
    if units is None:
        units = [None] * len(attributes)

        # Set output path to a temporary file in the same directory as the source shapefile if out_path is not provided
    src_dir = os.path.dirname(src_path)
    output_path = out_path or os.path.join(src_dir, "temp_output.shp")

    # Open the original shapefile
    with fio.open(src_path, 'r') as src:
        src_crs = CRS.from_user_input(src.crs)
        is_projected = src_crs.is_projected

        # Set up a transformer if reprojection is necessary
        transformer = None
        if not is_projected:
            transformer = Transformer.from_crs(src_crs, CRS.from_user_input(target_crs), always_xy=True)

        src_schema = src.schema.copy()

        # Add fields to the schema for each attribute
        for field_name in field_names:
            src_schema['properties'][field_name] = 'float'

        # Write to the new temporary shapefile with updated schema
        with fio.open(output_path, 'w', 'ESRI Shapefile', schema=src_schema,
                      crs=(target_crs if not is_projected else src.crs)) as dst:
            for feature in src:
                geom = shape(feature['geometry'])

                # Reproject geometry if the data is unprojected
                if transformer:
                    geom = shape(transformer.transform(*geom.xy))

                # Create a copy of the properties using `dict()`
                updated_properties = dict(feature['properties'])

                # Calculate each requested attribute and update the properties
                for i, attr in enumerate(attributes):
                    unit = units[i]
                    field_name = field_names[i]

                    if attr == 'length' and geom.geom_type in ['LineString', 'MultiLineString']:
                        length = geom.length
                        if unit == 'kilometers':
                            length /= 1000
                        elif unit == 'miles':
                            length *= 0.000621371
                        updated_properties[field_name] = length

                    elif attr == 'area' and geom.geom_type in ['Polygon', 'MultiPolygon']:
                        area = geom.area
                        if unit == 'square_kilometers':
                            area /= 1e6
                        elif unit == 'hectares':
                            area /= 1e4
                        elif unit == 'acres':
                            area *= 0.000247105
                        updated_properties[field_name] = area

                    elif attr == 'perimeter' and geom.geom_type in ['Polygon', 'MultiPolygon']:
                        perimeter = geom.length  # Perimeter is the length of the boundary
                        if unit == 'kilometers':
                            perimeter /= 1000
                        elif unit == 'miles':
                            perimeter *= 0.000621371
                        updated_properties[field_name] = perimeter

                    else:
                        updated_properties[field_name] = None  # Set to None if attribute not applicable

                # Update the feature with calculated attributes and write it to the output
                dst.write({
                    'properties': updated_properties,
                    'geometry': mapping(geom)
                })

    # Optionally overwrite the original shapefile with the temporary file
    if overwrite:
        for ext in ['.shp', '.shx', '.dbf', '.prj']:
            original_file = src_path.replace('.shp', ext)
            if os.path.exists(original_file):
                os.remove(original_file)

        for ext in ['.shp', '.shx', '.dbf', '.prj']:
            temp_file = output_path.replace('.shp', ext)
            os.rename(temp_file, src_path.replace('.shp', ext))
        out_path = src_path

    return fio.open(out_path, mode=mode)


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
    :param keep_fields: list of fields to keep in the Dataframe
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
    del gdb

    return df


def featureClassToGeoJSON(gdb_path: str,
                          feature_class: str,
                          keep_fields: Optional[list] = None,
                          out_path: Optional[str] = None) -> json:
    """
    Function to convert an ESRI feature class (in a file GDB) to a GeoJSON file
    :param gdb_path: path to the ESRI File GeoDatabase
    :param feature_class: name of the feature class
    :param keep_fields: list of fields to keep in the GeoJSON file
    :param out_path: list of fields to keep in the GeoJSON file
    :return: A GeoJSON version of the feature class
    """
    # Open the feature class using Fiona
    with fio.open(gdb_path, layer=feature_class, driver='OpenFileGDB') as src:
        features = []

        # Iterate through each feature in the source
        for feature in src:
            # If specific fields are to be kept, filter the feature properties
            if keep_fields:
                filtered_properties = {k: v for k, v in feature['properties'].items() if k in keep_fields}
                feature['properties'] = filtered_properties

            features.append(feature)

        # Convert to GeoJSON format
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        # Save to a .json file
        with open(out_path, 'w') as f:
            json.dump(geojson, f)

    return geojson


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


def featureExtentToPoly_GDF(gdf: gpd.GeoDataFrame,
                            buffer_size: float) -> gpd.GeoDataFrame:
    """
    Buffers each geometry in the GeoDataFrame by the buffer_size and returns a new GeoDataFrame
    with polygons representing the extent (bounding box) of the buffered geometries.

    :param gdf: input GeoDataFrame object
    :param buffer_size: the buffer size to apply around the geometries
    :return: a GeoDataFrame with polygons based on the extents (bounding boxes) of the buffered geometries.
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError('The gdf parameter must be a GeoDataFrame data type.')

    def buffer_and_get_extent(geometry):
        if geometry.is_empty or not geometry.is_valid:
            return None
        buffered = geometry.buffer(buffer_size)
        minx, miny, maxx, maxy = buffered.bounds
        return box(minx, miny, maxx, maxy)

    # Apply the buffer and extent function to each geometry
    gdf['extent'] = gdf['geometry'].apply(lambda geom: buffer_and_get_extent(geom) if geom else None)

    # Filter out rows with no valid extent (e.g., empty geometries)
    gdf = gdf.dropna(subset=['extent'])

    # Create a new GeoDataFrame with extents as geometry
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


def getNumberFeatures_GeoJSON(geojson_obj):
    """
    Returns the number of features in a GeoJSON object.

    :param geojson_obj: The GeoJSON object (as a Python dictionary).
    :return: The number of features in the GeoJSON object.
    """
    return len(geojson_obj['features'])


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
    :return: fiona collection object
    """
    return fio.open(in_path, mode=mode)


def intersectShapefiles(in_paths: list[str],
                       out_path: str,
                       out_crs_epsg: int = None,
                       mode: str = 'r') -> fio.collection:
    """
    Intersect features from multiple shapefiles and save the intersected geometries to a new shapefile.

    :param in_paths:  List of paths to input shapefiles.
    :param out_path: Path to the output shapefile where intersections will be saved.
    :param out_crs_epsg: EPSG code for output CRS. Uses the CRS from the first input if not provided.
    :param mode: mode to open the returned output shapefile with ('r', 'a', 'w'; default = 'r')
    :return: fiona collection object in read mode
    """
    # Initialize a combined schema with an unknown geometry type
    combined_schema = {'geometry': 'Unknown', 'properties': {}}

    # Combine all properties from input shapefiles into combined_schema
    for path in in_paths:
        with fio.open(path, "r") as src:
            # Update geometry type based on the first layer
            if combined_schema['geometry'] == 'Unknown':
                combined_schema['geometry'] = src.schema['geometry']

            for field_name, field_type in src.schema['properties'].items():
                if field_name not in combined_schema['properties']:
                    combined_schema['properties'][field_name] = field_type

    # Set output CRS
    crs = CRS.from_epsg(out_crs_epsg) if out_crs_epsg else fio.open(in_paths[0], 'r').crs

    # Write the intersected output with the combined schema
    with fio.open(out_path, 'w', 'ESRI Shapefile', schema=combined_schema, crs=crs) as output:
        # Read each feature from all input shapefiles
        features = []
        for filepath in in_paths:
            with fio.open(filepath, 'r') as layer:
                features.append([(shape(feature['geometry']), feature['properties']) for feature in layer])

        # Loop through all combinations of features across shapefiles to calculate intersections
        for i, (geom1, props1) in enumerate(features[0]):
            for j in range(1, len(features)):
                for geom2, props2 in features[j]:
                    intersection = geom1.intersection(geom2)
                    if not intersection.is_empty:
                        # Combine properties from both features
                        combined_properties = {**props1, **props2}
                        ordered_properties = {key: combined_properties.get(key, None) for key in
                                              combined_schema['properties'].keys()}

                        intersected_feature = {
                            'geometry': mapping(intersection),
                            'properties': ordered_properties
                        }
                        output.write(intersected_feature)

    return fio.open(out_path, mode=mode)


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


def listLayersInGDB(gdb_path: str) -> list:
    """
    Lists all feature classes and layers in a file geodatabase.
    :param gdb_path: The path to the file geodatabase (.gdb folder)
    :return: A list of feature classes/layers in the geodatabase
    """
    if not os.path.isdir(gdb_path):
        raise ValueError(f'The provided path "{gdb_path}" is not a valid directory.')

    # List to store the names of all layers in the geodatabase
    layers = []

    # Open the file geodatabase using Fiona
    try:
        # Check for available layers using Fiona's open method
        with fio.Env():
            for layer in fio.listlayers(gdb_path):
                layers.append(layer)
    except Exception as e:
        raise RuntimeError(f'Error reading the geodatabase: {e}')

    return layers


def pointsToBearingDistanceCSV(shapefile_paths: list,
                               x_field: str,
                               y_field: str,
                               distance: float,
                               bearing_interval: int,
                               output_csv_path: str) -> None:
    """
    Function generates a CSV file with bearings at specified intervals for each point in a shapefile.
    This function will read the shapefile, extract the X and Y fields, and generate a CSV with
    bearings from 0 to 360 degrees at the specified interval (e.g., every 5 degrees). The CSV
    will be saved to the path specified by `output_csv_path`.

    :param shapefile_paths: List of paths to the input point shapefiles
    :param x_field: Name of the field containing X coordinates in the shapefile
    :param y_field: Name of the field containing Y coordinates in the shapefile
    :param distance: Fixed distance to apply for all generated rows
    :param bearing_interval: Interval for bearings in degrees (e.g., 5 for bearings at 5-degree increments).
    :param output_csv_path: Path where the output CSV file will be saved.
    :return: None
    """
    # Initialize a list to store the output rows
    all_rows = []

    # Iterate over each shapefile in the list
    for shapefile_path in shapefile_paths:
        # Load the shapefile using geopandas
        gdf = gpd.read_file(shapefile_path)

        # Iterate over each point in the GeoDataFrame
        for index, row in gdf.iterrows():
            point_id = row['Point_ID']
            x = row[x_field]
            y = row[y_field]

            # Generate bearings at the specified interval
            for bearing in range(0, 361, bearing_interval):
                # Append each row to the list
                all_rows.append({
                    'Point_ID': point_id,
                    'X': x,
                    'Y': y,
                    'Distance': distance,
                    'Bearing': bearing
                })

    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(all_rows)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)

    return


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


def saveGeoJSON_ToGDB(src: dict,
                      gdb_path: str,
                      feature_class_name: str) -> None:
    """
    Saves a GeoJSON file as a feature class in a geodatabase.

    :param src: Path to the GeoJSON file.
    :param gdb_path: Path to the geodatabase.
    :param feature_class_name: Name for the new feature class in the geodatabase.
    """
    # Convert GeoJSON features to a list of shapes and properties
    features = src['features']
    records = []

    for feature in features:
        geometry = shape(feature['geometry'])  # Convert geometry
        properties = feature['properties']  # Get properties
        records.append({'geometry': geometry, **properties})  # Combine geometry and properties

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(records)

    # Save the GeoDataFrame to the geodatabase as a feature class
    gdf.to_file(gdb_path, layer=feature_class_name, driver='FileGDB')

    return


def shapefileToNumpyArray(in_path: str,
                          out_type: Optional[str] = None,
                          fields: Optional[list[str]] = None) -> np.ndarray:
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


def subsetByFieldValue_GeoJSON(src: dict,
                               field_name: str,
                               field_value: Union[int, float, str, None]) -> dict:
    """
    Returns a subset of the GeoJSON object where the specified field is None in the properties.

    :param src: The GeoJSON object (as a Python dictionary)
    :param field_name: The name of the field to check for None values
    :param field_value: The value in the field to search for
    :return: A new GeoJSON object containing only the features where the field is None
    """
    # Verify GeoJSON data type
    if _verifyGeoJSON(src) is None:
        raise TypeError('Invalid data type. The "src" parameter must be a valid GeoJSON dictionary')

    # Filter features where the specified field is None
    subset_features = [
        feature for feature in src['features']
        if feature['properties'].get(field_name) is None
    ]

    # Return a new GeoJSON object containing the filtered features
    return {
        'type': 'FeatureCollection',
        'features': subset_features
    }


def subsetPointsByBufferDistance(gdf: gpd.GeoDataFrame,
                                 buffer_dist: float,
                                 out_path: str,
                                 gen_all_points: bool = False,
                                 new_crs: Optional[int] = None) -> list[str]:
    """
    Function to generate a subset of points from a point geodataframe, such that each point
    is at least a given distance from all others.

    Note: The gdf must be in a projected coordinate system, and the distance value must be in the
    same units as the coordinate system. The new_crs parameter can be used to reproject the point
    dataset if it is not currently in a projected coordinate system.
    :param gdf: input GeoDataFrame object representing points
    :param buffer_dist: minimum distance between points
    :param out_path: output path to new shapefile
    :param gen_all_points: if True, rerun the iteration on remaining points until all have been processed
    :param new_crs: EPSG code to reproject the point dataset
    :return: a list of file paths to the output dataset(s)
    """
    # Reset index in case an index has been set
    gdf = gdf.copy().reset_index()

    if new_crs is not None:
        # Ensure the CRS is projected (for example, UTM)
        gdf = gdf.to_crs(epsg=new_crs)

        # Initialize batch counter for multiple shapefile outputs
    batch_count = 1
    output_list = []

    while not gdf.empty:
        # Create an empty GeoDataFrame to store selected points for this batch
        selected_points = gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)
        selected_indices = []

        # Iteratively select points
        for i, point in gdf.iterrows():
            if selected_points.empty:
                selected_points = pd.concat([selected_points, point.to_frame().T])
                selected_indices.append(i)
            else:
                # Check the minimum distance to already selected points
                min_distance = selected_points.distance(point.geometry).min()
                if min_distance >= buffer_dist:
                    selected_points = pd.concat([selected_points, point.to_frame().T])
                    selected_indices.append(i)

        # Save the resulting subset as a new shapefile
        if gen_all_points:
            output_path = f'{out_path.strip(".shp")}_{batch_count}.shp'
        else:
            output_path = out_path
        selected_points.set_crs(epsg=gdf.crs.to_epsg(), inplace=True)
        selected_points.to_file(output_path)
        output_list.append(output_path)

        # Remove selected points from the original GeoDataFrame by their indices
        gdf = gdf.drop(selected_indices)

        # If gen_all_points is False, exit after the first batch
        if not gen_all_points:
            break

        # Increment batch count
        batch_count += 1

    return output_list
