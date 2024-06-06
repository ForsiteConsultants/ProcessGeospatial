# -*- coding: utf-8 -*-
"""
Created on Mon Feb  21 09:00:00 2024

@author: Gregory A. Greene
"""
import fiona
import numpy as np
import fiona as fio
from fiona.crs import CRS
# from fiona import Feature, Geometry
from shapely.geometry import mapping, shape, Point
from pyproj import Transformer
import geopandas as gpd
from typing import Union


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


def copyShapefile(src: fio.Collection,
                  out_path: str) -> fio.Collection:
    """
    Function to add a field to an existing shapefile
    :param src: fiona collection object
    :param out_path: path to new output shapefile
    :return: updated fiona collection object in read mode
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
                 out_file: str) -> gpd.GeoDataFrame:
    """
    Function to save a GeoDataFrame as a shapefile
    :param gdf: Geopandas GeoDataFrame object
    :param out_file: location and name to save output shapefile
    :return: GeoDataFrame object
    """
    gdf.to_file(out_file)
    return gpd.read_file(out_file)
