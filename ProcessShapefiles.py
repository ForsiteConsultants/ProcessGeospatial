# -*- coding: utf-8 -*-
"""
Created on Mon Feb  21 09:00:00 2024

@author: Gregory A. Greene
"""
#import os
import numpy as np
import fiona
#from fiona import Feature, Geometry
from shapely.geometry import mapping, shape
import geopandas as gpd


def addField(src, out_path, new_field, dtype, field_value=np.nan):
    """
    Function to add a field to an existing shapefile.
    At present, this function only works by creating a new shapefile.
    :param src: fiona collection object
    :param out_path: string; path to new output shapefile
    :param new_field: string or list of strings; name of new field
    :param dtype: string or list of strings; data type of new field (default: 'float')
    :param field_value: value matching dtype; value to assign to new field
    :return: updated fiona collection object in read mode
    """
    src_crs = src.crs
    src_schema = src.schema.copy()
    if isinstance(new_field, list):
        for i, field in enumerate(new_field):
            src_schema['properties'][field] = dtype[i]
    else:
        src_schema['properties'][new_field] = dtype

    with fiona.open(out_path, 'w', 'ESRI Shapefile', src_schema, src_crs) as dst:
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

    return fiona.open(out_path, 'r')


def copyShapefile(src, out_path):
    """
    Function to add a field to an existing shapefile
    :param src: fiona collection object
    :param out_path: string; path to new output shapefile
    :return: updated fiona collection object in read mode
    """
    src_schema = src.schema.copy()

    with fiona.open(out_path, 'w', 'ESRI Shapefile', src_schema, src.crs) as dst:
        for elem in src:
            dst.write(
                {
                    'properties': elem['properties'],
                    'geometry': mapping(shape(elem['geometry']))
                }
            )

    return fiona.open(out_path, 'r')


def getShapeGDF(in_path):
    """
    Function returns a GeoDataFrame of the shapefile
    :param in_path: path to shapefile
    :return: GeoDataFrame object
    """
    return gpd.read_file(in_path)


def getShapefile(in_path):
    """
    Function returns a fiona collection object representing the shapefile
    :param in_path: path to shapefile
    :return: fiona collection object in read mode
    """
    return fiona.open(in_path, 'r')

def main():
    in_path = r'S:\1993\1\03_MappingAnalysisData\02_Data\04_ROS_Distance_Analysis\Scratch\CommunityCluster_Distance_Lines.shp'
    out_path = r'S:\1993\1\03_MappingAnalysisData\02_Data\04_ROS_Distance_Analysis\ROS_Distance_Modelling_Results\Inputs\Distance_Lines.shp'

    in_lines = getShapefile(in_path)
    new_lines = copyShapefile(in_lines, out_path)

    fields = ['rosDist30', 'rosDist60', 'rosDist120']
    dtypes = ['float', 'float', 'float']

    new_lines = addField(in_lines, out_path=out_path, new_field=fields, dtype=dtypes)


if __name__ == '__main__':
    main()