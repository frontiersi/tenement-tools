"""
   Converts a Layer to a Spatial DataFrame
"""
from __future__ import print_function
from __future__ import division
from .. import SpatialDataFrame
from arcgis.features.layer import FeatureLayer, Table
from ..utils import chunks
import pandas as pd
import numpy as np
import json
import warnings

_look_up_types = {
    "esriFieldTypeBlob" : "object",
    "esriFieldTypeDate" : "datetime64",
    "esriFieldTypeInteger" : "int64",
    "esriFieldTypeSmallInteger" : "int32",
    "esriFieldTypeDouble" : "float64",
    "esriFieldTypeSingle" :  "float32",
    "esriFieldTypeString" : "str",
    "esriFieldTypeGeometry" : "object",
    "esriFieldTypeOID" : "int64",
    "esriFieldTypeGlobalID" : "str",
    "esriFieldTypeRaster" : "object",
    "esriFieldTypeGUID" : "str",
    "esriFieldTypeXML" : "object"
}
#--------------------------------------------------------------------------
def from_layer(layer, **kwargs):
    """
    Converts a Feature Service Layer to a Pandas' DataFrame

    Parameters:
     :layer: FeatureLayer or Table object.  If the object is a FeatureLayer
      the function will return a Spatial DataFrame, if the object is of
      type Table, the function will return a Pandas' DataFrame

    Usage:
    >>> from arcgis.arcgisserver import Layer
    >>> from arcgis import from_layer
    >>> mylayer = Layer("https://sampleserver6.arcgisonline.com/arcgis/rest" +\
                        "/services/CommercialDamageAssessment/FeatureServer/0")
    >>> sdf = from_layer(mylayer)
    >>> print(sdf)
    """
    fields = []
    if isinstance(layer, (Table, FeatureLayer)) == False:
        raise ValueError("Invalid inputs: must be FeatureLayer or Table")
    if 'maxRecordCount' in layer.properties:
        max_records = layer.properties['maxRecordCount']
    else:
        max_records = 1000
    service_count = layer.query(return_count_only=True)
    if service_count > max_records:
        frames = []
        oid_info = layer.query(return_ids_only=True)
        for ids in chunks(oid_info['objectIds'], max_records):
            ids = [str(i) for i in ids]
            sql = "%s in (%s)" % (oid_info['objectIdFieldName'],
                                  ",".join(ids))
            frames.append(layer.query(where=sql).df)
        res = pd.concat(frames, ignore_index=True)
        res.reset_index(drop=True, inplace=True)
    else:
        res = layer.query().df
        res.reset_index(drop=True, inplace=True)
    dtypes = {}
    for field in layer.properties.fields:
        dtypes[field['name']] = _look_up_types[field['type']]
        if _look_up_types[field['type']] == 'datetime64':
            res[field['name']] = pd.to_datetime(res[field['name']]/1000, unit='s')
        del field
    return res
#----------------------------------------------------------------------
def to_layer(df,
             layer,
             update_existing=True,
             add_new=False,
             truncate=False):
    """
    Sends the Spatial DataFrame information to a published service

    :Parameters:
     :df: Spatial DataFrame object
     :layer: Feature Layer or Table Layer object
     :update_existing: boolean -
     :add_new: boolean
     :truncate: if true, all records will be deleted and the dataframe
      records will replace the service data
    Output:
     A layer object
    """
    if not isinstance(df, (SpatialDataFrame)):
        raise ValueError("df must be a SpatialDatframe")
    if not isinstance(layer, (Table, FeatureLayer)):
        raise ValueError("layer must be a FeatureLayer or Table Layer")
    if truncate:
        layer.delete_features(where='1=1')
        layer.edit_features(adds=df.to_featureset().features)
    elif update_existing:
        layer.edit_features(updates=df.to_featureset().features)
    elif add_new:
        layer.edit_features(adds=df.to_featureset().features)
    return layer
